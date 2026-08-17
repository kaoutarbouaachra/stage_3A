#!/usr/bin/env python3
"""
run_exact_warm.py  (TCG_carla)
==============================

TCG-Exact-Warm-CARLA: the CARLA sequential leximin driver of
:mod:`TCG_carla.run_exact`, but with the column pool warm-started from a
heuristic solution (by default the CARLA diagonal heuristic TCG_DIAG_CARLA).
Same loop, seeding, time guards and hyperparameters as
TCG_Exact.run_TCG_exact_warm; only the master rank encoding differs (cumulative
Ogryczak-Tamir, hardcoded in TCG_carla.master).

Time model (per instance)
-------------------------
* --wall-time : global wall-clock cap for THIS instance. A new rank is only
  STARTED while elapsed < wall-time, so the loop stops cleanly BETWEEN ranks.
  Default 1800s (matches the 30-min STELA/CARLA campaign).
* --time-limit : cap for ONE integer-master optimize() call. Keep it << wall.
  Default 180s.
* On external SIGINT (timeout -s SIGINT), srpaths + summary are still flushed
  from the ranks completed so far.

SCOPE: assumes h = 2.

Run as a module from Algorithm_codes/:
    python -m TCG_carla.run_exact_warm toy
    python -m TCG_carla.run_exact_warm setAP-001 --data-dir=../setAP/ --max-rank 5
    python -m TCG_carla.run_exact_warm setA-01 --warm-algo TCG_DIAG          # STELA seed
    python -m TCG_carla.run_exact_warm setA-01 --warm-srpaths /path/seed-srpaths.json

Outputs (always):
    results/<instance>/TCG_EXACT_WARM_CARLA/<instance>-srpaths.json
    results/<instance>/TCG_EXACT_WARM_CARLA/<instance>-summary.json
Logs (always):
    logs/TCG_EXACT_WARM_CARLA/<instance>_<timestamp>.log
"""

from __future__ import annotations

import argparse
import datetime
import json
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from core.paths import RESULTS_DIR, LOGS_DIR
from core.dataset import add_dataset_args, resolve_paths, instance_name

# Shared, unmodified data model (single source of truth).
from TCG_Exact.data import Column, DemandId, Period, TASRData, TASRError
# CARLA-hardcoded master lives in THIS package.
from .master import PairPathColumnGeneration

# Reuse the pure, ALGO-agnostic helpers from the CARLA cold driver so the two
# stay in lock-step (integer-solution reconstruction, srpaths I/O, stdout tee,
# license release, cumulative-theta).
from .run_exact import (
    _Tee,
    _teardown_logging,
    _release_gurobi,
    _selected_columns,
    _sorted_load_vector,
    _export_srpaths,
    _cumulative_theta,
)

# Hardcoded CARLA namespace 
ALGO_NAME = "TCG_EXACT_WARM_CARLA"
RANK_ENCODING = "carla"

DEFAULT_WARM_ALGO = "TCG_DIAG_CARLA"


_WARM_ALGO_MODULE = {
    "TCG_DIAG_CARLA": "TCG_carla.run_diag",
    "TCG_DIAG": "TCG_diag.run_TCG_Diag",
    "TCG_TOPK": "TCG_topk.run_TCG_TopK",
}


# ---------------------------------------------------------------------------
# Logging / results dir (carry the warm CARLA ALGO_NAME)
# ---------------------------------------------------------------------------
def _setup_logging(name: str) -> tuple:
    log_dir = LOGS_DIR / ALGO_NAME
    log_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = log_dir / f"{name}_{timestamp}.log"
    log_fh = open(log_path, "w", encoding="utf-8")
    sys.stdout = _Tee(sys.__stdout__, log_fh)
    return log_path, log_fh


def _results_dir_for(name: str) -> Path:
    d = RESULTS_DIR / name / ALGO_NAME
    d.mkdir(parents=True, exist_ok=True)
    return d


# ---------------------------------------------------------------------------
# Warm-start source resolution and parsing
# ---------------------------------------------------------------------------
def _resolve_warm_path(instance: str, args: argparse.Namespace) -> Path:
    if args.warm_srpaths:
        p = Path(args.warm_srpaths)
        if not p.exists():
            raise FileNotFoundError(f"Warm-start srpaths not found: {p}")
        return p

    tried: List[Path] = []
    for folder in (args.warm_algo, args.warm_algo.lower()):
        p = RESULTS_DIR / instance / folder / f"{instance}-srpaths.json"
        if p.exists():
            return p
        tried.append(p)

    module = _WARM_ALGO_MODULE.get(args.warm_algo.upper())
    hint = (f"    python -m {module} {instance}\n" if module else "")
    raise FileNotFoundError(
        "Warm-start srpaths not found. Looked in:\n"
        + "".join(f"    {p}\n" for p in tried)
        + "Run the heuristic first, e.g.\n" + hint
        + "or pass an explicit file with --warm-srpaths PATH."
    )


def _make_resolvers(data: TASRData):
    dmap: Dict[object, DemandId] = {}
    for d in data.demands:
        dmap[d.id] = d.id
        dmap[str(d.id)] = d.id
        try:
            dmap[int(d.id)] = d.id  # type: ignore[arg-type]
        except (TypeError, ValueError):
            pass
    nmap: Dict[object, object] = {}
    for n in data.nodes:
        nmap[n] = n
        nmap[str(n)] = n
        try:
            nmap[int(n)] = n  # type: ignore[arg-type]
        except (TypeError, ValueError):
            pass

    def resolve_demand(x):
        return dmap.get(x, dmap.get(str(x)))

    def resolve_node(x):
        return nmap.get(x, nmap.get(str(x)))

    return resolve_demand, resolve_node


def _parse_srpaths(path: Path, data: TASRData
                   ) -> Tuple[Dict[DemandId, Dict[Period, List]], int]:
    raw = json.loads(Path(path).read_text())
    entries = raw.get("srpaths", [])
    resolve_demand, resolve_node = _make_resolvers(data)

    warm: Dict[DemandId, Dict[Period, List]] = {
        d.id: {t: [] for t in data.periods} for d in data.demands
    }
    unresolved = 0
    for e in entries:
        d_id = resolve_demand(e.get("d"))
        if d_id is None:
            unresolved += 1
            continue
        t = int(e.get("t"))
        if t not in data.periods:
            unresolved += 1
            continue
        wps: List = []
        ok = True
        for w in e.get("w", []) or []:
            n = resolve_node(w)
            if n is None:
                ok = False
                break
            wps.append(n)
        if ok:
            warm[d_id][t] = wps
        else:
            unresolved += 1
    return warm, unresolved


# ---------------------------------------------------------------------------
# Seeding: heuristic waypoints -> validated pair-path columns -> pool
# ---------------------------------------------------------------------------
def _waypoints_to_segments(source, target, waypoints: List) -> Tuple:
    nodes = [source, *waypoints, target]
    return tuple((nodes[i], nodes[i + 1]) for i in range(len(nodes) - 1))


def _pool_column(cg: PairPathColumnGeneration, d_id: DemandId, key) -> Optional[Column]:
    for c in cg.columns.get(d_id, []):
        if c.key() == key:
            return c
    return None


def seed_columns_from_warm(cg: PairPathColumnGeneration,
                           data: TASRData,
                           warm: Dict[DemandId, Dict[Period, List]],
                           verbose: bool = True
                           ) -> Tuple[Dict[DemandId, Column], int, int]:
    """
    Add the heuristic trajectory of every demand to the pool, with per-(d, t)
    feasibility validation against feasible_real_segments and maxSeg. Refused
    trajectories fall back to the trivial column (so the warm incumbent is
    always well defined). Returns (heuristic_column_per_demand, n_added,
    n_refused).
    """
    heuristic_col: Dict[DemandId, Column] = {}
    n_added = 0
    n_refused = 0

    for d in data.demands:
        d_id = d.id
        by_t = warm.get(d_id, {})
        paths: Dict[Period, Tuple] = {}
        feasible = True

        for t in data.periods:
            wps = by_t.get(t, [])
            segs = _waypoints_to_segments(d.source, d.target, wps)

            if len(segs) > data.max_segments:
                feasible = False
                if verbose:
                    print(f"[warm][refuse] demand={d_id} t={t}: "
                          f"{len(segs)} segments > maxSeg={data.max_segments}; "
                          f"falling back to trivial for this demand.")
                break

            feas_set = data.feasible_real_segments(d_id, t)
            bad = [e for e in segs if e not in feas_set]
            if bad:
                feasible = False
                if verbose:
                    print(f"[warm][refuse] demand={d_id} t={t}: segment(s) "
                          f"{bad} not in the model's feasible set "
                          f"(interventions / adapter mismatch?); "
                          f"falling back to trivial for this demand.")
                break

            paths[t] = segs

        if not feasible:
            n_refused += 1
            paths = {t: _waypoints_to_segments(d.source, d.target, [])
                     for t in data.periods}

        col = data.make_column(d_id, paths[0], paths[1])
        if cg.add_column(col):
            n_added += 1
        heuristic_col[d_id] = _pool_column(cg, d_id, col.key()) or col

    return heuristic_col, n_added, n_refused


# ---------------------------------------------------------------------------
# Warm-started sequential leximin loop (CARLA)
# ---------------------------------------------------------------------------
def solve_leximin_warm(data: TASRData,
                       warm: Dict[DemandId, Dict[Period, List]],
                       max_rank: Optional[int] = None,
                       pricing_tol: float = 1e-7,
                       freeze_tol: float = 1e-6,
                       mip_gap: Optional[float] = 0.0,
                       time_limit: Optional[float] = None,
                       wall_time: Optional[float] = None,
                       verbose: bool = True
                       ) -> Tuple[List[float], Dict[DemandId, Column],
                                  PairPathColumnGeneration, bool]:
    """
    Same as TCG_carla.run_exact.solve_leximin, but the pool is warm-started with
    the heuristic columns before rank 1, and the loop is guarded by a per-instance
    wall-clock cap. Returns (z_star_list, chosen_columns, cg, interrupted).

    The certified sorted loads [z_1*, ..., z_{k-1}*] are passed as the master's
    `rank_values`; the CARLA master cumulates them into Theta_j* internally.
    """
    N = len(data.arc_times)
    if max_rank is None:
        max_rank = N
    max_rank = min(max_rank, N)

    t_start = time.perf_counter()
    wall_deadline = (t_start + wall_time) if wall_time else None

    # Cold construction first: this creates the trivial basic pool.
    cg = PairPathColumnGeneration(data, output=False, pricing_output=False,
                                  pricing_tol=pricing_tol)

    # ...then add the heuristic columns on top of it (additive warm start).
    heuristic_col, n_added, n_refused = seed_columns_from_warm(
        cg, data, warm, verbose=verbose)

    if verbose:
        pool_now = sum(len(c) for c in cg.columns.values())
        warm_vec = _sorted_load_vector(data, heuristic_col)
        head = ", ".join(f"{v:.6g}" for v in warm_vec[:min(5, len(warm_vec))])
        print(f"[warm] rank encoding: {RANK_ENCODING}")
        print(f"[warm] heuristic columns seeded: +{n_added} new "
              f"({n_refused} demand(s) refused -> trivial); pool now {pool_now} cols")
        print(f"[warm] heuristic incumbent sorted loads (top): {head}")
        if wall_deadline:
            print(f"[warm] wall cap: {wall_time:.0f}s   MIP cap/rank: "
                  f"{time_limit if time_limit else 'none'}s")

    z_star: List[float] = []   # certified sorted loads z_k* (challenge-facing)
    last_chosen: Dict[DemandId, Column] = {}
    interrupted = False

    try:
        for k in range(1, max_rank + 1):
            if wall_deadline is not None and time.perf_counter() >= wall_deadline:
                interrupted = True
                if verbose:
                    elapsed = time.perf_counter() - t_start
                    print(f"[wall] cap reached ({elapsed:.0f}s >= {wall_time:.0f}s) "
                          f"before rank {k}; stopping with {len(z_star)} rank(s).")
                break

            cg_res = cg.solve_lp(rank_values=list(z_star))
            mip = cg.solve_integer_master(rank_values=list(z_star),
                                          time_limit=time_limit, mip_gap=mip_gap)
            chosen = _selected_columns(cg, mip)
            last_chosen = chosen

            sorted_loads = _sorted_load_vector(data, chosen)
            zk = sorted_loads[k - 1] if k - 1 < len(sorted_loads) else 0.0
            z_star.append(float(zk))

            if verbose:
                elapsed = time.perf_counter() - t_start
                print(f"[rank {k:>3}] Theta={cg_res.objective_value:.6g}  "
                      f"z_{k}*={zk:.6g}  "
                      f"pool={sum(len(c) for c in cg.columns.values())} cols  "
                      f"iters={cg_res.iterations}  t={elapsed:.0f}s")

            if zk <= freeze_tol and all(v <= freeze_tol for v in sorted_loads[k - 1:]):
                if verbose:
                    print(f"[stop] remaining loads are ~0 at rank {k}; "
                          f"leximin vector complete.")
                break

    except KeyboardInterrupt:
        interrupted = True
        if verbose:
            print(f"\n[interrupt] SIGINT caught; returning {len(z_star)} "
                  f"completed rank(s).")

    return z_star, last_chosen, cg, interrupted


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------
def _run(args: argparse.Namespace, name: str,
         net_f: Path, tm_f: Path, scen_f: Path) -> None:
    res_dir = _results_dir_for(name)

    print("=" * 68)
    print("  T-ASR -- Exact Pair-Path CG, CARLA cumulative, WARM-started")
    print(f"  Algorithm    : TCG-Exact-Warm-CARLA  (folder: {ALGO_NAME})")
    print(f"  Rank encoding: {RANK_ENCODING}")
    print(f"  {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 68)
    print(f"  Instance     : {name}")
    print(f"  Dataset      : {net_f.parent}")
    print(f"  Results dir  : {res_dir}")
    print("=" * 68)

    from TCG_Exact.adapter import build_tasr_data_from_files

    print(f"[load] {net_f}")
    data = build_tasr_data_from_files(str(net_f), str(tm_f), str(scen_f))
    print(f"[data] nodes={len(data.nodes)} arcs={len(data.arcs)} "
          f"demands={len(data.demands)} K={data.max_segments} "
          f"budget={data.budget} N={len(data.arc_times)}")

    if len(data.periods) != 2:
        raise TASRError(
            f"this driver is specialised to h = 2, but instance '{name}' has "
            f"h = {len(data.periods)}. Use an exact solver "
            f"(ALEXA / STELA / CARLA) on this instance.")

    warm_path = _resolve_warm_path(name, args)
    warm, unresolved = _parse_srpaths(warm_path, data)
    print(f"[warm] source       : {warm_path}")
    if unresolved:
        print(f"[warm] note         : {unresolved} srpaths entry/entries could "
              f"not be resolved to this instance and were ignored.")

    t0 = datetime.datetime.now()
    z_star: List[float] = []
    chosen: Dict[DemandId, Column] = {}
    cg: Optional[PairPathColumnGeneration] = None
    interrupted = False
    try:
        z_star, chosen, cg, interrupted = solve_leximin_warm(
            data, warm, max_rank=args.max_rank, pricing_tol=args.pricing_tol,
            freeze_tol=args.freeze_tol, mip_gap=args.mip_gap,
            time_limit=args.time_limit, wall_time=args.wall_time,
        )
    finally:
        elapsed_s = (datetime.datetime.now() - t0).total_seconds()

        print("\n[leximin] z* prefix (sorted load vector, leading entries):")
        print("  " + ", ".join(f"{z:.6g}" for z in z_star[:20]))
        if len(z_star) > 20:
            print(f"  ... ({len(z_star)} ranks certified, N={len(data.arc_times)})")

        sol_path = res_dir / f"{name}-srpaths.json"
        if chosen:
            _export_srpaths(chosen, data, sol_path)

        pool_size = (sum(len(c) for c in cg.columns.values())
                     if cg is not None else 0)
        summary_path = res_dir / f"{name}-summary.json"
        with open(summary_path, "w") as f:
            json.dump({
                "algorithm":     ALGO_NAME,
                "rank_encoding": RANK_ENCODING,
                "instance":      name,
                "N":             len(data.arc_times),
                "ranks_solved":  len(z_star),
                "interrupted":   bool(interrupted),
                "z_stars":       z_star,
                "theta_stars":   _cumulative_theta(z_star),
                "pool_size":     pool_size,
                "warm_start": {
                    "source":             str(warm_path),
                    "warm_algo":          (None if args.warm_srpaths else args.warm_algo),
                    "unresolved_entries": unresolved,
                },
                "pricing_tol":   args.pricing_tol,
                "freeze_tol":    args.freeze_tol,
                "mip_gap":       args.mip_gap,
                "time_limit":    args.time_limit,
                "wall_time":     args.wall_time,
                "time_s":        round(elapsed_s, 2),
            }, f, indent=2)
        print(f"  Summary saved -> {summary_path}"
              f"{'  [INTERRUPTED]' if interrupted else ''}")

        if args.out and chosen:
            _export_srpaths(chosen, data, Path(args.out))

    print("=" * 68)


def main(argv: Optional[list] = None) -> int:
    ap = argparse.ArgumentParser(
        description="TCG-Exact-Warm-CARLA: sequential leximin via exact "
                    "pair-path CG with the cumulative Ogryczak-Tamir master, "
                    "pool warm-started from a heuristic srpaths solution.")
    add_dataset_args(ap)

    ap.add_argument("--max-rank", "--max-iter",
                    type=int, default=None, dest="max_rank",
                    help="stop after this many ranks (default: all N=|A|x|T|)")
    ap.add_argument("--pricing-tol", "--pricing_tol",
                    type=float, default=1e-7, dest="pricing_tol")
    ap.add_argument("--freeze-tol", "--freeze_tol",
                    type=float, default=1e-6, dest="freeze_tol")
    ap.add_argument("--mip-gap", "--mip_gap",
                    type=float, default=0.0, dest="mip_gap")
    ap.add_argument("--time-limit", "--time_limit",
                    type=float, default=180.0, dest="time_limit",
                    help="Cap (s) for ONE integer-master call (the per-rank "
                         "MIP-time). Keep it << --wall-time. Default 180.")
    ap.add_argument("--wall-time", "--wall_time",
                    type=float, default=1800.0, dest="wall_time",
                    help="Per-instance global wall-clock cap (s). A new rank is "
                         "only started while elapsed < wall-time. Default 1800 "
                         "(matches the 30-min campaign). Pass 0 to disable.")
    ap.add_argument("--warm-algo", "--warm_algo",
                    type=str, default=DEFAULT_WARM_ALGO, dest="warm_algo",
                    help=f"heuristic whose srpaths seeds the pool, read from "
                         f"results/<instance>/<warm-algo>/ "
                         f"(default: {DEFAULT_WARM_ALGO}; columns are "
                         f"encoding-independent, so TCG_DIAG also works)")
    ap.add_argument("--warm-srpaths", "--warm_srpaths",
                    type=str, default=None, dest="warm_srpaths",
                    help="explicit path to a srpaths.json to seed from "
                         "(overrides --warm-algo)")
    ap.add_argument("--out", type=str, default=None,
                    help="also write the srpaths JSON solution to this extra "
                         "path (the results/<instance>/<ALGO>/ copy is always "
                         "written)")
    args = ap.parse_args(argv)

    if args.wall_time is not None and args.wall_time <= 0:
        args.wall_time = None

    name = instance_name(args)

    net_f, tm_f, scen_f = resolve_paths(args)

    log_path, log_fh = _setup_logging(name)
    print(f"  Log          : {log_path}")
    try:
        _run(args, name, net_f, tm_f, scen_f)
    finally:
        _release_gurobi()
        _teardown_logging(log_fh)

    return 0


if __name__ == "__main__":
    sys.exit(main())
