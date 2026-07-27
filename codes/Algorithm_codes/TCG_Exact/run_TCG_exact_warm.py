#!/usr/bin/env python3
"""
run_TCG_exact_warm.py
=====================

TCG-Exact-Warm: the sequential leximin driver of :mod:`run_TCG_Exact`, but with
the column pool *warm-started* from a heuristic solution instead of starting
from the bare basic (IGP) pool.

Motivation
----------
``run_TCG_Exact`` builds its initial pool with one trivial pair-path per demand.
That is a cold start: the exact pricing has to discover every useful waypoint
trajectory from scratch, rank by rank. Here we additionally seed the pool with
the trajectories of a heuristic solution (by default the TCG-Diag output), so
the restricted master starts from a good incumbent and the pricing spends its
budget closing the gap rather than re-discovering structure.

Design decisions
----------------
* **Additive, not restrictive.** We keep the trivial basic pool that
  ``PairPathColumnGeneration.__init__`` creates and *add* the heuristic columns
  on top of it (``cg.add_column`` dedups by trajectory key). The trivial columns
  remain as a feasibility floor, and -- crucially -- exact pricing stays fully
  enabled at every rank, so the warm start accelerates without changing what the
  method can certify. A warm start must never shrink the search space.

* **Decoupled via the challenge output format.** The warm start is read from a
  ``<instance>-srpaths.json`` file (the official solution format), by default the
  one written by TCG-Diag under ``results/<instance>/TCG_DIAG/``. Any heuristic
  that writes srpaths can be used as the seed (``--warm-algo`` / ``--warm-srpaths``).

* **Feasibility-guarded seeding (correctness).** ``TASRData.make_column`` computes
  gamma from the split coefficients ``r`` and does NOT validate segments. A
  segment ``(u, v)`` that is not in ``data.feasible_real_segments(d, t)`` yields
  ``r == 0`` on every arc, i.e. a column with *under-counted* load -- which would
  look artificially good and could corrupt the certified z_k*. We therefore
  validate every heuristic segment against ``feasible_real_segments`` (and against
  ``maxSeg``) before seeding; an infeasible trajectory is refused (the demand
  falls back to its trivial column) and a warning is printed.

SCOPE: assumes h = 2, like the cold driver.

Everything downstream of the pool -- the rank-by-rank loop, the z_k* recovery
from the integer solution's sorted load vector, the freeze tolerance, the
outputs -- is identical to ``run_TCG_Exact``; only the initial pool differs.

Run as a module from Algorithm_codes/:
    python -m TCG_Exact.run_TCG_exact_warm toy
    python -m TCG_Exact.run_TCG_exact_warm setA-01 --max-rank 5 --time-limit 120
    python -m TCG_Exact.run_TCG_exact_warm setAP-001 --data-dir=../setAP/ --max-rank 5
    python -m TCG_Exact.run_TCG_exact_warm setA-01 --warm-algo TCG_TOPK
    python -m TCG_Exact.run_TCG_exact_warm setA-01 --warm-srpaths /path/to/seed-srpaths.json

Outputs (always):
    results/<instance>/TCG_EXACT_WARM/<instance>-srpaths.json
    results/<instance>/TCG_EXACT_WARM/<instance>-summary.json
Logs (always):
    logs/TCG_EXACT_WARM/<instance>_<timestamp>.log
"""

from __future__ import annotations

import argparse
import datetime
import json
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from core.paths import RESULTS_DIR, LOGS_DIR
from core.dataset import add_dataset_args, resolve_paths, instance_name

from .data import Column, DemandId, Period, TASRData, TASRError
from .master import PairPathColumnGeneration

# Reuse the pure, ALGO-agnostic helpers from the cold driver so the two stay in
# lock-step (integer-solution reconstruction, srpaths I/O, stdout tee, license
# release). Only the pieces that must carry the new ALGO_NAME are redefined.
from .run_TCG_Exact import (
    _Tee,
    _teardown_logging,
    _release_gurobi,
    _selected_columns,
    _sorted_load_vector,
    _export_srpaths,
)

# Official algorithm identifier -- its own results/log subfolder so warm and
# cold runs never overwrite each other and every summary is self-identifying.
ALGO_NAME = "TCG_EXACT_WARM"

# Default heuristic whose srpaths seeds the pool (results/<inst>/<this>/...).
DEFAULT_WARM_ALGO = "TCG_DIAG"

# results-folder name -> module path, used only to print an actionable hint
# when the seed file is missing. The results folder and the Python package are
# NOT the same string (folders are uppercase, packages keep their own casing).
_WARM_ALGO_MODULE = {
    "TCG_DIAG": "TCG_diag.run_TCG_Diag",
    "TCG_TOPK": "TCG_topk.run_TCG_TopK",
}


# ---------------------------------------------------------------------------
# Logging / results dir (carry the warm ALGO_NAME)
# ---------------------------------------------------------------------------
def _setup_logging(name: str) -> tuple:
    """Creates logs/<ALGO_NAME>/, opens the log file, tees sys.stdout to it."""
    log_dir = LOGS_DIR / ALGO_NAME
    log_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = log_dir / f"{name}_{timestamp}.log"
    log_fh = open(log_path, "w", encoding="utf-8")
    sys.stdout = _Tee(sys.__stdout__, log_fh)
    return log_path, log_fh


def _results_dir_for(name: str) -> Path:
    """results/<instance>/<ALGO_NAME>/ -- created on demand."""
    d = RESULTS_DIR / name / ALGO_NAME
    d.mkdir(parents=True, exist_ok=True)
    return d


# ---------------------------------------------------------------------------
# Warm-start source resolution and parsing
# ---------------------------------------------------------------------------
def _resolve_warm_path(instance: str, args: argparse.Namespace) -> Path:
    """
    Locate the srpaths.json that seeds the pool.

    The results folders were renamed to uppercase; runs produced before that
    rename still sit in the lowercase folder, so both spellings are tried.
    """
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
    """
    srpaths ids are JSON ints; data ids may be int/str/tuple. Build tolerant
    lookups so an int id in the file resolves to the matching data id whatever
    its concrete type is.
    """
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
    """
    Read a challenge srpaths file into ``{data_demand_id: {t: [node, ...]}}``,
    resolving ids to the concrete types used by ``data``. Demands/periods absent
    from the file default to an empty waypoint list (i.e. the trivial routing).
    Returns (warm, n_unresolved_entries).
    """
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
    """[w1, w2] for demand (s, t) -> ((s, w1), (w1, w2), (w2, t)); [] -> ((s, t),)."""
    nodes = [source, *waypoints, target]
    return tuple((nodes[i], nodes[i + 1]) for i in range(len(nodes) - 1))


def _pool_column(cg: PairPathColumnGeneration, d_id: DemandId, key) -> Optional[Column]:
    """Return the pooled Column for (d_id, key), or None if absent."""
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
    feasibility validation. Returns (heuristic_column_per_demand, n_added,
    n_refused). ``heuristic_column_per_demand[d]`` is the pooled Column that
    represents the heuristic's routing for d (its trivial column if the
    trajectory was refused or was itself trivial); together they form the warm
    incumbent whose sorted load vector we report.
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

            # Guard 1: number of segments must respect maxSeg.
            if len(segs) > data.max_segments:
                feasible = False
                if verbose:
                    print(f"[warm][refuse] demand={d_id} t={t}: "
                          f"{len(segs)} segments > maxSeg={data.max_segments}; "
                          f"falling back to trivial for this demand.")
                break

            # Guard 2: every segment must be feasible in the exact model, else
            # its split coefficients are all zero and the column's load is
            # under-counted -- which would corrupt z_k*.
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
            # Fall back to the trivial routing for this demand so the warm
            # incumbent is always well defined and always feasible.
            paths = {t: _waypoints_to_segments(d.source, d.target, [])
                     for t in data.periods}

        col = data.make_column(d_id, paths[0], paths[1])
        if cg.add_column(col):
            n_added += 1
        # add_column dedups by key; fetch the actually-pooled instance.
        heuristic_col[d_id] = _pool_column(cg, d_id, col.key()) or col

    return heuristic_col, n_added, n_refused


# ---------------------------------------------------------------------------
# Warm-started sequential leximin loop
# ---------------------------------------------------------------------------
def solve_leximin_warm(data: TASRData,
                       warm: Dict[DemandId, Dict[Period, List]],
                       max_rank: Optional[int] = None,
                       pricing_tol: float = 1e-7,
                       freeze_tol: float = 1e-6,
                       mip_gap: Optional[float] = 0.0,
                       time_limit: Optional[float] = None,
                       verbose: bool = True
                       ) -> Tuple[List[float], Dict[DemandId, Column], PairPathColumnGeneration]:
    """
    Same sequential leximin loop as run_TCG_Exact.solve_leximin, but the pool is
    warm-started with the heuristic columns before rank 1. Returns
    (z_star_list, chosen_columns, cg).

    NOTE: the rank loop below is duplicated from the cold driver. Any change to
    the loop must be applied in both files until the seeding is factored out
    into a hook on solve_leximin.
    """
    N = len(data.arc_times)
    if max_rank is None:
        max_rank = N
    max_rank = min(max_rank, N)

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
        print(f"[warm] heuristic columns seeded: +{n_added} new "
              f"({n_refused} demand(s) refused -> trivial); pool now {pool_now} cols")
        print(f"[warm] heuristic incumbent sorted loads (top): {head}")

    z_star: List[float] = []
    last_chosen: Dict[DemandId, Column] = {}

    for k in range(1, max_rank + 1):
        # 1-5: column generation for rank k (inherited z_1*..z_{k-1}*).
        cg_res = cg.solve_lp(rank_values=list(z_star))
        # 6: integer master over the current pool, same inherited ranks.
        mip = cg.solve_integer_master(rank_values=list(z_star),
                                      time_limit=time_limit, mip_gap=mip_gap)
        chosen = _selected_columns(cg, mip)
        last_chosen = chosen

        # Recover z_k* from the integer solution's sorted load vector rather
        # than from the master's z variable, which is only bracketed by
        # [z_{k+1}*, z_k*].
        sorted_loads = _sorted_load_vector(data, chosen)
        zk = sorted_loads[k - 1] if k - 1 < len(sorted_loads) else 0.0
        z_star.append(float(zk))

        if verbose:
            lp = cg_res.objective_value
            print(f"[rank {k:>3}] LP={lp:.6g}  z_{k}*={zk:.6g}  "
                  f"pool={sum(len(c) for c in cg.columns.values())} cols  "
                  f"iters={cg_res.iterations}")

        # Stop early once the tail is all zeros.
        if zk <= freeze_tol and all(v <= freeze_tol for v in sorted_loads[k - 1:]):
            if verbose:
                print(f"[stop] remaining loads are ~0 at rank {k}; "
                      f"leximin vector complete.")
            break

    return z_star, last_chosen, cg


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------
def _run(args: argparse.Namespace, name: str,
         net_f: Path, tm_f: Path, scen_f: Path) -> None:
    res_dir = _results_dir_for(name)

    print("=" * 68)
    print("  T-ASR -- Exact Pair-Path CG, WARM-started (sequential leximin)")
    print(f"  Algorithm    : TCG-Exact-Warm  (folder: {ALGO_NAME})")
    print(f"  {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 68)
    print(f"  Instance     : {name}")
    print(f"  Dataset      : {net_f.parent}")
    print(f"  Results dir  : {res_dir}")
    print("=" * 68)

    from .adapter import build_tasr_data_from_files

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

    # Resolve + parse the warm-start srpaths.
    warm_path = _resolve_warm_path(name, args)
    warm, unresolved = _parse_srpaths(warm_path, data)
    print(f"[warm] source       : {warm_path}")
    if unresolved:
        print(f"[warm] note         : {unresolved} srpaths entry/entries could "
              f"not be resolved to this instance and were ignored.")

    t0 = datetime.datetime.now()
    z_star, chosen, cg = solve_leximin_warm(
        data, warm, max_rank=args.max_rank, pricing_tol=args.pricing_tol,
        freeze_tol=args.freeze_tol, mip_gap=args.mip_gap,
        time_limit=args.time_limit,
    )
    elapsed_s = (datetime.datetime.now() - t0).total_seconds()

    print("\n[leximin] z* prefix (sorted load vector, leading entries):")
    print("  " + ", ".join(f"{z:.6g}" for z in z_star[:20]))
    if len(z_star) > 20:
        print(f"  ... ({len(z_star)} ranks certified, N={len(data.arc_times)})")

    # Always saved under results/<instance>/<ALGO_NAME>/ ...
    sol_path = res_dir / f"{name}-srpaths.json"
    _export_srpaths(chosen, data, sol_path)

    pool_size = sum(len(c) for c in cg.columns.values())
    summary_path = res_dir / f"{name}-summary.json"
    with open(summary_path, "w") as f:
        json.dump({
            "algorithm":    ALGO_NAME,
            "instance":     name,
            "N":            len(data.arc_times),
            "ranks_solved": len(z_star),
            "z_stars":      z_star,
            "pool_size":    pool_size,
            "warm_start": {
                "source":             str(warm_path),
                "warm_algo":          (None if args.warm_srpaths else args.warm_algo),
                "unresolved_entries": unresolved,
            },
            "pricing_tol":  args.pricing_tol,
            "freeze_tol":   args.freeze_tol,
            "mip_gap":      args.mip_gap,
            "time_limit":   args.time_limit,
            "time_s":       round(elapsed_s, 2),
        }, f, indent=2)
    print(f"  Summary saved -> {summary_path}")

    # ... plus an optional extra copy at a user-specified path.
    if args.out:
        _export_srpaths(chosen, data, Path(args.out))

    print("=" * 68)


def main(argv: Optional[list] = None) -> int:
    ap = argparse.ArgumentParser(
        description="TCG-Exact-Warm: sequential leximin via exact pair-path CG, "
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
                    type=float, default=None, dest="time_limit",
                    help="Time limit for ONE integer-master call, in seconds. "
                         "Not a global cap.")
    ap.add_argument("--warm-algo", "--warm_algo",
                    type=str, default=DEFAULT_WARM_ALGO, dest="warm_algo",
                    help=f"heuristic whose srpaths seeds the pool, read from "
                         f"results/<instance>/<warm-algo>/ "
                         f"(default: {DEFAULT_WARM_ALGO})")
    ap.add_argument("--warm-srpaths", "--warm_srpaths",
                    type=str, default=None, dest="warm_srpaths",
                    help="explicit path to a srpaths.json to seed from "
                         "(overrides --warm-algo)")
    ap.add_argument("--out", type=str, default=None,
                    help="also write the srpaths JSON solution to this extra "
                         "path (the results/<instance>/<ALGO>/ copy is always "
                         "written)")
    args = ap.parse_args(argv)

    name = instance_name(args)

    # Resolve BEFORE opening the log: resolve_paths exits on a missing
    # instance, and an empty log for a run that never started is just noise.
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