#!/usr/bin/env python3
"""
run_exact.py  (TCG_carla)
=========================

TCG-Exact-CARLA: sequential lexicographic (leximin) driver for the CARLA
cumulative pair-path column-generation master (h = 2 periods). Same loop, pool,
pricing and hyperparameters as TCG_Exact.run_TCG_Exact; the ONLY difference is
the master rank encoding, which is the cumulative Ogryczak-Tamir block hardcoded
in TCG_carla.master.


The exact pair-path pricer, the column data model and the file adapter are
reused as-is from the STELA package TCG_Exact, so pricing is byte-for-byte the
same code STELA uses -- the comparison is "all else equal" by construction.

SCOPE: assumes h = 2. 

Run as a module from Algorithm_codes/:
    python -m TCG_carla.run_exact toy
    python -m TCG_carla.run_exact setA-01 --max-rank 5 --time-limit 120
    python -m TCG_carla.run_exact setAP-001 --data-dir=../setAP/ --max-rank 5

Outputs (always):
    results/<instance>/TCG_EXACT_CARLA/<instance>-srpaths.json
    results/<instance>/TCG_EXACT_CARLA/<instance>-summary.json
Logs (always):
    logs/TCG_EXACT_CARLA/<instance>_<timestamp>.log
"""

from __future__ import annotations

import argparse
import datetime
import json
import os
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from core.paths import RESULTS_DIR, LOGS_DIR, CODES_DIR
from core.dataset import add_dataset_args, resolve_paths, instance_name

from TCG_Exact.data import ArcTime, Column, DemandId, TASRData, TASRError
# CARLA-hardcoded master lives in THIS package.
from .master import MasterResult, PairPathColumnGeneration


# Hardcoded CARLA namespace 
ALGO_NAME = "TCG_EXACT_CARLA"
RANK_ENCODING = "carla"


### Gurobi setup
def _setup_gurobi():
    for p in [Path.home() / "gurobi.lic", CODES_DIR / "gurobi.lic"]:
        if p.exists():
            os.environ["GRB_LICENSE_FILE"] = str(p)
            return str(p)
    return None


_lic = _setup_gurobi()



class _Tee:
    """Writes simultaneously to stdout and a log file."""

    def __init__(self, *files):
        self.files = files

    def write(self, obj):
        for f in self.files:
            f.write(obj)
            f.flush()

    def flush(self):
        for f in self.files:
            f.flush()

    def fileno(self):
        return sys.__stdout__.fileno()


def _setup_logging(name: str) -> tuple:
    """Creates logs/<ALGO_NAME>/, opens the log file, tees sys.stdout to it."""
    log_dir = LOGS_DIR / ALGO_NAME
    log_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = log_dir / f"{name}_{timestamp}.log"
    log_fh = open(log_path, "w", encoding="utf-8")
    sys.stdout = _Tee(sys.__stdout__, log_fh)
    return log_path, log_fh


def _teardown_logging(log_fh) -> None:
    """Restores sys.stdout and closes the log file."""
    sys.stdout = sys.__stdout__
    try:
        log_fh.close()
    except Exception:
        pass


def _release_gurobi() -> None:
    """Return the WLS license token as soon as the run is over."""
    try:
        import gurobipy as gp
        gp.disposeDefaultEnv()
    except Exception:
        pass


def _results_dir_for(name: str) -> Path:
    """results/<instance>/<ALGO_NAME>/ -- created on demand."""
    d = RESULTS_DIR / name / ALGO_NAME
    d.mkdir(parents=True, exist_ok=True)
    return d


# ---------------------------------------------------------------------------
# Load-vector reconstruction from an integer master solution
# ---------------------------------------------------------------------------
def _selected_columns(cg: PairPathColumnGeneration,
                      mip: MasterResult) -> Dict[DemandId, Column]:
    """Map each demand to the unique column it selected in the integer solution."""
    chosen: Dict[DemandId, Column] = {}
    by_key: Dict[Tuple[DemandId, Tuple], Column] = {}
    for d_id, cols in cg.columns.items():
        for col in cols:
            by_key[(d_id, col.key())] = col
    for (d_id, key), val in mip.xi_values.items():
        if val > 0.5:
            chosen[d_id] = by_key[(d_id, key)]
    return chosen


def _sorted_load_vector(data: TASRData,
                        chosen: Dict[DemandId, Column]) -> List[float]:
    """
    Aggregate the per-column gamma contributions into arc-time loads and
    return them sorted in nonincreasing order, over ALL arc-time coordinates
    (zeros included), matching the challenge's sorted-load definition.
    """
    load: Dict[ArcTime, float] = {at: 0.0 for at in data.arc_times}
    for col in chosen.values():
        for at, g in col.gamma.items():
            load[at] = load.get(at, 0.0) + float(g)
    return sorted(load.values(), reverse=True)


def _cumulative_theta(z_star: List[float]) -> List[float]:
    """Theta_k* = sum of the k largest loads = cumulative sum of the certified
    sorted loads. Reported as the CARLA cumulative master objective."""
    theta: List[float] = []
    acc = 0.0
    for zk in z_star:
        acc += float(zk)
        theta.append(acc)
    return theta


# ---------------------------------------------------------------------------
# srpaths export 
# ---------------------------------------------------------------------------
def _path_to_waypoints(path) -> List:
    """
    Convert a segment path ((u0,v0),(v0,v1),...) into the waypoint list the
    challenge expects: the intermediate nodes between source and target.
    A path of segments s->w1->w2->t has waypoints [w1, w2].
    """
    if not path:
        return []
    nodes = [path[0][0]] + [seg[1] for seg in path]
    return list(nodes[1:-1])


def _export_srpaths(chosen: Dict[DemandId, Column], data: TASRData,
                    out_path: Path) -> None:
    entries = []
    for d_id, col in sorted(chosen.items(), key=lambda kv: str(kv[0])):
        for t in data.periods:
            wps = _path_to_waypoints(col.paths[t])
            if wps:  # empty waypoint lists may be omitted
                entries.append({"d": int(d_id) if isinstance(d_id, int) else d_id,
                                "t": int(t), "w": [int(w) for w in wps]})
    out_path.write_text(json.dumps({"srpaths": entries}, indent=2))
    print(f"[out] wrote {out_path}")


# ---------------------------------------------------------------------------
# Sequential leximin driver (CARLA)
# ---------------------------------------------------------------------------
def solve_leximin(data: TASRData,
                  max_rank: Optional[int] = None,
                  pricing_tol: float = 1e-7,
                  freeze_tol: float = 1e-6,
                  mip_gap: Optional[float] = 0.0,
                  time_limit: Optional[float] = None,
                  verbose: bool = True
                  ) -> Tuple[List[float], Dict[DemandId, Column],
                             PairPathColumnGeneration]:
    """
    Run the sequential leximin loop with the CARLA cumulative master. Returns
    (z_star_list, chosen_columns, cg) where z_star_list = [z_1*, z_2*, ...] are
    the certified sorted loads (recovered from the integer incumbent).

    The certified sorted loads [z_1*, ..., z_{k-1}*] are passed as the master's
    `rank_values`; the CARLA master cumulates them into Theta_j* = sum_{r<=j} z_r*
    and freezes Theta_j <= Theta_j* one-sidedly. Because the certified prefix is
    preserved rank to rank, that cumulative equals the sum of the j largest loads
    of the rank-j incumbent, so no differenced/encoding variable is ever fed back.

    time_limit bounds ONE integer-master optimize() call. It does NOT bound the
    LP, the pricing sweep, or the whole run.
    """
    N = len(data.arc_times)
    if max_rank is None:
        max_rank = N
    max_rank = min(max_rank, N)

    cg = PairPathColumnGeneration(data, output=False, pricing_output=False,
                                  pricing_tol=pricing_tol)

    z_star: List[float] = []   # certified sorted loads z_k* (challenge-facing)
    last_chosen: Dict[DemandId, Column] = {}

    for k in range(1, max_rank + 1):
        # 1-5: column generation for rank k (inherited z_1*..z_{k-1}*).
        cg_res = cg.solve_lp(rank_values=list(z_star))
        # 6: integer master over the current pool, same inherited ranks.
        mip = cg.solve_integer_master(rank_values=list(z_star),
                                      time_limit=time_limit, mip_gap=mip_gap)
        chosen = _selected_columns(cg, mip)
        last_chosen = chosen

        # Recover z_k* from the integer solution's sorted load vector rather than
        # from the master's O-T variables: the returned routing lies in the
        # rank-k feasible set, so its k-th sorted load IS z_k*.
        sorted_loads = _sorted_load_vector(data, chosen)
        zk = sorted_loads[k - 1] if k - 1 < len(sorted_loads) else 0.0
        z_star.append(float(zk))

        if verbose:
            # The CARLA rank LP objective is Theta_k (the sum of the k largest
            # loads), i.e. the rank-k LP bound in the cumulative encoding.
            print(f"[rank {k:>3}] Theta={cg_res.objective_value:.6g}  "
                  f"z_{k}*={zk:.6g}  "
                  f"pool={sum(len(c) for c in cg.columns.values())} cols  "
                  f"iters={cg_res.iterations}")

        # Stop early once the tail is all zeros.
        if zk <= freeze_tol and all(v <= freeze_tol for v in sorted_loads[k - 1:]):
            if verbose:
                print(f"[stop] remaining loads are ~0 at rank {k}; "
                      f"leximin vector complete.")
            break

    return z_star, last_chosen, cg


def _run(args: argparse.Namespace, name: str,
         net_f: Path, tm_f: Path, scen_f: Path) -> None:
    res_dir = _results_dir_for(name)

    print("=" * 68)
    print("  T-ASR -- Exact Pair-Path CG, CARLA cumulative (sequential leximin)")
    print(f"  Algorithm    : TCG-Exact-CARLA  (folder: {ALGO_NAME})")
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
          f"budget(kappa1)={data.budget} N={len(data.arc_times)}")

    if len(data.periods) != 2:
        raise TASRError(
            f"this driver is specialised to h = 2, but instance '{name}' has "
            f"h = {len(data.periods)}. Use an exact solver "
            f"(ALEXA / STELA / CARLA) on this instance.")

    t0 = datetime.datetime.now()
    z_star, chosen, cg = solve_leximin(
        data, max_rank=args.max_rank, pricing_tol=args.pricing_tol,
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
            "algorithm":       ALGO_NAME,
            "rank_encoding":   RANK_ENCODING,
            "instance":        name,
            "N":               len(data.arc_times),
            "ranks_solved":    len(z_star),
            "z_stars":         z_star,
            "theta_stars":     _cumulative_theta(z_star),
            "pool_size":       pool_size,
            "pricing_tol":     args.pricing_tol,
            "freeze_tol":      args.freeze_tol,
            "mip_gap":         args.mip_gap,
            "time_limit":      args.time_limit,
            "time_s":          round(elapsed_s, 2),
        }, f, indent=2)
    print(f"  Summary saved -> {summary_path}")

    # ... plus an optional extra copy at a user-specified path.
    if args.out:
        _export_srpaths(chosen, data, Path(args.out))

    print("=" * 68)


def main(argv: Optional[list] = None) -> int:
    ap = argparse.ArgumentParser(
        description="TCG-Exact-CARLA: sequential leximin via exact pair-path CG "
                    "with the cumulative Ogryczak-Tamir master."
    )
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
                         "Not a global cap: total time grows with the number "
                         "of ranks.")
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
