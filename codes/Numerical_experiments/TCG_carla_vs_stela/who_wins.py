#!/usr/bin/env python3

"""
who_wins.py

Compare lexicographically:

    TCG_EXACT_WARM
        -> Stela-driven TCG

vs.

    TCG_EXACT_WARM_CARLA
        -> Carla-driven TCG

on setAP-001 ... setAP-125.

For each instance, the FULL sorted load vector is reconstructed
and compared lexicographically.

A single figure is produced:

    lexicographic_winner_by_N.png

For each network size N = 100, 200, 300, 400, 500, the figure
shows:

    - Stela wins
    - Ties
    - Carla wins
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np


# =============================================================================
# Locate Algorithm_codes
# =============================================================================

def find_algorithm_codes(*starts: Path) -> Optional[Path]:

    seen = set()

    for start in starts:

        try:
            cur = Path(start).resolve()
        except Exception:
            continue

        for base in [cur, *cur.parents]:

            for candidate in (
                base,
                base / "Algorithm_codes",
            ):

                key = str(candidate)

                if key in seen:
                    continue

                seen.add(key)

                if (
                    (candidate / "core" / "ecmp.py").exists()
                    and
                    (candidate / "TCG_Exact").exists()
                ):
                    return candidate

    return None


# =============================================================================
# Parse instances
# =============================================================================

def parse_instances(spec: str) -> List[str]:

    spec = spec.strip()

    if ".." in spec:

        lo, hi = spec.split("..")

        pre_lo, n_lo = lo.rsplit("-", 1)
        pre_hi, n_hi = hi.rsplit("-", 1)

        if pre_lo != pre_hi:
            raise ValueError(
                f"Range prefixes differ: {pre_lo} vs {pre_hi}"
            )

        width = len(n_lo)

        return [
            f"{pre_lo}-{i:0{width}d}"
            for i in range(
                int(n_lo),
                int(n_hi) + 1
            )
        ]

    if "," in spec:

        return [
            x.strip()
            for x in spec.split(",")
            if x.strip()
        ]

    return [spec]


# =============================================================================
# Dataset
# =============================================================================

def resolve_instance_paths(
    instance: str,
    data_dir: str,
):

    from core.dataset import (
        add_dataset_args,
        resolve_paths,
    )

    parser = argparse.ArgumentParser(add_help=False)

    add_dataset_args(parser)

    args = parser.parse_args(
        [
            instance,
            "--data-dir",
            data_dir,
        ]
    )

    return resolve_paths(args)


def load_instance_and_splits(
    instance: str,
    data_dir: str,
):

    from core.instance import load_instance
    from core.ecmp import compute_all_splits

    net_f, tm_f, scen_f = resolve_instance_paths(
        instance,
        data_dir,
    )

    inst = load_instance(
        net_f,
        tm_f,
        scen_f,
    )

    r = compute_all_splits(inst)

    demand_period_keys = [
        (d_id, t)
        for (
            d_id,
            _,
            _,
            _
        ) in inst.demands
        for t in inst.T
    ]

    return inst, r, demand_period_keys


# =============================================================================
# SR paths -> waypoints
# =============================================================================

def srpaths_to_waypoints(
    srpaths_path: Path,
    demand_period_keys: List[Tuple],
) -> Dict[Tuple, List]:

    raw = json.loads(
        Path(srpaths_path).read_text()
    )

    entries = raw.get(
        "srpaths",
        []
    )

    valid = set(
        demand_period_keys
    )

    d_ids = {
        d_id
        for d_id, _ in demand_period_keys
    }

    dmap = {}

    for d in d_ids:

        dmap[d] = d
        dmap[str(d)] = d

        try:
            dmap[int(d)] = d
        except (
            TypeError,
            ValueError,
        ):
            pass

    waypoints = {
        key: []
        for key in demand_period_keys
    }

    for entry in entries:

        d_raw = entry.get("d")

        d_id = dmap.get(
            d_raw,
            dmap.get(str(d_raw))
        )

        if d_id is None:
            continue

        t = int(
            entry.get("t")
        )

        if (d_id, t) in valid:

            waypoints[
                (d_id, t)
            ] = list(
                entry.get("w", []) or []
            )

    return waypoints


# =============================================================================
# Reconstruct complete sorted load vector
# =============================================================================

def reconstruct_sorted_vector(
    srpaths_path: Path,
    inst,
    r,
    demand_period_keys,
) -> np.ndarray:

    from core.ecmp import (
        compute_loads,
        sorted_load_vector,
    )

    waypoints = srpaths_to_waypoints(
        srpaths_path,
        demand_period_keys,
    )

    loads = compute_loads(
        waypoints,
        inst,
        r,
    )

    vector = sorted_load_vector(
        loads,
        inst,
    )

    return np.asarray(
        vector,
        dtype=float,
    )


# =============================================================================
# Lexicographic comparison
# =============================================================================

def equal_with_tolerance(
    a: float,
    b: float,
    tau: float,
) -> bool:

    return (
        abs(a - b)
        <=
        tau * max(
            1.0,
            abs(b)
        )
    )


def lex_winner(
    vec_a: np.ndarray,
    vec_b: np.ndarray,
    tau: float,
) -> str:

    n = min(
        len(vec_a),
        len(vec_b)
    )

    for k in range(n):

        if not equal_with_tolerance(
            vec_a[k],
            vec_b[k],
            tau,
        ):

            if vec_a[k] < vec_b[k]:
                return "stela"

            return "carla"

    # Same values up to the common length.
    # Check remaining entries if lengths differ.
    if len(vec_a) != len(vec_b):

        longer = (
            vec_a
            if len(vec_a) > len(vec_b)
            else vec_b
        )

        for k in range(
            n,
            len(longer)
        ):

            if abs(longer[k]) > tau:

                if len(vec_a) > len(vec_b):
                    return "carla"

                return "stela"

    return "tie"


# =============================================================================
# N mapping
# =============================================================================

def instance_to_N(
    instance: str,
) -> Optional[int]:

    try:
        number = int(
            instance.rsplit(
                "-",
                1
            )[1]
        )
    except Exception:
        return None

    if 1 <= number <= 25:
        return 100

    if 26 <= number <= 50:
        return 200

    if 51 <= number <= 75:
        return 300

    if 76 <= number <= 100:
        return 400

    if 101 <= number <= 125:
        return 500

    return None


# =============================================================================
# Main comparison
# =============================================================================

def main(
    argv: Optional[List[str]] = None,
) -> int:

    parser = argparse.ArgumentParser(
        description=(
            "Lexicographic comparison of "
            "Stela-driven TCG and "
            "Carla-driven TCG."
        )
    )

    parser.add_argument(
        "--algo-a",
        default="TCG_EXACT_WARM",
    )

    parser.add_argument(
        "--algo-b",
        default="TCG_EXACT_WARM_CARLA",
    )

    parser.add_argument(
        "--codes-dir",
        default=None,
    )

    parser.add_argument(
        "--data-dir",
        default=None,
    )

    parser.add_argument(
        "--results-dir",
        default=None,
    )

    parser.add_argument(
        "--instances",
        default="setAP-001..setAP-125",
    )

    parser.add_argument(
        "--out-dir",
        default=".",
    )

    parser.add_argument(
        "--tau",
        type=float,
        default=1e-6,
    )

    parser.add_argument(
        "--usetex",
        action="store_true",
    )

    args = parser.parse_args(argv)

    # =========================================================================
    # Locate Algorithm_codes
    # =========================================================================

    if args.codes_dir:

        codes_dir = Path(
            args.codes_dir
        ).resolve()

    else:

        script_dir = (
            Path(__file__)
            .resolve()
            .parent
        )

        codes_dir = find_algorithm_codes(
            script_dir,
            Path.cwd(),
        )

        if codes_dir is None:

            print(
                "FATAL: could not locate Algorithm_codes."
            )

            print(
                "Use --codes-dir /path/to/Algorithm_codes"
            )

            return 2

    if not (
        codes_dir
        / "core"
        / "ecmp.py"
    ).exists():

        print(
            f"FATAL: {codes_dir} does not contain core/ecmp.py"
        )

        return 2

    # =========================================================================
    # Python path
    # =========================================================================

    if str(codes_dir) not in sys.path:

        sys.path.insert(
            0,
            str(codes_dir)
        )

    # =========================================================================
    # Paths
    # =========================================================================

    if args.results_dir:

        results_dir = Path(
            args.results_dir
        ).resolve()

    else:

        results_dir = (
            codes_dir
            / "results"
        )

    if args.data_dir:

        data_dir = str(
            Path(
                args.data_dir
            ).resolve()
        )

    else:

        data_dir = str(
            codes_dir.parent
            / "setAP"
        )

    out_dir = Path(
        args.out_dir
    ).resolve()

    out_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    # =========================================================================
    # Instances
    # =========================================================================

    instances = parse_instances(
        args.instances
    )

    # =========================================================================
    # Statistics
    # =========================================================================

    N_values = [
        100,
        200,
        300,
        400,
        500,
    ]

    stats = {
        N: {
            "stela": 0,
            "tie": 0,
            "carla": 0,
        }
        for N in N_values
    }

    # =========================================================================
    # Compare all instances
    # =========================================================================

    print()
    print("=" * 70)
    print("LEXICOGRAPHIC COMPARISON")
    print("=" * 70)

    for i, instance in enumerate(
        instances,
        start=1,
    ):

        N = instance_to_N(instance)

        if N is None:
            continue

        srpaths_a = (
            results_dir
            / instance
            / args.algo_a
            / f"{instance}-srpaths.json"
        )

        srpaths_b = (
            results_dir
            / instance
            / args.algo_b
            / f"{instance}-srpaths.json"
        )

        if not srpaths_a.exists():
            print(
                f"[{i:3d}/{len(instances)}] "
                f"{instance}: missing Stela"
            )
            continue

        if not srpaths_b.exists():
            print(
                f"[{i:3d}/{len(instances)}] "
                f"{instance}: missing Carla"
            )
            continue

        try:

            inst, r, demand_period_keys = (
                load_instance_and_splits(
                    instance,
                    data_dir,
                )
            )

            vec_a = reconstruct_sorted_vector(
                srpaths_a,
                inst,
                r,
                demand_period_keys,
            )

            vec_b = reconstruct_sorted_vector(
                srpaths_b,
                inst,
                r,
                demand_period_keys,
            )

            winner = lex_winner(
                vec_a,
                vec_b,
                args.tau,
            )

            stats[N][winner] += 1

            print(
                f"[{i:3d}/{len(instances)}] "
                f"{instance} (N={N}) : {winner}"
            )

        except Exception as e:

            print(
                f"[{i:3d}/{len(instances)}] "
                f"{instance}: FAILED -- "
                f"{type(e).__name__}: {e}"
            )

    # =========================================================================
    # Print final counts
    # =========================================================================

    print()
    print("=" * 70)
    print("FINAL LEXICOGRAPHIC RESULTS")
    print("=" * 70)

    print(
        f"{'N':>8}"
        f"{'Stela':>12}"
        f"{'Tie':>12}"
        f"{'Carla':>12}"
    )

    print("-" * 44)

    for N in N_values:

        print(
            f"{N:>8}"
            f"{stats[N]['stela']:>12}"
            f"{stats[N]['tie']:>12}"
            f"{stats[N]['carla']:>12}"
        )

    print("=" * 70)

    # =========================================================================
    # One single figure
    # =========================================================================

    import matplotlib

    matplotlib.use("Agg")

    import matplotlib.pyplot as plt

    if args.usetex:

        matplotlib.rcParams.update({
            "text.usetex": True,
            "font.family": "serif",
        })

    matplotlib.rcParams.update({
        "font.size": 11,
        "axes.titlesize": 12,
        "axes.labelsize": 11,
        "figure.dpi": 160,
        "axes.spines.top": False,
        "axes.spines.right": False,
    })

    x = np.arange(
        len(N_values)
    )

    stela = [
        stats[N]["stela"]
        for N in N_values
    ]

    ties = [
        stats[N]["tie"]
        for N in N_values
    ]

    carla = [
        stats[N]["carla"]
        for N in N_values
    ]

    fig, ax = plt.subplots(
        figsize=(7.2, 4.5)
    )

    width = 0.25

    bars_stela = ax.bar(
        x - width,
        stela,
        width,
        label=r"Stela-driven TCG",
        color="0.35",
        edgecolor="black",
        linewidth=1.0,
        hatch="///",
    )

    bars_tie = ax.bar(
        x,
        ties,
        width,
        label="Tie",
        color="0.65",
        edgecolor="black",
        linewidth=1.0,
    )

    bars_carla = ax.bar(
        x + width,
        carla,
        width,
        label=r"Carla-driven TCG",
        color="0.85",
        edgecolor="black",
        linewidth=1.0,
        hatch="...",
    )

    # Numbers above bars
    for bars in (
        bars_stela,
        bars_tie,
        bars_carla,
    ):

        for bar in bars:

            height = bar.get_height()

            ax.text(
                bar.get_x()
                + bar.get_width() / 2,
                height + 0.3,
                str(int(height)),
                ha="center",
                va="bottom",
                fontsize=10,
            )

    ax.set_xticks(x)

    ax.set_xticklabels(
        [
            rf"$N={N}$"
            for N in N_values
        ]
    )

    ax.set_xlabel(
        "$N$"
    )

    ax.set_ylabel(
        "Number of instances"
    )



    ax.legend(
        frameon=False,
        ncol=3,
        loc="upper center",
        bbox_to_anchor=(0.5, 1.13),
    )

    maximum = max(
        max(stela),
        max(ties),
        max(carla),
        1,
    )

    ax.set_ylim(
        0,
        maximum * 1.20
    )

    fig.tight_layout()

    output_file = (
        out_dir
        / "lexicographic_winner_by_N.png"
    )

    fig.savefig(
        output_file,
        bbox_inches="tight",
    )

    plt.close(fig)

    print()
    print(
        f"Figure written to:\n{output_file}"
    )

    return 0


# =============================================================================
# Entry point
# =============================================================================

if __name__ == "__main__":

    sys.exit(
        main()
    )