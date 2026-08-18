#!/usr/bin/env python3

"""
by_how_much.py

Compare the complete sorted load vectors produced by:

    TCG_EXACT_WARM
        -> Stela-driven TCG

vs.

    TCG_EXACT_WARM_CARLA
        -> Carla-driven TCG

For every instance:

    1. Reconstruct the complete sorted load vectors.
    2. Find the first rank k* at which they differ.
    3. Compute the difference at that rank:

           Delta = lambda_Carla[k*] - lambda_Stela[k*]

    4. Aggregate the absolute differences by network size N.

A single publication-style figure is generated:

    by_how_much_at_first_difference_by_N.png

The figure reports:

    |lambda_Carla[k*] - lambda_Stela[k*]|

for:

    N = 100
    N = 200
    N = 300
    N = 400
    N = 500

Ties are excluded because they have no first differing rank.
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
                f"Range prefixes differ: "
                f"{pre_lo} vs {pre_hi}"
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

    parser = argparse.ArgumentParser(
        add_help=False
    )

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

    return (
        inst,
        r,
        demand_period_keys,
    )


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
                entry.get(
                    "w",
                    []
                ) or []
            )

    return waypoints


# =============================================================================
# Reconstruct complete sorted vector
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
# Numerical comparison
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


def first_differing_rank(
    vec_stela: np.ndarray,
    vec_carla: np.ndarray,
    tau: float,
) -> Optional[int]:

    n = min(
        len(vec_stela),
        len(vec_carla)
    )

    for k in range(n):

        if not equal_with_tolerance(
            vec_stela[k],
            vec_carla[k],
            tau,
        ):

            return k + 1

    # If lengths differ, check remaining entries.
    if len(vec_stela) != len(vec_carla):

        longer = (
            vec_stela
            if len(vec_stela) > len(vec_carla)
            else vec_carla
        )

        for k in range(
            n,
            len(longer)
        ):

            if abs(longer[k]) > tau:

                return k + 1

    return None


# =============================================================================
# Determine N
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
# Main
# =============================================================================

def main(
    argv: Optional[List[str]] = None,
) -> int:

    parser = argparse.ArgumentParser(
        description=(
            "Measure by how much the Stela-driven and "
            "Carla-driven TCG load vectors differ "
            "at their first differing rank."
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
            f"FATAL: {codes_dir} "
            "does not contain core/ecmp.py"
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
    # Store differences by N
    # =========================================================================

    N_values = [
        100,
        200,
        300,
        400,
        500,
    ]

    abs_diff_by_N = {
        N: []
        for N in N_values
    }

    signed_diff_by_N = {
        N: []
        for N in N_values
    }

    rank_by_N = {
        N: []
        for N in N_values
    }

    ties_by_N = {
        N: 0
        for N in N_values
    }

    # =========================================================================
    # Process instances
    # =========================================================================

    print()
    print("=" * 80)
    print("BY HOW MUCH DO THEY DIFFER?")
    print("=" * 80)

    for i, instance in enumerate(
        instances,
        start=1,
    ):

        N = instance_to_N(
            instance
        )

        if N is None:
            continue

        srpaths_stela = (
            results_dir
            / instance
            / args.algo_a
            / f"{instance}-srpaths.json"
        )

        srpaths_carla = (
            results_dir
            / instance
            / args.algo_b
            / f"{instance}-srpaths.json"
        )

        # ---------------------------------------------------------------------
        # Missing Stela
        # ---------------------------------------------------------------------

        if not srpaths_stela.exists():

            print(
                f"[{i:3d}/{len(instances)}] "
                f"{instance}: missing Stela"
            )

            continue

        # ---------------------------------------------------------------------
        # Missing Carla
        # ---------------------------------------------------------------------

        if not srpaths_carla.exists():

            print(
                f"[{i:3d}/{len(instances)}] "
                f"{instance}: missing Carla"
            )

            continue

        try:

            # ---------------------------------------------------------------
            # Load instance
            # ---------------------------------------------------------------

            (
                inst,
                r,
                demand_period_keys,
            ) = load_instance_and_splits(
                instance,
                data_dir,
            )

            # ---------------------------------------------------------------
            # Reconstruct vectors
            # ---------------------------------------------------------------

            vec_stela = reconstruct_sorted_vector(
                srpaths_stela,
                inst,
                r,
                demand_period_keys,
            )

            vec_carla = reconstruct_sorted_vector(
                srpaths_carla,
                inst,
                r,
                demand_period_keys,
            )

            # ---------------------------------------------------------------
            # First differing rank
            # ---------------------------------------------------------------

            k_star = first_differing_rank(
                vec_stela,
                vec_carla,
                args.tau,
            )

            # ---------------------------------------------------------------
            # Tie
            # ---------------------------------------------------------------

            if k_star is None:

                ties_by_N[N] += 1

                print(
                    f"[{i:3d}/{len(instances)}] "
                    f"{instance} (N={N}) : "
                    f"TIE"
                )

                continue

            # ---------------------------------------------------------------
            # Values at k*
            # ---------------------------------------------------------------

            k = k_star - 1

            stela_value = vec_stela[k]
            carla_value = vec_carla[k]

            signed_diff = (
                carla_value
                -
                stela_value
            )

            abs_diff = abs(
                signed_diff
            )

            # ---------------------------------------------------------------
            # Store
            # ---------------------------------------------------------------

            abs_diff_by_N[N].append(
                abs_diff
            )

            signed_diff_by_N[N].append(
                signed_diff
            )

            rank_by_N[N].append(
                k_star
            )

            # ---------------------------------------------------------------
            # Print
            # ---------------------------------------------------------------

            print(
                f"[{i:3d}/{len(instances)}] "
                f"{instance} (N={N}) : "
                f"k* = {k_star:3d} | "
                f"Stela = {stela_value:.8f} | "
                f"Carla = {carla_value:.8f} | "
                f"Δ = {signed_diff:+.8f} | "
                f"|Δ| = {abs_diff:.8f}"
            )

        except Exception as e:

            print(
                f"[{i:3d}/{len(instances)}] "
                f"{instance}: FAILED -- "
                f"{type(e).__name__}: {e}"
            )

    # =========================================================================
    # Summary
    # =========================================================================

    print()
    print("=" * 100)
    print("SUMMARY: DIFFERENCE AT THE FIRST DIFFERING RANK")
    print("=" * 100)

    print(
        f"{'N':>8}"
        f"{'Instances':>12}"
        f"{'Median |Δ|':>18}"
        f"{'Mean |Δ|':>18}"
        f"{'Min |Δ|':>18}"
        f"{'Max |Δ|':>18}"
        f"{'Median k*':>14}"
        f"{'Ties':>10}"
    )

    print("-" * 100)

    for N in N_values:

        values = abs_diff_by_N[N]
        ranks = rank_by_N[N]

        if values:

            print(
                f"{N:>8}"
                f"{len(values):>12}"
                f"{np.median(values):>18.8f}"
                f"{np.mean(values):>18.8f}"
                f"{min(values):>18.8f}"
                f"{max(values):>18.8f}"
                f"{np.median(ranks):>14.1f}"
                f"{ties_by_N[N]:>10}"
            )

        else:

            print(
                f"{N:>8}"
                f"{0:>12}"
                f"{'--':>18}"
                f"{'--':>18}"
                f"{'--':>18}"
                f"{'--':>18}"
                f"{'--':>14}"
                f"{ties_by_N[N]:>10}"
            )

    print("=" * 100)

    # =========================================================================
    # Plot
    # =========================================================================

    import matplotlib

    matplotlib.use("Agg")

    import matplotlib.pyplot as plt

    matplotlib.rcParams.update({
        "font.size": 11,
        "axes.titlesize": 12,
        "axes.labelsize": 11,
        "figure.dpi": 160,
        "axes.spines.top": False,
        "axes.spines.right": False,
    })

    data = [
        abs_diff_by_N[N]
        for N in N_values
    ]

    # -------------------------------------------------------------------------
    # Remove empty groups
    # -------------------------------------------------------------------------

    valid_positions = []
    valid_data = []
    valid_labels = []

    for position, N in enumerate(
        N_values,
        start=1,
    ):

        if len(abs_diff_by_N[N]) > 0:

            valid_positions.append(
                position
            )

            valid_data.append(
                abs_diff_by_N[N]
            )

            valid_labels.append(
                rf"$N={N}$"
            )

    # =========================================================================
    # Figure
    # =========================================================================

    fig, ax = plt.subplots(
        figsize=(7.2, 4.5)
    )

    bp = ax.boxplot(
        valid_data,
        positions=valid_positions,
        widths=0.55,
        patch_artist=True,
        showfliers=True,
        medianprops={
            "color": "black",
            "linewidth": 1.5,
        },
        boxprops={
            "edgecolor": "black",
            "linewidth": 1.0,
        },
        whiskerprops={
            "color": "black",
            "linewidth": 1.0,
        },
        capprops={
            "color": "black",
            "linewidth": 1.0,
        },
        flierprops={
            "marker": "o",
            "markersize": 4,
            "markerfacecolor": "white",
            "markeredgecolor": "black",
        },
    )

    for box in bp["boxes"]:

        box.set_facecolor(
            "0.80"
        )

    ax.set_xticks(
        valid_positions
    )

    ax.set_xticklabels(
        valid_labels
    )

    ax.set_xlabel(
        "$N$"
    )

    ax.set_ylabel(
        r"$|\lambda_{k^\star}^{\mathrm{Carla}}"
        r"-\lambda_{k^\star}^{\mathrm{Stela}}|$"
    )

    ax.grid(
        axis="y",
        linestyle=":",
        linewidth=0.7,
        alpha=0.6,
    )

    # -------------------------------------------------------------------------
    # Scientific notation if necessary
    # -------------------------------------------------------------------------

    ax.ticklabel_format(
        axis="y",
        style="sci",
        scilimits=(0, 0),
    )

    fig.tight_layout()

    output_file = (
        out_dir
        / "by_how_much_at_first_difference_by_N.png"
    )

    fig.savefig(
        output_file,
        bbox_inches="tight",
    )

    plt.close(fig)

    print()
    print(
        "[figure] Written to:"
    )

    print(
        output_file
    )

    return 0


# =============================================================================
# Entry point
# =============================================================================

if __name__ == "__main__":

    sys.exit(
        main()
    )