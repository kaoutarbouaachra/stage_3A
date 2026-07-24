#!/usr/bin/env python3
"""
Step 3 -- lexicographic value vectors: side-by-side table, figure, divergence report.

Reads vectors.csv (from make_summary.py) and writes:

    vectors_wide.csv   one row per (instance, rank): value_ALEXA, value_STELA,
                       abs_diff, rel_diff_pct, agree
    divergence.csv     one row per instance where the two vectors disagree:
                       first divergent rank and worst relative gap
    fig_vectors.pdf    one panel per size class, both vectors overlaid


Usage:
    python make_vectors.py vectors.csv --outdir figures --ranks 30
    python make_vectors.py vectors.csv --instance setAP-03      # single instance
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

BASELINE, CANDIDATE = "ALEXA", "STELA"


def natural_key(name: str):
    # tuple, not list: pandas sort keys must be hashable
    return tuple(int(p) if p.isdigit() else p for p in re.split(r"(\d+)", name))


def widen(vectors: pd.DataFrame, tol: float) -> pd.DataFrame:
    """One row per (instance, rank) with both values side by side."""
    wide = vectors.pivot_table(index=["instance", "rank"], columns="algo",
                               values="value", aggfunc="first")
    wide = wide.rename(columns={BASELINE: f"value_{BASELINE}",
                                CANDIDATE: f"value_{CANDIDATE}"}).reset_index()
    for column in (f"value_{BASELINE}", f"value_{CANDIDATE}"):
        if column not in wide:
            wide[column] = np.nan

    both = wide[f"value_{BASELINE}"].notna() & wide[f"value_{CANDIDATE}"].notna()
    wide["abs_diff"] = np.where(both, wide[f"value_{BASELINE}"] - wide[f"value_{CANDIDATE}"], np.nan)
    wide["rel_diff_pct"] = np.where(
        both & (wide[f"value_{CANDIDATE}"].abs() > 1e-12),
        100 * wide.abs_diff / wide[f"value_{CANDIDATE}"], np.nan)
    wide["agree"] = np.where(both, wide.abs_diff.abs() <= tol, pd.NA)
    return wide.sort_values(["instance", "rank"],
                            key=lambda s: s.map(natural_key) if s.name == "instance" else s)


def report(wide: pd.DataFrame) -> pd.DataFrame:
    """First divergent rank and worst relative gap, per instance."""
    rows = []
    for instance, group in wide.groupby("instance"):
        common = group[group.agree.notna()]
        bad = common[~common.agree.astype(bool)]
        if bad.empty:
            continue
        worst = bad.loc[bad.rel_diff_pct.abs().idxmax()]
        rows.append({
            "instance": instance,
            "ranks_compared": len(common),
            "n_divergent": len(bad),
            "first_divergent_rank": int(bad["rank"].min()),
            "worst_rank": int(worst["rank"]),
            "worst_rel_gap_pct": float(worst.rel_diff_pct),
        })
    frame = pd.DataFrame(rows)
    if not frame.empty:
        frame = frame.sort_values("worst_rel_gap_pct", key=abs, ascending=False)
    return frame


def pick_representatives(wide: pd.DataFrame, summary: pd.DataFrame | None) -> dict:
    """Median instance per size class by depth gap; falls back to ranks compared."""
    depth = wide.groupby("instance").agg(
        n_base=(f"value_{BASELINE}", "count"),
        n_cand=(f"value_{CANDIDATE}", "count")).reset_index()
    if summary is not None:
        sizes = summary.drop_duplicates("instance").set_index("instance").N
        depth["N"] = depth.instance.map(sizes)
    else:
        depth["N"] = 0
    depth = depth.dropna(subset=["N"])
    depth["gap"] = depth.n_cand - depth.n_base

    return {int(n): sorted(zip(group.gap, group.instance))[(len(group) - 1) // 2][1]
            for n, group in depth.groupby("N")}


def draw(wide: pd.DataFrame, reps: dict, n_ranks: int, outdir: Path) -> None:
    plt.rcParams.update({"font.family": "serif", "font.serif": ["DejaVu Serif"],
                         "mathtext.fontset": "dejavuserif", "font.size": 9,
                         "axes.linewidth": 0.7})
    classes = sorted(reps)
    fig, axes = plt.subplots(1, len(classes), figsize=(min(7.0, 2.4 * len(classes)), 2.0),
                             squeeze=False)
    axes = axes[0]

    for ax, size in zip(axes, classes):
        instance = reps[size]
        block = wide[wide.instance == instance]
        block = block[block["rank"] <= n_ranks]
        common = block[block.agree.notna()]["rank"]
        if len(common):
            ax.axvspan(0.5, common.max() + 0.5, color="0.91", lw=0, zorder=0)
        for algo, colour, style, marker in [(CANDIDATE, "k", "-", "o"),
                                            (BASELINE, "0.5", "--", "s")]:
            series = block.dropna(subset=[f"value_{algo}"])
            ax.plot(series["rank"], series[f"value_{algo}"], style, color=colour,
                    lw=1.0, marker=marker, ms=2.4, mew=0)
        ax.set_title(rf"$|L|={size}$ ({instance})", fontsize=7.5, pad=3)
        ax.set_xlabel("rank $j$", fontsize=8)
        ax.grid(alpha=0.22, lw=0.4)
        ax.tick_params(labelsize=6.5)

    axes[0].set_ylabel(r"$\bar{\lambda}_j$", fontsize=8)
    handles = [
        plt.Line2D([], [], color="k", ls="-", marker="o", ms=3, mew=0, label=CANDIDATE),
        plt.Line2D([], [], color="0.5", ls="--", marker="s", ms=3, mew=0, label=BASELINE),
        plt.Rectangle((0, 0), 1, 1, fc="0.91", ec="none", label="solved by both"),
    ]
    fig.legend(handles=handles, loc="lower center", ncol=3, frameon=False,
               fontsize=7.5, bbox_to_anchor=(0.5, -0.06))
    fig.tight_layout(rect=(0, 0.07, 1, 1))
    fig.savefig(outdir / "fig_vectors.pdf", bbox_inches="tight")
    fig.savefig(outdir / "fig_vectors.png", dpi=200, bbox_inches="tight")
    plt.close(fig)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("vectors", type=Path, nargs="?", default=Path("vectors.csv"))
    parser.add_argument("--summary", type=Path, default=None,
                        help="summary.csv, used to group representatives by size class")
    parser.add_argument("--outdir", type=Path, default=Path("."))
    parser.add_argument("--ranks", type=int, default=30, help="ranks shown in the figure")
    parser.add_argument("--tol", type=float, default=2e-6,
                        help="below this, a difference is display rounding")
    parser.add_argument("--instance", default=None,
                        help="print the full side-by-side table for one instance")
    args = parser.parse_args(argv)
    args.outdir.mkdir(parents=True, exist_ok=True)

    vectors = pd.read_csv(args.vectors)
    wide = widen(vectors, args.tol)

    summary_path = args.summary or args.vectors.with_name("summary.csv")
    summary = pd.read_csv(summary_path) if summary_path.exists() else None

    if args.instance:
        block = wide[wide.instance == args.instance]
        if block.empty:
            raise SystemExit(f"instance not found: {args.instance}")
        pd.set_option("display.width", 120)
        print(block.to_string(index=False, float_format=lambda v: f"{v:.6f}"))
        return 0

    wide.to_csv(args.outdir / "vectors_wide.csv", index=False)
    divergence = report(wide)
    divergence.to_csv(args.outdir / "divergence.csv", index=False)

    reps = pick_representatives(wide, summary)
    if reps:
        draw(wide, reps, args.ranks, args.outdir)
        print("representatives (median depth gap per size class):")
        for size, instance in sorted(reps.items()):
            print(f"  |L|={size}: {instance}")

    compared = wide.agree.notna().sum()
    agreeing = (wide.agree == True).sum()  # noqa: E712
    print(f"\nranks compared on both sides : {compared}")
    print(f"  agreeing (tol={args.tol:.0e})      : {agreeing} "
          f"({100 * agreeing / max(compared, 1):.1f}%)")

    n_instances = wide.instance.nunique()
    if divergence.empty:
        print("\nno disagreement: the two encodings return identical vectors")
    else:
        print(f"\ndisagreeing instances: {len(divergence)}/{n_instances}")
        print(divergence.head(20).to_string(index=False,
                                            float_format=lambda v: f"{v:.3f}"))
        print("\nNote: by uniqueness of the leximin value vector, two correct "
              "implementations\nmust agree on every rank both of them solved. "
              "Any row above is a bug.")

    print(f"\nwritten to {args.outdir.resolve()}")
    return 0


if __name__ == "__main__":
    sys.exit(main())