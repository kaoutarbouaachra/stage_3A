#!/usr/bin/env python3
"""
Step 2 -- boxplot of ranks solved by size class, plus paired statistics.

Reads summary.csv (from make_summary.py) and writes:

    fig_box.pdf/.png     ranks solved by size class, both algorithms
    by_class.csv         per-class aggregates
    table_depth.tex      LaTeX table 

Only instances present in both campaigns are used, so the comparison is paired.

Usage:
    python make_boxplot.py summary.csv --outdir figures
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
from scipy import stats

BASELINE, CANDIDATE = "ALEXA", "STELA"


def natural_key(name: str):
    # tuple, not list: pandas sort keys must be hashable
    return tuple(int(p) if p.isdigit() else p for p in re.split(r"(\d+)", name))


def pair(summary: pd.DataFrame) -> pd.DataFrame:
    """Wide table restricted to instances run by both algorithms."""
    wide = summary.pivot(index=["instance", "N"],
                         columns="algo", values=["k", "time", "status"])
    wide.columns = [f"{field}_{algo}" for field, algo in wide.columns]
    wide = wide.dropna(subset=[f"k_{BASELINE}", f"k_{CANDIDATE}"]).reset_index()
    wide[f"k_{BASELINE}"] = wide[f"k_{BASELINE}"].astype(int)
    wide[f"k_{CANDIDATE}"] = wide[f"k_{CANDIDATE}"].astype(int)
    wide["delta"] = wide[f"k_{CANDIDATE}"] - wide[f"k_{BASELINE}"]
    return wide.sort_values("instance", key=lambda s: s.map(natural_key))


def aggregate(paired: pd.DataFrame) -> pd.DataFrame:
    def agg(g):
        return pd.Series({
            "n_inst": len(g),
            f"median_{BASELINE}": g[f"k_{BASELINE}"].median(),
            f"median_{CANDIDATE}": g[f"k_{CANDIDATE}"].median(),
            f"mean_{BASELINE}": round(g[f"k_{BASELINE}"].mean(), 1),
            f"mean_{CANDIDATE}": round(g[f"k_{CANDIDATE}"].mean(), 1),
            f"zero_{BASELINE}": int((g[f"k_{BASELINE}"] == 0).sum()),
            f"zero_{CANDIDATE}": int((g[f"k_{CANDIDATE}"] == 0).sum()),
            f"conv_{BASELINE}": int((g[f"status_{BASELINE}"] == "CONVERGED").sum()),
            f"conv_{CANDIDATE}": int((g[f"status_{CANDIDATE}"] == "CONVERGED").sum()),
            "wins": int((g.delta > 0).sum()),
            "ties": int((g.delta == 0).sum()),
            "losses": int((g.delta < 0).sum()),
        })

    by_class = paired.groupby("N").apply(agg, include_groups=False)
    by_class.loc["all"] = agg(paired)
    return by_class


def paired_stats(paired: pd.DataFrame, seed: int = 0) -> dict:
    rng = np.random.default_rng(seed)

    def boot(values, func, n=5000):
        values = np.asarray(values, dtype=float)
        draws = [func(values[rng.integers(0, len(values), len(values))]) for _ in range(n)]
        return tuple(np.percentile(draws, [2.5, 97.5]))

    untied = paired[paired.delta != 0]
    ratio = ((paired[f"k_{CANDIDATE}"] + 1) / (paired[f"k_{BASELINE}"] + 1)).to_numpy()
    wilcoxon = stats.wilcoxon(paired[f"k_{CANDIDATE}"], paired[f"k_{BASELINE}"],
                              alternative="greater", zero_method="wilcox")
    return {
        "wins": int((paired.delta > 0).sum()),
        "ties": int((paired.delta == 0).sum()),
        "ties_at_zero": int(((paired.delta == 0) & (paired[f"k_{CANDIDATE}"] == 0)).sum()),
        "losses": int((paired.delta < 0).sum()),
        "loss_instances": paired.loc[paired.delta < 0, "instance"].tolist(),
        "wilcoxon_p": float(wilcoxon.pvalue),
        "sign_p": float(stats.binomtest(int((untied.delta > 0).sum()), len(untied),
                                        0.5, alternative="greater").pvalue) if len(untied) else float("nan"),
        "median_delta": float(paired.delta.median()),
        "median_delta_ci": boot(paired.delta, np.median),
        "geomean": float(stats.gmean(ratio)),
        "geomean_ci": boot(ratio, stats.gmean),
    }


def draw(paired: pd.DataFrame, outdir: Path) -> None:
    plt.rcParams.update({"font.family": "serif", "font.serif": ["DejaVu Serif"],
                         "mathtext.fontset": "dejavuserif", "font.size": 9,
                         "axes.linewidth": 0.7})
    classes = sorted(paired.N.unique())
    positions = np.arange(len(classes))
    fig, ax = plt.subplots(figsize=(3.4, 2.6))

    for offset, algo, facecolour in [(-0.18, BASELINE, "0.85"), (0.18, CANDIDATE, "0.45")]:
        data = [paired.loc[paired.N == c, f"k_{algo}"].to_numpy() for c in classes]
        bp = ax.boxplot(data, positions=positions + offset, widths=0.30,
                        patch_artist=True,
                        flierprops=dict(marker=".", ms=3, mec="0.4"),
                        medianprops=dict(color="k", lw=1.1))
        for box in bp["boxes"]:
            box.set(facecolor=facecolour, lw=0.6)
        for line in bp["whiskers"] + bp["caps"]:
            line.set(lw=0.6)
        ax.plot([], [], "s", c=facecolour, mec="k", ms=6, label=algo)

    ax.set_xticks(positions)
    ax.set_xticklabels(classes)
    ax.set_xlabel(r"$N$")
    ax.set_ylabel("ranks solved")
    ax.legend(frameon=False, fontsize=8)
    ax.grid(axis="y", alpha=0.25, lw=0.5)
    ax.tick_params(labelsize=7.5)
    fig.tight_layout()
    fig.savefig(outdir / "fig_box.pdf")
    fig.savefig(outdir / "fig_box.png", dpi=200)
    plt.close(fig)


def write_latex(by_class: pd.DataFrame, path: Path) -> None:
    body = []
    for idx, row in by_class.iterrows():
        if idx == "all":
            body.append(r"\midrule")
        name = r"\textbf{all}" if idx == "all" else str(idx)
        body.append(
            f"{name} & {int(row.n_inst)} & "
            f"{row[f'median_{BASELINE}']:.0f} & {row[f'median_{CANDIDATE}']:.0f} & "
            f"{int(row[f'zero_{BASELINE}'])} & {int(row[f'zero_{CANDIDATE}'])} & "
            f"{int(row[f'conv_{BASELINE}'])} & {int(row[f'conv_{CANDIDATE}'])} & "
            f"{int(row.wins)}/{int(row.ties)}/{int(row.losses)} \\\\")

    path.write_text(
        "% generated by make_boxplot.py\n"
        "\\begin{table}[t]\\centering\n"
        "\\caption{Ranks of the sorted load vector solved within the per-instance "
        "budget. $\\#\\{k=0\\}$ counts instances on which no rank was solved; "
        "\\emph{conv.} counts runs that determined the entire vector. W/T/L is the "
        f"paired outcome of {CANDIDATE} against {BASELINE}.}}\n"
        "\\label{tab:depth}\\small\n"
        "\\begin{tabular}{@{}lrrrrrrrr@{}}\n\\toprule\n"
        "& & \\multicolumn{2}{c}{median $k$} & \\multicolumn{2}{c}{$\\#\\{k=0\\}$}"
        " & \\multicolumn{2}{c}{conv.} & \\\\\n"
        "\\cmidrule(lr){3-4}\\cmidrule(lr){5-6}\\cmidrule(lr){7-8}\n"
        f"$|L|$ & \\#inst & {BASELINE[0]} & {CANDIDATE[0]} & {BASELINE[0]} & "
        f"{CANDIDATE[0]} & {BASELINE[0]} & {CANDIDATE[0]} & W/T/L \\\\\n\\midrule\n"
        + "\n".join(body)
        + "\n\\bottomrule\n\\end{tabular}\n\\end{table}\n", encoding="utf-8")


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("summary", type=Path, nargs="?", default=Path("summary.csv"))
    parser.add_argument("--outdir", type=Path, default=Path("."))
    args = parser.parse_args(argv)
    args.outdir.mkdir(parents=True, exist_ok=True)

    summary = pd.read_csv(args.summary)
    paired = pair(summary)
    if paired.empty:
        raise SystemExit("no instance present in both campaigns")

    by_class = aggregate(paired)
    result = paired_stats(paired)

    by_class.to_csv(args.outdir / "by_class.csv")
    write_latex(by_class, args.outdir / "table_depth.tex")
    draw(paired, args.outdir)

    print(f"paired instances: {len(paired)}\n")
    print(by_class.to_string())

    print(f"\n=== {CANDIDATE} vs {BASELINE} ===")
    print(f"  W/T/L                : {result['wins']}/{result['ties']}/{result['losses']}")
    print(f"  ties at zero ranks   : {result['ties_at_zero']}")
    if result["loss_instances"]:
        print(f"  losses               : {result['loss_instances']}")
    print(f"  Wilcoxon             : p={result['wilcoxon_p']:.3e}")
    print(f"  sign test            : p={result['sign_p']:.3e}")
    lo, hi = result["median_delta_ci"]
    print(f"  median rank gain     : {result['median_delta']:+.0f} [{lo:+.0f}, {hi:+.0f}]")
    lo, hi = result["geomean_ci"]
    print(f"  geometric mean ratio : {result['geomean']:.2f}x [{lo:.2f}, {hi:.2f}]")

    print(f"\nwritten to {args.outdir.resolve()}")
    return 0


if __name__ == "__main__":
    sys.exit(main())