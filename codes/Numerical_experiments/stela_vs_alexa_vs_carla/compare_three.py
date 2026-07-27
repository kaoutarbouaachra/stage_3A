#!/usr/bin/env python3
r"""
Three-way comparison of exact leximin algorithms (STELA / CARLA / ALEXA).

Parses the expe3 log files named ``<algo>-setAP-<NNN>-log.txt`` and reproduces,
for three algorithms, the certified-ranks boxplot previously made for
ALEXA vs STELA: number of lexicographic ranks certified within the wall-clock
limit, grouped by instance size class N.

Usage (PowerShell):
    python compare_three.py --logs logs\expe3 --outdir figs --format pdf

Outputs:
    summary.csv          one row per (algo, instance): N, ranks, time, mlu, timed_out
    fig_box_three.<fmt>  grouped boxplot of certified ranks by size class
    (stdout)             head-to-head counts and per-class medians
"""

from __future__ import annotations
import argparse
import re
import sys
from pathlib import Path

import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# --- fixed algorithm order and sober grayscale fills (print-friendly) ---------
ALGOS = ["STELA", "CARLA", "ALEXA"]
FILLS = {"STELA": "#d9d9d9", "CARLA": "#a6a6a6", "ALEXA": "#6f6f6f"}

# where the logs live relative to the repo root (used for auto-detection)
LOGS_REL = Path("Algorithm_codes") / "logs" / "expe3"


def resolve_logs_dir(override):
    """Return the logs directory.

    If ``override`` is given, use it. Otherwise walk up from the script's
    directory (and from the current working directory) looking for
    ``Algorithm_codes/logs/expe3``.
    """
    if override is not None:
        return override
    bases = [Path(__file__).resolve().parent, Path.cwd().resolve()]
    for base in bases:
        for parent in [base, *base.parents]:
            cand = parent / LOGS_REL
            if cand.is_dir():
                return cand
    return LOGS_REL  # last resort; will fail loudly downstream


# --- regexes ------------------------------------------------------------------
RE_INSTANCE = re.compile(r"(setAP-\d+)")
RE_N        = re.compile(r"\bN=(\d+)")
RE_DONE     = re.compile(r"Done\.\s+(\d+)\s+(?:iterations|ranks)")
RE_DONE_T   = re.compile(r"Done\.\s+\d+\s+(?:iterations|ranks)(?:\s+certified)?\s+in\s+([\d.]+)s")
RE_MLU      = re.compile(r"MLU\s*\([^)]*\)\s*=\s*([\d.]+)")
RE_STATUS11 = re.compile(r"Unexpected status 11")
# a per-rank data row: starts (after whitespace) with the rank index, ends OPTIMAL
RE_OPTROW   = re.compile(r"^\s*\d+\s+.*\bOPTIMAL\b")


def natural_key(name: str):
    """Sort 'setAP-2' before 'setAP-10'."""
    m = re.search(r"(\d+)", name)
    return (name[: m.start()], int(m.group(1))) if m else (name, 0)


def parse_log(path: Path, algo: str) -> dict | None:
    text = path.read_text(encoding="utf-8", errors="replace")

    m_inst = RE_INSTANCE.search(text)
    if m_inst is None:
        return None
    instance = m_inst.group(1)

    m_n = RE_N.search(text)
    N = int(m_n.group(1)) if m_n else None

    # certified ranks: trust the "Done. N ..." line; cross-check against the
    # number of OPTIMAL rows in the per-rank table.
    m_done = RE_DONE.search(text)
    n_optrows = sum(1 for ln in text.splitlines() if RE_OPTROW.match(ln))
    if m_done is not None:
        ranks = int(m_done.group(1))
        if ranks != n_optrows:
            print(f"  [warn] {path.name}: Done={ranks} but {n_optrows} OPTIMAL rows",
                  file=sys.stderr)
    else:
        ranks = n_optrows  # fallback: no Done line (e.g. rank 1 itself timed out)

    m_t = RE_DONE_T.search(text)
    solve_time = float(m_t.group(1)) if m_t else None

    m_mlu = RE_MLU.search(text)
    mlu = float(m_mlu.group(1)) if m_mlu else None

    timed_out = bool(RE_STATUS11.search(text))

    return {
        "algo": algo,
        "instance": instance,
        "N": N,
        "ranks": ranks,
        "solve_time": solve_time,
        "mlu": mlu,
        "timed_out": timed_out,
    }


def load(logs_dir: Path) -> pd.DataFrame:
    if not logs_dir.is_dir():
        raise SystemExit(f"logs directory not found: {logs_dir}")
    rows, skipped = [], []
    for path in sorted(logs_dir.glob("*-setAP-*-log.txt")):
        prefix = path.name.split("-", 1)[0].lower()
        algo = {"stela": "STELA", "carla": "CARLA", "alexa": "ALEXA"}.get(prefix)
        if algo is None:
            skipped.append(path.name)
            continue
        parsed = parse_log(path, algo)
        (rows if parsed else skipped).append(parsed or path.name)
    if not rows:
        raise SystemExit(f"no parsable logs found in {logs_dir}")
    if skipped:
        print(f"skipped {len(skipped)} file(s): {skipped[:5]}", file=sys.stderr)
    return pd.DataFrame(rows)


def restrict_to_triplets(df: pd.DataFrame) -> pd.DataFrame:
    """Keep only instances present for all three algorithms (paired comparison)."""
    present = df.groupby("instance").algo.nunique()
    complete = set(present[present == len(ALGOS)].index)
    dropped = sorted(set(df.instance) - complete, key=natural_key)
    if dropped:
        print(f"[info] {len(dropped)} instance(s) not present for all three algos, "
              f"excluded: {dropped[:8]}")
    return df[df.instance.isin(complete)].copy()


def head_to_head(df: pd.DataFrame, a: str, b: str) -> None:
    wide = df.pivot(index="instance", columns="algo", values="ranks")
    d = wide[a] - wide[b]
    print(f"  {a} vs {b}:  {a} wins {int((d > 0).sum())}, "
          f"ties {int((d == 0).sum())}, losses {int((d < 0).sum())}")


def make_boxplot(df: pd.DataFrame, out: Path) -> None:
    classes = sorted(df.N.dropna().unique())
    n_cls, n_alg = len(classes), len(ALGOS)
    width = 0.8 / n_alg

    fig, ax = plt.subplots(figsize=(3.4, 2.5))  # ~ \columnwidth, IEEEtran
    for j, algo in enumerate(ALGOS):
        data = [df[(df.N == c) & (df.algo == algo)].ranks.values for c in classes]
        positions = [i + (j - (n_alg - 1) / 2) * width for i in range(n_cls)]
        bp = ax.boxplot(
            data, positions=positions, widths=width * 0.9,
            patch_artist=True, showfliers=False,
            medianprops=dict(color="black", linewidth=1.1),
            whiskerprops=dict(color="black", linewidth=0.7),
            capprops=dict(color="black", linewidth=0.7),
            boxprops=dict(color="black", linewidth=0.7),
        )
        for box in bp["boxes"]:
            box.set_facecolor(FILLS[algo])
        # single legend proxy per algorithm
        ax.plot([], [], "s", color=FILLS[algo], markeredgecolor="black",
                markersize=7, label=algo)

    ax.set_xticks(range(n_cls))
    ax.set_xticklabels([f"{int(c)}" for c in classes])
    ax.set_xlabel(r"instance size $N=|\mathcal{A}_T|$")
    ax.set_ylabel("certified ranks")
    ax.set_ylim(bottom=0)
    ax.legend(frameon=False, fontsize=7, loc="upper right", handletextpad=0.3)
    ax.grid(axis="y", linewidth=0.4, alpha=0.5)
    fig.tight_layout()
    fig.savefig(out, bbox_inches="tight")
    print(f"figure written -> {out}")


def main(argv=None) -> int:
    script_dir = Path(__file__).resolve().parent
    ap = argparse.ArgumentParser()
    ap.add_argument("--logs", type=Path, default=None,
                    help="logs dir (default: auto-detect Algorithm_codes/logs/expe3)")
    ap.add_argument("--outdir", type=Path, default=None,
                    help="output dir (default: next to this script)")
    ap.add_argument("--format", default="pdf", choices=["pdf", "png"])
    args = ap.parse_args(argv)

    logs_dir = resolve_logs_dir(args.logs)
    outdir = args.outdir if args.outdir is not None else script_dir
    outdir.mkdir(parents=True, exist_ok=True)
    print(f"logs   : {logs_dir}")
    print(f"outdir : {outdir}")

    # Computer Modern, sober academic look
    plt.rcParams.update({
        "font.family": "serif",
        "mathtext.fontset": "cm",
        "font.size": 8,
        "axes.linewidth": 0.6,
        "figure.dpi": 200,
    })

    df = load(logs_dir)
    df.sort_values(["algo", "instance"],
                   key=lambda s: s.map(natural_key) if s.name == "instance" else s,
                   inplace=True)
    df.to_csv(outdir / "summary.csv", index=False)

    print("\nruns per algorithm:")
    print(df.groupby("algo").size().to_string())

    df = restrict_to_triplets(df)
    if df.empty:
        raise SystemExit("no instance is present for all three algorithms")

    print("\nmedian certified ranks by size class:")
    print(df.pivot_table(index="N", columns="algo", values="ranks",
                         aggfunc="median").reindex(columns=ALGOS).to_string())

    print("\nhead-to-head on certified ranks:")
    head_to_head(df, "STELA", "ALEXA")
    head_to_head(df, "STELA", "CARLA")
    head_to_head(df, "CARLA", "ALEXA")

    make_boxplot(df, outdir / f"fig_box_three.{args.format}")
    return 0


if __name__ == "__main__":
    sys.exit(main())