#!/usr/bin/env python3
r"""
STELA vs CARLA comparison on the expe30min campaign (works on partial runs).

Parses ``<algo>-setAP-<NNN>-log.txt`` for algo in {stela, carla}, keeps the
instances present for BOTH algorithms, and produces:
  - summary.csv                    one row per (algo, instance)
  - fig_box_two.<fmt>              certified-ranks boxplot by size class N
  - fig_itertime_vs_rank.<fmt>     median iter_time vs rank k (the crossover plot)
  - stdout                         head-to-head + per-class medians

Paths are auto-resolved: it locates Algorithm_codes/logs/expe30min by walking up
from the script, and writes outputs next to the script. Run with no arguments:

    python compare_two.py

Overrides: --logs <dir>  --outdir <dir>  --format pdf
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

ALGOS = ["STELA", "CARLA"]
FILLS = {"STELA": "#d9d9d9", "CARLA": "#a6a6a6"}
LINES = {"STELA": "-", "CARLA": "--"}
LOGS_REL = Path("Algorithm_codes") / "logs" / "expe30min"

RE_INSTANCE = re.compile(r"(setAP-\d+)")
RE_N        = re.compile(r"\bN=(\d+)")
RE_DONE     = re.compile(r"Done\.\s+(\d+)\s+(?:iterations|ranks)")
RE_DONE_T   = re.compile(r"Done\.\s+\d+\s+(?:iterations|ranks)(?:\s+certified)?\s+in\s+([\d.]+)s")
RE_MLU      = re.compile(r"MLU\s*\([^)]*\)\s*=\s*([\d.]+)")
RE_STATUS11 = re.compile(r"Unexpected status 11")
RE_OPTROW   = re.compile(r"^\s*\d+\s+.*\bOPTIMAL\b")
# per-rank data row -> (k, iter_time in seconds); iter_time is the 3rd/4th col
# ending with 's'. We capture k and the first "<number>s" on the row.
RE_KROW     = re.compile(r"^\s*(\d+)\s+.*?([\d.]+)s\s")


def resolve_logs_dir(override):
    if override is not None:
        return override
    for base in [Path(__file__).resolve().parent, Path.cwd().resolve()]:
        for parent in [base, *base.parents]:
            cand = parent / LOGS_REL
            if cand.is_dir():
                return cand
    return LOGS_REL


def natural_key(name: str):
    m = re.search(r"(\d+)", name)
    return (name[: m.start()], int(m.group(1))) if m else (name, 0)


def parse_log(path: Path, algo: str):
    text = path.read_text(encoding="utf-8", errors="replace")
    m_inst = RE_INSTANCE.search(text)
    if m_inst is None:
        return None, []
    instance = m_inst.group(1)
    m_n = RE_N.search(text)
    N = int(m_n.group(1)) if m_n else None

    m_done = RE_DONE.search(text)
    n_optrows = sum(1 for ln in text.splitlines() if RE_OPTROW.match(ln))
    if m_done is not None:
        ranks = int(m_done.group(1))
        if ranks != n_optrows:
            print(f"  [warn] {path.name}: Done={ranks} but {n_optrows} OPTIMAL rows",
                  file=sys.stderr)
    else:
        ranks = n_optrows

    m_t = RE_DONE_T.search(text)
    solve_time = float(m_t.group(1)) if m_t else None
    m_mlu = RE_MLU.search(text)
    mlu = float(m_mlu.group(1)) if m_mlu else None
    timed_out = bool(RE_STATUS11.search(text))

    # per-rank iter_time, only from genuine OPTIMAL rows
    per_rank = []
    for ln in text.splitlines():
        if "OPTIMAL" not in ln:
            continue
        m = RE_KROW.match(ln)
        if m:
            per_rank.append({"algo": algo, "instance": instance, "N": N,
                             "k": int(m.group(1)), "iter_time": float(m.group(2))})

    row = {"algo": algo, "instance": instance, "N": N, "ranks": ranks,
           "solve_time": solve_time, "mlu": mlu, "timed_out": timed_out}
    return row, per_rank


def load(logs_dir: Path):
    if not logs_dir.is_dir():
        raise SystemExit(f"logs directory not found: {logs_dir}")
    rows, per_rank = [], []
    for path in sorted(logs_dir.glob("*-setAP-*-log.txt")):
        prefix = path.name.split("-", 1)[0].lower()
        algo = {"stela": "STELA", "carla": "CARLA"}.get(prefix)
        if algo is None:
            continue
        row, pr = parse_log(path, algo)
        if row:
            rows.append(row)
            per_rank.extend(pr)
    if not rows:
        raise SystemExit(f"no STELA/CARLA logs found in {logs_dir}")
    return pd.DataFrame(rows), pd.DataFrame(per_rank)


def restrict_to_pairs(df: pd.DataFrame) -> pd.DataFrame:
    present = df.groupby("instance").algo.nunique()
    both = set(present[present == 2].index)
    only = sorted(set(df.instance) - both, key=natural_key)
    if only:
        print(f"[info] {len(only)} instance(s) not present for both algos, "
              f"excluded: {only[:8]}")
    return df[df.instance.isin(both)].copy()


def head_to_head(df: pd.DataFrame) -> None:
    wide = df.pivot(index="instance", columns="algo", values="ranks")
    d = wide["CARLA"] - wide["STELA"]
    print(f"  CARLA vs STELA (certified ranks):  "
          f"CARLA wins {int((d > 0).sum())}, ties {int((d == 0).sum())}, "
          f"losses {int((d < 0).sum())}  (n={len(wide)})")


def make_boxplot(df: pd.DataFrame, out: Path) -> None:
    classes = sorted(df.N.dropna().unique())
    n_cls, width = len(classes), 0.8 / len(ALGOS)
    fig, ax = plt.subplots(figsize=(3.4, 2.5))
    for j, algo in enumerate(ALGOS):
        data = [df[(df.N == c) & (df.algo == algo)].ranks.values for c in classes]
        pos = [i + (j - 0.5) * width for i in range(n_cls)]
        bp = ax.boxplot(data, positions=pos, widths=width * 0.9, patch_artist=True,
                        showfliers=False,
                        medianprops=dict(color="black", linewidth=1.1),
                        whiskerprops=dict(color="black", linewidth=0.7),
                        capprops=dict(color="black", linewidth=0.7),
                        boxprops=dict(color="black", linewidth=0.7))
        for box in bp["boxes"]:
            box.set_facecolor(FILLS[algo])
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


def make_itertime_plot(pr: pd.DataFrame, pairs: set, out: Path) -> None:
    """Median iter_time vs rank k, STELA vs CARLA (the crossover diagnostic)."""
    pr = pr[pr.instance.isin(pairs)]
    if pr.empty:
        print("[info] no per-rank iter_time parsed, skipping crossover plot")
        return
    fig, ax = plt.subplots(figsize=(3.4, 2.5))
    for algo in ALGOS:
        g = (pr[pr.algo == algo].groupby("k").iter_time
             .median().sort_index())
        # only plot ranks where enough instances contribute (>= 3)
        counts = pr[pr.algo == algo].groupby("k").size()
        g = g[counts.reindex(g.index).fillna(0) >= 3]
        ax.plot(g.index, g.values, LINES[algo], color="black" if algo == "STELA"
                else "#666666", linewidth=1.2, label=algo)
    ax.set_xlabel(r"rank $k$")
    ax.set_ylabel("median iter. time (s)")
    ax.set_yscale("log")
    ax.legend(frameon=False, fontsize=7)
    ax.grid(True, which="both", linewidth=0.3, alpha=0.4)
    fig.tight_layout()
    fig.savefig(out, bbox_inches="tight")
    print(f"figure written -> {out}")


def main(argv=None) -> int:
    script_dir = Path(__file__).resolve().parent
    ap = argparse.ArgumentParser()
    ap.add_argument("--logs", type=Path, default=None)
    ap.add_argument("--outdir", type=Path, default=None)
    ap.add_argument("--format", default="pdf", choices=["pdf", "png"])
    args = ap.parse_args(argv)

    logs_dir = resolve_logs_dir(args.logs)
    outdir = args.outdir if args.outdir is not None else script_dir
    outdir.mkdir(parents=True, exist_ok=True)
    print(f"logs   : {logs_dir}")
    print(f"outdir : {outdir}")

    plt.rcParams.update({"font.family": "serif", "mathtext.fontset": "cm",
                         "font.size": 8, "axes.linewidth": 0.6, "figure.dpi": 200})

    df, pr = load(logs_dir)
    df.sort_values(["algo", "instance"],
                   key=lambda s: s.map(natural_key) if s.name == "instance" else s,
                   inplace=True)
    df.to_csv(outdir / "summary.csv", index=False)

    print("\nruns per algorithm:")
    print(df.groupby("algo").size().to_string())

    df = restrict_to_pairs(df)
    if df.empty:
        raise SystemExit("no instance present for both STELA and CARLA")
    pairs = set(df.instance)

    print("\nmedian certified ranks by size class:")
    print(df.pivot_table(index="N", columns="algo", values="ranks",
                         aggfunc="median").reindex(columns=ALGOS).to_string())

    print("\nhead-to-head:")
    head_to_head(df)

    make_boxplot(df, outdir / f"fig_box_two.{args.format}")
    make_itertime_plot(pr, pairs, outdir / f"fig_itertime_vs_rank.{args.format}")
    return 0


if __name__ == "__main__":
    sys.exit(main())