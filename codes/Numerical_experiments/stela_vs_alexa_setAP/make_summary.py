#!/usr/bin/env python3
"""
Step 1 -- build the CSV summaries from the run logs.

Reads every "*-log.txt" under the two campaign directories and writes:

    summary.csv   one row per (instance, algo)
                  instance, N, algo, k, time, status, mlu
    vectors.csv   long format, one row per (instance, algo, rank)
                  instance, algo, rank, value, iter_time

Status is read from the log text, never inferred from elapsed time:
    CONVERGED    "All loads are 0"        whole vector determined
    INTERRUPTED  "Unexpected status 11"   global budget expired mid-rank
    STOPPED_<c>  any other unexpected status
    EXHAUSTED    loop ended with no marker

Usage (from Algorithm_codes/):
    python make_summary.py results --outdir ..\\Numerical_experiments\\stela_vs_alexa_setAP
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

import pandas as pd

RE_HEADER = re.compile(
    r"(?P<instance>\S+)\s*\|\s*(?P<nodes>\d+)\s+nodes,\s*(?P<arcs>\d+)\s+arcs,"
    r"\s*(?P<demands>\d+)\s+demands,\s*(?P<periods>\d+)\s+timesteps,\s*N=(?P<N>\d+)"
)
# STELA rows carry an extra signed delta column; ALEXA rows do not.
RE_ROW = re.compile(
    r"^\s*(?P<k>\d+)\s+(?P<z>\d+\.\d+)\s+(?P<iter>\d+\.\d+)s"
    r"\s+(?P<elapsed>\d+\.\d+)s\s+(?:[+-]\d+\.\d+\s+)?(?P<status>[A-Za-z][\w=]*)"
)
RE_DONE = re.compile(r"Done\.\s+(?P<count>\d+)\s+(?:iterations|ranks)\s+in\s+(?P<time>[\d.]+)s")
RE_MLU = re.compile(r"MLU baseline\s*:\s*(?P<mlu>[\d.]+)")
RE_UNEXPECTED = re.compile(r"Unexpected status\s+(?P<code>\d+)")
RE_CONVERGED = re.compile(r"All loads are 0")


def parse_log(path: Path, algo: str) -> tuple[dict, list[dict]] | None:
    text = path.read_text(encoding="utf-8", errors="replace")
    header = RE_HEADER.search(text)
    if header is None:
        return None

    # Key on the printed rank, not on append order: a log with a missing rank
    # would otherwise shift every later value and corrupt the vector.
    by_rank: dict[int, tuple[float, float]] = {}
    for line in text.splitlines():
        row = RE_ROW.match(line)
        if row is not None:
            by_rank[int(row["k"])] = (float(row["z"]), float(row["iter"]))

    if RE_CONVERGED.search(text):
        status = "CONVERGED"
    elif (unexpected := RE_UNEXPECTED.search(text)) is not None:
        code = unexpected["code"]
        status = "INTERRUPTED" if code == "11" else f"STOPPED_{code}"
    else:
        status = "EXHAUSTED"

    done, mlu = RE_DONE.search(text), RE_MLU.search(text)
    instance = header["instance"]

    summary = {
        "instance": instance,
        "N": int(header["N"]),
        "algo": algo,
        "k": int(done["count"]) if done else len(by_rank),
        "time": float(done["time"]) if done else float("nan"),
        "status": status,
        "mlu": float(mlu["mlu"]) if mlu else float("nan"),
        "arcs": int(header["arcs"]),
        "periods": int(header["periods"]),
    }
    vectors = [
        {"instance": instance, "algo": algo, "rank": k,
         "value": value, "iter_time": iter_time}
        for k, (value, iter_time) in sorted(by_rank.items())
    ]
    return summary, vectors


def natural_key(name: str):
    # tuple, not list: pandas sort keys must be hashable
    return tuple(int(p) if p.isdigit() else p for p in re.split(r"(\d+)", name))


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("root", type=Path, nargs="?", default=Path("results"))
    parser.add_argument("--alexa", default="expe_alexa_setAP")
    parser.add_argument("--stela", default="expe_stela_setAP")
    parser.add_argument("--outdir", type=Path, default=Path("."))
    args = parser.parse_args(argv)
    args.outdir.mkdir(parents=True, exist_ok=True)

    summaries, vectors, skipped = [], [], []
    for algo, subdir in (("ALEXA", args.alexa), ("STELA", args.stela)):
        folder = args.root / subdir
        if not folder.is_dir():
            raise SystemExit(f"directory not found: {folder}")
        for path in sorted(folder.glob("*-log.txt")):
            parsed = parse_log(path, algo)
            if parsed is None:
                skipped.append(path.name)
                continue
            summary, rows = parsed
            summaries.append(summary)
            vectors.extend(rows)

    if not summaries:
        raise SystemExit("no parsable logs found")

    summary_df = pd.DataFrame(summaries)
    summary_df = summary_df.sort_values(
        ["algo", "instance"], key=lambda s: s.map(natural_key) if s.name == "instance" else s)
    vectors_df = pd.DataFrame(vectors)

    summary_df.to_csv(args.outdir / "summary.csv", index=False)
    vectors_df.to_csv(args.outdir / "vectors.csv", index=False)

    print(f"parsed {len(summary_df)} runs, {len(vectors_df)} rank rows")
    if skipped:
        print(f"  skipped (no header): {len(skipped)} file(s) -> {skipped[:5]}")

    print("\nruns per algorithm:")
    print(summary_df.groupby("algo").size().to_string())

    print("\nstatus breakdown:")
    print(pd.crosstab(summary_df.algo, summary_df.status).to_string())

    counts = summary_df.groupby("instance").algo.nunique()
    unpaired = sorted(counts[counts < 2].index, key=natural_key)
    if unpaired:
        print(f"\ninstances present in only one campaign ({len(unpaired)}): {unpaired}")

    print(f"\nwritten to {args.outdir.resolve()}")
    return 0


if __name__ == "__main__":
    sys.exit(main())