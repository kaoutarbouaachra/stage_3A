"""
Batch launcher for the T-ASR algorithm suite.


The script auto-detects `Algorithm_codes/` (the directory holding the
algorithm packages) by walking up from its own location, and runs every
sub-process with that directory as cwd, so `python -m <pkg>.<module>` and
the relative `--data-dir` behave exactly as in an interactive session.

Usage (from anywhere)
---------------------
    python scripts/launch_batch.py --algo diag --start 1 --end 25
    python scripts/launch_batch.py --algo diag --start 26 --end 50

Each instance runs in its own process, so a crash, a Gurobi licence hiccup
or a timeout on one instance never kills the batch. The consolidated CSV is
rewritten after every instance, so a Ctrl-C loses nothing.
"""

from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

# --------------------------------------------------------------------------- #
# Algorithm registry:  key -> (results-folder tag, module for `python -m`)
# --------------------------------------------------------------------------- #
ALGOS: dict[str, tuple[str, str]] = {
    "diag":  ("TCG_Diag",  "TCG_Diag.run_TCG_Diag"),
    "topk":  ("TCG_TOPK",  "TCG_TopK.run_TCG_TopK"),
    "exact": ("TCG_EXACT", "TCG_Exact.run_TCG_Exact"),
    "exact_warm_start": ("TCG_EXACT", "TCG_Exact.run_TCG_exact_warm"),
    "stela": ("STELA",     "stela.main"),
    "alexa": ("ALEXA",     "alexa.main"),
    "carla": ("CARLA",     "carla.main")
}
# --------------------------------------------------------------------------- #


def find_algo_root(start: Path) -> Path:
    """Walk up from `start` until a directory named Algorithm_codes is found,
    or one that contains at least one known algorithm package."""
    known_pkgs = {"TCG_Diag", "TCG_TopK", "TCG_Exact", "STELA", "ALEXA"}
    for cand in [start, *start.parents]:
        if cand.name == "Algorithm_codes":
            return cand
        if any((cand / p).is_dir() for p in known_pkgs):
            return cand
    # fallback: parent of the scripts/ folder
    return start.parent


def instance_name(idx: int, prefix: str, pad: int) -> str:
    """(setAP, 3, 3) -> 'setAP-003'."""
    return f"{prefix}-{idx:0{pad}d}"


def flatten(prefix: str, obj, out: dict) -> None:
    """Flatten a nested dict/list into CSV-friendly scalar columns."""
    if isinstance(obj, dict):
        for k, v in obj.items():
            flatten(f"{prefix}.{k}" if prefix else str(k), v, out)
    elif isinstance(obj, (list, tuple)):
        out[prefix] = json.dumps(obj)
    else:
        out[prefix] = obj


class Batch:
    def __init__(self, args, extra_argv: list[str]):
        self.args = args
        self.extra_argv = extra_argv
        self.root = Path(args.algo_root).resolve() if args.algo_root \
            else find_algo_root(Path(__file__).resolve().parent)
        self.results_dir = self.root / "results"

    # ---------------- paths ---------------- #

    def summary_path(self, inst: str) -> Path:
        return self.results_dir / inst / self.args.algo_tag / f"{inst}-summary.json"

    def read_summary(self, inst: str) -> dict:
        p = self.summary_path(inst)
        if not p.is_file():
            return {}
        try:
            with p.open(encoding="utf-8") as fh:
                data = json.load(fh)
        except Exception as exc:  # corrupted / partial write
            return {"summary_read_error": repr(exc)}
        out: dict = {}
        flatten("", data, out)
        return out

    # ---------------- running ---------------- #

    def run_one(self, inst: str) -> dict:
        cmd = [sys.executable, "-m", self.args.module, inst,
               f"--data-dir={self.args.data_dir}"] + self.extra_argv

        row = {
            "instance": inst,
            "started_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "cmd": " ".join(cmd),
        }

        if self.args.dry_run:
            print(f"  [dry-run] (cwd={self.root}) {' '.join(cmd)}")
            row.update(status="dry-run", returncode="", wall_time_s=0.0)
            return row

        t0 = time.time()
        try:
            proc = subprocess.run(
                cmd,
                cwd=str(self.root),
                timeout=self.args.timeout if self.args.timeout > 0 else None,
            )
            row["returncode"] = proc.returncode
            row["status"] = "ok" if proc.returncode == 0 else "error"
        except subprocess.TimeoutExpired:
            row["returncode"] = ""
            row["status"] = "timeout"
        except KeyboardInterrupt:
            row["returncode"] = ""
            row["status"] = "interrupted"
            row["wall_time_s"] = round(time.time() - t0, 2)
            row.update(self.read_summary(inst))
            raise BatchInterrupt(row)

        row["wall_time_s"] = round(time.time() - t0, 2)
        row.update(self.read_summary(inst))
        return row


class BatchInterrupt(Exception):
    """Ctrl-C during an instance; carries the partial row."""

    def __init__(self, row: dict):
        super().__init__("interrupted")
        self.row = row


def write_csv(rows: list[dict], path: Path) -> None:
    """Rewrite the whole CSV; columns are the union of all row keys."""
    if not rows:
        return
    fixed = ["instance", "status", "returncode", "wall_time_s", "started_at"]
    extra = sorted({k for r in rows for k in r} - set(fixed))
    fields = fixed + extra
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        for r in rows:
            w.writerow(r)


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(
        description="Run one algorithm over a contiguous range of instances."
    )
    ap.add_argument("--algo", default="diag", choices=sorted(ALGOS),
                    help="algorithm key from the registry (default: diag)")
    ap.add_argument("--algo-tag", default=None,
                    help="override the results/<inst>/<TAG>/ folder name")
    ap.add_argument("--module", default=None,
                    help="override the module passed to `python -m`")
    ap.add_argument("--algo-root", default=None,
                    help="path to Algorithm_codes/ (auto-detected by default)")

    ap.add_argument("--start", type=int, required=True, help="first instance index (e.g. 1)")
    ap.add_argument("--end", type=int, required=True, help="last index, inclusive (e.g. 25)")
    ap.add_argument("--prefix", default="setAP", help="instance name prefix (default: setAP)")
    ap.add_argument("--pad", type=int, default=3,
                    help="zero-padding width (default: 3 -> setAP-001)")
    ap.add_argument("--data-dir", default="../setAP",
                    help="dataset dir, relative to Algorithm_codes/ (default: ../setAP)")

    ap.add_argument("--timeout", type=float, default=0,
                    help="hard per-instance wall-clock cap in seconds; 0 = none. "
                         "Set it slightly above the algorithm's internal time_limit.")
    ap.add_argument("--skip-done", action="store_true",
                    help="skip instances that already have a summary JSON (resume a batch)")
    ap.add_argument("--dry-run", action="store_true",
                    help="print the commands without running them")
    ap.add_argument("--tag", default="", help="optional suffix for the batch CSV name")
    return ap


def main() -> int:
    ap = build_parser()
    args, extra_argv = ap.parse_known_args()

    default_tag, default_module = ALGOS[args.algo]
    args.algo_tag = args.algo_tag or default_tag
    args.module = args.module or default_module

    if args.end < args.start:
        ap.error("--end must be >= --start")

    batch = Batch(args, extra_argv)

    if not (batch.root / args.module.split(".")[0]).is_dir():
        print(f"  !! package '{args.module.split('.')[0]}' not found in {batch.root}")
        print("     use --algo-root to point at Algorithm_codes/ explicitly.")
        return 2

    instances = [instance_name(i, args.prefix, args.pad)
                 for i in range(args.start, args.end + 1)]

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    suffix = f"_{args.tag}" if args.tag else ""
    csv_path = (batch.results_dir / "_batches" /
                f"{args.algo_tag}_{args.prefix}_"
                f"{args.start:0{args.pad}d}-{args.end:0{args.pad}d}_{stamp}{suffix}.csv")

    print("=" * 76)
    print(f"  BATCH {args.algo_tag}   {instances[0]} .. {instances[-1]}   "
          f"({len(instances)} instances)")
    print(f"  algo root  : {batch.root}")
    print(f"  module     : {args.module}")
    print(f"  data-dir   : {args.data_dir}")
    print(f"  timeout    : {'none' if args.timeout <= 0 else f'{args.timeout:.0f}s per instance'}")
    print(f"  extra args : {' '.join(extra_argv) if extra_argv else '(none)'}")
    print(f"  batch CSV  : {csv_path}")
    print("=" * 76)

    rows: list[dict] = []
    t_batch = time.time()

    for k, inst in enumerate(instances, start=1):
        head = f"[{k}/{len(instances)}] {inst}"

        if args.skip_done and batch.summary_path(inst).is_file():
            print(f"\n{head}  --> already done, skipped")
            row = {"instance": inst, "status": "skipped", "returncode": "",
                   "wall_time_s": 0.0, "started_at": ""}
            row.update(batch.read_summary(inst))
            rows.append(row)
            write_csv(rows, csv_path)
            continue

        print(f"\n{'-' * 76}\n{head}\n{'-' * 76}", flush=True)
        try:
            row = batch.run_one(inst)
        except BatchInterrupt as exc:
            rows.append(exc.row)
            write_csv(rows, csv_path)
            print("\n\n  Interrupted by user. Partial results written to:")
            print(f"  {csv_path}")
            return 130

        rows.append(row)
        write_csv(rows, csv_path)          # flush after every instance
        print(f"  --> {row['status']}  ({row['wall_time_s']}s)", flush=True)

    write_csv(rows, csv_path)

    n_ok = sum(1 for r in rows if r["status"] == "ok")
    n_bad = sum(1 for r in rows if r["status"] in ("error", "timeout"))
    n_skip = sum(1 for r in rows if r["status"] == "skipped")

    print("\n" + "=" * 76)
    print(f"  BATCH DONE in {time.time() - t_batch:.1f}s")
    print(f"  ok: {n_ok}   failed/timeout: {n_bad}   skipped: {n_skip}")
    if n_bad:
        print("  problematic instances:")
        for r in rows:
            if r["status"] in ("error", "timeout"):
                print(f"    - {r['instance']}: {r['status']} (rc={r['returncode']})")
    print(f"  CSV: {csv_path}")
    print("=" * 76)

    return 0 if n_bad == 0 else 1


if __name__ == "__main__":
    sys.exit(main())