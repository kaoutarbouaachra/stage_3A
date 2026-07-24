"""
Function: dataset.py
Role: Shared dataset CLI arguments and path resolution for every T-ASR
      algorithm package (ALEXA, STELA, CARLA, TCG, ...).

"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import List, Optional, Tuple

from core.paths import CODES_DIR

FILE_SUFFIXES = ("-net.json", "-tm.json", "-scenario.json")


#####################################################
###  CLI arguments
#####################################################

def add_dataset_args(parser) -> None:
    """
    Add the instance-selection arguments shared by every algorithm.

    Three accepted invocation styles:
        <prog> setAP-001 --data-dir ../setAP/      (positional)
        <prog> --instance setAP-001                (named)
        <prog> --net a.json --tm b.json --scenario c.json   (manual)
    """
    parser.add_argument(
        "instance_name", nargs="?", default=None,
        help="Instance name, positional form (e.g. setAP-001).",
    )
    parser.add_argument(
        "--instance", "-i", default=None,
        help="Instance name, named form. Equivalent to the positional arg.",
    )

    parser.add_argument("--net", default=None,
                        help="Path to <name>-net.json (manual mode).")
    parser.add_argument("--tm", default=None,
                        help="Path to <name>-tm.json (manual mode).")
    parser.add_argument("--scenario", default=None,
                        help="Path to <name>-scenario.json (manual mode).")

    # Both spellings are accepted: the packages historically disagreed and
    # existing launch scripts use either one.
    parser.add_argument(
        "--data-dir", "--data_dir", "-d",
        dest="data_dir", default=None,
        help="Base folder holding the instance files. Accepts both the flat "
             "layout (../setAP/setAP-001-net.json) and the per-instance "
             "subfolder layout (../setAP/setAP-001/setAP-001-net.json).",
    )


#####################################################
###  Path resolution
#####################################################

def _candidate_bases(name: str, data_dir: Optional[str]) -> List[Path]:
    """
    Return the candidate base folders, in priority order.

    Two candidates are always returned for a set-style name so that both the
    flat and the per-instance-subfolder layouts work without a flag.
    """
    if data_dir:
        base = Path(data_dir)
        return [base, base / name]

    if name.startswith("set"):
        # Split on the first '-' so that setA-01 -> setA and setAP-001 -> setAP.
        # A plain startswith("setA") test would send setAP-* into setA/.
        prefix = name.split("-", 1)[0]
        return [CODES_DIR / prefix / name, CODES_DIR / prefix]

    if name.startswith("ar"):
        return [CODES_DIR / "ar3_instance"]

    if name == "toy":
        return [CODES_DIR / "toy_instance"]

    return [CODES_DIR / f"{name}_instance"]


def _files_in(base: Path, name: str) -> Tuple[Path, Path, Path]:
    return tuple(base / f"{name}{suffix}" for suffix in FILE_SUFFIXES)


def resolve_paths(args) -> Tuple[Path, Path, Path]:
    """
    Resolve (net, tm, scenario) paths from the parsed arguments.

    Exits with an explicit diagnostic listing every folder tried if the
    instance cannot be located.
    """
    if getattr(args, "net", None):
        if not (args.tm and args.scenario):
            print("[ERROR] --net requires both --tm and --scenario.")
            sys.exit(1)
        return Path(args.net), Path(args.tm), Path(args.scenario)

    name = getattr(args, "instance", None) or getattr(args, "instance_name", None)
    if not name:
        print("[ERROR] No instance given. Use a positional NAME, --instance NAME, "
              "or the manual --net/--tm/--scenario triple.")
        sys.exit(1)

    tried: List[Path] = []
    for base in _candidate_bases(name, getattr(args, "data_dir", None)):
        files = _files_in(base, name)
        if all(p.exists() for p in files):
            return files
        tried.append(base)

    print(f"[ERROR] Instance '{name}' not found. Looked for "
          f"{name}-net.json, {name}-tm.json, {name}-scenario.json in:")
    for base in tried:
        print(f"          {base}")
    sys.exit(1)


def instance_name(args) -> str:
    """Name used for output namespacing: results/<instance>/<ALGO>/."""
    name = getattr(args, "instance", None) or getattr(args, "instance_name", None)
    if name:
        return name
    if getattr(args, "net", None):
        return Path(args.net).stem.removesuffix("-net")
    return "output"