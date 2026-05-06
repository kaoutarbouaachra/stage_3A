"""
checker.py
----------
Interface with the Orange C++ checker binary.

The checker must be compiled first:
Checker usage:
    ./checker --net net.json --tm tm.json --scenario scenario.json
              --srpaths srpaths.json --max-decimal-places 6
"""

import json
import platform
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional


# ============================================================
# Locate checker binary
# ============================================================

def find_checker(checker_dir: Path) -> Optional[Path]:
    """
    Find the compiled checker binary.
    Looks in checker/src/ for the compiled output.
    """
    candidates = [
        checker_dir / "src" / "checker",
        checker_dir / "src" / "checker.exe",
        checker_dir / "checker",
        checker_dir / "checker.exe",
    ]
    for p in candidates:
        if p.exists():
            return p
    return None


def build_instructions(checker_dir: Path) -> str:
    """Return compilation instructions for the user."""
    src = checker_dir / "src"
    if platform.system() == "Windows":
        return (
            f"The checker needs to be compiled.\n"
            f"Options:\n"
            f"  1. Use WSL (Windows Subsystem for Linux):\n"
            f"       cd '{src}' && make\n"
            f"  2. Use MinGW/MSYS2:\n"
            f"       cd '{src}' && make\n"
            f"  3. Use the Docker image provided by Orange.\n"
        )
    return (
        f"The checker needs to be compiled:\n"
        f"  cd '{src}' && make\n"
    )


# ============================================================
# Result container
# ============================================================

@dataclass
class CheckerResult:
    feasible    : bool
    loads_sorted: List[float]   # sorted load vector from checker
    max_load    : float
    raw_output  : str

    def summary(self) -> str:
        status = "FEASIBLE" if self.feasible else "INFEASIBLE"
        top5   = [round(v, 6) for v in self.loads_sorted[:5]]
        return f"[CHECKER] {status} | max={self.max_load:.6f} | top5={top5}"


# ============================================================
# Run checker
# ============================================================

def run_checker(
    net_file      : Path,
    tm_file       : Path,
    scenario_file : Path,
    srpaths_file  : Path,
    checker_bin   : Optional[Path],
    max_decimal   : int = 6,
) -> Optional[CheckerResult]:
    """
    Run the Orange checker on a solution file.

    Returns CheckerResult, or None if the checker is not available.
    """
    if checker_bin is None or not checker_bin.exists():
        print(f"  [CHECKER] Binary not found.")
        return None

    def to_wsl(p):
        return str(p).replace("\\", "/").replace("C:", "/mnt/c")

    if platform.system() == "Windows":
        cmd = [
            "wsl", to_wsl(checker_bin),
            "--net",      to_wsl(net_file),
            "--tm",       to_wsl(tm_file),
            "--scenario", to_wsl(scenario_file),
            "--srpaths",  to_wsl(srpaths_file),
            f"--max-decimal-places={max_decimal}",
        ]
    else:
        cmd = [
            str(checker_bin),
            "--net",      str(net_file),
            "--tm",       str(tm_file),
            "--scenario", str(scenario_file),
            "--srpaths",  str(srpaths_file),
            f"--max-decimal-places={max_decimal}",
        ]

    try:
        res = subprocess.run(
            cmd, capture_output=True, text=True, timeout=120
        )
        output = res.stdout + res.stderr

        # Feasibility: return code 0 and no "infeasible" keyword
        feasible = (
            res.returncode == 0
            and "infeasible" not in output.lower()
        )

        # Parse sorted load values from JSON output
        loads = []
        max_load = float('inf')
        try:
            data = json.loads(output)
            objectives = data.get("objectives", [])
            loads = sorted([obj["mlu"] for obj in objectives], reverse=True)
            max_load = loads[0] if loads else float('inf')
        except (json.JSONDecodeError, KeyError):
            # Fallback : cherche des floats ligne par ligne
            for line in output.splitlines():
                try:
                    loads.append(float(line.strip()))
                except ValueError:
                    pass
            max_load = loads[0] if loads else float('inf')

        result = CheckerResult(
            feasible=feasible,
            loads_sorted=loads,
            max_load=max_load,
            raw_output=output,
        )
        print(result.summary())
        return result

    except subprocess.TimeoutExpired:
        print("  [CHECKER] Timeout (>120s)")
        return None
    except FileNotFoundError:
        print(f"  [CHECKER] Binary not executable: {checker_bin}")
        return None
    except Exception as e:
        print(f"  [CHECKER] Error: {e}")
        return None