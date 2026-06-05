"""
Function: checker.py
Role: Interface with the Orange C++ checker binary.
Requirements: Compile the checker first: cd ../checker/src && make
"""
 
import json
import platform
import subprocess
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional
 
##################################### 
###  Locate checker binary
#####################################
 
def find_checker(checker_dir: Path) -> Optional[Path]:
    for p in [
        checker_dir / "src" / "checker",
        checker_dir / "src" / "checker.exe",
        checker_dir / "checker",
        checker_dir / "checker.exe",
    ]:
        if p.exists():
            return p
    return None
 
 
def build_instructions(checker_dir: Path) -> str:
    src = checker_dir / "src"
    if platform.system() == "Windows":
        return (
            f"Checker not compiled. Options:\n"
            f"  WSL  : cd '{src}' && make\n"
            f"  Docker: use the Orange Docker image\n"
        )
    return f"Checker not compiled:\n  cd '{src}' && make\n"
 


#####################################
###  Result container
#####################################
 
@dataclass
class CheckerResult:
    # Core
    valid        : bool
    raw_output   : str
    
    objectives   : List[dict] = field(default_factory=list)
    # [{"t": 0, "mlu": 0.929, "inv_load_cost": ..., "jain_index": ...}, ...]
 
    saturations  : List[dict] = field(default_factory=list)
    # [{"t": 0, "from": "A", "to": "B", "sat": 0.929}, ...]
 
    # Summary stats 
    max_load     : float = float('inf')   # MLU
    loads_sorted : List[float] = field(default_factory=list)  # sat values
 
    # Metadata
    total_cost    : Optional[int] = None
    total_srpaths : Optional[int] = None
    total_segments: Optional[int] = None
 
    @property
    def feasible(self) -> bool:
        return self.valid
 
    def summary(self) -> str:
        status = "FEASIBLE" if self.valid else "INFEASIBLE"
        top5   = [round(v, 6) for v in self.loads_sorted[:5]]
        return f"[CHECKER] {status} | MLU={self.max_load:.6f} | top5={top5}"
 
 
#####################################
###  Run checker
#####################################
 
def run_checker(
    net_file      : Path,
    tm_file       : Path,
    scenario_file : Path,
    srpaths_file  : Path,
    checker_bin   : Optional[Path],
    max_decimal   : int = 6,
    verbose       : bool = True,
) -> Optional[CheckerResult]:
 
    if checker_bin is None or not checker_bin.exists():
        print("  [CHECKER] Binary not found.")
        return None
 
    def to_wsl(p: Path) -> str:
        s = str(p).replace("\\", "/")
        if s[1] == ":":
            s = "/mnt/" + s[0].lower() + s[2:]
        return s
 
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
 
    if verbose:
        cmd.append("--verbose")
 
    try:
        proc   = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
        stdout = proc.stdout.strip()
        stderr = proc.stderr.strip()
        output = stdout or stderr
 
        # Parse JSON output
        result = _parse_output(output, proc.returncode)
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
 
def _parse_output(output: str, returncode: int) -> CheckerResult:
    # Extract JSON from output, ignoring any non-JSON text before it
    json_start = output.find('{')
    if json_start == -1:
        return CheckerResult(valid=False, raw_output=output)
    json_str = output[json_start:]
    try:
        data = json.loads(json_str)
    except json.JSONDecodeError:
        return CheckerResult(valid=False, raw_output=output)
    valid       = data.get("valid", False)
    objectives  = data.get("objectives",  [])
    saturations = data.get("saturations", [])
    mlu_per_t    = [obj["mlu"] for obj in objectives if "mlu" in obj]
    max_load     = max(mlu_per_t) if mlu_per_t else (float('inf') if not valid else 0.0)
    loads_sorted = sorted([s["sat"] for s in saturations if "sat" in s], reverse=True)
    return CheckerResult(
        valid         = valid,
        raw_output    = output,
        objectives    = objectives,
        saturations   = saturations,
        max_load      = max_load,
        loads_sorted  = loads_sorted,
        total_cost    = data.get("total_cost"),
        total_srpaths = data.get("total_srpaths"),
        total_segments= data.get("total_segments"),
    )