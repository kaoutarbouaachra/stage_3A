"""
Function: paths.py
Role: Chemins du projet, indépendants du sous-dossier qui lance un script.
"""
from pathlib import Path

CORE_DIR    = Path(__file__).resolve().parent   # .../Algorithm_codes/core
ALGO_DIR    = CORE_DIR.parent                    # .../Algorithm_codes
CODES_DIR   = ALGO_DIR.parent                    # .../codes  (instances + checker ici)
RESULTS_DIR = ALGO_DIR / "results"               # .../Algorithm_codes/results
CHECKER_DIR = CODES_DIR / "checker"
LOGS_DIR    = ALGO_DIR / "logs"         
