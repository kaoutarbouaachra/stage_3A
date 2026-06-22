"""
Function: ecmp.py
Role: Public ECMP interface. Selects the split-coefficient backend.

  - ecmp_nt           : nt4python (Orange networktools) — Linux / WSL only
  - ecmp_from_scratch : NetworkX — cross-platform (works on Windows)

Backend selection via the env var T_ASR_ECMP_BACKEND:
  "nt"       -> force nt4python
  "scratch"  -> force NetworkX
  unset/auto -> try nt4python, fall back to NetworkX

The active backend name is exposed as ECMP_BACKEND.
"""

import os

_BACKEND = os.environ.get("T_ASR_ECMP_BACKEND", "auto").lower()


def _load_backend():
    if _BACKEND == "scratch":
        from . import ecmp_from_scratch as b
        return b, "NetworkX (forced)"
    if _BACKEND == "nt":
        from . import ecmp_nt as b
        return b, "nt4python (forced)"
    # auto: prefer nt4python, fall back to NetworkX
    try:
        from . import ecmp_nt as b
        return b, "nt4python"
    except Exception as e:
        from . import ecmp_from_scratch as b
        return b, f"NetworkX (nt4python unavailable: {e})"


_backend, ECMP_BACKEND = _load_backend()

compute_all_splits = _backend.compute_all_splits
compute_loads      = _backend.compute_loads
sorted_load_vector = _backend.sorted_load_vector
SplitCoeffs        = _backend.SplitCoeffs