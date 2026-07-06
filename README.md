# T-ASR
 
This repository contains Python implementations for the **T-ASR routing problem**, developed as part of my third-year engineering internship at **École Polytechnique**. The internship was carried out at **Orange Garden Research**, within the **DATA&AI division** / **MORE team**.
 
The code is organized around an exact sequential lexicographic solver and two column-generation variants.
 
The main working directory is:
 
```bash
codes/
```
 
For detailed installation and execution instructions, see:
 
```bash
codes/README.md
codes/Algorithm_codes/README.md
```
 
## Quick start
 
From the repository root:
 
```bash
cd codes/Algorithm_codes
python -m exact.main --instance toy --time-limit 60
python -m cg_lex_heuristic_pricing.run_cg_lex_heuristic_pricing --instance toy --K 3 --time_limit 120
```
 
Most scripts must be launched with `python -m ...` from `codes/Algorithm_codes` so that Python imports resolve correctly.
 
## ECMP backend (nt4python or NetworkX)
 
ECMP split coefficients can be computed by two interchangeable backends, chosen at runtime through the `T_ASR_ECMP_BACKEND` environment variable:
 
- `scratch` — NetworkX, cross-platform (works on Windows).
- `nt` — nt4python (Orange NetworkTools), Linux / WSL only.
- unset / `auto` — tries nt4python, falls back to NetworkX if it is unavailable.
```bash
export T_ASR_ECMP_BACKEND=scratch   # recommended on Windows
```
 
All solver code imports from `core.ecmp` only; that module is a selector that loads the right backend. See `codes/README.md` for the full explanation.
