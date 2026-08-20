# Algorithm Codes

This folder contains the codes.

## Important launch rule

All solver commands should be executed from this directory:

```bash
cd codes/Algorithm_codes
```

Use module execution:

```bash
python -m package.module [arguments]
```

This avoids import errors and lets modules such as `core`, `exact`, `cg_lex_heuristic_pricing`, and `cg_bellman_ford` resolve correctly.

## Packages and folders

```text
Algorithm_codes/
  core/                       Shared utilities: instance loading, ECMP backend selector, checker interface, paths
  stela/                      Exact sequential lexicographic minimization solver (Threshold-Exceedance based)
  alexa/                      Exact sequential lexicographic minimization solver (Assignment based)
  TCG_TopK/                   Column-generation solver with heuristic diagonal/off-diagonal pricing
  TCG_Diag/                   Alternative column-generation solver with Bellman-Ford diagonal pricing
  TCG_Exact/                  Alternative column-generation solver with exact pricing
  TCG_carla/                  Carla-driven column-generation solver (with exact an diag pricings)  
  logs/                       Runtime logs produced by algos
  results/                    Solver outputs and checker results
```

## ECMP backend (nt4python or NetworkX)

ECMP split coefficients are computed by one of two interchangeable backends inside `core/`:

- `core/ecmp_nt.py` — **nt4python** (Orange NetworkTools), Linux / WSL only.
- `core/ecmp_from_scratch.py` — **NetworkX**, cross-platform (works on Windows).

`core/ecmp.py` is a **selector** that loads the right backend and re-exports `compute_all_splits`, `compute_loads`, `sorted_load_vector`, `SplitCoeffs`. Always import from `core.ecmp` — never from a specific backend.

Select the backend with the `T_ASR_ECMP_BACKEND` environment variable:

| Value | Behaviour |
|---|---|
| `scratch` | Force NetworkX (cross-platform) |
| `nt` | Force nt4python (Linux / WSL only) |
| unset / `auto` | Try nt4python, fall back to NetworkX |

```bash
export T_ASR_ECMP_BACKEND=scratch   # NetworkX 
export T_ASR_ECMP_BACKEND=nt        # nt4python (networktools)

# which backend is active?
python -c "import core.ecmp as e; print(e.ECMP_BACKEND)"
```