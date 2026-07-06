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
  exact/                      Exact sequential lexicographic minimization solver
  cg_lex_heuristic_pricing/   Column-generation solver with heuristic diagonal/off-diagonal pricing
  cg_bellman_ford/            Alternative column-generation solver with Bellman-Ford diagonal pricing
  logs/                       Runtime logs produced by `exact.run_all`
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
export T_ASR_ECMP_BACKEND=scratch   # NetworkX (recommended on Windows)
export T_ASR_ECMP_BACKEND=nt        # nt4python (Linux / WSL)

# which backend is active?
python -c "import core.ecmp as e; print(e.ECMP_BACKEND)"
```

On Windows, use `scratch` (nt4python has no Windows wheel; the selector also falls back automatically when the variable is unset).

## Main commands

### Exact solver, simple entry point

```bash
python -m exact.main --instance toy --time-limit 60
python -m exact.main --instance setA-01 --time-limit 300
```

### Exact solver with logs and figures

```bash
python -m exact.run_all --instance toy --time_limit 120
python -m exact.run_all --instance setA-01 --time_limit 300 --max_iter 10
```

### Column-generation solver (heuristic pricing)

```bash
python -m cg_lex_heuristic_pricing.run_cg_lex_heuristic_pricing --instance toy --K 3 --time_limit 120
python -m cg_lex_heuristic_pricing.run_cg_lex_heuristic_pricing --instance setA-01 --K 5 --time_limit 900
```

### Bellman-Ford column generation

```bash
python -m cg_bellman_ford.run_cg_lex --instance toy --K 3 --time_limit 120
```

### Checker only

```bash
python -m core.run_checker_local --instance toy
python -m core.run_checker_local --instance toy --srpaths results/toy/toy-srpaths-cglex.json
```

## Common command-line arguments

### `exact.main`

```text
--instance / -i       Standard instance name, e.g. toy, simple, ar3, setA-01
--net                 Manual path to a net JSON file
--tm                  Manual path to a traffic matrix JSON file
--scenario            Manual path to a scenario JSON file
--data-dir / -d       Override the auto-detected input folder
--out-dir / -o        Override the output folder
--checker-dir         Override the checker folder
--max-iter            Maximum number of lexicographic iterations
--time-limit          Gurobi time limit in seconds
--mip-gap             Gurobi MIP gap
--verbose-gurobi      Show Gurobi logs
```

### `exact.run_all`

```text
--instance            Instance name
--data_dir            Override the input folder
--checker_dir         Override the checker folder
--time_limit          Time limit in seconds
--max_iter            Maximum number of lexicographic iterations
--mip_gap             Gurobi MIP gap
```

### `cg_lex_heuristic_pricing.run_cg_lex_heuristic_pricing` and `cg_bellman_ford.run_cg_lex`

```text
--instance            Instance name
--data_dir            Override the input folder
--checker_dir         Override the checker folder
--K                   Number of lexicographic ranks to certify
--time_limit          Global time limit in seconds
--lp_time             Time limit per LP restricted master problem
--mip_time            Time limit per rank MIP
--max_cg_iter         Maximum CG iterations per rank
--topk_cols           Number of columns added per demand per iteration
--rc_tol              Reduced-cost tolerance
```

## Outputs

Outputs are written under:

```text
results/<instance>/
logs/<instance>/<timestamp>/
```

The exact solver usually writes:

```text
<instance>-srpaths.json
<instance>-summary.json
<instance>-checker-loads.json
```

The column-generation solvers usually write:

```text
<instance>-srpaths-cglex.json
<instance>-cglex-summary.json
```
