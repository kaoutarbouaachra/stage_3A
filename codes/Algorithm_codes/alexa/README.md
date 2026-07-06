# ALEXA

**Assignment-based LEXicographic Algorithm**

## Overview

ALEXA solves T-ASR directly on the **compact formulation** -- the binary
routing variables `x[d,t,i,j]` and the reconfiguration-linearization
variables `delta[d,t,i,j]`, same as STELA, not a column-generation
reformulation.

What defines ALEXA, and distinguishes it from STELA, is *how* it sorts the
load vector into lexicographic order: STELA counts threshold exceedances
rank by rank; ALEXA builds an explicit **permutation / assignment matrix**
`p[(a,t), k]` that matches every arc-time load to a rank position `k`, with
`L[k]` the load assigned to rank `k`. Constraints force `L[0] >= L[1] >=
... >= L[N-1]` (a full sort of all `N = |A| x |T|` loads), each `(a,t)` is
assigned to exactly one rank, each rank receives exactly one `(a,t)`, and a
big-M linking constraint ties `L[k]` to the utilization of whichever
`(a,t)` is assigned to it.

This full-sort structure is built **once**, covering all ranks from the
start (unlike STELA, which adds each rank's rows incrementally). The
sequential loop then minimizes `L[0]`, freezes it once solved, minimizes
`L[1]`, and so on -- the assignment/sorting apparatus doesn't change between
ranks, only which `L[k]` is being minimized and which earlier ones are
frozen.


## Usage

Run as a module from `Algorithm_codes/`:

```bash
cd .../codes/Algorithm_codes

python -m alexa.run_alexa toy
python -m alexa.run_alexa ar3 --max-rank 5
python -m alexa.run_alexa setA-01 --max-rank 5 --time-limit 120
```

### CLI arguments

| Flag | Default | Meaning |
|---|---|---|
| `instance` | *(required)* | instance name (e.g. `toy`, `ar3`, `setA-01`) |
| `--data_dir` | `None` | override the auto-detected instance folder |
| `--checker_dir` | `None` | path to the official ROADEF checker (falls back to `CHECKER_DIR`) |
| `--max-rank` | all `N = |A| x |T|` | stop after this many ranks |
| `--time-limit` | `300.0` | Gurobi time limit **per rank/iteration**, seconds |
| `--mip-gap` | `1e-7` | relative MIP gap, applied per rank |
| `--eps` | `1e-4` | one-sided freeze tolerance: `L[k] <= l_k_opt + eps` |

## Outputs

- `results/<instance>/alexa/<instance>-srpaths.json` -- final routing
  (last solved rank) in challenge srpaths format, written via
  `stela.solver.write_solution`.
- `results/<instance>/alexa/<instance>-summary.json` -- `algorithm`
  (`"alexa"`), `instance`, `mlu_baseline`, `mlu_solution`, `reduction_pct`,
  `z_stars`, `checker_valid`, `checker_mlu`.
- `logs/alexa/<instance>_<timestamp>.log` -- full console output, tee'd
  to a timestamped file on every run.

Namespaced under an `alexa` subfolder for both results and logs, so running
STELA/TCG-Diag/TCG-TopK/TCG-Exact/ALEXA on the same instance never
overwrites another method's output.

## Dependencies

`gurobipy` with a valid Gurobi license (auto-detected: `~/gurobi.lic` then
`codes/gurobi.lic`), `networkx` (used only in `segments.py`, for per-period
reachability), `core` (`core.instance`, `core.ecmp`, `core.constraints`,
`core.paths`), and `stela` (`stela.solver.write_solution` only).
