# ALEXA

**Assignment-based LEXicographic Algorithm**

Git folder: `alexa`

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

## Provenance

This package is a from-scratch port of a reference Pyomo/CPLEX script (the
algorithm and its assignment-based sorting mechanism were already correct
and are preserved). What changed:

- **`NUMBER_ITERATIONS` was hardcoded to `1`** in the reference script, so
  only rank 1 (the MLU) was ever solved despite the model containing the
  full sorting apparatus for every rank. Replaced by `--max-rank`.
- **The big-M was a flat, invalid constant** (`M_link = 1.0`), even though
  the same model bounded utilization variables up to 100 -- not a valid
  bound once any load exceeds 1.0, which the problem explicitly allows
  under congestion. Replaced by a proven, per-`(arc, period)` bound (see
  `bigm.py` and the note below).
- **Segments were unfiltered** (`x` built over all `|V|^2` pairs per
  period), the same blow-up STELA's own pruning avoids. Replaced by the
  same demand-aware, reachability-based pruning technique (own copy in
  `segments.py`, kept self-contained rather than importing from `stela`).
- **Solver backend switched from Pyomo/CPLEX to native `gurobipy`**, and
  instance loading / ECMP / flow / segment-count / budget / load-expression
  construction now reuse the shared `core.*` building blocks (the same ones
  `stela/solver.py` uses), instead of duplicating that logic. Reusing
  `core.constraints.build_load_exprs` -- which already returns each load as
  a ready-to-use utilization expression -- also means there is **no `u`
  variable and no utilization-defining equality constraint** in this
  version: the expression is used directly inside the big-M-linking
  constraint, the same way STELA uses it directly in its threshold rows.
- **`--time-limit` now applies per rank** (one Gurobi solve per iteration),
  not as a single option set once for the whole run.

## Algorithm

### 1. Segment pruning (`segments.useful_segments`)

Identical technique to `stela._useful_segments` (see that package's
README for the full description): per period, per-node reachability is
computed once and shared across demands; per demand, three segment types
are kept (direct 0-waypoint, first/last-hop 1-waypoint, intermediate-hop
2+-waypoint), bounded by `max_seg`. Prints the resulting reduction factor
vs. the naive `|V|^2` count.

### 2. Model construction (`model.build_alexa_model`, once)

- `x[d,t,i,j]`, `delta[d,t,i,j]` -- same roles and same construction as in
  STELA.
- Base constraints via `core.constraints`: flow conservation, segment-count,
  budget (shared code, not reimplemented).
- `lam` -- the `N` load/utilization expressions via
  `core.constraints.build_load_exprs` (already load/capacity, no separate
  `u` variable needed).
- `L[0..N-1]` -- continuous, sorted-order constraints `L[k] >= L[k+1]`.
- `p[(a,t), k]` -- binary assignment variables; `rank_assignment` (each
  `(a,t)` gets exactly one `k`) and `load_assignment` (each `k` gets exactly
  one `(a,t)`) constraints.
- `big_m_linking`: `L[k] >= lam(a,t) - M(a,t) * (1 - p[(a,t),k])` for every
  `(a,t)` and every `k`, using the per-`(a,t)` big-M from `bigm.py`.

### 3. Sequential rank loop (`solver.AssignmentLexSolver.solve`)

For `k = 0, ..., min(max_rank, N) - 1`:

- `m.setObjective(L[k], GRB.MINIMIZE)`, apply `time_limit` and `mip_gap` for
  this rank, solve.
- On success, freeze with a **one-sided** tolerance,
  `m.addConstr(L[k] <= l_k_opt + eps)` -- no artificial lower bound. This
  matches the reference script's own (correct) pattern, and is
  intentionally different from the current symmetric `+/-1e-6` band in
  `stela/solver.py` (see that package's README for the discrepancy this
  avoids).
- Recompute `best_waypoints` after every successful rank, so an early stop
  still returns the last certified routing.
- Stop early if `l_k_opt < 1e-9`.

### 4. Solution extraction

Same DFS-based reconstruction as STELA (own local copy of
`_reconstruct_path` in `solver.py`, not imported, keeping the package
self-contained), same waypoint convention (`path[1:-1]`). Output writing
reuses `stela.solver.write_solution` -- the one piece of cross-package
sharing, matching what TCG-Diag/TCG-TopK already do.

## The big-M (`bigm.py`)

**Proposition (valid big-M).** For every `(a,t)` and every feasible routing
`P`:

```
0 <= lambda(a,t; P) <= M(a,t)
   = (maxSeg / c(a)) * max_{i != j} r(i,j,a,t) * sum_d nu(d,t)
```

*Proof sketch:* a single demand's contribution to `lambda(a,t) * c(a)` is
`sum_{i,j} r(i,j,a,t) * nu(d,t) * x[d,t,i,j]`; the segment-count constraint
allows at most `maxSeg` of the `x[d,t,i,j]` to be 1, and each active segment
contributes at most `(max_{i,j} r(i,j,a,t)) * nu(d,t)`, so the demand's
total is at most `maxSeg * max_r * nu(d,t)`. Summing over demands gives the
bound.

An earlier version of this file used `M(a,t) = sum_d nu(d,t) / c(a)`, which
implicitly assumes a single demand's path touches a given arc with total
split-coefficient mass at most 1. That is **not** guaranteed by the model: a
demand's SR-path can legally revisit the same physical arc via more than
one of its segments (the trail-vs-repeated-node distinction used elsewhere
in this project), so a single demand's contribution can in principle exceed
`nu(d,t)/c(a)`. The `maxSeg * max_r` factor closes that gap, making the
bound unconditionally valid rather than valid only in the typical case.

This module returns `M(a,t)` **per** `(arc, period)`, not the single global
constant `M = max_{(a,t)} M(a,t)` used for a clean statement of the
proposition above. Nothing in the proof requires uniformity -- the linking
constraint at `(a,t)` only needs a valid bound on `lambda(a,t)` itself --
and the per-component version is strictly tighter whenever arc-time pairs
differ in capacity, demand volume, or split coefficients.

`compute_utilization_bigm(inst, r)` requires the ECMP split coefficients
(for `max_{i,j} r(i,j,a,t)`), so unlike a version depending only on static
instance data, it must be called after `compute_all_splits`. It remains
solve-free: no MILP is solved to derive it.

## Implementation notes

- **Gurobi status acceptance.** Same pattern as STELA and TCG-Diag: a rank
  is accepted whenever `status in (OPTIMAL, TIME_LIMIT, NODE_LIMIT,
  SOLUTION_LIMIT)` and `SolCount > 0`. Because ranks are frozen and never
  revisited, an accepted non-certified incumbent propagates forward with no
  opportunity for correction. Check `m.Status == GRB.OPTIMAL` explicitly if
  you need certified ranks on instances where `--time-limit` may bind.
- **Infeasibility aborts the run.** Same as STELA: `GRB.INFEASIBLE` triggers
  `computeIIS()`, writes `alexa_infeasible.ilp`, and stops.
- **Scales as O(N^2) in the assignment variables**, `N = |A| x |T|` --
  `p[(a,t),k]` has one entry per `(a,t), k)` pair, i.e. `N^2` binaries, plus
  `N^2` big-M-linking constraints, versus STELA's `O(N)` per-rank
  exceedance variables. Observed directly during testing: even the small
  `toy` instance (`N=44`) produces ~2,200 variables and already exceeds a
  size-limited Gurobi license; STELA's equivalent model is far smaller at
  the same instance size. Worth keeping in mind when comparing runtimes
  against STELA/TCG on larger instances -- the gap is structural, not
  incidental.

## Package layout

```
Algorithm_codes/
└── alexa/
    ├── __init__.py
    ├── bigm.py         <- compute_utilization_bigm(inst, r): proven per-(a,t) big-M
    ├── segments.py     <- useful_segments(inst): same pruning technique as STELA
    ├── model.py        <- build_alexa_model(inst, r): variables + constraints, once
    ├── solver.py        <- AssignmentLexSolver: the sequential rank loop
    └── run_alexa.py      <- CLI driver (logging, results, checker integration)
```

Import stack (bottom to top): `bigm.py` and `segments.py` depend only on
`core`; `model.py` imports both plus `core.constraints`; `solver.py`
imports `model.py`; `run_alexa.py` imports `solver.py` and
`stela.solver.write_solution`.

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