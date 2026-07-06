# STELA

**Sequential Threshold-Exceedance Leximin Algorithm**
## Overview

STELA solves T-ASR directly on the **compact formulation**: the binary
routing variables `x[d,t,i,j]` and the reconfiguration-linearization
variables `delta[d,t,i,j]` from the original problem statement, not a
column-generation reformulation. It minimizes the sorted load vector in
lexicographic order, one threshold rank at a time.

The defining architectural choice, compared to the trajectory-based CG
methods (TCG-Diag / TCG-TopK / TCG-Exact): **STELA builds a single Gurobi
model once and extends it in place across all ranks**, rather than
rebuilding a restricted master per rank. Each rank's threshold constraints
are added to the model and never removed; once a rank is solved, its
threshold variable is frozen to a narrow band, so the constraints from every
earlier rank stay active automatically. There is no need to reconstruct
inherited `(Tj, Nj)` rows at each rank -- the persistent model does that
implicitly, by construction.



## How to run

Run as a module from `Algorithm_codes/`:

python -m exact.main --instance setA-01

## Options

These are the tuning parameters exposed by `SequentialLexSolver`:

| Option        | Default | Description                                                                                                   |
| ------------- | ------- | ------------------------------------------------------------------------------------------------------------- |
| `time_limit`  | `300.0` | Gurobi time limit **per sub-problem** (seconds), i.e. applied to each iteration `k` independently.            |
| `mip_gap`     | `1e-7`  | Relative MIP optimality gap for each sub-problem.                                                             |
| `max_iter`    | `N`     | Stop after this many iterations. Default is `N = |A|·|T|` (all arc-time components).                          |
| `verbose`     | `False` | Print the full Gurobi log for each solve.
                                                              |
## Dependencies

`gurobipy` with a valid Gurobi license, `networkx` (used only in
`_useful_segments`, for the per-period reachability computation -- this is
the only package among STELA/TCG-Diag/TCG-TopK/TCG-Exact that depends on
networkx directly rather than going through `core.ecmp`), and `core`
(`core.instance.Instance`, `core.ecmp.compute_all_splits`/`SplitCoeffs`,
`core.constraints`, `core.paths.CODES_DIR`).
