# STELA

**Sequential Threshold-Exceedance Leximin Algorithm**

## How to run

Run as a module from `Algorithm_codes/`:

python -m stela.main --instance setA-01

## Options

These are the tuning parameters exposed by `SequentialLexSolver`:

| Option        | Default | Description                                                                                                   |
| ------------- | ------- | ------------------------------------------------------------------------------------------------------------- |
| `time_limit`  | `300.0` | Gurobi time limit **per sub-problem** (seconds), i.e. applied to each iteration `k` independently.            |
| `mip_gap`     | `1e-8`  | Relative MIP optimality gap for each sub-problem.                                                             |
| `epsilon`     | `1e-9`  | Half-width of the window used to freeze `z*_k` between iterations: after solving `P_k`, `z_k` is fixed to `[z*_k − epsilon, z*_k + epsilon]`. |
| `max_iter`    | `N`     | Stop after this many iterations. Default is `N = |A|·|T|` (all arc-time components).                          |
| `verbose`     | `False` | Print the full Gurobi log for each solve.                                                                      |
## Dependencies

`gurobipy` with a valid Gurobi license, `networkx` (used only in
`_useful_segments`, for the per-period reachability computation -- this is
the only package among STELA/TCG-Diag/TCG-TopK/TCG-Exact that depends on
networkx directly rather than going through `core.ecmp`), and `core`
(`core.instance.Instance`, `core.ecmp.compute_all_splits`/`SplitCoeffs`,
`core.constraints`, `core.paths.CODES_DIR`).
