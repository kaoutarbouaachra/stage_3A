# TCG-TopK

**Trajectory Column Generation with Fully Heuristic Top-K Pricing**



## Overview

TCG-TopK is a trajectory-based column-generation algorithm for T-ASR. The
decision unit is a trajectory `(p0, p1)`, one segment path per period, so the
reconfiguration budget is constant per column.

What defines TCG-TopK is its pricing oracle: **both** the diagonal case
(same path reused in both periods) and the off-diagonal case (different
paths per period) are priced through the **same top-K heuristic
enumeration** (`price_demand_topk`), returning up to `--topk_cols` candidate
trajectories per demand per column-generation iteration. There is no exact
sub-case -- unlike a hybrid diagonal/off-diagonal pricer, TCG-TopK does not
guarantee an exact shortest path even for the (structurally simpler)
diagonal case.

The heuristic is **sound but not complete**: any trajectory it returns has a
genuine negative reduced cost, but failing to find one does not prove none
exists. The per-rank LP bound this driver reports is therefore a heuristic
bound, not a certified one.

## Algorithm

A pool of trajectories, shared and monotonically growing across all
lexicographic ranks. For each rank `k = 1, ..., K`:

1. **Inner CG loop:** solve the LP relaxation of the rank-k restricted master
   (with inherited rows `(Tj, Nj)` for `j < k`), extract the stratified duals
   `mu_bar_k(a,t)`, price every demand with the top-K heuristic, add
   improving trajectories to the shared pool. Repeat until no demand yields
   an improving column.
2. Solve the rank-k **integer** master on the current pool to obtain `z_k*`.
3. Freeze `z_k*` as an inherited row for rank `k + 1`.



'''

Run as a module from `Algorithm_codes/` (so that the absolute imports
`from core import ...` and `from stela.solver import ...` resolve):

```bash
cd .../codes/Algorithm_codes
python -m TCG_topk.run_TCG_TopK --instance setA-01 --K 5
python -m TCG_topk.run_TCG_TopK --instance setA-10 --K 3 --time_limit 900
```

### CLI arguments

| Flag | Default | Meaning |
|---|---|---|
| `--instance` | *(required)* | instance name (e.g. `toy`, `ar3`, `setA-01`) |
| `--data_dir` | `None` | override the auto-detected instance folder |
| `--checker_dir` | `None` | path to the official ROADEF checker (falls back to `CHECKER_DIR`) |
| `--K` | `5` | number of lexicographic ranks to certify |
| `--time_limit` | `900.0` | global wall-clock budget (seconds) |
| `--lp_time` | `90.0` | max time per LP-RMP solve |
| `--mip_time` | `180.0` | max time per rank's integer master |
| `--max_cg_iter` | `20` | max CG iterations per rank |
| `--topk_cols` | `3` | trajectories added per demand per CG iteration |
| `--rc_tol` | `-1e-6` | reduced-cost tolerance for accepting a column |

## Outputs

- `results/<instance>/TCG_topk/<instance>-srpaths.json` -- solution in
  challenge srpaths format (written via `stela.solver.write_solution`).
- `results/<instance>/TCG_topk/<instance>-summary.json` -- run summary: MLU
  baseline/solution, `z_stars` (the certified rank prefix), top-5 sorted
  loads before/after, pool size, total time. Includes an `"algorithm":
  "TCG_topk"` field so a file found outside its folder is still
  self-identifying.
- If the official checker is found under `--checker_dir` (or `CHECKER_DIR`),
  its feasibility verdict and checker-computed MLU are printed, compared
  against the in-process value, and included in the summary JSON
  (`checker_valid`, `checker_mlu`).
- `logs/TCG_topk/<instance>_<timestamp>.log`.

Results and logs are namespaced under a `TCG_topk` subfolder specifically so
that running TCG-Diag, TCG-TopK, and TCG-Exact on the same instance never
overwrites another method's output.

## Dependencies

`gurobipy` with a valid Gurobi license, plus `core` (instance loading, ECMP)
and `stela` (solution writer, `write_solution`).
