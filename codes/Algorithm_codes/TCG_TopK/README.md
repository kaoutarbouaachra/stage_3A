# TCG-TopK

**Trajectory Column Generation with Fully Heuristic Top-K Pricing**

Git folder: `TCG_topk`

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

The reduced cost of a candidate trajectory `(p0, p1)` for demand `d` is

```
rc = -pi_d
     + sum_{(i,j) in segs(p0)} theta_d(i,j,0)
     + sum_{(i,j) in segs(p1)} theta_d(i,j,1)
     - nu(1) * dist(p0, p1)

theta_d(i,j,t) = sum_a (-mu_bar_k(a,t)) * nu(d,t)/c(a) * r(i,j,a,t)
```

where `nu(1) <= 0` is the budget dual (so `-nu(1) >= 0` acts as the budget
penalty).

## Implementation notes

- **Solver status check is shared between the LP and the MIP solve, and
  accepts `TIME_LIMIT` / `SUBOPTIMAL` for both:**
  ```python
  if m.Status not in (GRB.OPTIMAL, GRB.SUBOPTIMAL, GRB.TIME_LIMIT):
      return None
  ```
  LP duality only guarantees meaningful reduced costs at a proven-optimal LP
  vertex. If `--lp_time` is hit before Gurobi proves optimality, this driver
  will still read and use the duals from that time-limited solve, which can
  silently feed incorrect pricing signals into the column generation loop
  and propagate through the lexicographic chain. If certified pricing
  matters for your experiments, consider requiring `GRB.OPTIMAL` for the
  `binary=False` (LP) case specifically.
- **No explicit two-period guard.** Trajectories are built as fixed pairs
  `(traj[0], traj[1])` throughout, so the driver is implicitly specialized to
  h = 2, but there is no early, explicit check on `len(inst.T)` -- an
  instance with a different horizon would likely fail later with an
  index/key error rather than a clear message at load time.
- **Gamma/beta are recomputed for the whole pool on every model build.**
  `_precompute_gamma(pool)` iterates every trajectory of every demand fresh
  at the top of each `_build_model` call (LP and MIP alike), with no
  persistent cache across calls or ranks. Simple, but strictly more
  recomputation than an incremental per-trajectory cache as the pool and
  rank count grow.
- **Initial pool:** one trivial trajectory per demand, `([], [])` (pure IGP
  routing, no waypoints, both periods).
- **Instance resolution** (`detect_base`): `setA*` -> `CODES_DIR/setA/<name>`,
  `toy` -> `CODES_DIR/toy_instance`, else -> `CODES_DIR/<name>_instance`.
- **Logging:** every run writes a timestamped log to
  `Algorithm_codes/logs/TCG_topk/<instance>_<timestamp>.log`, tee'd to
  stdout. The console banner also prints "Algorithm: TCG-TopK" so console/log
  output is self-identifying.
- **Gurobi license:** `~/gurobi.lic` then `codes/gurobi.lic` (via
  `core.paths.CODES_DIR`), set as `GRB_LICENSE_FILE` before any Gurobi call.

## Package layout

```
Algorithm_codes/
└── TCG_topk/
    ├── cg_master0.py            <- TrajectoryPool; trajectory_load_contribution,
    │                               trajectory_budget_cost
    ├── pricing_dp_heuristic.py  <- price_demand_topk (top-K heuristic pricing
    │                               for both the diagonal and off-diagonal
    │                               cases)
    └── run_TCG_TopK.py          <- driver (this file)
```

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