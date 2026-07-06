# TCG-Diag

**Trajectory Column Generation with Exact-Diagonal / Top-K Off-Diagonal Pricing**

Git folder: `TCG_diag`

## Overview

TCG-Diag is a rank-by-rank lexicographic column-generation algorithm for
T-ASR, specialized to a two-period horizon (h = 2; the code asserts
`len(inst.T) == 2`). The decision unit is a trajectory `(p0, p1)`, one segment
path per period, so the reconfiguration budget is a constant per column.

What defines TCG-Diag is its pricing oracle:

- the **diagonal case** (the same path reused in both periods, zero
  reconfiguration cost) is priced **exactly**, via a Bellman-Ford bounded-hop
  shortest path;
- the **off-diagonal case** (the two periods use different paths) is priced
  **heuristically**, returning up to `--topk_cols` candidate trajectories per
  demand per CG iteration (`price_demand_topk`).

Because the off-diagonal oracle is a heuristic (sound but not complete: any
column it returns has a genuine negative reduced cost, but it may fail to
find one that exists), "no improving column added" is a stopping heuristic
for this method, not a proof that the LP relaxation has been solved to
optimality at that rank.

## Algorithm

A pool of trajectories, shared and monotonically growing across all
lexicographic ranks. For each rank `k = 1, ..., K`:

1. **Inner CG loop:** solve the LP relaxation of the rank-k restricted master
   (with inherited rows `(Tj, Nj)` for `j < k`), read the stratified duals
   `mu_bar_k(a,t) = mu_k(a,t) + sum_{j<k} eta_j(a,t)`, price every demand
   (exact diagonal + heuristic top-K off-diagonal), add improving
   trajectories to the shared pool. Repeat until no demand yields an
   improving column.
2. Solve the rank-k **integer** master on the current pool to obtain `z_k*`.
3. Freeze `z_k*` as an inherited row for rank `k + 1`.

## Implementation notes

- **Dual sign convention:** `sigma_d` free, `mu_k <= 0`, `eta_j <= 0`,
  `psi(t) <= 0`. Pricing uses `-mu_bar` as the per-`(a,t)` load pressure and
  `rho = -psi(1)` as the budget penalty for the single h=2 transition.
- **LP duals are only meaningful at proven optimality.** Duals are read from
  the LP relaxation only if Gurobi returns `GRB.OPTIMAL`; `mu = mu_bar` is
  only correct once the LP has actually converged.
- **Big-M in the inherited rows** is `z1_star` (the global max), not a
  tightened per-rank value: `lam[at] <= zj + z1_star * y[(j,)+at]`. Valid,
  but a looser LP relaxation at deep ranks than a tightened bound would give.
- **The rank-k MIP accepts any feasible incumbent, not only a proven-optimal
  one.** If `--mip_time` is hit before Gurobi proves optimality, `z_k*` is
  taken from whatever incumbent was found, and that value is then inherited
  as a hard constraint at rank k+1 -- so a time-limited, suboptimal `z_k*`
  can silently propagate down the lexicographic chain. Check
  `mip_result["status"] == GRB.OPTIMAL` if you need certified ranks,
  especially on larger instances where `--mip_time` may bind.
- **Initial pool:** one trivial trajectory per demand, `([], [])` (pure IGP
  routing, no waypoints, both periods).
- **Instance resolution** (`detect_base`): `setA*` -> `CODES_DIR/setA/<name>`,
  `toy` -> `CODES_DIR/toy_instance`, else -> `CODES_DIR/<name>_instance`.
- **Logging:** every run writes a timestamped log to
  `Algorithm_codes/logs/TCG_diag/<instance>_<timestamp>.log`, tee'd to
  stdout. The console banner also prints "Algorithm: TCG-Diag" so console/log
  output is self-identifying.
- **Gurobi license:** `~/gurobi.lic` then `codes/gurobi.lic` (via
  `core.paths.CODES_DIR`), set as `GRB_LICENSE_FILE` before any Gurobi call.

## Package layout

```
Algorithm_codes/
└── TCG_diag/
    ├── cg_master.py         <- TrajectoryPool, trajectory_budget_cost
    ├── pricing_dp.py        <- price_demand_topk, build_pricing_context
    │                           (exact diagonal Bellman-Ford + heuristic
    │                            top-K off-diagonal)
    └── run_tcg_diag.py       <- driver (this file)
```

## Usage

Run as a module from `Algorithm_codes/`:

```bash
cd .../codes/Algorithm_codes

python -m TCG_diag.run_tcg_diag --instance setA-01 --K 5
python -m TCG_diag.run_tcg_diag --instance setA-10 --K 3 --time_limit 900
```

### CLI arguments

| Flag | Default | Meaning |
|---|---|---|
| `--instance` | *(required)* | instance name (e.g. `toy`, `ar3`, `setA-01`) |
| `--data_dir` | `None` | override the auto-detected instance folder |
| `--checker_dir` | `../checker` | path to the official ROADEF checker |
| `--K` | `5` | number of lexicographic ranks to certify |
| `--time_limit` | `900.0` | global wall-clock budget (seconds) |
| `--lp_time` | `90.0` | max time per LP-RMP solve |
| `--mip_time` | `180.0` | max time per rank's integer master |
| `--max_cg_iter` | `20` | max CG iterations per rank |
| `--topk_cols` | `3` | trajectories added per demand per CG iteration |
| `--rc_tol` | `-1e-6` | reduced-cost tolerance for accepting a column |

## Outputs

- `results/<instance>/TCG_diag/<instance>-srpaths.json` -- solution in
  challenge srpaths format.
- `results/<instance>/TCG_diag/<instance>-summary.json` -- run summary: MLU
  baseline/solution, `z_stars` (the certified rank prefix), top-5 sorted
  loads before/after, pool size, total time. Includes an `"algorithm":
  "TCG_diag"` field so a file found outside its folder is still
  self-identifying.
- If the official checker is found under `--checker_dir`, its feasibility
  verdict and checker-computed MLU are printed, compared against the
  in-process value, and included in the summary JSON (`checker_valid`,
  `checker_mlu`).
- `logs/TCG_diag/<instance>_<timestamp>.log`.

Results and logs are namespaced under a `TCG_diag` subfolder specifically so
that running TCG-Diag, TCG-TopK, and TCG-Exact on the same instance never
overwrites another method's output.

## Dependencies

`gurobipy` with a valid Gurobi license, plus `core` (instance loading, ECMP)
and `stela` (solution writer, `write_solution`).