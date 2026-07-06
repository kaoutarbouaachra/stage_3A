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

## Algorithm

### 1. Segment pruning (`_useful_segments`)

Before building the model, STELA computes a reduced candidate-segment set
per period, instead of using all `|V|^2` node pairs. For each period `t` and
each demand `(src, tgt)`:

- reachability sets `reach_from[v]` / `reach_to[v]` are computed **once per
  period** (via `networkx`, shared across all demands of that period);
- **Type 1** -- the direct segment `(src, tgt)`, if reachable with 0
  waypoints;
- **Type 2** -- `(src, w)` and `(w, tgt)` for every valid waypoint `w`
  (reachable from `src` and able to reach `tgt`), if `max_seg - 1 >= 1`;
- **Type 3** -- `(w1, w2)` intermediate hops between two valid waypoints, if
  `max_seg - 1 >= 2`.

This is demand-aware and typically far smaller than `|V|^2` per period; the
function prints the resulting reduction factor
(`naive = |V|^2 x |T|` vs. the actual count). This pruning is what keeps the
compact formulation's variable count tractable on larger instances.

### 2. Model construction (once)

- `x[d,t,i,j]` -- binary, one per `(demand, period, segment)` triple, only
  over `valid_segs[t]`.
- `delta[d,t,i,j]` -- binary, linearizes `|x^dt_ij - x^{d,t-1}_ij|`, defined
  over `t in T_star` and over the *union* `valid_segs[t] | valid_segs[t-1]`
  (a segment can be newly added or newly removed between consecutive
  periods).
- `lam` -- the `N = |A| x |T|` load expressions, built by
  `core.constraints.build_load_exprs`.
- Base constraints added once, via `core.constraints`: flow conservation
  (C1), segment-count bound (C2), budget (C4, skipped for any period whose
  `kappa` is `None`, i.e. an unconstrained/infinite budget).

### 3. Sequential rank loop

For `k = 1, ..., max_iter` (default `max_iter = N`):

- A new variable `z_k` is created, with `ub` set to the *previous* rank's
  certified value (`z_stars[-1]`) or `+inf` at `k=1` -- a natural monotone
  bound, since each rank's threshold can only be less than or equal to the
  one before it.
- **Rank 1:** `lam[idx] <= z_k` for every load coordinate -- this is exactly
  the MLU. No exceedance variables.
- **Rank k >= 2:** binary exceedance variables `y_k[idx]`, constraint
  `lam[idx] <= z_k + z1_val * y_k[idx]` for every coordinate, plus
  `sum(y_k) <= k - 1`. `z1_val` is the *already-solved numeric value* of
  rank 1's threshold (not a symbolic reference), used as the big-M.
- Objective: minimize `z_k`. Solve.
- On success, `z_k.lb` and `z_k.ub` are both set to `z_k_val +/- 1e-6`,
  freezing the rank. Because the model is never rebuilt, every constraint
  from this and all earlier ranks stays in the model for the rest of the
  run -- this is the mechanism that plays the role of inherited rows in the
  rank-by-rank formulation, without ever restating them.
- Stops early if `z_k_val < 1e-9` (all remaining loads are already zero).

### 4. Solution extraction

`_extract_waypoints` reads the `.X > 0.5` values of `x` per
`(demand, period)`, reconstructs an explicit node path via DFS
(`_reconstruct_path`, following active segments from source to target,
backtracking on dead ends), and keeps only the interior nodes as waypoints
(`path[1:-1]`). Recomputed after every rank, so if the loop stops early for
any reason, the last successfully solved rank's routing is what gets
returned.

`write_solution` writes the challenge srpaths JSON format, omitting entries
with an empty waypoint list. This function is shared: TCG-Diag and TCG-TopK
both import it (`from stela.solver import write_solution`) rather than
duplicating srpaths-writing logic.

## Implementation notes

- **Gurobi status acceptance.** A rank is accepted whenever
  `status in (OPTIMAL, TIME_LIMIT, NODE_LIMIT, SOLUTION_LIMIT)` **and**
  `SolCount > 0` -- i.e. a time-limited, non-certified incumbent is accepted
  just like a proven-optimal one. Because the model persists and old
  constraints are never removed, an accepted suboptimal `z_k` is frozen
  permanently and cannot be revisited by a later rank -- there is no
  opportunity for it to be corrected downstream, unlike in a rebuild-per-rank
  scheme. Worth checking `m.Status == GRB.OPTIMAL` explicitly before trusting
  a run's `z_stars` on instances where `time_limit` may bind.
- **Freezing tolerance: please double-check against your own fix.** This
  version freezes each rank with a **symmetric** band,
  `z_k.lb = z_k_val - 1e-6`, `z_k.ub = z_k_val + 1e-6`, and defaults
  `mip_gap=1e-7`. If the one-sided tolerant freezing fix you've used
  elsewhere (`lb = 0`, `ub = z*_k + 1e-4`, with `MIPGap = 0`) was meant to
  land in this file, it doesn't appear to be present in the version shown
  here -- worth confirming which version is current before relying on
  multi-rank output from this file.
- **Infeasibility aborts the whole run.** If any rank comes back
  `GRB.INFEASIBLE`, STELA calls `computeIIS()`, writes `infeasible.ilp` to
  the working directory, and stops -- there is no fallback to the previous
  rank's solution beyond what was already stored in `best_waypoints`.
- **Big-M is not tightened.** Rank-k constraints use `z1_val` (the solved
  rank-1 value) directly, not a per-rank-tightened `z1_val - zj_val` bound --
  same untightened pattern as TCG-Diag/TCG-TopK.
- **Path reconstruction fallback.** If `_reconstruct_path` cannot find a
  source-to-target walk over the active segments (shouldn't happen if flow
  conservation holds, but not asserted), it returns `[]`, which silently
  becomes "no waypoints" for that demand/period rather than raising -- worth
  being aware of if a solution ever looks suspiciously sparse.
- **Gurobi license:** resolved at import time (top of `solver.py`, before
  other imports), same two candidate paths as the other packages:
  `~/gurobi.lic` then `codes/gurobi.lic` (via `core.paths.CODES_DIR`).

## Function / class reference

- `_useful_segments(inst) -> Dict[int, Set[Tuple[int,int]]]` -- the
  segment-pruning step described above; returns `valid_segs[t]`.
- `_reconstruct_path(src, tgt, active) -> List[int]` -- DFS path
  reconstruction from a set of active `(i,j)` segments.
- `SequentialLexSolver(inst, r, time_limit=300.0, mip_gap=1e-7, max_iter=None, verbose=False)`
  -- builds the persistent model on construction (`self.valid_segs`
  precomputed), runs the rank loop via `.solve()`.
  - `.solve() -> (waypoints, z_stars)` -- runs the full sequential loop;
    `waypoints` is `{(demand_id, t): [w1, w2, ...]}`, `z_stars` is the
    certified prefix `[z*_1, z*_2, ...]`.
  - `._extract_waypoints(x) -> Dict` -- solution extraction, described above.
- `write_solution(waypoints, inst, out_path)` -- writes challenge-format
  srpaths JSON. Shared with TCG-Diag / TCG-TopK.

## Dependencies

`gurobipy` with a valid Gurobi license, `networkx` (used only in
`_useful_segments`, for the per-period reachability computation -- this is
the only package among STELA/TCG-Diag/TCG-TopK/TCG-Exact that depends on
networkx directly rather than going through `core.ecmp`), and `core`
(`core.instance.Instance`, `core.ecmp.compute_all_splits`/`SplitCoeffs`,
`core.constraints`, `core.paths.CODES_DIR`).