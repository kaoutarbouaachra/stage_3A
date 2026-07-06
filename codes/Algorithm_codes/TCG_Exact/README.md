# TCG_exact 

**Official name (report):** Trajectory Column Generation with Exact Layered-MIP

TCG-Exact is the only trajectory-based method with a valid LP optimality
certificate: pricing is solved to global optimality (layered MIP with the
trail constraint, `MIPGap=0`), so "no negative reduced-cost column found"
provably means the restricted master LP equals the full master LP at that
rank (see the accompanying note, Corollary 1 / Corollary 2).

---

Column generation with **exact pricing** for T-ASR (horizon h=2, to be
generalized to h>2). A column is a *trajectory* (a pair of segment paths over
the two periods), which makes the reconfiguration budget linear in the
master. The pricing is a layered MIP with the **trail** constraint
(`sum_ell y[t,ell,e] <= 1`, no repeated segment). The rank-k master uses the
tightened big-M `z1* - zj*` for inherited ranks.

## Where to put the folder

`TCG_exact/` is a package that is a **sibling** of `core/`, next to your other
packages (`TCG_diag/`, `TCG_topk/`, `heuristique_by_preprocessing/`, ...):

```
codes/
├── Algorithm_codes/
│   ├── core/                     <- instance.py, ecmp.py, ecmp_from_scratch.py, ecmp_nt.py
│   ├── TCG_diag/
│   ├── TCG_topk/
│   ├── heuristique_by_preprocessing/
│   └── TCG_exact/                <- THIS FOLDER
│       ├── __init__.py
│       ├── data.py
│       ├── pricing.py
│       ├── master.py
│       ├── adapter.py            <- the only file that imports core
│       └── run_TCG_exact.py      <- the single CLI driver (sequential leximin)
├── toy_instance/                 <- toy-net.json, toy-tm.json, toy-scenario.json
├── ar3_instance/
└── setA/setA-01/ ...
```

There is a single runner, `run_TCG_exact.py`. It always runs the full
sequential leximin loop (rank by rank); there is no separate rank-1-only
tool and no `--smoketest` flag.

## How to run

Always **as a module**, from `Algorithm_codes/` (so that `from core import ...`
resolves, exactly like your other packages):

```bash
cd .../codes/Algorithm_codes

python -m TCG_exact.run_TCG_exact toy
python -m TCG_exact.run_TCG_exact ar3 --max-rank 5
python -m TCG_exact.run_TCG_exact setA-01 --max-rank 5 --time-limit 120
```

Options: `--max-rank` (stop after this many ranks; default: all `N=|A|x|T|`),
`--pricing-tol`, `--freeze-tol`, `--mip-gap`, `--time-limit`, `--out` (an
*extra* copy of the srpaths solution at a custom path -- the
`results/<instance>/TCG_exact/` copy described under Outputs is always
written regardless).

The instance is resolved by name in the `codes/` tree:
`codes/<name>_instance/` (toy, ar3) or `codes/<prefix>/<name>/` (setA-01 -> setA).

## Architecture (import stack)

```
data.py         pure structures (TASRData, Demand, Column) -- depends on NOTHING
   ^
pricing.py      ExactPairPathPricer (layered MIP + trail)   -- imports data
   ^
master.py       PairPathColumnGeneration (CG + integer)     -- imports data, pricing
   ^
adapter.py      bridge core -> TASRData                       -- imports core, data
   ^
run_TCG_exact.py  CLI (logging, results, sequential leximin) -- imports adapter, master
```

## What the runner does

`run_TCG_exact.py` always runs the full **sequential leximin loop**: for each
rank k, it runs CG to the LP certificate, solves the integer master on the
pool, recovers `z_k*` from the resulting sorted load vector (not from the
master's `z` variable, to avoid rank-to-rank drift), and freezes it as an
inherited constraint for rank k+1. It stops early once the remaining loads
are ~0. Logging and result-saving (see Outputs below) are automatic on every
run.

- The integer master is solved **over the generated pool only** at each rank
  (exact w.r.t. that pool, not yet a global integer optimality proof).
  Branch-and-price (pricing re-solved at each B&B node, branching on the
  original x^dt_ij variables via `forced_segment_values` in `pricing.py`) is
  the planned next step for full integer exactness.
- Currently hardcoded to h=2 periods; generalizing to h>2 (per the note's
  Section 12) is on the roadmap and touches `data.py` (Column.beta becomes a
  per-transition dict), `pricing.py` (the h-linearization becomes one block
  per consecutive period pair), and `adapter.py` (budget becomes
  {t: kappa(t)} instead of a single kappa(1)).

## Outputs

- `results/<instance>/TCG_exact/<instance>-srpaths.json` -- the final
  routing (last solved rank) in challenge srpaths format. Always written; an
  extra copy also goes to `--out <path>` if given.
- `results/<instance>/TCG_exact/<instance>-summary.json` -- `algorithm`
  (`"TCG_exact"`), `instance`, `ranks_certified`, `z_stars` (the certified
  rank prefix), `pool_size`, `time_s`.
- `logs/TCG_exact/<instance>_<timestamp>.log` -- full console output,
  tee'd to a timestamped file on every run.

Results and logs are namespaced under a `TCG_exact` subfolder specifically so
that running TCG-Diag, TCG-TopK, and TCG-Exact on the same instance never
overwrites another method's output.

## Dependencies

`gurobipy` with a valid Gurobi license, + your `core` (which brings in
nt4python or NetworkX depending on `T_ASR_ECMP_BACKEND`). The engine does not
use NetworkX directly; it goes through `core.ecmp.compute_all_splits(inst)`.

The Gurobi license is picked up automatically, same pattern as the other
packages: `data.py` looks for `~/gurobi.lic` then `codes/gurobi.lic` (via
`core.paths.CODES_DIR`) and sets `GRB_LICENSE_FILE` before importing
gurobipy. Without a named/WLS license there, Gurobi falls back to the
size-limited "Restricted license", which will error out on real instances
(too many variables for setA-01's ~40 demands x ~800 candidate segments).
