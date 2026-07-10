# TCG_Exact 

Trajectory Column Generation with Exact Layered-MIP

TCG-Exact is the trajectory-based method with a valid LP optimality
certificate: pricing is solved to global optimality (layered MIP with the
trail constraint).

---


## How to run

Always **as a module**, from `Algorithm_codes/` 

```bash
cd .../codes/Algorithm_codes

python -m TCG_Exact.run_TCG_Exact toy
python -m TCG_Exact.run_TCG_Exact ar3 --max-rank 5
python -m TCG_Exact.run_TCG_Exact setA-01 --max-rank 5 --time-limit 120
```

## Options

- `--max-rank` — stop after this many ranks (default: all `N = |A|×|T|`).
- `--pricing-tol` — reduced-cost tolerance for accepting an improving column.
- `--freeze-tol` — tolerance for treating a recovered load as zero (early stop).
- `--mip-gap` — relative MIP gap for the integer master (default: `0.0`).
- `--time-limit` — per-rank wall-clock cap (seconds) on the **integer master
  solve only**; the LP column-generation phase is always run to optimality and
  is never time-limited.
- `--out` — write an **extra** copy of the srpaths solution to a custom path;
  the `results/<instance>/TCG_exact/` copy (see **Outputs**) is always written
  regardless.

The instance is resolved by name in the `codes/` tree:
`codes/<name>_instance/` (toy, ar3) or `codes/<prefix>/<name>/` (setA-01 -> setA).


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


---

## Warm start

`run_TCG_exact_warm` is the **same** sequential leximin method as
`run_TCG_Exact` — same rank-by-rank STELA loop, same `z_k*` recovery from the
integer solution's sorted load vector, same freeze tolerance, same exact
pricing enabled at every rank — with **only** the initial pool changed. The
heuristic columns are *added on top of* the trivial basic pool (never
replacing it), so the warm start accelerates the search without shrinking what
the method can certify.

### How to run

```bash
cd .../codes/Algorithm_codes

# default: seed from results//TCG_diag/-srpaths.json
python -m TCG_Exact.run_TCG_exact_warm toy
python -m TCG_Exact.run_TCG_exact_warm setA-01 --max-rank 5 --time-limit 120

# seed from another heuristic's srpaths (must have been run first)
python -m TCG_Exact.run_TCG_exact_warm setA-01 --warm-algo TCG_topk

# seed from an arbitrary srpaths file (overrides --warm-algo)
python -m TCG_Exact.run_TCG_exact_warm setA-01 --warm-srpaths /path/to/seed-srpaths.json
```

### Extra options

All of the `run_TCG_Exact` options above apply, plus:

- `--warm-algo` — the heuristic whose srpaths seeds the pool, read from
  `results/<instance>/<warm-algo>/<instance>-srpaths.json`
  (default: `TCG_diag`).
- `--warm-srpaths` — explicit path to a srpaths.json to seed from. **Overrides**
  `--warm-algo` when given.

The warm start reads its seed from the official challenge srpaths format, so
any heuristic that writes srpaths can be used. It does **not** fabricate a
seed: the source file must already exist. With the default `--warm-algo`, that
means **TCG-Diag must have been run first** on the instance, otherwise the
driver exits with a `FileNotFoundError` telling you which heuristic to run.

### Warm-start outputs

Namespaced under a **separate** `TCG_exact_warm` subfolder so warm and cold
runs never overwrite each other:

- `results/<instance>/TCG_exact_warm/<instance>-srpaths.json` -- final routing
  in challenge srpaths format. Always written; extra copy to `--out <path>` if
  given.
- `results/<instance>/TCG_exact_warm/<instance>-summary.json` -- `algorithm`
  (`"TCG_exact_warm"`), `instance`, a `warm_start` block (`source`,
  `warm_algo` — `null` when `--warm-srpaths` is used, `unresolved_entries`),
  `ranks_certified`, `z_stars`, `pool_size`, `time_s`.
- `logs/TCG_exact_warm/<instance>_<timestamp>.log` -- full console output,
  tee'd to a timestamped file on every run.


## Dependencies

`gurobipy` with a valid Gurobi license, + your `core` (which brings in
nt4python or NetworkX depending on `T_ASR_ECMP_BACKEND`). The engine does not
use NetworkX directly; it goes through `core.ecmp.compute_all_splits(inst)`.

The Gurobi license is picked up automatically, same pattern as the other
packages: `data.py` looks for `~/gurobi.lic` then `codes/gurobi.lic` (via
`core.paths.CODES_DIR`) and sets `GRB_LICENSE_FILE` before importing
gurobipy. Without a named/WLS license there, Gurobi falls back to the
size-limited "Restricted license", which will error out on real instances.
