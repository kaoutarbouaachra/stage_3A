# TCG_exact 

**Official name (report):** Trajectory Column Generation with Exact Layered-MIP

TCG-Exact is the trajectory-based method with a valid LP optimality
certificate: pricing is solved to global optimality (layered MIP with the
trail constraint).

---


## How to run

Always **as a module**, from `Algorithm_codes/` 

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
size-limited "Restricted license", which will error out on real instances.
