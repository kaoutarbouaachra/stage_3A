# cg_carla — CARLA-driven trajectory column generation


## The one idea

The pricing subproblem `(SP_d)` has the **same form** as in the STELA-driven CG,
so the two oracles (`price_diag`, `price_exact`) are **identical in role** to the
STELA ones: they consume `(theta, rho, sigma_d)` and never see the rank rows.
The **only** new component is the master (`master.py`): the CARLA cumulative
rank-`k` block replaces STELA's threshold-exceedance rows.

| component        | file               | shared with STELA? |
|------------------|--------------------|--------------------|
| rank master      | `master.py`        | **no this is the new part!** |
| effective costs  | `costs.py`         | yes (formula)      |
| diag oracle      | `pricing_diag.py`  | yes                |
| exact oracle     | `pricing_exact.py` | yes                |
| lex loop         | `loop.py`          | yes (skeleton)     |

