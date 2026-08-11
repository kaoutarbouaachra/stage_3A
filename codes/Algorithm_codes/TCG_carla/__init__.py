"""
TCG_carla
=========

CARLA-hardcoded trajectory column generation for T-ASR (h = 2), structured like
the STELA packages TCG_Exact / TCG_diag but with the cumulative Ogryczak-Tamir
rank encoding baked in (no --rank-encoding flag).

Only the master rank rows and the drivers live here. The column data model, the
exact pricer, the adapter and the diagonal pricing are IDENTICAL to STELA, so
they are reused as-is from the existing packages rather than duplicated:

    from TCG_Exact.data     import ...      (Column, TASRData, ...)
    from TCG_Exact.pricing  import ExactPairPathPricer
    from TCG_Exact.adapter  import build_tasr_data_from_files
    from TCG_diag.cg_master import TrajectoryPool, trajectory_budget_cost
    from TCG_diag.pricing_dp import price_demand_topk, build_pricing_context

This keeps a single source of truth for the pricing oracle: TCG_carla and the
STELA packages price with the very same code, so a STELA-vs-CARLA comparison is
"all else equal" by construction. Run as modules from Algorithm_codes/:

    python -m TCG_carla.run_exact       <instance> ...
    python -m TCG_carla.run_exact_warm  <instance> ...
    python -m TCG_carla.run_diag        <instance> ...

Output namespaces (never collide with the STELA runs):
    results/<inst>/TCG_EXACT_CARLA/        logs/TCG_EXACT_CARLA/
    results/<inst>/TCG_EXACT_WARM_CARLA/   logs/TCG_EXACT_WARM_CARLA/
    results/<inst>/TCG_DIAG_CARLA/         logs/TCG_DIAG_CARLA/
"""
