"""
TCG_carla
=========

CARLA-hardcoded trajectory column generation for T-ASR (h = 2), structured like
the STELA packages TCG_Exact / TCG_diag but with the cumulative Ogryczak-Tamir
.

    python -m TCG_carla.run_exact       <instance> ...
    python -m TCG_carla.run_exact_warm  <instance> ...
    python -m TCG_carla.run_diag        <instance> ...

Output namespaces (never collide with the STELA runs):
    results/<inst>/TCG_EXACT_CARLA/        logs/TCG_EXACT_CARLA/
    results/<inst>/TCG_EXACT_WARM_CARLA/   logs/TCG_EXACT_WARM_CARLA/
    results/<inst>/TCG_DIAG_CARLA/         logs/TCG_DIAG_CARLA/
"""
