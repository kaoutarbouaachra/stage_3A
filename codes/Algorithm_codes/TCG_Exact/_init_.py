#!/usr/bin/env python3
"""
cg_exact
========

Exact pair-path column generation for two-period T-ASR.

Layers:
    data      : TASRData, Demand, Column, exceptions (engine, no core dep)
    pricing   : ExactPairPathPricer (corrected layered MIP, trail constraint)
    master    : PairPathColumnGeneration (root-node CG + integer-on-pool)
    adapter   : bridge from a core instance + ECMP to TASRData
    run_cg_exact : CLI runner

Run as a module from Algorithm_codes/:
    python -m cg_exact.run_cg_exact toy
    python -m cg_exact.run_cg_exact --smoketest
"""

from .data import (
    Column,
    Demand,
    GurobiUnavailableError,
    MasterInfeasibleError,
    PricingInfeasibleError,
    PricingNotOptimalError,
    TASRData,
    TASRError,
)
from .pricing import ExactPairPathPricer, PricingResult
from .master import (
    ColumnGenerationLogEntry,
    ColumnGenerationResult,
    MasterResult,
    PairPathColumnGeneration,
)

__all__ = [
    "TASRData",
    "Demand",
    "Column",
    "TASRError",
    "GurobiUnavailableError",
    "MasterInfeasibleError",
    "PricingInfeasibleError",
    "PricingNotOptimalError",
    "ExactPairPathPricer",
    "PricingResult",
    "PairPathColumnGeneration",
    "MasterResult",
    "ColumnGenerationLogEntry",
    "ColumnGenerationResult",
]