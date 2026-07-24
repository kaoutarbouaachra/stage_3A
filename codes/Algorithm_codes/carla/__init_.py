"""
CARLA -- Cumulative Aggregation of Ranked Loads Algorithm.

Cumulative (Ogryczak-Tamir) formulation for the lexicographic min-max
objective of T-ASR.
"""

from .solver import CumulativeLexSolver, write_solution

__all__ = ["CumulativeLexSolver", "write_solution"]