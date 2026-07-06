#!/usr/bin/env python3
"""
cg_exact.data
=============

Data structures for the two-period pair-path T-ASR formulation, plus the
Gurobi import guard shared by the rest of the package.

This module has NO dependency on ``core``; it is the pure engine layer.
The bridge to ``core`` (instance loading + ECMP split coefficients) lives in
:mod:`cg_exact.adapter`.

Nodes, arcs, and demand ids may be strings, integers, or tuples, as long as
they are hashable. Arcs are arbitrary hashable objects; for network arcs,
tuples ``(u, v)`` are natural and are what the adapter produces.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import (
    Any,
    Dict,
    Hashable,
    Iterable,
    List,
    Mapping,
    Optional,
    Sequence,
    Set,
    Tuple,
)

# ---------------------------------------------------------------------------
# Gurobi licence setup (same pattern as the other packages): point
# GRB_LICENSE_FILE at ~/gurobi.lic or codes/gurobi.lic if present, BEFORE
# importing gurobipy. This activates a named/WLS/academic licence and avoids
# the size-limited "Restricted license".
# ---------------------------------------------------------------------------
import os as _os
from pathlib import Path as _Path


def _setup_gurobi_licence():
    candidates = [_Path.home() / "gurobi.lic"]
    try:
        from core.paths import CODES_DIR as _CODES_DIR  # type: ignore
        candidates.append(_CODES_DIR / "gurobi.lic")
    except Exception:
        pass
    for p in candidates:
        if p.exists():
            _os.environ["GRB_LICENSE_FILE"] = str(p)
            return str(p)
    return None


_GUROBI_LICENCE = _setup_gurobi_licence()

# ---------------------------------------------------------------------------
# Gurobi import guard (shared by .pricing and .master via `from .data import`)
# ---------------------------------------------------------------------------
try:
    import gurobipy as gp
    from gurobipy import GRB
except Exception as exc:  # pragma: no cover
    gp = None
    GRB = None
    _GUROBI_IMPORT_ERROR = exc
else:
    _GUROBI_IMPORT_ERROR = None


# ---------------------------------------------------------------------------
# Type aliases
# ---------------------------------------------------------------------------
Node = Hashable
Arc = Hashable
DemandId = Hashable
Period = int
Segment = Tuple[Node, Node]
Path = Tuple[Segment, ...]
TrajectoryKey = Tuple[Path, Path]
ArcTime = Tuple[Arc, Period]


# ---------------------------------------------------------------------------
# Exceptions
# ---------------------------------------------------------------------------
class TASRError(RuntimeError):
    """Base exception for this package."""


class GurobiUnavailableError(TASRError):
    """Raised when gurobipy is unavailable."""


class MasterInfeasibleError(TASRError):
    """Raised when the restricted master is infeasible."""


class PricingInfeasibleError(TASRError):
    """Raised when a demand has no feasible pair-path in pricing."""


class PricingNotOptimalError(TASRError):
    """Raised when exact pricing is not solved to global optimality."""


def _require_gurobi() -> None:
    if gp is None or GRB is None:
        raise GurobiUnavailableError(
            "gurobipy could not be imported. Install gurobipy and ensure that "
            "a valid Gurobi license is available."
        ) from _GUROBI_IMPORT_ERROR


# ---------------------------------------------------------------------------
# Small helpers
# ---------------------------------------------------------------------------
def safe_name(obj: Any) -> str:
    """Create a Gurobi-safe name fragment."""
    s = str(obj)
    out = []
    for ch in s:
        out.append(ch if (ch.isalnum() or ch == "_") else "_")
    result = "".join(out).strip("_")
    return result[:80] if result else "x"


def padded_path(path: Path, target: Node, K: int) -> Tuple[Segment, ...]:
    """Pad a path with the dummy target self-loop up to K layers."""
    dummy = (target, target)
    if len(path) > K:
        raise ValueError(f"Path length {len(path)} exceeds K={K}.")
    return tuple(path) + tuple(dummy for _ in range(K - len(path)))


# ---------------------------------------------------------------------------
# Core data structures
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class Demand:
    """Traffic demand."""
    id: DemandId
    source: Node
    target: Node
    volume: Mapping[Period, float]


@dataclass(frozen=True)
class Column:
    """Pair-path trajectory column for one demand."""
    demand_id: DemandId
    paths: Mapping[Period, Path]
    beta: float
    gamma: Mapping[ArcTime, float]
    name: str = ""

    def key(self) -> TrajectoryKey:
        return (tuple(self.paths[0]), tuple(self.paths[1]))


@dataclass
class TASRData:
    """
    Input data for the two-period pair-path T-ASR formulation.

    routing_fraction is indexed as routing_fraction[(segment, arc, period)].
    Missing routing coefficients are interpreted as zero.
    """

    nodes: Sequence[Node]
    arcs: Sequence[Arc]
    capacities: Mapping[Arc, float]
    demands: Sequence[Demand]
    feasible_segments: Mapping[Tuple[DemandId, Period], Iterable[Segment]]
    routing_fraction: Mapping[Tuple[Segment, Arc, Period], float]
    max_segments: int
    budget: float
    periods: Tuple[Period, Period] = (0, 1)

    _demand_by_id: Dict[DemandId, Demand] = field(init=False, repr=False)
    _feasible_segment_sets: Dict[Tuple[DemandId, Period], Set[Segment]] = field(init=False, repr=False)

    def __post_init__(self) -> None:
        if tuple(self.periods) != (0, 1):
            raise ValueError("This implementation is for periods=(0, 1).")
        if self.max_segments <= 0:
            raise ValueError("max_segments must be positive.")
        if self.budget < 0:
            raise ValueError("budget must be nonnegative.")

        object.__setattr__(self, "_demand_by_id", {d.id: d for d in self.demands})
        if len(self._demand_by_id) != len(self.demands):
            raise ValueError("Demand ids must be unique.")

        fs: Dict[Tuple[DemandId, Period], Set[Segment]] = {}
        for d in self.demands:
            for t in self.periods:
                key = (d.id, t)
                if key not in self.feasible_segments:
                    raise ValueError(f"Missing feasible_segments for demand {d.id!r}, period {t}.")
                fs[key] = set(tuple(e) for e in self.feasible_segments[key])
                if t not in d.volume:
                    raise ValueError(f"Demand {d.id!r} has no volume for period {t}.")
        object.__setattr__(self, "_feasible_segment_sets", fs)

        for a in self.arcs:
            if a not in self.capacities:
                raise ValueError(f"Missing capacity for arc {a!r}.")
            if self.capacities[a] <= 0:
                raise ValueError(f"Capacity of arc {a!r} must be positive.")

    @property
    def demand_by_id(self) -> Mapping[DemandId, Demand]:
        return self._demand_by_id

    @property
    def arc_times(self) -> List[ArcTime]:
        return [(a, t) for t in self.periods for a in self.arcs]

    def feasible_real_segments(self, demand_id: DemandId, t: Period) -> Set[Segment]:
        return set(self._feasible_segment_sets[(demand_id, t)])

    def r(self, segment: Segment, arc: Arc, t: Period) -> float:
        return float(self.routing_fraction.get((segment, arc, t), 0.0))

    def segment_costs(
        self,
        demand_id: DemandId,
        aggregate_mu: Mapping[ArcTime, float],
    ) -> Dict[Period, Dict[Segment, float]]:
        """
        Compute theta_d(e,t) = sum_a (-mu(a,t)) * nu(d,t)/c(a) * r(e,a,t).
        """
        d = self.demand_by_id[demand_id]
        theta: Dict[Period, Dict[Segment, float]] = {}
        for t in self.periods:
            vol = float(d.volume[t])
            theta_t: Dict[Segment, float] = {}
            for e in self.feasible_real_segments(demand_id, t):
                cost = 0.0
                for a in self.arcs:
                    mu = float(aggregate_mu.get((a, t), 0.0))
                    if mu != 0.0:
                        cost += (-mu) * vol / float(self.capacities[a]) * self.r(e, a, t)
                if cost < 0.0 and cost > -1e-10:
                    cost = 0.0
                theta_t[e] = cost
            theta[t] = theta_t
        return theta

    @staticmethod
    def distance(p0: Path, p1: Path) -> int:
        return len(set(p0).symmetric_difference(set(p1)))

    def compute_gamma(self, demand_id: DemandId, paths: Mapping[Period, Path]) -> Dict[ArcTime, float]:
        d = self.demand_by_id[demand_id]
        gamma: Dict[ArcTime, float] = {}
        for t in self.periods:
            vol = float(d.volume[t])
            path_segments = set(paths[t])
            for a in self.arcs:
                total_fraction = sum(self.r(e, a, t) for e in path_segments)
                gamma[(a, t)] = vol / float(self.capacities[a]) * total_fraction if total_fraction else 0.0
        return gamma

    def make_column(self, demand_id: DemandId, p0: Iterable[Segment], p1: Iterable[Segment]) -> Column:
        path0: Path = tuple(tuple(e) for e in p0)
        path1: Path = tuple(tuple(e) for e in p1)
        paths = {0: path0, 1: path1}
        beta = float(self.distance(path0, path1))
        gamma = self.compute_gamma(demand_id, paths)
        name = f"col_{safe_name(demand_id)}_{abs(hash((path0, path1))) % 10**12}"
        return Column(demand_id=demand_id, paths=paths, beta=beta, gamma=gamma, name=name)