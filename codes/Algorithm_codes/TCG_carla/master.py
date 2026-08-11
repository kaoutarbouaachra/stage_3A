#!/usr/bin/env python3
"""
TCG_carla.master
================

Pair-path restricted master LP with the CARLA (cumulative Ogryczak-Tamir) rank
encoding hardcoded, root-node column generation with exact pricing, and a
pool-restricted integer solve.

    solve_lp(rank_values=None)             solves the MLU LP (rank 1).
    solve_lp(rank_values=[z1,...,z_{k-1}]) solves the rank-k CARLA LP.
    solve_integer_master(...)              solves the integer master over the
                                           currently generated pool.

CARLA rank rows
---------------
The columns (trajectories), the selection rows (S), the load-assembly expression
lambda(a,t) = sum Gamma*xi, the budget row (B) and the whole pricing interface
are identical to STELA. What is baked in here is the cumulative rank block: for
each active rank j <= k a free threshold q_j and nonnegative excesses d_{j,a,t}
with

    lambda(a,t) <= q_j + d_{j,a,t},

so that  Theta_j := j*q_j + sum_{a,t} d_{j,a,t}  equals the sum of the j largest
loads at the O-T optimum. Rank k minimises Theta_k; the inherited ranks j<k are
frozen one-sidedly by  Theta_j <= Theta_j*  with  Theta_j* = sum_{r<=j} z_r*
(cumulated internally from the certified sorted loads passed in `rank_values`).
No binary appears in the rank encoding; only xi toggles between the LP
relaxation and the integer master.

The duals fed to the pricer are read exactly as in STELA:
aggregate_mu(a,t) = sum over ALL load-bounding rows touching (a,t) of their Pi
(<= 0 in a minimisation). For CARLA that sum is sum_{j<=k} of the O-T row duals,
i.e. the stratified subgradient of the sum of the k largest loads -- so the
pricer (imported unchanged from TCG_Exact.pricing) never learns which encoding
produced the duals.

The column data model and the exact pricer are reused as-is from TCG_Exact.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Set, Tuple

# Shared, unmodified data model + exact pricer (single source of truth).
from TCG_Exact.data import (
    GRB,
    Arc,
    ArcTime,
    Column,
    DemandId,
    MasterInfeasibleError,
    Period,
    PricingInfeasibleError,
    TASRData,
    TASRError,
    TrajectoryKey,
    _require_gurobi,
    gp,
    safe_name,
)
from TCG_Exact.pricing import ExactPairPathPricer


@dataclass
class MasterResult:
    model: Any
    objective_value: float
    z_value: float
    sigma: Dict[DemandId, float]
    aggregate_mu: Dict[ArcTime, float]
    psi: float
    xi_values: Dict[Tuple[DemandId, TrajectoryKey], float]
    y_values: Dict[Any, float] = field(default_factory=dict)


@dataclass
class ColumnGenerationLogEntry:
    iteration: int
    objective_value: float
    added_columns: int
    min_reduced_cost: float
    pricing_values: Dict[DemandId, float]


@dataclass
class ColumnGenerationResult:
    objective_value: float
    z_value: float
    iterations: int
    columns: Dict[DemandId, List[Column]]
    xi_values: Dict[Tuple[DemandId, TrajectoryKey], float]
    log: List[ColumnGenerationLogEntry]
    master_result: MasterResult


# Small one-sided slack on the cumulative freeze rows. The certified levels
# z_r* are recovered by evaluation of an actual integer solution, so Theta_j*
# is consistent with a feasible point; the slack only guards against solver
# feasibility tolerance turning "<= Theta_j*" marginally infeasible. It loosens
# the freeze in the SAFE direction (never excludes the leximin optimum), in the
# spirit of the one-sided freezing rule.
_CARLA_FREEZE_EPS = 1e-6


class PairPathColumnGeneration:
    """Root-node column generation with corrected exact pricing (CARLA master)."""

    def __init__(self, data: TASRData,
                 initial_columns: Optional[Mapping[DemandId, Iterable[Column]]] = None,
                 output: bool = False,
                 pricing_output: bool = False,
                 pricing_tol: float = 1e-7,
                 max_iterations: int = 10000,
                 master_params: Optional[Mapping[str, Any]] = None,
                 pricing_params: Optional[Mapping[str, Any]] = None) -> None:
        _require_gurobi()
        self.data = data
        self.output = output
        self.pricing_output = pricing_output
        self.pricing_tol = pricing_tol
        self.max_iterations = max_iterations
        self.master_params = dict(master_params or {})
        self.pricing_params = dict(pricing_params or {})
        self.columns: Dict[DemandId, List[Column]] = {d.id: [] for d in data.demands}
        self.column_keys: Dict[DemandId, Set[TrajectoryKey]] = {d.id: set() for d in data.demands}

        if initial_columns is None:
            self._build_basic_initial_pool()
        else:
            for d_id, cols in initial_columns.items():
                for col in cols:
                    self.add_column(col)

    def add_column(self, col: Column) -> bool:
        if col.demand_id not in self.columns:
            raise ValueError(f"Unknown demand id {col.demand_id!r}.")
        key = col.key()
        if key in self.column_keys[col.demand_id]:
            return False
        self.columns[col.demand_id].append(col)
        self.column_keys[col.demand_id].add(key)
        return True

    def solve_lp(self, rank_values: Optional[Sequence[float]] = None) -> ColumnGenerationResult:
        rank_values = list(rank_values or [])
        log: List[ColumnGenerationLogEntry] = []
        pricer = ExactPairPathPricer(self.data, output=self.pricing_output,
                                     gurobi_params=self.pricing_params)
        last_master: Optional[MasterResult] = None

        for iteration in range(1, self.max_iterations + 1):
            master = self._solve_master(relax_integrality=True, rank_values=rank_values)
            last_master = master
            rho = max(0.0, -float(master.psi))
            added = 0
            min_rc = math.inf
            pricing_values: Dict[DemandId, float] = {}

            for d in self.data.demands:
                theta = self.data.segment_costs(d.id, master.aggregate_mu)
                forbidden: Set[TrajectoryKey] = set()
                while True:
                    pricing = pricer.solve(demand_id=d.id, theta=theta, rho=rho,
                                           sigma=master.sigma[d.id],
                                           forbidden_trajectories=forbidden,
                                           use_cutoff=False,
                                           pricing_tol=self.pricing_tol)
                    pricing_values[d.id] = pricing.value
                    min_rc = min(min_rc, pricing.reduced_cost)
                    if pricing.reduced_cost >= -self.pricing_tol:
                        break
                    if pricing.p0 is None or pricing.p1 is None:
                        break
                    col = self.data.make_column(d.id, pricing.p0, pricing.p1)
                    if self.add_column(col):
                        added += 1
                        break
                    forbidden.add(col.key())
                    if len(forbidden) > len(self.columns[d.id]) + 5:
                        break

            log.append(ColumnGenerationLogEntry(iteration, master.objective_value,
                                                added, min_rc, pricing_values))
            if self.output:
                print(f"[CG] iter={iteration:04d} obj={master.objective_value:.10g} "
                      f"added={added} min_rc={min_rc:.3e}")
            if added == 0:
                return ColumnGenerationResult(master.objective_value, master.z_value,
                                              iteration, self.columns, master.xi_values,
                                              log, master)

        raise TASRError(f"Column generation did not converge in {self.max_iterations} iterations.")

    def solve_integer_master(self, rank_values: Optional[Sequence[float]] = None,
                             time_limit: Optional[float] = None,
                             mip_gap: Optional[float] = None) -> MasterResult:
        extra: Dict[str, Any] = {}
        if time_limit is not None:
            extra["TimeLimit"] = time_limit
        if mip_gap is not None:
            extra["MIPGap"] = mip_gap
        old = self.master_params.copy()
        self.master_params.update(extra)
        try:
            return self._solve_master(relax_integrality=False, rank_values=list(rank_values or []))
        finally:
            self.master_params = old

    # ------------------------------------------------------------------
    # Master build: common structure, then the CARLA cumulative rank block.
    # ------------------------------------------------------------------
    def _solve_master(self, relax_integrality: bool, rank_values: Sequence[float]) -> MasterResult:
        data = self.data
        rank_values = list(rank_values)

        m, xi, selection_constr, budget_constr, load_expr = \
            self._build_common(relax_integrality)

        z_handle, load_constrs_by_at, extra = self._add_carla_rank_rows(
            m, load_expr, rank_values, relax_integrality)

        m.ModelSense = GRB.MINIMIZE
        m.optimize()

        if m.Status in (GRB.INFEASIBLE, GRB.INF_OR_UNBD):
            try:
                m.computeIIS()
                iis_rows = [c.ConstrName for c in m.getConstrs() if c.IISConstr]
            except Exception:
                iis_rows = []
            raise MasterInfeasibleError(
                "Restricted master is infeasible. Provide a feasible initial pool or check the budget. "
                "IIS rows: " + ", ".join(iis_rows[:20])
            )
        if m.Status != GRB.OPTIMAL:
            raise TASRError(f"Master did not solve to optimality. Gurobi status={m.Status}.")

        # Duals -- the pricing pressure on a load coordinate (a,t) is the sum of
        # the Pi of every load-bounding row that touches it (the stratified O-T
        # subgradient). Read exactly as in STELA.
        if relax_integrality:
            sigma = {d.id: float(selection_constr[d.id].Pi) for d in data.demands}
            psi = float(budget_constr.Pi)
            aggregate_mu = {at: sum(float(con.Pi) for con in load_constrs_by_at[at])
                            for at in data.arc_times}
        else:
            sigma = {d.id: float("nan") for d in data.demands}
            psi = float("nan")
            aggregate_mu = {at: float("nan") for at in data.arc_times}

        xi_values: Dict[Tuple[DemandId, TrajectoryKey], float] = {}
        for d in data.demands:
            for idx, col in enumerate(self.columns[d.id]):
                val = float(xi[(d.id, idx)].X)
                if abs(val) > 1e-10:
                    xi_values[(d.id, col.key())] = val

        # CARLA has no binary rank encoding, so no y-values to report.
        y_values: Dict[Any, float] = {}

        z_value = float(z_handle.X)
        return MasterResult(m, float(m.ObjVal), z_value, sigma, aggregate_mu,
                            psi, xi_values, y_values)

    # ------------------------------------------------------------------
    # Common (encoding-independent) structure: xi, (S), (B), and the load
    # expressions lambda(a,t) = sum Gamma*xi.  Identical to the STELA master.
    # ------------------------------------------------------------------
    def _build_common(self, relax_integrality: bool):
        data = self.data
        m = gp.Model("pair_path_restricted_master_carla")
        m.Params.OutputFlag = 1 if self.output else 0
        for key, val in self.master_params.items():
            setattr(m.Params, key, val)

        xi: Dict[Tuple[DemandId, int], gp.Var] = {}
        for d in data.demands:
            if not self.columns[d.id]:
                raise MasterInfeasibleError(f"No column in pool for demand {d.id!r}.")
            for idx, _ in enumerate(self.columns[d.id]):
                vtype = GRB.CONTINUOUS if relax_integrality else GRB.BINARY
                xi[(d.id, idx)] = m.addVar(lb=0.0, ub=1.0, obj=0.0, vtype=vtype,
                                           name=f"xi[{safe_name(d.id)},{idx}]")
        m.update()

        selection_constr: Dict[DemandId, gp.Constr] = {}
        for d in data.demands:
            selection_constr[d.id] = m.addConstr(
                gp.quicksum(xi[(d.id, idx)] for idx in range(len(self.columns[d.id]))) == 1,
                name=f"select[{safe_name(d.id)}]"
            )

        budget_expr = gp.LinExpr()
        for d in data.demands:
            for idx, col in enumerate(self.columns[d.id]):
                budget_expr += float(col.beta) * xi[(d.id, idx)]
        budget_constr = m.addConstr(budget_expr <= float(data.budget), name="budget")

        load_expr: Dict[ArcTime, gp.LinExpr] = {}
        for at in data.arc_times:
            expr = gp.LinExpr()
            for d in data.demands:
                for idx, col in enumerate(self.columns[d.id]):
                    coeff = float(col.gamma.get(at, 0.0))
                    if coeff:
                        expr += coeff * xi[(d.id, idx)]
            load_expr[at] = expr

        return m, xi, selection_constr, budget_constr, load_expr

    # ------------------------------------------------------------------
    # CARLA rank rows -- cumulative Ogryczak-Tamir block (hardcoded).
    # ------------------------------------------------------------------
    def _add_carla_rank_rows(self, m, load_expr, rank_values, relax_integrality):
        data = self.data
        ats = list(data.arc_times)
        rank_values = list(rank_values)
        k = len(rank_values) + 1

        # Cumulative certified levels: Theta_j* = sum_{r<=j} z_r*.
        theta_star: List[float] = []
        acc = 0.0
        for zr in rank_values:
            acc += float(zr)
            theta_star.append(acc)          # theta_star[j-1] = Theta_j*, j=1..k-1

        # Free thresholds q_j and nonnegative excesses d_{j,a,t}, for every
        # active rank j = 1..k. Always continuous: no binary in the rank
        # encoding -- only xi (built in _build_common) toggles integrality.
        q: Dict[int, gp.Var] = {}
        d: Dict[Tuple[int, ArcTime], gp.Var] = {}
        for j in range(1, k + 1):
            q[j] = m.addVar(lb=-GRB.INFINITY, vtype=GRB.CONTINUOUS, name=f"q[{j}]")
            for at in ats:
                d[(j, at)] = m.addVar(lb=0.0, vtype=GRB.CONTINUOUS,
                                      name=f"d[{j},{safe_name(at)}]")
        m.update()

        # Ogryczak-Tamir rows: lambda(a,t) <= q_j + d_{j,a,t}, for all active j.
        # Their duals (<= 0) are the stratified subgradient the pricer reads via
        # aggregate_mu(a,t) = sum_j Pi_{j,a,t}.
        load_constrs_by_at: Dict[ArcTime, List[Any]] = {at: [] for at in ats}
        for j in range(1, k + 1):
            for at in ats:
                con = m.addConstr(load_expr[at] <= q[j] + d[(j, at)],
                                  name=f"ot[{j},{safe_name(at)}]")
                load_constrs_by_at[at].append(con)

        # One-sided cumulative freeze of the inherited ranks:
        #   Theta_j = j*q_j + sum_{a,t} d_{j,a,t} <= Theta_j*   (j < k).
        for j in range(1, k):
            theta_j = j * q[j] + gp.quicksum(d[(j, at)] for at in ats)
            m.addConstr(theta_j <= theta_star[j - 1] + _CARLA_FREEZE_EPS,
                        name=f"freeze_theta[{j}]")

        # Objective: minimise Theta_k = k*q_k + sum d_{k,a,t}.
        theta_k = k * q[k] + gp.quicksum(d[(k, at)] for at in ats)
        m.setObjective(theta_k, GRB.MINIMIZE)

        # z_handle exposes a scalar for MasterResult.z_value; q_k is the k-th
        # largest load at the O-T optimum (a VaR-type threshold). It is NOT used
        # by the driver, which recovers z_k* from the integer solution's sorted
        # load vector.
        return q[k], load_constrs_by_at, {"q": q, "d": d, "theta_star": theta_star}

    def _build_basic_initial_pool(self) -> None:
        pricer = ExactPairPathPricer(self.data, output=self.pricing_output,
                                     gurobi_params=self.pricing_params)
        for d in self.data.demands:
            theta = {
                0: {e: 0.0 for e in self.data.feasible_real_segments(d.id, 0)},
                1: {e: 0.0 for e in self.data.feasible_real_segments(d.id, 1)},
            }
            pricing = pricer.solve(d.id, theta=theta, rho=1.0, sigma=0.0)
            if pricing.p0 is None or pricing.p1 is None:
                raise PricingInfeasibleError(f"Could not create initial pair-path for {d.id!r}.")
            self.add_column(self.data.make_column(d.id, pricing.p0, pricing.p1))
