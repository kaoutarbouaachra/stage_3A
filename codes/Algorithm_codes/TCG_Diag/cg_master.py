"""
cg_master.py
------------
Restricted Master Problem (RMP) for the pair-path Column Generation.
 
Objective:
  Phase 1 (pool generation, rank 1): MLU = max_{a,t} lambda(a, t)
  Phase 2 (lex k >= 2): not implemented here -- see the lex module, which must
  also return the inherited threshold duals eta_j for the stratified pricing.
 
DUAL SIGN CONVENTION (important for the pricing interface)
  For a minimisation model, Gurobi returns Pi <= 0 for a `<=` constraint and a
  free Pi for an `==` constraint. Hence:
     sigma_d (selection, (S))  : free
     mu(a, t) (load,      (L)) : <= 0
     psi(t)   (budget,    (B)) : <= 0
  These are returned RAW. The pricing subproblem is responsible for using the
  effective non-negative quantities  (-mu)  and  rho = -psi  (cf. eq:theta).
"""
 
from __future__ import annotations
 
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
 
import gurobipy as gp
from gurobipy import GRB
 
from core.instance import Instance
from core.ecmp import SplitCoeffs
from .pricing_dp import Path, Trajectory
 
 
# ──────────────────────────────────────────────────────────────────────────
#  TRAJECTORY POOL
# ──────────────────────────────────────────────────────────────────────────
 
@dataclass
class TrajectoryPool:
    """Pool of trajectories (p0, p1) per demand. Grows by append only, so the
    positional index `ti` of a trajectory is stable across CG iterations -- the
    RMP solver relies on this to cache column coefficients."""
    # pool[d] = list of (p0, p1)
    pool: Dict[int, List[Trajectory]] = field(default_factory=dict)
 
    def add(self, d_id: int, traj: Trajectory) -> bool:
        """Add a trajectory if not already present. Returns True if added."""
        if d_id not in self.pool:
            self.pool[d_id] = []
        # Deduplicate
        traj_norm = (tuple(traj[0]), tuple(traj[1]))
        for existing in self.pool[d_id]:
            if (tuple(existing[0]), tuple(existing[1])) == traj_norm:
                return False
        self.pool[d_id].append(traj)
        return True
 
    def get(self, d_id: int) -> List[Trajectory]:
        return self.pool.get(d_id, [])
 
    def total_size(self) -> int:
        return sum(len(v) for v in self.pool.values())
 
 
# ──────────────────────────────────────────────────────────────────────────
#  COLUMN COEFFICIENTS
# ──────────────────────────────────────────────────────────────────────────
 
def trajectory_budget_cost(traj: Trajectory, src: int, tgt: int) -> int:
    """beta_{d,1}(traj) = dist(p0, p1) = |segs(p0) symdiff segs(p1)|.
 
    h = 2 only: there is a single transition (period 0 -> period 1), so the
    budget cost of a trajectory is the single distance dist(p0, p1). For h >= 3
    this must become a per-period dist(p^{t-1}, p^{t}) and the signature must
    take t; the current form would wrongly reuse dist(p0, p1) at every t.
    """
    p0, p1 = traj
    nodes0 = [src] + list(p0) + [tgt]
    nodes1 = [src] + list(p1) + [tgt]
    segs0 = set(zip(nodes0[:-1], nodes0[1:]))
    segs1 = set(zip(nodes1[:-1], nodes1[1:]))
    return len(segs0.symmetric_difference(segs1))
 
 
# ──────────────────────────────────────────────────────────────────────────
#  RMP SOLVER
# ──────────────────────────────────────────────────────────────────────────
 
class RMPSolver:
    """
    Restricted Master Problem solver for the CG.
 
    The LP relaxation (continuous z) is solved to optimality to extract duals;
    the MIP (binary z) is solved to recover a feasible routing scheme.
    """
 
    def __init__(self, inst: Instance, r: Dict[int, SplitCoeffs]) -> None:
        # h = 2 is hard-wired throughout (trajectories are pairs, single budget
        # transition). Fail loudly rather than produce a silently wrong model.
        assert inst.h == 2, "cg_master.py is specialised to h = 2"
 
        self.inst = inst
        self.r = r
        self.demands_info = {
            d_id: (src, tgt, vols)
            for (d_id, src, tgt, vols) in inst.demands
        }
 
        # Build the capacity map ONCE (was rebuilt for every column before).
        self.capa: Dict[int, float] = {
            l['id']: l['capacity'] for l in inst.links.values()
        }
 
        # Index the split coefficients by (u, v) ONCE, per time step:
        #   r_index[t][(u, v)] = list of (arc_id, coeff)
        # This both speeds up Gamma (no full scan per segment) and makes the
        # all-pairs requirement explicit: every segment endpoint pair (i, j)
        # used as a waypoint hop must have an entry here. If r only covers
        # demand source/target pairs, waypoint segments contribute 0 and the
        # load is silently under-counted -- so we check for that case below.
        self.r_index: Dict[int, Dict[Tuple[int, int], List[Tuple[int, float]]]] = {}
        for t, r_t in self.r.items():
            idx: Dict[Tuple[int, int], List[Tuple[int, float]]] = {}
            for (u, v, arc_id), coeff in r_t.items():
                if coeff > 1e-12:
                    idx.setdefault((u, v), []).append((arc_id, coeff))
            self.r_index[t] = idx
 
        # Persistent coefficient caches, keyed by (d_id, ti). Gamma and beta are
        # intrinsic to the trajectory (independent of the duals and of the lex
        # rank k), so they are computed once and reused across all iterations.
        self._gamma_cache: Dict[Tuple[int, int], Dict[Tuple[int, int], float]] = {}
        self._beta_cache: Dict[Tuple[int, int], int] = {}
 
    # ----------------------------------------------------------------------
 
    def _gamma(self, d_id: int, ti: int, traj: Trajectory) -> Dict[Tuple[int, int], float]:
        """Gamma_dat(traj) for all (a, t), cached by (d_id, ti).
 
        Gamma_dat(traj) = (nu(d, t) / c(a)) * sum_{(i,j) in segs} r(i, j, a, t)
        """
        key = (d_id, ti)
        cached = self._gamma_cache.get(key)
        if cached is not None:
            return cached
 
        src, tgt, vols = self.demands_info[d_id]
        contrib: Dict[Tuple[int, int], float] = {}
 
        for t, path in enumerate(traj):          # h = 2: t = 0, 1
            nu_dt = vols[t]
            if nu_dt == 0.0:
                continue
            nodes = [src] + list(path) + [tgt]
            segments = list(zip(nodes[:-1], nodes[1:]))
            idx_t = self.r_index.get(t, {})
 
            for (i, j) in segments:
                arcs = idx_t.get((i, j))
                if arcs is None:
                    # No forwarding info for this hop. For a valid simple hop in
                    # G_t this should not happen if r is all-pairs; flag it so a
                    # missing-coefficient bug cannot pass unnoticed.
                    raise KeyError(
                        f"no split coefficients r({i},{j},.,{t}); "
                        f"is r computed for ALL node pairs in G_t?"
                    )
                for (arc_id, coeff) in arcs:
                    c_a = self.capa[arc_id]      # KeyError if arc_id unknown
                    k = (arc_id, t)
                    contrib[k] = contrib.get(k, 0.0) + (nu_dt / c_a) * coeff
 
        self._gamma_cache[key] = contrib
        return contrib
 
    def _beta(self, d_id: int, ti: int, traj: Trajectory) -> int:
        """beta for the single h = 2 transition, cached by (d_id, ti)."""
        key = (d_id, ti)
        cached = self._beta_cache.get(key)
        if cached is not None:
            return cached
        src, tgt, _ = self.demands_info[d_id]
        b = trajectory_budget_cost(traj, src, tgt)
        self._beta_cache[key] = b
        return b
 
    def _check_pool_covers_all_demands(self, pool: TrajectoryPool) -> None:
        """Every demand must have at least one column, otherwise it silently
        vanishes from the model (no z variable, no selection row) and the LP
        stays feasible while leaving the demand unrouted. The caller must seed
        the pool with the IGP trajectory (shortest, shortest) per demand."""
        for d_id in self.demands_info:
            if not pool.get(d_id):
                raise ValueError(
                    f"demand {d_id} has no column in the pool -- seed it with "
                    f"the IGP trajectory before solving the RMP"
                )
 
    # ----------------------------------------------------------------------
 
    def solve_lp(
        self,
        pool: TrajectoryPool,
        time_limit: float = 60.0,
        verbose: bool = False,
    ) -> Tuple[float, Dict, Dict, Dict, Dict]:
        """
        Solve the LP relaxation of the RMP (objective: minimise MLU).
        Returns (MLU_LP, z_values, sigma_duals, mu_duals, psi_duals).
 
        The LP MUST reach proven optimality: the duals feed the pricing oracle,
        and duals from a non-optimal LP are invalid (unsound pricing, lost
        convergence guarantee). On failure we return +inf and empty dicts.
        """
        self._check_pool_covers_all_demands(pool)
 
        m = gp.Model("RMP_LP")
        if not verbose:
            m.setParam("OutputFlag", 0)
        m.setParam("TimeLimit", time_limit)
 
        # ─── Variables ────────────────────────────────────────────────────
        # z[d, ti] where ti is the positional index in pool[d]
        z = {}
        for d_id in self.demands_info:
            for ti in range(len(pool.get(d_id))):
                z[(d_id, ti)] = m.addVar(
                    lb=0.0, ub=1.0,
                    vtype=GRB.CONTINUOUS,
                    name=f"z_{d_id}_{ti}",
                )
 
        # MLU variable
        mlu = m.addVar(lb=0.0, vtype=GRB.CONTINUOUS, name="MLU")
 
        # ─── (S) selection ────────────────────────────────────────────────
        cons_S = {}
        for d_id in self.demands_info:
            n = len(pool.get(d_id))
            cons_S[d_id] = m.addConstr(
                gp.quicksum(z[(d_id, ti)] for ti in range(n)) == 1,
                name=f"S_{d_id}",
            )
 
        # ─── (L) loads (precompute Gamma per column, cached) ──────────────
        gamma = {
            (d_id, ti): self._gamma(d_id, ti, traj)
            for d_id in self.demands_info
            for ti, traj in enumerate(pool.get(d_id))
        }
 
        cons_L = {}
        for l in self.inst.links.values():
            arc_id = l['id']
            for t in self.inst.T:
                # Skip arcs unavailable at t: pricing routes only inside G_t, so
                # their load is structurally 0 and they need no threshold row.
                if arc_id in self.inst.interv.get(t, set()):
                    continue
                expr = gp.LinExpr()
                for (d_id, ti), g in gamma.items():
                    val = g.get((arc_id, t), 0.0)
                    if val > 1e-12:
                        expr.add(z[(d_id, ti)], val)
                if expr.size() > 0:
                    cons_L[(arc_id, t)] = m.addConstr(
                        expr <= mlu, name=f"L_{arc_id}_{t}"
                    )
 
        # ─── (B) budget ───────────────────────────────────────────────────
        cons_B = {}
        for t in self.inst.T_star:               # h = 2: T_star = {1}
            kappa = self.inst.budget.get(t)
            if kappa is None:
                continue
            expr = gp.LinExpr()
            for d_id in self.demands_info:
                for ti, traj in enumerate(pool.get(d_id)):
                    beta = self._beta(d_id, ti, traj)
                    if beta > 0:
                        expr.add(z[(d_id, ti)], float(beta))
            if expr.size() > 0:
                cons_B[t] = m.addConstr(expr <= kappa, name=f"B_{t}")
 
        # ─── Objective and solve ──────────────────────────────────────────
        m.setObjective(mlu, GRB.MINIMIZE)
        m.optimize()
 
        # Require proven optimality for valid duals.
        if m.Status != GRB.OPTIMAL or m.SolCount == 0:
            return float('inf'), {}, {}, {}, {}
 
        # ─── Solution and duals ───────────────────────────────────────────
        mlu_val = mlu.X
        z_vals = {key: var.X for key, var in z.items()}
 
        # Raw Gurobi duals. Sign convention documented at the top of the file:
        # sigma free, mu <= 0, psi <= 0. The pricing must use (-mu) and (-psi).
        sigma_duals = {d_id: c.Pi for d_id, c in cons_S.items()}
        mu_duals = {key: c.Pi for key, c in cons_L.items()}
        psi_duals = {t: c.Pi for t, c in cons_B.items()}
 
        return mlu_val, z_vals, sigma_duals, mu_duals, psi_duals
 
    # ----------------------------------------------------------------------
 
    def solve_mip(
        self,
        pool: TrajectoryPool,
        time_limit: float = 300.0,
        verbose: bool = False,
    ) -> Tuple[float, Dict[int, Trajectory]]:
        """
        Solve the RMP as a MIP (binary z) to obtain a feasible routing scheme.
        Returns (MLU, solution_per_demand). No duals are extracted here (they
        would not be meaningful from a MIP).
        """
        self._check_pool_covers_all_demands(pool)
 
        m = gp.Model("RMP_MIP")
        if not verbose:
            m.setParam("OutputFlag", 0)
        m.setParam("TimeLimit", time_limit)
 
        z = {}
        for d_id in self.demands_info:
            for ti in range(len(pool.get(d_id))):
                z[(d_id, ti)] = m.addVar(vtype=GRB.BINARY, name=f"z_{d_id}_{ti}")
 
        mlu = m.addVar(lb=0.0, vtype=GRB.CONTINUOUS, name="MLU")
 
        # (S) selection
        for d_id in self.demands_info:
            n = len(pool.get(d_id))
            m.addConstr(
                gp.quicksum(z[(d_id, ti)] for ti in range(n)) == 1,
                name=f"S_{d_id}",
            )
 
        # (L) loads (reuse the cached Gamma)
        gamma = {
            (d_id, ti): self._gamma(d_id, ti, traj)
            for d_id in self.demands_info
            for ti, traj in enumerate(pool.get(d_id))
        }
        for l in self.inst.links.values():
            arc_id = l['id']
            for t in self.inst.T:
                if arc_id in self.inst.interv.get(t, set()):
                    continue
                expr = gp.LinExpr()
                for (d_id, ti), g in gamma.items():
                    val = g.get((arc_id, t), 0.0)
                    if val > 1e-12:
                        expr.add(z[(d_id, ti)], val)
                if expr.size() > 0:
                    m.addConstr(expr <= mlu, name=f"L_{arc_id}_{t}")
 
        # (B) budget
        for t in self.inst.T_star:
            kappa = self.inst.budget.get(t)
            if kappa is None:
                continue
            expr = gp.LinExpr()
            for d_id in self.demands_info:
                for ti, traj in enumerate(pool.get(d_id)):
                    beta = self._beta(d_id, ti, traj)
                    if beta > 0:
                        expr.add(z[(d_id, ti)], float(beta))
            if expr.size() > 0:
                m.addConstr(expr <= kappa, name=f"B_{t}")
 
        m.setObjective(mlu, GRB.MINIMIZE)
        m.optimize()
 
        # A MIP may stop at the time limit with an incumbent; accept any feasible
        # solution but require that one exists before reading values.
        if m.SolCount == 0:
            return float('inf'), {}
 
        mlu_val = mlu.X
        solution: Dict[int, Trajectory] = {}
        for d_id in self.demands_info:
            for ti, traj in enumerate(pool.get(d_id)):
                if z[(d_id, ti)].X > 0.5:
                    solution[d_id] = traj
                    break
        return mlu_val, solution