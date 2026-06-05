
"""
Function:cg_master0.py

Role: Restricted Master Problem (RMP) for the pair-path Column Generation.

Variables:
  z[d, traj_id] ∈ {0, 1}  for each trajectory (p0, p1) in the pool

Constraints:
  (S) Σ_traj z[d, traj] = 1            for each demand d
  (L) λ(a, t) = Σ_d,traj Γ_dat(traj) × z[d, traj]   for each (a, t)
  (B) Σ_d,traj β_dt(traj) × z[d, traj] ≤ κ(t)       for each t ∈ T*

Objective:
  Phase 1 (pool generation): MLU = max_at λ(a, t)
  Phase 2 (lex k≥2): lex-k objective with inherited constraints
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import gurobipy as gp
from gurobipy import GRB

from core.instance import Instance
from core.ecmp_from_scratch import SplitCoeffs
from .pricing0_dp import Path, Trajectory


# ──────────────────────────────────────────────────────────────────────────
#  TRAJECTORY POOL
# ──────────────────────────────────────────────────────────────────────────

@dataclass
class TrajectoryPool:
    """Pool of trajectories (p0, p1) per demand."""
    # pool[d] = list of (p0, p1)
    pool: Dict[int, List[Trajectory]] = field(default_factory=dict)

    def add(self, d_id: int, traj: Trajectory) -> bool:
        """Add a trajectory if it is not already present."""
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
#  COLUMN COEFFICIENT COMPUTATION
# ──────────────────────────────────────────────────────────────────────────

def trajectory_load_contribution(
    traj: Trajectory,
    src: int, tgt: int,
    vols: List[float],
    inst: Instance,
    r: Dict[int, SplitCoeffs],
) -> Dict[Tuple[int, int], float]:
    """
    Compute Γ_dat(traj) for all (a, t).
    Γ_dat(traj) = (ν(d, t) / c(a)) × Σ_segments r(i, j, a, t)
    """
    capa = {l['id']: l['capacity'] for l in inst.links.values()}
    contrib: Dict[Tuple[int, int], float] = {}

    for t, path in enumerate(traj):
        nu_dt = vols[t]
        if nu_dt == 0:
            continue
        nodes = [src] + list(path) + [tgt]
        segments = list(zip(nodes[:-1], nodes[1:]))
        r_t = r[t]

        for (i, j) in segments:
            for (u, v, arc_id), coeff in r_t.items():
                if u == i and v == j and coeff > 1e-12:
                    c_a = capa.get(arc_id, 1.0)
                    val = (nu_dt / c_a) * coeff
                    key = (arc_id, t)
                    contrib[key] = contrib.get(key, 0.0) + val
    return contrib


def trajectory_budget_cost(traj: Trajectory, src: int, tgt: int) -> int:
    """Compute β_d,1(traj) = dist(p0, p1) = |segs(p0) Δ segs(p1)|."""
    p0, p1 = traj
    nodes0 = [src] + list(p0) + [tgt]
    nodes1 = [src] + list(p1) + [tgt]
    segs0 = set(zip(nodes0[:-1], nodes0[1:]))
    segs1 = set(zip(nodes1[:-1], nodes1[1:]))
    return len(segs0.symmetric_difference(segs1))


# ──────────────────────────────────────────────────────────────────────────
#  RMP CONSTRUCTION AND SOLVING
# ──────────────────────────────────────────────────────────────────────────

class RMPSolver:
    """
    Restricted Master Problem solver for the CG.
    Uses an LP RELAXATION (continuous z) to extract the duals.
    """

    def __init__(self, inst: Instance, r: Dict[int, SplitCoeffs]) -> None:
        self.inst = inst
        self.r = r
        self.demands_info = {
            d_id: (src, tgt, vols)
            for (d_id, src, tgt, vols) in inst.demands
        }

    def solve_lp(
        self,
        pool: TrajectoryPool,
        time_limit: float = 60.0,
        verbose: bool = False,
    ) -> Tuple[float, Dict, Dict, Dict, Dict]:
        """
        Solve the LP relaxation of the RMP (objective: minimize MLU).
        Returns (MLU_LP, z_values, pi_duals, mu_duals, nu_duals).
        """
        m = gp.Model("RMP_LP")
        if not verbose:
            m.setParam("OutputFlag", 0)
        m.setParam("TimeLimit", time_limit)

        # ─── Variables ────────────────────────────────────────────────────
        # z[d, t_idx] where t_idx is the index in pool[d]
        z = {}
        for d_id in self.demands_info:
            trajs = pool.get(d_id)
            for ti, traj in enumerate(trajs):
                z[(d_id, ti)] = m.addVar(
                    lb=0.0, ub=1.0,
                    vtype=GRB.CONTINUOUS,
                    name=f"z_{d_id}_{ti}",
                )

        # MLU variable
        mlu = m.addVar(lb=0.0, vtype=GRB.CONTINUOUS, name="MLU")

        # ─── Constraints (S): selection ───────────────────────────────────
        cons_S = {}
        for d_id in self.demands_info:
            trajs = pool.get(d_id)
            if not trajs:
                continue
            cons_S[d_id] = m.addConstr(
                gp.quicksum(z[(d_id, ti)] for ti in range(len(trajs))) == 1,
                name=f"S_{d_id}",
            )

        # ─── Constraints (L): loads ───────────────────────────────────────
        # Pre-compute Γ_dat for each column
        gamma_cache: Dict[Tuple[int, int], Dict[Tuple[int, int], float]] = {}
        for d_id, (src, tgt, vols) in self.demands_info.items():
            trajs = pool.get(d_id)
            for ti, traj in enumerate(trajs):
                gamma_cache[(d_id, ti)] = trajectory_load_contribution(
                    traj, src, tgt, vols, self.inst, self.r
                )

        # For each (arc_id, t), build the MLU constraint
        cons_L = {}
        for l in self.inst.links.values():
            arc_id = l['id']
            for t in self.inst.T:
                # Skip unavailable arcs
                if arc_id in self.inst.interv.get(t, set()):
                    continue
                # Σ Γ × z ≤ MLU
                expr = gp.LinExpr()
                for (d_id, ti), gamma in gamma_cache.items():
                    val = gamma.get((arc_id, t), 0.0)
                    if val > 1e-12:
                        expr.add(z[(d_id, ti)], val)
                if expr.size() > 0:
                    cons_L[(arc_id, t)] = m.addConstr(
                        expr <= mlu, name=f"L_{arc_id}_{t}"
                    )

        # ─── Constraints (B): budget ──────────────────────────────────────
        cons_B = {}
        for t in self.inst.T_star:
            kappa = self.inst.budget.get(t)
            if kappa is None:
                continue
            expr = gp.LinExpr()
            for d_id, (src, tgt, vols) in self.demands_info.items():
                trajs = pool.get(d_id)
                for ti, traj in enumerate(trajs):
                    beta = trajectory_budget_cost(traj, src, tgt)
                    if beta > 0:
                        expr.add(z[(d_id, ti)], float(beta))
            if expr.size() > 0:
                cons_B[t] = m.addConstr(expr <= kappa, name=f"B_{t}")

        # ─── Objective ────────────────────────────────────────────────────
        m.setObjective(mlu, GRB.MINIMIZE)

        # ─── Solve ────────────────────────────────────────────────────────
        m.optimize()

        if m.Status not in (GRB.OPTIMAL, GRB.SUBOPTIMAL, GRB.TIME_LIMIT):
            return float('inf'), {}, {}, {}, {}

        # ─── Extract solution and duals ──────────────────────────────────
        mlu_val = mlu.X
        z_vals = {key: var.X for key, var in z.items()}

        pi_duals = {d_id: c.Pi for d_id, c in cons_S.items()}
        mu_duals = {key: c.Pi for key, c in cons_L.items()}
        nu_duals = {t: c.Pi for t, c in cons_B.items()}

        return mlu_val, z_vals, pi_duals, mu_duals, nu_duals

    def solve_mip(
        self,
        pool: TrajectoryPool,
        time_limit: float = 300.0,
        verbose: bool = False,
    ) -> Tuple[float, Dict[int, Trajectory]]:
        """
        Solve the RMP as a MIP (binary z) to obtain a feasible solution.
        Returns (MLU, solution_per_demand).
        """
        m = gp.Model("RMP_MIP")
        if not verbose:
            m.setParam("OutputFlag", 0)
        m.setParam("TimeLimit", time_limit)

        z = {}
        for d_id in self.demands_info:
            trajs = pool.get(d_id)
            for ti, traj in enumerate(trajs):
                z[(d_id, ti)] = m.addVar(
                    vtype=GRB.BINARY, name=f"z_{d_id}_{ti}"
                )

        mlu = m.addVar(lb=0.0, vtype=GRB.CONTINUOUS, name="MLU")

        # Selection
        for d_id in self.demands_info:
            trajs = pool.get(d_id)
            if not trajs:
                continue
            m.addConstr(
                gp.quicksum(z[(d_id, ti)] for ti in range(len(trajs))) == 1,
                name=f"S_{d_id}",
            )

        # Loads
        gamma_cache = {}
        for d_id, (src, tgt, vols) in self.demands_info.items():
            for ti, traj in enumerate(pool.get(d_id)):
                gamma_cache[(d_id, ti)] = trajectory_load_contribution(
                    traj, src, tgt, vols, self.inst, self.r
                )

        for l in self.inst.links.values():
            arc_id = l['id']
            for t in self.inst.T:
                if arc_id in self.inst.interv.get(t, set()):
                    continue
                expr = gp.LinExpr()
                for (d_id, ti), gamma in gamma_cache.items():
                    val = gamma.get((arc_id, t), 0.0)
                    if val > 1e-12:
                        expr.add(z[(d_id, ti)], val)
                if expr.size() > 0:
                    m.addConstr(expr <= mlu, name=f"L_{arc_id}_{t}")

        # Budget
        for t in self.inst.T_star:
            kappa = self.inst.budget.get(t)
            if kappa is None:
                continue
            expr = gp.LinExpr()
            for d_id, (src, tgt, vols) in self.demands_info.items():
                for ti, traj in enumerate(pool.get(d_id)):
                    beta = trajectory_budget_cost(traj, src, tgt)
                    if beta > 0:
                        expr.add(z[(d_id, ti)], float(beta))
            if expr.size() > 0:
                m.addConstr(expr <= kappa, name=f"B_{t}")

        m.setObjective(mlu, GRB.MINIMIZE)
        m.optimize()

        if m.Status not in (GRB.OPTIMAL, GRB.SUBOPTIMAL, GRB.TIME_LIMIT):
            return float('inf'), {}

        mlu_val = mlu.X
        solution: Dict[int, Trajectory] = {}
        for d_id in self.demands_info:
            for ti, traj in enumerate(pool.get(d_id)):
                if z[(d_id, ti)].X > 0.5:
                    solution[d_id] = traj
                    break
        return mlu_val, solution