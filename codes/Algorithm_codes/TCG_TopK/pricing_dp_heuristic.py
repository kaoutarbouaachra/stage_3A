"""
Function:pricing_dp_heuristic.py 
Role: Pricing for lexicographic CG with diagonal/off-diagonal with heuristic path enumeration.
---------------------------------------
  1. Explores paths with 3, 4, 5 waypoints (not only 0-2)
  2. Better candidate node selection (dual-based)
  3. Adds several columns per demand (top-K)
  4. More aggressive off-diagonal pricing
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
import itertools
import networkx as nx 
Path = List[int]
Trajectory = Tuple[Path, Path]


@dataclass
class PricingResult:
    trajectory: Trajectory
    reduced_cost: float
    is_diagonal: bool

    def __repr__(self):
        p0, p1 = self.trajectory
        tag = "diag" if self.is_diagonal else "off"
        return f"PricingResult(rc={self.reduced_cost:.6f},{tag},p0={p0},p1={p1})"


######################################
###  Segment costs
######################################

def compute_segment_costs(
    d_info, inst, r, mu, t
) -> Dict[Tuple[int, int], float]:
    """θ_d(i,j,t) = Σ_a (-μ(a,t)) × ν(d,t)/c(a) × r(i,j,a,t) ≥ 0"""
    d_id, src, tgt, vols = d_info
    nu_dt = vols[t]
    if nu_dt == 0:
        return {}

    r_t = r[t]
    capa = {l['id']: l['capacity'] for l in inst.links.values()}

    theta: Dict[Tuple[int, int], float] = {}
    for (u, v, arc_id), coeff in r_t.items():
        if coeff < 1e-12:
            continue
        mu_at = mu.get((arc_id, t), 0.0)
        if mu_at >= -1e-12:
            continue
        c_a = capa.get(arc_id, 1.0)
        contrib = (nu_dt / c_a) * (-mu_at) * coeff
        theta[(u, v)] = theta.get((u, v), 0.0) + contrib
    return theta


def seg_cost(theta: Dict, i: int, j: int) -> float:
    return theta.get((i, j), 0.0)


def path_cost(theta: Dict, src: int, tgt: int, wps: Path) -> float:
    nodes = [src] + wps + [tgt]
    return sum(seg_cost(theta, nodes[k], nodes[k+1])
               for k in range(len(nodes)-1))


def path_segments(src: int, tgt: int, wps: Path):
    nodes = [src] + wps + [tgt]
    return set(zip(nodes[:-1], nodes[1:]))


###################################################
###  Candidate nodes ranked by usefulness (dual-based)
###################################################

def ranked_waypoints(
    src: int, tgt: int,
    inst,
    theta: Dict[Tuple[int, int], float],
    t: int,
    top_k: int = 30,
) -> List[int]:
    """
    Return the best waypoint nodes ranked by total cost
    θ(src,w) + θ(w,tgt) in increasing order.
    Takes the top_k with enough diversity.
    """
    unavail = inst.interv.get(t, set())
    G = nx.DiGraph()
    for nid in inst.nodes:
        G.add_node(nid)
    for l in inst.links.values():
        if l['id'] not in unavail:
            G.add_edge(l['from'], l['to'], weight=l['metric'])

    if src not in G.nodes or tgt not in G.nodes:
        return []

    try:
        reach_src = set(nx.single_source_shortest_path_length(G, src).keys())
        reach_tgt = set(nx.single_source_shortest_path_length(
            G.reverse(copy=False), tgt).keys())
    except Exception:
        return []

    cands = sorted(
        (reach_src & reach_tgt) - {src, tgt},
        key=lambda w: seg_cost(theta, src, w) + seg_cost(theta, w, tgt)
    )
    return cands[:top_k]


#####################################################
###  Path enumeration for one period
#####################################################

def enumerate_paths_for_period(
    src: int, tgt: int,
    inst,
    theta: Dict[Tuple[int, int], float],
    t: int,
    top_k: int = 20,
) -> List[Tuple[Path, float]]:
    """
    Enumerate the best SR paths from src→tgt at period t.
    Covers 0, 1, 2, 3 waypoints (up to maxSeg-1).
    Returns a list of (waypoints, θ_cost) sorted by increasing cost.
    """
    max_wps = inst.max_seg - 1
    candidates = []

    # 0 waypoint
    c0 = seg_cost(theta, src, tgt)
    candidates.append(([], c0))

    wps = ranked_waypoints(src, tgt, inst, theta, t, top_k=30)
    if not wps:
        return sorted(candidates, key=lambda x: x[1])[:top_k]

    # 1 waypoint 
    if max_wps >= 1:
        for w in wps:
            cost = seg_cost(theta, src, w) + seg_cost(theta, w, tgt)
            candidates.append(([w], cost))

    # 2 waypoints 
    if max_wps >= 2:
        # Limit to top-20 to avoid O(n²)
        wps2 = wps[:20]
        for w1, w2 in itertools.permutations(wps2, 2):
            cost = (seg_cost(theta, src, w1)
                  + seg_cost(theta, w1, w2)
                  + seg_cost(theta, w2, tgt))
            candidates.append(([w1, w2], cost))

    # 3 waypoints 
    if max_wps >= 3:
        # Top-10 only to avoid blow-up
        wps3 = wps[:10]
        for w1, w2, w3 in itertools.permutations(wps3, 3):
            cost = (seg_cost(theta, src, w1)
                  + seg_cost(theta, w1, w2)
                  + seg_cost(theta, w2, w3)
                  + seg_cost(theta, w3, tgt))
            candidates.append(([w1, w2, w3], cost))

    # 4 waypoints (only if maxSeg >= 5) 
    if max_wps >= 4:
        wps4 = wps[:6]
        for w1, w2, w3, w4 in itertools.permutations(wps4, 4):
            cost = (seg_cost(theta, src, w1)
                  + seg_cost(theta, w1, w2)
                  + seg_cost(theta, w2, w3)
                  + seg_cost(theta, w3, w4)
                  + seg_cost(theta, w4, tgt))
            candidates.append(([w1, w2, w3, w4], cost))

    candidates.sort(key=lambda x: x[1])
    return candidates[:top_k]


##############################################
### MAIN PRICING: returns top-K trajectories
##############################################

def price_demand_topk(
    d_info,
    inst,
    r: Dict,
    mu: Dict[Tuple[int, int], float],
    pi_d: float,
    nu_1: float,
    top_k: int = 5,
    rc_tol: float = -1e-6,
) -> List[PricingResult]:
    """
    Full pricing for one demand.
    Returns the top_k best trajectories with rc < rc_tol.
    """
    d_id, src, tgt, vols = d_info
    theta0 = compute_segment_costs(d_info, inst, r, mu, 0)
    theta1 = compute_segment_costs(d_info, inst, r, mu, 1)
    paths0 = enumerate_paths_for_period(src, tgt, inst, theta0, 0, top_k=20)
    paths1 = enumerate_paths_for_period(src, tgt, inst, theta1, 1, top_k=20)
    results = []
    
    # Diagonal: p0 == p1 
    for wps, cost0 in paths0:
        cost1 = path_cost(theta1, src, tgt, wps)
        # dist(p, p) = 0 → no budget term
        rc = -pi_d + cost0 + cost1
        if rc < rc_tol:
            results.append(PricingResult(
                trajectory=(wps, list(wps)),
                reduced_cost=rc,
                is_diagonal=True,
            ))

    # Off-diagonal: p0 ≠ p1 
    # Combine the best ones from each period
    for wps0, cost0 in paths0[:10]:
        segs0 = path_segments(src, tgt, wps0)
        for wps1, cost1 in paths1[:10]:
            if wps0 == wps1:
                continue
            segs1 = path_segments(src, tgt, wps1)
            dist = len(segs0.symmetric_difference(segs1))
            # nu_1 ≤ 0, -nu_1 ≥ 0 → penalty on changes
            rc = -pi_d + cost0 + cost1 - nu_1 * dist
            if rc < rc_tol:
                results.append(PricingResult(
                    trajectory=(wps0, list(wps1)),
                    reduced_cost=rc,
                    is_diagonal=False,
                ))

    # Sort by increasing rc and return top_k
    results.sort(key=lambda x: x.reduced_cost)
    return results[:top_k]


###############################################
###  Backward compat: old price_demand interface
##############################################

def price_demand(d_info, inst, r, mu, pi_d, nu_1,
                 rc_tol=-1e-6) -> Optional[PricingResult]:
    results = price_demand_topk(d_info, inst, r, mu, pi_d, nu_1,
                                top_k=1, rc_tol=rc_tol)
    return results[0] if results else None