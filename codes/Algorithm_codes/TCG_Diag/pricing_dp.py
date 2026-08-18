"""
pricing_dp.py  (exact diagonal pricing-off diagonal enumeration (topK))
---------------------------------------------
"""
 
from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
import itertools
import networkx as nx
 
 
# ─────────────────────────────────────────────────────────────────
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
 
 
# ─────────────────────────────────────────────────────────────────
#  PRICING CONTEXT  (precomputed once, reused across all iterations)
# ─────────────────────────────────────────────────────────────────
 
@dataclass
class PricingContext:
    """Static data that does not depend on the duals:
      reach[t][i] = set of nodes reachable from i in G_t (includes i)
      capa[arc_id] = capacity
    A segment (i, j) is a usable hop at period t iff j in reach[t][i]."""
    reach: Dict[int, Dict[int, set]]
    capa: Dict[int, float]
 
 
def _reachability(inst, t: int) -> Dict[int, set]:
    """Transitive reachability in G_t (instance minus arcs down at t)."""
    unavail = inst.interv.get(t, set())
    G = nx.DiGraph()
    G.add_nodes_from(inst.nodes)
    for l in inst.links.values():
        if l['id'] not in unavail:
            G.add_edge(l['from'], l['to'])
    reach: Dict[int, set] = {}
    for i in inst.nodes:
        reach[i] = nx.descendants(G, i) | {i}
    return reach
 
 
def build_pricing_context(inst) -> PricingContext:
    """Build reachability for every period and the capacity map, once."""
    capa = {l['id']: l['capacity'] for l in inst.links.values()}
    reach = {t: _reachability(inst, t) for t in inst.T}
    return PricingContext(reach=reach, capa=capa)
 
 
# ─────────────────────────────────────────────────────────────────
#  SEGMENT COSTS  theta_d(i, j, t)
# ─────────────────────────────────────────────────────────────────
 
def compute_segment_costs(
    d_info, inst, r, mu, t, ctx: Optional[PricingContext] = None,
) -> Dict[Tuple[int, int], float]:
    """theta_d(i,j,t) = sum_a (-mu(a,t)) * (nu(d,t)/c(a)) * r(i,j,a,t) >= 0.
 
    Only pairs (i,j) with at least one loaded arc (mu < 0) appear in the result;
    any other valid segment has theta = 0 (no load pressure)."""
    d_id, src, tgt, vols = d_info
    nu_dt = vols[t]
    if nu_dt == 0.0:
        return {}
 
    r_t = r[t]
    capa = ctx.capa if ctx is not None else {
        l['id']: l['capacity'] for l in inst.links.values()
    }
 
    theta: Dict[Tuple[int, int], float] = {}
    for (u, v, arc_id), coeff in r_t.items():
        if coeff < 1e-12:
            continue
        mu_at = mu.get((arc_id, t), 0.0)
        if mu_at >= -1e-12:          # mu ~ 0 or (numerically) positive -> skip
            continue
        c_a = capa[arc_id]           # KeyError on unknown arc, rather than 1.0
        theta[(u, v)] = theta.get((u, v), 0.0) + (nu_dt / c_a) * (-mu_at) * coeff
    return theta
 
 
def seg_cost(theta: Dict, i: int, j: int) -> float:
    return theta.get((i, j), 0.0)
 
 
def path_cost(theta: Dict, src: int, tgt: int, wps: Path) -> float:
    nodes = [src] + wps + [tgt]
    return sum(seg_cost(theta, nodes[k], nodes[k + 1])
               for k in range(len(nodes) - 1))
 
 
def path_segments(src: int, tgt: int, wps: Path):
    nodes = [src] + wps + [tgt]
    return set(zip(nodes[:-1], nodes[1:]))
 
 
# ─────────────────────────────────────────────────────────────────
#  EXACT DIAGONAL PRICING  (bounded-hop Bellman-Ford)
# ─────────────────────────────────────────────────────────────────
 
def _diagonal_dp(
    src: int, tgt: int, maxseg: int,
    theta0: Dict[Tuple[int, int], float],
    theta1: Dict[Tuple[int, int], float],
    reach0: Dict[int, set],
    reach1: Dict[int, set],
    nodes: List[int],
) -> Tuple[Optional[Path], float]:
    """Min-cost s->t segment path of <= maxseg segments under the combined cost
    theta^comb(i,j) = theta0(i,j) + theta1(i,j) >= 0, where a hop (i,j) is allowed
    iff j is reachable from i in BOTH G_0 and G_1.
 
    Layered DP (Proposition "Diagonal pricing"):
        D[v,0] = 0 if v == s else +inf
        D[j,k+1] = min_i { D[i,k] + theta^comb(i,j) }
    optimal value = min_{1<=k<=maxseg} D[t,k]. Costs are non-negative, so the
    smallest-k minimiser is a simple path; we reconstruct it via parent pointers.
 
    Returns (waypoints, best_combined_cost). waypoints is None if t is not
    reachable on the diagonal within maxseg segments.
    Complexity O(maxseg * |V|^2).
    """
    INF = float('inf')
    # D[k] : node -> best cost reaching it from src with exactly k segments
    D: List[Dict[int, float]] = [dict.fromkeys(nodes, INF) for _ in range(maxseg + 1)]
    parent: List[Dict[int, int]] = [dict() for _ in range(maxseg + 1)]
    D[0][src] = 0.0
 
    for k in range(1, maxseg + 1):
        Dk, Dprev, pk = D[k], D[k - 1], parent[k]
        for i in nodes:
            di = Dprev[i]
            if di == INF:
                continue
            # valid successors: reachable in both periods (a usable segment)
            succ = reach0[i] & reach1[i]
            for j in succ:
                if j == i:
                    continue
                c = theta0.get((i, j), 0.0) + theta1.get((i, j), 0.0)
                nd = di + c
                if nd < Dk[j] - 1e-15:
                    Dk[j] = nd
                    pk[j] = i
 
    # pick the cheapest, breaking ties toward the fewest segments
    best_k, best_cost = None, INF
    for k in range(1, maxseg + 1):
        c = D[k][tgt]
        if c < best_cost - 1e-12:
            best_cost, best_k = c, k
    if best_k is None or best_cost == INF:
        return None, INF
 
    # reconstruct src -> ... -> tgt and strip endpoints to get waypoints
    chain = [tgt]
    v, k = tgt, best_k
    while k > 0 and v != src:
        u = parent[k].get(v)
        if u is None:
            return None, INF        # should not happen for a finite best_cost
        chain.append(u)
        v, k = u, k - 1
    chain.reverse()
    wps = chain[1:-1] if len(chain) >= 2 else []
    return wps, best_cost
 
 
def price_diagonal_exact(
    d_info, inst, r, mu, pi_d,
    ctx: PricingContext,
    theta0: Optional[Dict] = None,
    theta1: Optional[Dict] = None,
) -> Tuple[Optional[PricingResult], float]:
    """Exact diagonal pricing for one demand.
 
    Returns (result, opt_reduced_cost) where opt_reduced_cost is the EXACT
    minimum reduced cost over all diagonal trajectories (even if >= 0, so it can
    feed the lower bound). result is a PricingResult iff that optimum is < 0,
    else None.
    """
    d_id, src, tgt, vols = d_info
    if theta0 is None:
        theta0 = compute_segment_costs(d_info, inst, r, mu, 0, ctx)
    if theta1 is None:
        theta1 = compute_segment_costs(d_info, inst, r, mu, 1, ctx)
 
    nodes = list(inst.nodes)
    wps, best_cost = _diagonal_dp(
        src, tgt, inst.max_seg, theta0, theta1,
        ctx.reach[0], ctx.reach[1], nodes,
    )
    if wps is None:
        return None, float('inf')
 
    rc = -pi_d + best_cost          # dist(p,p) = 0, no budget term
    res = PricingResult(trajectory=(wps, list(wps)),
                        reduced_cost=rc, is_diagonal=True) if rc < 0 else None
    return res, rc
 
 
def best_diagonal_reduced_cost(
    d_info, inst, r, mu, pi_d, ctx: PricingContext,
) -> float:
    """Exact min reduced cost over diagonal trajectories for one demand.
 
    NOTE on the lower bound: with EXACT pricing the Lagrangian bound
    LB = z_RMP_LP + sum_d (min reduced cost)_d is valid only when the per-demand
    minimum is exact over ALL trajectories. This function is exact over the
    DIAGONAL only; the off-diagonal here is heuristic. So sum_d of this value is
    a valid bound only in the diagonal regime (e.g. tight budget / rho large,
    where the optimum is on the diagonal). For an unconditional certificate, add
    the exact product-DP off-diagonal and take the min of the two.
    """
    _, rc = price_diagonal_exact(d_info, inst, r, mu, pi_d, ctx)
    return rc
 
 
# ─────────────────────────────────────────────────────────────────
#  HEURISTIC OFF-DIAGONAL ENUMERATION  (fast column generator, NOT exact)
# ─────────────────────────────────────────────────────────────────
 
def ranked_waypoints(
    src: int, tgt: int, inst,
    theta: Dict[Tuple[int, int], float], t: int,
    ctx: PricingContext, top_k: int = 30,
) -> List[int]:
    """Candidate waypoints reachable both from src and to tgt in G_t, ranked by
    theta(src,w) + theta(w,tgt)."""
    reach_t = ctx.reach[t]
    reach_src = reach_t.get(src, {src})
    # nodes that can reach tgt = {w : tgt in reach_t[w]}
    cands = [w for w in inst.nodes
             if w != src and w != tgt
             and w in reach_src and tgt in reach_t.get(w, ())]
    cands.sort(key=lambda w: seg_cost(theta, src, w) + seg_cost(theta, w, tgt))
    return cands[:top_k]
 
 
def enumerate_paths_for_period(
    src: int, tgt: int, inst,
    theta: Dict[Tuple[int, int], float], t: int,
    ctx: PricingContext, top_k: int = 20,
) -> List[Tuple[Path, float]]:
    """Heuristic enumeration of low-cost SR paths src->tgt at period t, covering
    0..min(maxSeg-1, 4) waypoints over truncated candidate sets. Every emitted
    path has all hops reachable in G_t (so it is a valid path, even if the set is
    not exhaustive)."""
    max_wps = inst.max_seg - 1
    reach_t = ctx.reach[t]
    candidates: List[Tuple[Path, float]] = [([], seg_cost(theta, src, tgt))]
 
    wps = ranked_waypoints(src, tgt, inst, theta, t, ctx, top_k=30)
    if not wps:
        return sorted(candidates, key=lambda x: x[1])[:top_k]
 
    def hop_ok(i, j):
        return j in reach_t.get(i, ())
 
    if max_wps >= 1:
        for w in wps:
            candidates.append(([w], seg_cost(theta, src, w) + seg_cost(theta, w, tgt)))
 
    if max_wps >= 2:
        for w1, w2 in itertools.permutations(wps[:20], 2):
            if not hop_ok(w1, w2):
                continue
            candidates.append(([w1, w2],
                seg_cost(theta, src, w1) + seg_cost(theta, w1, w2)
                + seg_cost(theta, w2, tgt)))
 
    if max_wps >= 3:
        for w1, w2, w3 in itertools.permutations(wps[:10], 3):
            if not (hop_ok(w1, w2) and hop_ok(w2, w3)):
                continue
            candidates.append(([w1, w2, w3],
                seg_cost(theta, src, w1) + seg_cost(theta, w1, w2)
                + seg_cost(theta, w2, w3) + seg_cost(theta, w3, tgt)))
 
    if max_wps >= 4:
        for w1, w2, w3, w4 in itertools.permutations(wps[:6], 4):
            if not (hop_ok(w1, w2) and hop_ok(w2, w3) and hop_ok(w3, w4)):
                continue
            candidates.append(([w1, w2, w3, w4],
                seg_cost(theta, src, w1) + seg_cost(theta, w1, w2)
                + seg_cost(theta, w2, w3) + seg_cost(theta, w3, w4)
                + seg_cost(theta, w4, tgt)))
 
    candidates.sort(key=lambda x: x[1])
    return candidates[:top_k]
 
 
# ─────────────────────────────────────────────────────────────────
#  MAIN PRICING : top-K trajectories (exact diagonal + heuristic off-diag)
# ─────────────────────────────────────────────────────────────────
 
def price_demand_topk(
    d_info, inst, r: Dict,
    mu: Dict[Tuple[int, int], float],
    pi_d: float, psi_1: float,
    ctx: Optional[PricingContext] = None,
    top_k: int = 5, rc_tol: float = -1e-6,
) -> List[PricingResult]:
    """Up to top_k improving trajectories (reduced cost < rc_tol) for one demand.
 
    The single best DIAGONAL column is found exactly; off-diagonal columns come
    from the heuristic enumeration. `psi_1` is the raw budget dual psi(1) <= 0.
    """
    if ctx is None:
        ctx = build_pricing_context(inst)     # caller should build once & reuse
 
    d_id, src, tgt, vols = d_info
    theta0 = compute_segment_costs(d_info, inst, r, mu, 0, ctx)
    theta1 = compute_segment_costs(d_info, inst, r, mu, 1, ctx)
 
    results: List[PricingResult] = []
 
    # ── Diagonal: exact ───────────────────────────────────────────
    diag_res, _ = price_diagonal_exact(d_info, inst, r, mu, pi_d, ctx,
                                       theta0=theta0, theta1=theta1)
    if diag_res is not None and diag_res.reduced_cost < rc_tol:
        results.append(diag_res)
 
    # ── Off-diagonal: heuristic enumeration ───────────────────────
    paths0 = enumerate_paths_for_period(src, tgt, inst, theta0, 0, ctx, top_k=20)
    paths1 = enumerate_paths_for_period(src, tgt, inst, theta1, 1, ctx, top_k=20)
    rho = -psi_1                              # >= 0, per-shared-segment discount
    for wps0, cost0 in paths0[:10]:
        segs0 = path_segments(src, tgt, wps0)
        for wps1, cost1 in paths1[:10]:
            if wps0 == wps1:
                continue                       # diagonal handled above (exactly)
            segs1 = path_segments(src, tgt, wps1)
            dist = len(segs0.symmetric_difference(segs1))
            rc = -pi_d + cost0 + cost1 + rho * dist   # rho*dist = -psi_1*dist >= 0
            if rc < rc_tol:
                results.append(PricingResult(
                    trajectory=(wps0, list(wps1)),
                    reduced_cost=rc, is_diagonal=False))
 
    results.sort(key=lambda x: x.reduced_cost)
    return results[:top_k]
 
 
# ─────────────────────────────────────────────────────────────────
#  Backward-compatible single-column 
# ─────────────────────────────────────────────────────────────────
 
def price_demand(d_info, inst, r, mu, pi_d, psi_1,
                 ctx: Optional[PricingContext] = None,
                 rc_tol=-1e-6) -> Optional[PricingResult]:
    results = price_demand_topk(d_info, inst, r, mu, pi_d, psi_1,
                                ctx=ctx, top_k=1, rc_tol=rc_tol)
    return results[0] if results else None