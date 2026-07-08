"""
Function: pricing_dp_nt.py 
Role:pricing with nt4python k-hop Bellman-Ford on the diagonal

-------------------------

Important modelling point:
The nt4python Bellman-Ford is used only for DIAGONAL pricing.
It is not run on the physical topology.  For each demand and each dual vector,
we build an auxiliary segment graph:
    node = router
    auxiliary arc (i, j) exists iff j is reachable from i in BOTH periods 0 and 1
    weight(i, j) = theta0(i, j) + theta1(i, j)
    hop limit = inst.max_seg

This is exactly the same diagonal pricing problem as the previous Python DP.
The off-diagonal part remains heuristic, with the same function names and same
return types as pricing_dp.py, so the rest of the code can stay unchanged.

If nt4python is not installed or fails, this file automatically falls back to the
pure-Python diagonal DP.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
import itertools
import json
import os
import tempfile
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


@dataclass
class PricingContext:
    """Static data that does not depend on the duals.

    reach[t][i] = set of nodes reachable from i in G_t, including i.
    capa[arc_id] = capacity.
    """
    reach: Dict[int, Dict[int, set]]
    capa: Dict[int, float]


# ──────────────────────────────────────────────────────────────────────────────
# Optional nt4python import
# ──────────────────────────────────────────────────────────────────────────────

try:
    from nt4python import *  # type: ignore  # noqa: F401,F403
    HAS_NT4PYTHON = True
except Exception:
    HAS_NT4PYTHON = False
_NT_INITIALIZED = False


def _ensure_nt_init() -> bool:
    """Initialise nt4python once if the binding exposes nt_init().
    Some nt4python examples do not call nt_init explicitly, so this function is
    defensive: if nt_init does not exist, we simply continue.
    """
    global _NT_INITIALIZED

    if not HAS_NT4PYTHON:
        return False

    if _NT_INITIALIZED:
        return True

    init_fun = globals().get("nt_init")
    if init_fun is None:
        _NT_INITIALIZED = True
        return True

    # Different builds may expose nt_init with different signatures.
    for args in [(), (0, None)]:
        try:
            out = init_fun(*args)
            if out is False:
                continue
            _NT_INITIALIZED = True
            return True
        except TypeError:
            continue
        except Exception:
            return False

    return False


def _safe_free(function_name: str, obj) -> None:
    """Call an nt4python free function if available, ignoring cleanup errors."""
    if obj is None:
        return
    fun = globals().get(function_name)
    if fun is None:
        return
    try:
        fun(obj)
    except Exception:
        pass


# ──────────────────────────────────────────────────────────────────────────────
# Reachability and segment costs
# ──────────────────────────────────────────────────────────────────────────────

def _reachability(inst, t: int) -> Dict[int, set]:
    """Transitive reachability in G_t, i.e. topology without arcs down at t."""
    unavail = inst.interv.get(t, set())
    G = nx.DiGraph()
    G.add_nodes_from(inst.nodes)
    for l in inst.links.values():
        if l["id"] not in unavail:
            G.add_edge(l["from"], l["to"])

    reach: Dict[int, set] = {}
    for i in inst.nodes:
        reach[i] = nx.descendants(G, i) | {i}
    return reach


def build_pricing_context(inst) -> PricingContext:
    """Build reachability for every period and the capacity map, once."""
    capa = {l["id"]: l["capacity"] for l in inst.links.values()}
    reach = {t: _reachability(inst, t) for t in inst.T}
    return PricingContext(reach=reach, capa=capa)


def compute_segment_costs(
    d_info,
    inst,
    r,
    mu,
    t,
    ctx: Optional[PricingContext] = None,
) -> Dict[Tuple[int, int], float]:
    """Compute theta_d(i,j,t).

    theta_d(i,j,t) = sum_a (-mu(a,t)) * (nu(d,t)/c(a)) * r(i,j,a,t).

    The dual mu is passed raw from Gurobi, so mu(a,t) <= 0.  We use (-mu) as a
    non-negative pressure.  Segments with zero pressure are omitted and treated
    as cost 0 by seg_cost().
    """
    _d_id, _src, _tgt, vols = d_info
    nu_dt = vols[t]
    if nu_dt == 0.0:
        return {}

    r_t = r[t]
    capa = ctx.capa if ctx is not None else {
        l["id"]: l["capacity"] for l in inst.links.values()
    }

    theta: Dict[Tuple[int, int], float] = {}
    for (u, v, arc_id), coeff in r_t.items():
        if coeff < 1e-12:
            continue

        mu_at = mu.get((arc_id, t), 0.0)
        if mu_at >= -1e-12:
            continue

        c_a = capa[arc_id]
        theta[(u, v)] = theta.get((u, v), 0.0) + (nu_dt / c_a) * (-mu_at) * coeff

    return theta


def seg_cost(theta: Dict[Tuple[int, int], float], i: int, j: int) -> float:
    return theta.get((i, j), 0.0)


def path_cost(theta: Dict[Tuple[int, int], float], src: int, tgt: int, wps: Path) -> float:
    nodes = [src] + list(wps) + [tgt]
    return sum(seg_cost(theta, nodes[k], nodes[k + 1]) for k in range(len(nodes) - 1))


def path_segments(src: int, tgt: int, wps: Path):
    nodes = [src] + list(wps) + [tgt]
    return set(zip(nodes[:-1], nodes[1:]))


# ──────────────────────────────────────────────────────────────────────────────
# Pure-Python diagonal fallback
# ──────────────────────────────────────────────────────────────────────────────

def _remove_cycles_from_chain(chain: List[int]) -> List[int]:
    """Remove repeated-node loops from a chain.

    Segment costs are non-negative, so removing a loop cannot increase the cost.
    This is mostly useful when many segment weights are zero.
    """
    out: List[int] = []
    pos: Dict[int, int] = {}

    for node in chain:
        if node in pos:
            cut = pos[node]
            for old in out[cut + 1:]:
                pos.pop(old, None)
            out = out[:cut + 1]
        else:
            pos[node] = len(out)
            out.append(node)

    return out


def _diagonal_dp(
    src: int,
    tgt: int,
    maxseg: int,
    theta0: Dict[Tuple[int, int], float],
    theta1: Dict[Tuple[int, int], float],
    reach0: Dict[int, set],
    reach1: Dict[int, set],
    nodes: List[int],
) -> Tuple[Optional[Path], float]:
    """Exact diagonal pricing with a pure-Python bounded-hop DP."""
    INF = float("inf")
    D: List[Dict[int, float]] = [dict.fromkeys(nodes, INF) for _ in range(maxseg + 1)]
    parent: List[Dict[int, int]] = [dict() for _ in range(maxseg + 1)]
    D[0][src] = 0.0

    for k in range(1, maxseg + 1):
        Dk, Dprev, pk = D[k], D[k - 1], parent[k]
        for i in nodes:
            di = Dprev[i]
            if di == INF:
                continue

            succ = reach0.get(i, set()) & reach1.get(i, set())
            for j in succ:
                if j == i:
                    continue
                c = theta0.get((i, j), 0.0) + theta1.get((i, j), 0.0)
                nd = di + c
                if nd < Dk[j] - 1e-15:
                    Dk[j] = nd
                    pk[j] = i

    best_k, best_cost = None, INF
    for k in range(1, maxseg + 1):
        c = D[k][tgt]
        if c < best_cost - 1e-12:
            best_cost, best_k = c, k

    if best_k is None or best_cost == INF:
        return None, INF

    chain = [tgt]
    v, k = tgt, best_k
    while k > 0 and v != src:
        u = parent[k].get(v)
        if u is None:
            return None, INF
        chain.append(u)
        v, k = u, k - 1

    chain.reverse()
    chain = _remove_cycles_from_chain(chain)
    return chain[1:-1], best_cost


# ──────────────────────────────────────────────────────────────────────────────
# nt4python diagonal Bellman-Ford
# ──────────────────────────────────────────────────────────────────────────────

def _write_aux_graph_json(
    filename: str,
    nodes: List[int],
    theta0: Dict[Tuple[int, int], float],
    theta1: Dict[Tuple[int, int], float],
    ctx: PricingContext,
) -> None:
    """Write the auxiliary segment graph in a JSON format readable by nt_rnx."""
    links = []
    arc_key = 0

    for i in nodes:
        succ = ctx.reach[0].get(i, set()) & ctx.reach[1].get(i, set())
        for j in succ:
            if i == j:
                continue
            links.append({
                "from": int(i),
                "to": int(j),
                "key": int(arc_key),
                "weight": float(theta0.get((i, j), 0.0) + theta1.get((i, j), 0.0)),
            })
            arc_key += 1

    graph_json = {
        "directed": True,
        "multigraph": True,
        "graph": {},
        "nodes": [{"id": int(n)} for n in nodes],
        "links": links,
    }

    with open(filename, "w", encoding="utf-8") as f:
        json.dump(graph_json, f)


def _diagonal_nt_bellman_ford(
    src: int,
    tgt: int,
    maxseg: int,
    theta0: Dict[Tuple[int, int], float],
    theta1: Dict[Tuple[int, int], float],
    ctx: PricingContext,
    nodes: List[int],
) -> Tuple[Optional[Path], float]:
    """Exact diagonal pricing using nt4python's k-hop Bellman-Ford.

    This function uses only the nt4python calls shown in the example from the
    tutor: nt_rnx_read_file, nt_rnx_extract_arc_map_double, and
    nt_bellman_ford_run_limited_with_paths.
    """
    if not _ensure_nt_init():
        return _diagonal_dp(src, tgt, maxseg, theta0, theta1,
                            ctx.reach[0], ctx.reach[1], nodes)

    tmp_path = None
    g = None
    rnx = None
    weight_map = None
    bf = None

    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            suffix="_pricing_diag_aux.json",
            delete=False,
            encoding="utf-8",
        ) as tmp:
            tmp_path = tmp.name

        _write_aux_graph_json(tmp_path, nodes, theta0, theta1, ctx)

        g = nt_digraph_create()
        rnx = nt_rnx_create()
        nt_rnx_set_source_attr(rnx, "from")
        nt_rnx_set_target_attr(rnx, "to")

        if not nt_rnx_read_file(rnx, tmp_path, g):
            return _diagonal_dp(src, tgt, maxseg, theta0, theta1,
                                ctx.reach[0], ctx.reach[1], nodes)

        weight_map = nt_arcmap_double_create(g)
        nt_rnx_extract_arc_map_double(rnx, g, "weight", weight_map, -1.0)

        s_h = nt_digraph_node_from_id(int(src))
        t_h = nt_digraph_node_from_id(int(tgt))

        bf = nt_bellman_ford_create(g, weight_map)
        nt_bellman_ford_run_limited_with_paths(bf, s_h, int(maxseg))

        if not nt_bellman_ford_reached(bf, t_h):
            return None, float("inf")

        best_cost = float(nt_bellman_ford_dist(bf, t_h))
        arc_count = int(nt_bellman_ford_limited_path_arc_num(bf, t_h))

        chain: List[int] = []
        for idx in range(arc_count):
            arc = nt_bellman_ford_limited_path_arc(bf, t_h, idx)
            u = int(nt_digraph_node_id(nt_digraph_arc_source(g, arc)))
            v = int(nt_digraph_node_id(nt_digraph_arc_target(g, arc)))
            if idx == 0:
                chain.append(u)
            chain.append(v)

        if not chain:
            return None, float("inf")

        if chain[0] != src or chain[-1] != tgt:
            # Defensive fallback in case a binding version returns arcs in another order.
            return _diagonal_dp(src, tgt, maxseg, theta0, theta1,
                                ctx.reach[0], ctx.reach[1], nodes)

        chain = _remove_cycles_from_chain(chain)
        return chain[1:-1], best_cost

    except Exception:
        # Keep the algorithm usable even if nt4python is absent, misconfigured,
        # or the local binding differs from the expected one.
        return _diagonal_dp(src, tgt, maxseg, theta0, theta1,
                            ctx.reach[0], ctx.reach[1], nodes)

    finally:
        _safe_free("nt_bellman_ford_free", bf)
        _safe_free("nt_arcmap_double_free", weight_map)
        _safe_free("nt_rnx_free", rnx)
        _safe_free("nt_digraph_free", g)
        if tmp_path is not None:
            try:
                os.remove(tmp_path)
            except OSError:
                pass


# ──────────────────────────────────────────────────────────────────────────────
# Public diagonal pricing API
# ──────────────────────────────────────────────────────────────────────────────

def price_diagonal_exact(
    d_info,
    inst,
    r,
    mu,
    pi_d,
    ctx: PricingContext,
    theta0: Optional[Dict] = None,
    theta1: Optional[Dict] = None,
) -> Tuple[Optional[PricingResult], float]:
    """Exact diagonal pricing for one demand.

    Returns (result, opt_reduced_cost).  result is returned only when the best
    diagonal reduced cost is negative.
    """
    _d_id, src, tgt, _vols = d_info

    if theta0 is None:
        theta0 = compute_segment_costs(d_info, inst, r, mu, 0, ctx)
    if theta1 is None:
        theta1 = compute_segment_costs(d_info, inst, r, mu, 1, ctx)

    nodes = list(inst.nodes)

    if HAS_NT4PYTHON:
        wps, best_cost = _diagonal_nt_bellman_ford(
            src, tgt, inst.max_seg, theta0, theta1, ctx, nodes,
        )
    else:
        wps, best_cost = _diagonal_dp(
            src, tgt, inst.max_seg, theta0, theta1,
            ctx.reach[0], ctx.reach[1], nodes,
        )

    if wps is None:
        return None, float("inf")

    rc = -pi_d + best_cost
    res = PricingResult(
        trajectory=(wps, list(wps)),
        reduced_cost=rc,
        is_diagonal=True,
    ) if rc < 0 else None

    return res, rc


def best_diagonal_reduced_cost(d_info, inst, r, mu, pi_d, ctx: PricingContext) -> float:
    """Exact minimum reduced cost over diagonal trajectories for one demand."""
    _res, rc = price_diagonal_exact(d_info, inst, r, mu, pi_d, ctx)
    return rc


# ──────────────────────────────────────────────────────────────────────────────
# Off-diagonal heuristic: same as pricing_dp.py
# ──────────────────────────────────────────────────────────────────────────────

def ranked_waypoints(
    src: int,
    tgt: int,
    inst,
    theta: Dict[Tuple[int, int], float],
    t: int,
    ctx: PricingContext,
    top_k: int = 30,
) -> List[int]:
    """Candidate waypoints reachable from src and able to reach tgt at period t."""
    reach_t = ctx.reach[t]
    reach_src = reach_t.get(src, {src})

    cands = [
        w for w in inst.nodes
        if w != src
        and w != tgt
        and w in reach_src
        and tgt in reach_t.get(w, set())
    ]
    cands.sort(key=lambda w: seg_cost(theta, src, w) + seg_cost(theta, w, tgt))
    return cands[:top_k]


def enumerate_paths_for_period(
    src: int,
    tgt: int,
    inst,
    theta: Dict[Tuple[int, int], float],
    t: int,
    ctx: PricingContext,
    top_k: int = 20,
) -> List[Tuple[Path, float]]:
    """Heuristic enumeration of low-cost SR paths for one period."""
    max_wps = inst.max_seg - 1
    reach_t = ctx.reach[t]
    candidates: List[Tuple[Path, float]] = [([], seg_cost(theta, src, tgt))]

    wps = ranked_waypoints(src, tgt, inst, theta, t, ctx, top_k=30)
    if not wps:
        return sorted(candidates, key=lambda x: x[1])[:top_k]

    def hop_ok(i, j):
        return j in reach_t.get(i, set())

    if max_wps >= 1:
        for w in wps:
            candidates.append(([w], seg_cost(theta, src, w) + seg_cost(theta, w, tgt)))

    if max_wps >= 2:
        for w1, w2 in itertools.permutations(wps[:20], 2):
            if not hop_ok(w1, w2):
                continue
            candidates.append((
                [w1, w2],
                seg_cost(theta, src, w1)
                + seg_cost(theta, w1, w2)
                + seg_cost(theta, w2, tgt),
            ))

    if max_wps >= 3:
        for w1, w2, w3 in itertools.permutations(wps[:10], 3):
            if not (hop_ok(w1, w2) and hop_ok(w2, w3)):
                continue
            candidates.append((
                [w1, w2, w3],
                seg_cost(theta, src, w1)
                + seg_cost(theta, w1, w2)
                + seg_cost(theta, w2, w3)
                + seg_cost(theta, w3, tgt),
            ))

    if max_wps >= 4:
        for w1, w2, w3, w4 in itertools.permutations(wps[:6], 4):
            if not (hop_ok(w1, w2) and hop_ok(w2, w3) and hop_ok(w3, w4)):
                continue
            candidates.append((
                [w1, w2, w3, w4],
                seg_cost(theta, src, w1)
                + seg_cost(theta, w1, w2)
                + seg_cost(theta, w2, w3)
                + seg_cost(theta, w3, w4)
                + seg_cost(theta, w4, tgt),
            ))

    candidates.sort(key=lambda x: x[1])
    return candidates[:top_k]


# ──────────────────────────────────────────────────────────────────────────────
# Main public pricing functions: same names/signatures as pricing_dp.py
# ──────────────────────────────────────────────────────────────────────────────

def price_demand_topk(
    d_info,
    inst,
    r: Dict,
    mu: Dict[Tuple[int, int], float],
    pi_d: float,
    psi_1: float,
    ctx: Optional[PricingContext] = None,
    top_k: int = 5,
    rc_tol: float = -1e-6,
) -> List[PricingResult]:
    """Return up to top_k improving trajectories for one demand.

    Diagonal: exact k-hop Bellman-Ford through nt4python when available.
    Off-diagonal: same heuristic enumeration as the original pricing_dp.py.
    """
    if ctx is None:
        ctx = build_pricing_context(inst)

    _d_id, src, tgt, _vols = d_info
    theta0 = compute_segment_costs(d_info, inst, r, mu, 0, ctx)
    theta1 = compute_segment_costs(d_info, inst, r, mu, 1, ctx)

    results: List[PricingResult] = []

    # Exact diagonal pricing.
    diag_res, _diag_rc = price_diagonal_exact(
        d_info, inst, r, mu, pi_d, ctx,
        theta0=theta0,
        theta1=theta1,
    )
    if diag_res is not None and diag_res.reduced_cost < rc_tol:
        results.append(diag_res)

    # Heuristic off-diagonal pricing.
    paths0 = enumerate_paths_for_period(src, tgt, inst, theta0, 0, ctx, top_k=20)
    paths1 = enumerate_paths_for_period(src, tgt, inst, theta1, 1, ctx, top_k=20)
    rho = -psi_1

    for wps0, cost0 in paths0[:10]:
        segs0 = path_segments(src, tgt, wps0)
        for wps1, cost1 in paths1[:10]:
            if wps0 == wps1:
                continue

            segs1 = path_segments(src, tgt, wps1)
            dist = len(segs0.symmetric_difference(segs1))
            rc = -pi_d + cost0 + cost1 + rho * dist

            if rc < rc_tol:
                results.append(PricingResult(
                    trajectory=(wps0, list(wps1)),
                    reduced_cost=rc,
                    is_diagonal=False,
                ))

    results.sort(key=lambda x: x.reduced_cost)
    return results[:top_k]


def price_demand(
    d_info,
    inst,
    r,
    mu,
    pi_d,
    psi_1,
    ctx: Optional[PricingContext] = None,
    rc_tol=-1e-6,
) -> Optional[PricingResult]:
    """Backward-compatible single-column interface."""
    results = price_demand_topk(
        d_info, inst, r, mu, pi_d, psi_1,
        ctx=ctx,
        top_k=1,
        rc_tol=rc_tol,
    )
    return results[0] if results else None
