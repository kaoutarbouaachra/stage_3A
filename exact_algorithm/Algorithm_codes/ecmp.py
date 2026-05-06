"""
ecmp.py
-------
Compute ECMP split coefficients r(u, v, arc_id, t) using NetworkX.

Three NetworkX tools used:
  1. nx.single_source_dijkstra_path_length  ->  forward distances d(u, *)
  2. G.reverse() + Dijkstra                 ->  backward distances d(*, v)
  3. nx.topological_sort(FG)                ->  ECMP flow propagation on DAG
"""

from collections import defaultdict
from typing import Dict, List, Set, Tuple

import networkx as nx

from instance import Instance

# Type alias: r[(u, v, arc_id)] = fraction in [0, 1]
SplitCoeffs = Dict[Tuple[int, int, int], float]


# ============================================================
# Public API
# ============================================================

def compute_all_splits(inst: Instance) -> Dict[int, SplitCoeffs]:
    """
    Compute r[t] for every timestep t in inst.T.

    Returns a dict: t -> SplitCoeffs
    """
    r = {}
    for t in inst.T:
        G = _build_graph(inst, unavailable=inst.interv.get(t, set()))
        r[t] = _compute_splits(G, inst.nodes)
        print(f"    t={t}: {len(r[t])} non-zero ECMP coefficients")
    return r


def compute_loads(
    waypoints: Dict[Tuple[int, int], List[int]],
    inst: Instance,
    r: Dict[int, SplitCoeffs],
) -> Dict[Tuple[int, int], float]:
    """
    Compute arc loads lambda(a, t) from a waypoints solution.

    Parameters
    ----------
    waypoints : {(demand_id, t): [w1, w2, ...]}
        Empty list = pure IGP routing (no waypoint).

    Returns
    -------
    loads : {(arc_id, t): load_value}
    """
    loads: Dict[Tuple[int, int], float] = defaultdict(float)

    for (d_id, src, tgt, vols) in inst.demands:
        for t in inst.T:
            volume = vols[t]
            if volume == 0:
                continue

            # Build segment list from waypoints
            wps  = waypoints.get((d_id, t), [])
            path = [src] + wps + [tgt]
            segments = [(path[i], path[i + 1]) for i in range(len(path) - 1)]

            # Accumulate contribution of each segment on each arc
            for (u, v) in segments:
                for (uu, vv, arc_id), frac in r[t].items():
                    if uu == u and vv == v:
                        cap = inst.links[arc_id]['capacity']
                        loads[(arc_id, t)] += frac * volume / cap

    return dict(loads)


def sorted_load_vector(
    loads: Dict[Tuple[int, int], float],
    inst: Instance,
) -> List[float]:
    """
    Return all arc loads sorted in decreasing order (lambda' vector).
    Includes zeros for arcs with no traffic.
    """
    all_values = [
        loads.get((a, t), 0.0)
        for a in inst.arc_ids
        for t in inst.T
    ]
    return sorted(all_values, reverse=True)


# ============================================================
# Internal helpers
# ============================================================

def _build_graph(inst: Instance, unavailable: Set[int]) -> nx.DiGraph:
    """Build a NetworkX DiGraph from the instance, excluding unavailable arcs."""
    G = nx.DiGraph()
    for nid in inst.nodes:
        G.add_node(nid, name=inst.node_name[nid])
    for l in inst.links.values():
        if l['id'] in unavailable:
            continue
        G.add_edge(
            l['from'], l['to'],
            id=l['id'],
            metric=l['metric'],
            capacity=l['capacity'],
        )
    return G


def _compute_splits(G: nx.DiGraph, nodes: List[int]) -> SplitCoeffs:
    """
    Compute r[(u, v, arc_id)] for all (u, v) pairs in the given graph G.

    Algorithm:
      1. Forward Dijkstra from each u  ->  dist_from[u][v] = d(u, v)
      2. Backward Dijkstra from each v  ->  dist_to[v][b]  = d(b, v)
         (= forward Dijkstra on G.reverse())
      3. For each (u, v): build FG(u,v) = arcs on any shortest path u->v
         Condition: d(u,a) + metric(a,b) + d(b,v) == d(u,v)
      4. Propagate flow=1.0 from u through FG(u,v) in topological order.
         At each node: split flow equally among outgoing arcs (ECMP rule).
    """
    # Step 1: forward distances
    dist_from: Dict[int, Dict[int, float]] = {}
    for u in nodes:
        try:
            dist_from[u] = dict(
                nx.single_source_dijkstra_path_length(G, u, weight='metric')
            )
        except Exception:
            dist_from[u] = {}

    # Step 2: backward distances via reversed graph
    Gr = G.reverse(copy=False)
    dist_to: Dict[int, Dict[int, float]] = {}
    for v in nodes:
        try:
            dist_to[v] = dict(
                nx.single_source_dijkstra_path_length(Gr, v, weight='metric')
            )
        except Exception:
            dist_to[v] = {}

    r: SplitCoeffs = {}

    for u in nodes:
        for v in nodes:
            if u == v:
                continue
            if v not in dist_from.get(u, {}):
                continue  # no path u -> v
            d_uv = dist_from[u][v]

            # Step 3: build FG(u, v)
            fg = nx.DiGraph()
            for (a, b, data) in G.edges(data=True):
                da = dist_from[u].get(a, float('inf'))
                db = dist_to[v].get(b, float('inf'))
                if abs(da + data['metric'] + db - d_uv) < 1e-9:
                    fg.add_edge(a, b, id=data['id'])

            if fg.number_of_nodes() == 0:
                continue

            # Step 4: ECMP flow propagation on DAG
            flow = defaultdict(float)
            flow[u] = 1.0
            arc_flow: Dict[int, float] = defaultdict(float)

            try:
                topo_order = list(nx.topological_sort(fg))
            except nx.NetworkXUnfeasible:
                continue  # should not happen (FG is a Directed Acyclic Graph)

            for node in topo_order:
                if flow[node] < 1e-12:
                    continue
                out_edges = list(fg.out_edges(node, data=True))
                if not out_edges:
                    continue
                share = flow[node] / len(out_edges)  # equal split (ECMP)
                for (_, nb, edata) in out_edges:
                    arc_flow[edata['id']] += share
                    flow[nb]              += share

            for arc_id, frac in arc_flow.items():
                if frac > 1e-12:
                    r[(u, v, arc_id)] = frac

    return r
