"""
Function: ecmp.py
Role: Complete replacement for ecmp_from_scratch.py.

Goal
----
Compute ECMP split coefficients:
    r[t][(u, v, arc_id)] = fraction

using nt4python instead of the manual NetworkX/from-scratch computation.

Public functions kept with the same names:
    - compute_all_splits(inst)
    - compute_loads(waypoints, inst, r)
    - sorted_load_vector(loads, inst)
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from typing import Any, Dict, List, Set, Tuple

from nt4python import *


# Type alias: r[(u, v, arc_id)] = fraction in [0, 1]
SplitCoeffs = Dict[Tuple[int, int, int], float]

_EPS = 1e-12
_NT_INITIALIZED = False


@dataclass
class _NtGraphData:
    """Container for the nt4python graph and its useful mappings."""

    g: Any
    metrics: Any
    node_of_id: Dict[int, Any]
    arcs: List[Tuple[int, Any]]


def _ensure_nt_initialized() -> None:
    """Initialize ntlib only once."""
    global _NT_INITIALIZED

    if not _NT_INITIALIZED:
        ok = nt_init(0, None)
        if ok is False:
            raise RuntimeError("nt_init(0, None) failed")
        _NT_INITIALIZED = True


def compute_all_splits(inst) -> Dict[int, SplitCoeffs]:
    """
    Compute ECMP split coefficients for every timestep in inst.T.

    Returns
    -------
    dict
        r[t][(u, v, arc_id)] = ECMP fraction from u to v using arc_id.
    """
    r: Dict[int, SplitCoeffs] = {}

    for t in inst.T:
        unavailable = inst.interv.get(t, set())
        ntg = _build_graph(inst, unavailable=unavailable)

        try:
            r[t] = _compute_splits(ntg, inst.nodes)
        finally:
            _free_graph(ntg)

        print(f"    t={t}: {len(r[t])} non-zero ECMP coefficients")

    return r


def compute_loads(
    waypoints: Dict[Tuple[int, int], List[int]],
    inst,
    r: Dict[int, SplitCoeffs],
) -> Dict[Tuple[int, int], float]:
    """
    Compute arc loads lambda(a, t) from a waypoint solution.

    Parameters
    ----------
    waypoints : dict
        Dictionary of the form {(demand_id, t): [w1, w2, ...]}.
        An empty list means pure IGP routing, with no waypoint.

    inst : Instance-like object
        Network and demand instance.

    r : dict
        ECMP split coefficients returned by compute_all_splits.

    Returns
    -------
    dict
        loads[(arc_id, t)] = load_value.
    """
    loads: Dict[Tuple[int, int], float] = defaultdict(float)

    for (d_id, src, tgt, vols) in inst.demands:
        for t in inst.T:
            volume = vols[t]
            if volume == 0:
                continue

            # Build the segment list induced by the selected waypoints.
            wps = waypoints.get((d_id, t), [])
            path = [src] + wps + [tgt]
            segments = [(path[i], path[i + 1]) for i in range(len(path) - 1)]

            # Accumulate the contribution of each segment on each physical arc.
            for (u, v) in segments:
                for (uu, vv, arc_id), frac in r[t].items():
                    if uu == u and vv == v:
                        cap = inst.links[arc_id]["capacity"]
                        loads[(arc_id, t)] += frac * volume / cap

    return dict(loads)


def sorted_load_vector(
    loads: Dict[Tuple[int, int], float],
    inst,
) -> List[float]:
    """
    Return all arc loads sorted in decreasing order.

    Zero values are included for arcs with no traffic.
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


def _build_graph(inst, unavailable: Set[int]) -> _NtGraphData:
    """
    Build a directed nt4python graph from the instance.

    Arcs whose IDs are listed in unavailable are excluded.
    """
    _ensure_nt_initialized()

    g = nt_digraph_create()

    active_links = [
        link for link in inst.links.values()
        if link["id"] not in unavailable
    ]

    nt_digraph_reserve(g, len(inst.nodes), len(active_links))

    # Create nt4python nodes and keep the mapping from original node ID to handle.
    node_of_id: Dict[int, Any] = {}
    for nid in inst.nodes:
        node_of_id[nid] = nt_digraph_add_node(g)

    # Create nt4python arcs and keep the mapping from original arc ID to handle.
    arcs: List[Tuple[int, Any]] = []
    arc_metrics: List[Tuple[Any, float]] = []

    for link in active_links:
        u = node_of_id[link["from"]]
        v = node_of_id[link["to"]]

        a = nt_digraph_add_arc(g, u, v)

        original_arc_id = link["id"]
        metric = float(link["metric"])

        arcs.append((original_arc_id, a))
        arc_metrics.append((a, metric))

    # Store IGP metrics in a double arc map.
    metrics = nt_arcmap_double_create_ex(g, 0.0)

    for a, metric in arc_metrics:
        nt_arcmap_double_set(metrics, a, metric)

    return _NtGraphData(
        g=g,
        metrics=metrics,
        node_of_id=node_of_id,
        arcs=arcs,
    )


def _free_graph(ntg: _NtGraphData) -> None:
    """Free nt4python objects created by _build_graph."""
    nt_arcmap_double_free(ntg.metrics)
    nt_digraph_free(ntg.g)


def _compute_splits(ntg: _NtGraphData, nodes: List[int]) -> SplitCoeffs:
    """
    Compute r[(u, v, arc_id)] for all ordered node pairs (u, v).

    This replaces the from-scratch computation with:

        nt_ecmp_run(ecmp, dijk, s, t, 1.0, node_flow, arc_flow)

    Since the routed flow is 1.0, each value in arc_flow is directly the ECMP
    fraction carried by that arc.
    """
    r: SplitCoeffs = {}

    dijk = nt_dijkstra_create(ntg.g, ntg.metrics)
    ecmp = nt_ecmp_create()

    node_flow = nt_nodemap_double_create_ex(ntg.g, 0.0)
    arc_flow = nt_arcmap_double_create_ex(ntg.g, 0.0)

    try:
        for u in nodes:
            s = ntg.node_of_id[u]

            # Run Dijkstra once per source node.
            nt_dijkstra_run(dijk, s)

            for v in nodes:
                if u == v:
                    continue

                t = ntg.node_of_id[v]

                if not nt_dijkstra_reached(dijk, t):
                    continue

                # Reset temporary flow maps before each source-target computation.
                nt_nodemap_double_zeros(node_flow)
                nt_arcmap_double_zeros(arc_flow)

                # Compute ECMP split coefficients for one unit of flow.
                nt_ecmp_run(ecmp, dijk, s, t, 1.0, node_flow, arc_flow)

                # Convert nt4python arc handles back to the original arc IDs.
                for original_arc_id, a in ntg.arcs:
                    frac = float(nt_arcmap_double_get(arc_flow, a))
                    if frac > _EPS:
                        r[(u, v, original_arc_id)] = frac

    finally:
        nt_arcmap_double_free(arc_flow)
        nt_nodemap_double_free(node_flow)
        nt_ecmp_free(ecmp)
        nt_dijkstra_free(dijk)

    return r
