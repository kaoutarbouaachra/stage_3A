"""
Function: segments.py
Role: Demand-aware, reachability-based segment pruning.
"""

from __future__ import annotations

from typing import Dict, Set, Tuple

import networkx as nx

from core.instance import Instance


def useful_segments(inst: Instance) -> Dict[int, Set[Tuple[int, int]]]:
    """
    Returns valid_segs[t]: the set of (i, j) node pairs that can appear as a
    segment in some demand's SR-path at period t, given inst.max_seg.

    Type 1: (s, t)          -- direct, 0 waypoints
    Type 2: (s, w), (w, t)  -- first/last segment, >= 1 waypoint
    Type 3: (w1, w2)        -- intermediate hop, >= 2 waypoints
    """
    max_seg = inst.max_seg
    max_wps = max_seg - 1
    valid: Dict[int, Set[Tuple[int, int]]] = {}

    for t in inst.T:
        unavail = inst.interv.get(t, set())
        G = nx.DiGraph()
        for nid in inst.nodes:
            G.add_node(nid)
        for l in inst.links.values():
            if l["id"] not in unavail:
                G.add_edge(l["from"], l["to"], weight=l["metric"])

        Gr = G.reverse(copy=False)
        reach_from: Dict[int, Set[int]] = {}
        reach_to: Dict[int, Set[int]] = {}
        for nid in inst.nodes:
            reach_from[nid] = set(nx.single_source_shortest_path_length(G, nid).keys())
            reach_to[nid] = set(nx.single_source_shortest_path_length(Gr, nid).keys())

        segs: Set[Tuple[int, int]] = set()
        for (_d_id, src, tgt, _vols) in inst.demands:
            valid_wps = (reach_from[src] & reach_to[tgt]) - {src, tgt}

            if tgt in reach_from[src]:
                segs.add((src, tgt))

            if max_wps >= 1:
                for w in valid_wps:
                    segs.add((src, w))
                    segs.add((w, tgt))

            if max_wps >= 2:
                for w1 in valid_wps:
                    for w2 in valid_wps:
                        if w2 != w1 and w2 in reach_from[w1]:
                            segs.add((w1, w2))
        valid[t] = segs

    total = sum(len(v) for v in valid.values())
    naive = len(inst.nodes) ** 2 * len(inst.T)
    factor = naive // max(total, 1)
    print(f"  -> {total:,} useful segment slots  "
          f"(naive: {naive:,} -> reduction x{factor})")
    return valid
