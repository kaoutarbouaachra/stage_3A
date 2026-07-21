#!/usr/bin/env python3
"""
cg_exact.adapter
================

Bridge from a ``core`` instance (network + traffic matrix + scenario) and the
``core`` ECMP split coefficients to the engine's :class:`TASRData`.

This version is wired to the ACTUAL core API:

    core.instance.load_instance(net, tm, scenario) -> Instance
        Instance.nodes    : List[int]
        Instance.links    : Dict[arc_id, {from, to, metric, capacity}]
        Instance.arc_ids  : List[int]
        Instance.T        : List[int]      # [0, 1, ...]
        Instance.demands  : List[(d_id, s, t, [v0, v1, ...])]
        Instance.max_seg  : int
        Instance.budget   : Dict[t, value]
        Instance.interv   : Dict[t, Set[arc_id]]

    core.ecmp.compute_all_splits(inst) -> Dict[t, Dict[(u, v, arc_id), frac]]
        Already loops over inst.T AND removes intervention arcs per period
        internally (see ecmp_nt / ecmp_from_scratch), so the adapter does NOT
        recompute subgraphs.

"""

from __future__ import annotations

from typing import Any, Dict, FrozenSet, List, Set, Tuple

from .data import (
    Arc,
    Demand,
    DemandId,
    Node,
    Period,
    Segment,
    TASRData,
    TASRError,
)

from core import instance as core_instance
from core import ecmp as core_ecmp


def build_tasr_data_from_files(net_path: str, tm_path: str, scenario_path: str) -> TASRData:
    from pathlib import Path
    inst = core_instance.load_instance(Path(net_path), Path(tm_path), Path(scenario_path))
    return build_tasr_data_from_instance(inst)


def build_tasr_data_from_instance(inst: Any) -> TASRData:
    periods: Tuple[Period, Period] = (0, 1)
    if len(inst.T) < 2:
        raise TASRError(f"Instance has horizon h={len(inst.T)}; this engine needs exactly 2 periods.")

    # --- arcs and capacities (arc = arc_id) ---------------------------------
    arcs: List[Arc] = list(inst.arc_ids)
    capacities: Dict[Arc, float] = {
        arc_id: float(inst.links[arc_id]["capacity"]) for arc_id in inst.arc_ids
    }
    nodes: List[Node] = list(inst.nodes)

    # --- ECMP split coefficients (core handles per-period interventions) -----
    # r[t][(u, v, arc_id)] = fraction
    r: Dict[int, Dict[Tuple[Node, Node, Arc], float]] = core_ecmp.compute_all_splits(inst)

    # --- routing_fraction[(segment, arc_id, t)] and feasible_segments[(d,t)] -
    routing_fraction: Dict[Tuple[Segment, Arc, Period], float] = {}
    feasible_per_t: Dict[Period, Set[Segment]] = {t: set() for t in periods}
    for t in periods:
        for (u, v, arc_id), frac in r[t].items():
            if u == v:
                continue
            seg: Segment = (u, v)
            routing_fraction[(seg, arc_id, t)] = float(frac)
            feasible_per_t[t].add(seg)

    # --- demands and per-demand candidate segments --------------------------
    # The feasible set is demand-independent here, so we build ONE immutable
    # set per period and let every demand share the same reference. Using
    # set(feasible_per_t[t]) per demand made #demands x 2 full copies of a
    # set with hundreds of thousands of entries -> MemoryError on setA-20.
    feas_by_t: Dict[Period, FrozenSet[Segment]] = {
        t: frozenset(feasible_per_t[t]) for t in periods
    }

    demands: List[Demand] = []
    feasible_segments: Dict[Tuple[DemandId, Period], FrozenSet[Segment]] = {}
    for (d_id, s, tgt, vols) in inst.demands:
        if len(vols) < 2:
            raise TASRError(f"Demand {d_id} has fewer than 2 traffic values (need t=0 and t=1).")
        demands.append(Demand(id=d_id, source=s, target=tgt,
                              volume={0: float(vols[0]), 1: float(vols[1])}))
        for t in periods:
            feasible_segments[(d_id, t)] = feas_by_t[t]  # shared reference, no copy

    # --- budget = kappa(1) ---------------------------------------------------
    budget = float(inst.budget.get(1, 0.0))

    return TASRData(
        nodes=nodes,
        arcs=arcs,
        capacities=capacities,
        demands=demands,
        feasible_segments=feasible_segments,
        routing_fraction=routing_fraction,
        max_segments=int(inst.max_seg),
        budget=budget,
        periods=periods,
    )