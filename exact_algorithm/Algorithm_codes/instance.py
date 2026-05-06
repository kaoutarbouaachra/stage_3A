"""
instance.py
-----------
Load and parse the three JSON input files of a T-ASR instance.
"""

import json
from pathlib import Path
from dataclasses import dataclass, field
from typing import Dict, List, Set, Tuple


@dataclass
class Instance:
    """All data for one T-ASR instance."""

    # ---- network ----
    nodes    : List[int]
    node_name: Dict[int, str]
    links    : Dict[int, dict]   # link_id -> {from, to, metric, capacity}
    arc_ids  : List[int]

    # ---- time ----
    T      : List[int]           # [0, 1, ..., h-1]
    T_star : List[int]           # T \ {0}

    # ---- demands ----
    # each demand: (demand_id, source, target, [volume_t0, volume_t1, ...])
    demands : List[Tuple]

    # ---- scenario ----
    max_seg : int
    budget  : Dict[int, int]       # t -> max reconfig cost
    interv  : Dict[int, Set[int]]  # t -> set of unavailable link ids

    # ---- instance name (for file naming) ----
    name : str = "instance"


def load_instance(net_file: Path, tm_file: Path,
                  scenario_file: Path) -> Instance:
    """
    Load the three JSON files and return a parsed Instance object.

    Parameters
    ----------
    net_file      : path to *-net.json
    tm_file       : path to *-tm.json
    scenario_file : path to *-scenario.json

    Returns
    -------
    Instance
    """
    with open(net_file)      as f: net      = json.load(f)
    with open(tm_file)       as f: tm       = json.load(f)
    with open(scenario_file) as f: scenario = json.load(f)

    # ---- network ----
    nodes     = [n['id'] for n in net['nodes']]
    node_name = {n['id']: n['name'] for n in net['nodes']}
    links     = {l['id']: l for l in net['links']}
    arc_ids   = list(links.keys())

    # ---- time ----
    T      = list(range(tm['num_time_slots']))
    T_star = T[1:]

    # ---- demands ----
    demands = [
        (i, d['s'], d['t'], d['v'])
        for i, d in enumerate(tm['demands'])
    ]

    # ---- scenario ----
    max_seg = scenario.get('max_segments', 4)
    budget  = {b['t']: b['value'] for b in scenario.get('budget', [])}
    interv  = {
        iv['t']: set(iv['links'])
        for iv in scenario.get('interventions', [])
    }

    # ---- name from filename ----
    name = net_file.name[: -len('-net.json')]

    return Instance(
        nodes=nodes, node_name=node_name, links=links, arc_ids=arc_ids,
        T=T, T_star=T_star, demands=demands,
        max_seg=max_seg, budget=budget, interv=interv,
        name=name,
    )


def summary(inst: Instance) -> str:
    """Return a one-line summary string."""
    N = len(inst.arc_ids) * len(inst.T)
    return (
        f"{inst.name} | "
        f"{len(inst.nodes)} nodes, {len(inst.arc_ids)} arcs, "
        f"{len(inst.demands)} demands, {len(inst.T)} timesteps, "
        f"N={N}"
    )
