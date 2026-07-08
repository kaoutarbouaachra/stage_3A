"""
Function: bigm.py
Role: Valid, per-(arc, period) big-M constants for ALEXA's rank-assignment
linking constraint.

"""

from __future__ import annotations

from typing import Dict, Tuple

from core.ecmp import SplitCoeffs
from core.instance import Instance

ArcTime = Tuple[int, int]  # (arc_id, t)


def compute_utilization_bigm(inst: Instance, r: Dict[int, SplitCoeffs]) -> Dict[ArcTime, float]:
    """
    M(a, t) = (maxSeg / c(a)) * max_{i != j} r(i, j, a, t) * sum_d nu(d, t)

    Requires the ECMP split coefficients r, which are already computed
    before the model is built -- this remains solve-free: no MILP is solved
    to derive it, only the split-coefficient precomputation that happens
    anyway as part of loading the instance.
    """
    max_seg = inst.max_seg

    total_volume_by_t: Dict[int, float] = {t: 0.0 for t in inst.T}
    for (_d_id, _src, _tgt, vols) in inst.demands:
        for t in inst.T:
            total_volume_by_t[t] += float(vols[t])

    # max_{i,j} r(i,j,a,t): read directly off the (sparse) split-coefficient
    # tables. Only reachable/nonzero pairs are stored there, which is
    # exactly the set the proposition's max needs to range over.
    max_r: Dict[ArcTime, float] = {}
    for t, r_t in r.items():
        for (_i, _j, arc_id), coeff in r_t.items():
            key = (arc_id, t)
            if coeff > max_r.get(key, 0.0):
                max_r[key] = coeff

    bigm: Dict[ArcTime, float] = {}
    for link in inst.links.values():
        cap = float(link["capacity"])
        if cap <= 0:
            continue
        arc_id = link["id"]
        for t in inst.T:
            m_r = max_r.get((arc_id, t), 0.0)
            bigm[(arc_id, t)] = (max_seg / cap) * m_r * total_volume_by_t[t]
    return bigm
