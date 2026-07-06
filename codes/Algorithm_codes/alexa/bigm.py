"""
Function: bigm.py
Role: Valid, per-(arc, period) big-M constants for ALEXA's rank-assignment
linking constraint.

Proposition (Valid Big-M constant). For every (a, t) and every feasible
routing P,

    0 <= lambda(a, t; P) <= M(a, t)

where

    M(a,t) = (maxSeg / c(a)) * max_{i != j in V} r(i,j,a,t) * sum_{d in D} nu(d,t).

Proof sketch: fix a demand d. Its contribution to lambda(a,t) * c(a) is
sum_{i,j} r(i,j,a,t) * nu(d,t) * x[d,t,i,j]. The segment-count constraint
allows at most maxSeg of the x[d,t,i,j] to be 1, and each active segment
contributes at most (max_{i,j} r(i,j,a,t)) * nu(d,t), so demand d's total
contribution is at most maxSeg * max_{i,j} r(i,j,a,t) * nu(d,t). Summing
over demands gives the bound above.

Correction vs. an earlier version of this file: that version used
M(a,t) = sum_d nu(d,t) / c(a), which implicitly assumes a single demand's
path touches arc a with total split-coefficient mass at most 1. That is not
guaranteed by the model: a demand's SR-path can legally revisit the same
physical arc via more than one of its (up to maxSeg) segments (see the
trail-vs-repeated-node distinction used elsewhere in this project), so a
single demand's contribution to lambda(a,t) can in principle exceed
nu(d,t)/c(a). The maxSeg * max_r factor closes that gap and makes the bound
unconditionally valid rather than valid only in the typical case.

Design choice vs. the proposition as stated: the proposition defines a
single GLOBAL constant M = max_{(a,t)} M(a,t), convenient for a clean,
uniform statement. This module instead returns M(a,t) PER (arc, period).
Nothing in the proof requires a uniform constant -- the linking constraint
at (a,t) only needs a valid bound on lambda(a,t) itself -- and using M(a,t)
directly is strictly tighter whenever arc-time pairs differ in capacity,
demand volume, or split coefficients, which is the typical case.
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
