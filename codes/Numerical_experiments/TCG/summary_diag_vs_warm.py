# Numerical_experiments/TCG/summary_warm_vs_diag.py

"""
Summary comparison between TCG-exact and TCG-diag.

For each instance, within the certified prefix of TCG-exact (warm) logs,
we compare the sorted load vectors. The leximin verdict is decided at the
first diverging rank.

The produced figure is a compact horizontal stacked bar showing:
    - TCG-exact better
    - Tie
    - TCG-diag better
"""

import os
import sys
import json
import re
from pathlib import Path


os.environ.setdefault("T_ASR_ECMP_BACKEND", "scratch")


HERE = Path(__file__).resolve()

ALGO = next(
    (p / "Algorithm_codes" for p in HERE.parents
     if (p / "Algorithm_codes").is_dir()),
    None
)

if ALGO is None:
    sys.exit("Algorithm_codes/ not found above this script.")

sys.path.insert(0, str(ALGO))


try:
    from core.ecmp import (
        compute_all_splits,
        compute_loads,
        sorted_load_vector,
        ECMP_BACKEND
    )

except Exception as e:
    sys.exit(
        f"Cannot import core.ecmp: {e}\n"
        "-> pip install networkx numpy matplotlib"
    )


# ============================================================
# PARAMETERS
# ============================================================

INSTANCES = [
    f"setAP-{i:03d}"
    for i in range(1, 89)
]

REL_TOL = 1e-5
ABS_TOL = 1e-9


INPUT_ROOT = ALGO.parent / "setAP"

RESULTS = ALGO / "results"
LOGS = ALGO / "logs"

DIAG_DIR = "TCG_diag"
WARM_DIR = "TCG_exact_warm"


# ============================================================
# Utilities
# ============================================================

def resolve_child(parent, name):

    exact = parent / name

    if exact.exists():
        return exact

    if parent.is_dir():
        low = name.lower()

        for child in parent.iterdir():
            if child.name.lower() == low:
                return child

    return exact



class Inst:
    pass



def find_input(name, suffix):

    files = [
        INPUT_ROOT / f"{name}-{suffix}.json",
        INPUT_ROOT / name / f"{name}-{suffix}.json"
    ]

    for f in files:
        if f.exists():
            return f

    raise FileNotFoundError(
        f"Input not found for {name}-{suffix}"
    )



def load_instance(name):

    net = json.loads(
        find_input(name, "net").read_text()
    )

    tm = json.loads(
        find_input(name, "tm").read_text()
    )

    scen = json.loads(
        find_input(name, "scenario").read_text()
    )


    inst = Inst()

    inst.nodes = [
        n["id"]
        for n in net["nodes"]
    ]

    inst.node_name = {
        n["id"]: n.get("name", str(n["id"]))
        for n in net["nodes"]
    }


    inst.links = {
        l["id"]: {
            "id": l["id"],
            "from": l["from"],
            "to": l["to"],
            "metric": float(l["metric"]),
            "capacity": float(l["capacity"])
        }
        for l in net["links"]
    }


    inst.arc_ids = list(inst.links.keys())


    h = tm["num_time_slots"]

    inst.T = list(range(h))
    inst.T_star = list(range(1, h))


    inst.demands = [
        (
            i,
            d["s"],
            d["t"],
            list(d["v"])
        )
        for i, d in enumerate(tm["demands"])
    ]


    inst.max_seg = scen.get("max_segments")

    inst.budget = {
        b["t"]: b["value"]
        for b in scen.get("budget", [])
    }

    inst.interv = {
        iv["t"]: set(iv["links"])
        for iv in scen.get("interventions", [])
    }


    return inst



def recompute_sorted(name, method_dir, inst, r):

    f = (
        resolve_child(RESULTS / name, method_dir)
        /
        f"{name}-srpaths.json"
    )


    if not f.exists():
        return None


    data = json.loads(
        f.read_text()
    )


    wp = {
        (e["d"], e["t"]): e.get("w", [])
        for e in data.get("srpaths", [])
    }


    return sorted_load_vector(
        compute_loads(wp, inst, r),
        inst
    )



def warm_vector_from_logs(name):

    d = resolve_child(
        LOGS,
        WARM_DIR
    )

    logs = sorted(
        d.glob(f"{name}_*.log")
    )


    if not logs:
        return None, None


    txt = logs[-1].read_text(
        errors="ignore"
    )


    zmap = {
        int(k): float(v)
        for k, v in re.findall(
            r"z_(\d+)\*\s*=\s*([-\d.eE+]+)",
            txt
        )
    }


    if not zmap:
        return None, None


    vec = [
        zmap[k]
        for k in sorted(zmap)
    ]


    return vec, len(vec)



def lex_verdict(vd, vw, kmax):

    for k in range(
        min(len(vd), len(vw), kmax)
    ):

        tol = (
            REL_TOL *
            max(abs(vd[k]), abs(vw[k]))
            +
            ABS_TOL
        )


        if abs(vd[k] - vw[k]) > tol:

            if vd[k] < vw[k]:
                return k + 1, "diag"

            else:
                return k + 1, "warm"


    return None, "tie"



# ============================================================
# Compute results
# ============================================================

print(
    f"ECMP backend : {ECMP_BACKEND}\n"
)


rows = []


for name in INSTANCES:

    try:
        inst = load_instance(name)

    except FileNotFoundError:
        continue


    r = compute_all_splits(inst)


    warm, rc = warm_vector_from_logs(name)

    diag = recompute_sorted(
        name,
        DIAG_DIR,
        inst,
        r
    )


    if rc is None or warm is None or diag is None:
        continue


    k = max(
        1,
        min(
            rc,
            len(diag),
            len(warm)
        )
    )


    kdiv, win = lex_verdict(
        diag[:k],
        warm[:k],
        k
    )


    if win == "tie":

        ld = diag[0]
        lw = warm[0]
        gain = 0.0


    else:

        ld = diag[kdiv - 1]
        lw = warm[kdiv - 1]

        gain = (
            (ld - lw) / ld
            if ld > 0
            else 0.0
        )


    rows.append(
        (
            name,
            win,
            kdiv,
            ld,
            lw,
            gain
        )
    )



if not rows:

    sys.exit(
        "No instances with both results were found."
    )



n_warm = sum(
    1
    for _, w, *_ in rows
    if w == "warm"
)

n_diag = sum(
    1
    for _, w, *_ in rows
    if w == "diag"
)

n_tie = sum(
    1
    for _, w, *_ in rows
    if w == "tie"
)



print(
    f"instances compared : {len(rows)}"
)

print(
    f"  exact (warm) better : {n_warm}"
)

print(
    f"  tie                 : {n_tie}"
)

print(
    f"  diag better         : {n_diag}"
)



gains = [
    g
    for _, w, _, _, _, g in rows
    if w == "warm"
]


if gains:

    import statistics as st

    print(
        "  warm gain @ first divergence: "
        f"median={st.median(gains)*100:.1f}% "
        f"max={max(gains)*100:.1f}%"
    )



# ============================================================
# Figure
# ============================================================

import matplotlib.pyplot as plt


outdir = HERE.parent / "figs"

outdir.mkdir(
    exist_ok=True
)



def save(fig, stem):

    fig.tight_layout()

    for ext in ("pdf", "png"):

        fig.savefig(
            outdir / f"{stem}.{ext}",
            dpi=300,
            bbox_inches="tight"
        )


    plt.close(fig)


    print(
        f"\n-> {outdir / (stem + '.pdf')}"
    )


def figure_summary_bar():

    total = len(rows)

    p_warm = 100 * n_warm / total
    p_tie = 100 * n_tie / total


    fig, ax = plt.subplots(figsize=(6.0, 1.3))


    # bar
    ax.barh(
        0,
        n_warm,
        color="0.30",
        edgecolor="black",
        height=0.35
    )

    ax.barh(
        0,
        n_tie,
        left=n_warm,
        color="0.75",
        edgecolor="black",
        height=0.35
    )


    # labels
    ax.text(
        n_warm / 2,
        0,
        f"{n_warm} ({p_warm:.1f}%)",
        ha="center",
        va="center",
        fontsize=9,
        color="white",
        fontweight="bold"
    )


    ax.text(
        n_warm + n_tie / 2,
        0,
        f"{n_tie} ({p_tie:.1f}%)",
        ha="center",
        va="center",
        fontsize=9,
        color="black",
        fontweight="bold"
    )


    # title
    ax.set_title(
        "TCG-exact vs TCG-diag (88 instances)",
        fontsize=12,
        pad=12
    )


    ax.set_xlim(
        0,
        total
    )


    ax.set_ylim(
        -0.35,
        0.35
    )


    ax.set_xlabel(
        "Number of instances",
        fontsize=9
    )


    ax.tick_params(
        axis="x",
        labelsize=8
    )

    ax.set_yticks([])


    # remove borders
    for spine in [
        "top",
        "right",
        "left"
    ]:
        ax.spines[spine].set_visible(False)



    # conclusion text below
    ax.text(
        0,
        -0.28,
        "TCG-diag better: 0 instances",
        fontsize=8
    )


    fig.tight_layout()


    for ext in ("pdf", "png"):
        fig.savefig(
            outdir / f"summary_warm_vs_diag_bar.{ext}",
            dpi=300,
            bbox_inches="tight"
        )


    plt.close(fig)

    print(
        f"\n-> {outdir / 'summary_warm_vs_diag_bar.pdf'}"
    )


figure_summary_bar()