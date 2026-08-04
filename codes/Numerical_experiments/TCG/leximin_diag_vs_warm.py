# Numerical_experiments/TCG/leximin_diag_vs_warm.py
import os, sys, json, re
from pathlib import Path

os.environ.setdefault("T_ASR_ECMP_BACKEND", "scratch")  # Windows laptop -> NetworkX

HERE = Path(__file__).resolve()
ALGO = next((p / "Algorithm_codes" for p in HERE.parents
             if (p / "Algorithm_codes").is_dir()), None)
if ALGO is None:
    sys.exit("Algorithm_codes/ not found above this script.")
sys.path.insert(0, str(ALGO))

try:
    from core.ecmp import (compute_all_splits, compute_loads,   # type: ignore
                           sorted_load_vector, ECMP_BACKEND)
except Exception as e:
    sys.exit(f"Cannot import core.ecmp: {e}\n-> pip install networkx numpy matplotlib")

# ============================================================
#  USER KNOBS
# ============================================================
K_FIGURE = 15   # display cap; per instance shows min(K_FIGURE, ranks in log)
EPS      = 1e-6
INSTANCES = ["setA-01", "setA-02", "setA-03", "setA-04"]
# ============================================================

INPUT_ROOT = ALGO.parent / "setA"
RESULTS    = ALGO / "results"
LOGS       = ALGO / "logs"

class Inst:
    pass

def find_input(name, suffix):
    f = INPUT_ROOT / name / f"{name}-{suffix}.json"
    if f.exists():
        return f
    raise FileNotFoundError(f"Input file not found: {f}")

def load_instance(name):
    net  = json.loads(find_input(name, "net").read_text())
    tm   = json.loads(find_input(name, "tm").read_text())
    scen = json.loads(find_input(name, "scenario").read_text())
    inst = Inst()
    inst.nodes = [n["id"] for n in net["nodes"]]
    inst.node_name = {n["id"]: n.get("name", str(n["id"])) for n in net["nodes"]}
    inst.links = {l["id"]: {"id": l["id"], "from": l["from"], "to": l["to"],
                            "metric": float(l["metric"]),
                            "capacity": float(l["capacity"])}
                  for l in net["links"]}
    inst.arc_ids = list(inst.links.keys())
    h = tm["num_time_slots"]
    inst.T = list(range(h)); inst.T_star = list(range(1, h))
    inst.demands = [(i, d["s"], d["t"], list(d["v"]))
                    for i, d in enumerate(tm["demands"])]
    inst.max_seg = scen.get("max_segments")
    inst.budget  = {b["t"]: b["value"] for b in scen.get("budget", [])}
    inst.interv  = {iv["t"]: set(iv["links"]) for iv in scen.get("interventions", [])}
    return inst

def recompute_sorted(name, method, inst, r):
    """Sorted load vector from the WRITTEN srpaths (checker-aligned)."""
    f = RESULTS / name / method / f"{name}-srpaths.json"
    if not f.exists():
        return None
    data = json.loads(f.read_text())
    wp = {(e["d"], e["t"]): e.get("w", []) for e in data.get("srpaths", [])}
    return sorted_load_vector(compute_loads(wp, inst, r), inst)

def warm_vector_from_logs(name):
    """Read z_k* (k=1,2,...) from the most recent warm log, in rank order.
    Returns (vector, last_rank_reached)."""
    d = LOGS / "TCG_exact_warm"
    logs = sorted(d.glob(f"{name}_*.log"))
    if not logs:
        return None, None
    txt = logs[-1].read_text(errors="ignore")
    zmap = {int(k): float(v)
            for k, v in re.findall(r"z_(\d+)\*\s*=\s*([-\d.eE+]+)", txt)}
    if not zmap:
        return None, None
    vec = [zmap[k] for k in sorted(zmap)]
    return vec, len(vec)

def lex_verdict(vd, vw, kmax):
    for k in range(min(len(vd), len(vw), kmax)):
        if abs(vd[k] - vw[k]) > EPS:
            return k + 1, ("diag" if vd[k] < vw[k] else "warm")
    return None, "tie"

# ============================================================
#  Compute
# ============================================================
print(f"ECMP backend : {ECMP_BACKEND}   K_FIGURE = {K_FIGURE}\n")

data = {}
for name in INSTANCES:
    inst = load_instance(name)
    r = compute_all_splits(inst)

    warm_log, rc = warm_vector_from_logs(name)
    diag         = recompute_sorted(name, "TCG_diag", inst, r)

    if rc is None or warm_log is None or diag is None:
        print(f"[{name}] MISSING (rc={rc}, warm_log={warm_log is not None}, "
              f"diag={diag is not None})\n")
        data[name] = None
        continue

    k = max(1, min(K_FIGURE, rc, len(warm_log), len(diag)))
    data[name] = (diag[:k], warm_log[:k], k)

    print(f"[{name}] ranks in log={rc}  -> showing {k}")
    print(f"  diag (srpaths) : " + ", ".join(f"{v:.4f}" for v in diag[:k]))
    print(f"  warm (log z_k*): " + ", ".join(f"{v:.4f}" for v in warm_log[:k]))
    kdiv, win = lex_verdict(diag[:k], warm_log[:k], k)
    print("  -> leximin: " + ("identical" if win == "tie"
                              else f"{win} wins at rank {kdiv}") + "\n")

# ============================================================
#  Figures — ONE PER INSTANCE
# ============================================================
import numpy as np
import matplotlib.pyplot as plt

C_DIAG, C_WARM = "0.70", "0.30"
width  = 0.4
outdir = HERE.parent / "figs"; outdir.mkdir(exist_ok=True)

def make_figure(name, diag, warm, k):
    ranks = np.arange(1, k + 1)
    d = np.asarray(diag, float); w = np.asarray(warm, float)

    fig, ax = plt.subplots(figsize=(max(4, 4), 4))
    ax.bar(ranks - width/2, d, width, label="TCG-diag",
           color=C_DIAG, edgecolor="black", linewidth=0.6)
    ax.bar(ranks + width/2, w, width, label="TCG-exact (warm)",
           color=C_WARM, edgecolor="black", linewidth=0.6, hatch="//")

    ax.set_title(name, fontsize=11,pad=24)
    ax.set_xlabel("Rank $k$")
    ax.set_ylabel(r"Sorted load $\lambda^{\downarrow}_k$")
    ax.set_xticks(ranks); ax.set_xlim(0.5, k + 0.5)
    ax.grid(axis="y", linestyle=":", linewidth=0.5, color="0.8")
    ax.set_axisbelow(True)

    kdiv, win = lex_verdict(d, w, k)
    if win == "tie":
        txt = "identical prefix"
    else:
        txt = f"{win} wins @ rank {kdiv}"
        ax.axvline(kdiv, color="black", linestyle="--", linewidth=0.8, alpha=0.55)
    ax.annotate(txt, xy=(0.03, 0.96), xycoords="axes fraction",
                fontsize=8, ha="left", va="top",
                bbox=dict(boxstyle="round,pad=0.25", fc="white", ec="0.6", lw=0.5))

    ax.legend(loc="lower center", bbox_to_anchor=(0.5, 1.02),
              ncol=2, frameon=False, fontsize=8)
    fig.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(outdir / f"leximin_{name}.{ext}", bbox_inches="tight", dpi=150)
    plt.close(fig)
    print(f"-> {outdir / f'leximin_{name}.pdf'}")

for name in INSTANCES:
    if data[name] is None:
        continue
    diag, warm, k = data[name]
    make_figure(name, diag, warm, k)