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
# Comparison/figure window = warm's CERTIFIED prefix only (ranks 1..rc, where
# rc = number of z_k* entries in the warm log). Beyond rc, warm is not proven
# optimal, so we neither compare nor plot there. K_FIGURE is an OPTIONAL extra
# cap on the number of ranks shown and never exceeds rc; None => full prefix.
K_FIGURE = None
# Uniform x-axis across ALL figures so they share ONE size (clean tiling in the
# report). The common upper bound X_MAX is chosen by:
#   "difference" -> largest rank where ANY instance still shows a diag/warm
#                   difference (ranks beyond it tie everywhere, so cropping them
#                   loses nothing) -- this is "jusqu'au max de rangs de diff."
#   "shown"      -> largest certified-prefix length shown across instances
#   <int>        -> a fixed common value
COMMON_RANKS = "difference"
# z_k* in the warm log is printed to ~6 significant figures. Compare loads with
# a RELATIVE tolerance so that log-rounding never spuriously breaks a genuine
# tie (an absolute 1e-6 is already exceeded by rounding at values ~1-2).
REL_TOL  = 1e-5
ABS_TOL  = 1e-9
# setAP instances are named with a capital 'P' and 3-digit zero-padding
# (setAP-001 ... setAP-088), matching the campaign logs.
INSTANCES = [f"setAP-{i:03d}" for i in range(1, 89)]
# Tolerance above which the warm srpaths vector and the logged z_k* prefix
# are considered inconsistent (one-sided freezing leaves at most ~EPS slack,
# so anything larger signals a real srpaths<->log mismatch).
WARM_CONSISTENCY_TOL = 1e-4
# ============================================================

# setAP inputs live flat under codes/setAP/ (e.g. setAP/setAP-079-net.json).
INPUT_ROOT = ALGO.parent / "setAP"
RESULTS    = ALGO / "results"
LOGS       = ALGO / "logs"

# Result/log folder names as they appear on disk (lowercase). Lookups below
# are made case-insensitive so either casing works on Windows or Linux.
DIAG_DIR = "TCG_diag"
WARM_DIR = "TCG_exact_warm"

def resolve_child(parent, name):
    """Return parent/name, falling back to a case-insensitive match among the
    children of `parent` if the exact-cased folder is absent."""
    exact = parent / name
    if exact.exists():
        return exact
    if parent.is_dir():
        low = name.lower()
        for child in parent.iterdir():
            if child.name.lower() == low:
                return child
    return exact  # non-existent; caller handles the miss

class Inst:
    pass

def find_input(name, suffix):
    """Locate an input file, tolerating both a flat layout
    (setAP/<name>-<suffix>.json) and a nested one (setAP/<name>/<name>-<suffix>.json)."""
    candidates = [
        INPUT_ROOT / f"{name}-{suffix}.json",          # flat (setAP)
        INPUT_ROOT / name / f"{name}-{suffix}.json",   # nested (legacy setA)
    ]
    for f in candidates:
        if f.exists():
            return f
    raise FileNotFoundError(
        f"Input file not found for {name}-{suffix}; tried: "
        + " | ".join(str(c) for c in candidates))

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

def recompute_sorted(name, method_dir, inst, r):
    """Sorted load vector from the WRITTEN srpaths (checker-aligned)."""
    f = resolve_child(RESULTS / name, method_dir) / f"{name}-srpaths.json"
    if not f.exists():
        return None
    data = json.loads(f.read_text())
    wp = {(e["d"], e["t"]): e.get("w", []) for e in data.get("srpaths", [])}
    return sorted_load_vector(compute_loads(wp, inst, r), inst)

def warm_vector_from_logs(name):
    """Read z_k* (k=1,2,...) from the most recent warm log, in rank order.
    Returns (vector, last_rank_reached)."""
    d = resolve_child(LOGS, WARM_DIR)
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
        tol = REL_TOL * max(abs(vd[k]), abs(vw[k])) + ABS_TOL
        if abs(vd[k] - vw[k]) > tol:
            return k + 1, ("diag" if vd[k] < vw[k] else "warm")
    return None, "tie"

# ============================================================
#  Compute
# ============================================================
print(f"ECMP backend : {ECMP_BACKEND}   K_FIGURE = {K_FIGURE}\n")

data = {}
for name in INSTANCES:
    try:
        inst = load_instance(name)
    except FileNotFoundError as e:
        print(f"[{name}] input MISSING: {e}\n")
        data[name] = None
        continue
    r = compute_all_splits(inst)

    warm_log, rc = warm_vector_from_logs(name)   # warm vector = z_k* read from the log
    diag         = recompute_sorted(name, DIAG_DIR, inst, r)  # diag recomputed from srpaths

    if rc is None or warm_log is None or diag is None:
        print(f"[{name}] MISSING (rc={rc}, warm_log={warm_log is not None}, "
              f"diag={diag is not None})\n")
        data[name] = None
        continue

    # Optional rigor guard (does NOT affect the comparison, which uses the log):
    # the warm srpaths, if present, should reproduce the certified z_k* prefix
    # within the one-sided freezing slack.
    warm_srp = recompute_sorted(name, WARM_DIR, inst, r)
    if warm_srp is not None:
        m = min(rc, len(warm_srp), len(warm_log))
        dmax = max((abs(warm_srp[i] - warm_log[i]) for i in range(m)), default=0.0)
        if dmax > WARM_CONSISTENCY_TOL:
            print(f"[{name}] WARN warm srpaths vs log z_k* diverge "
                  f"(max|Δ|={dmax:.2e} over {m} ranks) -> check freezing/srpaths")

    cap = rc if K_FIGURE is None else min(K_FIGURE, rc)
    k = max(1, min(cap, len(diag)))          # never beyond the certified prefix
    data[name] = (diag[:k], warm_log[:k], k, rc)

    print(f"[{name}] certified ranks rc={rc}  -> comparing/plotting {k}")
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

def last_diff_rank(diag, warm, k):
    """Highest rank (1-based, within 1..k) where diag and warm still differ."""
    last = 0
    for i in range(min(len(diag), len(warm), k)):
        tol = REL_TOL * max(abs(diag[i]), abs(warm[i])) + ABS_TOL
        if abs(diag[i] - warm[i]) > tol:
            last = i + 1
    return last

# ---- common x-axis so every figure has the SAME size --------------------
valid = [d for d in data.values() if d is not None]
if not valid:
    print("[fig] nothing to plot."); sys.exit(0)
if COMMON_RANKS == "difference":
    X_MAX = max(last_diff_rank(dg, wm, k) for (dg, wm, k, rc) in valid)
elif COMMON_RANKS == "shown":
    X_MAX = max(k for (dg, wm, k, rc) in valid)
else:
    X_MAX = int(COMMON_RANKS)
X_MAX = max(1, X_MAX)
print(f"[fig] common x-axis X_MAX = {X_MAX} ranks (mode='{COMMON_RANKS}')\n")

# One figure size and one tick set, shared by all instances.
FIGW = min(22.0, max(4.0, 0.30 * X_MAX))
FIGH = 4.0
_all_ticks = np.arange(1, X_MAX + 1)
XT = _all_ticks if X_MAX <= 25 else _all_ticks[::int(np.ceil(X_MAX / 20))]

def make_figure(name, diag, warm, k, rc):
    kk    = min(k, X_MAX)                     # clip drawing to the common window
    ranks = np.arange(1, kk + 1)
    d = np.asarray(diag[:kk], float); w = np.asarray(warm[:kk], float)

    fig, ax = plt.subplots(figsize=(FIGW, FIGH))
    ax.bar(ranks - width/2, d, width, label="TCG-diag",
           color=C_DIAG, edgecolor="black", linewidth=0.6)
    ax.bar(ranks + width/2, w, width, label="TCG-exact (warm)",
           color=C_WARM, edgecolor="black", linewidth=0.6, hatch="//")

    ax.set_title(name, fontsize=11, pad=24)
    ax.set_xlabel(rf"Rank $k$  (certified prefix, $k\leq{rc}$)")
    ax.set_ylabel(r"Sorted load $\lambda^{\downarrow}_k$")
    ax.set_xticks(XT)
    ax.set_xlim(0.5, X_MAX + 0.5)            # SAME axis for every figure
    ax.grid(axis="y", linestyle=":", linewidth=0.5, color="0.8")
    ax.set_axisbelow(True)

    kdiv, win = lex_verdict(d, w, kk)
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
        fig.savefig(outdir / f"leximin_{name}.{ext}", dpi=150)
    plt.close(fig)
    print(f"-> {outdir / f'leximin_{name}.pdf'}")

for name in INSTANCES:
    if data[name] is None:
        continue
    diag, warm, k, rc = data[name]
    make_figure(name, diag, warm, k, rc)