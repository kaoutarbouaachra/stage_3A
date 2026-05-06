"""
main.py
-------
Entry point for the T-ASR Sequential Lex-Minimax solver.

Usage
-----
    python main.py --instance toy
    python main.py --instance toy --max-iter 20 --time-limit 60
    python main.py --instance toy --node-limit 5000
    python main.py --net path/to/net.json --tm path/to/tm.json --scenario path/to/scen.json
"""

import argparse
import json
import sys
from pathlib import Path

from instance import load_instance, summary
from ecmp     import compute_all_splits, compute_loads, sorted_load_vector
from solver   import SequentialLexSolver, write_solution
from checker  import find_checker, run_checker


# ─────────────────────────────────────────────────────────────────────────────
#  CLI
# ─────────────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="T-ASR Sequential Lex-Minimax Solver")

    grp = p.add_mutually_exclusive_group(required=True)
    grp.add_argument(
        "--instance", "-i",
        help="Préfixe de l'instance (ex: 'toy' → toy-net.json / toy-tm.json / toy-scenario.json)",
    )
    grp.add_argument("--net", help="Chemin vers *-net.json")

    p.add_argument("--tm",       help="Chemin vers *-tm.json (requis avec --net)")
    p.add_argument("--scenario", help="Chemin vers *-scenario.json (requis avec --net)")
    p.add_argument(
        "--data-dir", "-d", default=".",
        help="Dossier contenant les fichiers JSON (défaut : dossier courant)",
    )
    p.add_argument(
        "--out-dir", "-o", default="results",
        help="Dossier de sortie pour la solution (défaut : results/)",
    )
    p.add_argument(
        "--checker-dir", default="../checker",
        help="Dossier contenant le binaire checker Orange (défaut : ../checker)",
    )
    p.add_argument(
        "--max-iter", type=int, default=None,
        help="Nombre max d'itérations lex (défaut : N = |A|·|T|)",
    )
    p.add_argument(
        "--time-limit", type=float, default=300.0,
        help="Limite de temps Gurobi par sous-problème en secondes (défaut : 300)",
    )
    p.add_argument(
        "--mip-gap", type=float, default=1e-4,
        help="Tolérance MIP relative Gurobi (défaut : 1e-4)",
    )
    # ── AJOUT : node-limit ────────────────────────────────────────────────
    p.add_argument(
        "--node-limit", type=int, default=None,
        help=(
            "Limite de nœuds Branch-and-Bound Gurobi par sous-problème "
            "(défaut : illimité). "
            "À utiliser si Out-of-Memory persiste après le fix _useful_segments. "
            "Exemple : --node-limit 5000"
        ),
    )
    # ─────────────────────────────────────────────────────────────────────
    p.add_argument(
        "--verbose-gurobi", action="store_true",
        help="Afficher le log Gurobi",
    )
    return p.parse_args()


# ─────────────────────────────────────────────────────────────────────────────
#  Helpers
# ─────────────────────────────────────────────────────────────────────────────

def resolve_paths(args):
    """Résout les chemins vers les 3 fichiers JSON de l'instance."""
    if args.instance:
        base   = Path(args.data_dir)
        name   = args.instance
        net_f  = base / f"{name}-net.json"
        tm_f   = base / f"{name}-tm.json"
        scen_f = base / f"{name}-scenario.json"
    else:
        net_f  = Path(args.net)
        tm_f   = Path(args.tm)
        scen_f = Path(args.scenario)

    for p in (net_f, tm_f, scen_f):
        if not p.exists():
            print(f"[ERREUR] Fichier introuvable : {p}", file=sys.stderr)
            sys.exit(1)

    return net_f, tm_f, scen_f


def section(title: str):
    print(f"\n{'═' * 60}")
    print(f"  {title}")
    print(f"{'═' * 60}")


def print_z_stars(z_stars, N):
    """Affiche les premières valeurs du vecteur lex avec une barre visuelle."""
    print("\n  Vecteur lex-optimal (premières valeurs) :")
    for k, z in enumerate(z_stars[:20], start=1):
        bar = "█" * int(z * 30)
        print(f"    z*_{k:3d} = {z:.6f}  {bar}")
    if len(z_stars) > 20:
        print(f"    … ({len(z_stars)} valeurs au total, N={N})")


def print_checker_comparison(checker_result, mlu_python):
    """Compare le MLU calculé en Python avec celui du checker officiel."""
    ecart = abs(checker_result.max_load - mlu_python)
    print(f"\n  Comparaison Python vs checker officiel :")
    print(f"    MLU Python              : {mlu_python:.6f}")
    print(f"    MLU checker (C++)       : {checker_result.max_load:.6f}")
    print(f"    Écart absolu            : {ecart:.2e}")
    if ecart < 1e-4:
        print("    → ✓ Cohérent (écart < 1e-4)")
    else:
        print("    → ⚠  Écart suspect — vérifier compute_loads() ou _extract_waypoints()")


# ─────────────────────────────────────────────────────────────────────────────
#  Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    args    = parse_args()
    net_f, tm_f, scen_f = resolve_paths(args)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # ── 1. Chargement de l'instance ───────────────────────────────────────
    section("1. Chargement de l'instance")
    inst = load_instance(net_f, tm_f, scen_f)
    print(f"  {summary(inst)}")
    print(f"  Interventions : {dict(inst.interv)}")
    print(f"  Budgets       : {dict(inst.budget)}")

    # ── 2. Coefficients ECMP ─────────────────────────────────────────────
    section("2. Calcul des coefficients ECMP")
    r = compute_all_splits(inst)

    # ── 3. Baseline (aucun waypoint) ──────────────────────────────────────
    section("3. Baseline — aucun waypoint (routage IGP pur)")
    trivial_wp     = {(d_id, t): []
                      for (d_id, _, _, _) in inst.demands
                      for t in inst.T}
    baseline_loads = compute_loads(trivial_wp, inst, r)
    baseline_vec   = sorted_load_vector(baseline_loads, inst)
    print(f"  MLU baseline = {baseline_vec[0]:.6f}")

    # ── 4. Résolution ─────────────────────────────────────────────────────
    section("4. Solveur séquentiel lex-minimax")

    # Affiche le node_limit utilisé pour traçabilité
    if args.node_limit is not None:
        print(f"  node_limit = {args.node_limit}  "
              f"(Gurobi s'arrêtera après {args.node_limit} nœuds B&B)")
    else:
        print(f"  node_limit = illimité")

    solver = SequentialLexSolver(
        inst, r,
        time_limit = args.time_limit,
        mip_gap    = args.mip_gap,
        max_iter   = args.max_iter,
        node_limit = args.node_limit,   # ← AJOUT
        verbose    = args.verbose_gurobi,
    )
    waypoints, z_stars = solver.solve()

    # ── 5. Résultats ──────────────────────────────────────────────────────
    section("5. Résultats")
    print_z_stars(z_stars, solver.N)

    sol_loads = compute_loads(waypoints, inst, r)
    sol_vec   = sorted_load_vector(sol_loads, inst)
    pct       = (baseline_vec[0] - sol_vec[0]) / max(baseline_vec[0], 1e-12) * 100

    print(f"\n  MLU baseline  = {baseline_vec[0]:.6f}")
    print(f"  MLU solution  = {sol_vec[0]:.6f}")
    print(f"  Réduction MLU = {pct:.2f}%")

    # ── 6. Écriture de la solution ────────────────────────────────────────
    section("6. Écriture de la solution")
    sol_path = out_dir / f"{inst.name}-srpaths.json"
    write_solution(waypoints, inst, sol_path)

    # ── 7. Checker officiel Orange ────────────────────────────────────────
    section("7. Vérification avec le checker officiel Orange")
    checker_bin = find_checker(Path(args.checker_dir))

    if checker_bin is None:
        print(f"  ⚠  Checker non trouvé dans : {args.checker_dir}")
        print(f"  Pour compiler :")
        print(f"    cd {Path(args.checker_dir) / 'src'} && make")
    else:
        print(f"  Binaire trouvé : {checker_bin}")
        checker_result = run_checker(
            net_file      = net_f,
            tm_file       = tm_f,
            scenario_file = scen_f,
            srpaths_file  = sol_path,
            checker_bin   = checker_bin,
        )
        if checker_result:
            print(f"  Faisable : {checker_result.feasible}")
            print_checker_comparison(checker_result, sol_vec[0])

            # Sauvegarder le vecteur officiel pour traçabilité et notebook
            checker_out = out_dir / f"{inst.name}-checker-loads.json"
            with open(checker_out, "w") as f:
                json.dump({
                    "feasible":     checker_result.feasible,
                    "max_load":     checker_result.max_load,
                    "loads_sorted": checker_result.loads_sorted,
                }, f, indent=2)
            print(f"  Vecteur checker sauvegardé → {checker_out}")

    # ── Fin ───────────────────────────────────────────────────────────────
    print(f"\n{'═' * 60}")
    print("  Terminé.")
    print(f"{'═' * 60}\n")


if __name__ == "__main__":
    main()