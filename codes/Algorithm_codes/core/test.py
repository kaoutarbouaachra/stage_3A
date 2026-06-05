"""
test_ecmp.py — membre du package `core`. Valide le backend ECMP nt4python.

  Test B (toujours)  : nt4python == from-scratch, toutes paires & periodes.
  Test A (toy seult) : nt4python reproduit la Table 2 du sujet.

A placer dans core/ et a lancer EN MODULE depuis Algorithm_codes/ :
    cd .../codes/Algorithm_codes
    python -m core.test_ecmp toy
    python -m core.test_ecmp setA-01
    python -m core.test_ecmp /chemin/vers/un/dossier/instance   # aussi accepte

Le nom (toy, setA-01, ...) est resolu automatiquement dans l'arborescence.
"""
import sys
from pathlib import Path

from .instance import load_instance
from .ecmp_from_scratch import compute_all_splits as scratch_splits
from .nt_adapter import compute_all_splits_ntlib as nt_splits

GOLDEN_TOY = {0: 1.00, 2: 0.50, 3: 0.50, 6: 0.75, 18: 0.25, 9: 0.25, 21: 0.25}
TOL = 1e-9
_SUFFIX = "-net.json"


def find_instance(arg):
    """arg = un nom (toy, setA-01) OU un dossier. Renvoie (dossier, prefixe)."""
    p = Path(arg)
    if p.is_dir():
        nets = sorted(p.glob("*-net.json"))
        if nets:
            return p, nets[0].name[:-len(_SUFFIX)]
        raise SystemExit(f"aucun *-net.json dans {p}")

    name = arg
    # codes/ = core -> Algorithm_codes -> codes
    codes = Path(__file__).resolve().parents[2]
    candidates = [
        codes / f"{name}_instance",     # toy      -> codes/toy_instance
        codes / name,                   # direct   -> codes/<name>
        codes / "setA" / name,          # setA-01  -> codes/setA/setA-01
    ]
    for c in candidates:
        if (c / f"{name}{_SUFFIX}").exists():
            return c, name
    # dernier recours : recherche recursive
    hits = list(codes.rglob(f"{name}{_SUFFIX}"))
    if hits:
        return hits[0].parent, name
    raise SystemExit(f"instance '{arg}' introuvable sous {codes}")


def main():
    arg = sys.argv[1] if len(sys.argv) > 1 else "toy"
    base, name = find_instance(arg)
    print(f"Instance : {name}   (dossier : {base})")

    inst = load_instance(base / f"{name}-net.json",
                         base / f"{name}-tm.json",
                         base / f"{name}-scenario.json")

    print("Calcul de r (from scratch)...")
    a = scratch_splits(inst)
    print("Calcul de r (nt4python)...")
    b = nt_splits(inst)

    # ---- Test B : equivalence (toute instance) ---------------------------
    print("\n=== Test B : nt4python vs from-scratch (toutes paires, toutes periodes) ===")
    total, mism, worst = 0, 0, 0.0
    for t in inst.T:
        for k in sorted(set(a[t]) | set(b[t])):
            total += 1
            va, vb = a[t].get(k, 0.0), b[t].get(k, 0.0)
            d = abs(va - vb)
            worst = max(worst, d)
            if d > TOL:
                mism += 1
                if mism <= 30:
                    print(f"  ECART t={t} {k}: scratch={va:.6f}  nt={vb:.6f}")
    B_ok = (mism == 0)
    print(f"  {total} coefficients compares, {mism} ecarts, ecart max = {worst:.2e}")
    print(" -> Test B :", "PASS" if B_ok else "FAIL")

    # ---- Test A : Table 2 (toy uniquement) -------------------------------
    A_ok = True
    if name == "toy":
        print("\n=== Test A : nt4python vs Table 2 du sujet (toy, t=0, 0->5) ===")
        got = {aid: c for (u, v, aid), c in b[0].items() if u == 0 and v == 5}
        print(" arc   nt4python   golden    ok")
        for aid in sorted(set(got) | set(GOLDEN_TOY)):
            g, e = got.get(aid, 0.0), GOLDEN_TOY.get(aid, 0.0)
            ok = abs(g - e) <= TOL
            A_ok = A_ok and ok
            print(f" {aid:>3}    {g:7.4f}   {e:7.4f}    {'OK' if ok else 'XX'}")
        if set(got) - set(GOLDEN_TOY):
            A_ok = False
            print(f" arcs inattendus : {sorted(set(got) - set(GOLDEN_TOY))}")
        print(" -> Test A :", "PASS" if A_ok else "FAIL")
    else:
        print("\n(Test A = Table 2 : ignore, specifique a toy)")

    print("\n" + "=" * 52)
    print(" TOUT PASSE" if (A_ok and B_ok) else " ECHECS")
    print("=" * 52)
    sys.exit(0 if (A_ok and B_ok) else 1)


if __name__ == "__main__":
    main()