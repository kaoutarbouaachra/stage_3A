#!/usr/bin/python3
# coding: utf-8
"""
Author: Yannick

Self-contained file
"""

import os
import re
import glob

# Répertoire contenant les fichiers (répertoire courant par défaut)
REPERTOIRE = "."
FICHIER_SORTIE = "data.txt"


def extraire_infos(chemin_fichier):
    """
    Extrait les informations d'un fichier setAP-*.txt.
    Retourne un dictionnaire avec les infos trouvées, ou None en cas d'échec.
    """
    infos = {
        "nom_instance": None,
        "N": None,
        "nb_iterations": None,
        "duree": None,
    }

    try:
        with open(chemin_fichier, "r", encoding="utf-8") as f:
            contenu = f.read()

        # --- Nom de l'instance et valeur de N ---
        # Exemple : "setAP-01 | 10 nodes, 50 arcs, 10 demands, 2 timesteps, N=100"
        match_instance = re.search(
            r"(setAP-\d+)\s*\|.*?N=(\d+)",
            contenu
        )
        if match_instance:
            infos["nom_instance"] = match_instance.group(1)
            infos["N"] = int(match_instance.group(2))

        # --- Nombre d'itérations et durée ---
        # Exemple : "Done.  37 iterations in 20.9s"
        match_iterations = re.search(
            r"Done\.\s+(\d+)\s+ranks\s+in\s+([\d.]+)s",
            contenu
        )
        if match_iterations:
            infos["nb_iterations"] = int(match_iterations.group(1))
            infos["duree"] = float(match_iterations.group(2))

    except Exception as e:
        print(f"  [ERREUR] Impossible de lire {chemin_fichier} : {e}")
        return None

    # Vérification que toutes les infos ont été trouvées
    if any(v is None for v in infos.values()):
        print(f"  [AVERTISSEMENT] Informations incomplètes dans : {chemin_fichier}")
        print(f"    → Infos extraites : {infos}")

    return infos


def main():
    # Recherche de tous les fichiers setAP-*.txt
    pattern = os.path.join(REPERTOIRE, "setAP-*.txt")
    # fichiers = sorted(glob.glob(pattern))
    fichiers = sorted(glob.glob(pattern),
                      key=lambda f: int(re.search(r"setAP-(\d+)", os.path.basename(f)).group(1)))

    if not fichiers:
        print(f"Aucun fichier correspondant à 'setAP-*.txt' trouvé dans '{REPERTOIRE}'.")
        return

    print(f"{len(fichiers)} fichier(s) trouvé(s).\n")

    lignes_sortie = []
    # En-tête du fichier de sortie
    entete = f"{'Instance':<20} {'N':>6} {'Itérations':>12} {'Durée (s)':>12}"
    lignes_sortie.append(entete)
    lignes_sortie.append("-" * len(entete))

    for chemin in fichiers:
        nom_fichier = os.path.basename(chemin)
        print(f"Traitement de : {nom_fichier}")

        infos = extraire_infos(chemin)

        if infos is None:
            print(f"  → Fichier ignoré.\n")
            continue

        ligne = (
            f"{infos['nom_instance'] or 'N/A':<20}"
            f"{str(infos['N']) if infos['N'] is not None else 'N/A':>6}"
            f"{str(infos['nb_iterations']) if infos['nb_iterations'] is not None else 'N/A':>12}"
            f"{str(infos['duree']) if infos['duree'] is not None else 'N/A':>12}"
        )
        lignes_sortie.append(ligne)
        print(f"  → {ligne}\n")

    # Écriture dans data.txt
    with open(FICHIER_SORTIE, "w", encoding="utf-8") as f:
        f.write("\n".join(lignes_sortie) + "\n")

    print(f"\nRésultats écrits dans : {FICHIER_SORTIE}")


if __name__ == "__main__":
    main()


# parcourir tous les fichiers dans le répertoire de type setAP-*.txt
# pour chaque fichier, trouver les infos suivantes :
#  - nom de l'instance (e.g. setAP-01)
# - valeur de N (e.g. 100)
# - nombre d'itérations réalisées (e.g. 37)
# - durée (e.g. 20.9)
# ecrire les infos dans data.txt, une ligne par instance
# exemple format du fichier setAP-01-srpaths.json.txt:
"""


"""


