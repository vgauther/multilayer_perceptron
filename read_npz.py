#!/usr/bin/env python3

"""
read_npz.py

Programme de lecture d'un modèle sauvegardé au format .npz.
Affiche :
- l'architecture du réseau
- les poids et biais de chaque couche
"""

import argparse
import numpy as np


def read_npz(file_path):
    print(f"[INFO] Lecture du fichier : {file_path}\n")

    # Chargement du fichier npz
    data = np.load(file_path, allow_pickle=True)

    # Affichage des clés disponibles
    print("[INFO] Contenu du fichier NPZ :")
    for key in data.files:
        print(f" - {key}")
    print()

    # Lecture de l'architecture
    architecture = data["architecture"]
    weights = data["weights"]
    biases = data["biases"]

    print("=== Architecture du réseau ===")
    print(" -> ".join(map(str, architecture)))
    print("==============================\n")

    # Affichage des poids et biais couche par couche
    for i in range(len(weights)):
        print(f"--- Couche {i + 1} ---")

        print("Poids (weights) :")
        print(weights[i])
        print(f"Shape : {weights[i].shape}\n")

        print("Biais (bias) :")
        print(biases[i])
        print(f"Shape : {biases[i].shape}")

        print("---------------------------\n")


def main():
    parser = argparse.ArgumentParser(description="Lire et afficher un fichier .npz")
    parser.add_argument(
        "model",
        type=str,
        help="Chemin vers le fichier .npz du modèle"
    )
    args = parser.parse_args()

    read_npz(args.model)


if __name__ == "__main__":
    main()
