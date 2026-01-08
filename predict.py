#!/usr/bin/env python3

"""
predict.py

- Recharge un modèle entraîné avec --layer
- Architecture dynamique
- Sigmoid → ReLU → Softmax
- Affiche les prédictions
- Calcule BCE et MSE globales
"""

import argparse
import csv
import numpy as np

from mlp import MLP
from ml_math import binary_cross_entropy, mean_squared_error


# =========================
# MAIN
# =========================
def main():
    parser = argparse.ArgumentParser(description="MLP Prediction")
    parser.add_argument("model", help="Fichier modèle (.npy)")
    parser.add_argument("dataset", help="Dataset CSV")
    args = parser.parse_args()

    # =========================
    # CHARGEMENT DU MODÈLE
    # =========================
    mlp, mean, std, architecture = MLP.load(args.model)

    print("\n=== ARCHITECTURE DU MODÈLE ===")
    print(" -> ".join(map(str, architecture)))
    print("=============================\n")

    # =========================
    # CHARGEMENT DATASET
    # =========================
    X, y_true = [], []

    with open(args.dataset, newline="", encoding="utf-8") as f:
        reader = csv.reader(f)
        next(reader)
        for row in reader:
            y_true.append(int(row[0]))
            X.append([float(v) for v in row[1:]])

    X = np.array(X)
    y_true = np.array(y_true)

    # Normalisation IDENTIQUE au training
    X = (X - mean) / std

    # =========================
    # PRÉDICTIONS
    # =========================
    correct = 0
    bce_sum = 0.0
    mse_sum = 0.0

    print("=== PRÉDICTIONS ===")

    for i in range(len(X)):
        probs = mlp.forward(X[i:i + 1]).flatten()

        y_pred = int(np.argmax(probs))
        p1 = probs[1]

        # métriques (silencieuses par ligne)
        bce_sum += binary_cross_entropy(y_true[i], p1)
        mse_sum += mean_squared_error(y_true[i], p1)

        if y_pred == y_true[i]:
            correct += 1
            status = "OK"
        else:
            status = "KO"

        print(
            f"Ligne {i+1:3d} | "
            f"Attendu: {y_true[i]} | "
            f"Prédit: {y_pred} | "
            f"Softmax: [{probs[0]:.4f}, {probs[1]:.4f}] | "
            f"{status}"
        )

    n = len(X)

    # =========================
    # RÉSULTATS GLOBAUX
    # =========================
    print("\n===================")
    print(f"Tumeurs bien classées : {correct} / {n}")
    print(f"Accuracy : {correct / n * 100:.2f}%")
    print(f"Mean Binary Cross-Entropy : {bce_sum / n:.4f}")
    print(f"Mean Squared Error       : {mse_sum / n:.4f}")


if __name__ == "__main__":
    main()
