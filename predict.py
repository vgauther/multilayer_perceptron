#!/usr/bin/env python3

"""
predict.py

Script de prédiction avec le MLP entraîné
"""

import argparse
import csv
import numpy as np

from mlp import MLP
from ml_math import binary_cross_entropy, mean_squared_error


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("model")
    parser.add_argument("dataset")
    args = parser.parse_args()

    mlp, mean, std, arch = MLP.load(args.model)

    X, y_true = [], []
    with open(args.dataset, newline="", encoding="utf-8") as f:
        reader = csv.reader(f)
        next(reader)
        for row in reader:
            y_true.append(int(row[0]))
            X.append([float(v) for v in row[1:]])

    X = (np.array(X) - mean) / std
    y_true = np.array(y_true)

    correct = 0
    bce_sum = 0
    mse_sum = 0

    print("\n=== PRÉDICTIONS ===")

    for i in range(len(X)):
        probs = mlp.forward(X[i:i+1]).flatten()
        y_pred = int(np.argmax(probs))
        p1 = probs[1]

        bce_sum += binary_cross_entropy(y_true[i], p1)
        mse_sum += mean_squared_error(y_true[i], p1)

        status = "OK" if y_pred == y_true[i] else "KO"
        correct += (status == "OK")

        print(
            f"Ligne {i+1:3d} | "
            f"Attendu: {y_true[i]} | "
            f"Prédit: {y_pred} | "
            f"Softmax: [{probs[0]:.4f}, {probs[1]:.4f}] | "
            f"{status}"
        )

    print("\n===================")
    print(f"Accuracy : {correct / len(X) * 100:.2f}%")
    print(f"Mean BCE : {bce_sum / len(X):.4f}")
    print(f"Mean MSE : {mse_sum / len(X):.4f}")


if __name__ == "__main__":
    main()
