import argparse
import csv
import numpy as np

from mlp import MLP
from ml_math import categorical_cross_entropy


def binary_cross_entropy(y, p):
    eps = 1e-15
    p = np.clip(p, eps, 1 - eps)
    return -(y * np.log(p) + (1 - y) * np.log(1 - p))


def mean_squared_error(y, p):
    return (y - p) ** 2


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("model")
    parser.add_argument("dataset")
    args = parser.parse_args()

    # ===== Chargement du modèle =====
    mlp, mean, std, architecture = MLP.load(args.model)

    print("\n=== ARCHITECTURE ===")
    print(" -> ".join(map(str, architecture)))
    print("====================\n")

    # ===== Chargement du dataset =====
    X, y_true = [], []
    with open(args.dataset) as f:
        reader = csv.reader(f)
        next(reader)
        for row in reader:
            y_true.append(int(row[0]))
            X.append([float(v) for v in row[1:]])

    X = np.array(X)
    y_true = np.array(y_true)

    # Normalisation identique au training
    X = (X - mean) / std

    # One-hot pour la loss catégorielle finale
    y_one_hot = np.zeros((len(y_true), 2))
    y_one_hot[np.arange(len(y_true)), y_true] = 1

    # ===== Prédictions =====
    preds = mlp.forward(X)

    correct = 0
    bce_losses = []
    mse_losses = []

    print("=== PRÉDICTIONS ===")

    for i in range(len(X)):
        p0, p1 = preds[i]
        y_pred = int(np.argmax(preds[i]))
        expected = y_true[i]

        # métriques sur la classe positive
        bce_losses.append(binary_cross_entropy(expected, p1))
        mse_losses.append(mean_squared_error(expected, p1))

        if y_pred == expected:
            status = "OK"
            correct += 1
        else:
            status = "KO"

        print(
            f"Ligne {i+1:3d} | "
            f"Attendu: {expected} | "
            f"Prédit: {y_pred} | "
            f"Softmax: [{p0:.4f}, {p1:.4f}] | "
            f"{status}"
        )

    # ===== Résultats globaux =====
    accuracy = correct / len(y_true)
    cat_loss = np.mean(categorical_cross_entropy(y_one_hot, preds))
    mean_bce = np.mean(bce_losses)
    mean_mse = np.mean(mse_losses)

    print("\n===================")
    print(f"Tumeurs bien classées : {correct} / {len(y_true)}")
    print(f"Accuracy : {accuracy * 100:.2f}%")
    print(f"Categorical Cross-Entropy : {cat_loss:.4f}")
    print(f"Binary Cross-Entropy      : {mean_bce:.4f}")
    print(f"Mean Squared Error        : {mean_mse:.4f}")


if __name__ == "__main__":
    main()
