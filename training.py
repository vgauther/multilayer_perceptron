import argparse
import csv
import numpy as np
import matplotlib.pyplot as plt

from mlp import MLP
from ml_math import categorical_cross_entropy


# ===== Chargement du dataset =====
def load_dataset(path):
    X, y = [], []

    with open(path) as f:
        reader = csv.reader(f)
        next(reader)
        for row in reader:
            y.append(int(row[0]))          # 0 ou 1
            X.append([float(v) for v in row[1:]])

    X = np.array(X)
    y = np.array(y)

    # One-hot encoding
    y_one_hot = np.zeros((len(y), 2))
    y_one_hot[np.arange(len(y)), y] = 1

    return X, y_one_hot, y


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset")
    parser.add_argument("--layer", type=int, nargs="+", required=True)
    parser.add_argument("--activation", choices=["sigmoid", "relu"], default="sigmoid")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--lr", type=float, default=0.01)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--save-model", default="model.npy")
    args = parser.parse_args()

    # ===== Chargement =====
    X, y_one_hot, y_labels = load_dataset(args.dataset)

    # ===== Split 80 / 20 (SANS mélange) =====
    split = int(0.8 * len(X))
    X_train, X_val = X[:split], X[split:]
    y_train, y_val = y_one_hot[:split], y_one_hot[split:]
    y_train_lbl, y_val_lbl = y_labels[:split], y_labels[split:]

    # ===== Normalisation =====
    mean = X_train.mean(axis=0)
    std = X_train.std(axis=0) + 1e-8

    X_train = (X_train - mean) / std
    X_val = (X_val - mean) / std

    # ===== Création du réseau =====
    layers = [(X.shape[1], args.layer[0], args.activation)]
    for i in range(len(args.layer) - 1):
        layers.append((args.layer[i], args.layer[i + 1], args.activation))
    layers.append((args.layer[-1], 2, "softmax"))

    mlp = MLP(layers)

    # ===== Tracking =====
    train_losses, val_losses = [], []
    train_accs, val_accs = [], []

    print("\n=== DÉBUT DU TRAINING ===")

    # ===== Training =====
    for epoch in range(args.epochs):
        losses = []
        correct = 0

        for i in range(0, len(X_train), args.batch_size):
            xb = X_train[i:i + args.batch_size]
            yb = y_train[i:i + args.batch_size]
            yb_lbl = y_train_lbl[i:i + args.batch_size]

            # Forward
            y_pred = mlp.forward(xb)

            # Loss
            loss = np.mean(categorical_cross_entropy(yb, y_pred))
            losses.append(loss)

            # Backward
            mlp.backward(yb, y_pred, args.lr)

            # Accuracy
            correct += np.sum(np.argmax(y_pred, axis=1) == yb_lbl)

        # Metrics train
        train_losses.append(np.mean(losses))
        train_accs.append(correct / len(X_train))

        # ===== Validation =====
        val_pred = mlp.forward(X_val)
        val_losses.append(np.mean(categorical_cross_entropy(y_val, val_pred)))
        val_accs.append(np.mean(np.argmax(val_pred, axis=1) == y_val_lbl))

        print(
            f"Epoch {epoch+1:3d}/{args.epochs} | "
            f"loss: {train_losses[-1]:.4f} | "
            f"val_loss: {val_losses[-1]:.4f} | "
            f"acc: {train_accs[-1]*100:.2f}% | "
            f"val_acc: {val_accs[-1]*100:.2f}%"
        )

    # ===== Sauvegarde =====
    mlp.save(args.save_model, mean, std)
    print(f"\n[INFO] Modèle sauvegardé dans : {args.save_model}")

    # ===== Graphs =====
    plt.figure()
    plt.plot(train_losses, label="Train Loss")
    plt.plot(val_losses, label="Validation Loss")
    plt.legend()
    plt.grid()
    plt.show()

    plt.figure()
    plt.plot(train_accs, label="Train Accuracy")
    plt.plot(val_accs, label="Validation Accuracy")
    plt.legend()
    plt.grid()
    plt.show()


if __name__ == "__main__":
    main()
