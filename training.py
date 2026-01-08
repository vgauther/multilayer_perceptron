#!/usr/bin/env python3

"""
training.py

Script d'entraînement du MLP
"""

import argparse
import csv
import numpy as np
import matplotlib.pyplot as plt

from mlp import MLP
from ml_math import categorical_cross_entropy


def load_dataset(path):
    X, y = [], []

    with open(path, newline="", encoding="utf-8") as f:
        reader = csv.reader(f)
        next(reader)
        for row in reader:
            y.append(int(row[0]))
            X.append([float(v) for v in row[1:]])

    X = np.array(X)
    y = np.array(y)

    y_one_hot = np.zeros((len(y), 2))
    y_one_hot[np.arange(len(y)), y] = 1

    mean = X.mean(axis=0)
    std = X.std(axis=0) + 1e-8
    X = (X - mean) / std

    return X, y_one_hot, y, mean, std


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--lr", type=float, default=0.01)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--save-model", default="model.npy")
    args = parser.parse_args()

    X, y, y_lbl, mean, std = load_dataset(args.dataset)

    split = int(0.8 * len(X))
    X_train, X_val = X[:split], X[split:]
    y_train, y_val = y[:split], y[split:]
    y_train_lbl, y_val_lbl = y_lbl[:split], y_lbl[split:]

    layers_config = [
        (X.shape[1], 16, "relu"),
        (16, 16, "relu"),
        (16, 2, "softmax")
    ]

    mlp = MLP(layers_config)

    train_losses, val_losses = [], []
    train_accs, val_accs = [], []

    print("=== DÉBUT DU TRAINING ===")

    for epoch in range(args.epochs):
        perm = np.random.permutation(len(X_train))
        X_train, y_train, y_train_lbl = X_train[perm], y_train[perm], y_train_lbl[perm]

        loss_sum = 0
        correct = 0

        for i in range(0, len(X_train), args.batch_size):
            Xb = X_train[i:i+args.batch_size]
            yb = y_train[i:i+args.batch_size]
            yb_lbl = y_train_lbl[i:i+args.batch_size]

            y_pred = mlp.forward(Xb)
            loss = np.mean(categorical_cross_entropy(yb, y_pred))
            loss_sum += loss

            mlp.backward(yb, y_pred, args.lr)
            correct += np.sum(np.argmax(y_pred, axis=1) == yb_lbl)

        train_loss = loss_sum / (len(X_train) / args.batch_size)
        train_acc = correct / len(X_train)

        y_val_pred = mlp.forward(X_val)
        val_loss = np.mean(categorical_cross_entropy(y_val, y_val_pred))
        val_acc = np.mean(np.argmax(y_val_pred, axis=1) == y_val_lbl)

        train_losses.append(train_loss)
        val_losses.append(val_loss)
        train_accs.append(train_acc)
        val_accs.append(val_acc)

        print(
            f"Epoch {epoch+1:3d}/{args.epochs} | "
            f"loss: {train_loss:.4f} | val_loss: {val_loss:.4f} | "
            f"acc: {train_acc*100:.2f}% | val_acc: {val_acc*100:.2f}%"
        )

    mlp.save(args.save_model, mean, std)

    plt.figure()
    plt.plot(train_losses, label="Train Loss")
    plt.plot(val_losses, label="Val Loss")
    plt.legend()
    plt.show()

    plt.figure()
    plt.plot(train_accs, label="Train Acc")
    plt.plot(val_accs, label="Val Acc")
    plt.legend()
    plt.show()


if __name__ == "__main__":
    main()
