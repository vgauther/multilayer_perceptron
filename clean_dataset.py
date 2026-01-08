#!/usr/bin/env python3

import argparse
import pandas as pd
from sklearn.model_selection import train_test_split

def main(input_csv, seed):
    print(f"[INFO] Chargement du dataset : {input_csv}")
    df = pd.read_csv(input_csv, header=None)

    print(f"[INFO] Shape initiale : {df.shape}")

    # Suppression première colonne (ID)
    first_col = df.columns[0]
    df = df.drop(columns=[first_col])
    print(f"[INFO] Suppression de la colonne ID : {first_col}")

    # Encodage du label M/B -> 1/0
    label_col = df.columns[0]
    df[label_col] = df[label_col].map({"M": 1, "B": 0})
    print("[INFO] Encodage du label : M -> 1 | B -> 0")

    # Split 80 / 20 avec random + seed
    train_df, predict_df = train_test_split(
        df,
        test_size=0.2,
        shuffle=True,
        random_state=seed
    )

    print(f"[INFO] Seed utilisée : {seed}")
    print(f"[INFO] Dataset train : {train_df.shape}")
    print(f"[INFO] Dataset predict : {predict_df.shape}")

    # Sauvegarde
    train_df.to_csv("dataset_train.csv", index=False, header=False)
    predict_df.to_csv("dataset_predict.csv", index=False, header=False)

    print("[INFO] Fichiers générés :")
    print("  - dataset_train.csv")
    print("  - dataset_predict.csv")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset", help="CSV d'entrée")
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Seed pour la séparation aléatoire"
    )
    args = parser.parse_args()

    main(args.dataset, args.seed)
