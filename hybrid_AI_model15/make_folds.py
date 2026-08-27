"""Create the one authoritative fold manifest shared by Qwen and meta models."""
import os
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold
from config import ARTIFACT_DIR, N_FOLDS, SEED, TARGET
from data_loader import prepare_dataset


def assign_folds(df: pd.DataFrame) -> np.ndarray:
    bins = pd.qcut(np.log1p(df[TARGET]), q=10, labels=False, duplicates="drop")
    folds = np.full(len(df), -1, dtype=int)
    splitter = StratifiedGroupKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
    for fold, (_, val_idx) in enumerate(splitter.split(df, bins, df["duplicate_group"])):
        folds[val_idx] = fold
    if (folds < 0).any():
        raise AssertionError("Unassigned rows in fold manifest")
    return folds


def main():
    df = prepare_dataset()
    folds = assign_folds(df)
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    manifest = pd.DataFrame({
        "ticket_id": df["ticket_id"],
        "duplicate_group": df["duplicate_group"],
        "fold": folds,
    })
    target = ARTIFACT_DIR / "folds.csv"
    temporary = ARTIFACT_DIR / "folds.model15.tmp.csv"
    manifest.to_csv(temporary, index=False)
    # Atomic replacement also detaches any legacy hard-link to Model14.
    os.replace(temporary, target)
    print(manifest.groupby("fold").size())


if __name__ == "__main__":
    main()
