"""
Data Preprocessor — Normalize, merge, and split landmark data.

Usage:
    python src/data_preprocessor.py

Steps:
    1. Load all per-letter CSVs from data/raw/
    2. Apply wrist-origin normalization (position + scale invariance)
    3. Merge into a single dataset
    4. Stratified 80/20 train/test split
    5. Save to data/processed/train.csv and data/processed/test.csv
"""

import os
import sys
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

# ── Paths ────────────────────────────────────────────────────────────
PROJECT_ROOT = os.path.join(os.path.dirname(__file__), "..")
RAW_DIR = os.path.join(PROJECT_ROOT, "data", "raw")
PROCESSED_DIR = os.path.join(PROJECT_ROOT, "data", "processed")


def normalize_wrist_origin(row_landmarks):
    """Normalize 63 landmark values: subtract wrist, divide by max distance.

    Args:
        row_landmarks: numpy array of shape (63,) — 21 landmarks × (x, y, z)

    Returns:
        numpy array of shape (63,) — normalized landmarks.
        Wrist is at (0, 0, 0), all values scaled by max fingertip distance.
    """
    # Reshape to (21, 3)
    points = row_landmarks.reshape(21, 3)

    # Step 1: Subtract wrist (landmark 0) → position invariance
    wrist = points[0].copy()
    points = points - wrist

    # Step 2: Divide by max distance from wrist → scale invariance
    # Use fingertip landmarks (4, 8, 12, 16, 20) for max distance
    distances = np.linalg.norm(points, axis=1)
    max_dist = np.max(distances)

    if max_dist > 0:
        points = points / max_dist

    # Flatten back to (63,)
    return points.flatten()


def load_raw_csvs():
    """Load all CSV files from data/raw/ and merge into a single DataFrame."""
    if not os.path.isdir(RAW_DIR):
        print(f"ERROR: Raw data directory not found: {RAW_DIR}")
        print("Run kaggle_extractor.py or data_collector.py first.")
        sys.exit(1)

    csv_files = sorted([f for f in os.listdir(RAW_DIR) if f.endswith(".csv")])
    if not csv_files:
        print(f"ERROR: No CSV files found in {RAW_DIR}")
        sys.exit(1)

    print(f"Found {len(csv_files)} CSV files in {RAW_DIR}")

    dfs = []
    for csv_file in csv_files:
        path = os.path.join(RAW_DIR, csv_file)
        df = pd.read_csv(path)
        dfs.append(df)
        label = csv_file.replace(".csv", "")
        print(f"  {label:<12}: {len(df):>6,} samples")

    merged = pd.concat(dfs, ignore_index=True)
    print(f"\nTotal samples: {len(merged):,}")
    return merged


def clean_data(df):
    """Remove rows with missing/invalid values."""
    original_len = len(df)

    # Drop rows with any NaN
    df = df.dropna()

    # Get feature columns (all except 'label')
    feature_cols = [c for c in df.columns if c != "label"]

    # Drop rows where all landmarks are zero (bad detection)
    all_zero_mask = (df[feature_cols] == 0).all(axis=1)
    df = df[~all_zero_mask]

    removed = original_len - len(df)
    if removed > 0:
        print(f"Cleaned: removed {removed:,} invalid rows ({len(df):,} remaining)")
    else:
        print(f"Cleaned: all {len(df):,} rows valid, none removed")

    return df.reset_index(drop=True)


def apply_normalization(df):
    """Apply wrist-origin normalization to all rows."""
    feature_cols = [c for c in df.columns if c != "label"]
    labels = df["label"].values

    print("Applying wrist-origin normalization...", end=" ", flush=True)

    features = df[feature_cols].values.astype(np.float32)
    normalized = np.array([normalize_wrist_origin(row) for row in features])

    # Rebuild DataFrame
    result = pd.DataFrame(normalized, columns=feature_cols)
    result["label"] = labels

    # Verify wrist is at origin
    wrist_max = max(
        result["x0"].abs().max(),
        result["y0"].abs().max(),
        result["z0"].abs().max(),
    )
    print(f"done (wrist max deviation: {wrist_max:.2e})")

    return result


def split_and_save(df, test_size=0.2):
    """Stratified train/test split and save to CSV."""
    os.makedirs(PROCESSED_DIR, exist_ok=True)

    feature_cols = [c for c in df.columns if c != "label"]
    X = df[feature_cols]
    y = df["label"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=42, stratify=y
    )

    # Reassemble DataFrames
    train_df = pd.concat([X_train, y_train], axis=1).reset_index(drop=True)
    test_df = pd.concat([X_test, y_test], axis=1).reset_index(drop=True)

    # Save
    train_path = os.path.join(PROCESSED_DIR, "train.csv")
    test_path = os.path.join(PROCESSED_DIR, "test.csv")
    train_df.to_csv(train_path, index=False)
    test_df.to_csv(test_path, index=False)

    print(f"\nSaved:")
    print(f"  Train: {train_path} ({len(train_df):,} samples)")
    print(f"  Test:  {test_path} ({len(test_df):,} samples)")

    # Class distribution
    print(f"\nClass distribution (train):")
    train_counts = train_df["label"].value_counts().sort_index()
    print(f"  {'Class':<12} {'Train':>8} {'Test':>8}")
    print(f"  {'-'*12} {'-'*8} {'-'*8}")
    test_counts = test_df["label"].value_counts().sort_index()
    for label in train_counts.index:
        tr = train_counts.get(label, 0)
        te = test_counts.get(label, 0)
        print(f"  {label:<12} {tr:>8,} {te:>8,}")

    return train_df, test_df


def main():
    print("=" * 50)
    print("ASL Data Preprocessor")
    print("=" * 50)
    print()

    # Step 1: Load
    df = load_raw_csvs()
    print()

    # Step 2: Clean
    df = clean_data(df)
    print()

    # Step 3: Normalize
    df = apply_normalization(df)
    print()

    # Step 4: Split and save
    train_df, test_df = split_and_save(df)

    print("\nDone! Next step: run model_trainer.py to train a classifier.")


if __name__ == "__main__":
    main()
