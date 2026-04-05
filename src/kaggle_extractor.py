"""
Kaggle ASL Alphabet Extractor — Batch extract MediaPipe landmarks from images.

Usage:
    python src/kaggle_extractor.py

Expects the Kaggle "ASL Alphabet" dataset extracted to:
    data/kaggle/asl_alphabet_train/asl_alphabet_train/{A,B,...,Z,del,nothing,space}/

Each folder contains ~3,000 images (200×200px).
This script runs MediaPipe HandLandmarker on every image, extracts 63 landmark
features (21 × 3), and saves them to data/raw/{LABEL}.csv.

Images where no hand is detected are skipped and counted.
"""

import os
import sys
import csv
import time
import cv2
import numpy as np
from mediapipe.tasks.python import BaseOptions
from mediapipe.tasks.python.vision import (
    HandLandmarker,
    HandLandmarkerOptions,
    RunningMode,
)
import mediapipe as mp

# ── Paths ────────────────────────────────────────────────────────────
PROJECT_ROOT = os.path.join(os.path.dirname(__file__), "..")
MODEL_PATH = os.path.join(PROJECT_ROOT, "assets", "hand_landmarker.task")

# Try both common extraction structures
KAGGLE_DIR_OPTIONS = [
    os.path.join(PROJECT_ROOT, "data", "kaggle", "asl_alphabet_train", "asl_alphabet_train"),
    os.path.join(PROJECT_ROOT, "data", "kaggle", "asl_alphabet_train"),
    os.path.join(PROJECT_ROOT, "data", "kaggle"),
]

RAW_DIR = os.path.join(PROJECT_ROOT, "data", "raw")

# Column headers: x0, y0, z0, x1, y1, z1, ..., x20, y20, z20, label
HEADERS = []
for i in range(21):
    HEADERS.extend([f"x{i}", f"y{i}", f"z{i}"])
HEADERS.append("label")


def find_kaggle_dir():
    """Find the correct Kaggle dataset directory."""
    for path in KAGGLE_DIR_OPTIONS:
        abs_path = os.path.abspath(path)
        if os.path.isdir(abs_path):
            # Check if it contains letter folders
            subfolders = [f for f in os.listdir(abs_path) if os.path.isdir(os.path.join(abs_path, f))]
            if any(f in subfolders for f in ["A", "B", "C"]):
                return abs_path
    return None


def create_landmarker():
    """Create a MediaPipe HandLandmarker in IMAGE mode for static images."""
    options = HandLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=MODEL_PATH),
        running_mode=RunningMode.IMAGE,
        num_hands=1,
        min_hand_detection_confidence=0.5,
        min_hand_presence_confidence=0.5,
    )
    return HandLandmarker.create_from_options(options)


def extract_landmarks(landmarker, image_path):
    """Extract 63 landmark values from a single image.

    Returns:
        list of 63 floats, or None if no hand detected.
    """
    img = cv2.imread(image_path)
    if img is None:
        return None

    rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
    result = landmarker.detect(mp_image)

    if not result.hand_landmarks:
        return None

    hand = result.hand_landmarks[0]
    landmarks = []
    for lm in hand:
        landmarks.extend([lm.x, lm.y, lm.z])
    return landmarks


def process_class(landmarker, class_dir, label, output_csv, max_images=None):
    """Process all images in a class folder and write landmarks to CSV.

    Returns:
        (success_count, skip_count)
    """
    image_files = sorted([
        f for f in os.listdir(class_dir)
        if f.lower().endswith((".jpg", ".jpeg", ".png", ".bmp"))
    ])

    if max_images:
        image_files = image_files[:max_images]

    success = 0
    skipped = 0

    with open(output_csv, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(HEADERS)

        for img_name in image_files:
            img_path = os.path.join(class_dir, img_name)
            landmarks = extract_landmarks(landmarker, img_path)

            if landmarks is None:
                skipped += 1
                continue

            writer.writerow(landmarks + [label])
            success += 1

    return success, skipped


def main():
    # Find the Kaggle dataset
    kaggle_dir = find_kaggle_dir()
    if kaggle_dir is None:
        print("ERROR: Kaggle dataset not found!")
        print()
        print("Please download the ASL Alphabet dataset from Kaggle and extract it:")
        print("  1. Go to: https://www.kaggle.com/datasets/grassknoted/asl-alphabet")
        print("  2. Download and extract to: data/kaggle/")
        print("  3. Expected structure: data/kaggle/asl_alphabet_train/asl_alphabet_train/{A,B,...}/")
        sys.exit(1)

    print(f"Found Kaggle dataset at: {kaggle_dir}")

    # Get all class folders
    class_folders = sorted([
        f for f in os.listdir(kaggle_dir)
        if os.path.isdir(os.path.join(kaggle_dir, f))
    ])
    print(f"Found {len(class_folders)} classes: {', '.join(class_folders)}")
    print()

    # Create output directory
    os.makedirs(RAW_DIR, exist_ok=True)

    # Create landmarker
    print("Loading MediaPipe HandLandmarker...")
    landmarker = create_landmarker()

    # Optional: limit images per class for testing (set to None for all)
    max_images = None
    if "--test" in sys.argv:
        max_images = 50
        print(f"TEST MODE: Processing only {max_images} images per class\n")

    # Process each class
    total_success = 0
    total_skipped = 0
    start_time = time.time()
    results_summary = []

    for i, class_name in enumerate(class_folders):
        class_dir = os.path.join(kaggle_dir, class_name)
        label = class_name  # A, B, ..., Z, del, nothing, space

        # Count images
        image_count = len([
            f for f in os.listdir(class_dir)
            if f.lower().endswith((".jpg", ".jpeg", ".png", ".bmp"))
        ])

        output_csv = os.path.join(RAW_DIR, f"{label}.csv")

        print(f"[{i+1}/{len(class_folders)}] Processing '{label}' ({image_count} images)...", end=" ", flush=True)

        class_start = time.time()
        success, skipped = process_class(landmarker, class_dir, label, output_csv, max_images)
        class_time = time.time() - class_start

        total_success += success
        total_skipped += skipped
        results_summary.append((label, success, skipped))

        print(f"✓ {success} extracted, {skipped} skipped ({class_time:.1f}s)")

    # Summary
    elapsed = time.time() - start_time
    print()
    print("=" * 60)
    print("EXTRACTION COMPLETE")
    print("=" * 60)
    print(f"  Total extracted:  {total_success:,}")
    print(f"  Total skipped:    {total_skipped:,}")
    print(f"  Success rate:     {total_success/(total_success+total_skipped)*100:.1f}%")
    print(f"  Time elapsed:     {elapsed:.0f}s ({elapsed/60:.1f} min)")
    print(f"  Output directory: {os.path.abspath(RAW_DIR)}")
    print()

    # Per-class breakdown
    print("Per-class breakdown:")
    print(f"  {'Class':<12} {'Extracted':>10} {'Skipped':>10} {'Rate':>8}")
    print(f"  {'-'*12} {'-'*10} {'-'*10} {'-'*8}")
    for label, success, skipped in results_summary:
        total = success + skipped
        rate = (success / total * 100) if total > 0 else 0
        print(f"  {label:<12} {success:>10,} {skipped:>10,} {rate:>7.1f}%")

    landmarker.close()
    print("\nDone! Next step: run data_preprocessor.py to normalize and split the data.")


if __name__ == "__main__":
    main()
