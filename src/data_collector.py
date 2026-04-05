"""
Data Collector — Record ASL hand landmarks from webcam to CSV.

Usage:
    python src/data_collector.py

Controls:
    A-Z keys  — set the target letter to record
    SPACEBAR  — burst-capture 30 frames of landmarks for the current letter
    q         — quit and save

Saves landmarks to data/raw/{LETTER}.csv (appends if file exists).
Useful for supplementing the Kaggle dataset with your own hand data,
or collecting extra samples for letters the model struggles with.
"""

import os
import sys
import csv
import time
import cv2
import numpy as np
from hand_tracker import HandTracker
from utils import LETTERS, COLORS

# ── Config ───────────────────────────────────────────────────────────
PROJECT_ROOT = os.path.join(os.path.dirname(__file__), "..")
RAW_DIR = os.path.join(PROJECT_ROOT, "data", "raw")
BURST_COUNT = 30  # Frames per spacebar press

# CSV headers
HEADERS = []
for i in range(21):
    HEADERS.extend([f"x{i}", f"y{i}", f"z{i}"])
HEADERS.append("label")


def append_to_csv(filepath, rows, label):
    """Append landmark rows to a CSV file (create with headers if new)."""
    file_exists = os.path.isfile(filepath)

    with open(filepath, "a", newline="") as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow(HEADERS)
        for row in rows:
            writer.writerow(row + [label])


def main():
    os.makedirs(RAW_DIR, exist_ok=True)
    tracker = HandTracker(max_hands=1)
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("ERROR: Cannot open webcam.")
        return

    current_letter = "A"
    recording = False
    burst_frames = []
    burst_target = 0
    total_collected = {letter: 0 for letter in LETTERS}

    # Count existing samples
    for letter in LETTERS:
        csv_path = os.path.join(RAW_DIR, f"{letter}.csv")
        if os.path.isfile(csv_path):
            with open(csv_path, "r") as f:
                total_collected[letter] = max(0, sum(1 for _ in f) - 1)  # minus header

    print("ASL Data Collector")
    print("  A-Z  = set target letter")
    print("  SPACE = burst-capture 30 frames")
    print("  Q     = quit")
    print()

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.flip(frame, 1)
        result = tracker.process(frame)
        frame = tracker.draw_landmarks(frame, result)
        landmarks = tracker.get_landmarks_array(result)

        # Burst recording
        if recording and landmarks is not None:
            burst_frames.append(landmarks.tolist())
            burst_target -= 1
            if burst_target <= 0:
                # Save burst
                csv_path = os.path.join(RAW_DIR, f"{current_letter}.csv")
                append_to_csv(csv_path, burst_frames, current_letter)
                total_collected[current_letter] += len(burst_frames)
                print(f"  ✓ Saved {len(burst_frames)} frames for '{current_letter}' "
                      f"(total: {total_collected[current_letter]})")
                burst_frames = []
                recording = False

        # UI: current letter and count
        h, w, _ = frame.shape
        color = COLORS["red"] if recording else COLORS["green"]
        cv2.putText(frame, f"Letter: {current_letter}", (10, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.2, color, 3)
        cv2.putText(frame, f"Samples: {total_collected[current_letter]}", (10, 80),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, COLORS["white"], 2)

        status = "Hand detected" if landmarks is not None else "No hand - show your hand"
        cv2.putText(frame, status, (10, h - 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, COLORS["yellow"], 1)

        if recording:
            remaining = burst_target
            cv2.putText(frame, f"RECORDING... {remaining} frames left", (10, 120),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, COLORS["red"], 2)

        # Instructions
        cv2.putText(frame, "SPACE=record  A-Z=letter  Q=quit", (10, h - 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, COLORS["gray"], 1)

        cv2.imshow("ASL Data Collector", frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break
        elif key == ord(" ") and not recording:
            if landmarks is not None:
                recording = True
                burst_frames = [landmarks.tolist()]
                burst_target = BURST_COUNT - 1
                print(f"  Recording {BURST_COUNT} frames for '{current_letter}'...")
            else:
                print("  No hand detected — show your hand and try again.")
        elif chr(key).upper() in LETTERS if key < 128 else False:
            new_letter = chr(key).upper()
            if new_letter in LETTERS:
                current_letter = new_letter
                print(f"  Target letter: {current_letter}")

    cap.release()
    cv2.destroyAllWindows()
    tracker.release()

    # Final summary
    print("\n── Collection Summary ──")
    for letter in LETTERS:
        count = total_collected[letter]
        if count > 0:
            print(f"  {letter}: {count} samples")
    total = sum(total_collected.values())
    print(f"  Total: {total} samples")
    print("Done!")


if __name__ == "__main__":
    main()
