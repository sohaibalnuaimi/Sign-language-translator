"""
Hand Tracker — Webcam + MediaPipe Hands landmark detection and overlay.

Run directly to see a live webcam feed with hand landmarks drawn:
    python src/hand_tracker.py

Controls:
    q  — quit
    p  — print landmarks for the current frame to the console
"""

import os
import cv2
import numpy as np
from mediapipe.tasks.python import BaseOptions
from mediapipe.tasks.python.vision import (
    HandLandmarker,
    HandLandmarkerOptions,
    RunningMode,
)
import mediapipe as mp

from utils import LANDMARK_NAMES, COLORS

# ── Path to the hand landmarker model ────────────────────────────────
MODEL_PATH = os.path.join(
    os.path.dirname(__file__), "..", "assets", "hand_landmarker.task"
)

# ── Hand connections for drawing (21 landmarks) ──────────────────────
HAND_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4),        # thumb
    (0, 5), (5, 6), (6, 7), (7, 8),        # index
    (0, 9), (9, 10), (10, 11), (11, 12),   # middle  (fix: wrist to middle MCP)
    (0, 13), (13, 14), (14, 15), (15, 16), # ring    (fix: wrist to ring MCP)
    (0, 17), (17, 18), (18, 19), (19, 20), # pinky   (fix: wrist to pinky MCP)
    (5, 9), (9, 13), (13, 17),             # palm cross-connections
]


class HandTracker:
    """Wraps MediaPipe HandLandmarker (tasks API) for landmark extraction."""

    def __init__(self, max_hands=1, min_detection_conf=0.7, min_tracking_conf=0.5):
        options = HandLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=MODEL_PATH),
            running_mode=RunningMode.VIDEO,
            num_hands=max_hands,
            min_hand_detection_confidence=min_detection_conf,
            min_hand_presence_confidence=min_detection_conf,
            min_tracking_confidence=min_tracking_conf,
        )
        self.landmarker = HandLandmarker.create_from_options(options)
        self._frame_timestamp = 0

    def process(self, frame):
        """Run hand detection on a BGR frame.

        Returns:
            result — HandLandmarkerResult with .hand_landmarks list.
        """
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        self._frame_timestamp += 33  # ~30fps in milliseconds
        result = self.landmarker.detect_for_video(mp_image, self._frame_timestamp)
        return result

    def get_landmarks_array(self, result, hand_index=0):
        """Extract a flat numpy array of shape (63,) from detection result.

        Returns:
            np.ndarray of 63 values (21 landmarks x 3 coords) or None.
        """
        if not result.hand_landmarks:
            return None
        if hand_index >= len(result.hand_landmarks):
            return None

        hand = result.hand_landmarks[hand_index]
        landmarks = []
        for lm in hand:
            landmarks.extend([lm.x, lm.y, lm.z])
        return np.array(landmarks, dtype=np.float32)

    def draw_landmarks(self, frame, result):
        """Draw all detected hands' landmarks and connections on the frame."""
        if not result.hand_landmarks:
            return frame

        h, w, _ = frame.shape
        for hand in result.hand_landmarks:
            # Draw connections
            for start_idx, end_idx in HAND_CONNECTIONS:
                x1, y1 = int(hand[start_idx].x * w), int(hand[start_idx].y * h)
                x2, y2 = int(hand[end_idx].x * w), int(hand[end_idx].y * h)
                cv2.line(frame, (x1, y1), (x2, y2), COLORS["cyan"], 2)

            # Draw landmark dots
            for lm in hand:
                cx, cy = int(lm.x * w), int(lm.y * h)
                cv2.circle(frame, (cx, cy), 5, COLORS["magenta"], -1)
                cv2.circle(frame, (cx, cy), 5, COLORS["white"], 1)

        return frame

    def release(self):
        """Clean up MediaPipe resources."""
        self.landmarker.close()


def print_landmarks(landmarks_array):
    """Pretty-print 21 landmarks with names to the console."""
    if landmarks_array is None:
        print("  No hand detected.")
        return
    for i in range(21):
        x = landmarks_array[i * 3]
        y = landmarks_array[i * 3 + 1]
        z = landmarks_array[i * 3 + 2]
        name = LANDMARK_NAMES[i]
        print(f"  {i:2d} {name:25s}  x={x:.4f}  y={y:.4f}  z={z:.4f}")


# ── Main: run standalone to test ─────────────────────────────────────
def main():
    tracker = HandTracker(max_hands=1)
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("ERROR: Cannot open webcam. Check that a camera is connected.")
        return

    print("Hand Tracker running. Press 'q' to quit, 'p' to print landmarks.")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("ERROR: Failed to read frame from webcam.")
            break

        # Flip horizontally so it feels like a mirror
        frame = cv2.flip(frame, 1)

        # Detect hands
        result = tracker.process(frame)

        # Draw landmarks on frame
        frame = tracker.draw_landmarks(frame, result)

        # Show status
        landmarks = tracker.get_landmarks_array(result)
        status = "Hand detected" if landmarks is not None else "No hand"
        cv2.putText(frame, status, (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, COLORS["green"], 2)

        # Display
        cv2.imshow("ASL Hand Tracker", frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break
        elif key == ord("p"):
            print("\n── Current Landmarks ──")
            print_landmarks(landmarks)
            print()

    cap.release()
    cv2.destroyAllWindows()
    tracker.release()
    print("Hand Tracker stopped.")


if __name__ == "__main__":
    main()
