"""
Hand Tracker — Webcam + MediaPipe Hands landmark detection and overlay.

Run directly to see a live webcam feed with hand landmarks drawn:
    python src/hand_tracker.py

Controls:
    q  — quit
    p  — print landmarks for the current frame to the console
"""

import cv2
import mediapipe as mp
import numpy as np
from utils import LANDMARK_NAMES, COLORS

# ── MediaPipe setup ──────────────────────────────────────────────────
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles


class HandTracker:
    """Wraps MediaPipe Hands for easy landmark extraction."""

    def __init__(self, max_hands=1, model_complexity=1, min_detection_conf=0.7, min_tracking_conf=0.5):
        self.hands = mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=max_hands,
            model_complexity=model_complexity,
            min_detection_confidence=min_detection_conf,
            min_tracking_confidence=min_tracking_conf,
        )

    def process(self, frame):
        """Run hand detection on a BGR frame.

        Returns:
            results — MediaPipe results object (or None if no hands found).
            The caller can access results.multi_hand_landmarks for drawing.
        """
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        rgb.flags.writeable = False          # small perf boost
        results = self.hands.process(rgb)
        rgb.flags.writeable = True
        return results

    def get_landmarks_array(self, results, hand_index=0):
        """Extract a flat numpy array of shape (63,) from detection results.

        Returns:
            np.ndarray of 63 values (21 landmarks × 3 coords) or None.
        """
        if not results.multi_hand_landmarks:
            return None
        if hand_index >= len(results.multi_hand_landmarks):
            return None

        hand = results.multi_hand_landmarks[hand_index]
        landmarks = []
        for lm in hand.landmark:
            landmarks.extend([lm.x, lm.y, lm.z])
        return np.array(landmarks, dtype=np.float32)

    def draw_landmarks(self, frame, results):
        """Draw all detected hands' landmarks and connections on the frame."""
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                mp_drawing.draw_landmarks(
                    frame,
                    hand_landmarks,
                    mp_hands.HAND_CONNECTIONS,
                    mp_drawing_styles.get_default_hand_landmarks_style(),
                    mp_drawing_styles.get_default_hand_connections_style(),
                )
        return frame

    def release(self):
        """Clean up MediaPipe resources."""
        self.hands.close()


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
    tracker = HandTracker(max_hands=1, model_complexity=1)
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
        results = tracker.process(frame)

        # Draw landmarks on frame
        frame = tracker.draw_landmarks(frame, results)

        # Show FPS-style info
        landmarks = tracker.get_landmarks_array(results)
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
