"""
Shared constants and helpers for the ASL Alphabet Translator.
"""

# ── Hand Landmark Constants ──────────────────────────────────────────
LANDMARK_COUNT = 21          # MediaPipe Hands outputs 21 keypoints
COORDS_PER_LANDMARK = 3      # x, y, z per landmark
FEATURE_COUNT = LANDMARK_COUNT * COORDS_PER_LANDMARK  # 63 features

# ── ASL Alphabet ─────────────────────────────────────────────────────
LETTERS = list("ABCDEFGHIJKLMNOPQRSTUVWXYZ")
NUM_CLASSES = len(LETTERS)    # 26

# ── Color Palette (BGR for OpenCV) ───────────────────────────────────
COLORS = {
    "green":   (0, 255, 0),
    "red":     (0, 0, 255),
    "blue":    (255, 0, 0),
    "yellow":  (0, 255, 255),
    "white":   (255, 255, 255),
    "black":   (0, 0, 0),
    "orange":  (0, 165, 255),
    "cyan":    (255, 255, 0),
    "magenta": (255, 0, 255),
    "gray":    (128, 128, 128),
}

# ── UI Themes ────────────────────────────────────────────────────────
THEME_LIGHT = {
    "bg":          (245, 245, 245),
    "text":        (30, 30, 30),
    "accent":      (255, 100, 0),    # orange
    "panel":       (220, 220, 220),
    "conf_high":   (0, 200, 0),
    "conf_mid":    (0, 220, 220),
    "conf_low":    (0, 0, 220),
}

THEME_DARK = {
    "bg":          (30, 30, 30),
    "text":        (230, 230, 230),
    "accent":      (0, 180, 255),    # warm blue
    "panel":       (50, 50, 50),
    "conf_high":   (0, 220, 0),
    "conf_mid":    (0, 200, 200),
    "conf_low":    (0, 0, 200),
}

THEMES = [THEME_LIGHT, THEME_DARK]

# ── MediaPipe Landmark Names (for reference / debugging) ─────────────
LANDMARK_NAMES = [
    "WRIST",
    "THUMB_CMC", "THUMB_MCP", "THUMB_IP", "THUMB_TIP",
    "INDEX_FINGER_MCP", "INDEX_FINGER_PIP", "INDEX_FINGER_DIP", "INDEX_FINGER_TIP",
    "MIDDLE_FINGER_MCP", "MIDDLE_FINGER_PIP", "MIDDLE_FINGER_DIP", "MIDDLE_FINGER_TIP",
    "RING_FINGER_MCP", "RING_FINGER_PIP", "RING_FINGER_DIP", "RING_FINGER_TIP",
    "PINKY_MCP", "PINKY_PIP", "PINKY_DIP", "PINKY_TIP",
]
