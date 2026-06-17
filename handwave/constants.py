"""Tunable constants for HandWave.

All magic numbers used across the application live here so behaviour can be
adjusted in a single place without touching logic code.
"""

from __future__ import annotations

# --- Gesture smoothing ------------------------------------------------------
#: Number of recent frames kept in the rolling smoothing window.
SMOOTHING_WINDOW: int = 5
#: A gesture must be detected this many consecutive frames before it is emitted.
SMOOTHING_CONSISTENCY: int = 3

# --- Action cooldown --------------------------------------------------------
#: Seconds to ignore a repeated discrete gesture (fist / three fingers) after
#: it fires, preventing rapid re-triggering.
GESTURE_COOLDOWN_S: float = 1.5

# --- Pinch → volume mapping -------------------------------------------------
#: Normalised pinch distance (thumb↔index) treated as 0% volume.
PINCH_MIN_DIST: float = 20.0
#: Normalised pinch distance treated as 100% volume.
PINCH_MAX_DIST: float = 200.0

# --- MediaPipe detection ----------------------------------------------------
MIN_DETECTION_CONFIDENCE: float = 0.7
MIN_TRACKING_CONFIDENCE: float = 0.5
#: Maximum number of hands tracked simultaneously.
MAX_HANDS: int = 1

# --- Camera defaults --------------------------------------------------------
DEFAULT_CAMERA_INDEX: int = 0
DEFAULT_FRAME_WIDTH: int = 1280
DEFAULT_FRAME_HEIGHT: int = 720

# --- Gesture classification thresholds -------------------------------------
#: Reference hand-size landmarks used to normalise pixel distances so gestures
#: work at any distance from the camera (wrist → middle-finger MCP).
HAND_SIZE_LANDMARK_A: int = 0
HAND_SIZE_LANDMARK_B: int = 9
#: A finger is considered "closed" for the pinch/three-finger checks when its
#: tip-to-PIP relationship indicates a folded finger.
FINGER_CLOSED_MARGIN: float = 0.0

# --- MediaPipe landmark indices --------------------------------------------
WRIST: int = 0
THUMB_TIP: int = 4
INDEX_TIP: int = 8
INDEX_PIP: int = 6
INDEX_MCP: int = 5
MIDDLE_TIP: int = 12
MIDDLE_PIP: int = 10
MIDDLE_MCP: int = 9
RING_TIP: int = 16
RING_PIP: int = 14
RING_MCP: int = 13
PINKY_TIP: int = 20
PINKY_PIP: int = 18
PINKY_MCP: int = 17
THUMB_IP: int = 3
