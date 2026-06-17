"""Gesture classification and temporal smoothing.

:class:`GestureClassifier` maps a single frame of hand landmarks to a
:class:`GestureType`. :class:`GestureSmoothing` debounces the raw per-frame
classifications so a gesture is only emitted once it has been stable for a few
consecutive frames.
"""

from __future__ import annotations

import logging
from collections import deque
from enum import Enum
from typing import Deque, Optional, Sequence

from . import constants
from .utils import Landmark, finger_up, normalised_distance

logger = logging.getLogger(__name__)


class GestureType(Enum):
    """The set of gestures HandWave can recognise.

    Each non-idle gesture is defined by a *unique* combination of which fingers
    are extended, so they never collide with one another:

    * ``PINCH``     — middle, ring and pinky curled (thumb/index free) → volume
    * ``OPEN_PALM`` — all four fingers extended → play / pause
    * ``PEACE``     — index and middle extended, ring and pinky curled → next
    """

    IDLE = "idle"
    PINCH = "pinch"
    OPEN_PALM = "open_palm"
    PEACE = "peace"


class GestureClassifier:
    """Classify a frame of hand landmarks into a :class:`GestureType`.

    Gestures are evaluated from most to least restrictive: OPEN_PALM → PEACE →
    PINCH → IDLE. The first match wins. Because each gesture corresponds to a
    distinct set of extended fingers, the ordering only resolves transitional
    frames, not genuine ambiguity.

    Args:
        frame_width: Frame width in pixels, used for distance computations.
        frame_height: Frame height in pixels, used for distance computations.
        pinch_min_dist: Normalised pinch distance mapped to 0% volume.
        pinch_max_dist: Normalised pinch distance mapped to 100% volume.
    """

    def __init__(
        self,
        frame_width: int,
        frame_height: int,
        pinch_min_dist: float = constants.PINCH_MIN_DIST,
        pinch_max_dist: float = constants.PINCH_MAX_DIST,
    ) -> None:
        self.frame_width = frame_width
        self.frame_height = frame_height
        self.pinch_min_dist = pinch_min_dist
        self.pinch_max_dist = pinch_max_dist
        #: Most recent normalised pinch distance, exposed for the overlay/volume
        #: mapping. Updated on every :meth:`classify` call.
        self.last_pinch_distance: float = 0.0

    def update_frame_size(self, width: int, height: int) -> None:
        """Update the cached frame dimensions used for distance math."""
        self.frame_width = width
        self.frame_height = height

    def _pinch_distance(self, landmarks: Sequence[Landmark]) -> float:
        """Scale-invariant thumb-tip ↔ index-tip distance."""
        return normalised_distance(
            landmarks[constants.THUMB_TIP],
            landmarks[constants.INDEX_TIP],
            landmarks[constants.HAND_SIZE_LANDMARK_A],
            landmarks[constants.HAND_SIZE_LANDMARK_B],
            self.frame_width,
            self.frame_height,
        )

    @staticmethod
    def _fingers_up(
        landmarks: Sequence[Landmark],
    ) -> tuple[bool, bool, bool, bool]:
        """Return whether index, middle, ring and pinky are each extended.

        A finger is extended when its tip sits above (smaller Y than) its PIP
        joint. The thumb is intentionally ignored — it is used only for the
        pinch distance, never for gating a gesture, since its posture is the
        least reliable.
        """
        index = finger_up(landmarks, constants.INDEX_TIP, constants.INDEX_PIP)
        middle = finger_up(
            landmarks, constants.MIDDLE_TIP, constants.MIDDLE_PIP
        )
        ring = finger_up(landmarks, constants.RING_TIP, constants.RING_PIP)
        pinky = finger_up(landmarks, constants.PINKY_TIP, constants.PINKY_PIP)
        return index, middle, ring, pinky

    @staticmethod
    def _is_open_palm(
        index: bool, middle: bool, ring: bool, pinky: bool
    ) -> bool:
        """All four fingers extended."""
        return index and middle and ring and pinky

    @staticmethod
    def _is_peace(
        index: bool, middle: bool, ring: bool, pinky: bool
    ) -> bool:
        """Index and middle extended; ring and pinky curled (a ``V`` sign)."""
        return index and middle and not ring and not pinky

    @staticmethod
    def _is_pinch(
        index: bool, middle: bool, ring: bool, pinky: bool
    ) -> bool:
        """A pinch posture: middle, ring and pinky curled.

        The index and thumb are left free because the index naturally curls as
        the pinch closes (low volume) and extends as it opens (high volume).
        """
        return (not middle) and (not ring) and (not pinky)

    def classify(self, landmarks: Optional[Sequence[Landmark]]) -> GestureType:
        """Classify a single frame of landmarks.

        Args:
            landmarks: The 21 normalised hand landmarks, or ``None`` when no
                hand is detected.

        Returns:
            The recognised :class:`GestureType` (``IDLE`` if none match).
        """
        if landmarks is None:
            self.last_pinch_distance = 0.0
            return GestureType.IDLE

        self.last_pinch_distance = self._pinch_distance(landmarks)
        index, middle, ring, pinky = self._fingers_up(landmarks)

        # Evaluate most restrictive first.
        if self._is_open_palm(index, middle, ring, pinky):
            logger.debug("Classified OPEN_PALM")
            return GestureType.OPEN_PALM
        if self._is_peace(index, middle, ring, pinky):
            logger.debug("Classified PEACE")
            return GestureType.PEACE
        if self._is_pinch(index, middle, ring, pinky):
            logger.debug(
                "Classified PINCH (dist=%.1f)", self.last_pinch_distance
            )
            return GestureType.PINCH
        return GestureType.IDLE


class GestureSmoothing:
    """Debounce raw per-frame gestures using a rolling window.

    A gesture is only emitted once it has appeared in at least ``min_consistent``
    of the most recent frames *and* as the last ``min_consistent`` consecutive
    frames, removing single-frame jitter.

    Args:
        window: Number of recent frames retained.
        min_consistent: Consecutive identical frames required to emit a gesture.
    """

    def __init__(
        self,
        window: int = constants.SMOOTHING_WINDOW,
        min_consistent: int = constants.SMOOTHING_CONSISTENCY,
    ) -> None:
        self.window = window
        self.min_consistent = min_consistent
        self._history: Deque[GestureType] = deque(maxlen=window)
        self._stable: GestureType = GestureType.IDLE

    def update(self, gesture: GestureType) -> GestureType:
        """Feed the latest raw gesture and return the current stable gesture.

        Args:
            gesture: The raw per-frame classification.

        Returns:
            The smoothed/stable gesture after applying the consistency rule.
        """
        self._history.append(gesture)

        if len(self._history) >= self.min_consistent:
            recent = list(self._history)[-self.min_consistent :]
            if all(g == recent[0] for g in recent):
                if recent[0] != self._stable:
                    logger.info("Gesture changed: %s", recent[0].value)
                self._stable = recent[0]
        return self._stable

    @property
    def stable(self) -> GestureType:
        """The current stable gesture."""
        return self._stable

    def reset(self) -> None:
        """Clear the smoothing history."""
        self._history.clear()
        self._stable = GestureType.IDLE
