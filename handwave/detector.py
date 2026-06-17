"""Hand landmark detection via MediaPipe.

Thin wrapper around ``mediapipe.solutions.hands`` exposing a simple
:meth:`HandDetector.detect` interface that returns normalised landmarks for the
primary detected hand.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Optional, Sequence

from . import constants

if TYPE_CHECKING:  # pragma: no cover - typing only
    import numpy as np
    from .utils import Landmark

logger = logging.getLogger(__name__)

try:
    import mediapipe as mp
except ImportError as exc:  # pragma: no cover - environment dependent
    raise ImportError(
        "mediapipe is required for HandWave. Install it with "
        "`pip install mediapipe` (Python 3.10+)."
    ) from exc


class HandDetector:
    """Detect hand landmarks in BGR video frames using MediaPipe Hands.

    Args:
        max_hands: Maximum number of hands to track.
        min_detection_confidence: Minimum confidence for the initial detection.
        min_tracking_confidence: Minimum confidence for landmark tracking.
    """

    def __init__(
        self,
        max_hands: int = constants.MAX_HANDS,
        min_detection_confidence: float = constants.MIN_DETECTION_CONFIDENCE,
        min_tracking_confidence: float = constants.MIN_TRACKING_CONFIDENCE,
    ) -> None:
        self._mp_hands = mp.solutions.hands
        self._mp_draw = mp.solutions.drawing_utils
        self._mp_styles = mp.solutions.drawing_styles
        self._hands = self._mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=max_hands,
            min_detection_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence,
        )
        logger.debug(
            "HandDetector initialised (max_hands=%d, det=%.2f, track=%.2f)",
            max_hands,
            min_detection_confidence,
            min_tracking_confidence,
        )

    def detect(
        self, frame: "np.ndarray", draw: bool = True
    ) -> Optional[Sequence["Landmark"]]:
        """Detect the primary hand's landmarks in ``frame``.

        Args:
            frame: A BGR image as returned by OpenCV.
            draw: When ``True``, draw the landmark skeleton onto ``frame``
                in place.

        Returns:
            The list of 21 normalised landmarks for the first detected hand, or
            ``None`` if no hand is found.
        """
        # MediaPipe expects RGB; OpenCV frames are BGR.
        import cv2  # local import keeps detector importable without a camera

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        rgb.flags.writeable = False
        results = self._hands.process(rgb)

        if not results.multi_hand_landmarks:
            return None

        hand = results.multi_hand_landmarks[0]
        if draw:
            self._mp_draw.draw_landmarks(
                frame,
                hand,
                self._mp_hands.HAND_CONNECTIONS,
                self._mp_styles.get_default_hand_landmarks_style(),
                self._mp_styles.get_default_hand_connections_style(),
            )

        if logger.isEnabledFor(logging.DEBUG):
            logger.debug(
                "Detected hand: wrist=(%.3f, %.3f)",
                hand.landmark[constants.WRIST].x,
                hand.landmark[constants.WRIST].y,
            )
        return hand.landmark

    def close(self) -> None:
        """Release MediaPipe resources."""
        self._hands.close()
        logger.debug("HandDetector closed")

    def __enter__(self) -> "HandDetector":
        return self

    def __exit__(self, *_exc: object) -> None:
        self.close()
