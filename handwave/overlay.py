"""OpenCV visual overlay rendering.

:class:`Overlay` draws gesture feedback, a volume bar, the pinch measurement
line, an FPS counter and the current gesture label onto a BGR frame in place.
"""

from __future__ import annotations

import logging
from typing import Optional, Sequence

import cv2
import numpy as np

from . import constants
from .gestures import GestureType
from .utils import Landmark, landmark_to_xy

logger = logging.getLogger(__name__)

# BGR colours per gesture.
_BORDER_COLORS: dict[GestureType, Optional[tuple[int, int, int]]] = {
    GestureType.IDLE: None,
    GestureType.PINCH: (255, 160, 0),  # blue-ish
    GestureType.FIST: (0, 140, 255),  # orange
    GestureType.THREE_FINGERS: (0, 200, 0),  # green
}

_GESTURE_LABELS: dict[GestureType, str] = {
    GestureType.IDLE: "IDLE",
    GestureType.PINCH: "PINCH - VOLUME",
    GestureType.FIST: "FIST - PLAY/PAUSE",
    GestureType.THREE_FINGERS: "THREE - NEXT TRACK",
}

_ACTION_LABELS: dict[GestureType, str] = {
    GestureType.PINCH: "VOLUME",
    GestureType.FIST: "PAUSE / PLAY",
    GestureType.THREE_FINGERS: "NEXT TRACK",
}

_WHITE = (255, 255, 255)
_BLACK = (0, 0, 0)
_FONT = cv2.FONT_HERSHEY_SIMPLEX


class Overlay:
    """Render the HandWave heads-up display onto OpenCV frames."""

    def __init__(self) -> None:
        self.border_thickness = 12

    def draw(
        self,
        frame: "np.ndarray",
        gesture: GestureType,
        volume: float,
        fps: float,
        landmarks: Optional[Sequence[Landmark]] = None,
        pinch_distance: float = 0.0,
    ) -> "np.ndarray":
        """Draw the full overlay onto ``frame`` in place.

        Args:
            frame: The BGR frame to annotate.
            gesture: The current stable gesture.
            volume: Current volume in ``[0.0, 1.0]``.
            fps: Measured frames per second.
            landmarks: Optional landmarks, used to draw the pinch line.
            pinch_distance: Current normalised pinch distance.

        Returns:
            The same ``frame`` object, annotated.
        """
        self._draw_border(frame, gesture)
        self._draw_volume_bar(frame, volume)
        self._draw_gesture_label(frame, gesture)
        self._draw_action_label(frame, gesture)
        self._draw_fps(frame, fps)
        self._draw_volume_number(frame, volume)
        if gesture == GestureType.PINCH and landmarks is not None:
            self._draw_pinch_line(frame, landmarks, pinch_distance)
        return frame

    def _draw_border(self, frame: "np.ndarray", gesture: GestureType) -> None:
        color = _BORDER_COLORS.get(gesture)
        if color is None:
            return
        h, w = frame.shape[:2]
        cv2.rectangle(
            frame,
            (0, 0),
            (w - 1, h - 1),
            color,
            self.border_thickness,
        )

    def _draw_volume_bar(self, frame: "np.ndarray", volume: float) -> None:
        h, w = frame.shape[:2]
        bar_x = 40
        bar_top = int(h * 0.20)
        bar_bottom = int(h * 0.80)
        bar_w = 30
        # Background track.
        cv2.rectangle(
            frame,
            (bar_x, bar_top),
            (bar_x + bar_w, bar_bottom),
            (60, 60, 60),
            -1,
        )
        # Filled portion.
        fill_h = int((bar_bottom - bar_top) * float(np.clip(volume, 0.0, 1.0)))
        cv2.rectangle(
            frame,
            (bar_x, bar_bottom - fill_h),
            (bar_x + bar_w, bar_bottom),
            (0, 220, 120),
            -1,
        )
        cv2.rectangle(
            frame,
            (bar_x, bar_top),
            (bar_x + bar_w, bar_bottom),
            _WHITE,
            2,
        )
        pct = int(round(volume * 100))
        cv2.putText(
            frame,
            f"{pct}%",
            (bar_x - 5, bar_top - 12),
            _FONT,
            0.6,
            _WHITE,
            2,
            cv2.LINE_AA,
        )

    def _draw_gesture_label(
        self, frame: "np.ndarray", gesture: GestureType
    ) -> None:
        label = _GESTURE_LABELS.get(gesture, gesture.value.upper())
        # Semi-transparent background box in the top-left.
        (tw, th), _ = cv2.getTextSize(label, _FONT, 0.9, 2)
        x, y = 90, 50
        overlay = frame.copy()
        cv2.rectangle(
            overlay,
            (x - 10, y - th - 14),
            (x + tw + 10, y + 12),
            _BLACK,
            -1,
        )
        cv2.addWeighted(overlay, 0.45, frame, 0.55, 0, frame)
        cv2.putText(
            frame, label, (x, y), _FONT, 0.9, _WHITE, 2, cv2.LINE_AA
        )

    def _draw_action_label(
        self, frame: "np.ndarray", gesture: GestureType
    ) -> None:
        action = _ACTION_LABELS.get(gesture)
        if action is None:
            return
        h, w = frame.shape[:2]
        (tw, th), _ = cv2.getTextSize(action, _FONT, 1.1, 3)
        x = (w - tw) // 2
        y = int(h * 0.5)
        cv2.putText(
            frame, action, (x, y), _FONT, 1.1, _BLACK, 6, cv2.LINE_AA
        )
        cv2.putText(
            frame, action, (x, y), _FONT, 1.1, _WHITE, 3, cv2.LINE_AA
        )

    def _draw_fps(self, frame: "np.ndarray", fps: float) -> None:
        h, w = frame.shape[:2]
        text = f"{fps:5.1f} FPS"
        (tw, _th), _ = cv2.getTextSize(text, _FONT, 0.6, 2)
        cv2.putText(
            frame,
            text,
            (w - tw - 20, 35),
            _FONT,
            0.6,
            (0, 255, 255),
            2,
            cv2.LINE_AA,
        )

    def _draw_volume_number(self, frame: "np.ndarray", volume: float) -> None:
        h, _w = frame.shape[:2]
        pct = int(round(volume * 100))
        cv2.putText(
            frame,
            f"VOL {pct}%",
            (30, h - 25),
            _FONT,
            0.8,
            _WHITE,
            2,
            cv2.LINE_AA,
        )

    def _draw_pinch_line(
        self,
        frame: "np.ndarray",
        landmarks: Sequence[Landmark],
        pinch_distance: float,
    ) -> None:
        h, w = frame.shape[:2]
        thumb = landmark_to_xy(landmarks[constants.THUMB_TIP], w, h)
        index = landmark_to_xy(landmarks[constants.INDEX_TIP], w, h)
        cv2.line(frame, thumb, index, (255, 160, 0), 3)
        cv2.circle(frame, thumb, 8, (255, 160, 0), -1)
        cv2.circle(frame, index, 8, (255, 160, 0), -1)
        mid = ((thumb[0] + index[0]) // 2, (thumb[1] + index[1]) // 2)
        cv2.putText(
            frame,
            f"{pinch_distance:.0f}",
            (mid[0] + 10, mid[1]),
            _FONT,
            0.6,
            _WHITE,
            2,
            cv2.LINE_AA,
        )
