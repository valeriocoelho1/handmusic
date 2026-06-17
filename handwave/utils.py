"""Geometric helper functions for landmark math.

These helpers operate on MediaPipe-style normalised landmarks (objects exposing
``x``, ``y`` and ``z`` attributes in the ``[0, 1]`` range) and convert them to
pixel space where needed.
"""

from __future__ import annotations

from typing import Protocol, Sequence

import numpy as np


class Landmark(Protocol):
    """Structural type for a single normalised hand landmark."""

    x: float
    y: float
    z: float


def landmark_to_xy(
    landmark: Landmark, frame_width: int, frame_height: int
) -> tuple[int, int]:
    """Convert a normalised landmark to integer pixel coordinates.

    Args:
        landmark: A landmark with normalised ``x``/``y`` in ``[0, 1]``.
        frame_width: Width of the frame in pixels.
        frame_height: Height of the frame in pixels.

    Returns:
        The ``(x, y)`` pixel coordinates as integers.
    """
    return int(landmark.x * frame_width), int(landmark.y * frame_height)


def distance(
    a: Landmark,
    b: Landmark,
    frame_width: int,
    frame_height: int,
) -> float:
    """Euclidean pixel distance between two normalised landmarks.

    Args:
        a: First landmark.
        b: Second landmark.
        frame_width: Frame width in pixels.
        frame_height: Frame height in pixels.

    Returns:
        The distance between ``a`` and ``b`` in pixels.
    """
    ax, ay = landmark_to_xy(a, frame_width, frame_height)
    bx, by = landmark_to_xy(b, frame_width, frame_height)
    return float(np.hypot(ax - bx, ay - by))


def finger_up(
    landmarks: Sequence[Landmark], tip_idx: int, pip_idx: int
) -> bool:
    """Return whether a finger is extended (tip above its PIP joint).

    In image space the Y axis grows downward, so an extended finger has a tip
    with a *smaller* Y value than its PIP joint.

    Args:
        landmarks: The full sequence of 21 hand landmarks.
        tip_idx: Index of the finger tip landmark.
        pip_idx: Index of the finger PIP (or MCP) joint landmark.

    Returns:
        ``True`` if the finger is extended, ``False`` otherwise.
    """
    return landmarks[tip_idx].y < landmarks[pip_idx].y


def normalised_distance(
    a: Landmark,
    b: Landmark,
    ref_a: Landmark,
    ref_b: Landmark,
    frame_width: int,
    frame_height: int,
) -> float:
    """Scale-invariant distance between ``a`` and ``b``.

    The raw pixel distance between ``a`` and ``b`` is divided by a reference
    distance (typically the hand size) and re-scaled to a stable pseudo-pixel
    range so thresholds hold regardless of how far the hand is from the camera.

    Args:
        a: First landmark of the measured pair.
        b: Second landmark of the measured pair.
        ref_a: First landmark of the reference (hand-size) pair.
        ref_b: Second landmark of the reference (hand-size) pair.
        frame_width: Frame width in pixels.
        frame_height: Frame height in pixels.

    Returns:
        The normalised distance. Returns ``0.0`` if the reference distance is
        degenerate (zero) to avoid division by zero.
    """
    raw = distance(a, b, frame_width, frame_height)
    ref = distance(ref_a, ref_b, frame_width, frame_height)
    if ref <= 1e-6:
        return 0.0
    # Re-scale by a nominal hand size (100px) so the value lands in a range
    # comparable to PINCH_MIN_DIST / PINCH_MAX_DIST.
    return (raw / ref) * 100.0


def clamp(value: float, low: float, high: float) -> float:
    """Clamp ``value`` into the inclusive ``[low, high]`` range."""
    return max(low, min(high, value))
