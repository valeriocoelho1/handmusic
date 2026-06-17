"""Tests for gesture classification and smoothing.

Landmark data is mocked: each fixture builds the 21 normalised landmarks that
characterise a gesture's geometry, so no camera or MediaPipe is required.
"""

from __future__ import annotations

from dataclasses import dataclass

import pytest

from handwave.gestures import (
    GestureClassifier,
    GestureSmoothing,
    GestureType,
)

FRAME_W = 1280
FRAME_H = 720


@dataclass
class LM:
    """Minimal stand-in for a MediaPipe normalised landmark."""

    x: float
    y: float
    z: float = 0.0


def _base_open_hand() -> list[LM]:
    """An open hand with all five fingers extended (default layout)."""
    return [
        LM(0.50, 0.90),  # 0 wrist
        LM(0.42, 0.75),  # 1 thumb cmc
        LM(0.38, 0.68),  # 2 thumb mcp
        LM(0.35, 0.62),  # 3 thumb ip
        LM(0.32, 0.58),  # 4 thumb tip (extended: y < ip)
        LM(0.45, 0.55),  # 5 index mcp
        LM(0.45, 0.45),  # 6 index pip
        LM(0.45, 0.35),  # 7 index dip
        LM(0.45, 0.25),  # 8 index tip (up)
        LM(0.50, 0.55),  # 9 middle mcp
        LM(0.50, 0.45),  # 10 middle pip
        LM(0.50, 0.35),  # 11 middle dip
        LM(0.50, 0.25),  # 12 middle tip (up)
        LM(0.55, 0.55),  # 13 ring mcp
        LM(0.55, 0.45),  # 14 ring pip
        LM(0.55, 0.35),  # 15 ring dip
        LM(0.55, 0.25),  # 16 ring tip (up)
        LM(0.60, 0.55),  # 17 pinky mcp
        LM(0.60, 0.47),  # 18 pinky pip
        LM(0.60, 0.37),  # 19 pinky dip
        LM(0.60, 0.27),  # 20 pinky tip (up)
    ]


def make_fist() -> list[LM]:
    """All four fingers folded below their MCP joints."""
    hand = _base_open_hand()
    for tip in (8, 12, 16, 20):
        hand[tip].y = 0.62  # below the MCP row at 0.55
    hand[4].y = 0.70  # thumb folded too
    return hand


def make_three_fingers() -> list[LM]:
    """Index, middle, ring up; pinky and thumb folded."""
    hand = _base_open_hand()
    hand[20].y = 0.62  # pinky tip below its pip (0.47) → down
    hand[4].y = 0.70  # thumb tip below its ip (0.62) → down
    return hand


def make_pinch() -> list[LM]:
    """Open hand with thumb tip drawn close to the index tip."""
    hand = _base_open_hand()
    hand[4].x, hand[4].y = 0.45, 0.27  # near index tip (0.45, 0.25)
    return hand


def make_idle_spread() -> list[LM]:
    """Hand open with thumb and index splayed far apart (no gesture)."""
    hand = _base_open_hand()
    hand[4].x, hand[4].y = 0.05, 0.50  # far-left thumb
    hand[8].x, hand[8].y = 0.95, 0.25  # far-right index
    return hand


@pytest.fixture()
def classifier() -> GestureClassifier:
    return GestureClassifier(FRAME_W, FRAME_H)


def test_classify_fist(classifier: GestureClassifier) -> None:
    assert classifier.classify(make_fist()) == GestureType.FIST


def test_classify_three_fingers(classifier: GestureClassifier) -> None:
    assert classifier.classify(make_three_fingers()) == GestureType.THREE_FINGERS


def test_classify_pinch(classifier: GestureClassifier) -> None:
    assert classifier.classify(make_pinch()) == GestureType.PINCH


def test_classify_idle_when_no_hand(classifier: GestureClassifier) -> None:
    assert classifier.classify(None) == GestureType.IDLE


def test_classify_idle_when_spread(classifier: GestureClassifier) -> None:
    assert classifier.classify(make_idle_spread()) == GestureType.IDLE


def test_pinch_distance_is_scale_invariant() -> None:
    """The same gesture at different camera distances yields similar values."""
    near = GestureClassifier(FRAME_W, FRAME_H)
    near.classify(make_pinch())
    near_dist = near.last_pinch_distance

    # Shrink the whole hand toward the centre to simulate a farther hand.
    far_hand = make_pinch()
    for lm in far_hand:
        lm.x = 0.5 + (lm.x - 0.5) * 0.5
        lm.y = 0.5 + (lm.y - 0.5) * 0.5
    far = GestureClassifier(FRAME_W, FRAME_H)
    far.classify(far_hand)
    far_dist = far.last_pinch_distance

    assert near_dist == pytest.approx(far_dist, abs=5.0)


def test_priority_fist_over_pinch(classifier: GestureClassifier) -> None:
    """Fist must win even though a folded hand also has nearby thumb/index."""
    assert classifier.classify(make_fist()) == GestureType.FIST


def test_smoothing_requires_consecutive_frames() -> None:
    smoothing = GestureSmoothing(window=5, min_consistent=3)
    # Two frames are not enough to flip away from the IDLE default.
    assert smoothing.update(GestureType.PINCH) == GestureType.IDLE
    assert smoothing.update(GestureType.PINCH) == GestureType.IDLE
    # Third consecutive frame promotes the gesture to stable.
    assert smoothing.update(GestureType.PINCH) == GestureType.PINCH


def test_smoothing_ignores_single_frame_jitter() -> None:
    smoothing = GestureSmoothing(window=5, min_consistent=3)
    for _ in range(3):
        smoothing.update(GestureType.FIST)
    assert smoothing.stable == GestureType.FIST
    # A single spurious frame must not change the stable gesture.
    assert smoothing.update(GestureType.PINCH) == GestureType.FIST


def test_smoothing_reset() -> None:
    smoothing = GestureSmoothing(window=5, min_consistent=3)
    for _ in range(3):
        smoothing.update(GestureType.FIST)
    smoothing.reset()
    assert smoothing.stable == GestureType.IDLE
