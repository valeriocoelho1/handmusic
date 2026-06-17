"""Tests for gesture classification and smoothing.

Landmark data is mocked: each fixture builds the 21 normalised landmarks that
characterise a gesture's geometry, so no camera or MediaPipe is required.

Gesture model (each is a unique set of extended fingers):
    * OPEN_PALM — index, middle, ring, pinky all up      → play / pause
    * PEACE     — index + middle up, ring + pinky down    → next track
    * PINCH     — middle, ring, pinky down (index free)   → volume
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


def _fold(hand: list[LM], tip_idx: int, y: float = 0.62) -> None:
    """Curl a finger by pushing its tip below its PIP joint (greater Y)."""
    hand[tip_idx].y = y


def make_open_palm() -> list[LM]:
    """All four fingers extended."""
    return _base_open_hand()


def make_peace() -> list[LM]:
    """Index + middle up; ring + pinky curled."""
    hand = _base_open_hand()
    _fold(hand, 16)  # ring tip down
    _fold(hand, 20)  # pinky tip down
    return hand


def make_pinch() -> list[LM]:
    """Middle, ring and pinky curled; thumb drawn near the index tip."""
    hand = _base_open_hand()
    _fold(hand, 12)  # middle down
    _fold(hand, 16)  # ring down
    _fold(hand, 20)  # pinky down
    hand[4].x, hand[4].y = 0.45, 0.27  # thumb near index tip (0.45, 0.25)
    return hand


def make_idle() -> list[LM]:
    """A pose matching no gesture (only the ring finger extended)."""
    hand = _base_open_hand()
    _fold(hand, 8)  # index down
    _fold(hand, 12)  # middle down
    _fold(hand, 20)  # pinky down
    return hand  # ring stays up → not open, not peace, not pinch


@pytest.fixture()
def classifier() -> GestureClassifier:
    return GestureClassifier(FRAME_W, FRAME_H)


def test_classify_open_palm(classifier: GestureClassifier) -> None:
    assert classifier.classify(make_open_palm()) == GestureType.OPEN_PALM


def test_classify_peace(classifier: GestureClassifier) -> None:
    assert classifier.classify(make_peace()) == GestureType.PEACE


def test_classify_pinch(classifier: GestureClassifier) -> None:
    assert classifier.classify(make_pinch()) == GestureType.PINCH


def test_classify_idle_when_no_hand(classifier: GestureClassifier) -> None:
    assert classifier.classify(None) == GestureType.IDLE


def test_classify_idle_for_unknown_pose(classifier: GestureClassifier) -> None:
    assert classifier.classify(make_idle()) == GestureType.IDLE


def test_open_palm_is_not_pinch(classifier: GestureClassifier) -> None:
    """An open hand with thumb near index must stay OPEN_PALM, not PINCH."""
    hand = make_open_palm()
    hand[4].x, hand[4].y = 0.45, 0.27  # thumb near index tip
    assert classifier.classify(hand) == GestureType.OPEN_PALM


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
        smoothing.update(GestureType.OPEN_PALM)
    assert smoothing.stable == GestureType.OPEN_PALM
    # A single spurious frame must not change the stable gesture.
    assert smoothing.update(GestureType.PINCH) == GestureType.OPEN_PALM


def test_smoothing_reset() -> None:
    smoothing = GestureSmoothing(window=5, min_consistent=3)
    for _ in range(3):
        smoothing.update(GestureType.OPEN_PALM)
    smoothing.reset()
    assert smoothing.stable == GestureType.IDLE
