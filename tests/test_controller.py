"""Tests for the cross-platform volume and media controllers.

All OS calls are mocked: we assert the correct subprocess commands are issued
for each platform without touching the real system audio.
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import pytest

from handwave import controller as controller_mod
from handwave.controller import MediaController, VolumeController


@pytest.fixture()
def recorded_calls(monkeypatch: pytest.MonkeyPatch) -> list[list[str]]:
    """Patch subprocess.run and record the argv of every call."""
    calls: list[list[str]] = []

    def fake_run(args: Any, *_a: Any, **_k: Any) -> SimpleNamespace:
        calls.append(list(args))
        return SimpleNamespace(returncode=0, stdout="", stderr="")

    monkeypatch.setattr(controller_mod.subprocess, "run", fake_run)
    return calls


def _make_volume(system: str) -> VolumeController:
    vc = VolumeController()
    vc._system = system  # force platform for the test
    return vc


def test_macos_set_volume_command(recorded_calls: list[list[str]]) -> None:
    vc = _make_volume("Darwin")
    vc.set_volume(0.5)
    assert recorded_calls == [
        ["osascript", "-e", "set volume output volume 50"]
    ]


def test_macos_set_volume_clamps_high(recorded_calls: list[list[str]]) -> None:
    vc = _make_volume("Darwin")
    vc.set_volume(2.0)  # over-range
    assert recorded_calls[0] == [
        "osascript",
        "-e",
        "set volume output volume 100",
    ]


def test_macos_set_volume_clamps_low(recorded_calls: list[list[str]]) -> None:
    vc = _make_volume("Darwin")
    vc.set_volume(-1.0)  # under-range
    assert recorded_calls[0] == [
        "osascript",
        "-e",
        "set volume output volume 0",
    ]


def test_linux_set_volume_command(recorded_calls: list[list[str]]) -> None:
    vc = _make_volume("Linux")
    vc.set_volume(0.7)
    assert recorded_calls[0] == [
        "pactl",
        "set-sink-volume",
        "@DEFAULT_SINK@",
        "70%",
    ]


def test_macos_get_volume_parses_output(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fake_run(args: Any, *_a: Any, **_k: Any) -> SimpleNamespace:
        return SimpleNamespace(returncode=0, stdout="42\n", stderr="")

    monkeypatch.setattr(controller_mod.subprocess, "run", fake_run)
    vc = _make_volume("Darwin")
    assert vc.get_volume() == pytest.approx(0.42)


def test_linux_get_volume_parses_output(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    sample = "Volume: front-left: 45875 /  70% / -9.29 dB,   front-right: ..."

    def fake_run(args: Any, *_a: Any, **_k: Any) -> SimpleNamespace:
        return SimpleNamespace(returncode=0, stdout=sample, stderr="")

    monkeypatch.setattr(controller_mod.subprocess, "run", fake_run)
    vc = _make_volume("Linux")
    assert vc.get_volume() == pytest.approx(0.70)


def test_media_cooldown_blocks_rapid_fire(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fired: list[str] = []
    mc = MediaController(cooldown_s=1.5)
    mc._system = "TestOS"
    mc._keyboard = None
    monkeypatch.setattr(mc, "_send_play_pause", lambda: fired.append("pp"))

    assert mc.play_pause() is True  # first fires
    assert mc.play_pause() is False  # within cooldown, blocked
    assert fired == ["pp"]


def test_media_cooldown_expires(monkeypatch: pytest.MonkeyPatch) -> None:
    fired: list[str] = []
    fake_time = {"now": 1000.0}
    monkeypatch.setattr(
        controller_mod.time, "monotonic", lambda: fake_time["now"]
    )
    mc = MediaController(cooldown_s=1.5)
    mc._system = "TestOS"
    mc._keyboard = None
    monkeypatch.setattr(mc, "_send_next_track", lambda: fired.append("next"))

    assert mc.next_track() is True
    fake_time["now"] += 2.0  # advance past the cooldown
    assert mc.next_track() is True
    assert fired == ["next", "next"]


def test_media_play_pause_and_next_independent_cooldowns(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    mc = MediaController(cooldown_s=1.5)
    mc._system = "TestOS"
    mc._keyboard = None
    monkeypatch.setattr(mc, "_send_play_pause", lambda: None)
    monkeypatch.setattr(mc, "_send_next_track", lambda: None)

    assert mc.play_pause() is True
    # A different action is not blocked by play_pause's cooldown.
    assert mc.next_track() is True
