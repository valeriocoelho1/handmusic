"""Cross-platform volume and media control.

:class:`VolumeController` sets and reads the system output volume on Windows
(pycaw), macOS (osascript) and Linux (pactl/amixer). :class:`MediaController`
sends play/pause and next-track media keys, with a cooldown to prevent rapid
re-firing.
"""

from __future__ import annotations

import logging
import platform
import subprocess
import time
from typing import Optional

from . import constants
from .utils import clamp

logger = logging.getLogger(__name__)

_SYSTEM = platform.system()  # "Windows" | "Darwin" | "Linux"


class VolumeController:
    """Read and set the system output volume on the current platform.

    All public methods accept and return a scalar volume in the ``[0.0, 1.0]``
    range, clamping out-of-range inputs automatically.
    """

    def __init__(self) -> None:
        self._system = _SYSTEM
        self._win_volume = None
        if self._system == "Windows":
            self._init_windows()
        logger.debug("VolumeController initialised for %s", self._system)

    # -- Windows (pycaw) ----------------------------------------------------
    def _init_windows(self) -> None:
        """Acquire the pycaw master-volume endpoint."""
        try:
            from ctypes import cast, POINTER

            from comtypes import CLSCTX_ALL
            from pycaw.pycaw import AudioUtilities, IAudioEndpointVolume

            devices = AudioUtilities.GetSpeakers()
            interface = devices.Activate(
                IAudioEndpointVolume._iid_, CLSCTX_ALL, None
            )
            self._win_volume = cast(interface, POINTER(IAudioEndpointVolume))
        except Exception as exc:  # pragma: no cover - Windows only
            logger.error("Failed to initialise pycaw: %s", exc)
            self._win_volume = None

    def set_volume(self, scalar: float) -> None:
        """Set the system volume.

        Args:
            scalar: Desired volume in ``[0.0, 1.0]``. Values outside the range
                are clamped.
        """
        scalar = clamp(scalar, 0.0, 1.0)
        try:
            if self._system == "Windows":
                self._set_windows(scalar)
            elif self._system == "Darwin":
                self._set_macos(scalar)
            elif self._system == "Linux":
                self._set_linux(scalar)
            else:  # pragma: no cover - exotic platforms
                logger.warning("Unsupported platform: %s", self._system)
        except Exception as exc:  # pragma: no cover - platform dependent
            logger.error("Failed to set volume: %s", exc)

    def _set_windows(self, scalar: float) -> None:  # pragma: no cover
        if self._win_volume is not None:
            self._win_volume.SetMasterVolumeLevelScalar(scalar, None)

    def _set_macos(self, scalar: float) -> None:  # pragma: no cover
        pct = int(round(scalar * 100))
        subprocess.run(
            ["osascript", "-e", f"set volume output volume {pct}"],
            check=False,
        )

    def _set_linux(self, scalar: float) -> None:  # pragma: no cover
        pct = int(round(scalar * 100))
        result = subprocess.run(
            [
                "pactl",
                "set-sink-volume",
                "@DEFAULT_SINK@",
                f"{pct}%",
            ],
            check=False,
        )
        if result.returncode != 0:
            # Fall back to ALSA's amixer.
            subprocess.run(
                ["amixer", "set", "Master", f"{pct}%"],
                check=False,
            )

    def get_volume(self) -> float:
        """Return the current system volume in ``[0.0, 1.0]``.

        Returns ``0.0`` when the volume cannot be read on the current platform.
        """
        try:
            if self._system == "Windows":
                return self._get_windows()
            if self._system == "Darwin":
                return self._get_macos()
            if self._system == "Linux":
                return self._get_linux()
        except Exception as exc:  # pragma: no cover - platform dependent
            logger.error("Failed to read volume: %s", exc)
        return 0.0

    def _get_windows(self) -> float:  # pragma: no cover
        if self._win_volume is not None:
            return float(self._win_volume.GetMasterVolumeLevelScalar())
        return 0.0

    def _get_macos(self) -> float:  # pragma: no cover
        result = subprocess.run(
            ["osascript", "-e", "output volume of (get volume settings)"],
            capture_output=True,
            text=True,
            check=False,
        )
        try:
            return clamp(int(result.stdout.strip()) / 100.0, 0.0, 1.0)
        except (ValueError, TypeError):
            return 0.0

    def _get_linux(self) -> float:  # pragma: no cover
        result = subprocess.run(
            ["pactl", "get-sink-volume", "@DEFAULT_SINK@"],
            capture_output=True,
            text=True,
            check=False,
        )
        # Example output: "Volume: front-left: 45875 /  70% / ..."
        for token in result.stdout.split():
            if token.endswith("%"):
                try:
                    return clamp(int(token.rstrip("%")) / 100.0, 0.0, 1.0)
                except ValueError:
                    continue
        return 0.0


class MediaController:
    """Send media-key actions with a per-action cooldown.

    Args:
        cooldown_s: Seconds to ignore repeated triggers of the same action.
    """

    def __init__(self, cooldown_s: float = constants.GESTURE_COOLDOWN_S) -> None:
        self._system = _SYSTEM
        self.cooldown_s = cooldown_s
        self._last_fired: dict[str, float] = {}
        self._keyboard = self._try_import_keyboard()
        logger.debug("MediaController initialised for %s", self._system)

    @staticmethod
    def _try_import_keyboard():
        """Import the optional ``keyboard`` module, returning ``None`` on fail."""
        try:
            import keyboard  # type: ignore

            return keyboard
        except Exception as exc:  # pragma: no cover - optional dep / perms
            logger.warning(
                "`keyboard` unavailable (%s); media keys may not work on this "
                "platform without it.",
                exc,
            )
            return None

    def _cooldown_ok(self, action: str) -> bool:
        """Return whether ``action`` may fire given its last trigger time."""
        now = time.monotonic()
        # Default to -inf so a never-fired action always passes, even when
        # time.monotonic() is still near zero early in the process lifetime.
        last = self._last_fired.get(action, float("-inf"))
        if now - last < self.cooldown_s:
            return False
        self._last_fired[action] = now
        return True

    def play_pause(self) -> bool:
        """Toggle play/pause. Returns ``True`` if the action actually fired."""
        if not self._cooldown_ok("play_pause"):
            return False
        logger.info("Media: play/pause")
        self._send_play_pause()
        return True

    def next_track(self) -> bool:
        """Skip to the next track. Returns ``True`` if the action fired."""
        if not self._cooldown_ok("next_track"):
            return False
        logger.info("Media: next track")
        self._send_next_track()
        return True

    # -- Platform key delivery ---------------------------------------------
    def _send_play_pause(self) -> None:  # pragma: no cover - platform dependent
        if self._system == "Darwin":
            self._macos_key(16)  # NX_KEYTYPE_PLAY
        elif self._system == "Linux" and self._playerctl("play-pause"):
            return
        elif self._keyboard is not None:
            self._keyboard.send("play/pause media")

    def _send_next_track(self) -> None:  # pragma: no cover - platform dependent
        if self._system == "Darwin":
            self._macos_key(17)  # NX_KEYTYPE_NEXT
        elif self._system == "Linux" and self._playerctl("next"):
            return
        elif self._keyboard is not None:
            self._keyboard.send("next track")

    def _macos_key(self, key_code: int) -> None:  # pragma: no cover
        """Emit a macOS media key via the system event subsystem."""
        script = (
            'tell application "System Events" to key code '
            f"{key_code} using {{}}"
        )
        # Media keys on macOS are special HID keys; osascript `key code` covers
        # the standard play/next codes used by most media players.
        subprocess.run(["osascript", "-e", script], check=False)

    def _playerctl(self, command: str) -> bool:  # pragma: no cover
        """Try to control playback via playerctl/dbus. Returns success."""
        try:
            result = subprocess.run(
                ["playerctl", command],
                capture_output=True,
                check=False,
            )
            return result.returncode == 0
        except FileNotFoundError:
            return False

    def reset_cooldowns(self) -> None:
        """Clear all cooldown timers (mainly for tests)."""
        self._last_fired.clear()
