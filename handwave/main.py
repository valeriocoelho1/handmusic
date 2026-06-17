"""HandWave application entry point.

Captures webcam frames, detects a hand, classifies the gesture, executes the
mapped system action and renders a visual overlay. Run with::

    python -m handwave
    python -m handwave --camera 0 --width 1280 --height 720 --no-overlay
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
from typing import Optional

import numpy as np

from . import constants
from .controller import MediaController, VolumeController
from .gestures import GestureClassifier, GestureSmoothing, GestureType

logger = logging.getLogger("handwave")

try:
    from rich.console import Console
    from rich.logging import RichHandler

    _console: Optional["Console"] = Console()
except Exception:  # pragma: no cover - rich is optional at runtime
    _console = None


def build_parser() -> argparse.ArgumentParser:
    """Construct the command-line argument parser."""
    parser = argparse.ArgumentParser(
        prog="handwave",
        description="Control system volume and media with hand gestures.",
    )
    parser.add_argument(
        "--camera",
        type=int,
        default=constants.DEFAULT_CAMERA_INDEX,
        help="Webcam device index (default: 0).",
    )
    parser.add_argument(
        "--width",
        type=int,
        default=constants.DEFAULT_FRAME_WIDTH,
        help="Capture width in pixels (default: 1280).",
    )
    parser.add_argument(
        "--height",
        type=int,
        default=constants.DEFAULT_FRAME_HEIGHT,
        help="Capture height in pixels (default: 720).",
    )
    parser.add_argument(
        "--no-overlay",
        action="store_true",
        help="Disable the visual overlay window (headless control only).",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable DEBUG logging (landmark coordinates, etc.).",
    )
    parser.add_argument(
        "--smoothing-window",
        type=int,
        default=constants.SMOOTHING_WINDOW,
        help="Rolling window size for gesture smoothing (default: 5).",
    )
    return parser


def configure_logging(debug: bool) -> None:
    """Configure application logging, preferring rich output when available."""
    level = logging.DEBUG if debug else logging.INFO
    if _console is not None:
        logging.basicConfig(
            level=level,
            format="%(message)s",
            datefmt="[%X]",
            handlers=[RichHandler(console=_console, rich_tracebacks=True)],
        )
    else:  # pragma: no cover - fallback path
        logging.basicConfig(
            level=level,
            format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        )


def run(args: argparse.Namespace) -> int:
    """Run the main capture-and-control loop.

    Args:
        args: Parsed command-line arguments.

    Returns:
        Process exit code (0 on clean exit).
    """
    import cv2

    from .detector import HandDetector

    cap = cv2.VideoCapture(args.camera)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, args.width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, args.height)
    if not cap.isOpened():
        logger.error(
            "Could not open camera %d. Is it connected and not in use?",
            args.camera,
        )
        return 1

    detector = HandDetector()
    classifier = GestureClassifier(args.width, args.height)
    smoothing = GestureSmoothing(window=args.smoothing_window)
    volume = VolumeController()
    media = MediaController()

    overlay = None
    if not args.no_overlay:
        from .overlay import Overlay

        overlay = Overlay()

    current_volume = volume.get_volume()
    prev_time = time.monotonic()
    fps = 0.0

    if _console is not None:
        _console.print(
            "[bold green]HandWave[/] running — press [bold]q[/] or "
            "[bold]ESC[/] to quit."
        )
    else:  # pragma: no cover
        print("HandWave running — press q or ESC to quit.")

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                logger.warning("Failed to read frame from camera.")
                break

            frame = cv2.flip(frame, 1)  # mirror mode
            h, w = frame.shape[:2]
            classifier.update_frame_size(w, h)

            landmarks = detector.detect(frame, draw=overlay is not None)
            raw_gesture = classifier.classify(landmarks)
            gesture = smoothing.update(raw_gesture)

            current_volume = _handle_gesture(
                gesture,
                classifier,
                volume,
                media,
                current_volume,
            )

            # FPS measurement.
            now = time.monotonic()
            dt = now - prev_time
            prev_time = now
            if dt > 0:
                fps = 0.9 * fps + 0.1 * (1.0 / dt) if fps else (1.0 / dt)

            if overlay is not None:
                overlay.draw(
                    frame,
                    gesture,
                    current_volume,
                    fps,
                    landmarks=landmarks,
                    pinch_distance=classifier.last_pinch_distance,
                )
                cv2.imshow("HandWave", frame)
                key = cv2.waitKey(1) & 0xFF
                if key in (ord("q"), 27):  # q or ESC
                    break
    except KeyboardInterrupt:  # pragma: no cover - interactive
        logger.info("Interrupted by user.")
    finally:
        cap.release()
        detector.close()
        if overlay is not None:
            cv2.destroyAllWindows()
        logger.info("HandWave stopped cleanly.")
    return 0


def _handle_gesture(
    gesture: GestureType,
    classifier: GestureClassifier,
    volume: VolumeController,
    media: MediaController,
    current_volume: float,
) -> float:
    """Execute the system action mapped to ``gesture``.

    Args:
        gesture: The current stable gesture.
        classifier: Classifier (source of the latest pinch distance).
        volume: Volume controller.
        media: Media controller.
        current_volume: The last applied volume scalar.

    Returns:
        The possibly-updated volume scalar.
    """
    if gesture == GestureType.PINCH:
        pct = float(
            np.interp(
                classifier.last_pinch_distance,
                [constants.PINCH_MIN_DIST, constants.PINCH_MAX_DIST],
                [0.0, 100.0],
            )
        )
        scalar = pct / 100.0
        volume.set_volume(scalar)
        return scalar
    if gesture == GestureType.FIST:
        media.play_pause()
    elif gesture == GestureType.THREE_FINGERS:
        media.next_track()
    return current_volume


def main(argv: Optional[list[str]] = None) -> int:
    """Console-script entry point.

    Args:
        argv: Optional argument vector (defaults to ``sys.argv``).

    Returns:
        Process exit code.
    """
    parser = build_parser()
    args = parser.parse_args(argv)
    configure_logging(args.debug)
    return run(args)


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
