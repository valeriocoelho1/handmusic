<div align="center">

```
| | | | __ _ _ __   __| \ \      / /_ ___   _____
| |_| |/ _` | '_ \ / _` |\ \ /\ / / _` \ \ / / _ \
|  _  | (_| | | | | (_| | \ V  V / (_| |\ V /  __/
|_| |_|\__,_|_| |_|\__,_|  \_/\_/ \__,_| \_/ \___|
```

**Control your system volume and media playback with hand gestures — fully local, no cloud.**

</div>

---

HandWave uses your webcam and real-time hand-landmark detection (MediaPipe) to
turn simple hand gestures into system actions: pinch to set the volume, make a
fist to play/pause, and raise three fingers to skip to the next track.

## ✨ Features

- 🤏 **Pinch (thumb + index)** → smoothly **control system volume**
- ✋ **Open palm (all fingers up)** → **play / pause** media
- ✌️ **Peace / V sign (index + middle up)** → **next track**
- 🖥️ Live OpenCV overlay: gesture border, volume bar, pinch line, FPS counter
- 🧈 Temporal smoothing to eliminate jitter and accidental triggers
- 🔁 Per-action cooldown so play/pause and next-track fire **once** per gesture
- 🌍 Cross-platform: **Windows · macOS · Linux**
- 🔒 100% offline — no internet connection or cloud API is ever used

## 📋 Requirements

- **Python 3.10+**
- A webcam

| OS | Volume backend | Media keys backend |
|----|----------------|--------------------|
| Windows | `pycaw` | `keyboard` |
| macOS | `osascript` | `osascript` (key codes) |
| Linux | `pactl` (PulseAudio/PipeWire), falls back to `amixer` | `playerctl`, falls back to `keyboard` |

## 🚀 Installation

```bash
git clone https://github.com/yourusername/handwave
cd handwave
pip install -r requirements.txt
python -m handwave
```

Or install as a package (provides the `handwave` command):

```bash
pip install -e .
handwave
```

> **Tip:** use a virtual environment (`python -m venv .venv && source .venv/bin/activate`).

## 🖐️ Gesture Reference

| Gesture | Visual | Action | Detection logic |
|---|---|---|---|
| 🤏 **Pinch** | Thumb + index, with middle/ring/pinky curled | Set volume | Normalised distance between landmark 4 (thumb tip) and 8 (index tip), mapped to 0–100%. Active whenever middle, ring and pinky are curled |
| ✋ **Open palm** | All four fingers extended | Play / Pause | Index, middle, ring and pinky tips all above their PIP joints |
| ✌️ **Peace / V** | Index + middle up, ring + pinky curled | Next track | Index and middle tips above their PIP joints; ring and pinky curled |

Each gesture is defined by a **unique set of extended fingers**, so they never
collide with one another (the thumb is ignored when gating gestures because its
posture is the least reliable).

**Priority** (most restrictive first): `Open palm → Peace → Pinch → Idle`.
If no gesture matches, nothing happens.

Pinch-to-volume mapping uses linear interpolation between a full pinch
(`~20` normalised px → 0%) and a fully open span (`~200` normalised px → 100%).
Distances are normalised by hand size so the mapping holds at any distance from
the camera.

## ⚙️ Configuration

All behaviour is driven by command-line flags:

| Flag | Default | Description |
|------|---------|-------------|
| `--camera N` | `0` | Webcam device index |
| `--width N` | `1280` | Capture width in pixels |
| `--height N` | `720` | Capture height in pixels |
| `--no-overlay` | off | Run headless (control only, no preview window) |
| `--debug` | off | Verbose DEBUG logging (landmark coords, gesture events) |
| `--smoothing-window N` | `5` | Rolling window size for gesture smoothing |

```bash
python -m handwave --camera 1 --width 1280 --height 720
python -m handwave --no-overlay --debug
```

Deeper tuning constants (cooldown duration, pinch thresholds, detection
confidence) live in [`handwave/constants.py`](handwave/constants.py).

## 🧠 How it Works

```
Webcam frame (OpenCV)
        │  flip horizontally (mirror)
        ▼
HandDetector  ──►  MediaPipe Hands → 21 normalised landmarks
        ▼
GestureClassifier  ──►  geometric rules → GestureType
        ▼
GestureSmoothing  ──►  emit only after N consistent frames
        ▼
VolumeController / MediaController  ──►  OS-specific system control
        ▼
Overlay  ──►  draw feedback → cv2.imshow
```

1. **Detection** — MediaPipe returns 21 normalised landmarks for the hand.
2. **Classification** — purely geometric rules compare finger tips against
   their joints (evaluated most-restrictive first).
3. **Smoothing** — a rolling window requires the same gesture for several
   consecutive frames before it is acted upon, removing flicker.
4. **Control** — the mapped action is dispatched through the platform-specific
   backend, with a cooldown on discrete actions.

## 🛠️ Troubleshooting

**Camera not found / black window**
- Try a different `--camera` index (`0`, `1`, `2`…).
- Close other apps that may hold the webcam (Zoom, browser tabs, etc.).
- On macOS, grant camera permission to your terminal under
  *System Settings → Privacy & Security → Camera*.

**Volume not changing on Linux**
- Ensure PulseAudio/PipeWire is running and `pactl` is installed
  (`sudo apt install pulseaudio-utils`).
- If you use ALSA only, HandWave falls back to `amixer` (`alsa-utils`).

**Media keys do nothing**
- **Linux:** install `playerctl` (`sudo apt install playerctl`) and make sure
  your player supports MPRIS.
- **Windows/Linux fallback:** the `keyboard` package may require elevated
  privileges to emit media keys (run the terminal as administrator / with
  `sudo`).
- **macOS:** grant *Accessibility* permission to your terminal under
  *System Settings → Privacy & Security → Accessibility*.

**Permission errors importing `keyboard`**
- HandWave degrades gracefully and logs a warning; volume control still works.

**Low FPS**
- Lower the capture resolution: `--width 640 --height 480`.
- Run with `--no-overlay` for headless control.

## 🤝 Contributing

Contributions are welcome!

1. Fork and clone the repo.
2. Create a feature branch: `git checkout -b feature/my-gesture`.
3. Install dev dependencies: `pip install -e ".[dev]"`.
4. Run the tests: `pytest`.
5. Keep type hints and Google-style docstrings; put tunables in
   `constants.py` (no magic numbers in logic code).
6. Open a pull request describing your change.

## 📄 License

Released under the [MIT License](LICENSE).
