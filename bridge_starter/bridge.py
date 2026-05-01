#!/usr/bin/env python3
"""
bikeped collision-warning bridge — adaptation starter kit.

A slim, self-contained version of the deployed bikeped pipeline aimed at
teams adapting the system to a new intersection or research environment.

Pipeline (one frame):
    capture -> [optional preprocess] -> YOLO detection ->
    fisheye -> ground BEV projection (LUT) -> ground-plane tracker ->
    DecisionPipeline (IDLE / SAFE / WARN / ALERT) ->
    feedback (SimulatedLight, optional sound) + MQTT publish + latency log

To adapt:
    1. Edit the USER CONFIG block below (camera intrinsics, MQTT broker,
       sound paths, model name, etc.).
    2. Calibrate your own fisheye lens with calibrate_fisheye.py from the
       main repo (or hand-set CAMERA_* if you trust the manufacturer
       values), then plug the values into the USER CONFIG.
    3. Run ``python bridge.py``.

The first run downloads the YOLO weights from Hugging Face and (best-
effort) exports them to a TensorRT engine. Subsequent runs reuse the
cached engine.

This script keeps the latency profiler from the production system so you
can produce the same latency reports with ``latency_report.py``.
"""

# ============================================================================
# USER CONFIG — EDIT THESE FOR YOUR DEPLOYMENT
# ============================================================================

# --- Video source -----------------------------------------------------------
USE_WEBCAM = True              # True = live camera; False = play VIDEO_FILE
VIDEO_FILE = "input.mp4"       # Path to your offline video (used when USE_WEBCAM=False)

# --- Camera intrinsics & geometry ------------------------------------------
# Drop in the values from your calibration. Defaults below match the
# bikeped reference deployment (200° fisheye, 12 ft pole, level pitch).
CAMERA_HEIGHT_M    = 3.6576    # mounting height above ground (metres)
CAMERA_FOV_DEG     = 197.9     # horizontal field of view
CAMERA_RADIUS_PX   = 1750      # fisheye projection-circle radius (pixels)
CAMERA_PITCH_DEG   = 0.0       # downward tilt (negative = looking down)
CAMERA_CX          = 1752.69   # optical centre x in the preprocessed crop
CAMERA_CY          = 1804.54   # optical centre y
CAMERA_K1          = 0.0       # optional Kannala-Brandt distortion (0 = off)
CAMERA_K2          = 0.0
CAMERA_PROJECTION  = "linear"  # linear | equalarea | orthographic | stereographic
INPUT_PAD_SIZE     = 3840      # preprocessing pad dimension (square)
INPUT_CROP_SIZE    = 3500      # preprocessing centre-crop dimension (square)

# --- YOLO model -------------------------------------------------------------
# Auto-downloaded from Hugging Face if missing, then auto-exported to a
# TensorRT engine on first run. Pretty much "set the repo and forget".
YOLO_HF_REPO     = "mehmetkeremturkcan/YOLO11xxl"
YOLO_HF_FILE     = "yolo11xxl.pt"
YOLO_LOCAL_DIR   = "models"
YOLO_IMGSZ       = 1280
YOLO_CONF        = 0.1
YOLO_DEVICE      = "cuda"      # "cuda" or "cpu"
# COCO class IDs we keep:  person, bicycle, car, motorcycle, bus, truck.
# Add 16 (dog), 31 (handbag), etc. if your scene needs them.
YOLO_CLASSES     = [0, 1, 2, 3, 5, 7]

# --- MQTT publishing --------------------------------------------------------
# Set the broker hostname/port for your team's MQTT infrastructure.
MQTT_ENABLED   = True
MQTT_BROKER    = "mqtt.example.com"   # <-- CHANGE ME
MQTT_PORT      = 1883
MQTT_TOPIC     = "bikeped/intersection/tracks"
MQTT_USERNAME  = None                 # set to enable auth
MQTT_PASSWORD  = None
MQTT_CLIENT_ID = "bikeped_bridge"
MQTT_QOS       = 0

# --- Audio feedback ---------------------------------------------------------
# Both files are optional. If a file is missing or play_sounds is not
# installed, the system silently skips audio and continues running.
ALERT_SOUND = "alert.wav"      # played when ALERT fires (e.g. bike-bell.wav)
SAFE_SOUND  = "safe.wav"       # played when SAFE state is entered (e.g. go.wav)
SOUND_THROTTLE_FRAMES = 60     # min frames between repeated sound triggers

# --- Decision pipeline parameters ------------------------------------------
# Defaults match the optimizer-derived values from the paper. Tune for
# your road geometry / cyclist mix as needed.
#
# IMPORTANT — DEC_PROX_MIN_M and the rider-on-bike problem:
#   YOLO almost always detects the cyclist's torso as a separate `person`
#   on top of the `bicycle`/`motorcycle` box. Without filtering, the
#   cyclist's body and their own bike will read as a person<->bike pair
#   roughly 0.3-0.7 m apart, which would otherwise trip the proximity
#   check on every cyclist that enters the scene.
#
#   We mitigate this two ways, BOTH of which you should tune for your
#   detector / camera height:
#     1. Rider-overlap suppression below tags persons whose 2D box
#        overlaps a bike's 2D box and excludes them from the pedestrian
#        list. See `find_rider_track_ids()`.
#     2. DEC_PROX_MIN_M sets a hard floor on the bike<->ped distance the
#        decision pipeline will alert on. Anything closer is treated as
#        rider self-overlap. With a wide-angle ceiling-mounted camera the
#        paper's value of 1.9 m is comfortable; with a low pole-mounted
#        camera or narrower lens you may need to RAISE it (try 2.5-3 m)
#        because the bbox bottom-centre projection error grows.
DEC_BIKE_MEMORY_FRAMES = 58
DEC_PROX_MIN_M         = 1.9
DEC_PROX_MAX_M         = 24.8
DEC_PROX_SPEED_MIN     = 0.147
DEC_LOOKBACK_FRAMES    = 2
DEC_SPEED_ADAPTIVE     = True
DEC_REACTION_S         = 0.84   # 85th-pctl field PRT (Martin & Bigazzi 2025)
DEC_DECEL_MS2          = 1.96   # 85th-pctl field decel; e-bike scenarios use 6.0
DEC_MAX_RANGE_M        = 25.0
DEC_FPS                = 30
RIDER_X_OVERLAP_FRAC   = 0.6    # |Δcx| < frac * max(width)  -> overlap candidate
RIDER_Y_OVERLAP_FRAC   = 0.3    # person foot >= bike top - frac * bike height

# --- Latency profiling ------------------------------------------------------
LATENCY_ENABLED        = True
LATENCY_SAVE_PATH      = "latency_data.npy"
LATENCY_SAVE_INTERVAL  = 500
LATENCY_WARMUP_FRAMES  = 30

# --- Display ---------------------------------------------------------------
SHOW_WINDOW   = True           # cv2.imshow with bbox overlay
WINDOW_NAME   = "bikeped bridge"
DEBUG         = False
VERBOSE       = False

# ============================================================================
# Implementation below — most adapters won't need to edit anything past this.
# ============================================================================

import argparse
import json
import sys
import time
from collections import deque
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
from numpy import pi, sin, cos, tan, sqrt

# Heavy deps imported lazily so --help still works without them.
try:
    from ultralytics import YOLO
except ImportError:
    YOLO = None

try:
    import paho.mqtt.client as mqtt
except ImportError:
    mqtt = None

from decision_pipeline import DecisionPipeline


# ---------------------------------------------------------------------------
# Simulated indicator light (drop-in replacement for real hardware)
# ---------------------------------------------------------------------------
class SimulatedLight:
    """Stand-in for a hardware indicator light.

    Prints the colour transitions to stdout. To wire up a real device,
    subclass this and override ``set_color(rgb)``. The interface matches
    the busylight ``on()`` API so the call sites don't need to change.
    """

    COLOR_NAMES = {
        (0xff, 0, 0):    "RED",
        (0xff, 0xa5, 0): "ORANGE",
        (0, 0xff, 0):    "GREEN",
        (0, 0, 0xff):    "BLUE",
        (0, 0, 0):       "OFF",
    }

    def __init__(self):
        self._last_name = None

    def set_color(self, rgb):
        rgb = tuple(int(c) for c in rgb)
        name = self.COLOR_NAMES.get(rgb, f"RGB{rgb}")
        if name != self._last_name:
            print(f"[light] {name}")
            self._last_name = name

    # busylight-compatible alias
    on = set_color


light = SimulatedLight()


# ---------------------------------------------------------------------------
# Optional audio playback
# ---------------------------------------------------------------------------
class SoundPlayer:
    """Plays short audio cues; degrades silently if files / library missing."""

    def __init__(self):
        self._missing_warned = set()
        try:
            from play_sounds import play_file as _play
            self._play = _play
        except ImportError:
            self._play = None
            print("[sound] 'play_sounds' not installed — audio feedback disabled. "
                  "To enable: pip install play-sounds")

    def play(self, path):
        if self._play is None or not path:
            return
        p = Path(path)
        if not p.exists():
            if str(p) not in self._missing_warned:
                self._missing_warned.add(str(p))
                print(f"[sound] {p} not found — skipping (this warning prints once)")
            return
        try:
            self._play(p, block=False)
        except Exception as e:
            print(f"[sound] failed to play {p}: {e}")


sounds = SoundPlayer()


# ---------------------------------------------------------------------------
# YOLO model: download from Hugging Face + best-effort TensorRT export
# ---------------------------------------------------------------------------
def fetch_weights(repo: str, filename: str, dest_dir: Path) -> Path:
    """Download ``filename`` from ``huggingface.co/<repo>`` into ``dest_dir``."""
    dest_dir.mkdir(parents=True, exist_ok=True)
    local = dest_dir / filename
    if local.exists():
        return local

    print(f"[model] {filename} not found locally — downloading from huggingface.co/{repo}")
    try:
        from huggingface_hub import hf_hub_download
    except ImportError:
        sys.exit("[model] huggingface_hub not installed. "
                 "Install with:  pip install huggingface_hub  "
                 f"or download {filename} manually to {local}.")
    cached = hf_hub_download(repo_id=repo, filename=filename)
    # Symlink (or copy on Windows) into our local folder so the path is stable.
    try:
        local.symlink_to(cached)
    except (OSError, NotImplementedError):
        import shutil
        shutil.copy2(cached, local)
    print(f"[model] saved to {local}")
    return local


def ensure_model(weights: Path, imgsz: int) -> Path:
    """Return a runnable model path. Tries to export ``.pt`` -> ``.engine``
    on first call; falls back to the original ``.pt`` if export fails (with
    a warning so the user knows inference will be slower).
    """
    if YOLO is None:
        sys.exit("[model] ultralytics not installed. Install with:  pip install ultralytics")
    if weights.suffix == ".engine" and weights.exists():
        return weights
    engine = weights.with_suffix(".engine")
    if engine.exists():
        return engine
    print(f"[model] no TensorRT engine found at {engine}")
    print(f"[model] attempting one-time export {weights.name} -> {engine.name} (this can take a few minutes) ...")
    try:
        m = YOLO(str(weights))
        m.export(format="engine", imgsz=imgsz, half=True, device=0)
        if engine.exists():
            print(f"[model] export OK: {engine}")
            return engine
        raise RuntimeError("export call returned but the .engine file was not produced")
    except Exception as e:
        print(f"[model] WARNING: TensorRT export failed ({type(e).__name__}: {e}).")
        print(f"[model] WARNING: falling back to the .pt weights at {weights} — "
              f"inference will be 2-5x slower. Install TensorRT and rerun to fix.")
        return weights


# ---------------------------------------------------------------------------
# Latency compensator (first-order kinematic predictor)
# ---------------------------------------------------------------------------
class LatencyCompensator:
    """Per-track EMA-velocity predictor that extrapolates by a constant
    integer-frame delay. O(N) per update."""

    def __init__(self, delay_frames: int, alpha: float = 0.5):
        self.delay_frames = delay_frames
        self.alpha = alpha
        self._prev = {}
        self._vel = {}
        self._comp = {}

    def update(self, tid, x, y):
        if self.delay_frames <= 0:
            self._comp[tid] = (x, y)
            return
        if tid in self._prev:
            px, py = self._prev[tid]
            raw_vx, raw_vy = x - px, y - py
            if tid in self._vel:
                ovx, ovy = self._vel[tid]
                vx = self.alpha * raw_vx + (1 - self.alpha) * ovx
                vy = self.alpha * raw_vy + (1 - self.alpha) * ovy
            else:
                vx, vy = raw_vx, raw_vy
            self._vel[tid] = (vx, vy)
        else:
            vx, vy = 0.0, 0.0
        self._prev[tid] = (x, y)
        self._comp[tid] = (x + vx * self.delay_frames,
                           y + vy * self.delay_frames)

    def get(self, tid, x, y):
        return self._comp.get(tid, (x, y))


# ---------------------------------------------------------------------------
# Ground-plane tracker (BEV nearest-neighbor + EMA velocity)
# ---------------------------------------------------------------------------
class GroundPlaneTracker:
    """Greedy nearest-neighbor tracker in metric ground-plane coordinates."""

    MAX_SPEED_MS = {0: 6.0, 1: 17.0, 2: 28.0, 3: 17.0, 5: 20.0, 7: 20.0}
    MAX_SPEED_DEFAULT = 28.0

    def __init__(self, match_radius=3.0, max_lost=300, alpha=0.5,
                 fps=30, min_hits=2):
        self.match_radius = match_radius
        self.max_lost = max_lost
        self.alpha = alpha
        self.fps = fps
        self.min_hits = min_hits
        self._tracks = {}
        self._next_id = 1

    def update(self, detections):
        for trk in self._tracks.values():
            if trk["lost"] > 0:
                trk["x"] += trk["vx"]
                trk["y"] += trk["vy"]
            trk["lost"] += 1

        used_trk, used_det, matches = set(), set(), []
        for di, (dx, dy, dcls) in enumerate(detections):
            best_d, best_tid = self.match_radius, None
            for tid, trk in self._tracks.items():
                if tid in used_trk or trk["class_id"] != dcls:
                    continue
                d = sqrt((dx - trk["x"])**2 + (dy - trk["y"])**2)
                if d < best_d:
                    best_d, best_tid = d, tid
            if best_tid is not None:
                matches.append((di, best_tid))
                used_trk.add(best_tid)
                used_det.add(di)

        # Apply matches with speed gating
        rejects = set()
        for di, tid in matches:
            dx, dy, dcls = detections[di]
            trk = self._tracks[tid]
            new_vx, new_vy = dx - trk["x"], dy - trk["y"]
            v = sqrt(new_vx**2 + new_vy**2) * self.fps
            if v > self.MAX_SPEED_MS.get(dcls, self.MAX_SPEED_DEFAULT):
                rejects.add((di, tid))
                continue
            trk["vx"] = self.alpha * trk["vx"] + (1 - self.alpha) * new_vx
            trk["vy"] = self.alpha * trk["vy"] + (1 - self.alpha) * new_vy
            trk["x"], trk["y"] = dx, dy
            trk["lost"] = 0
            trk["age"] = trk.get("age", 0) + 1
        for di, tid in rejects:
            used_trk.discard(tid)
            used_det.discard(di)

        for di, (dx, dy, dcls) in enumerate(detections):
            if di not in used_det:
                self._tracks[self._next_id] = {
                    "x": dx, "y": dy, "vx": 0.0, "vy": 0.0,
                    "class_id": dcls, "lost": 0, "age": 0,
                }
                self._next_id += 1

        for tid in [t for t, trk in self._tracks.items()
                    if trk["lost"] > self.max_lost
                    or sqrt(trk["vx"]**2 + trk["vy"]**2) * self.fps
                       > self.MAX_SPEED_MS.get(trk["class_id"], self.MAX_SPEED_DEFAULT)]:
            del self._tracks[tid]

        return [(tid, t["x"], t["y"], t["class_id"])
                for tid, t in self._tracks.items()
                if t.get("age", 0) >= self.min_hits]


# ---------------------------------------------------------------------------
# Coordinate mapping: original frame -> preprocessed (pad+crop) frame
# ---------------------------------------------------------------------------
class CoordinateMapper:
    def __init__(self, orig_h, orig_w, pad, crop):
        self.crop = crop
        self.x_offset = (pad - orig_w) // 2 - max(0, (pad - crop) // 2)
        self.y_offset = (pad - orig_h) // 2 - max(0, (pad - crop) // 2)

    def to_preprocessed(self, x, y):
        xp = x + self.x_offset
        yp = y + self.y_offset
        if 0 <= xp < self.crop and 0 <= yp < self.crop:
            return int(xp), int(yp)
        return None, None


# ---------------------------------------------------------------------------
# Pixel -> ground-plane LUT (equidistant fisheye + optional k1/k2)
# ---------------------------------------------------------------------------
def fisheye_pixel_to_ground(fw, fh, ch, cp_deg, fov, cx, cy, radius,
                            dtype="linear", k1=0.0, k2=0.0):
    """Build a (H, W, 3) ground-coordinate map and a validity mask.

    Coordinate frame (camera-relative): X = forward, Y = right, Z = up.
    """
    yg, xg = np.meshgrid(np.arange(fh), np.arange(fw), indexing="ij")
    dx, dy = xg - cx, yg - cy
    r = np.sqrt(dx**2 + dy**2)
    phi = np.arctan2(dy, dx)
    dim = 2 * radius
    if dtype == "linear":
        ifoc = dim * 180 / (fov * pi)
        r_norm = r / ifoc
        if k1 != 0.0 or k2 != 0.0:
            theta = r_norm.copy()
            for _ in range(6):
                t2 = theta * theta
                f = theta * (1.0 + t2 * (k1 + k2 * t2)) - r_norm
                fp = 1.0 + t2 * (3.0 * k1 + 5.0 * k2 * t2)
                theta = theta - f / fp
        else:
            theta = r_norm
    elif dtype == "equalarea":
        ifoc = dim / (2.0 * sin(fov * pi / 720))
        theta = 2 * np.arcsin(np.clip(r / (2.0 * ifoc), -1, 1))
    elif dtype == "orthographic":
        ifoc = dim / (2.0 * sin(fov * pi / 360))
        theta = np.arcsin(np.clip(r / ifoc, -1, 1))
    elif dtype == "stereographic":
        ifoc = dim / (2.0 * tan(fov * pi / 720))
        theta = 2 * np.arctan(r / (2.0 * ifoc))
    else:
        raise ValueError(f"Unknown projection: {dtype}")

    st, ct = np.sin(theta), np.cos(theta)
    sp, cp_ = np.sin(phi), np.cos(phi)
    rays = np.stack([ct, st * cp_, -st * sp], axis=-1)
    rays = rays / (np.linalg.norm(rays, axis=-1, keepdims=True) + 1e-10)
    pr = cp_deg * pi / 180
    cpr, spr = cos(pr), sin(pr)
    rp = np.stack([rays[..., 0] * cpr - rays[..., 2] * spr,
                   rays[..., 1],
                   rays[..., 0] * spr + rays[..., 2] * cpr], axis=-1)
    rz = rp[..., 2]
    valid = rz < -1e-6
    t = np.zeros_like(rz)
    t[valid] = -ch / rz[valid]
    g = np.zeros((fh, fw, 3))
    for i in range(3):
        g[..., i] = np.array([0, 0, ch])[i] + t * rp[..., i]
    g[~valid] = np.nan
    return g, valid


# ---------------------------------------------------------------------------
# MQTT helpers
# ---------------------------------------------------------------------------
def setup_mqtt():
    if not MQTT_ENABLED:
        return None
    if mqtt is None:
        print("[mqtt] paho-mqtt not installed; publishing disabled. "
              "Install with:  pip install paho-mqtt")
        return None
    try:
        client = mqtt.Client(client_id=MQTT_CLIENT_ID)
        if MQTT_USERNAME and MQTT_PASSWORD:
            client.username_pw_set(MQTT_USERNAME, MQTT_PASSWORD)
        print(f"[mqtt] connecting to {MQTT_BROKER}:{MQTT_PORT} ...")
        client.connect(MQTT_BROKER, MQTT_PORT, keepalive=60)
        client.loop_start()
        return client
    except Exception as e:
        print(f"[mqtt] connect failed ({e}); continuing without MQTT.")
        return None


def publish_tracks(client, frame_num, tracks):
    if client is None:
        return
    payload = {
        "timestamp": datetime.utcnow().isoformat() + "Z",
        "frame_number": int(frame_num),
        "objects": [
            {"track_id": int(tid), "class_id": int(cls),
             "position": {"x": float(x), "y": float(y),
                          "distance": float(sqrt(x*x + y*y))}}
            for (tid, x, y, cls) in tracks
        ],
    }
    try:
        client.publish(MQTT_TOPIC, json.dumps(payload), qos=MQTT_QOS)
    except Exception as e:
        if VERBOSE:
            print(f"[mqtt] publish failed: {e}")


# ---------------------------------------------------------------------------
# Display helpers
# ---------------------------------------------------------------------------
CLASS_NAMES = {0: "person", 1: "bike", 2: "car", 3: "motorbike",
               5: "bus", 7: "truck"}


# ---------------------------------------------------------------------------
# Rider-on-bike suppression
# ---------------------------------------------------------------------------
def find_rider_indices(boxes_data, x_overlap=RIDER_X_OVERLAP_FRAC,
                       y_overlap=RIDER_Y_OVERLAP_FRAC):
    """Return the indices of `person` detections that are sitting on a bike.

    YOLO emits a separate ``person`` box for the cyclist's torso on top of
    the ``bicycle`` / ``motorcycle`` box. Without filtering, the cyclist
    self-trips the proximity check. This helper detects the rider boxes
    by 2D-box overlap so the caller can drop them from the pedestrian
    list before running the decision pipeline.

    Heuristic (matches mqtt_bridge.py):
      person overlaps bike when
        |person_cx - bike_cx|  <  x_overlap * max(person_w, bike_w)
      AND
        person_bottom >= bike_top - y_overlap * bike_height

    Args:
        boxes_data: (N, 6+) array of YOLO output [x1, y1, x2, y2, conf, cls].
    Returns:
        set of row indices into boxes_data that should be treated as riders.
    """
    riders = set()
    if boxes_data is None or len(boxes_data) == 0:
        return riders
    persons, bikes = [], []
    for i, box in enumerate(boxes_data):
        if len(box) < 6:
            continue
        cls = int(box[5])
        if cls == 0:
            persons.append(i)
        elif cls in (1, 3):  # bicycle, motorcycle
            bikes.append(i)
    if not persons or not bikes:
        return riders
    for pi in persons:
        px1, _, px2, py2 = boxes_data[pi, :4]
        p_cx, p_w = (px1 + px2) * 0.5, px2 - px1
        for bi in bikes:
            bx1, by1, bx2, by2 = boxes_data[bi, :4]
            b_cx, b_w, b_h = (bx1 + bx2) * 0.5, bx2 - bx1, by2 - by1
            if abs(p_cx - b_cx) < x_overlap * max(p_w, b_w):
                if py2 >= by1 - y_overlap * b_h:
                    riders.add(pi)
                    break
    return riders


def draw_overlay(frame, boxes, decision_state):
    if frame is None:
        return frame
    for x1, y1, x2, y2, conf, cls in boxes:
        cls = int(cls)
        col = (0, 255, 0) if cls == 0 else (0, 165, 255)
        cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), col, 2)
        cv2.putText(frame, f"{CLASS_NAMES.get(cls, cls)} {conf:.2f}",
                    (int(x1), max(0, int(y1) - 6)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, col, 1, cv2.LINE_AA)
    badge_color = {"alert": (0, 0, 255), "warning": (0, 165, 255),
                   "safe": (0, 200, 0), "idle": (200, 100, 0)}
    cv2.putText(frame, decision_state.upper(), (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1.0,
                badge_color.get(decision_state, (255, 255, 255)),
                2, cv2.LINE_AA)
    return frame


# ---------------------------------------------------------------------------
# State -> light + sound mapping
# ---------------------------------------------------------------------------
def apply_feedback(state: str, sound_cooldown: dict):
    """Drive light + (optional) sound based on the decision state."""
    if state == "alert":
        light.set_color((0xff, 0, 0))
        if sound_cooldown.get("alert", 0) <= 0:
            sounds.play(ALERT_SOUND)
            sound_cooldown["alert"] = SOUND_THROTTLE_FRAMES
    elif state == "warning":
        light.set_color((0xff, 0xa5, 0))
    elif state == "safe":
        light.set_color((0, 0xff, 0))
        if sound_cooldown.get("safe", 0) <= 0:
            sounds.play(SAFE_SOUND)
            sound_cooldown["safe"] = SOUND_THROTTLE_FRAMES
    else:  # idle
        light.set_color((0, 0, 0xff))
    for k in list(sound_cooldown):
        sound_cooldown[k] = max(0, sound_cooldown[k] - 1)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--video", help="Override VIDEO_FILE (forces USE_WEBCAM=False)")
    ap.add_argument("--no-mqtt", action="store_true", help="Disable MQTT publishing")
    ap.add_argument("--no-window", action="store_true", help="Run without a display window")
    args = ap.parse_args()

    use_webcam = USE_WEBCAM
    video_file = VIDEO_FILE
    if args.video:
        use_webcam = False
        video_file = args.video
    show_window = SHOW_WINDOW and not args.no_window
    mqtt_client = None if args.no_mqtt else setup_mqtt()

    # 1. Resolve the model
    weights = fetch_weights(YOLO_HF_REPO, YOLO_HF_FILE, Path(YOLO_LOCAL_DIR))
    model_path = ensure_model(weights, YOLO_IMGSZ)
    model = YOLO(str(model_path))

    # 2. Build BEV LUT
    print(f"[bev] building ground LUT ({INPUT_CROP_SIZE}x{INPUT_CROP_SIZE}) ...")
    ground, valid_mask = fisheye_pixel_to_ground(
        INPUT_CROP_SIZE, INPUT_CROP_SIZE, CAMERA_HEIGHT_M, CAMERA_PITCH_DEG,
        CAMERA_FOV_DEG, CAMERA_CX, CAMERA_CY, CAMERA_RADIUS_PX,
        dtype=CAMERA_PROJECTION, k1=CAMERA_K1, k2=CAMERA_K2)
    print(f"[bev] valid pixels: {valid_mask.sum()}/{valid_mask.size}")

    # 3. Open video
    src = 0 if use_webcam else video_file
    cap = cv2.VideoCapture(src)
    if not cap.isOpened():
        sys.exit(f"[video] could not open source: {src}")
    fps = cap.get(cv2.CAP_PROP_FPS) or DEC_FPS

    ret, frame0 = cap.read()
    if not ret:
        sys.exit("[video] could not read first frame")
    orig_h, orig_w = frame0.shape[:2]
    coord_mapper = CoordinateMapper(orig_h, orig_w, INPUT_PAD_SIZE, INPUT_CROP_SIZE)
    cap.release()
    cap = cv2.VideoCapture(src)

    # 4. Decision pipeline + tracker + latency compensator
    pipeline = DecisionPipeline(
        bike_memory_frames=DEC_BIKE_MEMORY_FRAMES,
        prox_min=DEC_PROX_MIN_M, prox_max=DEC_PROX_MAX_M,
        prox_speed_min=DEC_PROX_SPEED_MIN, lookback_frames=DEC_LOOKBACK_FRAMES,
        max_range=DEC_MAX_RANGE_M, speed_adaptive=DEC_SPEED_ADAPTIVE,
        cyclist_reaction_s=DEC_REACTION_S, cyclist_decel_ms2=DEC_DECEL_MS2,
        fps=DEC_FPS,
    )
    tracker = GroundPlaneTracker(fps=DEC_FPS)
    compensator = LatencyCompensator(delay_frames=0)

    latency_records = []
    sound_cooldown = {}
    frame_count = 0
    proc_count = 0

    print("=" * 70)
    print(f" Source: {'WEBCAM' if use_webcam else video_file}")
    print(f" Model:  {model_path}")
    print(f" MQTT:   {'enabled (' + MQTT_BROKER + ')' if mqtt_client else 'disabled'}")
    print(f" Window: {'on' if show_window else 'off'}  (q to quit)")
    print("=" * 70)

    while cap.isOpened():
        t0 = time.perf_counter()
        success, frame = cap.read()
        t_acq = time.perf_counter()
        if not success:
            print("[video] end of stream")
            break
        frame_count += 1
        proc_count += 1

        t_pre = time.perf_counter()

        # Detection
        results = model.predict(frame, conf=YOLO_CONF, imgsz=YOLO_IMGSZ,
                                classes=YOLO_CLASSES, verbose=VERBOSE,
                                device=YOLO_DEVICE)
        t_yolo = time.perf_counter()

        boxes_data = []
        bev_dets = []
        if hasattr(results[0], "boxes") and hasattr(results[0].boxes, "data"):
            boxes_data = results[0].boxes.data.cpu().numpy()
            # Identify YOLO 'person' boxes that are actually a cyclist's torso
            # on top of a bike (see find_rider_indices docstring).
            rider_box_idx = find_rider_indices(boxes_data)
            for i, box in enumerate(boxes_data):
                if box.shape[0] < 6:
                    continue
                x1, y1, x2, y2, conf, cls = box[:6]
                cls = int(cls)
                # A YOLO person box that overlaps a bike is the rider — re-
                # label as bike so tracker / decision logic treat them as
                # one cyclist rather than a person<->bike proximity event.
                if i in rider_box_idx and cls == 0:
                    cls = 1
                cx = int((x1 + x2) / 2)
                cy = int(y2)
                xp, yp = coord_mapper.to_preprocessed(cx, cy)
                if xp is None or not valid_mask[yp, xp]:
                    continue
                bev_dets.append((float(ground[yp, xp, 0]),
                                 float(ground[yp, xp, 1]),
                                 cls))
        tracked = tracker.update(bev_dets)
        t_track = time.perf_counter()

        # Decision pipeline
        sim_t = proc_count / fps
        persons, bikes, bike_in_scene = [], [], False
        for tid, x, y, cls in tracked:
            compensator.update(tid, x, y)
            xc, yc = compensator.get(tid, x, y)
            entry = {"track_id": tid, "x": xc, "y": yc}
            pipeline.update_track(tid, xc, yc, sim_t)
            if cls == 0:
                persons.append(entry)
            elif cls in (1, 3):  # bike, motorcycle
                bikes.append(entry)
                bike_in_scene = True
        state, _alert_pair = pipeline.evaluate(persons, bikes, bike_in_scene, sim_t)
        t_dec = time.perf_counter()

        # Feedback
        apply_feedback(state, sound_cooldown)
        publish_tracks(mqtt_client, frame_count, tracked)

        # Display
        if show_window:
            disp = draw_overlay(frame.copy(), boxes_data, state)
            cv2.imshow(WINDOW_NAME, disp)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
        t_end = time.perf_counter()

        # Latency records
        if LATENCY_ENABLED and proc_count > LATENCY_WARMUP_FRAMES:
            speed = results[0].speed if results else {}
            latency_records.append([
                (t_acq - t0) * 1000.0,                   # acquire
                (t_pre - t_acq) * 1000.0,                # preprocess
                speed.get("preprocess", 0.0),            # yolo pre
                speed.get("inference", 0.0),             # yolo inference
                speed.get("postprocess", 0.0),           # yolo post
                (t_track - t_yolo) * 1000.0,             # bev+track
                (t_dec - t_track) * 1000.0,              # decision
                (t_end - t_dec) * 1000.0,                # display+publish
                (t_end - t0) * 1000.0,                   # total
            ])
            if len(latency_records) % LATENCY_SAVE_INTERVAL == 0:
                np.save(LATENCY_SAVE_PATH, np.array(latency_records))
                print(f"[latency] checkpoint: {len(latency_records)} records")

    # Cleanup
    cap.release()
    if show_window:
        cv2.destroyAllWindows()
    if mqtt_client is not None:
        mqtt_client.loop_stop()
        mqtt_client.disconnect()
    if LATENCY_ENABLED and latency_records:
        np.save(LATENCY_SAVE_PATH, np.array(latency_records))
        print(f"[latency] final save: {len(latency_records)} records -> {LATENCY_SAVE_PATH}")
    print(f"frames={frame_count}  processed={proc_count}")


if __name__ == "__main__":
    main()
