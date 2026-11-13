"""Writes one rendered run to disk: videos, ground truth, stills and the manifest.

Layout of a run directory (``run_id = <engine>_<scenario-id>_<weather>``)::

    <out_dir>/<run_id>/
      manifest.json             run metadata, camera models, per-frame records, timeline
      video/<view>.mp4          clean view, H.264
      video/<view>_overlay.mp4  view with the HUD (when overlay is on)
      video/composite.mp4       the chosen layout
      frames/<view>/000000.jpg  only when frames are saved
      gt/<view>.jsonl           one line per frame: 2D boxes, 3D corners, visibility
      stills/                   first frame, first alert, closest bike-pedestrian approach

Videos are encoded with ``ffmpeg`` (libx264, yuv420p) through a pipe; when no
ffmpeg executable is found, ``cv2.VideoWriter`` with the ``mp4v`` codec is
used instead and a warning is logged.
"""

from __future__ import annotations

import json
import logging
import math
import shutil
import subprocess
import tempfile
from collections import deque
from dataclasses import replace
from pathlib import Path
from typing import Any, Deque, Dict, List, Optional, Sequence, Tuple, Union

import cv2
import numpy as np

from .compose import LAYOUTS, compose, fit_size
from .groundtruth import closest_pair, frame_ground_truth
from .types import FrameRecord, RunInfo

_LOG = logging.getLogger(__name__)

PathLike = Union[str, Path]
SAVE_FRAME_FORMATS = ("none", "jpg", "png")


def find_ffmpeg() -> Optional[str]:
    """Path of an ffmpeg executable: the one on ``PATH``, else the ``imageio-ffmpeg`` binary."""
    exe = shutil.which("ffmpeg")
    if exe:
        return exe
    try:
        import imageio_ffmpeg

        return imageio_ffmpeg.get_ffmpeg_exe()
    except Exception:
        return None


def _even_frame(image: np.ndarray, size: Tuple[int, int]) -> np.ndarray:
    """Fit ``image`` to the writer size: pad odd sides by edge replication, resize otherwise."""
    h, w = image.shape[:2]
    if (w, h) == size:
        return image
    if size[0] - w in (0, 1) and size[1] - h in (0, 1):
        return cv2.copyMakeBorder(image, 0, size[1] - h, 0, size[0] - w, cv2.BORDER_REPLICATE)
    return cv2.resize(image, size, interpolation=cv2.INTER_AREA)


class VideoSink:
    """An H.264 mp4 writer fed with BGR ``uint8`` frames of a fixed size.

    The size is the first frame's size rounded up to even numbers, as
    yuv420p requires; later frames are padded or resized to it.
    """

    def __init__(self, path: PathLike, fps: float, crf: int = 18,
                 ffmpeg: Optional[str] = None) -> None:
        self.path = Path(path)
        self.fps = float(fps)
        self.crf = int(crf)
        self.ffmpeg = ffmpeg if ffmpeg is not None else find_ffmpeg()
        self.size: Optional[Tuple[int, int]] = None
        self.frames = 0
        self._proc: Optional[subprocess.Popen] = None
        self._stderr: Any = None
        self._cv: Optional[cv2.VideoWriter] = None
        self._closed = False

    def _open(self, size: Tuple[int, int]) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.size = size
        if self.ffmpeg:
            self._stderr = tempfile.TemporaryFile()
            cmd = [self.ffmpeg, "-y", "-loglevel", "error", "-f", "rawvideo",
                   "-pix_fmt", "bgr24", "-s", f"{size[0]}x{size[1]}", "-r", f"{self.fps:g}",
                   "-i", "-", "-an", "-c:v", "libx264", "-pix_fmt", "yuv420p",
                   "-crf", str(self.crf), "-preset", "medium", "-movflags", "+faststart",
                   str(self.path)]
            self._proc = subprocess.Popen(cmd, stdin=subprocess.PIPE, stderr=self._stderr)
        else:
            _LOG.warning("ffmpeg not found; writing %s with OpenCV mp4v instead of H.264",
                         self.path.name)
            self._cv = cv2.VideoWriter(str(self.path), cv2.VideoWriter_fourcc(*"mp4v"),
                                       self.fps, size)
            if not self._cv.isOpened():
                raise RuntimeError(f"cannot open a video writer for {self.path}")

    def _ffmpeg_error(self) -> str:
        if self._stderr is None:
            return ""
        self._stderr.seek(0)
        return self._stderr.read().decode(errors="replace").strip()

    def write(self, image: np.ndarray) -> None:
        if self._closed:
            raise RuntimeError(f"video {self.path} is closed")
        if image.ndim == 2:
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
        image = np.ascontiguousarray(image[..., :3], dtype=np.uint8)
        if self.size is None:
            h, w = image.shape[:2]
            self._open((w + w % 2, h + h % 2))
        frame = np.ascontiguousarray(_even_frame(image, self.size))
        if self._proc is not None:
            try:
                self._proc.stdin.write(frame.tobytes())
            except (BrokenPipeError, OSError) as exc:
                raise RuntimeError(f"ffmpeg failed on {self.path}: {self._ffmpeg_error()}") from exc
        else:
            self._cv.write(frame)
        self.frames += 1

    def close(self) -> None:
        """Finish the file. Safe to call more than once."""
        if self._closed:
            return
        self._closed = True
        if self._proc is not None:
            try:
                self._proc.stdin.close()
            except OSError:
                pass
            code = self._proc.wait()
            message = self._ffmpeg_error()
            self._stderr.close()
            if code != 0:
                _LOG.error("ffmpeg exited with %d writing %s: %s", code, self.path, message)
        if self._cv is not None:
            self._cv.release()


def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(v) for v in value]
    if isinstance(value, np.ndarray):
        return _json_safe(value.tolist())
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    if isinstance(value, (int, np.integer)):
        return int(value)
    if isinstance(value, (float, np.floating)):
        v = float(value)
        return v if math.isfinite(v) else None
    if value is None or isinstance(value, str):
        return value
    return str(value)


def _intervals(times: Sequence[float], flags: Sequence[bool], dt: float) -> List[List[float]]:
    """Merge flagged frames into ``[t_start, t_end]`` intervals (seconds)."""
    out: List[List[float]] = []
    start: Optional[float] = None
    for t, flag in zip(times, flags):
        if flag and start is None:
            start = t
        elif not flag and start is not None:
            out.append([round(start, 4), round(t, 4)])
            start = None
    if start is not None and times:
        out.append([round(start, 4), round(times[-1] + dt, 4)])
    return out


class Recorder:
    """Collects :class:`FrameRecord` objects of one run and writes the run directory.

    ``views`` are the view names to record from ``frame.images``; the first
    is the primary view used by the composite. ``overlay`` adds a HUD video
    per view, ``layout`` is one of ``single``, ``split`` (primary view plus a
    plan view) or ``quad`` (up to three views plus a plan view).
    ``save_frames`` is ``none``, ``jpg`` or ``png``. ``crf`` is the x264
    quality (lower is better). ``scenario`` is the scenario dict; it feeds the
    HUD timeline, the plan view and the manifest. ``composite_height`` sets
    the composite's height in pixels (default: the primary view's height).

    Use as a context manager, or call :meth:`close`; the manifest is written
    on close even when rendering stopped with an exception.
    """

    def __init__(self, out_dir: PathLike, run: RunInfo, *, views: Sequence[str],
                 overlay: bool = True, layout: str = "single", save_frames: str = "none",
                 crf: int = 18, scenario: Optional[Dict[str, Any]] = None,
                 style: Any = None, composite_height: Optional[int] = None,
                 run_id: Optional[str] = None) -> None:
        if layout not in LAYOUTS:
            raise ValueError(f"unknown layout {layout!r}; expected one of {LAYOUTS}")
        if save_frames not in SAVE_FRAME_FORMATS:
            raise ValueError(f"save_frames must be one of {SAVE_FRAME_FORMATS}")
        if not views:
            raise ValueError("Recorder needs at least one view")
        self.run = run
        self.views = list(views)
        self.overlay = bool(overlay)
        self.layout = layout
        self.save_frames = save_frames
        self.crf = int(crf)
        self.scenario = scenario
        self.style = style
        self.composite_height = composite_height
        self.run_id = run_id or f"{run.engine}_{run.scenario_id}_{run.weather}"
        self.run_dir = Path(out_dir) / self.run_id
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self.fps = float(run.fps or 30)
        self._ffmpeg = find_ffmpeg()
        self._sinks: Dict[str, VideoSink] = {}
        self._gt_files: Dict[str, Any] = {}
        self._cameras: Dict[str, Dict[str, Any]] = {}
        self._records: List[Dict[str, Any]] = []
        self._history: Deque[FrameRecord] = deque(maxlen=max(2, int(round(1.5 * self.fps)) + 1))
        self._stills: Dict[str, np.ndarray] = {}
        self._closest: Optional[float] = None
        self._timeline: Optional[Dict[str, Any]] = None
        self._missing: set = set()
        self._closed = False
        self._result: Optional[Dict[str, Any]] = None
        self._error: Optional[str] = None

    def __enter__(self) -> "Recorder":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        if exc is not None:
            self._error = f"{exc_type.__name__}: {exc}"
        self.close()

    def _sink(self, key: str) -> VideoSink:
        if key not in self._sinks:
            self._sinks[key] = VideoSink(self.run_dir / "video" / f"{key}.mp4", self.fps,
                                         self.crf, ffmpeg=self._ffmpeg)
        return self._sinks[key]

    def _gt(self, view: str):
        if view not in self._gt_files:
            path = self.run_dir / "gt" / f"{view}.jsonl"
            path.parent.mkdir(parents=True, exist_ok=True)
            self._gt_files[view] = open(path, "w")
        return self._gt_files[view]

    def _hud_timeline(self) -> Optional[Dict[str, Any]]:
        if self._timeline is None and self.scenario is not None:
            from .overlay import build_timeline

            self._timeline = build_timeline(self.scenario, int(round(self.fps)))
        return self._timeline

    def _draw(self, image: np.ndarray, frame: FrameRecord, view: str) -> np.ndarray:
        from .overlay import DEFAULT_STYLE, draw_overlay

        return draw_overlay(image, frame, view, run=self.run, scenario=self.scenario,
                            style=self.style or DEFAULT_STYLE, history=list(self._history),
                            timeline=self._hud_timeline())

    def _composite(self, frame: FrameRecord, shown: Dict[str, np.ndarray]) -> np.ndarray:
        primary = shown[self.views[0]] if self.views[0] in shown else next(iter(shown.values()))
        h, w = primary.shape[:2]
        height = int(self.composite_height or h)
        if self.layout == "single":
            size = (int(round(w * height / h / 2.0)) * 2, height)
            return compose("single", [primary], size=size)
        from .overlay import DEFAULT_STYLE, draw_bev

        side = height - 2 * max(2, int(round(0.015 * height)))
        bev = draw_bev(frame, run=self.run, scenario=self.scenario, size=(side, side),
                       style=self.style or DEFAULT_STYLE, history=list(self._history),
                       timeline=self._hud_timeline())
        images = [shown[v] for v in self.views if v in shown][:3]
        size = fit_size(self.layout, (w, h), height)
        return compose(self.layout, images, bev, size=size)

    def write(self, frame: FrameRecord) -> None:
        """Record one frame: videos, ground truth, saved frames, still candidates."""
        if self._closed:
            raise RuntimeError("Recorder is closed")
        shown: Dict[str, np.ndarray] = {}
        for view in self.views:
            image = frame.images.get(view)
            if image is None:
                if view not in self._missing:
                    _LOG.warning("frame %d has no image for view %r", frame.index, view)
                    self._missing.add(view)
                continue
            cam = frame.cameras.get(view)
            if cam is not None and view not in self._cameras:
                self._cameras[view] = cam.to_dict()
            self._sink(view).write(image)
            h, w = image.shape[:2]
            line = {"index": frame.index, "t": round(float(frame.t), 4), "width": w, "height": h,
                    "objects": frame_ground_truth(frame, view)}
            self._gt(view).write(json.dumps(_json_safe(line)) + "\n")
            if self.save_frames != "none":
                folder = self.run_dir / "frames" / view
                folder.mkdir(parents=True, exist_ok=True)
                params = [cv2.IMWRITE_JPEG_QUALITY, 95] if self.save_frames == "jpg" else []
                cv2.imwrite(str(folder / f"{frame.index:06d}.{self.save_frames}"), image, params)
            if self.overlay:
                annotated = self._draw(image, frame, view)
                self._sink(f"{view}_overlay").write(annotated)
                shown[view] = annotated
            else:
                shown[view] = image
        if shown:
            composite = self._composite(frame, shown)
            self._sink("composite").write(composite)
            self._note_stills(frame, composite)
        self._records.append(frame.to_dict())
        self._history.append(replace(frame, images={}, aux={}))

    def _note_stills(self, frame: FrameRecord, composite: np.ndarray) -> None:
        if "first_frame" not in self._stills:
            self._stills["first_frame"] = composite.copy()
        if frame.state == "alert" and "first_alert" not in self._stills:
            self._stills["first_alert"] = composite.copy()
        pair = closest_pair(frame)
        if pair is not None and (self._closest is None or pair[2] < self._closest):
            self._closest = pair[2]
            self._stills["closest_approach"] = composite.copy()

    def timeline_summary(self) -> Dict[str, Any]:
        """Frame counts per state, alert and danger intervals, first alert and lead time.

        ``lead_time_s`` is the onset of the first ground-truth danger interval
        minus the first alert time (positive when the alert came first); it is
        ``None`` unless both exist. Times are seconds of scenario time.
        """
        from .types import STATES

        times = [r["t"] for r in self._records]
        states = [r["state"] for r in self._records]
        danger = [bool((r.get("danger") or {}).get("is_danger")) for r in self._records]
        actionable = [bool((r.get("danger") or {}).get("actionable")) for r in self._records]
        dt = 1.0 / self.fps
        alert = _intervals(times, [s == "alert" for s in states], dt)
        danger_iv = _intervals(times, danger, dt)
        first_alert = alert[0][0] if alert else None
        first_danger = danger_iv[0][0] if danger_iv else None
        closest = None
        best = None
        for r in self._records:
            people = [a for a in r["agents"] if a["kind"] == "person" and a["role"] == "scripted"]
            bikes = [a for a in r["agents"] if a["kind"] == "bike" and a["role"] == "scripted"]
            for b in bikes:
                for p in people:
                    d = math.hypot(b["sx"] - p["sx"], b["sy"] - p["sy"])
                    if best is None or d < best:
                        best = d
                        closest = {"index": r["index"], "t": r["t"], "bike_id": b["id"],
                                   "person_id": p["id"], "distance_m": round(d, 3)}
        return {
            "frames": len(self._records),
            "state_counts": {s: states.count(s) for s in STATES},
            "alert_intervals": alert,
            "warning_intervals": _intervals(times, [s == "warning" for s in states], dt),
            "danger_intervals": danger_iv,
            "actionable_intervals": _intervals(times, actionable, dt),
            "first_alert_t": first_alert,
            "first_danger_t": first_danger,
            "lead_time_s": (round(first_danger - first_alert, 4)
                            if first_alert is not None and first_danger is not None else None),
            "closest_approach": closest,
        }

    def _write_stills(self) -> List[str]:
        paths: List[str] = []
        if not self._stills:
            return paths
        folder = self.run_dir / "stills"
        folder.mkdir(parents=True, exist_ok=True)
        for key, image in self._stills.items():
            jpg, png = folder / f"{key}.jpg", folder / f"{key}.png"
            cv2.imwrite(str(jpg), image, [cv2.IMWRITE_JPEG_QUALITY, 95])
            cv2.imwrite(str(png), image, [cv2.IMWRITE_PNG_COMPRESSION, 3])
            paths += [str(jpg), str(png)]
        return paths

    def close(self) -> Dict[str, Any]:
        """Finish all files and write the manifest; returns paths and counts.

        Returns ``{'run_dir', 'manifest', 'videos', 'stills', 'frames'}``.
        Calling it again returns the same result.
        """
        if self._result is not None:
            return self._result
        self._closed = True
        for sink in self._sinks.values():
            try:
                sink.close()
            except Exception as exc:
                _LOG.error("closing %s failed: %s", sink.path, exc)
        for fh in self._gt_files.values():
            fh.close()
        stills = self._write_stills()
        videos = {k: str(s.path.relative_to(self.run_dir)) for k, s in self._sinks.items()
                  if s.frames > 0}
        frames_dir = {}
        if self.save_frames != "none":
            frames_dir = {v: f"frames/{v}" for v in self.views
                          if (self.run_dir / "frames" / v).is_dir()}
        manifest = {
            "run_id": self.run_id,
            "run": self.run.to_dict(),
            "scenario": self.scenario,
            "layout": self.layout,
            "overlay": self.overlay,
            "views": self.views,
            "cameras": self._cameras,
            "videos": videos,
            "frames_dir": frames_dir,
            "frame_ext": None if self.save_frames == "none" else self.save_frames,
            "gt": {v: f"gt/{v}.jsonl" for v in self._gt_files},
            "stills": [str(Path(p).relative_to(self.run_dir)) for p in stills],
            "timeline": self.timeline_summary(),
            "complete": self._error is None,
            "error": self._error,
            "frame_count": len(self._records),
            "frames": self._records,
        }
        path = self.run_dir / "manifest.json"
        with open(path, "w") as fh:
            json.dump(_json_safe(manifest), fh, indent=1)
        self._result = {
            "run_dir": str(self.run_dir),
            "manifest": str(path),
            "videos": [str(self.run_dir / v) for v in videos.values()],
            "stills": stills,
            "frames": len(self._records),
        }
        return self._result
