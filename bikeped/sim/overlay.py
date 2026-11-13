"""HUD overlay and schematic top-down panel for rendered scenario frames.

``draw_overlay`` annotates one camera view with 3D boxes, id chips, ground
trails, the bike-pedestrian connector, the decision state badge and a timeline
of ground-truth danger against the system's alerts. ``draw_bev`` renders the
same frame as a plan view in the scenario frame. Both work on BGR ``uint8``
arrays, return new arrays and scale every size with the image height, so 720p
and 4K output look the same.

Pixel coordinates follow ``bikeped.sim.projection``: continuous, with pixel
``i`` covering ``[i, i + 1)``.

Run ``python -m bikeped.sim.overlay --demo OUT_DIR`` to render a still and a
short clip over a synthetic backdrop without a simulator.
"""

from __future__ import annotations

import argparse
import logging
import math
import os
import sys
from collections import OrderedDict
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np

from .groundtruth import closest_pair
from .projection import (
    BOX_BASE,
    BOX_EDGES,
    box_corners_world,
    carla_rotation_to_basis,
    pixel_rays,
    pixel_to_ground,
    project_box,
    project_points,
    project_polyline,
)
from .types import AgentState, CameraModel, FrameRecord, RunInfo

try:
    from PIL import Image, ImageDraw, ImageFont
except ImportError:  # Pillow is optional; text falls back to Hershey strokes.
    Image = ImageDraw = ImageFont = None

_LOG = logging.getLogger(__name__)

BGR = Tuple[int, int, int]
Rect = Tuple[float, float, float, float]

_SHIFT = 4
_ONE = 1 << _SHIFT
_STATE_LABELS = {"idle": "IDLE", "safe": "SAFE", "warning": "WARNING", "alert": "ALERT"}
_KIND_LETTER = {"person": "P", "bike": "B"}
_ROAD_HALF_WIDTH_M = 4.0
_CROSSWALK_X_M = 5.0
_CROSSWALK_HALF_M = 1.5


def _hex_to_bgr(value: str) -> BGR:
    value = value.lstrip("#")
    return int(value[4:6], 16), int(value[2:4], 16), int(value[0:2], 16)


def _mix(a: BGR, b: BGR, t: float) -> BGR:
    return tuple(int(round(a[i] + (b[i] - a[i]) * t)) for i in range(3))


@dataclass(frozen=True)
class OverlayStyle:
    """Appearance of the HUD and of the top-down panel.

    Colours are ``#rrggbb`` strings and match the project site. The boolean
    fields switch single elements. ``rings`` forces the range rings on for any
    view; without it they are drawn only on the views named in ``ring_views``,
    where the camera footprint is the sensor position. ``scale`` multiplies
    every size on top of the automatic scaling with image height, which is
    useful when a view is shown small inside a composite.
    """

    background: str = "#0a0b0d"
    text: str = "#ecebe8"
    muted: str = "#9b9ca3"
    cyclist: str = "#ffb547"
    pedestrian: str = "#6fe3ff"
    alert: str = "#ff5a4f"
    warning: str = "#ffb547"
    safe: str = "#5ad19a"
    idle: str = "#6c6e76"
    font_path: Optional[str] = None
    boxes: bool = True
    labels: bool = True
    trails: bool = True
    connector: bool = True
    rings: bool = False
    ring_views: Tuple[str, ...] = ("fisheye",)
    badge: bool = True
    timeline: bool = True
    caption: bool = True
    trail_seconds: float = 1.5
    connector_max_m: float = 30.0
    scale: float = 1.0

    def bgr(self, name: str) -> BGR:
        """BGR tuple of the palette entry ``name``."""
        return _hex_to_bgr(getattr(self, name))

    def state_color(self, state: str) -> BGR:
        """Colour of a decision state; unknown states use the idle colour."""
        return self.bgr(state if state in _STATE_LABELS else "idle")

    def agent_color(self, kind: str, role: str = "scripted") -> BGR:
        """Colour of an agent: cyan pedestrians, amber cyclists, muted ambient traffic."""
        if role != "scripted":
            return self.bgr("muted")
        return self.bgr("pedestrian" if kind == "person" else "cyclist")


DEFAULT_STYLE = OverlayStyle()


def _font_candidates() -> List[Tuple[str, str]]:
    pairs: List[Tuple[str, str]] = []
    try:
        import matplotlib

        base = os.path.join(matplotlib.get_data_path(), "fonts", "ttf")
        pairs.append((os.path.join(base, "DejaVuSans.ttf"),
                      os.path.join(base, "DejaVuSans-Bold.ttf")))
    except ImportError:
        pass
    for folder in (
        "/usr/share/fonts/truetype/dejavu",
        "/usr/share/fonts/dejavu",
        "/usr/share/fonts/TTF",
        "/usr/share/fonts/dejavu-sans-fonts",
    ):
        pairs.append((f"{folder}/DejaVuSans.ttf", f"{folder}/DejaVuSans-Bold.ttf"))
    pairs += [
        ("/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf",
         "/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf"),
        ("/System/Library/Fonts/Supplemental/Arial.ttf",
         "/System/Library/Fonts/Supplemental/Arial Bold.ttf"),
        ("/Library/Fonts/Arial.ttf", "/Library/Fonts/Arial Bold.ttf"),
        ("C:/Windows/Fonts/segoeui.ttf", "C:/Windows/Fonts/segoeuib.ttf"),
        ("C:/Windows/Fonts/arial.ttf", "C:/Windows/Fonts/arialbd.ttf"),
    ]
    return pairs


class _Fonts:
    """Rasterises text to alpha masks, with Pillow when possible and Hershey otherwise."""

    def __init__(self, font_path: Optional[str]) -> None:
        self.regular: Optional[str] = None
        self.bold: Optional[str] = None
        if ImageFont is not None:
            candidates = _font_candidates()
            if font_path:
                stem, ext = os.path.splitext(font_path)
                candidates.insert(0, (font_path, f"{stem}-Bold{ext}"))
            for regular, bold in candidates:
                if os.path.isfile(regular):
                    self.regular = regular
                    self.bold = bold if os.path.isfile(bold) else regular
                    break
        self._faces: Dict[Tuple[bool, int], Any] = {}
        self._masks: "OrderedDict[Tuple[str, int, bool], np.ndarray]" = OrderedDict()

    @property
    def truetype(self) -> bool:
        return self.regular is not None

    def mask(self, text: str, px: int, bold: bool = False) -> np.ndarray:
        """Alpha mask of one line of text, ``(line_height, width)`` ``uint8``."""
        key = (text, px, bold)
        cached = self._masks.get(key)
        if cached is not None:
            self._masks.move_to_end(key)
            return cached
        mask = self._render_truetype(text, px, bold) if self.truetype else self._render_hershey(
            text, px, bold)
        self._masks[key] = mask
        if len(self._masks) > 512:
            self._masks.popitem(last=False)
        return mask

    def _render_truetype(self, text: str, px: int, bold: bool) -> np.ndarray:
        face = self._faces.get((bold, px))
        if face is None:
            face = ImageFont.truetype(self.bold if bold else self.regular, px)
            self._faces[(bold, px)] = face
        ascent, descent = face.getmetrics()
        left, _, right, _ = face.getbbox(text or " ")
        origin = 1 - min(left, 0)
        width = max(1, int(math.ceil(right)) + origin + 1)
        canvas = Image.new("L", (width, ascent + descent), 0)
        ImageDraw.Draw(canvas).text((origin, 0), text, font=face, fill=255)
        return np.asarray(canvas, dtype=np.uint8)

    @staticmethod
    def _render_hershey(text: str, px: int, bold: bool) -> np.ndarray:
        text = text.encode("ascii", "replace").decode("ascii")
        scale = px / 30.0
        thickness = max(1, int(round(px / 14.0))) + (1 if bold else 0)
        (width, height), baseline = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, scale,
                                                    thickness)
        pad = thickness + 1
        line = int(round(px * 1.17))
        mask = np.zeros((max(line, height + baseline + 2 * pad), width + 2 * pad), np.uint8)
        origin_y = int(round(px * 0.93))
        cv2.putText(mask, text, (pad, origin_y), cv2.FONT_HERSHEY_SIMPLEX, scale, 255, thickness,
                    cv2.LINE_AA)
        return mask


_FONTS: Dict[Optional[str], _Fonts] = {}


def _fonts(style: OverlayStyle) -> _Fonts:
    fonts = _FONTS.get(style.font_path)
    if fonts is None:
        fonts = _FONTS[style.font_path] = _Fonts(style.font_path)
        if not fonts.truetype:
            _LOG.warning("no TrueType font found; HUD text uses OpenCV Hershey strokes")
    return fonts


def _fixed(points: Any, offset: Tuple[float, float] = (0.0, 0.0)) -> np.ndarray:
    """Continuous pixel coordinates to OpenCV fixed-point centres relative to ``offset``."""
    pts = np.asarray(points, dtype=np.float64).reshape(-1, 2) - np.asarray(offset) - 0.5
    return np.round(np.clip(pts, -30000.0, 30000.0) * _ONE).astype(np.int32)


def _rounded_rect(x: float, y: float, w: float, h: float, r: float) -> np.ndarray:
    r = max(0.0, min(r, w / 2.0, h / 2.0))
    arc = np.linspace(0.0, np.pi / 2.0, 7)
    parts = []
    for cx, cy, start in (
        (x + w - r, y + r, -np.pi / 2.0),
        (x + w - r, y + h - r, 0.0),
        (x + r, y + h - r, np.pi / 2.0),
        (x + r, y + r, np.pi),
    ):
        parts.append(np.stack([cx + r * np.cos(start + arc), cy + r * np.sin(start + arc)], axis=1))
    return np.concatenate(parts)


def _overlaps(a: Rect, b: Rect, pad: float = 0.0) -> bool:
    return not (a[2] + pad <= b[0] or b[2] + pad <= a[0]
                or a[3] + pad <= b[1] or b[3] + pad <= a[1])


class _Canvas:
    """Anti-aliased drawing primitives on a BGR image, sized relative to its height."""

    def __init__(self, image: np.ndarray, style: OverlayStyle) -> None:
        self.img = image
        self.h, self.w = image.shape[:2]
        self.style = style
        self.s = style.scale * self.h / 1080.0
        self.fonts = _fonts(style)

    def px(self, value: float, minimum: int = 1) -> int:
        """``value`` design pixels at 1080p, scaled to this image."""
        return max(minimum, int(round(value * self.s)))

    def _blend(self, bounds: Rect, alpha: float,
               draw: Callable[[np.ndarray, Tuple[float, float]], None]) -> None:
        if alpha <= 0.0:
            return
        if alpha >= 0.995:
            draw(self.img, (0.0, 0.0))
            return
        x0 = int(max(0, math.floor(bounds[0]) - 3))
        y0 = int(max(0, math.floor(bounds[1]) - 3))
        x1 = int(min(self.w, math.ceil(bounds[2]) + 3))
        y1 = int(min(self.h, math.ceil(bounds[3]) + 3))
        if x1 <= x0 or y1 <= y0:
            return
        region = self.img[y0:y1, x0:x1]
        layer = region.copy()
        draw(layer, (float(x0), float(y0)))
        region[:] = cv2.addWeighted(layer, alpha, region, 1.0 - alpha, 0.0)

    @staticmethod
    def _bounds(arrays: Sequence[np.ndarray], pad: float) -> Rect:
        stacked = np.concatenate([np.asarray(a, dtype=np.float64).reshape(-1, 2) for a in arrays])
        return (stacked[:, 0].min() - pad, stacked[:, 1].min() - pad,
                stacked[:, 0].max() + pad, stacked[:, 1].max() + pad)

    def lines(self, polylines: Sequence[np.ndarray], color: BGR, thickness: int = 1,
              alpha: float = 1.0, closed: bool = False) -> None:
        """Stroke image polylines given as ``(M, 2)`` arrays."""
        polylines = [np.asarray(p, dtype=np.float64) for p in polylines if len(p) >= 2]
        if not polylines:
            return

        def draw(img: np.ndarray, offset: Tuple[float, float]) -> None:
            cv2.polylines(img, [_fixed(p, offset) for p in polylines], closed, color, thickness,
                          cv2.LINE_AA, _SHIFT)

        self._blend(self._bounds(polylines, thickness), alpha, draw)

    def polygon(self, points: np.ndarray, color: BGR, alpha: float = 1.0) -> None:
        """Fill one image polygon."""
        points = np.asarray(points, dtype=np.float64)
        if len(points) < 3:
            return

        def draw(img: np.ndarray, offset: Tuple[float, float]) -> None:
            cv2.fillPoly(img, [_fixed(points, offset)], color, cv2.LINE_AA, _SHIFT)

        self._blend(self._bounds([points], 1.0), alpha, draw)

    def circle(self, center: Tuple[float, float], radius: float, color: BGR, alpha: float = 1.0,
               thickness: int = -1) -> None:
        """Filled (``thickness=-1``) or stroked circle."""
        cx, cy = center
        pad = radius + max(thickness, 0) + 1

        def draw(img: np.ndarray, offset: Tuple[float, float]) -> None:
            c = _fixed([[cx, cy]], offset)[0]
            cv2.circle(img, (int(c[0]), int(c[1])), int(round(radius * _ONE)), color, thickness,
                       cv2.LINE_AA, _SHIFT)

        self._blend((cx - pad, cy - pad, cx + pad, cy + pad), alpha, draw)

    def plate(self, rect: Rect, color: BGR, alpha: float = 1.0, radius: Optional[float] = None,
              ) -> None:
        """Rounded rectangle ``(x0, y0, x1, y1)``."""
        x0, y0, x1, y1 = rect
        r = (y1 - y0) / 2.0 if radius is None else radius
        self.polygon(_rounded_rect(x0, y0, x1 - x0, y1 - y0, r), color, alpha)

    def text_size(self, text: str, px: int, bold: bool = False) -> Tuple[int, int]:
        """Width and line height of ``text`` in pixels."""
        mask = self.fonts.mask(text, px, bold)
        return mask.shape[1], mask.shape[0]

    def text(self, x: float, y: float, text: str, px: int, color: BGR, bold: bool = False,
             alpha: float = 1.0) -> Tuple[int, int]:
        """Draw one line with the top-left of its line box at ``(x, y)``."""
        mask = self.fonts.mask(text, px, bold)
        mh, mw = mask.shape
        x0, y0 = int(round(x)), int(round(y))
        sx0, sy0 = max(0, -x0), max(0, -y0)
        sx1, sy1 = min(mw, self.w - x0), min(mh, self.h - y0)
        if sx1 > sx0 and sy1 > sy0:
            region = self.img[y0 + sy0:y0 + sy1, x0 + sx0:x0 + sx1]
            a = mask[sy0:sy1, sx0:sx1].astype(np.float32)[..., None] * (alpha / 255.0)
            blended = region.astype(np.float32) * (1.0 - a) + np.asarray(color, np.float32) * a
            region[:] = np.clip(blended + 0.5, 0, 255).astype(np.uint8)
        return mw, mh

    def text_centered(self, rect: Rect, text: str, px: int, color: BGR, bold: bool = False,
                      ) -> None:
        """Draw one line centred in ``rect``, optically centred on the capitals."""
        tw, th = self.text_size(text, px, bold)
        x = (rect[0] + rect[2] - tw) / 2.0
        y = (rect[1] + rect[3] - th) / 2.0 - 0.02 * px
        self.text(x, y, text, px, color, bold)

    def scrim(self, y0: int, y1: int, alpha0: float, alpha1: float) -> None:
        """Darken rows ``y0..y1`` towards the background colour with a vertical ramp."""
        y0, y1 = max(0, y0), min(self.h, y1)
        if y1 <= y0:
            return
        ramp = np.linspace(alpha0, alpha1, y1 - y0, dtype=np.float32)[:, None, None]
        region = self.img[y0:y1]
        bg = np.asarray(self.style.bgr("background"), np.float32)
        region[:] = (region.astype(np.float32) * (1.0 - ramp) + bg * ramp + 0.5).astype(np.uint8)


def _testbench() -> Any:
    """The top-level ``decision_testbench`` module, importable from the repository root."""
    try:
        import decision_testbench
    except ImportError:
        root = str(Path(__file__).resolve().parents[2])
        if root not in sys.path:
            sys.path.insert(0, root)
        import decision_testbench
    return decision_testbench


def _intervals(flags: Sequence[bool], dt: float) -> List[List[float]]:
    out: List[List[float]] = []
    start: Optional[int] = None
    for i, flag in enumerate(list(flags) + [False]):
        if flag and start is None:
            start = i
        elif not flag and start is not None:
            out.append([round(start * dt, 4), round(i * dt, 4)])
            start = None
    return out


_TIMELINES: Dict[Tuple[str, int], Dict[str, Any]] = {}
_DANGER: Dict[Tuple[str, int, int], Tuple[List[bool], List[bool]]] = {}


def _scenario_key(scenario: Dict[str, Any]) -> str:
    return str(scenario.get("id") or scenario["name"])


def build_timeline(scenario: Dict[str, Any], fps: int,
                   states: Optional[Sequence[str]] = None) -> Dict[str, Any]:
    """Precompute what the timeline strip shows for one scenario.

    Frame ``i`` is at ``t = i / fps``. Ground-truth danger comes from
    ``decision_testbench.compute_ground_truth_danger``. ``states`` is the
    decision state per frame as recorded by a run; when it is ``None`` the
    states come from ``decision_testbench.simulate_scenario`` with the default
    pipeline configuration and no detection noise, and the result is cached
    per scenario and frame rate.

    Returns a dict with ``duration`` (s), ``fps``, ``n`` (frames), ``states``,
    the interval lists ``danger``, ``actionable``, ``alert`` and ``warning``
    (``[t0, t1]`` pairs in seconds, ``t1`` exclusive), and ``first_alert_t``
    and ``first_danger_t`` (seconds or ``None``).
    """
    tb = _testbench()
    fps = int(fps)
    key = (_scenario_key(scenario), fps)
    if states is None and key in _TIMELINES:
        return _TIMELINES[key]
    dt = 1.0 / fps
    duration = float(tb.get_scenario_duration(scenario))
    if states is None:
        result = tb.simulate_scenario(scenario, tb.PIPELINE_CONFIGS["default"], fps=fps)
        state_list = [str(s) for s in result["states"]]
        danger = [bool(d) for d in result["dangers"]]
        actionable = [bool(a.actionable) for a in result["assessments"]]
        _DANGER[(key[0], fps, len(state_list))] = (danger, actionable)
    else:
        state_list = [str(s) for s in states]
        cached = _DANGER.get((key[0], fps, len(state_list)))
        if cached is None:
            assessments = [tb.compute_ground_truth_danger(scenario, i * dt)
                           for i in range(len(state_list))]
            cached = ([bool(a.is_danger) for a in assessments],
                      [bool(a.actionable) for a in assessments])
            _DANGER[(key[0], fps, len(state_list))] = cached
        danger, actionable = cached
    n = len(state_list)
    alert = [s == "alert" for s in state_list]
    timeline = {
        "duration": max(duration, n * dt),
        "fps": fps,
        "n": n,
        "states": state_list,
        "danger": _intervals(danger, dt),
        "actionable": _intervals(actionable, dt),
        "alert": _intervals(alert, dt),
        "warning": _intervals([s == "warning" for s in state_list], dt),
        "first_alert_t": round(alert.index(True) * dt, 4) if any(alert) else None,
        "first_danger_t": round(danger.index(True) * dt, 4) if any(danger) else None,
    }
    if states is None:
        _TIMELINES[key] = timeline
    return timeline


def _bottom_center(agent: AgentState) -> np.ndarray:
    """World position of the centre of the bottom face of the agent's box."""
    up = np.asarray(carla_rotation_to_basis(*agent.rotation)[2], dtype=np.float64)
    return np.asarray(agent.location, dtype=np.float64) - up * agent.extent[2]


def _chip_text(agent: AgentState) -> str:
    return f"{_KIND_LETTER.get(agent.kind, agent.kind[:1].upper())}{agent.id}"


def _place_chip(canvas: _Canvas, size: Tuple[float, float], bbox: Sequence[float],
                occupied: List[Rect]) -> Rect:
    """First free spot for an id chip around a 2D box, trying above, the sides, then below."""
    cw, ch = size
    gap = canvas.px(6)
    x1, y1, x2, y2 = bbox
    ax = (x1 + x2) / 2.0
    spots = [(ax - cw / 2.0, y1 - gap - ch - k * (ch + canvas.px(4))) for k in range(4)]
    spots += [(x2 + gap, y1), (x1 - gap - cw, y1), (ax - cw / 2.0, y2 + gap)]
    edge = canvas.px(6)
    rects = []
    for x, y in spots:
        x = min(max(x, edge), canvas.w - cw - edge)
        y = min(max(y, edge), canvas.h - ch - edge)
        rects.append((x, y, x + cw, y + ch))
    for rect in rects:
        if not any(_overlaps(rect, other, canvas.px(3)) for other in occupied):
            return rect
    return rects[0]


def _draw_chip(canvas: _Canvas, rect: Rect, text: str, color: BGR, outline: Optional[BGR] = None,
               ) -> None:
    style = canvas.style
    if outline is not None:
        grow = canvas.px(2)
        canvas.plate((rect[0] - grow, rect[1] - grow, rect[2] + grow, rect[3] + grow), outline,
                     radius=canvas.px(7))
    canvas.plate(rect, color, radius=canvas.px(5))
    canvas.text_centered(rect, text, canvas.px(14, 9), style.bgr("background"), bold=True)


def _draw_badge(canvas: _Canvas, x: float, y: float, state: str, align_right: bool = False,
                ) -> Rect:
    """State light and label on a dark pill; returns the pill rectangle."""
    style = canvas.style
    label = _STATE_LABELS.get(state, state.upper())
    font = canvas.px(19, 10)
    height = canvas.px(42, 20)
    pad = canvas.px(16)
    light = canvas.px(7, 3)
    tw, _ = canvas.text_size(label, font, bold=True)
    width = pad + 2 * light + canvas.px(11) + tw + pad
    if align_right:
        x -= width
    rect = (x, y, x + width, y + height)
    color = style.state_color(state)
    canvas.plate(rect, style.bgr("background"), alpha=0.78)
    center = (x + pad + light, y + height / 2.0)
    canvas.circle(center, light * 2.1, color, alpha=0.22)
    canvas.circle(center, light, color)
    text_rect = (x + pad + 2 * light + canvas.px(11), y, x + width - pad, y + height)
    canvas.text_centered(text_rect, label, font, style.bgr("text"), bold=True)
    return rect


def _draw_timeline(canvas: _Canvas, timeline: Dict[str, Any], t: float, x0: float, x1: float,
                   y: float) -> None:
    """Two lanes, ground-truth danger over system state, with a playhead at ``t``."""
    style = canvas.style
    lane = canvas.px(6, 3)
    gap = canvas.px(5, 2)
    font = canvas.px(11, 8)
    label_w = max(canvas.text_size(s, font, True)[0] for s in ("TRUTH", "SYSTEM"))
    track0 = x0 + label_w + canvas.px(12)
    duration = max(float(timeline["duration"]), 1e-6)

    def to_x(seconds: float) -> float:
        return track0 + (x1 - track0) * min(max(seconds / duration, 0.0), 1.0)

    rows = (
        ("TRUTH", y, (("danger", "alert", 0.42), ("actionable", "alert", 1.0))),
        ("SYSTEM", y + lane + gap, (("warning", "warning", 1.0), ("alert", "alert", 1.0))),
    )
    for label, top, layers in rows:
        _, th = canvas.text_size(label, font, True)
        canvas.text(x0, top + lane / 2.0 - th / 2.0, label, font, style.bgr("muted"), bold=True)
        canvas.plate((track0, top, x1, top + lane), style.bgr("text"), alpha=0.13)
        for key, color, alpha in layers:
            for start, end in timeline.get(key, ()):
                left, right = to_x(start), max(to_x(end), to_x(start) + 1.0)
                canvas.plate((left, top, right, top + lane), style.bgr(color), alpha=alpha,
                             radius=min(lane / 2.0, (right - left) / 2.0))
    head = to_x(t)
    half = canvas.px(2) / 2.0
    top, bottom = y - canvas.px(5), y + 2 * lane + gap + canvas.px(5)
    canvas.plate((head - half - 1, top - 1, head + half + 1, bottom + 1), style.bgr("background"),
                 alpha=0.6, radius=half + 1)
    canvas.plate((head - half, top, head + half, bottom), style.bgr("text"), radius=half)


def _draw_trails(canvas: _Canvas, frame: FrameRecord, cam: CameraModel,
                 history: Sequence[FrameRecord]) -> None:
    style = canvas.style
    recent = [f for f in history if frame.t - style.trail_seconds <= f.t < frame.t]
    if not recent:
        return
    bands = 5
    for agent in frame.agents:
        if agent.role != "scripted":
            continue
        points = [_bottom_center(a) for f in recent for a in f.agents if a.id == agent.id]
        points.append(_bottom_center(agent))
        if len(points) < 2:
            continue
        world = np.asarray(points)
        color = style.agent_color(agent.kind)
        edges = np.linspace(0, len(world) - 1, bands + 1).round().astype(int)
        for band in range(bands):
            chunk = world[edges[band]:edges[band + 1] + 1]
            if len(chunk) >= 2:
                canvas.lines(project_polyline(chunk, cam), color, canvas.px(3),
                             alpha=0.15 + 0.6 * (band + 1) / bands)


def _draw_box(canvas: _Canvas, agent: AgentState, cam: CameraModel, color: BGR,
              base_color: BGR, base_alpha: float) -> None:
    corners = box_corners_world(agent.location, agent.rotation, agent.extent)
    base_loop = corners[list(BOX_BASE)]
    base = project_polyline(base_loop, cam, closed=True)
    for piece in base:
        canvas.polygon(piece, base_color, alpha=base_alpha)
    upper = []
    for i, j in BOX_EDGES:
        if i in BOX_BASE and j in BOX_BASE:
            continue
        upper.extend(project_polyline(corners[[i, j]], cam))
    canvas.lines(upper, color, canvas.px(1.5), alpha=0.72)
    canvas.lines(base, color, canvas.px(2))


def _draw_rings(canvas: _Canvas, cam: CameraModel, ground_z: float,
                occupied: List[Rect], avoid: Sequence[Rect] = ()) -> None:
    """Rings of the default pipeline's proximity band on the ground below the camera.

    Each ring's label goes on the highest visible point of the ring that does
    not overlap ``occupied`` (HUD plates) or ``avoid`` (agent boxes).
    """
    style = canvas.style
    params = _testbench().PIPELINE_CONFIGS["default"]
    angles = np.linspace(0.0, 2.0 * np.pi, 181)[:-1]
    font = canvas.px(12, 8)
    for radius in (params.prox_min, params.prox_max):
        ring = np.stack([cam.position[0] + radius * np.cos(angles),
                         cam.position[1] + radius * np.sin(angles),
                         np.full_like(angles, ground_z)], axis=1)
        pieces = project_polyline(ring, cam, closed=True)
        canvas.lines(pieces, style.bgr("text"), canvas.px(1.5), alpha=0.42)
        inside = [p[(p[:, 0] > 0) & (p[:, 0] < canvas.w) & (p[:, 1] > 0) & (p[:, 1] < canvas.h)]
                  for p in pieces]
        inside = [p for p in inside if len(p)]
        if not inside:
            continue
        points = np.vstack(inside)
        points = points[np.argsort(points[:, 1])]
        label = f"{radius:g} m"
        tw, th = canvas.text_size(label, font)
        pad = canvas.px(6)
        blockers = list(occupied) + list(avoid)
        rect = None
        for anchor in points[::max(1, len(points) // 48)]:
            candidate = (anchor[0] - tw / 2.0 - pad, anchor[1] - th / 2.0 - pad / 2.0,
                         anchor[0] + tw / 2.0 + pad, anchor[1] + th / 2.0 + pad / 2.0)
            inside_image = candidate[0] >= 0 and candidate[2] <= canvas.w
            if inside_image and not any(_overlaps(candidate, o, pad) for o in blockers):
                rect = candidate
                break
        if rect is None:
            continue
        canvas.plate(rect, style.bgr("background"), alpha=0.7)
        canvas.text_centered(rect, label, font, style.bgr("muted"))
        occupied.append(rect)


def _pair(frame: FrameRecord) -> Optional[Tuple[AgentState, AgentState, float]]:
    """The bike and pedestrian to connect: the alert pair when set, else the closest pair."""
    scripted = {a.id: a for a in frame.agents if a.role == "scripted"}
    ids: Optional[Tuple[int, int]] = None
    if frame.alert_pair and all(i in scripted for i in frame.alert_pair):
        ids = (frame.alert_pair[0], frame.alert_pair[1])
    else:
        found = closest_pair(frame)
        if found is not None and found[0] in scripted and found[1] in scripted:
            ids = (found[0], found[1])
    if ids is None:
        return None
    a, b = scripted[ids[0]], scripted[ids[1]]
    return a, b, float(math.hypot(a.sx - b.sx, a.sy - b.sy))


def _connector_color(style: OverlayStyle, state: str) -> BGR:
    return style.state_color(state) if state in ("alert", "warning") else style.bgr("text")


def _caption_parts(frame: FrameRecord, run: RunInfo, sep: str) -> Tuple[str, str]:
    engine = {"ue4": "UE4", "ue5": "UE5"}.get(run.engine, run.engine.upper())
    details = [p for p in (engine, run.weather.replace("_", " ")) if p]
    return run.scenario_name, sep.join(details)


def draw_overlay(image: np.ndarray, frame: FrameRecord, view: str, *, run: RunInfo,
                 scenario: Optional[Dict[str, Any]] = None,
                 style: Optional[OverlayStyle] = None,
                 history: Optional[Sequence[FrameRecord]] = None,
                 timeline: Optional[Dict[str, Any]] = None) -> np.ndarray:
    """Draw the HUD over one rendered view and return the annotated copy.

    ``image`` is the BGR ``uint8`` frame of ``view`` and is not modified. The
    camera model is ``frame.cameras[view]``; without one only the screen-space
    elements (badge, caption, timeline) are drawn. ``history`` holds earlier
    frames, oldest first, and feeds the ground trails from the bottom centre of
    each agent's box in the world frame. ``timeline`` is the result of
    ``build_timeline``; when it is ``None`` and ``scenario`` is given it is
    built from the scenario at ``run.fps``.
    """
    style = style or DEFAULT_STYLE
    out = np.ascontiguousarray(image).copy()
    if out.ndim == 2:
        out = cv2.cvtColor(out, cv2.COLOR_GRAY2BGR)
    canvas = _Canvas(out, style)
    cam = frame.cameras.get(view)
    if timeline is None and scenario is not None and style.timeline:
        timeline = build_timeline(scenario, run.fps)
    show_timeline = style.timeline and timeline is not None

    margin = canvas.px(28)
    lanes_h = 2 * canvas.px(6, 3) + canvas.px(5, 2)
    timeline_y = canvas.h - margin - lanes_h
    caption_font = canvas.px(16, 9)
    caption_h = canvas.text_size("Ag", caption_font)[1]
    caption_y = (timeline_y - canvas.px(14) - caption_h) if show_timeline else (
        canvas.h - margin - caption_h)
    hud_top = caption_y if style.caption else (timeline_y if show_timeline else canvas.h)
    if style.caption or show_timeline:
        canvas.scrim(int(hud_top - canvas.px(70)), canvas.h, 0.0, 0.66)

    occupied: List[Rect] = []
    if style.caption or show_timeline:
        occupied.append((0.0, hud_top - canvas.px(6), float(canvas.w), float(canvas.h)))
    badge_h = canvas.px(42, 20)
    if style.badge:
        occupied.append((0.0, 0.0, margin + canvas.px(250), margin + badge_h + canvas.px(40)))

    scripted = [a for a in frame.agents if a.role == "scripted"]
    if cam is not None:
        ground_z = run.site.get("ground_z")
        if ground_z is None:
            ground_z = min((_bottom_center(a)[2] for a in frame.agents), default=0.0)
        if style.rings or view in style.ring_views:
            boxes = [project_box(a, cam) for a in frame.agents]
            _draw_rings(canvas, cam, float(ground_z), occupied,
                        [tuple(b["bbox"]) for b in boxes if b is not None])
        if style.trails and history:
            _draw_trails(canvas, frame, cam, history)

        pair = _pair(frame) if style.connector else None
        highlighted = set()
        if pair is not None and pair[2] <= style.connector_max_m:
            a, b, distance = pair
            color = _connector_color(style, frame.state)
            if frame.state in ("alert", "warning"):
                highlighted = {a.id, b.id}
            ends = np.stack([_bottom_center(a), _bottom_center(b)])
            pieces = project_polyline(ends, cam)
            canvas.lines(pieces, style.bgr("background"), canvas.px(5), alpha=0.45)
            canvas.lines(pieces, color, canvas.px(2))
            uv, valid, _ = project_points(ends.mean(axis=0, keepdims=True), cam)
            if valid[0] and 0 <= uv[0, 0] < canvas.w and 0 <= uv[0, 1] < canvas.h:
                label = f"{distance:.1f} m"
                font = canvas.px(15, 9)
                tw, th = canvas.text_size(label, font, bold=True)
                pad = canvas.px(9)
                rect = (uv[0, 0] - tw / 2.0 - pad, uv[0, 1] - th / 2.0 - canvas.px(4),
                        uv[0, 0] + tw / 2.0 + pad, uv[0, 1] + th / 2.0 + canvas.px(4))
                canvas.plate(rect, style.bgr("background"), alpha=0.86)
                canvas.text_centered(rect, label, font, color, bold=True)
                occupied.append(rect)

        cam_pos = np.asarray(cam.position, dtype=np.float64)
        ordered = sorted(frame.agents, reverse=True,
                         key=lambda ag: float(np.linalg.norm(np.asarray(ag.location) - cam_pos)))
        chips: List[Tuple[AgentState, Sequence[float]]] = []
        for agent in ordered:
            color = style.agent_color(agent.kind, agent.role)
            hot = agent.id in highlighted and agent.role == "scripted"
            if style.boxes:
                _draw_box(canvas, agent, cam, color,
                          style.state_color(frame.state) if hot else color, 0.34 if hot else 0.2)
            if style.labels and agent.role == "scripted":
                box = project_box(agent, cam)
                if box is not None:
                    chips.append((agent, box["bbox"]))
        font = canvas.px(14, 9)
        for agent, bbox in reversed(chips):
            text = _chip_text(agent)
            tw, _ = canvas.text_size(text, font, bold=True)
            size = (tw + 2 * canvas.px(8), canvas.px(22, 13))
            rect = _place_chip(canvas, size, bbox, occupied)
            occupied.append(rect)
            color = style.agent_color(agent.kind)
            anchor = ((bbox[0] + bbox[2]) / 2.0, bbox[1])
            foot = ((rect[0] + rect[2]) / 2.0, rect[3])
            if math.hypot(anchor[0] - foot[0], anchor[1] - foot[1]) > canvas.px(10):
                canvas.lines([np.array([foot, anchor])], color, canvas.px(1.25), alpha=0.6)
            outline = style.state_color(frame.state) if agent.id in highlighted else None
            _draw_chip(canvas, rect, text, color, outline)

    if style.badge:
        canvas.scrim(0, margin + badge_h + canvas.px(50), 0.3, 0.0)
        badge = _draw_badge(canvas, margin, margin, frame.state)
        danger = frame.danger or {}
        if danger.get("is_danger"):
            sep = " \u00b7 " if canvas.fonts.truetype else " | "
            note = "Ground truth: danger"
            if danger.get("ttc") is not None and math.isfinite(float(danger["ttc"])):
                note += f"{sep}TTC {float(danger['ttc']):.1f} s"
            font = canvas.px(13, 9)
            tw, th = canvas.text_size(note, font)
            pad = canvas.px(12)
            top = badge[3] + canvas.px(8)
            rect = (margin, top, margin + tw + 2 * pad, top + th + canvas.px(10))
            canvas.plate(rect, style.bgr("background"), alpha=0.7)
            canvas.text_centered(rect, note, font, style.bgr("alert"))

    if style.caption:
        sep = "  \u00b7  " if canvas.fonts.truetype else "  |  "
        title, details = _caption_parts(frame, run, sep)
        tw, _ = canvas.text(margin, caption_y, title, caption_font, style.bgr("text"), bold=True)
        if details:
            canvas.text(margin + tw, caption_y, sep + details, caption_font, style.bgr("muted"))
        total = timeline["duration"] if timeline else run.duration_s
        clock = f"{frame.t:5.2f} s" + (f" / {total:.2f} s" if total else "")
        cw, _ = canvas.text_size(clock, caption_font)
        canvas.text(canvas.w - margin - cw, caption_y, clock, caption_font, style.bgr("text"))
        tag = view.upper()
        vw, _ = canvas.text_size(tag, canvas.px(12, 8), bold=True)
        canvas.text(canvas.w - margin - cw - canvas.px(18) - vw,
                    caption_y + (caption_h - canvas.text_size(tag, canvas.px(12, 8), True)[1]) / 2,
                    tag, canvas.px(12, 8), style.bgr("muted"), bold=True)
    if show_timeline:
        _draw_timeline(canvas, timeline, frame.t, margin, canvas.w - margin, timeline_y)
    if scripted and cam is None:
        _LOG.debug("view %s has no camera model; drew screen-space HUD only", view)
    return out


def _camera_xy(run: RunInfo) -> Tuple[float, float]:
    xy = run.site.get("camera_xy") if run.site else None
    if xy is None:
        ref = _testbench().FisheyeCamera()
        return float(ref.cam_x), float(ref.cam_y)
    return float(xy[0]), float(xy[1])


def _bev_window(scenario: Optional[Dict[str, Any]], frame: FrameRecord,
                cam_xy: Tuple[float, float]) -> Tuple[float, float, float]:
    """Centre and half-size in metres of the square plan-view window."""
    xs, ys = [cam_xy[0], _CROSSWALK_X_M], [cam_xy[1], 0.0]
    if scenario is not None:
        for agent in scenario["agents"]:
            xs += [p["x"] for p in agent["path"]]
            ys += [p["y"] for p in agent["path"]]
    else:
        xs += [a.sx for a in frame.agents] + [-10.0, 30.0]
        ys += [a.sy for a in frame.agents]
    half = max(max(xs) - min(xs), max(ys) - min(ys), 24.0) / 2.0 + 3.0
    return (max(xs) + min(xs)) / 2.0, (max(ys) + min(ys)) / 2.0, half


_BEV_STATIC: "OrderedDict[Any, np.ndarray]" = OrderedDict()


def _bev_static(canvas_size: Tuple[int, int], style: OverlayStyle,
                scenario: Optional[Dict[str, Any]], window: Tuple[float, float, float],
                cam_xy: Tuple[float, float]) -> np.ndarray:
    """Background of the plan view: grid, road, crosswalk, rings, camera, planned paths."""
    width, height = canvas_size
    image = np.empty((height, width, 3), np.uint8)
    bg, fg = style.bgr("background"), style.bgr("text")
    image[:] = bg
    canvas = _Canvas(image, style)
    canvas.s = style.scale * min(width, height) / 1080.0
    to_px = _bev_mapper(canvas_size, window)
    xc, yc, half = window
    x0, x1, y0, y1 = xc - half * 2, xc + half * 2, yc - half * 2, yc + half * 2

    for g in np.arange(math.floor(x0 / 5.0) * 5.0, x1, 5.0):
        canvas.lines([to_px([[g, y0], [g, y1]])], _mix(bg, fg, 0.055), 1)
    for g in np.arange(math.floor(y0 / 5.0) * 5.0, y1, 5.0):
        canvas.lines([to_px([[x0, g], [x1, g]])], _mix(bg, fg, 0.055), 1)

    hw = _ROAD_HALF_WIDTH_M
    canvas.polygon(to_px([[x0, -hw], [x1, -hw], [x1, hw], [x0, hw]]), _mix(bg, fg, 0.085))
    for side in (-hw, hw):
        canvas.lines([to_px([[x0, side], [x1, side]])], _mix(bg, fg, 0.26), canvas.px(1.5))
    for start in np.arange(math.floor(x0 / 4.0) * 4.0, x1, 4.0):
        if abs(start + 1.0 - _CROSSWALK_X_M) > _CROSSWALK_HALF_M + 1.5:
            canvas.lines([to_px([[start, 0.0], [start + 2.0, 0.0]])], _mix(bg, fg, 0.2),
                         canvas.px(1.5))
    cx0, cx1 = _CROSSWALK_X_M - _CROSSWALK_HALF_M, _CROSSWALK_X_M + _CROSSWALK_HALF_M
    for stripe in np.arange(-hw + 0.25, hw - 0.25, 1.0):
        canvas.polygon(to_px([[cx0, stripe], [cx1, stripe], [cx1, stripe + 0.5],
                              [cx0, stripe + 0.5]]), _mix(bg, fg, 0.36))

    params = _testbench().PIPELINE_CONFIGS["default"]
    angles = np.linspace(0.0, 2.0 * np.pi, 241)
    font = canvas.px(13, 8)
    for radius in (params.prox_min, params.prox_max):
        ring = np.stack([cam_xy[0] + radius * np.cos(angles),
                         cam_xy[1] + radius * np.sin(angles)], axis=1)
        canvas.lines([to_px(ring)], style.bgr("muted"), canvas.px(1.25), alpha=0.55)
        for deg in (62.0, 90.0, 118.0, 35.0, 145.0):
            anchor = to_px([[cam_xy[0] + radius * math.cos(math.radians(deg)),
                             cam_xy[1] + radius * math.sin(math.radians(deg))]])[0]
            label = f"{radius:g} m"
            tw, th = canvas.text_size(label, font)
            rect = (anchor[0] - tw / 2.0 - canvas.px(6), anchor[1] - th / 2.0 - canvas.px(2),
                    anchor[0] + tw / 2.0 + canvas.px(6), anchor[1] + th / 2.0 + canvas.px(2))
            pad = canvas.px(40)
            if rect[0] > pad and rect[1] > pad and rect[2] < width - pad and rect[3] < height - pad:
                canvas.plate(rect, bg, radius=canvas.px(4))
                canvas.text_centered(rect, label, font, style.bgr("muted"))
                break

    if scenario is not None:
        for agent in scenario["agents"]:
            path = to_px([[p["x"], p["y"]] for p in agent["path"]])
            color = style.agent_color("person" if agent["type"] == "person" else "bike")
            canvas.lines([path], color, canvas.px(1.5), alpha=0.3)
            canvas.circle((path[0][0], path[0][1]), canvas.px(3), color, alpha=0.45)

    cam_px = to_px([cam_xy])[0]
    canvas.circle((cam_px[0], cam_px[1]), canvas.px(11), fg, thickness=canvas.px(2))
    canvas.circle((cam_px[0], cam_px[1]), canvas.px(4), fg)
    label_font = canvas.px(12, 8)
    canvas.text(cam_px[0] + canvas.px(18), cam_px[1] - canvas.text_size("CAMERA", label_font,
                                                                        True)[1] / 2.0,
                "CAMERA", label_font, style.bgr("muted"), bold=True)

    pad = canvas.px(28)
    canvas.text(pad, pad, "PLAN VIEW", canvas.px(13, 8), style.bgr("muted"), bold=True)
    scale = to_px([[0.0, 0.0], [5.0, 0.0]])
    bar = float(scale[1][0] - scale[0][0])
    bx1, by = width - pad, height - pad
    tick = canvas.px(5)
    canvas.lines([np.array([[bx1 - bar, by - tick], [bx1 - bar, by], [bx1, by], [bx1, by - tick]])],
                 style.bgr("muted"), canvas.px(1.5))
    tw, th = canvas.text_size("5 m", label_font)
    canvas.text(bx1 - bar / 2.0 - tw / 2.0, by - tick - th - canvas.px(2), "5 m", label_font,
                style.bgr("muted"))
    return image


def _bev_mapper(canvas_size: Tuple[int, int],
                window: Tuple[float, float, float]) -> Callable[[Any], np.ndarray]:
    """Map scenario ``(x, y)`` in metres to panel pixels, ``y`` pointing up."""
    width, height = canvas_size
    xc, yc, half = window
    scale = min(width, height) / (2.0 * half)

    def to_px(points: Any) -> np.ndarray:
        pts = np.asarray(points, dtype=np.float64).reshape(-1, 2)
        return np.stack([width / 2.0 + (pts[:, 0] - xc) * scale,
                         height / 2.0 - (pts[:, 1] - yc) * scale], axis=1)

    return to_px


def _bev_trail(frame: FrameRecord, agent: AgentState, scenario: Optional[Dict[str, Any]],
               history: Optional[Sequence[FrameRecord]], seconds: float) -> np.ndarray:
    """Recent scenario-frame positions of ``agent``, oldest first, ending at the current one."""
    points: List[Tuple[float, float]] = []
    definition = None
    if scenario is not None:
        definition = next((a for a in scenario["agents"] if a["id"] == agent.id), None)
    if definition is not None:
        tb = _testbench()
        for t in np.linspace(frame.t - seconds, frame.t, 16)[:-1]:
            pose = tb.interpolate_agent(definition, float(t))
            if pose is not None:
                points.append((pose["x"], pose["y"]))
    elif history:
        for past in history:
            if frame.t - seconds <= past.t < frame.t:
                points += [(a.sx, a.sy) for a in past.agents if a.id == agent.id]
    points.append((agent.sx, agent.sy))
    return np.asarray(points, dtype=np.float64)


def draw_bev(frame: FrameRecord, *, run: RunInfo, scenario: Optional[Dict[str, Any]],
             size: Tuple[int, int] = (1080, 1080), style: Optional[OverlayStyle] = None,
             history: Optional[Sequence[FrameRecord]] = None,
             timeline: Optional[Dict[str, Any]] = None) -> np.ndarray:
    """Render the frame as a schematic plan view in the scenario frame.

    ``size`` is ``(width, height)`` in pixels. The panel shows the road, the
    crosswalk at ``x = 5``, the camera marker at ``run.site['camera_xy']``
    (the testbench camera position when absent) with the proximity rings of
    the default pipeline, the scripted paths, and every agent as an oriented
    glyph with a short trail. Scenario ``x`` points right and ``y`` up. Trails
    come from the scenario paths when ``scenario`` is given and from
    ``history`` otherwise. ``timeline`` adds the timeline strip at the bottom.
    """
    style = style or DEFAULT_STYLE
    width, height = int(size[0]), int(size[1])
    cam_xy = _camera_xy(run)
    window = _bev_window(scenario, frame, cam_xy)
    key = (width, height, style, _scenario_key(scenario) if scenario else None,
           tuple(round(v, 3) for v in window), cam_xy)
    static = _BEV_STATIC.get(key)
    if static is None:
        static = _BEV_STATIC[key] = _bev_static((width, height), style, scenario, window, cam_xy)
        if len(_BEV_STATIC) > 8:
            _BEV_STATIC.popitem(last=False)
    out = static.copy()
    canvas = _Canvas(out, style)
    canvas.s = style.scale * min(width, height) / 1080.0
    to_px = _bev_mapper((width, height), window)
    metre = float(to_px([[1.0, 0.0]])[0][0] - to_px([[0.0, 0.0]])[0][0])
    bg = style.bgr("background")

    pair = _pair(frame) if style.connector else None
    highlighted = set()
    if pair is not None and pair[2] <= style.connector_max_m:
        a, b, distance = pair
        color = _connector_color(style, frame.state)
        if frame.state in ("alert", "warning"):
            highlighted = {a.id, b.id}
        ends = to_px([[a.sx, a.sy], [b.sx, b.sy]])
        canvas.lines([ends], color, canvas.px(2), alpha=0.9)
        mid = ends.mean(axis=0)
        label = f"{distance:.1f} m"
        font = canvas.px(15, 9)
        tw, th = canvas.text_size(label, font, bold=True)
        normal = np.array([-(ends[1][1] - ends[0][1]), ends[1][0] - ends[0][0]])
        length = float(np.linalg.norm(normal))
        if length > 1e-6:
            normal = normal / length
            if normal[1] > 0:
                normal = -normal
            mid = mid + normal * (th / 2.0 + canvas.px(12))
        rect = (mid[0] - tw / 2.0 - canvas.px(9), mid[1] - th / 2.0 - canvas.px(4),
                mid[0] + tw / 2.0 + canvas.px(9), mid[1] + th / 2.0 + canvas.px(4))
        canvas.plate(rect, bg, alpha=0.9)
        canvas.text_centered(rect, label, font, color, bold=True)

    for agent in frame.agents:
        color = style.agent_color(agent.kind, agent.role)
        if style.trails and agent.role == "scripted":
            trail = to_px(_bev_trail(frame, agent, scenario, history, style.trail_seconds))
            edges = np.linspace(0, len(trail) - 1, 6).round().astype(int)
            for band in range(5):
                chunk = trail[edges[band]:edges[band + 1] + 1]
                canvas.lines([chunk], color, canvas.px(4), alpha=0.18 + 0.62 * (band + 1) / 5)

    font = canvas.px(14, 9)
    for agent in frame.agents:
        color = style.agent_color(agent.kind, agent.role)
        center = to_px([[agent.sx, agent.sy]])[0]
        direction = np.array([math.cos(agent.heading), -math.sin(agent.heading)])
        side = np.array([-direction[1], direction[0]])
        if agent.kind == "person":
            radius = max(canvas.px(8), 0.3 * metre)
            reach = radius
            if agent.id in highlighted:
                canvas.circle((center[0], center[1]), radius * 2.1,
                              style.state_color(frame.state), thickness=canvas.px(2))
            canvas.circle((center[0], center[1]), radius + canvas.px(2), bg)
            canvas.circle((center[0], center[1]), radius, color)
            if agent.speed > 0.05:
                tip = center + direction * radius * 1.9
                canvas.lines([np.stack([center + direction * radius, tip])], color, canvas.px(3))
        else:
            length = max(canvas.px(30), 2.0 * agent.extent[0] * metre)
            half_w = max(canvas.px(7), agent.extent[1] * metre)
            reach = length / 2.0
            glyph = np.stack([center + direction * length / 2.0,
                              center - direction * length / 2.0 + side * half_w,
                              center - direction * length / 4.0,
                              center - direction * length / 2.0 - side * half_w])
            if agent.id in highlighted:
                canvas.circle((center[0], center[1]), length * 0.78,
                              style.state_color(frame.state), thickness=canvas.px(2))
            canvas.lines([glyph], bg, canvas.px(5), closed=True)
            canvas.polygon(glyph, color)
        if style.labels and agent.role == "scripted":
            text = _chip_text(agent)
            tw, _ = canvas.text_size(text, font, bold=True)
            cw, ch = tw + 2 * canvas.px(8), canvas.px(22, 13)
            x = center[0] + reach * 0.75 + canvas.px(8)
            y = center[1] - reach * 0.75 - canvas.px(8) - ch
            x = min(max(x, canvas.px(6)), width - cw - canvas.px(6))
            y = min(max(y, canvas.px(6)), height - ch - canvas.px(6))
            _draw_chip(canvas, (x, y, x + cw, y + ch), text, color)

    pad = canvas.px(28)
    if style.badge:
        _draw_badge(canvas, width - pad, pad, frame.state, align_right=True)
    if timeline is not None and style.timeline:
        lanes_h = 2 * canvas.px(6, 3) + canvas.px(5, 2)
        bar = float(to_px([[5.0, 0.0]])[0][0] - to_px([[0.0, 0.0]])[0][0])
        _draw_timeline(canvas, timeline, frame.t, pad, width - pad - bar - canvas.px(28),
                       height - pad - lanes_h)
    return out


def _find_scenario(scenario_id: str) -> Dict[str, Any]:
    """Look a scenario up by name in the conformance set, then in the scenario registry."""
    for scenario in _testbench().SCENARIOS:
        if scenario["name"] == scenario_id:
            return scenario
    try:
        from bikeped.scenarios import get_scenario
    except ImportError as exc:
        raise KeyError(f"unknown scenario {scenario_id!r}") from exc
    return get_scenario(scenario_id)


def _look_at(name: str, position: Sequence[float], target: Sequence[float], width: int,
             height: int, fov_deg: float) -> CameraModel:
    """Pinhole camera at ``position`` (world frame) aimed at ``target`` with no roll."""
    d = np.asarray(target, dtype=np.float64) - np.asarray(position, dtype=np.float64)
    pitch = math.degrees(math.atan2(d[2], math.hypot(d[0], d[1])))
    yaw = math.degrees(math.atan2(d[1], d[0]))
    forward, right, up = carla_rotation_to_basis(pitch, yaw, 0.0)
    return CameraModel(name=name, kind="pinhole", width=width, height=height,
                       position=tuple(float(v) for v in position),
                       forward=tuple(forward), right=tuple(right), up=tuple(up), fov_deg=fov_deg)


def _synthetic_backdrop(cam: CameraModel, supersample: int = 2) -> np.ndarray:
    """Analytic street scene for ``cam``: a road along world ``x`` with a zebra crossing.

    The ground is shaded per pixel from its world position (scenario ``y`` is
    minus world ``y``), with distance haze, and the sky is a vertical gradient.
    """
    w, h = cam.width * supersample, cam.height * supersample
    out = np.empty((h, w, 3), np.float32)
    haze = np.array([176.0, 166.0, 156.0], np.float32)
    zenith = np.array([138.0, 104.0, 78.0], np.float32)
    us = (np.arange(w) + 0.5) / supersample
    hw = _ROAD_HALF_WIDTH_M
    for r0 in range(0, h, 128):
        vs = (np.arange(r0, min(r0 + 128, h)) + 0.5) / supersample
        uu, vv = np.meshgrid(us, vs)
        pts, valid = pixel_to_ground(np.stack([uu.ravel(), vv.ravel()], axis=1), cam, 0.0)
        x = np.where(valid, pts[:, 0], 0.0)
        y = np.where(valid, -pts[:, 1], 0.0)
        dist = np.hypot(x - cam.position[0], pts[:, 1] - cam.position[1])
        dist = np.where(valid, dist, 1e6)
        ay = np.abs(y)
        col = np.empty((len(x), 3), np.float32)
        col[:] = (58.0, 74.0, 60.0)
        walk = (ay >= hw + 0.2) & (ay < hw + 3.2)
        col[walk] = (138.0, 136.0, 132.0)
        joints = walk & ((np.mod(x, 1.5) < 0.04) | (np.abs(ay - hw - 1.7) < 0.02))
        col[joints] = (112.0, 110.0, 106.0)
        col[(ay >= hw) & (ay < hw + 0.2)] = (172.0, 170.0, 166.0)
        road = ay < hw
        grain = 4.0 * np.sin(x * 1.7 + y * 2.3) * np.sin(x * 0.9 - y * 1.3)
        col[road] = np.array([66.0, 63.0, 61.0], np.float32) + grain[road, None]
        dx = x - _CROSSWALK_X_M
        dash = road & (ay < 0.07) & (np.mod(x, 4.0) < 2.0) & (np.abs(dx) > _CROSSWALK_HALF_M + 2.0)
        col[dash] = (96.0, 176.0, 208.0)
        stop = road & (((np.abs(dx + 3.3) < 0.18) & (y < 0))
                       | ((np.abs(dx - 3.3) < 0.18) & (y > 0)))
        col[stop] = (214.0, 216.0, 218.0)
        zebra = road & (np.abs(dx) < _CROSSWALK_HALF_M) & (ay < hw - 0.2) & (
            np.mod(y + hw - 0.25, 1.0) < 0.5)
        col[zebra] = (222.0, 224.0, 226.0)
        fog = (1.0 - np.exp(-dist / 85.0))[:, None].astype(np.float32)
        col = col * (1.0 - fog) + haze * fog
        # Shade the sky by ray elevation, not image row, so that cube faces
        # rendered separately join without seams after the fisheye remap.
        rays, _ = pixel_rays(np.stack([uu.ravel(), vv.ravel()], axis=1), cam)
        elevation = np.clip(rays[:, 2], 0.0, 1.0)
        sky = haze + (zenith - haze) * np.clip(elevation * 2.2, 0.0, 1.0)[:, None] ** 0.8
        col = np.where(valid[:, None], col, sky)
        out[r0:r0 + len(vs)] = col.reshape(len(vs), w, 3)
    image = np.clip(out, 0, 255).astype(np.uint8)
    if supersample > 1:
        image = cv2.resize(image, (cam.width, cam.height), interpolation=cv2.INTER_AREA)
    return image


_BOX_FACES = ((0, 1, 3, 2), (4, 5, 7, 6), (0, 1, 5, 4), (2, 3, 7, 6), (0, 2, 6, 4), (1, 3, 7, 5))


def _paint_stand_ins(image: np.ndarray, frame: FrameRecord, cam: CameraModel) -> None:
    """Paint each agent as a shaded solid box with a ground shadow, far to near."""
    canvas = _Canvas(image, DEFAULT_STYLE)
    cam_pos = np.asarray(cam.position, dtype=np.float64)
    agents = sorted(frame.agents, reverse=True,
                    key=lambda ag: float(np.linalg.norm(np.asarray(ag.location) - cam_pos)))
    for agent in agents:
        corners = box_corners_world(agent.location, agent.rotation, agent.extent)
        uv, valid, depth = project_points(corners, cam)
        if not valid.all():
            continue
        base = corners[list(BOX_BASE)]
        shadow = base.mean(axis=0) + (base - base.mean(axis=0)) * 1.25 + np.array([0.25, 0.2, 0.0])
        suv, svalid, _ = project_points(shadow, cam)
        if svalid.all():
            canvas.polygon(suv, (24, 22, 20), alpha=0.42)
        body = (86, 78, 70) if agent.kind == "person" else (70, 74, 84)
        faces = sorted(_BOX_FACES, key=lambda f: -float(depth[list(f)].mean()))
        for face in faces:
            normal = np.cross(corners[face[1]] - corners[face[0]],
                              corners[face[2]] - corners[face[1]])
            normal = normal / max(float(np.linalg.norm(normal)), 1e-9)
            shade = 0.72 + 0.28 * abs(float(normal[2])) + 0.12 * abs(float(normal[0]))
            canvas.polygon(uv[list(face)], tuple(int(min(255, c * shade)) for c in body))


def render_demo(out_dir: Any, scenario_id: str = "Bike Approaches Ped", fps: int = 30,
                size: Tuple[int, int] = (1920, 1080)) -> Dict[str, Any]:
    """Render the HUD and the plan view for one scenario without a simulator.

    A pole-height pinhole camera at the testbench camera position looks at the
    crosswalk over an analytic street backdrop, with agents placed from the
    scenario paths. Writes ``demo_view.png`` (the camera view with the HUD),
    ``demo_bev.png``, ``demo_still.png`` (split layout), ``demo_fisheye.png``
    when the fisheye module is available, and ``demo.mp4``. Returns the paths
    under ``files`` together with the timeline of the scenario.
    """
    from .groundtruth import scenario_to_world, synthetic_frame

    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    scenario = _find_scenario(scenario_id)
    timeline = build_timeline(scenario, fps)
    width, height = int(size[0]), int(size[1])
    ref = _testbench().FisheyeCamera()
    position = scenario_to_world(ref.cam_x, ref.cam_y, ref.height)
    target = scenario_to_world(_CROSSWALK_X_M + 3.0, 0.0, 0.6)
    pole = _look_at("pole", position, target, width, height, fov_deg=90.0)
    cameras = {"pole": pole}
    run = RunInfo(scenario_name=scenario["name"],
                  scenario_id=str(scenario.get("id", scenario["name"].lower().replace(" ", "-"))),
                  fps=fps, duration_s=timeline["duration"], engine="synthetic",
                  weather="clear_noon", views=["pole"],
                  site={"camera_xy": (ref.cam_x, ref.cam_y), "ground_z": 0.0})
    backdrop = _synthetic_backdrop(pole)
    style = DEFAULT_STYLE
    bev_side = min(height, 1080)

    from .compose import compose, fit_size

    out_size = fit_size("split", (width, height), height)

    def render(index: int, frames: List[FrameRecord]) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        frame = frames[index]
        base = backdrop.copy()
        _paint_stand_ins(base, frame, pole)
        history = frames[max(0, index - int(style.trail_seconds * fps)):index]
        view = draw_overlay(base, frame, "pole", run=run, scenario=scenario, style=style,
                            history=history, timeline=timeline)
        bev = draw_bev(frame, run=run, scenario=scenario, size=(bev_side, bev_side), style=style)
        composite = compose("split", [view], bev, size=out_size)
        return view, bev, composite

    frames = [synthetic_frame(scenario, i, fps, cameras=cameras, states=timeline["states"])
              for i in range(timeline["n"])]
    alerts = timeline["alert"]
    if alerts:
        start, end = max(alerts, key=lambda iv: iv[1] - iv[0])
        still_index = int(round((start + 0.55 * (end - start)) * fps))
    else:
        still_index = timeline["n"] // 2
    still_index = min(max(still_index, 0), timeline["n"] - 1)
    view, bev, composite = render(still_index, frames)
    files = {"view": out / "demo_view.png", "bev": out / "demo_bev.png",
             "still": out / "demo_still.png", "clip": out / "demo.mp4"}
    cv2.imwrite(str(files["view"]), view)
    cv2.imwrite(str(files["bev"]), bev)
    cv2.imwrite(str(files["still"]), composite)

    try:
        from .fisheye import FisheyeRemap, deployed_fisheye_model

        model = deployed_fisheye_model(position, -90.0, scale=0.5)
        remap = FisheyeRemap(model, face_size=768)
        frame = replace(frames[still_index], cameras={"fisheye": model})
        faces = {}
        for face in remap.faces:
            face_cam = remap.face_camera(face)
            faces[face] = _synthetic_backdrop(face_cam, supersample=1)
            _paint_stand_ins(faces[face], frame, face_cam)
        fisheye = remap.remap(faces)
        files["fisheye"] = out / "demo_fisheye.png"
        cv2.imwrite(str(files["fisheye"]), draw_overlay(
            fisheye, frame, "fisheye", run=run, scenario=scenario, style=style,
            history=frames[max(0, still_index - int(style.trail_seconds * fps)):still_index],
            timeline=timeline))
    except ImportError:
        _LOG.info("fisheye module not available; skipping the fisheye still")

    from .recorder import VideoSink

    writer = VideoSink(files["clip"], fps)
    try:
        for index in range(timeline["n"]):
            writer.write(render(index, frames)[2])
    finally:
        writer.close()
    return {"files": {k: str(v) for k, v in files.items()}, "timeline": timeline,
            "still_index": still_index}


def main(argv: Optional[Sequence[str]] = None) -> int:
    """Command line entry point: ``python -m bikeped.sim.overlay --demo OUT_DIR``."""
    parser = argparse.ArgumentParser(
        prog="python -m bikeped.sim.overlay",
        description="Render the HUD and plan view over a synthetic backdrop, without a simulator.")
    parser.add_argument("--demo", metavar="OUT_DIR", required=True,
                        help="directory that receives the stills and the clip")
    parser.add_argument("--scenario", default="Bike Approaches Ped",
                        help="scenario name or id (default: %(default)s)")
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--size", default="1920x1080", help="output size as WIDTHxHEIGHT")
    args = parser.parse_args(argv)
    width, height = (int(v) for v in args.size.lower().split("x"))
    result = render_demo(args.demo, args.scenario, fps=args.fps, size=(width, height))
    for path in result["files"].values():
        print(path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
