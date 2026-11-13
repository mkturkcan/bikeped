"""Path builders that emit the testbench waypoint format.

Every builder works in the scenario frame used by ``decision_testbench``: metres, ``x`` along the
road (bikes ride along it), ``y`` across the road (pedestrians cross along it), crosswalk centre at
``(5, 0)``. Times are seconds from the start of the scenario and speeds are metres per second.

The testbench interpolates paths linearly in time, so curved or speed-varying motion is sampled
densely (every ``step`` seconds, 0.25 s by default). Headings are in degrees, counter-clockwise
from ``+x``: 0 rides towards ``+x``, 180 towards ``-x`` (the direction of the conformance bikes),
90 walks towards ``+y``.

The fluent :class:`Path` keeps scenario definitions short::

    bike = (Path.start(34, 1.5, heading=180).ride(14, 6.0)
            .arc(radius=6, angle=90).ride(8).build())
"""

from __future__ import annotations

import math
from typing import Sequence

import numpy as np

Waypoint = dict


def _round(v: float) -> float:
    return float(round(v, 3))


class Path:
    """Fluent builder for one agent path.

    The builder tracks the current position ``(x, y)`` in metres, time ``t`` in seconds, heading in
    degrees and speed in m/s. Each motion primitive appends waypoints sampled every ``step``
    seconds and leaves the state at its end, so primitives chain. :meth:`build` returns the
    waypoint list with strictly increasing times.
    """

    def __init__(self, x: float, y: float, t: float = 0.0, heading: float = 0.0,
                 speed: float = 0.0, step: float = 0.25) -> None:
        self.x, self.y, self.t = float(x), float(y), float(t)
        self.heading = float(heading)
        self.speed = float(speed)
        self.step = float(step)
        self._points: list[tuple[float, float, float]] = [(self.x, self.y, self.t)]

    @classmethod
    def start(cls, x: float, y: float, t: float = 0.0, heading: float = 0.0,
              speed: float = 0.0, step: float = 0.25) -> Path:
        """Start a path at ``(x, y)`` at time ``t`` (s), facing ``heading`` (deg)."""
        return cls(x, y, t, heading, speed, step)

    def _emit(self, x: float, y: float, t: float) -> None:
        self.x, self.y, self.t = x, y, t
        if t - self._points[-1][2] > 1e-6:
            self._points.append((x, y, t))

    def _unit(self) -> tuple[float, float]:
        h = math.radians(self.heading)
        return math.cos(h), math.sin(h)

    def ride(self, distance: float, speed: float | None = None) -> Path:
        """Move ``distance`` metres along the current heading at constant ``speed`` (m/s)."""
        if speed is not None:
            self.speed = float(speed)
        if self.speed <= 0:
            raise ValueError('ride() needs a positive speed')
        ux, uy = self._unit()
        x0, y0, t0 = self.x, self.y, self.t
        dur = distance / self.speed
        n = max(1, math.ceil(dur / self.step))
        for i in range(1, n + 1):
            f = i / n
            self._emit(x0 + ux * distance * f, y0 + uy * distance * f, t0 + dur * f)
        return self

    walk = ride

    def to(self, x: float, y: float, speed: float | None = None) -> Path:
        """Move in a straight line to ``(x, y)`` at constant ``speed``; the heading follows."""
        dx, dy = x - self.x, y - self.y
        dist = math.hypot(dx, dy)
        if dist < 1e-9:
            return self
        self.heading = math.degrees(math.atan2(dy, dx))
        return self.ride(dist, speed)

    def face(self, heading: float) -> Path:
        """Set the heading (deg) without moving."""
        self.heading = float(heading)
        return self

    def wait(self, duration: float) -> Path:
        """Stand still for ``duration`` seconds; the speed becomes zero."""
        self.speed = 0.0
        self._emit(self.x, self.y, self.t + duration)
        return self

    def appear(self, t: float) -> Path:
        """Shift the whole path so that it begins at time ``t``.

        Use it for late emergence: the agent is absent (not detected) until ``t``, as if hidden
        behind a parked van or a crowd. Call it before any motion primitive.
        """
        if len(self._points) != 1:
            raise ValueError('appear() must be called before the first motion')
        self.t = float(t)
        self._points = [(self.x, self.y, self.t)]
        return self

    def change_speed(self, v_end: float, accel: float) -> Path:
        """Accelerate or brake along the heading to ``v_end`` m/s at ``|accel|`` m/s^2.

        The distance covered follows from constant acceleration; the path ends with
        ``speed == v_end``. Braking to zero leaves the agent stopped at the end point.
        """
        a = abs(float(accel))
        v0 = self.speed
        if a <= 0 or abs(v_end - v0) < 1e-9:
            return self
        sign = 1.0 if v_end > v0 else -1.0
        dur = abs(v_end - v0) / a
        ux, uy = self._unit()
        x0, y0, t0 = self.x, self.y, self.t
        n = max(1, math.ceil(dur / self.step))
        for i in range(1, n + 1):
            tt = dur * i / n
            s = v0 * tt + 0.5 * sign * a * tt * tt
            self._emit(x0 + ux * s, y0 + uy * s, t0 + tt)
        self.speed = float(v_end)
        return self

    def accelerate(self, v_end: float, accel: float = 1.5) -> Path:
        """Speed up to ``v_end`` m/s at ``accel`` m/s^2 along the heading."""
        return self.change_speed(v_end, accel)

    def brake(self, v_end: float = 0.0, decel: float = 2.0) -> Path:
        """Slow down to ``v_end`` m/s (default a full stop) at ``decel`` m/s^2."""
        return self.change_speed(v_end, decel)

    def brake_to(self, x: float | None = None, y: float | None = None,
                 v_end: float = 0.0) -> Path:
        """Brake at constant deceleration so the agent reaches ``v_end`` at a given coordinate.

        Give ``x`` (for agents moving along the road) or ``y`` (for agents moving across it). The
        deceleration is whatever stops the agent exactly there.
        """
        ux, uy = self._unit()
        if x is not None:
            dist = (x - self.x) / ux if abs(ux) > 1e-6 else math.nan
        elif y is not None:
            dist = (y - self.y) / uy if abs(uy) > 1e-6 else math.nan
        else:
            raise ValueError('brake_to() needs x or y')
        if not dist > 0:
            raise ValueError('brake_to() target lies behind the agent')
        decel = (self.speed ** 2 - v_end ** 2) / (2.0 * dist)
        if decel <= 0:
            raise ValueError('brake_to() needs v_end below the current speed')
        return self.change_speed(v_end, decel)

    def arc(self, radius: float, angle: float, speed: float | None = None) -> Path:
        """Turn on a circular arc of ``radius`` metres through ``angle`` degrees.

        Positive angles turn left (counter-clockwise seen from above, ``+y`` up), negative angles
        turn right. Speed is constant along the arc.
        """
        if speed is not None:
            self.speed = float(speed)
        if self.speed <= 0:
            raise ValueError('arc() needs a positive speed')
        sgn = 1.0 if angle >= 0 else -1.0
        h0 = math.radians(self.heading)
        cx = self.x - sgn * radius * math.sin(h0)
        cy = self.y + sgn * radius * math.cos(h0)
        length = radius * math.radians(abs(angle))
        dur = length / self.speed
        n = max(2, math.ceil(dur / self.step))
        t0 = self.t
        for i in range(1, n + 1):
            phi = h0 + sgn * math.radians(abs(angle)) * i / n
            self._emit(cx + sgn * radius * math.sin(phi), cy - sgn * radius * math.cos(phi),
                       t0 + dur * i / n)
        self.heading += angle
        return self

    def lane_change(self, lateral: float, length: float, speed: float | None = None) -> Path:
        """Shift sideways by ``lateral`` metres (positive to the left) over ``length`` metres.

        The lateral offset follows a half-cosine, which gives a smooth swerve or lane change; the
        heading is unchanged at the end.
        """
        if speed is not None:
            self.speed = float(speed)
        ux, uy = self._unit()
        lx, ly = -uy, ux
        x0, y0, t0 = self.x, self.y, self.t
        dur = math.hypot(length, lateral) / self.speed
        n = max(4, math.ceil(dur / self.step))
        for i in range(1, n + 1):
            f = i / n
            off = lateral * 0.5 * (1.0 - math.cos(math.pi * f))
            self._emit(x0 + ux * length * f + lx * off, y0 + uy * length * f + ly * off,
                       t0 + dur * f)
        return self

    def zigzag(self, distance: float, amplitude: float, wavelength: float,
               speed: float | None = None) -> Path:
        """Weave about the current heading: sinusoidal lateral offset of ``amplitude`` metres.

        The agent covers ``distance`` metres along the heading; one full weave takes
        ``wavelength`` metres. Speed is the speed along the heading.
        """
        if speed is not None:
            self.speed = float(speed)
        ux, uy = self._unit()
        lx, ly = -uy, ux
        x0, y0, t0 = self.x, self.y, self.t
        dur = distance / self.speed
        n = max(4, math.ceil(dur / self.step))
        for i in range(1, n + 1):
            s = distance * i / n
            off = amplitude * math.sin(2.0 * math.pi * s / wavelength)
            self._emit(x0 + ux * s + lx * off, y0 + uy * s + ly * off, t0 + dur * i / n)
        return self

    def through(self, points: Sequence[tuple[float, float]], speed: float | None = None) -> Path:
        """Follow a Catmull-Rom spline from the current position through ``points`` at ``speed``.

        The spline passes through every control point. Arc length is resampled so the speed is
        constant; the heading at the end follows the last segment.
        """
        if speed is not None:
            self.speed = float(speed)
        xy = catmull_rom([(self.x, self.y), *points], samples_per_segment=24)
        seg = np.hypot(np.diff(xy[:, 0]), np.diff(xy[:, 1]))
        s = np.concatenate([[0.0], np.cumsum(seg)])
        total = float(s[-1])
        dur = total / self.speed
        n = max(2, math.ceil(dur / self.step))
        t0 = self.t
        for i in range(1, n + 1):
            si = total * i / n
            self._emit(float(np.interp(si, s, xy[:, 0])), float(np.interp(si, s, xy[:, 1])),
                       t0 + dur * i / n)
        dx, dy = xy[-1] - xy[-2]
        self.heading = math.degrees(math.atan2(dy, dx))
        return self

    def build(self) -> list[Waypoint]:
        """Return the path as testbench waypoints ``[{'x', 'y', 't'}, ...]`` (rounded to mm, ms)."""
        out: list[Waypoint] = []
        for x, y, t in self._points:
            p = {'x': _round(x), 'y': _round(y), 't': _round(t)}
            if out and p['t'] <= out[-1]['t']:
                continue
            out.append(p)
        if len(out) < 2:
            raise ValueError('a path needs at least two waypoints with increasing time')
        return out


def catmull_rom(points: Sequence[tuple[float, float]], samples_per_segment: int = 16) -> np.ndarray:
    """Sample a centripetal-free (uniform) Catmull-Rom spline through ``points``.

    Returns an ``(N, 2)`` array in the same units as the input. End points are duplicated so the
    curve starts and ends exactly at the first and last control points.
    """
    p = np.asarray(points, dtype=float)
    if len(p) < 2:
        raise ValueError('catmull_rom() needs at least two points')
    ext = np.vstack([p[0], p, p[-1]])
    out = [p[0]]
    for i in range(1, len(ext) - 2):
        p0, p1, p2, p3 = ext[i - 1], ext[i], ext[i + 1], ext[i + 2]
        for k in range(1, samples_per_segment + 1):
            u = k / samples_per_segment
            u2, u3 = u * u, u * u * u
            out.append(0.5 * ((2 * p1) + (-p0 + p2) * u + (2 * p0 - 5 * p1 + 4 * p2 - p3) * u2
                              + (-p0 + 3 * p1 - 3 * p2 + p3) * u3))
    return np.asarray(out)


def straight(x0: float, y0: float, x1: float, y1: float, speed: float,
             t0: float = 0.0) -> list[Waypoint]:
    """Constant-speed straight leg from ``(x0, y0)`` to ``(x1, y1)`` starting at ``t0``."""
    return Path.start(x0, y0, t0).to(x1, y1, speed).build()


def crossing(x: float = 5.0, y0: float = -6.0, y1: float = 6.0, speed: float = 1.35,
             t0: float = 0.0) -> list[Waypoint]:
    """Pedestrian crossing along ``y`` at road position ``x`` at constant ``speed``."""
    return straight(x, y0, x, y1, speed, t0)


def shift(path: Sequence[Waypoint], dx: float = 0.0, dy: float = 0.0,
          dt: float = 0.0) -> list[Waypoint]:
    """Return a copy of ``path`` translated by ``(dx, dy)`` metres and ``dt`` seconds."""
    return [{'x': _round(p['x'] + dx), 'y': _round(p['y'] + dy), 't': _round(p['t'] + dt)}
            for p in path]


def group(path: Sequence[Waypoint], offsets: Sequence[tuple[float, float, float]]
          ) -> list[list[Waypoint]]:
    """Copies of ``path`` for a group that moves together, one per ``(dx, dy, dt)`` offset."""
    return [shift(path, dx, dy, dt) for dx, dy, dt in offsets]


def platoon(path: Sequence[Waypoint], n: int, headway_s: float = 1.0, lateral: float = 0.0,
            jitter_m: float = 0.0, jitter_s: float = 0.0, seed: int = 0,
            lateral_axis: str = 'y') -> list[list[Waypoint]]:
    """``n`` riders on the same line, each ``headway_s`` seconds behind the previous one.

    ``lateral`` staggers alternate riders sideways (metres along ``lateral_axis``); ``jitter_m``
    and ``jitter_s`` add seeded uniform noise to each rider's lateral position and start time so
    the group does not look cloned. The result is deterministic for a given ``seed``.
    """
    rng = np.random.default_rng(seed)
    out = []
    for i in range(n):
        side = lateral * (1 if i % 2 else 0)
        off = side + (rng.uniform(-jitter_m, jitter_m) if jitter_m else 0.0)
        dt = i * headway_s + (rng.uniform(0, jitter_s) if jitter_s and i else 0.0)
        if lateral_axis == 'y':
            out.append(shift(path, dy=off, dt=dt))
        else:
            out.append(shift(path, dx=off, dt=dt))
    return out


def path_speeds(path: Sequence[Waypoint]) -> np.ndarray:
    """Segment speeds (m/s) of a waypoint path, one per segment."""
    xs = np.array([p['x'] for p in path], dtype=float)
    ys = np.array([p['y'] for p in path], dtype=float)
    ts = np.array([p['t'] for p in path], dtype=float)
    return np.hypot(np.diff(xs), np.diff(ys)) / np.diff(ts)


def path_accelerations(path: Sequence[Waypoint]) -> np.ndarray:
    """Magnitude of the velocity change (m/s^2) at each interior waypoint.

    The velocity change between consecutive segments is divided by the time between segment
    midpoints, so turns contribute their centripetal acceleration.
    """
    xs = np.array([p['x'] for p in path], dtype=float)
    ys = np.array([p['y'] for p in path], dtype=float)
    ts = np.array([p['t'] for p in path], dtype=float)
    if len(ts) < 3:
        return np.zeros(0)
    dts = np.diff(ts)
    vx, vy = np.diff(xs) / dts, np.diff(ys) / dts
    mid = 0.5 * (dts[:-1] + dts[1:])
    return np.hypot(np.diff(vx), np.diff(vy)) / mid
