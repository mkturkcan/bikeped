"""Smooth rendering poses for scripted scenario agents.

Positions come straight from the scripted piecewise-linear path with the same
semantics as ``decision_testbench.interpolate_agent``, so the ground truth stays
exact. Only quantities that the testbench does not use are smoothed: the
heading (paths turn instantly at waypoints, people and bicycles do not) and the
lean of a two-wheeler, derived from speed and yaw rate. All angles are in the
scenario frame, radians, counter-clockwise from ``+x``.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

GRAVITY = 9.81
MAX_LEAN_RAD = math.radians(25.0)


def wrap_angle(a: float) -> float:
    """Wrap an angle to ``(-pi, pi]``."""
    return math.atan2(math.sin(a), math.cos(a))


@dataclass
class Pose:
    """Pose of one agent at one instant (scenario frame)."""

    t: float
    active: bool
    x: float
    y: float
    vx: float
    vy: float
    speed: float
    path_heading: float
    heading: float
    yaw_rate: float
    lean: float


class Trajectory:
    """Rendering trajectory of one scenario agent sampled on a fixed time grid.

    ``time_constant`` is the time constant (s) of the critically damped heading
    filter and ``max_turn_rate`` its slew limit (rad/s). Headings are computed
    once over the whole active window at ``fps`` so that :meth:`sample` is pure
    and repeatable.
    """

    def __init__(self, agent: Dict[str, Any], fps: int = 30, time_constant: Optional[float] = None,
                 max_turn_rate: Optional[float] = None) -> None:
        self.agent = agent
        self.id = int(agent["id"])
        self.kind = str(agent["type"])
        path = agent["path"]
        self.t = np.array([float(p["t"]) for p in path])
        self.xy = np.array([[float(p["x"]), float(p["y"])] for p in path])
        self.fps = int(fps)
        bike = self.kind == "bike"
        self.time_constant = time_constant if time_constant is not None else (0.35 if bike else 0.2)
        self.max_turn_rate = max_turn_rate if max_turn_rate is not None else (1.6 if bike else 4.0)
        self._grid, self._heading, self._rate = self._filter_headings()

    @property
    def start(self) -> float:
        return float(self.t[0])

    @property
    def end(self) -> float:
        return float(self.t[-1])

    def active(self, t: float) -> bool:
        """True inside the scripted window, where ``interpolate_agent`` returns a pose."""
        return self.start <= t <= self.end

    def _segment(self, t: float) -> int:
        i = int(np.searchsorted(self.t, t, side="right") - 1)
        return max(0, min(i, len(self.t) - 2))

    def position(self, t: float) -> Tuple[float, float]:
        """Scripted position, clamped to the path ends outside the active window."""
        if len(self.t) == 1:
            return float(self.xy[0, 0]), float(self.xy[0, 1])
        tc = min(max(t, self.start), self.end)
        i = self._segment(tc)
        span = self.t[i + 1] - self.t[i]
        frac = 0.0 if span <= 0 else (tc - self.t[i]) / span
        p = self.xy[i] + (self.xy[i + 1] - self.xy[i]) * frac
        return float(p[0]), float(p[1])

    def velocity(self, t: float) -> Tuple[float, float]:
        """Velocity (m/s) of the segment containing ``t``; zero outside the window."""
        if len(self.t) == 1 or not self.active(t):
            return 0.0, 0.0
        i = self._segment(t)
        span = self.t[i + 1] - self.t[i]
        if span <= 0:
            return 0.0, 0.0
        v = (self.xy[i + 1] - self.xy[i]) / span
        return float(v[0]), float(v[1])

    def path_heading(self, t: float) -> float:
        """Heading of the segment containing ``t`` (what ``interpolate_agent`` reports)."""
        if len(self.t) == 1:
            return 0.0
        i = self._segment(min(max(t, self.start), self.end))
        d = self.xy[i + 1] - self.xy[i]
        if not np.any(d):
            for j in list(range(i, -1, -1)) + list(range(i + 1, len(self.t) - 1)):
                d = self.xy[j + 1] - self.xy[j]
                if np.any(d):
                    break
        return float(math.atan2(d[1], d[0]))

    def _filter_headings(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        dt = 1.0 / self.fps
        n = max(2, int(math.ceil((self.end - self.start) * self.fps)) + 2)
        grid = self.start + np.arange(n) * dt
        omega = 1.0 / max(self.time_constant, 1e-3)
        heading = self.path_heading(self.start)
        rate = 0.0
        headings, rates = np.empty(n), np.empty(n)
        for k, t in enumerate(grid):
            target = self.path_heading(t)
            error = wrap_angle(target - heading)
            accel = omega * omega * error - 2.0 * omega * rate
            rate = float(np.clip(rate + accel * dt, -self.max_turn_rate, self.max_turn_rate))
            heading = wrap_angle(heading + rate * dt)
            headings[k], rates[k] = heading, rate
        return grid, headings, rates

    def heading(self, t: float) -> Tuple[float, float]:
        """Smoothed heading (rad) and yaw rate (rad/s) at ``t``."""
        k = int(round((min(max(t, self.start), self._grid[-1]) - self.start) * self.fps))
        k = max(0, min(k, len(self._grid) - 1))
        return float(self._heading[k]), float(self._rate[k])

    def sample(self, t: float) -> Pose:
        """Full pose at time ``t``."""
        x, y = self.position(t)
        vx, vy = self.velocity(t)
        speed = math.hypot(vx, vy)
        heading, rate = self.heading(t)
        lean = 0.0
        if self.kind == "bike":
            lean = float(np.clip(math.atan(speed * rate / GRAVITY), -MAX_LEAN_RAD, MAX_LEAN_RAD))
        return Pose(t=t, active=self.active(t), x=x, y=y, vx=vx, vy=vy, speed=speed,
                    path_heading=self.path_heading(t), heading=heading, yaw_rate=rate, lean=lean)


def trajectories(scenario: Dict[str, Any], fps: int = 30) -> List[Trajectory]:
    """One :class:`Trajectory` per scenario agent, in scenario order."""
    return [Trajectory(a, fps) for a in scenario["agents"]]


def scripted_agents(scenario: Dict[str, Any], t: float) -> List[Dict[str, Any]]:
    """Agents active at ``t`` exactly as ``interpolate_agent`` reports them."""
    import decision_testbench as tb

    out = []
    for agent in scenario["agents"]:
        pos = tb.interpolate_agent(agent, t)
        if pos is not None:
            out.append(pos)
    return out
