"""Placement of the scenario frame on a CARLA crosswalk.

The scenario frame (see :mod:`bikeped.sim.types`) is anchored on one crosswalk:
scenario ``+y`` runs along the crosswalk (the direction pedestrians cross),
scenario ``+x`` runs along the road, and the crosswalk centre is scenario
``(5, 0)``. As in the original ``carla_scenario.py``, the reference short edge
of the crosswalk polygon defines ``-y`` (the pole stands there). The road axis
is the cross axis turned by ``(a, b) -> (-b, a)`` in world coordinates: the
world frame is left-handed, so this is the choice that keeps the scene
unmirrored (seen from above, turning from scenario ``+x`` to ``+y`` is a
counter-clockwise turn in both frames, and a cyclist approaching from scenario
``+x`` enters the pole view from the right, as in the testbench camera).

Everything here is plain numpy. :meth:`Site.from_world` is the only method that
touches a CARLA world, and only through ``world.get_map().get_crosswalks()``.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Sequence, Tuple, Union

import numpy as np

#: Pole height of the deployed camera (12 ft), metres.
POLE_HEIGHT_M = 3.6576


def _testbench_pole() -> Tuple[float, float, float]:
    """Camera position assumed by ``decision_testbench.FisheyeCamera`` (scenario frame)."""
    try:
        import decision_testbench as tb
    except ImportError:
        return 5.0, -13.0, POLE_HEIGHT_M
    cam = tb.FisheyeCamera
    return float(cam.cam_x), float(cam.cam_y), float(cam.height)


@dataclass(frozen=True)
class Site:
    """A crosswalk with the scenario frame attached to it.

    ``origin`` is the world ``(X, Y)`` of scenario ``(0, 0)``; ``road_axis`` and
    ``cross_axis`` are the world unit vectors of scenario ``+x`` and ``+y``.
    ``ground_z`` is the world height of the crosswalk surface and
    ``road_half_width`` half the crosswalk length (kerb to kerb), in metres.
    """

    origin: Tuple[float, float]
    road_axis: Tuple[float, float]
    cross_axis: Tuple[float, float]
    ground_z: float
    road_half_width: float
    crosswalk_id: Any = None
    pole_xy: Tuple[float, float] = (5.0, -13.0)
    pole_height: float = POLE_HEIGHT_M

    @classmethod
    def from_crosswalk(cls, entry: Dict[str, Any]) -> "Site":
        """Build a site from one ``crosswalks.json`` entry (``corners``, ``center``)."""
        corners = [np.asarray(c[:2], dtype=np.float64) for c in entry["corners"]]
        edges = []
        for i in range(4):
            a, b = corners[i], corners[(i + 1) % 4]
            edges.append((float(np.linalg.norm(b - a)), (a + b) / 2.0))
        edges.sort(key=lambda e: e[0])
        cam_edge = edges[0][1]
        center3 = list(entry["center"])
        center = np.asarray(center3[:2], dtype=np.float64)
        to_center = center - cam_edge
        half = float(np.linalg.norm(to_center))
        cross = to_center / half
        road = np.array([-cross[1], cross[0]])
        origin = center - 5.0 * road
        px, py, ph = _testbench_pole()
        return cls(
            origin=(float(origin[0]), float(origin[1])),
            road_axis=(float(road[0]), float(road[1])),
            cross_axis=(float(cross[0]), float(cross[1])),
            ground_z=float(center3[2]) if len(center3) > 2 else 0.0,
            road_half_width=half,
            crosswalk_id=entry.get("id"),
            pole_xy=(px, py),
            pole_height=ph,
        )

    @classmethod
    def from_json(cls, path: Union[str, Path], index: int) -> "Site":
        """Load entry ``index`` (list position or ``id``) of a ``crosswalks.json`` file."""
        entries = json.loads(Path(path).read_text())
        for entry in entries:
            if entry.get("id") == index:
                return cls.from_crosswalk(entry)
        return cls.from_crosswalk(entries[index])

    @classmethod
    def from_world(cls, world: Any, index: int) -> "Site":
        """Build a site from crosswalk ``index`` of the loaded map.

        ``Map.get_crosswalks()`` returns closed polygons as a flat list of
        locations; rectangles come with their first point repeated (five points).
        """
        entries = crosswalks_from_points(
            [(p.x, p.y, p.z) for p in world.get_map().get_crosswalks()])
        if not entries:
            raise RuntimeError("the loaded map exposes no crosswalk polygons")
        if not 0 <= index < len(entries):
            raise IndexError(f"crosswalk {index} not found, the map has {len(entries)}")
        return cls.from_crosswalk(entries[index])

    def to_world(self, x: float, y: float, z: float = 0.0) -> Tuple[float, float, float]:
        """Scenario ``(x, y)`` and height above ground to world ``(X, Y, Z)``."""
        X = self.origin[0] + x * self.road_axis[0] + y * self.cross_axis[0]
        Y = self.origin[1] + x * self.road_axis[1] + y * self.cross_axis[1]
        return float(X), float(Y), float(self.ground_z + z)

    def to_scenario(self, X: float, Y: float) -> Tuple[float, float]:
        """World ``(X, Y)`` to scenario ``(x, y)``."""
        dx, dy = X - self.origin[0], Y - self.origin[1]
        return (float(dx * self.road_axis[0] + dy * self.road_axis[1]),
                float(dx * self.cross_axis[0] + dy * self.cross_axis[1]))

    def heading_to_yaw(self, heading: float) -> float:
        """Scenario heading (radians, 0 = +x) to CARLA yaw (degrees, clockwise from above)."""
        dx = math.cos(heading) * self.road_axis[0] + math.sin(heading) * self.cross_axis[0]
        dy = math.cos(heading) * self.road_axis[1] + math.sin(heading) * self.cross_axis[1]
        return math.degrees(math.atan2(dy, dx))

    def yaw_to_heading(self, yaw_deg: float) -> float:
        """Inverse of :meth:`heading_to_yaw`."""
        c, s = math.cos(math.radians(yaw_deg)), math.sin(math.radians(yaw_deg))
        x = c * self.road_axis[0] + s * self.road_axis[1]
        y = c * self.cross_axis[0] + s * self.cross_axis[1]
        return math.atan2(y, x)

    def direction(self, heading: float) -> np.ndarray:
        """World-frame unit vector (X, Y, 0) of a scenario heading."""
        yaw = math.radians(self.heading_to_yaw(heading))
        return np.array([math.cos(yaw), math.sin(yaw), 0.0])

    @property
    def pole_world(self) -> Tuple[float, float, float]:
        """World position of the pole camera (the testbench camera position)."""
        return self.to_world(self.pole_xy[0], self.pole_xy[1], self.pole_height)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "crosswalk_id": self.crosswalk_id,
            "origin": list(self.origin),
            "road_axis": list(self.road_axis),
            "cross_axis": list(self.cross_axis),
            "ground_z": self.ground_z,
            "road_half_width": self.road_half_width,
            "pole_scenario": [self.pole_xy[0], self.pole_xy[1], self.pole_height],
            "pole_world": list(self.pole_world),
        }


def crosswalks_from_points(points: Sequence[Tuple[float, float, float]]) -> list:
    """Split CARLA's flat crosswalk point list into ``crosswalks.json``-like entries.

    A polygon ends where a point repeats the polygon's first point. Polygons with
    other than four distinct corners are approximated by their minimum-area
    rectangle so the edge logic of :meth:`Site.from_crosswalk` applies.
    """
    entries = []
    current: list = []
    for p in points:
        if current and np.allclose(p[:2], current[0][:2], atol=1e-3):
            entries.append(current)
            current = []
        else:
            current.append(tuple(float(v) for v in p))
    out = []
    for poly in entries:
        if len(poly) < 3:
            continue
        pts = np.asarray(poly, dtype=np.float64)
        corners = pts[:, :2] if len(poly) == 4 else _min_area_rect(pts[:, :2])
        z = float(pts[:, 2].mean())
        center = corners.mean(axis=0)
        out.append({
            "id": len(out),
            "center": [float(center[0]), float(center[1]), z],
            "corners": [[float(c[0]), float(c[1]), z] for c in corners],
        })
    return out


def _min_area_rect(xy: np.ndarray) -> np.ndarray:
    best = None
    for i in range(len(xy)):
        edge = xy[(i + 1) % len(xy)] - xy[i]
        n = float(np.linalg.norm(edge))
        if n < 1e-6:
            continue
        u = edge / n
        v = np.array([-u[1], u[0]])
        a, b = xy @ u, xy @ v
        area = (a.max() - a.min()) * (b.max() - b.min())
        if best is None or area < best[0]:
            best = (area, u, v, a.min(), a.max(), b.min(), b.max())
    _, u, v, a0, a1, b0, b1 = best
    return np.array([u * a0 + v * b0, u * a1 + v * b0, u * a1 + v * b1, u * a0 + v * b1])
