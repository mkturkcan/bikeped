"""Per-frame ground truth: 2D boxes, visibility, danger labels and the closest pair.

Boxes come from :func:`~bikeped.sim.projection.project_box` on each agent's
oriented 3D box in the world frame. Danger labels come from
``decision_testbench.compute_ground_truth_danger`` evaluated on the scripted
scenario, so the rendered runs carry the same labels as the paper's
evaluation. The synthetic helpers place testbench boxes in a world without a
simulator, for tests and demos.
"""

from __future__ import annotations

import math
from dataclasses import asdict
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import cv2
import numpy as np

from .projection import BOX_EDGES, box_corners_world, project_points, project_polyline
from .types import AgentState, CameraModel, FrameRecord

#: Testbench object type used for each agent kind.
KIND_TO_OBJECT = {"person": "pedestrian", "bike": "cyclist"}


def _testbench():
    import decision_testbench

    return decision_testbench


def _json_number(value: Any) -> Any:
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    if isinstance(value, (int, float, np.integer, np.floating)):
        v = float(value)
        return round(v, 4) if math.isfinite(v) else None
    return value


def danger_record(scenario: Mapping[str, Any], t: float) -> Dict[str, Any]:
    """Ground-truth danger at time ``t`` (seconds) as a JSON-safe dict.

    Wraps ``decision_testbench.compute_ground_truth_danger``; the keys are the
    fields of its ``DangerAssessment`` (distances in metres, speeds in m/s,
    times in seconds) and infinite values become ``None``.
    """
    assessment = _testbench().compute_ground_truth_danger(dict(scenario), float(t))
    return {k: _json_number(v) for k, v in asdict(assessment).items()}


def closest_pair(frame: FrameRecord) -> Optional[Tuple[int, int, float]]:
    """Nearest scripted ``(bike_id, person_id, distance_m)`` in the scenario frame, or ``None``."""
    bikes = [a for a in frame.agents if a.role == "scripted" and a.kind == "bike"]
    people = [a for a in frame.agents if a.role == "scripted" and a.kind == "person"]
    best: Optional[Tuple[int, int, float]] = None
    for b in bikes:
        for p in people:
            d = math.hypot(b.sx - p.sx, b.sy - p.sy)
            if best is None or d < best[2]:
                best = (b.id, p.id, d)
    return best


def scenario_to_world(x: float, y: float, z: float = 0.0) -> Tuple[float, float, float]:
    """Default embedding of the right-handed scenario frame in the left-handed world.

    Flips ``y``. Used by the synthetic helpers; the CARLA runtime places
    scenarios with its own site transform. A scenario heading ``h`` (radians)
    becomes the world yaw ``-degrees(h)``.
    """
    return float(x), float(-y), float(z)


def synthetic_agents(scenario: Mapping[str, Any], t: float,
                     ground_z: float = 0.0) -> List[AgentState]:
    """Scripted agents at time ``t`` with testbench box sizes.

    Positions are placed with :func:`scenario_to_world`; speeds are in m/s.
    """
    tb = _testbench()
    agents: List[AgentState] = []
    for agent_def in scenario["agents"]:
        pose = tb.interpolate_agent(agent_def, t)
        if pose is None:
            continue
        obj = tb.OBJECT_TYPES[KIND_TO_OBJECT.get(agent_def["type"], "pedestrian")]
        agents.append(AgentState(
            id=int(agent_def["id"]), kind=agent_def["type"], sx=float(pose["x"]),
            sy=float(pose["y"]), heading=float(pose["heading"]),
            speed=float(tb.estimate_agent_speed(agent_def, t)),
            location=scenario_to_world(pose["x"], pose["y"], ground_z + obj.height / 2.0),
            rotation=(0.0, -math.degrees(pose["heading"]), 0.0),
            extent=(obj.length / 2.0, obj.width / 2.0, obj.height / 2.0),
            model=str(agent_def.get("model", "")),
        ))
    return agents


def synthetic_frame(scenario: Mapping[str, Any], index: int, fps: int,
                    cameras: Optional[Dict[str, CameraModel]] = None,
                    states: Optional[Sequence[str]] = None) -> FrameRecord:
    """A :class:`FrameRecord` at ``t = index / fps`` built from the scenario alone, without images.

    ``state`` is ``states[index]`` when given, else ``"idle"``; in the alert
    state ``alert_pair`` is the closest bike and pedestrian.
    """
    t = index / float(fps)
    state = states[index] if states is not None and index < len(states) else "idle"
    frame = FrameRecord(index=index, t=t, agents=synthetic_agents(scenario, t), state=state,
                        states={"default": state}, danger=danger_record(scenario, t),
                        cameras=dict(cameras or {}))
    if state == "alert":
        pair = closest_pair(frame)
        if pair is not None:
            frame.alert_pair = (pair[0], pair[1])
    return frame


def _face_samples(agent: AgentState, cam: CameraModel, n: int = 5) -> np.ndarray:
    """Points on the box faces that face the camera, world frame ``(M, 3)``."""
    corners = box_corners_world(agent.location, agent.rotation, agent.extent)
    centre = corners.mean(axis=0)
    pos = np.asarray(cam.position, dtype=np.float64)
    grid = (np.arange(n) + 0.5) / n
    a, b = np.meshgrid(grid, grid)
    a, b = a.ravel()[:, None], b.ravel()[:, None]
    out = []
    for face in ((0, 2, 6, 4), (1, 3, 7, 5), (0, 1, 3, 2), (4, 5, 7, 6),
                 (0, 1, 5, 4), (2, 3, 7, 6)):
        p0, p1, _, p3 = corners[list(face)]
        fc = corners[list(face)].mean(axis=0)
        if np.dot(fc - centre, pos - fc) <= 0:
            continue
        out.append(p0 + a * (p1 - p0) + b * (p3 - p0))
    return np.vstack(out) if out else corners


def _visible_from_depth(agent: AgentState, cam: CameraModel, depth: np.ndarray) -> Optional[float]:
    pts = _face_samples(agent, cam)
    uv, valid, d = project_points(pts, cam)
    h, w = depth.shape[:2]
    if cam.kind == "pinhole":
        cols, rows = np.floor(uv[:, 0]), np.floor(uv[:, 1])
    else:
        cols, rows = np.round(uv[:, 0]), np.round(uv[:, 1])
    inside = valid & (cols >= 0) & (cols < w) & (rows >= 0) & (rows < h)
    if not inside.any():
        return None
    rendered = depth[rows[inside].astype(int), cols[inside].astype(int)].astype(np.float64)
    tol = 0.25 + 0.02 * d[inside]
    return float(np.mean(rendered >= d[inside] - tol))


def _silhouette(agent: AgentState, cam: CameraModel) -> Optional[np.ndarray]:
    corners = box_corners_world(agent.location, agent.rotation, agent.extent)
    pieces = [p for i, j in BOX_EDGES
              for p in project_polyline(corners[[i, j]], cam, max_step_px=4.0)]
    if not pieces:
        return None
    hull = cv2.convexHull(np.vstack(pieces).astype(np.float32))
    mask = np.zeros((cam.height, cam.width), dtype=np.uint8)
    cv2.fillConvexPoly(mask, np.round(hull).astype(np.int32), 1)
    return mask.astype(bool)


def _visible_from_instance(agent: AgentState, cam: CameraModel,
                           instance: np.ndarray) -> Optional[float]:
    if agent.instance_id is None:
        return None
    mask = _silhouette(agent, cam)
    if mask is None or instance.shape[:2] != mask.shape or not mask.any():
        return None
    ids = instance if instance.ndim == 2 else instance[..., 0]
    return float(min(1.0, np.count_nonzero(ids[mask] == agent.instance_id) / mask.sum()))


def frame_ground_truth(frame: FrameRecord, view: str) -> List[Dict[str, Any]]:
    """Per-agent ground truth of one view, JSON-safe.

    Each entry has ``id``, ``kind``, ``role``, ``bbox`` (pixels, clipped to
    the image), ``corners_image`` (8 ``[u, v]`` or ``None``), ``distance_m``,
    ``truncation`` and ``visible_fraction``. The visible fraction is the share
    of the agent's projected silhouette carrying its id in
    ``aux['<view>/instance']`` when that buffer and ``AgentState.instance_id``
    exist, else the share of camera-facing box surface samples not occluded in
    ``aux['<view>/depth']`` (planar depth for pinhole views, range for
    fisheye views), else ``None``. Agents outside the view are omitted.
    """
    from .projection import project_box

    cam = frame.cameras.get(view)
    if cam is None:
        return []
    depth = frame.aux.get(f"{view}/depth")
    instance = frame.aux.get(f"{view}/instance")
    objects: List[Dict[str, Any]] = []
    for agent in frame.agents:
        box = project_box(agent, cam)
        if box is None:
            continue
        visible: Optional[float] = None
        if instance is not None:
            visible = _visible_from_instance(agent, cam, instance)
        if visible is None and depth is not None:
            visible = _visible_from_depth(agent, cam, depth)
        corners = [None if not np.all(np.isfinite(c)) else [round(float(c[0]), 2),
                                                             round(float(c[1]), 2)]
                   for c in box["corners_image"]]
        objects.append({
            "id": agent.id, "kind": agent.kind, "role": agent.role,
            "bbox": [round(float(v), 2) for v in box["bbox"]],
            "corners_image": corners,
            "distance_m": round(float(box["distance_m"]), 3),
            "truncation": round(float(box["truncation"]), 4),
            "visible_fraction": None if visible is None else round(visible, 4),
        })
    return objects
