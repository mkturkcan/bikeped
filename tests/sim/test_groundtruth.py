"""Tests for per-frame ground truth; no simulator needed."""

from __future__ import annotations

import json
import math

import numpy as np
import pytest

import decision_testbench as tb
from bikeped.sim.groundtruth import (
    closest_pair,
    danger_record,
    frame_ground_truth,
    scenario_to_world,
    synthetic_agents,
    synthetic_frame,
)
from bikeped.sim.projection import carla_rotation_to_basis, project_points
from bikeped.sim.types import CameraModel

SCENARIO = next(s for s in tb.SCENARIOS if s["name"] == "Bike Approaches Ped")


def _pole(width: int = 640, height: int = 360) -> CameraModel:
    f, r, u = carla_rotation_to_basis(-16.0, -90.0, 0.0)
    return CameraModel(name="pole", kind="pinhole", width=width, height=height,
                       position=scenario_to_world(5.0, -13.0, 3.6576), forward=f, right=r, up=u,
                       fov_deg=90.0)


def test_synthetic_agents_match_testbench():
    agents = synthetic_agents(SCENARIO, 3.0)
    assert {a.kind for a in agents} == {"person", "bike"}
    for agent in agents:
        pose = tb.interpolate_agent(next(d for d in SCENARIO["agents"] if d["id"] == agent.id), 3.0)
        assert (agent.sx, agent.sy) == pytest.approx((pose["x"], pose["y"]))
        assert agent.location[:2] == pytest.approx((pose["x"], -pose["y"]))
        assert agent.location[2] == pytest.approx(agent.extent[2])
        assert agent.rotation[1] == pytest.approx(-math.degrees(pose["heading"]))
    bike = next(a for a in agents if a.kind == "bike")
    assert 2 * bike.extent[0] == pytest.approx(tb.OBJECT_TYPES["cyclist"].length)


def test_danger_record_is_json_safe():
    for t in (0.0, 4.5, 6.0):
        record = danger_record(SCENARIO, t)
        json.dumps(record, allow_nan=False)
        ref = tb.compute_ground_truth_danger(SCENARIO, t)
        assert record["is_danger"] == bool(ref.is_danger)
        if math.isfinite(ref.distance):
            assert record["distance"] == pytest.approx(ref.distance, abs=1e-3)
        else:
            assert record["distance"] is None
    assert any(danger_record(SCENARIO, i * 0.1)["is_danger"] for i in range(70))


def test_closest_pair_and_alert_pair():
    frame = synthetic_frame(SCENARIO, 45, 10, states=["alert"] * 100)
    bike_id, person_id, dist = closest_pair(frame)
    assert (bike_id, person_id) == (2, 1)
    bike = next(a for a in frame.agents if a.id == 2)
    person = next(a for a in frame.agents if a.id == 1)
    assert dist == pytest.approx(math.hypot(bike.sx - person.sx, bike.sy - person.sy))
    assert frame.alert_pair == (2, 1)
    assert synthetic_frame(SCENARIO, 45, 10).alert_pair is None


def test_frame_ground_truth_boxes():
    cam = _pole()
    frame = synthetic_frame(SCENARIO, 30, 10, cameras={"pole": cam})
    objects = frame_ground_truth(frame, "pole")
    assert {o["id"] for o in objects} == {1, 2}
    for obj in objects:
        x1, y1, x2, y2 = obj["bbox"]
        assert 0 <= x1 < x2 <= cam.width and 0 <= y1 < y2 <= cam.height
        assert len(obj["corners_image"]) == 8
        assert obj["visible_fraction"] is None
    json.dumps(objects, allow_nan=False)
    assert frame_ground_truth(frame, "missing") == []


def test_visible_fraction_from_depth():
    cam = _pole()
    frame = synthetic_frame(SCENARIO, 30, 10, cameras={"pole": cam})
    far = np.full((cam.height, cam.width), 1000.0, dtype=np.float32)
    frame.aux["pole/depth"] = far
    assert all(o["visible_fraction"] == 1.0 for o in frame_ground_truth(frame, "pole"))
    near = np.full((cam.height, cam.width), 0.5, dtype=np.float32)
    frame.aux["pole/depth"] = near
    assert all(o["visible_fraction"] == 0.0 for o in frame_ground_truth(frame, "pole"))


def test_visible_fraction_from_instance():
    cam = _pole()
    frame = synthetic_frame(SCENARIO, 30, 10, cameras={"pole": cam})
    person = next(a for a in frame.agents if a.kind == "person")
    person.instance_id = 77
    ids = np.zeros((cam.height, cam.width), dtype=np.int32)
    centre, _, _ = project_points(np.array([person.location]), cam)
    u, v = centre[0].astype(int)
    ids[:, :u] = 77
    frame.aux["pole/instance"] = ids
    objects = {o["id"]: o for o in frame_ground_truth(frame, "pole")}
    assert 0.2 < objects[person.id]["visible_fraction"] < 0.8
