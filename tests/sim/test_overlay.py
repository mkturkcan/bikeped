"""Tests for the HUD overlay and the plan view; no simulator needed."""

from __future__ import annotations

import numpy as np
import pytest

import decision_testbench as tb
from bikeped.sim.groundtruth import scenario_to_world, synthetic_frame
from bikeped.sim.overlay import (
    OverlayStyle,
    build_timeline,
    draw_bev,
    draw_overlay,
)
from bikeped.sim.projection import carla_rotation_to_basis, project_box
from bikeped.sim.types import CameraModel, FrameRecord, RunInfo

FPS = 10
SIZE = (640, 360)


def _scenario(name: str = "Bike Approaches Ped") -> dict:
    return next(s for s in tb.SCENARIOS if s["name"] == name)


def _pole_camera(width: int = SIZE[0], height: int = SIZE[1]) -> CameraModel:
    ref = tb.FisheyeCamera()
    forward, right, up = carla_rotation_to_basis(-16.0, -90.0, 0.0)
    return CameraModel(name="pole", kind="pinhole", width=width, height=height,
                       position=scenario_to_world(ref.cam_x, ref.cam_y, ref.height),
                       forward=tuple(forward), right=tuple(right), up=tuple(up), fov_deg=90.0)


def _fisheye_camera(width: int = 480, height: int = 480) -> CameraModel:
    ref = tb.FisheyeCamera()
    forward, right, up = carla_rotation_to_basis(-90.0, 0.0, 0.0)
    radius = width / 2.0
    return CameraModel(name="fisheye", kind="fisheye", width=width, height=height,
                       position=scenario_to_world(ref.cam_x, ref.cam_y, ref.height),
                       forward=tuple(forward), right=tuple(right), up=tuple(up), fov_deg=197.9,
                       f_px_per_rad=radius / np.radians(197.9 / 2.0), circle_radius_px=radius)


def _run(scenario: dict) -> RunInfo:
    return RunInfo(scenario_name=scenario["name"], scenario_id="bike-approaches-ped", fps=FPS,
                   engine="ue5", weather="clear_noon", site={"ground_z": 0.0})


@pytest.fixture(scope="module")
def scenario() -> dict:
    return _scenario()


@pytest.fixture(scope="module")
def timeline(scenario: dict) -> dict:
    return build_timeline(scenario, FPS)


def _frames(scenario: dict, timeline: dict, cameras: dict) -> list:
    return [synthetic_frame(scenario, i, FPS, cameras=cameras, states=timeline["states"])
            for i in range(timeline["n"])]


def test_timeline_intervals_are_ordered(scenario: dict, timeline: dict) -> None:
    assert timeline["n"] == len(timeline["states"]) > 0
    assert timeline["alert"], "the scenario is expected to raise an alert"
    assert timeline["danger"], "the scenario is expected to contain ground-truth danger"
    for key in ("danger", "actionable", "alert", "warning"):
        previous_end = -1.0
        for start, end in timeline[key]:
            assert previous_end <= start < end <= timeline["duration"] + 1.0 / FPS
            previous_end = end
    assert timeline["first_alert_t"] == timeline["alert"][0][0]
    assert timeline["first_danger_t"] == timeline["danger"][0][0]


def test_timeline_from_recorded_states(scenario: dict, timeline: dict) -> None:
    states = ["idle"] * timeline["n"]
    states[3:6] = ["alert"] * 3
    custom = build_timeline(scenario, FPS, states=states)
    assert custom["alert"] == [[0.3, 0.6]]
    assert custom["first_alert_t"] == 0.3
    assert custom["danger"] == timeline["danger"]
    assert build_timeline(scenario, FPS) is timeline


def test_overlay_keeps_shape_and_input(scenario: dict, timeline: dict) -> None:
    cam = _pole_camera()
    frames = _frames(scenario, timeline, {"pole": cam})
    index = int(timeline["alert"][0][0] * FPS) + 2
    image = np.full((SIZE[1], SIZE[0], 3), 90, np.uint8)
    before = image.copy()
    out = draw_overlay(image, frames[index], "pole", run=_run(scenario), scenario=scenario,
                       history=frames[max(0, index - 15):index], timeline=timeline)
    assert out.shape == image.shape and out.dtype == np.uint8
    assert np.array_equal(image, before)
    assert out is not image


def test_overlay_draws_near_projected_agent(scenario: dict, timeline: dict) -> None:
    cam = _pole_camera()
    frames = _frames(scenario, timeline, {"pole": cam})
    frame = frames[int(3.0 * FPS)]
    person = next(a for a in frame.agents if a.kind == "person")
    box = project_box(person, cam)
    assert box is not None
    x1, y1, x2, y2 = (int(round(v)) for v in box["bbox"])
    image = np.full((SIZE[1], SIZE[0], 3), 90, np.uint8)
    style = OverlayStyle(badge=False, timeline=False, caption=False, labels=False,
                         connector=False, trails=False)
    out = draw_overlay(image, frame, "pole", run=_run(scenario), style=style)
    changed = np.any(out != image, axis=2)
    assert changed[max(y1 - 2, 0):y2 + 3, max(x1 - 2, 0):x2 + 3].any()
    outside = changed.copy()
    for agent in frame.agents:
        other = project_box(agent, cam)
        if other is not None:
            ax1, ay1, ax2, ay2 = (int(round(v)) for v in other["bbox"])
            outside[max(ay1 - 4, 0):ay2 + 5, max(ax1 - 4, 0):ax2 + 5] = False
    assert not outside.any()


def test_overlay_hud_only_without_camera(scenario: dict, timeline: dict) -> None:
    frames = _frames(scenario, timeline, {})
    image = np.zeros((SIZE[1], SIZE[0], 3), np.uint8)
    out = draw_overlay(image, frames[5], "chase", run=_run(scenario), scenario=scenario)
    assert out.shape == image.shape
    assert out[: SIZE[1] // 4].any(), "state badge missing"
    assert out[-SIZE[1] // 5:].any(), "caption and timeline missing"


def test_overlay_on_fisheye_view(scenario: dict, timeline: dict) -> None:
    cam = _fisheye_camera()
    frames = _frames(scenario, timeline, {"fisheye": cam})
    index = int(3.0 * FPS)
    image = np.zeros((cam.height, cam.width, 3), np.uint8)
    out = draw_overlay(image, frames[index], "fisheye", run=_run(scenario), scenario=scenario,
                       history=frames[index - 10:index], timeline=timeline)
    assert out.shape == image.shape and out.dtype == np.uint8
    middle = out[cam.height // 4: 3 * cam.height // 4]
    assert middle.any(), "boxes and rings missing on the fisheye view"


def test_overlay_with_no_agents(scenario: dict) -> None:
    cam = _pole_camera()
    frame = FrameRecord(index=0, t=0.0, agents=[], cameras={"pole": cam})
    image = np.zeros((SIZE[1], SIZE[0], 3), np.uint8)
    out = draw_overlay(image, frame, "pole", run=_run(scenario), style=OverlayStyle(rings=True))
    assert out.shape == image.shape
    bev = draw_bev(frame, run=_run(scenario), scenario=None, size=(320, 240))
    assert bev.shape == (240, 320, 3)


def test_bev_size_and_content(scenario: dict, timeline: dict) -> None:
    frames = _frames(scenario, timeline, {})
    index = int(timeline["alert"][0][0] * FPS) + 2
    bev = draw_bev(frames[index], run=_run(scenario), scenario=scenario, size=(400, 300),
                   timeline=timeline)
    assert bev.shape == (300, 400, 3) and bev.dtype == np.uint8
    empty = draw_bev(FrameRecord(index=0, t=0.0, agents=[]), run=_run(scenario),
                     scenario=scenario, size=(400, 300))
    assert np.any(bev != empty), "agents missing from the plan view"


@pytest.mark.parametrize("height", [360, 720])
def test_hud_scales_with_height(scenario: dict, timeline: dict, height: int) -> None:
    width = height * 16 // 9
    frames = _frames(scenario, timeline, {})
    image = np.zeros((height, width, 3), np.uint8)
    out = draw_overlay(image, frames[0], "pole", run=_run(scenario), scenario=scenario)
    rows = np.flatnonzero(out[: height // 3].any(axis=(1, 2)))
    assert rows.size
    badge_height = (rows.max() - rows.min() + 1) / height
    assert 0.02 < badge_height < 0.12
