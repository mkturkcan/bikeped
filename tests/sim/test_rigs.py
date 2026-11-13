"""Camera rigs: framing and basis conventions (no server needed)."""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pytest

import decision_testbench as tb
from bikeped.sim.projection import project_points
from bikeped.sim.rigs import QUALITY_PRESETS, RIG_NAMES, make_rig, pole_framing
from bikeped.sim.site import Site

ROOT = Path(__file__).resolve().parents[2]


@pytest.fixture(scope="module")
def site():
    return Site.from_crosswalk(json.loads((ROOT / "crosswalks.json").read_text())[4])


def test_presets():
    assert set(QUALITY_PRESETS) == {"draft", "standard", "cinematic"}
    assert set(RIG_NAMES) == {"pole", "fisheye", "chase", "orbit", "street", "bev"}


def test_pole_sits_at_the_testbench_camera(site):
    rig = make_rig("pole", site, tb.SCENARIOS[1], "draft")
    cam = rig.camera(0.0)
    expected = site.to_world(tb.FisheyeCamera.cam_x, tb.FisheyeCamera.cam_y,
                             tb.FisheyeCamera.height)
    assert cam.position == pytest.approx(expected)


def test_pole_frames_every_path_point(site):
    scenario = tb.SCENARIOS[1]
    cam = make_rig("pole", site, scenario, "draft").camera(0.0)
    pts = np.array([site.to_world(p["x"], p["y"], 0.9)
                    for a in scenario["agents"] for p in a["path"]])
    uv, valid, _ = project_points(pts, cam)
    assert valid.all()
    assert (uv[:, 0] >= 0).all() and (uv[:, 0] < cam.width).all()


def test_pole_framing_looks_across_the_road(site):
    heading, pitch, fov = pole_framing(site, tb.SCENARIOS[0])
    assert abs(heading - math.pi / 2) < math.radians(15)
    assert pitch < 0 and 60.0 <= fov <= 100.0


def test_bev_image_up_is_scenario_plus_y(site):
    cam = make_rig("bev", site, tb.SCENARIOS[1], "draft").camera(0.0)
    a, b = site.to_world(5.0, 0.0), site.to_world(5.0, 5.0)
    uv, valid, _ = project_points(np.array([a, b]), cam)
    assert valid.all() and uv[1, 1] < uv[0, 1]


def test_fisheye_faces(site):
    rig = make_rig("fisheye", site, tb.SCENARIOS[1], "draft")
    assert set(rig.sensor_models(0.0)) == {f"fisheye/{f}" for f in rig.remap.faces}
