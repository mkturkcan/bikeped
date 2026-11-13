"""Tests for the deployed fisheye model and the cube-face remap; no simulator needed."""

from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np
import pytest

import decision_testbench as tb
from bikeped.sim.fisheye import FisheyeRemap, deployed_fisheye_model, synthetic_faces
from bikeped.sim.groundtruth import scenario_to_world
from bikeped.sim.projection import carla_rotation_to_basis, project_points

ROOT = Path(__file__).resolve().parents[2]


def _testbench_camera() -> tb.FisheyeCamera:
    cal = json.loads((ROOT / "camera_calibration.json").read_text())
    return tb.FisheyeCamera(cx_override=cal["cx"], cy_override=cal["cy"], k1=cal["k1"],
                            k2=cal["k2"], fov_deg=cal["fov_deg"], radius_px=int(cal["radius_px"]))


def test_matches_testbench_on_ground_grid():
    ref = _testbench_camera()
    model = deployed_fisheye_model(scenario_to_world(ref.cam_x, ref.cam_y, ref.height), -90.0)
    assert (model.width, model.height) == (3840, 2160)
    xs, ys = np.meshgrid(np.linspace(-25, 35, 41), np.linspace(-12.9, 20, 31))
    ground = np.column_stack([xs.ravel(), ys.ravel(), np.zeros(xs.size)])
    expected, ok_ref = ref.world_to_pixel(ground)
    world = np.array([scenario_to_world(*p) for p in ground])
    uv, ok, _ = project_points(world, model)
    assert ok_ref.sum() > 100
    assert np.array_equal(ok, ok_ref)
    # The sensor image is the calibration frame shifted by the pad-and-crop offsets.
    assert np.abs(uv[ok] - (expected[ok] + [170.0, -670.0])).max() < 1e-6


def test_sensor_band_matches_testbench():
    ref = _testbench_camera()
    model = deployed_fisheye_model(scenario_to_world(ref.cam_x, ref.cam_y, ref.height), -90.0)
    assert ref.sensor_v_min == 670 and ref.sensor_v_max == 2830
    assert model.cy == pytest.approx(ref.ycenter - ref.sensor_v_min)
    assert model.cx == pytest.approx(ref.xcenter - ref.sensor_u_min)


def test_scale_halves_intrinsics():
    full = deployed_fisheye_model((0, 0, 4))
    half = deployed_fisheye_model((0, 0, 4), scale=0.5)
    assert (half.width, half.height) == (1920, 1080)
    assert half.f_px_per_rad == pytest.approx(full.f_px_per_rad / 2)
    point = np.array([[3.0, -2.0, 0.0]])
    a, _, _ = project_points(point, full)
    b, _, _ = project_points(point, half)
    assert np.allclose((b + 0.5) * 2, a + 0.5, atol=1e-9)


@pytest.fixture(scope="module")
def remap() -> FisheyeRemap:
    model = deployed_fisheye_model((0.0, 0.0, 4.0), 30.0, scale=0.25)
    return FisheyeRemap(model, face_size=256, supersample=2)


def test_faces_cover_the_circle(remap):
    # 197.9 degrees reach just past the side faces but never into the back face.
    assert remap.faces == ("down", "front", "right", "left", "up")
    nadir = deployed_fisheye_model((0, 0, 4), pitch_deg=-90.0, scale=0.1)
    narrow = nadir.__class__(**{**nadir.to_dict(), "fov_deg": 80.0,
                                "f_px_per_rad": None, "circle_radius_px": 100.0})
    assert FisheyeRemap(narrow, face_size=64).faces == ("front",)


def test_face_cameras_are_square_pinholes(remap):
    for face in remap.faces:
        cam = remap.face_camera(face)
        assert cam.kind == "pinhole" and cam.width == cam.height == 256
        assert cam.fov_deg == 90.0 and cam.position == remap.model.position
        rot = remap.face_rotation(face)
        assert np.allclose(carla_rotation_to_basis(*rot), cam.basis(), atol=1e-9)
    assert np.allclose(remap.face_camera("down").forward, (0, 0, -1))


def test_remap_places_ground_markers(remap):
    markers = [(3.0, 2.0), (-4.0, 6.0), (8.0, -3.0), (2.0, 4.5)]
    image = remap.remap(synthetic_faces(remap, markers=markers))
    assert image.shape == (remap.model.height, remap.model.width, 3)
    expected, ok, _ = project_points(np.array([[x, y, 0.0] for x, y in markers]), remap.model)
    assert ok.all()
    red = (image[..., 2] > 150) & (image[..., 1] < 90) & (image[..., 0] < 90)
    count, labels = cv2.connectedComponents(red.astype(np.uint8))
    centroids = []
    for k in range(1, count):
        rows, cols = np.nonzero(labels == k)
        centroids.append((cols.mean(), rows.mean()))
    assert len(centroids) == len(markers)
    for uv in expected:
        assert min(np.hypot(c[0] - uv[0], c[1] - uv[1]) for c in centroids) < 1.0


def test_black_outside_circle(remap):
    image = remap.remap(synthetic_faces(remap, pattern="checker"))
    cx, cy = remap.model.principal_point
    r = remap.model.circle_radius_px
    assert image[int(cy), int(cx + r + 4)].max() == 0
    assert image[int(cy), int(cx)].max() > 0


def test_missing_face_raises(remap):
    faces = synthetic_faces(remap)
    faces.pop("down")
    with pytest.raises(KeyError):
        remap.remap(faces)
