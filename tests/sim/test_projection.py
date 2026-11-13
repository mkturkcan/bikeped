"""Tests for rotation conventions and projection; no simulator needed."""

from __future__ import annotations

import numpy as np
import pytest

from bikeped.sim.projection import (
    BOX_BASE,
    BOX_EDGES,
    basis_to_carla_rotation,
    box_corners_world,
    carla_rotation_to_basis,
    pixel_to_ground,
    project_box,
    project_points,
    project_polyline,
    world_to_camera,
)
from bikeped.sim.types import AgentState, CameraModel

RNG = np.random.default_rng(7)
ANGLES = [tuple(a) for a in np.column_stack([RNG.uniform(-89, 89, 40), RNG.uniform(-180, 180, 40),
                                             RNG.uniform(-180, 180, 40)])]


def _pinhole(position=(0.0, 0.0, 2.0), rotation=(-10.0, 0.0, 0.0), width=640, height=360,
             fov=90.0) -> CameraModel:
    f, r, u = carla_rotation_to_basis(*rotation)
    return CameraModel(name="cam", kind="pinhole", width=width, height=height,
                       position=position, forward=f, right=r, up=u, fov_deg=fov)


def _fisheye(position=(0.0, 0.0, 4.0), rotation=(-90.0, 0.0, 0.0), size=600) -> CameraModel:
    f, r, u = carla_rotation_to_basis(*rotation)
    return CameraModel(name="fe", kind="fisheye", width=size, height=size, position=position,
                       forward=f, right=r, up=u, fov_deg=190.0, circle_radius_px=size / 2.0,
                       k1=0.01, k2=-0.002)


@pytest.mark.parametrize("angles", ANGLES)
def test_basis_is_orthonormal_and_inverts(angles):
    f, r, u = (np.array(v) for v in carla_rotation_to_basis(*angles))
    m = np.array([f, r, u])
    assert np.allclose(m @ m.T, np.eye(3), atol=1e-12)
    back = carla_rotation_to_basis(*basis_to_carla_rotation(f, r, u))
    assert np.allclose(back, (f, r, u), atol=1e-9)


def test_basis_known_values():
    f, r, u = carla_rotation_to_basis(0.0, 90.0, 0.0)
    assert np.allclose(f, (0, 1, 0)) and np.allclose(r, (-1, 0, 0)) and np.allclose(u, (0, 0, 1))
    f, r, u = carla_rotation_to_basis(-90.0, 0.0, 0.0)
    assert np.allclose(f, (0, 0, -1)) and np.allclose(r, (0, 1, 0)) and np.allclose(u, (1, 0, 0))


def test_nadir_rotation_round_trip():
    basis = carla_rotation_to_basis(-90.0, 37.0, 0.0)
    pitch, yaw, roll = basis_to_carla_rotation(*basis)
    assert pitch == pytest.approx(-90.0) and roll == 0.0
    assert np.allclose(carla_rotation_to_basis(pitch, yaw, roll), basis, atol=1e-9)


def test_matches_carla_client():
    carla = pytest.importorskip("carla")
    for pitch, yaw, roll in ANGLES:
        tr = carla.Transform(carla.Location(), carla.Rotation(pitch=pitch, yaw=yaw, roll=roll))
        ref = [tr.get_forward_vector(), tr.get_right_vector(), tr.get_up_vector()]
        for vec, mine in zip(ref, carla_rotation_to_basis(pitch, yaw, roll)):
            assert np.allclose([vec.x, vec.y, vec.z], mine, atol=1e-5)


def test_box_corners_layout():
    corners = box_corners_world((1.0, 2.0, 1.0), (0.0, 90.0, 0.0), (2.0, 1.0, 0.5))
    assert corners.shape == (8, 3)
    assert np.allclose(corners.mean(axis=0), (1, 2, 1))
    assert np.allclose(corners[list(BOX_BASE), 2], 0.5)
    assert np.allclose(corners[4] - corners[0], (0, 4, 0), atol=1e-12)
    assert len(BOX_EDGES) == 12
    for i, j in BOX_EDGES:
        assert bin(i ^ j).count("1") == 1


def test_pinhole_centre_and_axes():
    cam = _pinhole(rotation=(0.0, 0.0, 0.0))
    pts = np.array([[10.0, 0.0, 2.0], [10.0, 1.0, 2.0], [10.0, 0.0, 3.0], [-1.0, 0.0, 2.0]])
    uv, valid, depth = project_points(pts, cam)
    assert np.allclose(uv[0], (320, 180))
    assert uv[1, 0] > 320 and uv[2, 1] < 180
    assert valid.tolist() == [True, True, True, False]
    assert np.allclose(depth[:3], 10.0)
    assert np.allclose(world_to_camera(pts[:1], cam), [[10, 0, 0]])


@pytest.mark.parametrize("make", [_pinhole, _fisheye])
def test_pixel_to_ground_inverts_projection(make):
    cam = make()
    xs, ys = np.meshgrid(np.linspace(3, 12, 6), np.linspace(-4, 4, 5))
    ground = np.column_stack([xs.ravel(), ys.ravel(), np.zeros(xs.size)])
    uv, valid, _ = project_points(ground, cam)
    back, ok = pixel_to_ground(uv[valid], cam, 0.0)
    assert ok.all()
    assert np.allclose(back, ground[valid], atol=1e-6)


def test_pixel_to_ground_rejects_sky():
    cam = _pinhole(rotation=(0.0, 0.0, 0.0))
    pts, ok = pixel_to_ground(np.array([[320.0, 10.0], [320.0, 350.0]]), cam)
    assert ok.tolist() == [False, True]
    assert np.isnan(pts[0]).all()


def _agent(location, yaw=0.0, extent=(0.9, 0.3, 0.85)) -> AgentState:
    return AgentState(id=1, kind="bike", sx=0.0, sy=0.0, heading=0.0, speed=0.0,
                      location=location, rotation=(0.0, yaw, 0.0), extent=extent)


def test_project_box_inside_image():
    cam = _pinhole()
    box = project_box(_agent((10.0, 0.0, 0.85)), cam)
    x1, y1, x2, y2 = box["bbox"]
    assert 0 <= x1 < x2 <= cam.width and 0 <= y1 < y2 <= cam.height
    assert box["truncation"] == pytest.approx(0.0)
    assert box["distance_m"] == pytest.approx(np.linalg.norm([10.0, 0.0, -1.15]))
    finite = box["corners_image"][np.isfinite(box["corners_image"][:, 0])]
    assert np.allclose([finite[:, 0].min(), finite[:, 1].min()], [x1, y1])


def test_project_box_clips_against_near_plane():
    cam = _pinhole(rotation=(0.0, 0.0, 0.0))
    # A long box that passes beside and behind the camera: corners behind the
    # near plane must not be dropped, the edges must be cut where they cross it.
    box = project_box(_agent((0.0, 1.5, 1.0), extent=(5.0, 0.3, 0.5)), cam)
    assert box is not None
    assert np.isnan(box["corners_image"]).any()
    assert box["truncation"] > 0.5
    assert box["bbox"][2] == pytest.approx(cam.width)
    assert project_box(_agent((-10.0, 0.0, 1.0)), cam) is None


def test_project_box_truncated_at_border():
    cam = _pinhole(rotation=(0.0, 0.0, 0.0))
    box = project_box(_agent((6.0, 6.0, 1.0)), cam)
    assert box is not None and 0.0 < box["truncation"] < 1.0
    assert box["bbox"][2] == pytest.approx(cam.width)


def test_fisheye_box_uses_curved_edges():
    cam = _fisheye()
    box = project_box(_agent((2.0, 1.0, 0.85), yaw=30.0), cam)
    corners = box["corners_image"]
    x1, y1, x2, y2 = box["bbox"]
    assert x1 <= np.nanmin(corners[:, 0]) + 1e-6 and x2 >= np.nanmax(corners[:, 0]) - 1e-6
    assert y1 <= np.nanmin(corners[:, 1]) + 1e-6 and y2 >= np.nanmax(corners[:, 1]) - 1e-6


def test_polyline_pinhole_clipped_and_fisheye_dense():
    cam = _pinhole(rotation=(0.0, 0.0, 0.0))
    line = np.array([[-5.0, 1.0, 0.0], [10.0, 1.0, 0.0]])
    pieces = project_polyline(line, cam)
    assert len(pieces) == 1 and pieces[0].shape == (2, 2)
    fe = _fisheye()
    pieces = project_polyline(np.array([[-6.0, -3.0, 0.0], [6.0, -3.0, 0.0]]), fe)
    assert len(pieces) == 1 and len(pieces[0]) > 20
    steps = np.linalg.norm(np.diff(pieces[0], axis=0), axis=1)
    # Sampling is uniform in angle, so pixel steps stay within a small factor of max_step_px.
    assert steps.max() < 12.0
    ring = project_polyline(np.array([[1, 1, 0], [1, -1, 0], [-1, -1, 0], [-1, 1, 0.0]]), fe,
                            closed=True)
    assert np.allclose(ring[0][0], ring[0][-1])
