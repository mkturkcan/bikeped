"""Rotation conventions and camera projection for rendered views.

All world quantities are in the CARLA world frame (metres, left-handed, ``z``
up) and rotations are ``(pitch, yaw, roll)`` in degrees with the Unreal
convention of ``carla.Rotation``. A camera is a :class:`~bikeped.sim.types.CameraModel`;
its camera coordinates are ``(f, r, u)`` along ``forward``, ``right`` and ``up``.

Pixel conventions follow the source of the intrinsics. Pinhole models use the
CARLA convention, principal point at ``(width / 2, height / 2)``. Fisheye models
use the calibration convention of ``camera_calibration.json`` and
``decision_testbench.FisheyeCamera``, in which pixel centres sit at integer
coordinates.
"""

from __future__ import annotations

import math
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from .types import AgentState, CameraModel

#: Pinhole near plane in metres. Geometry closer than this is clipped.
NEAR_PLANE_M = 0.05

#: The 12 edges of a box as pairs of corner indices (see :func:`box_corners_world`).
BOX_EDGES: Tuple[Tuple[int, int], ...] = tuple(
    (i, i | bit) for i in range(8) for bit in (4, 2, 1) if not i & bit
)

#: The bottom face of a box as a closed loop of corner indices.
BOX_BASE: Tuple[int, int, int, int] = (0, 2, 6, 4)

Vec = Tuple[float, float, float]


def carla_rotation_to_basis(pitch: float, yaw: float, roll: float) -> Tuple[Vec, Vec, Vec]:
    """Return the world-frame ``(forward, right, up)`` unit vectors of a rotation.

    Angles are in degrees. The result equals ``carla.Transform(rotation=...)``
    ``.get_forward_vector()``, ``.get_right_vector()`` and ``.get_up_vector()``.
    """
    p, y, r = np.radians([pitch, yaw, roll])
    cp, sp = math.cos(p), math.sin(p)
    cy, sy = math.cos(y), math.sin(y)
    cr, sr = math.cos(r), math.sin(r)
    forward = (cp * cy, cp * sy, sp)
    right = (sr * sp * cy - cr * sy, sr * sp * sy + cr * cy, -sr * cp)
    up = (-(cr * sp * cy + sr * sy), cy * sr - cr * sp * sy, cr * cp)
    return forward, right, up


def basis_to_carla_rotation(forward: Sequence[float], right: Sequence[float],
                            up: Sequence[float]) -> Tuple[float, float, float]:
    """Return ``(pitch, yaw, roll)`` in degrees for an orthonormal camera basis.

    Inverse of :func:`carla_rotation_to_basis`. At pitch of plus or minus 90
    degrees only ``yaw + roll`` (or ``yaw - roll``) is defined; roll is then set
    to zero and the whole in-plane rotation goes into yaw.
    """
    f = np.asarray(forward, dtype=np.float64)
    r = np.asarray(right, dtype=np.float64)
    u = np.asarray(up, dtype=np.float64)
    pitch = math.degrees(math.asin(float(np.clip(f[2], -1.0, 1.0))))
    if math.hypot(f[0], f[1]) > 1e-9:
        yaw = math.degrees(math.atan2(f[1], f[0]))
        roll = math.degrees(math.atan2(-r[2], u[2]))
    else:
        # With roll = 0 the right vector is (-sin yaw, cos yaw, 0) at any pitch.
        yaw = math.degrees(math.atan2(-r[0], r[1]))
        roll = 0.0
    return pitch, yaw, roll


def box_corners_world(location: Sequence[float], rotation: Sequence[float],
                      extent: Sequence[float]) -> np.ndarray:
    """Return the 8 world-frame corners of an oriented box as an ``(8, 3)`` array.

    ``location`` is the box centre, ``rotation`` ``(pitch, yaw, roll)`` in
    degrees, ``extent`` the half sizes along the box forward, right and up axes.
    Corner ``4*ix + 2*iy + iz`` lies on the minus (0) or plus (1) side of each
    axis, so corners 0, 2, 6, 4 form the bottom face.
    """
    basis = np.array(carla_rotation_to_basis(*rotation), dtype=np.float64)
    ext = np.asarray(extent, dtype=np.float64)
    signs = np.array([[1 if i & 4 else -1, 1 if i & 2 else -1, 1 if i & 1 else -1]
                      for i in range(8)], dtype=np.float64)
    return np.asarray(location, dtype=np.float64) + (signs * ext) @ basis


def world_to_camera(points_w: np.ndarray, cam: CameraModel) -> np.ndarray:
    """Return camera coordinates ``(f, r, u)`` in metres of ``(N, 3)`` world points."""
    pts = np.asarray(points_w, dtype=np.float64).reshape(-1, 3)
    return (pts - np.asarray(cam.position, dtype=np.float64)) @ cam.basis().T


def fisheye_focal(cam: CameraModel) -> float:
    """Equidistant focal length in pixels per radian of a fisheye model."""
    if cam.f_px_per_rad is not None:
        return float(cam.f_px_per_rad)
    if cam.circle_radius_px is None:
        raise ValueError(f"fisheye camera {cam.name!r} needs f_px_per_rad or circle_radius_px")
    return float(cam.circle_radius_px) / math.radians(cam.fov_deg / 2.0)


def _fisheye_radius(theta: np.ndarray, cam: CameraModel) -> np.ndarray:
    t2 = theta * theta
    return fisheye_focal(cam) * theta * (1.0 + t2 * (cam.k1 + cam.k2 * t2))


def _fisheye_theta(radius: np.ndarray, cam: CameraModel) -> np.ndarray:
    norm = radius / fisheye_focal(cam)
    theta = norm.copy()
    if cam.k1 or cam.k2:
        for _ in range(8):
            t2 = theta * theta
            value = theta * (1.0 + t2 * (cam.k1 + cam.k2 * t2)) - norm
            slope = 1.0 + t2 * (3.0 * cam.k1 + 5.0 * cam.k2 * t2)
            theta = theta - value / slope
    return theta


def camera_to_image(fru: np.ndarray, cam: CameraModel) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Project camera coordinates ``(N, 3)`` to pixels; see :func:`project_points`."""
    fru = np.asarray(fru, dtype=np.float64).reshape(-1, 3)
    f, r, u = fru[:, 0], fru[:, 1], fru[:, 2]
    cx, cy = cam.principal_point
    if cam.kind == "pinhole":
        valid = f > NEAR_PLANE_M
        safe = np.where(valid, f, 1.0)
        fpx = cam.focal_px
        uv = np.stack([cx + fpx * r / safe, cy - fpx * u / safe], axis=1)
        return uv, valid, f.copy()
    if cam.kind == "fisheye":
        rng = np.linalg.norm(fru, axis=1)
        lateral = np.hypot(r, u)
        theta = np.arctan2(lateral, f)
        rad = _fisheye_radius(theta, cam)
        scale = np.where(lateral > 1e-12, rad / np.maximum(lateral, 1e-12), 0.0)
        uv = np.stack([cx + scale * r, cy - scale * u], axis=1)
        valid = (theta <= math.radians(cam.fov_deg / 2.0)) & (rng > 1e-9)
        return uv, valid, rng
    raise ValueError(f"unknown camera kind {cam.kind!r}")


def project_points(points_w: np.ndarray,
                   cam: CameraModel) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Project world points to pixels.

    Returns ``uv`` ``(N, 2)``, ``valid`` ``(N,)`` and ``depth`` ``(N,)``.
    For a pinhole camera a point is valid in front of the near plane and depth
    is the distance along the optical axis, as in a CARLA depth image. For a
    fisheye a point is valid inside the field of view and depth is the
    Euclidean range. Validity does not test the image bounds.
    """
    return camera_to_image(world_to_camera(points_w, cam), cam)


def _clip_near(a: np.ndarray, b: np.ndarray) -> Optional[Tuple[np.ndarray, np.ndarray]]:
    """Clip a camera-frame segment to ``f >= NEAR_PLANE_M``."""
    fa, fb = a[0] - NEAR_PLANE_M, b[0] - NEAR_PLANE_M
    if fa < 0 and fb < 0:
        return None
    if fa < 0:
        a = a + (b - a) * (fa / (fa - fb))
    elif fb < 0:
        b = b + (a - b) * (fb / (fb - fa))
    return a, b


def _segment_samples(a: np.ndarray, b: np.ndarray, cam: CameraModel, max_step_px: float) -> int:
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    if na < 1e-9 or nb < 1e-9:
        return 32
    angle = math.acos(float(np.clip(np.dot(a, b) / (na * nb), -1.0, 1.0)))
    return int(np.clip(math.ceil(angle * fisheye_focal(cam) / max_step_px), 1, 512)) + 1


def project_polyline(points_w: np.ndarray, cam: CameraModel, *, max_step_px: float = 6.0,
                     closed: bool = False) -> List[np.ndarray]:
    """Project a world polyline to a list of image polylines ``(M, 2)``, ``M >= 2``.

    Pinhole segments are clipped exactly at the near plane and stay straight.
    Fisheye segments are resampled so that consecutive samples are at most
    about ``max_step_px`` apart, since straight lines map to curves, and the
    polyline is split where it leaves the field of view.
    """
    pts = np.asarray(points_w, dtype=np.float64).reshape(-1, 3)
    if closed and len(pts) > 2:
        pts = np.vstack([pts, pts[:1]])
    if len(pts) < 2:
        return []
    fru = world_to_camera(pts, cam)
    pieces: List[np.ndarray] = []
    current: List[np.ndarray] = []

    def flush() -> None:
        if len(current) >= 2:
            pieces.append(np.vstack(current))
        current.clear()

    for a, b in zip(fru[:-1], fru[1:]):
        if cam.kind == "pinhole":
            clipped = _clip_near(a, b)
            if clipped is None:
                flush()
                continue
            ca, cb = clipped
            uv, _, _ = camera_to_image(np.vstack([ca, cb]), cam)
            if current and ca is not a:
                flush()
            if not current:
                current.append(uv[:1])
            current.append(uv[1:])
            if cb is not b:
                flush()
        else:
            n = _segment_samples(a, b, cam, max_step_px)
            s = np.linspace(0.0, 1.0, n + 1)[:, None]
            uv, valid, _ = camera_to_image(a + (b - a) * s, cam)
            start = 1 if current else 0
            for k in range(start, len(uv)):
                if valid[k]:
                    current.append(uv[k:k + 1])
                else:
                    flush()
    flush()
    return pieces


def project_box(agent: AgentState, cam: CameraModel) -> Optional[Dict[str, object]]:
    """Project an agent's oriented 3D box into a view.

    Returns ``None`` when no part of the box is in front of the camera or
    inside the image. Otherwise a dict with ``bbox`` ``[x1, y1, x2, y2]``
    clipped to the image, ``corners_image`` ``(8, 2)`` with NaN rows for
    corners behind the near plane or outside the field of view,
    ``truncation`` (fraction of the unclipped box area outside the image) and
    ``distance_m`` (camera to box centre). Pinhole edges are clipped against
    the near plane one by one, so a box that straddles the camera keeps its
    true extent; fisheye edges are sampled densely because they curve.
    """
    corners = box_corners_world(agent.location, agent.rotation, agent.extent)
    uv, valid, _ = project_points(corners, cam)
    corners_image = np.where(valid[:, None], uv, np.nan)
    samples: List[np.ndarray] = []
    for i, j in BOX_EDGES:
        for piece in project_polyline(corners[[i, j]], cam, max_step_px=4.0):
            samples.append(piece)
    if not samples:
        return None
    allpts = np.vstack(samples)
    x1, y1 = allpts.min(axis=0)
    x2, y2 = allpts.max(axis=0)
    cx1, cy1 = max(x1, 0.0), max(y1, 0.0)
    cx2, cy2 = min(x2, float(cam.width)), min(y2, float(cam.height))
    if cx2 <= cx1 or cy2 <= cy1:
        return None
    full = max((x2 - x1) * (y2 - y1), 1e-12)
    inside = (cx2 - cx1) * (cy2 - cy1)
    distance = float(np.linalg.norm(np.asarray(agent.location) - np.asarray(cam.position)))
    return {
        "bbox": [float(cx1), float(cy1), float(cx2), float(cy2)],
        "corners_image": corners_image,
        "truncation": float(np.clip(1.0 - inside / full, 0.0, 1.0)),
        "distance_m": distance,
    }


def pixel_rays(uv: np.ndarray, cam: CameraModel) -> Tuple[np.ndarray, np.ndarray]:
    """Return world-frame unit ray directions ``(N, 3)`` and a validity mask for pixels."""
    pix = np.asarray(uv, dtype=np.float64).reshape(-1, 2)
    cx, cy = cam.principal_point
    du, dv = pix[:, 0] - cx, pix[:, 1] - cy
    if cam.kind == "pinhole":
        fpx = cam.focal_px
        fru = np.stack([np.ones(len(pix)), du / fpx, -dv / fpx], axis=1)
        valid = np.ones(len(pix), dtype=bool)
    elif cam.kind == "fisheye":
        rad = np.hypot(du, dv)
        theta = _fisheye_theta(rad, cam)
        phi = np.arctan2(dv, du)
        st = np.sin(theta)
        fru = np.stack([np.cos(theta), st * np.cos(phi), -st * np.sin(phi)], axis=1)
        valid = theta <= math.radians(cam.fov_deg / 2.0)
    else:
        raise ValueError(f"unknown camera kind {cam.kind!r}")
    rays = fru @ cam.basis()
    rays /= np.linalg.norm(rays, axis=1, keepdims=True)
    return rays, valid


def pixel_to_ground(uv: np.ndarray, cam: CameraModel,
                    ground_z: float = 0.0) -> Tuple[np.ndarray, np.ndarray]:
    """Intersect pixel rays with the horizontal plane ``z = ground_z``.

    Returns world points ``(N, 3)`` (NaN where invalid) and ``valid`` ``(N,)``,
    false for rays outside the field of view or not pointing at the plane.
    """
    rays, valid = pixel_rays(uv, cam)
    pos = np.asarray(cam.position, dtype=np.float64)
    rz = rays[:, 2]
    with np.errstate(divide="ignore", invalid="ignore"):
        t = (ground_z - pos[2]) / rz
    valid = valid & (np.abs(rz) > 1e-9) & (t > 0)
    pts = pos + rays * np.where(valid, t, 0.0)[:, None]
    pts[~valid] = np.nan
    return pts, valid
