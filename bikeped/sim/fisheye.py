"""The deployed fisheye lens as a sim camera, and cube-face to fisheye remapping.

CARLA has no equidistant fisheye sensor, so the fisheye view is rendered as a
set of 90 degree pinhole cameras sharing one position (the faces of a cube
around the fisheye's optical axis) and resampled into the fisheye image with
a lookup table built once by :class:`FisheyeRemap`.

:func:`deployed_fisheye_model` reproduces the lens of the field deployment.
The calibration describes a 3500 x 3500 frame centred on the image circle; the
physical sensor is 3840 x 2160, which is that frame padded to 3840 x 3840 and
cropped to the middle 2160 rows, so the top and bottom of the circle are cut
(rows 670 to 2829 of the 3500 px frame are on the sensor). The sim camera
renders the sensor image, so its principal point is the calibrated one
shifted by ``(+170, -670)`` pixels. With the same pose it agrees with
``decision_testbench.FisheyeCamera.world_to_pixel`` after that shift.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Dict, Optional, Sequence, Tuple, Union

import cv2
import numpy as np

from .projection import basis_to_carla_rotation, carla_rotation_to_basis, pixel_rays
from .types import CameraModel

_REPO_ROOT = Path(__file__).resolve().parents[2]

#: Physical sensor size in pixels.
SENSOR_SIZE: Tuple[int, int] = (3840, 2160)
#: Side of the square calibration frame and of the padded frame, in pixels.
CROP_SIZE = 3500
PAD_SIZE = 3840

#: Cube faces in the fisheye camera frame: (forward, right, up) of each face as
#: signed combinations of the fisheye axes (F, R, U). All are proper rotations.
_FACE_AXES: Dict[str, Tuple[Tuple[int, int], Tuple[int, int], Tuple[int, int]]] = {
    "front": ((0, 1), (1, 1), (2, 1)),
    "right": ((1, 1), (0, -1), (2, 1)),
    "back": ((0, -1), (1, -1), (2, 1)),
    "left": ((1, -1), (0, 1), (2, 1)),
    "up": ((2, 1), (1, 1), (0, -1)),
    "down": ((2, -1), (1, 1), (0, 1)),
}

PathLike = Union[str, Path]


def _resolve(path: PathLike) -> Path:
    p = Path(path)
    if not p.exists() and not p.is_absolute():
        p = _REPO_ROOT / p
    return p


def _load_camera_parameters(calibration: Optional[PathLike],
                            config: Optional[PathLike]) -> Dict[str, float]:
    params: Dict[str, float] = {"fov_deg": 197.9, "radius_px": 1750.0, "cx": (CROP_SIZE - 1) / 2.0,
                                "cy": (CROP_SIZE - 1) / 2.0, "k1": 0.0, "k2": 0.0,
                                "pitch_deg": 0.0}
    if config is not None and _resolve(config).exists():
        import yaml

        with open(_resolve(config)) as fh:
            cam = (yaml.safe_load(fh) or {}).get("camera", {}) or {}
        for src, dst in (("fov", "fov_deg"), ("radius", "radius_px"), ("cx", "cx"), ("cy", "cy"),
                         ("k1", "k1"), ("k2", "k2"), ("pitch_deg", "pitch_deg")):
            if cam.get(src) is not None:
                params[dst] = float(cam[src])
    if calibration is not None and _resolve(calibration).exists():
        with open(_resolve(calibration)) as fh:
            cal = json.load(fh)
        for key in ("fov_deg", "radius_px", "cx", "cy", "k1", "k2"):
            if cal.get(key) is not None:
                params[key] = float(cal[key])
    return params


def deployed_fisheye_model(position: Sequence[float], yaw_deg: float = 0.0, *,
                           pitch_deg: Optional[float] = None,
                           calibration: Optional[PathLike] = "camera_calibration.json",
                           config: Optional[PathLike] = "config.yaml",
                           scale: float = 1.0, name: str = "fisheye") -> CameraModel:
    """Return a :class:`CameraModel` of the deployed fisheye at ``position``.

    ``position`` is in world metres and ``yaw_deg`` is the CARLA yaw of the
    optical axis. ``pitch_deg`` defaults to ``camera.pitch_deg`` in the config
    (0, a level camera, as in the deployment and in the testbench). Intrinsics
    come from the calibration file, falling back to the config; relative paths
    are resolved against the repository root when not found in the working
    directory. The projection is equidistant with optional ``k1``/``k2``
    terms and ``f = radius / (fov / 2)``, the same focal length the testbench
    uses. ``scale`` resamples the 3840 x 2160 sensor (0.5 gives 1920 x 1080)
    and scales the intrinsics to match.

    Mapping to the testbench: a testbench heading ``h`` (scenario frame,
    counter-clockwise from ``+x``) is CARLA yaw ``-h`` once the scenario
    ``y`` axis is flipped into the left-handed world, so the testbench default
    (heading 90 degrees) is ``yaw_deg=-90``.
    """
    p = _load_camera_parameters(calibration, config)
    pitch = p["pitch_deg"] if pitch_deg is None else float(pitch_deg)
    forward, right, up = carla_rotation_to_basis(pitch, yaw_deg, 0.0)
    x_off = (PAD_SIZE - CROP_SIZE) // 2
    y_off = (PAD_SIZE - CROP_SIZE) // 2 - (PAD_SIZE - SENSOR_SIZE[1]) // 2
    width = max(2, int(round(SENSOR_SIZE[0] * scale / 2.0)) * 2)
    height = max(2, int(round(SENSOR_SIZE[1] * scale / 2.0)) * 2)
    sx, sy = width / SENSOR_SIZE[0], height / SENSOR_SIZE[1]
    # Pixel centres sit at integer coordinates, so scaling acts on (x + 0.5).
    cx = (p["cx"] + x_off + 0.5) * sx - 0.5
    cy = (p["cy"] + y_off + 0.5) * sy - 0.5
    radius = p["radius_px"] * sx
    return CameraModel(
        name=name, kind="fisheye", width=width, height=height,
        position=tuple(float(v) for v in position), forward=forward, right=right, up=up,
        fov_deg=p["fov_deg"], cx=cx, cy=cy,
        f_px_per_rad=radius / math.radians(p["fov_deg"] / 2.0),
        k1=p["k1"], k2=p["k2"], circle_radius_px=radius,
    )


_MAP_COLS = 4096


def _as_map(values: np.ndarray) -> np.ndarray:
    """Pack a 1D coordinate list into a 2D float32 map; cv2.remap limits each side to 32767."""
    rows = -(-values.size // _MAP_COLS)
    packed = np.full(rows * _MAP_COLS, -1.0, dtype=np.float32)
    packed[:values.size] = values
    return packed.reshape(rows, _MAP_COLS)


def _output_grid(model: CameraModel, supersample: int) -> Tuple[np.ndarray, np.ndarray]:
    s = int(supersample)
    xs = (np.arange(model.width * s, dtype=np.float64) + 0.5) / s - 0.5
    ys = (np.arange(model.height * s, dtype=np.float64) + 0.5) / s - 0.5
    return np.meshgrid(xs, ys)


class FisheyeRemap:
    """Lookup table from cube-face pinhole images to a fisheye image.

    Every face is a square ``face_size`` pinhole camera with a 90 degree field
    of view at the fisheye position. Faces are oriented relative to the
    fisheye's optical axis (``front``, ``right``, ``back``, ``left``, ``up``,
    ``down``); :attr:`faces` lists only those that receive pixels inside the
    image circle and on the sensor. ``supersample`` renders the lookup at an
    integer multiple of the output resolution and averages down, which
    anti-aliases fine texture near the rim of the circle.
    """

    def __init__(self, model: CameraModel, face_size: int = 1024, supersample: int = 1,
                 interpolation: int = cv2.INTER_LINEAR, feather_px: float = 1.5) -> None:
        if model.kind != "fisheye":
            raise ValueError("FisheyeRemap needs a fisheye CameraModel")
        self.model = model
        self.face_size = int(face_size)
        self.supersample = max(1, int(supersample))
        self.interpolation = interpolation
        gx, gy = _output_grid(model, self.supersample)
        shape = gx.shape
        rays, valid = pixel_rays(np.stack([gx.ravel(), gy.ravel()], axis=1), model)
        fru = rays @ model.basis().T
        signed = np.concatenate([fru, -fru], axis=1)
        choice = np.argmax(signed, axis=1)
        names = {(0, 1): "front", (1, 1): "right", (2, 1): "up",
                 (0, -1): "back", (1, -1): "left", (2, -1): "down"}
        self._lookup: Dict[str, Tuple[np.ndarray, np.ndarray, np.ndarray]] = {}
        half = self.face_size / 2.0
        for k in range(6):
            face = names[(k % 3, 1 if k < 3 else -1)]
            idx = np.flatnonzero(valid & (choice == k))
            if idx.size == 0:
                continue
            axes = _FACE_AXES[face]
            comp = [fru[idx, a] * s for a, s in axes]
            f = np.maximum(comp[0], 1e-9)
            # Face cameras follow the CARLA pinhole convention: principal point
            # at the image centre and pixel i covering [i, i + 1).
            mx = half + half * comp[1] / f - 0.5
            my = half - half * comp[2] / f - 0.5
            self._lookup[face] = (idx, _as_map(mx), _as_map(my))
        order = ("down", "front", "right", "back", "left", "up")
        self.faces: Tuple[str, ...] = tuple(f for f in order if f in self._lookup)
        self._shape = shape
        cx, cy = model.principal_point
        ox, oy = _output_grid(model, 1)
        radius = np.hypot(ox - cx, oy - cy)
        edge = float(model.circle_radius_px) if model.circle_radius_px else float(radius.max())
        feather = max(float(feather_px), 1e-3)
        self.mask = np.clip((edge - radius) / feather + 0.5, 0.0, 1.0).astype(np.float32)

    def face_camera(self, face: str) -> CameraModel:
        """Return the 90 degree square pinhole :class:`CameraModel` of a cube face.

        Its basis converts to a CARLA rotation with
        :func:`~bikeped.sim.projection.basis_to_carla_rotation`.
        """
        if face not in _FACE_AXES:
            raise KeyError(f"unknown face {face!r}")
        b = self.model.basis()
        f, r, u = (b[a] * s for a, s in _FACE_AXES[face])
        return CameraModel(name=f"{self.model.name}/{face}", kind="pinhole",
                           width=self.face_size, height=self.face_size,
                           position=self.model.position, forward=tuple(f), right=tuple(r),
                           up=tuple(u), fov_deg=90.0)

    def face_rotation(self, face: str) -> Tuple[float, float, float]:
        """CARLA ``(pitch, yaw, roll)`` in degrees of a face camera."""
        cam = self.face_camera(face)
        return basis_to_carla_rotation(cam.forward, cam.right, cam.up)

    def remap(self, face_images: Dict[str, np.ndarray]) -> np.ndarray:
        """Resample BGR face images into the fisheye image (``uint8``, black outside the circle).

        ``face_images`` must hold every face in :attr:`faces`. Images of a
        different size than ``face_size`` are accepted and sampled in proportion.
        """
        h, w = self._shape
        out = np.zeros((h * w, 3), dtype=np.float32)
        for face in self.faces:
            if face not in face_images:
                raise KeyError(f"missing face image {face!r}; need {self.faces}")
            img = np.asarray(face_images[face])
            if img.ndim == 2:
                img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
            img = img[..., :3]
            idx, mx, my = self._lookup[face]
            sx, sy = img.shape[1] / self.face_size, img.shape[0] / self.face_size
            if sx != 1.0 or sy != 1.0:
                mx = ((mx + 0.5) * sx - 0.5).astype(np.float32)
                my = ((my + 0.5) * sy - 0.5).astype(np.float32)
            sampled = cv2.remap(img, mx, my, self.interpolation, borderMode=cv2.BORDER_REPLICATE)
            out[idx] = sampled.reshape(-1, 3)[:idx.size]
        out = out.reshape(h, w, 3)
        if self.supersample > 1:
            out = cv2.resize(out, (self.model.width, self.model.height),
                             interpolation=cv2.INTER_AREA)
        out *= self.mask[..., None]
        return np.clip(out + 0.5, 0, 255).astype(np.uint8)


def synthetic_faces(remap: FisheyeRemap, pattern: str = "grid", *, ground_z: float = 0.0,
                    cell_m: float = 1.0,
                    markers: Sequence[Sequence[float]] = ()) -> Dict[str, np.ndarray]:
    """Render face images of an analytic scene, for tests and demos.

    The scene is the plane ``z = ground_z`` (world metres) painted as a
    checkerboard (``pattern='checker'``) or a light grid on asphalt grey
    (``pattern='grid'``) with ``cell_m`` cells, under a vertical sky gradient.
    Each entry of ``markers`` is a world ``(x, y)`` drawn as a pure red
    (BGR ``(0, 0, 255)``) disc of 0.15 m radius.
    """
    if pattern not in ("grid", "checker"):
        raise ValueError(f"unknown pattern {pattern!r}")
    images: Dict[str, np.ndarray] = {}
    n = remap.face_size
    xs = np.arange(n, dtype=np.float64) + 0.5
    gu, gv = np.meshgrid(xs, xs)
    uv = np.stack([gu.ravel(), gv.ravel()], axis=1)
    pos = np.asarray(remap.model.position, dtype=np.float64)
    for face in remap.faces:
        cam = remap.face_camera(face)
        rays, _ = pixel_rays(uv, cam)
        elev = np.clip(rays[:, 2], -1.0, 1.0)
        sky = np.stack([235 - 60 * elev, 205 - 70 * elev, 170 - 90 * elev], axis=1)
        img = np.clip(sky, 0, 255)
        with np.errstate(divide="ignore", invalid="ignore"):
            t = (ground_z - pos[2]) / rays[:, 2]
        hit = np.isfinite(t) & (t > 0)
        gx = pos[0] + rays[:, 0] * np.where(hit, t, 0.0)
        gy = pos[1] + rays[:, 1] * np.where(hit, t, 0.0)
        if pattern == "checker":
            parity = (np.floor(gx / cell_m) + np.floor(gy / cell_m)) % 2
            ground = np.where(parity[:, None] > 0, [[200, 200, 200]], [[60, 60, 60]])
        else:
            fx = np.abs(gx / cell_m - np.round(gx / cell_m))
            fy = np.abs(gy / cell_m - np.round(gy / cell_m))
            line = np.minimum(fx, fy) * cell_m < 0.03
            ground = np.where(line[:, None], [[215, 215, 210]], [[78, 76, 74]])
        for mx_, my_ in markers:
            disc = np.hypot(gx - mx_, gy - my_) < 0.15
            ground = np.where(disc[:, None], [[0, 0, 255]], ground)
        img = np.where(hit[:, None], ground, img)
        images[face] = img.reshape(n, n, 3).astype(np.uint8)
    return images
