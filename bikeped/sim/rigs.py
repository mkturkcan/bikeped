"""Camera rigs: named views, their camera models and the CARLA sensors behind them.

A rig yields one :class:`~bikeped.sim.types.CameraModel` per tick (world frame,
see :mod:`bikeped.sim.types`) and owns the sensors that render it. Static rigs
(``pole``, ``street``, ``bev``, ``fisheye``) never move; ``chase`` and
``orbit`` move their sensor before every tick.

Views:

``pole``
    Pinhole camera on the pole at the testbench camera position (12 ft), aimed
    so the crosswalk and every scripted path fit in the frame.
``fisheye``
    The deployed fisheye lens, rendered as cube faces and resampled with
    :class:`~bikeped.sim.fisheye.FisheyeRemap`.
``chase``
    Follows the first cyclist (or pedestrian) from behind and above.
``orbit``
    Slow eye-level arc around the conflict point with a longer lens.
``street``
    Static eye-level tripod on the kerb.
``bev``
    Straight down from 60 m with a narrow lens, image up along scenario ``+y``.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Sequence, Tuple

import numpy as np

from .motion import Trajectory
from .session import SensorHub, depth_to_metres, image_to_bgr, instance_ids
from .site import Site
from .types import CameraModel

log = logging.getLogger(__name__)

RIG_NAMES = ("pole", "fisheye", "chase", "orbit", "street", "bev")

QUALITY_PRESETS: Dict[str, Dict[str, Any]] = {
    "draft": dict(width=960, height=540, face=512, fisheye_scale=0.25,
                  post=dict(motion_blur_intensity=0.0, bloom_intensity=0.0,
                            lens_flare_intensity=0.0)),
    "standard": dict(width=1920, height=1080, face=1024, fisheye_scale=0.5,
                     post=dict(motion_blur_intensity=0.2, motion_blur_max_distortion=0.2,
                               bloom_intensity=0.4, lens_flare_intensity=0.0, gamma=2.2)),
    "cinematic": dict(width=2560, height=1440, face=1536, fisheye_scale=0.5,
                      post=dict(motion_blur_intensity=0.35, motion_blur_max_distortion=0.25,
                                bloom_intensity=0.55, lens_flare_intensity=0.08, gamma=2.2,
                                chromatic_aberration_intensity=0.15)),
}


def rotation_to_basis(pitch: float, yaw: float, roll: float = 0.0) -> Tuple[Tuple[float, ...], ...]:
    """Unreal ``(pitch, yaw, roll)`` degrees to world ``(forward, right, up)``."""
    from .projection import carla_rotation_to_basis

    return carla_rotation_to_basis(pitch, yaw, roll)


def basis_to_rotation(forward: Sequence[float], right: Sequence[float],
                      up: Sequence[float]) -> Tuple[float, float, float]:
    from .projection import basis_to_carla_rotation

    return basis_to_carla_rotation(forward, right, up)


def look_at(position: Sequence[float], target: Sequence[float]) -> Tuple[float, float]:
    """CARLA ``(pitch, yaw)`` in degrees of a level-horizon camera aimed at ``target``."""
    d = np.asarray(target, dtype=float) - np.asarray(position, dtype=float)
    yaw = math.degrees(math.atan2(d[1], d[0]))
    pitch = math.degrees(math.atan2(d[2], math.hypot(d[0], d[1])))
    return pitch, yaw


def pinhole(name: str, position: Sequence[float], pitch: float, yaw: float, width: int,
            height: int, fov: float, roll: float = 0.0) -> CameraModel:
    f, r, u = rotation_to_basis(pitch, yaw, roll)
    return CameraModel(name=name, kind="pinhole", width=width, height=height,
                       position=tuple(float(v) for v in position), forward=tuple(f),
                       right=tuple(r), up=tuple(u), fov_deg=float(fov))


def scenario_points(scenario: Dict[str, Any]) -> np.ndarray:
    return np.array([[p["x"], p["y"]] for a in scenario["agents"] for p in a["path"]], dtype=float)


def pole_framing(site: Site, scenario: Dict[str, Any], aspect: float = 16 / 9,
                 reach_m: float = 40.0) -> Tuple[float, float, float]:
    """Heading (rad, scenario frame), pitch (deg) and horizontal FOV (deg) of the pole view.

    The view always contains the crosswalk (scenario ``x = 5``, ``|y| <= 4``) and
    every path point within ``reach_m`` of the pole.
    """
    px, py = site.pole_xy
    pts = scenario_points(scenario)
    pts = np.vstack([pts, [[5.0, -4.0], [5.0, 4.0], [3.0, 0.0], [7.0, 0.0]]])
    d = pts - np.array([px, py])
    dist = np.hypot(d[:, 0], d[:, 1])
    d = d[(dist < reach_m) & (dist > 2.0)]
    centre = math.pi / 2.0
    angles = np.arctan2(d[:, 1], d[:, 0]) - centre
    angles = (angles + math.pi) % (2 * math.pi) - math.pi
    lo, hi = float(angles.min()), float(angles.max())
    heading = centre + (lo + hi) / 2.0
    fov = float(np.clip(math.degrees(hi - lo) + 14.0, 60.0, 100.0))
    near = float(np.percentile(np.hypot(d[:, 0], d[:, 1]), 50))
    pitch = -math.degrees(math.atan2(site.pole_height - 1.0, max(near, 4.0)))
    return heading, pitch, fov


@dataclass
class SensorSpec:
    key: str
    blueprint: str
    model: CameraModel
    decode: Callable[[Any], Any]


class Rig:
    """One named view: the camera model(s) per tick and their sensors."""

    name = "rig"

    def __init__(self, site: Site, scenario: Dict[str, Any], quality: str,
                 ground_truth: bool = False) -> None:
        self.post_overrides: Dict[str, Any] = {}
        self.site = site
        self.scenario = scenario
        self.quality = QUALITY_PRESETS[quality]
        self.ground_truth = ground_truth
        self.sensors: Dict[str, Any] = {}
        self.specs: List[SensorSpec] = []

    def camera(self, t: float) -> CameraModel:
        raise NotImplementedError

    def sensor_models(self, t: float) -> Dict[str, CameraModel]:
        """Models of the sensors to render, keyed by sensor key."""
        return {self.name: self.camera(t)}

    def spawn(self, session: Any, hub: SensorHub, caps: Any) -> None:
        carla = session.carla
        library = session.world.get_blueprint_library()
        for key, model in self.sensor_models(0.0).items():
            kinds = [("sensor.camera.rgb", "", image_to_bgr)]
            if self.ground_truth and caps.depth_camera:
                kinds.append(("sensor.camera.depth", "/depth", depth_to_metres))
            if self.ground_truth and caps.instance_camera:
                kinds.append(("sensor.camera.instance_segmentation", "/instance", instance_ids))
            for bp_id, suffix, decode in kinds:
                bp = library.find(bp_id)
                values = {"image_size_x": model.width, "image_size_y": model.height,
                          "fov": model.fov_deg, "sensor_tick": 0.0}
                if suffix == "":
                    values.update(self.quality["post"])
                    values.update(self.post_overrides)
                    values["enable_postprocess_effects"] = "true"
                for attr, value in values.items():
                    if bp.has_attribute(attr):
                        bp.set_attribute(attr, str(value))
                actor = session.spawn(bp, self.transform(carla, model))
                hub.add(key + suffix, actor, decode)
                self.sensors[key + suffix] = actor

    @staticmethod
    def transform(carla: Any, model: CameraModel) -> Any:
        pitch, yaw, roll = basis_to_rotation(model.forward, model.right, model.up)
        x, y, z = model.position
        return carla.Transform(carla.Location(x=x, y=y, z=z),
                               carla.Rotation(pitch=pitch, yaw=yaw, roll=roll))

    @property
    def moving(self) -> bool:
        return False

    def move(self, session: Any, t: float) -> None:
        """Place moving sensors for the frame rendered at ``t``."""
        if not self.moving:
            return
        models = self.sensor_models(t)
        for key, actor in self.sensors.items():
            base = key.split("/")[0] if key.split("/")[-1] in ("depth", "instance") else key
            actor.set_transform(self.transform(session.carla, models[base]))

    def frame(self, data: Dict[str, Any], t: float) -> Tuple[np.ndarray, Dict[str, np.ndarray]]:
        """BGR image of this view and its auxiliary buffers from collected sensor data."""
        aux = {k: v for k, v in data.items()
               if k.startswith(self.name + "/") and k.split("/")[-1] in ("depth", "instance")}
        return data[self.name], aux


class PoleRig(Rig):
    name = "pole"

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        heading, pitch, fov = pole_framing(self.site, self.scenario,
                                           self.quality["width"] / self.quality["height"])
        self._model = pinhole(self.name, self.site.pole_world, pitch,
                              self.site.heading_to_yaw(heading), self.quality["width"],
                              self.quality["height"], fov)

    def camera(self, t: float) -> CameraModel:
        return self._model


class StreetRig(Rig):
    name = "street"

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        sx, sy = -3.0, -(self.site.road_half_width + 1.5)
        pos = self.site.to_world(sx, sy, 1.6)
        target = self.site.to_world(12.0, 0.5, 1.0)
        pitch, yaw = look_at(pos, target)
        self._model = pinhole(self.name, pos, pitch, yaw, self.quality["width"],
                              self.quality["height"], 62.0)

    def camera(self, t: float) -> CameraModel:
        return self._model


class BevRig(Rig):
    name = "bev"

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        pts = scenario_points(self.scenario)
        cx, cy = float(np.clip(pts[:, 0].mean(), -5, 25)), 0.0
        height = 60.0
        w, h = self.quality["width"], self.quality["height"]
        span = float(np.clip(np.ptp(pts[:, 0]) + 10.0, 30.0, 70.0))
        fov = math.degrees(2 * math.atan(span / 2.0 / height))
        x, y, z = self.site.to_world(cx, cy, height)
        forward = (0.0, 0.0, -1.0)
        up = tuple(self.site.direction(math.pi / 2.0))
        right = _right_from(forward, up)
        self._model = CameraModel(name=self.name, kind="pinhole", width=w, height=h,
                                  position=(x, y, z), forward=forward, right=right, up=up,
                                  fov_deg=fov)

    def camera(self, t: float) -> CameraModel:
        return self._model


def _right_from(forward: Sequence[float], up: Sequence[float]) -> Tuple[float, float, float]:
    """Right vector completing a CARLA (left-handed) camera basis: right = up x forward."""
    r = np.cross(np.asarray(up, float), np.asarray(forward, float))
    r = r / np.linalg.norm(r)
    return float(r[0]), float(r[1]), float(r[2])


class FollowRig(Rig):
    """Base of the moving views: an exponentially smoothed camera path."""

    def __init__(self, site: Site, scenario: Dict[str, Any], quality: str,
                 ground_truth: bool = False, trajectories: Sequence[Trajectory] = (),
                 fps: int = 30) -> None:
        super().__init__(site, scenario, quality, ground_truth)
        bikes = [t for t in trajectories if t.kind == "bike"]
        self.target = (bikes or list(trajectories))[0] if trajectories else None
        self.fps = fps
        self._cache: Dict[int, CameraModel] = {}

    @property
    def moving(self) -> bool:
        return True


class ChaseRig(FollowRig):
    name = "chase"

    def camera(self, t: float) -> CameraModel:
        k = int(round(t * self.fps))
        if k in self._cache:
            return self._cache[k]
        traj = self.target
        dt = 1.0 / self.fps
        pos = np.zeros(3)
        look = np.zeros(3)
        alpha = 1.0 - math.exp(-dt / 0.25)
        start = traj.start if traj else 0.0
        steps = max(0, int(round((t - start) * self.fps)))
        for i in range(steps + 1):
            tt = start + i * dt
            x, y = traj.position(tt)
            h, _ = traj.heading(tt)
            want = np.array([x - 4.5 * math.cos(h), y - 4.5 * math.sin(h), 2.2])
            tgt = np.array([x + 2.0 * math.cos(h), y + 2.0 * math.sin(h), 1.0])
            if i == 0:
                pos, look = want, tgt
            else:
                pos += alpha * (want - pos)
                look += alpha * (tgt - look)
        wp = self.site.to_world(*pos)
        wl = self.site.to_world(*look)
        pitch, yaw = look_at(wp, wl)
        model = pinhole(self.name, wp, pitch, yaw, self.quality["width"], self.quality["height"],
                        70.0)
        self._cache[k] = model
        return model


class OrbitRig(FollowRig):
    name = "orbit"

    def camera(self, t: float) -> CameraModel:
        centre = (5.0, 0.0)
        duration = max(1.0, max((p["t"] for a in self.scenario["agents"] for p in a["path"]),
                                default=1.0))
        # The arc stays on the near half of the carriageway so the camera never
        # ends up inside kerbside trees or hedges.
        angle = math.radians(-50.0 + 40.0 * min(t / duration, 1.0))
        radius = 14.0
        pos = self.site.to_world(centre[0] + radius * math.cos(angle),
                                 centre[1] + radius * math.sin(angle), 1.7)
        target = self.site.to_world(centre[0] + 3.0, centre[1], 1.0)
        pitch, yaw = look_at(pos, target)
        return pinhole(self.name, pos, pitch, yaw, self.quality["width"], self.quality["height"],
                       45.0)


class FisheyeRig(Rig):
    name = "fisheye"

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        from .fisheye import FisheyeRemap, deployed_fisheye_model

        yaw = self.site.heading_to_yaw(math.pi / 2.0)
        self._model = deployed_fisheye_model(self.site.pole_world, yaw,
                                             scale=self.quality["fisheye_scale"])
        self.remap = FisheyeRemap(self._model, face_size=self.quality["face"])

    def camera(self, t: float) -> CameraModel:
        return self._model

    def sensor_models(self, t: float) -> Dict[str, CameraModel]:
        return {f"{self.name}/{face}": self.remap.face_camera(face) for face in self.remap.faces}

    def spawn(self, session: Any, hub: SensorHub, caps: Any) -> None:
        gt, self.ground_truth = self.ground_truth, False
        super().spawn(session, hub, caps)
        self.ground_truth = gt

    def frame(self, data: Dict[str, Any], t: float) -> Tuple[np.ndarray, Dict[str, np.ndarray]]:
        faces = {face: data[f"{self.name}/{face}"] for face in self.remap.faces}
        return self.remap.remap(faces), {}


RIG_CLASSES = {"pole": PoleRig, "fisheye": FisheyeRig, "chase": ChaseRig, "orbit": OrbitRig,
               "street": StreetRig, "bev": BevRig}


def make_rig(name: str, site: Site, scenario: Dict[str, Any], quality: str, *,
             ground_truth: bool = False, trajectories: Sequence[Trajectory] = (),
             fps: int = 30) -> Rig:
    """Build the rig called ``name`` (one of :data:`RIG_NAMES`)."""
    if name not in RIG_CLASSES:
        raise KeyError(f"unknown rig '{name}', choose from {', '.join(RIG_NAMES)}")
    cls = RIG_CLASSES[name]
    if issubclass(cls, FollowRig):
        return cls(site, scenario, quality, ground_truth, trajectories, fps)
    return cls(site, scenario, quality, ground_truth)


def describe_rigs() -> List[Tuple[str, str]]:
    """Name and first docstring line of every rig, for ``presets``."""
    docs = {
        "pole": "pinhole on the 12 ft pole at the testbench camera position",
        "fisheye": "deployed fisheye lens, cube faces remapped",
        "chase": "smoothed camera behind the first cyclist",
        "orbit": "slow eye-level arc around the conflict point",
        "street": "static eye-level tripod on the kerb",
        "bev": "straight down from 60 m, image up along scenario +y",
    }
    return [(n, docs[n]) for n in RIG_NAMES]

