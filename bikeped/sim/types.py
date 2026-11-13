"""Data types shared by the CARLA runtime and the imaging code.

Nothing in this module imports ``carla``. The runtime (session, actors, rigs,
runner) fills these records from the simulator, and the imaging code
(projection, ground truth, overlay, recorder, export) consumes them, so both
halves can be developed and tested without a server.

Frames of reference
-------------------
Scenario frame
    The frame of the testbench scenarios: metres, right-handed, ``x`` along the
    road (cyclists travel along it), ``y`` across the road (pedestrians cross
    along it), crosswalk centre at ``(5, 0)``.

World frame
    The CARLA world frame: metres, left-handed, ``x`` forward, ``y`` right,
    ``z`` up. Rotations are ``(pitch, yaw, roll)`` in degrees with the Unreal
    convention used by ``carla.Rotation``.

Camera basis
    A camera is described by its position and by three orthonormal world-frame
    vectors ``forward``, ``right`` and ``up``. A world point ``P`` has camera
    coordinates ``(f, r, u) = (forward . d, right . d, up . d)`` with
    ``d = P - position``. Image ``u`` grows with ``r`` and image ``v`` grows
    with ``-u``.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

Vec3 = Tuple[float, float, float]

#: Decision states in order of severity.
STATES: Tuple[str, ...] = ("idle", "safe", "warning", "alert")

#: Agent kinds understood by the decision pipeline.
AGENT_KINDS: Tuple[str, ...] = ("person", "bike")


@dataclass(frozen=True)
class CameraModel:
    """Intrinsics and world pose of one rendered view.

    ``kind`` is ``"pinhole"`` or ``"fisheye"``.

    Pinhole: ``fov_deg`` is the horizontal field of view, and the focal length
    in pixels is ``width / (2 tan(fov / 2))``.

    Fisheye: the optical axis is ``forward`` and the projection is the
    equidistant model with odd polynomial terms,
    ``r = f_px_per_rad * (theta + k1 theta^3 + k2 theta^5)``, where ``theta``
    is the angle between the ray and the optical axis. ``fov_deg`` is the full
    field of view of the image circle and ``circle_radius_px`` its radius.
    ``cx`` and ``cy`` default to the image centre.
    """

    name: str
    kind: str
    width: int
    height: int
    position: Vec3
    forward: Vec3
    right: Vec3
    up: Vec3
    fov_deg: float = 90.0
    cx: Optional[float] = None
    cy: Optional[float] = None
    f_px_per_rad: Optional[float] = None
    k1: float = 0.0
    k2: float = 0.0
    circle_radius_px: Optional[float] = None

    @property
    def principal_point(self) -> Tuple[float, float]:
        cx = self.width / 2.0 if self.cx is None else self.cx
        cy = self.height / 2.0 if self.cy is None else self.cy
        return cx, cy

    @property
    def focal_px(self) -> float:
        """Pinhole focal length in pixels."""
        return self.width / (2.0 * np.tan(np.radians(self.fov_deg) / 2.0))

    def basis(self) -> np.ndarray:
        """3x3 array whose rows are ``forward``, ``right`` and ``up``."""
        return np.array([self.forward, self.right, self.up], dtype=np.float64)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class AgentState:
    """One agent at one frame.

    ``sx``, ``sy`` and ``heading`` are the scripted pose in the scenario frame
    and are what the decision pipeline and the ground-truth danger model see.
    ``location``, ``rotation`` and ``extent`` describe the oriented 3D box of
    the simulated actor in the world frame: ``location`` is the box centre,
    ``extent`` the half sizes along the actor's forward, right and up axes.
    ``track_error_m`` is the ground-plane distance between the scripted and the
    simulated position.
    """

    id: int
    kind: str
    sx: float
    sy: float
    heading: float
    speed: float
    location: Vec3
    rotation: Vec3
    extent: Vec3
    role: str = "scripted"          # "scripted" or "ambient"
    model: str = ""                 # rendering hint from the scenario
    blueprint: str = ""             # CARLA blueprint id actually spawned
    track_error_m: float = 0.0
    instance_id: Optional[int] = None  # id of the actor in "<view>/instance" buffers

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class FrameRecord:
    """Everything known about one simulation step.

    ``images`` maps a view name to a BGR ``uint8`` array. ``aux`` holds
    optional per-view buffers under ``"<view>/depth"`` (float32 metres) and
    ``"<view>/instance"`` (integer ids). Neither is serialised to the manifest.
    """

    index: int
    t: float
    agents: List[AgentState]
    state: str = "idle"
    states: Dict[str, str] = field(default_factory=dict)
    alert_pair: Optional[Tuple[int, int]] = None
    danger: Optional[Dict[str, Any]] = None
    cameras: Dict[str, CameraModel] = field(default_factory=dict)
    images: Dict[str, np.ndarray] = field(default_factory=dict)
    aux: Dict[str, np.ndarray] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "index": self.index,
            "t": round(float(self.t), 4),
            "state": self.state,
            "states": dict(self.states),
            "alert_pair": list(self.alert_pair) if self.alert_pair else None,
            "danger": self.danger,
            "agents": [a.to_dict() for a in self.agents],
        }


@dataclass
class RunInfo:
    """Metadata of one rendered scenario, written at the top of the manifest."""

    scenario_name: str
    scenario_id: str
    suite: str = "conformance"
    fps: int = 30
    duration_s: float = 0.0
    engine: str = "ue4"             # "ue4" or "ue5"
    server_version: str = ""
    client_version: str = ""
    map: str = ""
    crosswalk_id: Optional[int] = None
    weather: str = "clear_noon"
    quality: str = "standard"
    drive: str = "velocity"
    ambient: str = "none"
    seed: int = 0
    views: List[str] = field(default_factory=list)
    site: Dict[str, Any] = field(default_factory=dict)
    pipeline_config: str = "default"
    extra: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)
