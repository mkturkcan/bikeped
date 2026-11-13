"""In-memory stand-in for the CARLA Python API.

The simulator code reaches CARLA through one accessor,
``bikeped.sim.engine.carla_module()``. Tests point that accessor at the object
returned by :func:`make_fake_carla`, which behaves like the ``carla`` module
talking to one server: a world with settings, weather, a blueprint library,
actors with transforms and bounding boxes, and camera sensors that deliver one
synthetic image per ``world.tick()`` carrying the frame id of that tick.

Two configurations matter for the tests. The default one resembles a CARLA
0.9.16 (Unreal Engine 4) server with bicycles and motorcycles. The other
resembles CARLA 0.10.0 (Unreal Engine 5): ``make_fake_carla(version="0.10.0",
two_wheelers=False)`` has four-wheeled vehicles only and fewer camera
attributes.

Rotations follow the Unreal convention of ``carla.Rotation``: degrees,
left-handed world, yaw clockwise seen from above, positive pitch looking up.
Setting an attribute that a blueprint does not have raises ``IndexError`` as the
real client does, so code that forgets ``has_attribute`` fails here too, and
operating on a destroyed actor raises ``RuntimeError``.
"""

from __future__ import annotations

import enum
import fnmatch
import math
import types
from typing import Any, Callable, Dict, Iterable, Iterator, List, Optional, Sequence, Tuple

import numpy as np

#: Corners of crosswalk 8 of Town10HD_Opt as listed in ``crosswalks.json`` (world frame, metres).
CROSSWALK_8: Tuple[Tuple[float, float, float], ...] = (
    (86.4061, 9.1364, 0.0),
    (88.9116, 8.6202, 0.0),
    (88.6616, 32.8860, 0.0),
    (86.1637, 32.6735, 0.0),
)

#: Depth written by the fake depth camera at every pixel, in metres.
FAKE_DEPTH_M = 10.0

_WALKER_EXTENT = (0.18, 0.18, 0.93)
_TWO_WHEELER_EXTENT = (0.02, 0.02, 0.02)
_CAR_EXTENT = (2.4, 1.0, 0.75)


def _basis(pitch: float, yaw: float, roll: float) -> np.ndarray:
    """Rotation matrix whose columns are the forward, right and up vectors (degrees in)."""
    p, y, r = math.radians(pitch), math.radians(yaw), math.radians(roll)
    cp, sp = math.cos(p), math.sin(p)
    cy, sy = math.cos(y), math.sin(y)
    cr, sr = math.cos(r), math.sin(r)
    forward = (cp * cy, cp * sy, sp)
    right = (cy * sp * sr - sy * cr, sy * sp * sr + cy * cr, -cp * sr)
    up = (-cy * sp * cr - sy * sr, -sy * sp * cr + cy * sr, cp * cr)
    return np.array([forward, right, up], dtype=np.float64).T


def _angles(matrix: np.ndarray) -> Tuple[float, float, float]:
    """Inverse of :func:`_basis`: ``(pitch, yaw, roll)`` in degrees."""
    forward, right, up = matrix[:, 0], matrix[:, 1], matrix[:, 2]
    pitch = math.degrees(math.asin(float(np.clip(forward[2], -1.0, 1.0))))
    if abs(forward[2]) > 1.0 - 1e-9:
        return pitch, math.degrees(math.atan2(-right[0], right[1])), 0.0
    yaw = math.degrees(math.atan2(forward[1], forward[0]))
    roll = math.degrees(math.atan2(-right[2], up[2]))
    return pitch, yaw, roll


class Vector3D:
    """Three floats with the arithmetic of ``carla.Vector3D``."""

    def __init__(self, x: float = 0.0, y: float = 0.0, z: float = 0.0) -> None:
        self.x = float(x)
        self.y = float(y)
        self.z = float(z)

    def _like(self, x: float, y: float, z: float) -> "Vector3D":
        return type(self)(x, y, z)

    def __add__(self, other: "Vector3D") -> "Vector3D":
        return self._like(self.x + other.x, self.y + other.y, self.z + other.z)

    def __sub__(self, other: "Vector3D") -> "Vector3D":
        return self._like(self.x - other.x, self.y - other.y, self.z - other.z)

    def __mul__(self, other: Any) -> "Vector3D":
        if isinstance(other, Vector3D):
            return self._like(self.x * other.x, self.y * other.y, self.z * other.z)
        return self._like(self.x * other, self.y * other, self.z * other)

    __rmul__ = __mul__

    def __truediv__(self, other: float) -> "Vector3D":
        return self._like(self.x / other, self.y / other, self.z / other)

    def __neg__(self) -> "Vector3D":
        return self._like(-self.x, -self.y, -self.z)

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Vector3D):
            return NotImplemented
        return (self.x, self.y, self.z) == (other.x, other.y, other.z)

    def __repr__(self) -> str:
        return f"{type(self).__name__}(x={self.x:.4f}, y={self.y:.4f}, z={self.z:.4f})"

    def _array(self) -> np.ndarray:
        return np.array([self.x, self.y, self.z], dtype=np.float64)

    def length(self) -> float:
        return math.sqrt(self.squared_length())

    def squared_length(self) -> float:
        return self.x * self.x + self.y * self.y + self.z * self.z

    def make_unit_vector(self) -> "Vector3D":
        norm = self.length()
        return self._like(self.x / norm, self.y / norm, self.z / norm)

    def dot(self, other: "Vector3D") -> float:
        return self.x * other.x + self.y * other.y + self.z * other.z

    def cross(self, other: "Vector3D") -> "Vector3D":
        return self._like(
            self.y * other.z - self.z * other.y,
            self.z * other.x - self.x * other.z,
            self.x * other.y - self.y * other.x,
        )

    def distance(self, other: "Vector3D") -> float:
        return (self - other).length()

    def distance_2d(self, other: "Vector3D") -> float:
        return math.hypot(self.x - other.x, self.y - other.y)


class Location(Vector3D):
    """A point of the world frame in metres."""


class Rotation:
    """Unreal rotation in degrees."""

    def __init__(self, pitch: float = 0.0, yaw: float = 0.0, roll: float = 0.0) -> None:
        self.pitch = float(pitch)
        self.yaw = float(yaw)
        self.roll = float(roll)

    def _matrix(self) -> np.ndarray:
        return _basis(self.pitch, self.yaw, self.roll)

    def get_forward_vector(self) -> Vector3D:
        return Vector3D(*self._matrix()[:, 0])

    def get_right_vector(self) -> Vector3D:
        return Vector3D(*self._matrix()[:, 1])

    def get_up_vector(self) -> Vector3D:
        return Vector3D(*self._matrix()[:, 2])

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Rotation):
            return NotImplemented
        return (self.pitch, self.yaw, self.roll) == (other.pitch, other.yaw, other.roll)

    def __repr__(self) -> str:
        return f"Rotation(pitch={self.pitch:.4f}, yaw={self.yaw:.4f}, roll={self.roll:.4f})"


class Transform:
    """Location plus rotation. The constructor copies both, as the real type does."""

    def __init__(
        self, location: Optional[Vector3D] = None, rotation: Optional[Rotation] = None
    ) -> None:
        location = location if location is not None else Location()
        rotation = rotation if rotation is not None else Rotation()
        self.location = Location(location.x, location.y, location.z)
        self.rotation = Rotation(rotation.pitch, rotation.yaw, rotation.roll)

    def transform(self, point: Vector3D) -> Location:
        """Map a point of the local frame (forward, right, up) to the world frame."""
        world = self.location._array() + self.rotation._matrix() @ point._array()
        return Location(*world)

    def transform_vector(self, vector: Vector3D) -> Vector3D:
        return Vector3D(*(self.rotation._matrix() @ vector._array()))

    def inverse_transform(self, point: Vector3D) -> Location:
        local = self.rotation._matrix().T @ (point._array() - self.location._array())
        return Location(*local)

    def get_forward_vector(self) -> Vector3D:
        return self.rotation.get_forward_vector()

    def get_right_vector(self) -> Vector3D:
        return self.rotation.get_right_vector()

    def get_up_vector(self) -> Vector3D:
        return self.rotation.get_up_vector()

    def _matrix4(self) -> np.ndarray:
        matrix = np.eye(4)
        matrix[:3, :3] = self.rotation._matrix()
        matrix[:3, 3] = self.location._array()
        return matrix

    def get_matrix(self) -> List[List[float]]:
        return self._matrix4().tolist()

    def get_inverse_matrix(self) -> List[List[float]]:
        return np.linalg.inv(self._matrix4()).tolist()

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Transform):
            return NotImplemented
        return self.location == other.location and self.rotation == other.rotation

    def __repr__(self) -> str:
        return f"Transform({self.location!r}, {self.rotation!r})"


def _compose(parent: Transform, child: Transform) -> Transform:
    """World transform of ``child`` expressed relative to ``parent``."""
    matrix = parent.rotation._matrix() @ child.rotation._matrix()
    return Transform(parent.transform(child.location), Rotation(*_angles(matrix)))


class BoundingBox:
    """Oriented box: centre relative to the actor origin and half sizes in metres."""

    def __init__(
        self,
        location: Optional[Vector3D] = None,
        extent: Optional[Vector3D] = None,
        rotation: Optional[Rotation] = None,
    ) -> None:
        self.location = Location(*(location._array() if location is not None else (0, 0, 0)))
        self.extent = Vector3D(*(extent._array() if extent is not None else (0, 0, 0)))
        self.rotation = rotation if rotation is not None else Rotation()

    def get_local_vertices(self) -> List[Location]:
        e = self.extent
        return [
            Location(self.location.x + sx * e.x, self.location.y + sy * e.y,
                     self.location.z + sz * e.z)
            for sx in (-1, 1) for sy in (-1, 1) for sz in (-1, 1)
        ]

    def get_world_vertices(self, transform: Transform) -> List[Location]:
        return [transform.transform(v) for v in self.get_local_vertices()]


class WorldSettings:
    """Simulation settings with the defaults of a freshly loaded map."""

    def __init__(
        self,
        synchronous_mode: bool = False,
        no_rendering_mode: bool = False,
        fixed_delta_seconds: Optional[float] = None,
        substepping: bool = True,
        max_substep_delta_time: float = 0.01,
        max_substeps: int = 10,
        max_culling_distance: float = 0.0,
        deterministic_ragdolls: bool = True,
        tile_stream_distance: float = 3000.0,
        actor_active_distance: float = 2000.0,
        spectator_as_ego: bool = True,
    ) -> None:
        self.synchronous_mode = synchronous_mode
        self.no_rendering_mode = no_rendering_mode
        self.fixed_delta_seconds = fixed_delta_seconds
        self.substepping = substepping
        self.max_substep_delta_time = max_substep_delta_time
        self.max_substeps = max_substeps
        self.max_culling_distance = max_culling_distance
        self.deterministic_ragdolls = deterministic_ragdolls
        self.tile_stream_distance = tile_stream_distance
        self.actor_active_distance = actor_active_distance
        self.spectator_as_ego = spectator_as_ego

    def _copy(self) -> "WorldSettings":
        return WorldSettings(**self.__dict__)

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, WorldSettings):
            return NotImplemented
        return self.__dict__ == other.__dict__

    def __repr__(self) -> str:
        return f"WorldSettings({self.__dict__})"


class WeatherParameters:
    """Plain attribute bag with the fields of ``carla.WeatherParameters``."""

    FIELDS: Tuple[str, ...] = (
        "cloudiness", "precipitation", "precipitation_deposits", "wind_intensity",
        "sun_azimuth_angle", "sun_altitude_angle", "fog_density", "fog_distance",
        "fog_falloff", "wetness", "scattering_intensity", "mie_scattering_scale",
        "rayleigh_scattering_scale", "dust_storm",
    )
    _DEFAULTS = {
        "sun_altitude_angle": 45.0, "fog_falloff": 0.2, "scattering_intensity": 1.0,
        "mie_scattering_scale": 0.03, "rayleigh_scattering_scale": 0.0331,
    }

    def __init__(self, **values: float) -> None:
        for name in self.FIELDS:
            setattr(self, name, float(self._DEFAULTS.get(name, 0.0)))
        for name, value in values.items():
            if name not in self.FIELDS:
                raise TypeError(f"WeatherParameters has no field {name!r}")
            setattr(self, name, float(value))

    def _copy(self) -> "WeatherParameters":
        return WeatherParameters(**{name: getattr(self, name) for name in self.FIELDS})

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, WeatherParameters):
            return NotImplemented
        return all(getattr(self, n) == getattr(other, n) for n in self.FIELDS)


WeatherParameters.ClearNoon = WeatherParameters(cloudiness=5.0, sun_altitude_angle=45.0)
WeatherParameters.CloudyNoon = WeatherParameters(cloudiness=60.0, sun_altitude_angle=45.0)
WeatherParameters.ClearSunset = WeatherParameters(cloudiness=5.0, sun_altitude_angle=15.0)
WeatherParameters.ClearNight = WeatherParameters(cloudiness=5.0, sun_altitude_angle=-90.0)


class WalkerControl:
    """Direction (world frame), speed in m/s and jump flag of a walker."""

    def __init__(
        self, direction: Optional[Vector3D] = None, speed: float = 0.0, jump: bool = False
    ) -> None:
        self.direction = direction if direction is not None else Vector3D(1.0, 0.0, 0.0)
        self.speed = float(speed)
        self.jump = bool(jump)


class VehicleControl:
    """Driver inputs of a vehicle."""

    def __init__(
        self,
        throttle: float = 0.0,
        steer: float = 0.0,
        brake: float = 0.0,
        hand_brake: bool = False,
        reverse: bool = False,
        manual_gear_shift: bool = False,
        gear: int = 0,
    ) -> None:
        self.throttle = float(throttle)
        self.steer = float(steer)
        self.brake = float(brake)
        self.hand_brake = bool(hand_brake)
        self.reverse = bool(reverse)
        self.manual_gear_shift = bool(manual_gear_shift)
        self.gear = int(gear)


class AttachmentType(enum.Enum):
    Rigid = 0
    SpringArm = 1
    SpringArmGhost = 2


class ColorConverter(enum.Enum):
    Raw = 0
    Depth = 1
    LogarithmicDepth = 2
    CityScapesPalette = 3


class VehicleLightState(enum.IntFlag):
    NONE = 0
    Position = 0x1
    LowBeam = 0x2
    HighBeam = 0x4
    Brake = 0x8
    RightBlinker = 0x10
    LeftBlinker = 0x20
    Reverse = 0x40
    Fog = 0x80
    Interior = 0x100
    Special1 = 0x200
    Special2 = 0x400
    All = 0xFFFFFFFF


class LightGroup(enum.Enum):
    NONE = 0
    Vehicle = 1
    Street = 2
    Building = 3
    Other = 4


class LaneType(enum.IntFlag):
    NONE = 0x1
    Driving = 0x2
    Shoulder = 0x8
    Sidewalk = 0x20
    Any = 0xFFFFFFFE


class MapLayer(enum.IntFlag):
    NONE = 0
    Buildings = 0x1
    Decals = 0x2
    Foliage = 0x4
    Ground = 0x8
    ParkedVehicles = 0x10
    Particles = 0x20
    Props = 0x40
    StreetLights = 0x80
    Walls = 0x100
    All = 0xFFFF


class ActorAttribute:
    """One blueprint attribute. Values are stored as strings, as on the wire."""

    def __init__(
        self,
        attribute_id: str,
        value: Any,
        recommended_values: Sequence[Any] = (),
        is_modifiable: bool = True,
    ) -> None:
        self.id = attribute_id
        self._value = str(value)
        self.recommended_values = [str(v) for v in recommended_values]
        self.is_modifiable = is_modifiable

    def as_str(self) -> str:
        return self._value

    def as_int(self) -> int:
        return int(float(self._value))

    def as_float(self) -> float:
        return float(self._value)

    def as_bool(self) -> bool:
        return self._value.strip().lower() == "true"

    def __str__(self) -> str:
        return self._value

    def __int__(self) -> int:
        return self.as_int()

    def __float__(self) -> float:
        return self.as_float()

    def __bool__(self) -> bool:
        return self.as_bool()

    def __eq__(self, other: object) -> bool:
        if isinstance(other, ActorAttribute):
            return self._value == other._value
        if isinstance(other, bool):
            return self.as_bool() == other
        if isinstance(other, (int, float)):
            return self.as_float() == other
        return self._value == str(other)

    def __hash__(self) -> int:
        return hash((self.id, self._value))

    def _copy(self) -> "ActorAttribute":
        return ActorAttribute(self.id, self._value, self.recommended_values, self.is_modifiable)


class ActorBlueprint:
    """Blueprint with string attributes."""

    def __init__(self, blueprint_id: str, attributes: Iterable[ActorAttribute]) -> None:
        self.id = blueprint_id
        self.tags = blueprint_id.split(".")
        self._attributes: Dict[str, ActorAttribute] = {a.id: a for a in attributes}

    def has_attribute(self, name: str) -> bool:
        return name in self._attributes

    def get_attribute(self, name: str) -> ActorAttribute:
        if name not in self._attributes:
            raise IndexError(f"attribute '{name}' not found in blueprint '{self.id}'")
        return self._attributes[name]

    def set_attribute(self, name: str, value: Any) -> None:
        attribute = self.get_attribute(name)
        if not attribute.is_modifiable:
            raise RuntimeError(f"attribute '{name}' of blueprint '{self.id}' is not modifiable")
        if isinstance(value, bool):
            value = "true" if value else "false"
        attribute._value = str(value)

    def has_tag(self, tag: str) -> bool:
        return tag in self.tags

    def match_tags(self, pattern: str) -> bool:
        return fnmatch.fnmatchcase(self.id, pattern) or any(
            fnmatch.fnmatchcase(tag, pattern) for tag in self.tags
        )

    def __iter__(self) -> Iterator[ActorAttribute]:
        return iter(self._attributes.values())

    def __len__(self) -> int:
        return len(self._attributes)

    def __repr__(self) -> str:
        return f"ActorBlueprint(id={self.id})"

    def _copy(self) -> "ActorBlueprint":
        return ActorBlueprint(self.id, [a._copy() for a in self._attributes.values()])


class BlueprintLibrary:
    """Blueprints of the server. Lookups return copies, so edits never leak back."""

    def __init__(self, blueprints: Iterable[ActorBlueprint]) -> None:
        self._blueprints = list(blueprints)

    def filter(self, wildcard_pattern: str) -> "BlueprintLibrary":
        return BlueprintLibrary(
            bp._copy() for bp in self._blueprints if bp.match_tags(wildcard_pattern)
        )

    def filter_by_attribute(self, name: str, value: Any) -> "BlueprintLibrary":
        return BlueprintLibrary(
            bp._copy() for bp in self._blueprints
            if bp.has_attribute(name) and bp.get_attribute(name) == value
        )

    def find(self, blueprint_id: str) -> ActorBlueprint:
        for bp in self._blueprints:
            if bp.id == blueprint_id:
                return bp._copy()
        raise IndexError(f"blueprint '{blueprint_id}' not found")

    def __len__(self) -> int:
        return len(self._blueprints)

    def __iter__(self) -> Iterator[ActorBlueprint]:
        return iter([bp._copy() for bp in self._blueprints])

    def __getitem__(self, index: int) -> ActorBlueprint:
        return self._blueprints[index]._copy()


class Timestamp:
    """Simulation time of one tick."""

    def __init__(self, frame: int, elapsed_seconds: float, delta_seconds: float) -> None:
        self.frame = frame
        self.frame_count = frame
        self.elapsed_seconds = elapsed_seconds
        self.delta_seconds = delta_seconds
        self.platform_timestamp = elapsed_seconds


class Image:
    """Camera output: ``raw_data`` is a BGRA ``uint8`` buffer of ``height x width x 4``."""

    def __init__(
        self,
        frame: int,
        timestamp: float,
        transform: Transform,
        width: int,
        height: int,
        fov: float,
        raw_data: bytes,
    ) -> None:
        self.frame = frame
        self.frame_number = frame
        self.timestamp = timestamp
        self.transform = transform
        self.width = width
        self.height = height
        self.fov = fov
        self.raw_data = raw_data
        self.conversions: List[Any] = []
        self.saved_to: List[str] = []

    def convert(self, color_converter: Any) -> None:
        """Record the request. The buffer is left as delivered."""
        self.conversions.append(color_converter)

    def save_to_disk(self, path: str, color_converter: Any = None) -> str:
        """Record the path without writing a file."""
        self.saved_to.append(path)
        return path

    def __len__(self) -> int:
        return self.width * self.height


class Actor:
    """Base actor: pose, physics flags and the controls it received."""

    def __init__(
        self,
        world: "World",
        actor_id: int,
        blueprint: ActorBlueprint,
        transform: Transform,
        parent: Optional["Actor"] = None,
    ) -> None:
        self._world = world
        self.id = actor_id
        self.type_id = blueprint.id
        self.attributes: Dict[str, str] = {a.id: str(a) for a in blueprint}
        self.parent = parent
        self.is_alive = True
        self.is_active = True
        self.semantic_tags: List[int] = []
        self._transform = Transform(transform.location, transform.rotation)
        self._velocity = Vector3D()
        self._angular_velocity = Vector3D()
        self.simulate_physics = True
        self.gravity = True
        self.autopilot = False
        self.light_state = VehicleLightState.NONE
        self.controls: List[Any] = []
        self.transform_updates = 0
        self.bounding_box = BoundingBox(Location(), Vector3D(*_extent_for(blueprint)))
        if self.type_id.startswith("vehicle.") and not _is_two_wheeler(blueprint):
            self.bounding_box.location = Location(0.0, 0.0, _CAR_EXTENT[2])

    def _require_alive(self) -> None:
        if not self.is_alive:
            raise RuntimeError(
                f"trying to operate on a destroyed actor ({self.type_id}, id {self.id})"
            )

    def get_world(self) -> "World":
        return self._world

    def get_transform(self) -> Transform:
        if self.parent is not None:
            return _compose(self.parent.get_transform(), self._transform)
        return Transform(self._transform.location, self._transform.rotation)

    def set_transform(self, transform: Transform) -> None:
        self._require_alive()
        self._transform = Transform(transform.location, transform.rotation)
        self.transform_updates += 1

    def get_location(self) -> Location:
        return self.get_transform().location

    def set_location(self, location: Vector3D) -> None:
        self._require_alive()
        self._transform = Transform(location, self._transform.rotation)
        self.transform_updates += 1

    def get_velocity(self) -> Vector3D:
        return Vector3D(self._velocity.x, self._velocity.y, self._velocity.z)

    def get_angular_velocity(self) -> Vector3D:
        w = self._angular_velocity
        return Vector3D(w.x, w.y, w.z)

    def get_acceleration(self) -> Vector3D:
        return Vector3D()

    def set_simulate_physics(self, enabled: bool = True) -> None:
        self._require_alive()
        self.simulate_physics = bool(enabled)

    def set_enable_gravity(self, enabled: bool) -> None:
        self._require_alive()
        self.gravity = bool(enabled)

    def set_collisions(self, enabled: bool) -> None:
        self._require_alive()
        self.collisions = bool(enabled)

    def apply_control(self, control: Any) -> None:
        self._require_alive()
        self.controls.append(control)
        if isinstance(control, WalkerControl):
            direction = control.direction
            norm = direction.length()
            scale = control.speed / norm if norm > 1e-9 else 0.0
            self._velocity = Vector3D(direction.x * scale, direction.y * scale, 0.0)

    def get_control(self) -> Any:
        return self.controls[-1] if self.controls else None

    def set_target_velocity(self, velocity: Vector3D) -> None:
        self._require_alive()
        self._velocity = Vector3D(velocity.x, velocity.y, velocity.z)

    def set_target_angular_velocity(self, angular_velocity: Vector3D) -> None:
        self._require_alive()
        w = angular_velocity
        self._angular_velocity = Vector3D(w.x, w.y, w.z)

    def enable_constant_velocity(self, velocity: Vector3D) -> None:
        self.set_target_velocity(velocity)

    def disable_constant_velocity(self) -> None:
        self._require_alive()
        self._velocity = Vector3D()

    def set_autopilot(self, enabled: bool = True, port: int = 8000) -> None:
        self._require_alive()
        self.autopilot = bool(enabled)

    def set_light_state(self, state: Any) -> None:
        self._require_alive()
        self.light_state = state

    def get_light_state(self) -> Any:
        return self.light_state

    def _step(self, dt: float) -> None:
        """Advance the pose by the commanded velocity (yaw rate in degrees per second)."""
        if self.parent is not None or not self.simulate_physics:
            return
        loc, rot = self._transform.location, self._transform.rotation
        self._transform = Transform(
            Location(loc.x + self._velocity.x * dt, loc.y + self._velocity.y * dt, loc.z),
            Rotation(rot.pitch, rot.yaw + self._angular_velocity.z * dt, rot.roll),
        )

    def destroy(self) -> bool:
        if not self.is_alive:
            return False
        self.is_alive = False
        self.is_active = False
        self._world._forget(self)
        return True

    def __repr__(self) -> str:
        return f"Actor(id={self.id}, type={self.type_id})"


class Sensor(Actor):
    """Camera sensor that hands a synthetic image to its callback on every tick."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self._callback: Optional[Callable[[Image], None]] = None
        self.width = int(float(self.attributes.get("image_size_x", "800")))
        self.height = int(float(self.attributes.get("image_size_y", "600")))
        self.fov = float(self.attributes.get("fov", "90.0"))
        self.delivered = 0

    @property
    def is_listening(self) -> bool:
        return self._callback is not None

    def listen(self, callback: Callable[[Image], None]) -> None:
        self._require_alive()
        self._callback = callback

    def stop(self) -> None:
        self._callback = None

    def destroy(self) -> bool:
        self._callback = None
        return super().destroy()

    def _pixels(self, frame: int) -> np.ndarray:
        buffer = np.empty((self.height, self.width, 4), dtype=np.uint8)
        buffer[..., 3] = 255
        if self.type_id == "sensor.camera.depth":
            value = int(FAKE_DEPTH_M / 1000.0 * (256 ** 3 - 1))
            buffer[..., 2] = value % 256
            buffer[..., 1] = (value // 256) % 256
            buffer[..., 0] = value // 65536
        elif self.type_id == "sensor.camera.rgb":
            buffer[..., 0] = np.linspace(0, 255, self.width, dtype=np.uint8)[None, :]
            buffer[..., 1] = np.linspace(0, 255, self.height, dtype=np.uint8)[:, None]
            buffer[..., 2] = (frame * 7) % 256
        else:
            buffer[..., :3] = 0
        return buffer

    def _emit(self, frame: int, timestamp: Timestamp) -> None:
        if self._callback is None:
            return
        image = Image(
            frame, timestamp.elapsed_seconds, self.get_transform(),
            self.width, self.height, self.fov, self._pixels(frame).tobytes(),
        )
        self.delivered += 1
        self._callback(image)


class WalkerAIController(Actor):
    """Controller actor of an AI walker. It records what it was asked to do."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.started = False
        self.destination: Optional[Location] = None
        self.max_speed: Optional[float] = None

    def start(self) -> None:
        self._require_alive()
        self.started = True

    def stop(self) -> None:
        self.started = False

    def go_to_location(self, destination: Vector3D) -> None:
        self._require_alive()
        self.destination = Location(destination.x, destination.y, destination.z)

    def set_max_speed(self, speed: float = 1.4) -> None:
        self._require_alive()
        self.max_speed = float(speed)


class ActorList(list):
    """List of actors with the ``filter`` and ``find`` helpers of ``carla.ActorList``."""

    def filter(self, wildcard_pattern: str) -> "ActorList":
        return ActorList(a for a in self if fnmatch.fnmatchcase(a.type_id, wildcard_pattern))

    def find(self, actor_id: int) -> Optional[Actor]:
        for actor in self:
            if actor.id == actor_id:
                return actor
        return None


class Waypoint:
    """Lane sample of the fake map: a straight lane through the given point."""

    def __init__(self, location: Vector3D, yaw: float = 0.0) -> None:
        self.transform = Transform(Location(location.x, location.y, 0.0), Rotation(yaw=yaw))
        self.lane_width = 3.5
        self.road_id = 1
        self.lane_id = 1
        self.section_id = 0
        self.s = 0.0
        self.is_junction = False
        self.is_intersection = False
        self.lane_type = LaneType.Driving

    def _ahead(self, distance: float) -> "Waypoint":
        forward = self.transform.get_forward_vector()
        return Waypoint(self.transform.location + forward * distance, self.transform.rotation.yaw)

    def next(self, distance: float) -> List["Waypoint"]:
        return [self._ahead(distance)]

    def previous(self, distance: float) -> List["Waypoint"]:
        return [self._ahead(-distance)]

    def get_left_lane(self) -> None:
        return None

    def get_right_lane(self) -> None:
        return None


class Map:
    """Map with crosswalk polygons and a ring of spawn points around the first crosswalk."""

    def __init__(self, name: str, crosswalks: Sequence[Sequence[Sequence[float]]]) -> None:
        self.name = f"Carla/Maps/{name}"
        self._crosswalks = [[tuple(float(v) for v in p) for p in poly] for poly in crosswalks]

    def get_crosswalks(self) -> List[Location]:
        """Flat list of polygon points. Each polygon repeats its first point at the end."""
        points: List[Location] = []
        for polygon in self._crosswalks:
            for corner in list(polygon) + [polygon[0]]:
                points.append(Location(*corner))
        return points

    def _centre(self) -> np.ndarray:
        if not self._crosswalks:
            return np.zeros(3)
        return np.mean(np.array(self._crosswalks[0], dtype=np.float64), axis=0)

    def get_spawn_points(self) -> List[Transform]:
        centre = self._centre()
        points = []
        for ring, radius in enumerate((12.0, 45.0, 90.0)):
            for k in range(8):
                angle = 2.0 * math.pi * (k + 0.5 * ring) / 8.0
                points.append(Transform(
                    Location(centre[0] + radius * math.cos(angle),
                             centre[1] + radius * math.sin(angle), 0.6),
                    Rotation(yaw=math.degrees(angle) + 90.0),
                ))
        return points

    def get_waypoint(
        self, location: Vector3D, project_to_road: bool = True, lane_type: Any = LaneType.Driving
    ) -> Waypoint:
        return Waypoint(location)

    def generate_waypoints(self, distance: float) -> List[Waypoint]:
        return [Waypoint(t.location, t.rotation.yaw) for t in self.get_spawn_points()]

    def get_topology(self) -> List[Tuple[Waypoint, Waypoint]]:
        waypoints = self.generate_waypoints(2.0)
        return list(zip(waypoints[:-1], waypoints[1:]))


class Light:
    """One street or building light."""

    def __init__(self, light_id: int, location: Location, light_group: LightGroup) -> None:
        self.id = light_id
        self.location = location
        self.light_group = light_group
        self.is_on = False
        self.intensity = 0.0

    def turn_on(self) -> None:
        self.is_on = True

    def turn_off(self) -> None:
        self.is_on = False

    def set_intensity(self, intensity: float) -> None:
        self.intensity = float(intensity)


class LightManager:
    """Light manager with a handful of street and building lights."""

    def __init__(self) -> None:
        self.lights = [
            Light(i, Location(10.0 * i, 0.0, 6.0),
                  LightGroup.Street if i < 6 else LightGroup.Building)
            for i in range(8)
        ]
        self.day_night_cycle = True

    def get_all_lights(self, light_group: LightGroup = LightGroup.NONE) -> List[Light]:
        if light_group == LightGroup.NONE:
            return list(self.lights)
        return [light for light in self.lights if light.light_group == light_group]

    def turn_on(self, lights: Iterable[Light]) -> None:
        for light in lights:
            light.is_on = True

    def turn_off(self, lights: Iterable[Light]) -> None:
        for light in lights:
            light.is_on = False

    def is_active(self, light: Light) -> bool:
        return light.is_on

    def set_intensity(self, lights: Iterable[Light], intensity: float) -> None:
        for light in lights:
            light.intensity = float(intensity)

    def set_day_night_cycle(self, active: bool) -> None:
        self.day_night_cycle = bool(active)


class TrafficManager:
    """Traffic Manager stub. Every call is appended to ``calls`` as ``(name, args)``."""

    def __init__(self, port: int) -> None:
        self._port = port
        self.synchronous_mode = False
        self.seed: Optional[int] = None
        self.calls: List[Tuple[str, Tuple[Any, ...]]] = []

    def get_port(self) -> int:
        return self._port

    def set_synchronous_mode(self, enabled: bool = True) -> None:
        self.calls.append(("set_synchronous_mode", (enabled,)))
        self.synchronous_mode = bool(enabled)

    def set_random_device_seed(self, seed: int) -> None:
        self.calls.append(("set_random_device_seed", (seed,)))
        self.seed = int(seed)

    def __getattr__(self, name: str) -> Callable[..., None]:
        if name.startswith("_"):
            raise AttributeError(name)

        def record(*args: Any, **kwargs: Any) -> None:
            self.calls.append((name, args))

        return record


class LabelledPoint:
    """Result of a ground projection."""

    def __init__(self, location: Location, label: int = 0) -> None:
        self.location = location
        self.label = label


class ActorSnapshot:
    """Pose and velocity of one actor inside a world snapshot."""

    def __init__(self, actor: Actor) -> None:
        self.id = actor.id
        self._transform = actor.get_transform()
        self._velocity = actor.get_velocity()

    def get_transform(self) -> Transform:
        return self._transform

    def get_velocity(self) -> Vector3D:
        return self._velocity


class WorldSnapshot:
    """State of the world after one tick."""

    def __init__(self, world: "World") -> None:
        self.frame = world.frame
        self.id = world.id
        self.timestamp = Timestamp(world.frame, world.elapsed_seconds, world.delta_seconds)
        self._actors = {a.id: ActorSnapshot(a) for a in world.actors.values()}

    def find(self, actor_id: int) -> Optional[ActorSnapshot]:
        return self._actors.get(actor_id)

    def has_actor(self, actor_id: int) -> bool:
        return actor_id in self._actors

    def __iter__(self) -> Iterator[ActorSnapshot]:
        return iter(self._actors.values())

    def __len__(self) -> int:
        return len(self._actors)


class World:
    """World of the fake server.

    ``actors`` maps the id of every live spawned actor to the actor,
    ``spawned_total`` and ``destroyed_total`` count spawns and destructions,
    ``settings_history`` keeps every settings object applied and
    ``weather_history`` every weather set. Setting ``spawn_failures`` to ``n``
    makes the next ``n`` spawn attempts fail the way a collision does.
    """

    def __init__(self, server: "FakeServer", map_name: str) -> None:
        self._server = server
        self.id = server.next_world_id()
        self._map = Map(map_name, server.crosswalks)
        self._settings = WorldSettings()
        self.settings_history: List[WorldSettings] = []
        self._weather = WeatherParameters()
        self.weather_history: List[WeatherParameters] = []
        self.actors: Dict[int, Actor] = {}
        self.spawned_total = 0
        self.destroyed_total = 0
        self.spawn_failures = 0
        self.frame = 0
        self.tick_count = 0
        self.elapsed_seconds = 0.0
        self.delta_seconds = 0.05
        self.ground_z = 0.0
        self.pedestrians_seed: Optional[int] = None
        self.pedestrians_cross_factor: Optional[float] = None
        self.unloaded_layers: List[Any] = []
        self._next_actor_id = 100
        self._rng = np.random.default_rng(0)
        self._tick_callbacks: Dict[int, Callable[[WorldSnapshot], None]] = {}
        self._spectator = Actor(self, 1, ActorBlueprint("spectator", []), Transform())
        if server.with_ground_projection:
            self.ground_projection = self._ground_projection
            self.project_point = self._project_point
        if server.with_light_manager:
            self._light_manager = LightManager()
            self.get_lightmanager = self._get_lightmanager

    def get_settings(self) -> WorldSettings:
        return self._settings._copy()

    def apply_settings(self, settings: WorldSettings, seconds: float = 0.0) -> int:
        self._settings = settings._copy()
        self.settings_history.append(settings._copy())
        return self.frame

    def get_map(self) -> Map:
        return self._map

    def get_blueprint_library(self) -> BlueprintLibrary:
        return self._server.library

    def get_spectator(self) -> Actor:
        return self._spectator

    def get_weather(self) -> WeatherParameters:
        return self._weather._copy()

    def set_weather(self, weather: WeatherParameters) -> None:
        self._weather = weather._copy()
        self.weather_history.append(weather._copy())

    def _make_actor(
        self, blueprint: ActorBlueprint, transform: Transform, parent: Optional[Actor]
    ) -> Actor:
        if blueprint.id.startswith("sensor."):
            cls: type = Sensor
        elif blueprint.id.startswith("controller.ai.walker"):
            cls = WalkerAIController
        else:
            cls = Actor
        actor = cls(self, self._next_actor_id, blueprint, transform, parent)
        self._next_actor_id += 1
        self.actors[actor.id] = actor
        self.spawned_total += 1
        return actor

    def try_spawn_actor(
        self,
        blueprint: ActorBlueprint,
        transform: Transform,
        attach_to: Optional[Actor] = None,
        attachment_type: Any = AttachmentType.Rigid,
        **_: Any,
    ) -> Optional[Actor]:
        if self.spawn_failures > 0:
            self.spawn_failures -= 1
            return None
        return self._make_actor(blueprint, transform, attach_to)

    def spawn_actor(
        self,
        blueprint: ActorBlueprint,
        transform: Transform,
        attach_to: Optional[Actor] = None,
        attachment_type: Any = AttachmentType.Rigid,
        **_: Any,
    ) -> Actor:
        actor = self.try_spawn_actor(blueprint, transform, attach_to, attachment_type)
        if actor is None:
            raise RuntimeError("Spawn failed because of collision at spawn position")
        return actor

    def _forget(self, actor: Actor) -> None:
        if self.actors.pop(actor.id, None) is not None:
            self.destroyed_total += 1

    def get_actors(self, actor_ids: Optional[Iterable[int]] = None) -> ActorList:
        actors = [self._spectator] + list(self.actors.values())
        if actor_ids is not None:
            wanted = set(actor_ids)
            actors = [a for a in actors if a.id in wanted]
        return ActorList(actors)

    def get_actor(self, actor_id: int) -> Optional[Actor]:
        if actor_id == self._spectator.id:
            return self._spectator
        return self.actors.get(actor_id)

    def tick(self, seconds: float = 10.0) -> int:
        """Advance one step and deliver one image per listening sensor before returning."""
        self.delta_seconds = self._settings.fixed_delta_seconds or 0.05
        self.frame += 1
        self.tick_count += 1
        self.elapsed_seconds += self.delta_seconds
        for actor in list(self.actors.values()):
            actor._step(self.delta_seconds)
        timestamp = Timestamp(self.frame, self.elapsed_seconds, self.delta_seconds)
        for actor in list(self.actors.values()):
            if isinstance(actor, Sensor):
                actor._emit(self.frame, timestamp)
        if self._tick_callbacks:
            snapshot = self.get_snapshot()
            for callback in list(self._tick_callbacks.values()):
                callback(snapshot)
        return self.frame

    def wait_for_tick(self, seconds: float = 10.0) -> WorldSnapshot:
        self.tick(seconds)
        return self.get_snapshot()

    def on_tick(self, callback: Callable[[WorldSnapshot], None]) -> int:
        callback_id = len(self._tick_callbacks) + 1
        self._tick_callbacks[callback_id] = callback
        return callback_id

    def remove_on_tick(self, callback_id: int) -> None:
        self._tick_callbacks.pop(callback_id, None)

    def get_snapshot(self) -> WorldSnapshot:
        return WorldSnapshot(self)

    def _ground_projection(
        self, location: Vector3D, search_distance: float = 10000.0
    ) -> Optional[LabelledPoint]:
        return LabelledPoint(Location(location.x, location.y, self.ground_z))

    def _project_point(
        self, location: Vector3D, direction: Vector3D, search_distance: float = 10000.0
    ) -> Optional[LabelledPoint]:
        return LabelledPoint(Location(location.x, location.y, self.ground_z))

    def _get_lightmanager(self) -> LightManager:
        return self._light_manager

    def set_pedestrians_seed(self, seed: int) -> None:
        self.pedestrians_seed = int(seed)

    def set_pedestrians_cross_factor(self, percentage: float) -> None:
        self.pedestrians_cross_factor = float(percentage)

    def get_random_location_from_navigation(self) -> Location:
        centre = self._map._centre()
        angle = float(self._rng.uniform(0.0, 2.0 * math.pi))
        radius = float(self._rng.uniform(30.0, 80.0))
        return Location(centre[0] + radius * math.cos(angle),
                        centre[1] + radius * math.sin(angle), 0.2)

    def unload_map_layer(self, layers: Any) -> None:
        self.unloaded_layers.append(layers)

    def load_map_layer(self, layers: Any) -> None:
        if layers in self.unloaded_layers:
            self.unloaded_layers.remove(layers)

    def get_traffic_lights(self) -> ActorList:
        return ActorList()

    def freeze_all_traffic_lights(self, frozen: bool) -> None:
        self.traffic_lights_frozen = bool(frozen)

    def _resolve(self, actor: Any) -> Optional[Actor]:
        return actor if isinstance(actor, Actor) else self.get_actor(int(actor))


class Response:
    """Outcome of one batched command."""

    def __init__(self, actor_id: int = 0, error: str = "") -> None:
        self.actor_id = actor_id
        self.error = error

    def has_error(self) -> bool:
        return bool(self.error)


class _Command:
    """Batched command. ``run`` applies it to the world and returns a :class:`Response`."""

    def __init__(self, *args: Any) -> None:
        self.args = args
        self.followups: List["_Command"] = []

    def then(self, command: "_Command") -> "_Command":
        self.followups.append(command)
        return self

    def _apply(self, world: World, actor: Optional[Actor]) -> None:
        raise NotImplementedError

    def run(self, world: World) -> Response:
        actor = world._resolve(self.args[0])
        if actor is None or not actor.is_alive:
            return Response(0, "actor not found")
        self._apply(world, actor)
        return Response(actor.id)


class _SpawnActor(_Command):
    def run(self, world: World) -> Response:
        blueprint, transform = self.args[0], self.args[1]
        parent = world._resolve(self.args[2]) if len(self.args) > 2 and self.args[2] else None
        actor = world.try_spawn_actor(blueprint, transform, parent)
        if actor is None:
            return Response(0, "Spawn failed because of collision at spawn position")
        for command in self.followups:
            if command.args and command.args[0] is _FUTURE_ACTOR:
                command.args = (actor,) + tuple(command.args[1:])
            command.run(world)
        return Response(actor.id)


class _DestroyActor(_Command):
    def _apply(self, world: World, actor: Optional[Actor]) -> None:
        actor.destroy()


class _SetAutopilot(_Command):
    def _apply(self, world: World, actor: Optional[Actor]) -> None:
        actor.set_autopilot(self.args[1])


class _ApplyTransform(_Command):
    def _apply(self, world: World, actor: Optional[Actor]) -> None:
        actor.set_transform(self.args[1])


class _ApplyControl(_Command):
    def _apply(self, world: World, actor: Optional[Actor]) -> None:
        actor.apply_control(self.args[1])


class _ApplyTargetVelocity(_Command):
    def _apply(self, world: World, actor: Optional[Actor]) -> None:
        actor.set_target_velocity(self.args[1])


class _ApplyTargetAngularVelocity(_Command):
    def _apply(self, world: World, actor: Optional[Actor]) -> None:
        actor.set_target_angular_velocity(self.args[1])


class _SetSimulatePhysics(_Command):
    def _apply(self, world: World, actor: Optional[Actor]) -> None:
        actor.set_simulate_physics(self.args[1])


class _SetEnableGravity(_Command):
    def _apply(self, world: World, actor: Optional[Actor]) -> None:
        actor.set_enable_gravity(self.args[1])


_FUTURE_ACTOR = object()

command = types.SimpleNamespace(
    SpawnActor=_SpawnActor,
    DestroyActor=_DestroyActor,
    SetAutopilot=_SetAutopilot,
    ApplyTransform=_ApplyTransform,
    ApplyWalkerControl=_ApplyControl,
    ApplyVehicleControl=_ApplyControl,
    ApplyTargetVelocity=_ApplyTargetVelocity,
    ApplyTargetAngularVelocity=_ApplyTargetAngularVelocity,
    SetSimulatePhysics=_SetSimulatePhysics,
    SetEnableGravity=_SetEnableGravity,
    FutureActor=_FUTURE_ACTOR,
    Response=Response,
)


def _is_two_wheeler(blueprint: ActorBlueprint) -> bool:
    return (
        blueprint.has_attribute("number_of_wheels")
        and blueprint.get_attribute("number_of_wheels").as_int() == 2
    )


def _extent_for(blueprint: ActorBlueprint) -> Tuple[float, float, float]:
    if blueprint.id.startswith("walker."):
        return _WALKER_EXTENT
    if blueprint.id.startswith("vehicle."):
        return _TWO_WHEELER_EXTENT if _is_two_wheeler(blueprint) else _CAR_EXTENT
    return (0.0, 0.0, 0.0)


def _walker_blueprints(count: int = 20) -> List[ActorBlueprint]:
    blueprints = []
    for index in range(1, count + 1):
        child = index in (9, 10, 11, 12)
        wheelchair = index in (15, 16, 17)
        blueprints.append(ActorBlueprint(f"walker.pedestrian.{index:04d}", [
            ActorAttribute("is_invincible", "true", ["true", "false"]),
            ActorAttribute("speed", "0.0", ["0.0", "1.1" if child else "1.4", "2.2"]),
            ActorAttribute("age", "child" if child else "adult", is_modifiable=False),
            ActorAttribute("gender", "female" if index % 2 else "male", is_modifiable=False),
            ActorAttribute("can_use_wheelchair", "true" if wheelchair else "false",
                           ["true"] if wheelchair else [], is_modifiable=False),
            ActorAttribute("use_wheelchair", "false", ["true", "false"]),
            ActorAttribute("role_name", "walker"),
            ActorAttribute("generation", "2" if index > 14 else "1", is_modifiable=False),
        ]))
    return blueprints


def _vehicle_blueprint(blueprint_id: str, wheels: int, base_type: str) -> ActorBlueprint:
    return ActorBlueprint(blueprint_id, [
        ActorAttribute("number_of_wheels", str(wheels), is_modifiable=False),
        ActorAttribute("base_type", base_type, is_modifiable=False),
        ActorAttribute("role_name", "autopilot", ["autopilot", "scenario", "hero"]),
        ActorAttribute("color", "0,0,0", ["0,0,0", "180,30,30", "30,60,160", "230,230,230"]),
        ActorAttribute("generation", "1", is_modifiable=False),
        ActorAttribute("sticky_control", "true", ["true", "false"]),
    ])


def _camera_blueprint(blueprint_id: str, post_process: Sequence[Tuple[str, str]]) -> ActorBlueprint:
    attributes = [
        ActorAttribute("image_size_x", "800"),
        ActorAttribute("image_size_y", "600"),
        ActorAttribute("fov", "90.0"),
        ActorAttribute("sensor_tick", "0.0"),
        ActorAttribute("role_name", "default"),
    ]
    attributes.extend(ActorAttribute(name, value) for name, value in post_process)
    return ActorBlueprint(blueprint_id, attributes)


_RGB_ATTRIBUTES_UE4: Tuple[Tuple[str, str], ...] = (
    ("motion_blur_intensity", "0.45"),
    ("motion_blur_max_distortion", "0.35"),
    ("motion_blur_min_object_screen_size", "0.1"),
    ("bloom_intensity", "0.675"),
    ("lens_flare_intensity", "0.1"),
    ("exposure_mode", "histogram"),
    ("exposure_compensation", "0.0"),
    ("exposure_min_bright", "10.0"),
    ("exposure_max_bright", "12.0"),
    ("exposure_speed_up", "3.0"),
    ("exposure_speed_down", "1.0"),
    ("gamma", "2.2"),
    ("shutter_speed", "200.0"),
    ("iso", "100.0"),
    ("fstop", "1.4"),
    ("enable_postprocess_effects", "true"),
)

_RGB_ATTRIBUTES_UE5: Tuple[Tuple[str, str], ...] = tuple(
    item for item in _RGB_ATTRIBUTES_UE4
    if item[0] not in ("lens_flare_intensity", "motion_blur_max_distortion", "gamma")
)


def _build_library(
    version: str, two_wheelers: bool, with_instance_segmentation: bool
) -> BlueprintLibrary:
    ue5 = version.startswith("0.10")
    blueprints = _walker_blueprints()
    blueprints.append(ActorBlueprint("controller.ai.walker", [ActorAttribute("role_name", "ai")]))
    if ue5:
        cars = ("vehicle.dodge.charger", "vehicle.mini.cooper", "vehicle.nissan.patrol")
        blueprints.append(ActorBlueprint("static.prop.yamaha", [
            ActorAttribute("role_name", "prop"),
        ]))
    else:
        cars = ("vehicle.tesla.model3", "vehicle.audi.tt")
    blueprints.extend(_vehicle_blueprint(car, 4, "car") for car in cars)
    if two_wheelers:
        for bicycle in ("vehicle.diamondback.century", "vehicle.bh.crossbike",
                        "vehicle.gazelle.omafiets"):
            blueprints.append(_vehicle_blueprint(bicycle, 2, "bicycle"))
        for motorcycle in ("vehicle.vespa.zx125", "vehicle.yamaha.yzf"):
            blueprints.append(_vehicle_blueprint(motorcycle, 2, "motorcycle"))
    rgb = _RGB_ATTRIBUTES_UE5 if ue5 else _RGB_ATTRIBUTES_UE4
    blueprints.append(_camera_blueprint("sensor.camera.rgb", rgb))
    blueprints.append(_camera_blueprint("sensor.camera.depth", ()))
    blueprints.append(_camera_blueprint("sensor.camera.semantic_segmentation", ()))
    if with_instance_segmentation:
        blueprints.append(_camera_blueprint("sensor.camera.instance_segmentation", ()))
    return BlueprintLibrary(blueprints)


class FakeServer:
    """State shared by every client created from one fake module.

    ``connection_failures`` makes the next ``n`` connection attempts time out,
    which exercises retry loops. ``load_count`` counts ``load_world`` calls.
    """

    def __init__(
        self,
        version: str,
        client_version: str,
        maps: Sequence[str],
        library: BlueprintLibrary,
        crosswalks: Sequence[Sequence[Sequence[float]]],
        with_ground_projection: bool,
        with_light_manager: bool,
    ) -> None:
        self.version = version
        self.client_version = client_version
        self.maps = list(maps)
        self.library = library
        self.crosswalks = crosswalks
        self.with_ground_projection = with_ground_projection
        self.with_light_manager = with_light_manager
        self.connection_failures = 0
        self.connection_attempts = 0
        self.load_count = 0
        self.clients: List[Any] = []
        self.traffic_managers: Dict[int, TrafficManager] = {}
        self._world_ids = 0
        self.world = World(self, self.maps[0])

    def next_world_id(self) -> int:
        self._world_ids += 1
        return self._world_ids

    def connect(self) -> None:
        self.connection_attempts += 1
        if self.connection_failures > 0:
            self.connection_failures -= 1
            raise RuntimeError(
                "time-out of 2000ms while waiting for the simulator, make sure the simulator "
                "is ready and connected to 127.0.0.1:2000"
            )

    def traffic_manager(self, port: int) -> TrafficManager:
        if port not in self.traffic_managers:
            self.traffic_managers[port] = TrafficManager(port)
        return self.traffic_managers[port]


class _Client:
    """Client bound to the :class:`FakeServer` of the module it was created from."""

    _server: FakeServer

    def __init__(self, host: str = "127.0.0.1", port: int = 2000, worker_threads: int = 0) -> None:
        self.host = host
        self.port = port
        self.timeout: Optional[float] = None
        self._server.clients.append(self)

    def set_timeout(self, seconds: float) -> None:
        self.timeout = float(seconds)

    def get_server_version(self) -> str:
        self._server.connect()
        return self._server.version

    def get_client_version(self) -> str:
        return self._server.client_version

    def get_world(self) -> World:
        self._server.connect()
        return self._server.world

    def get_available_maps(self) -> List[str]:
        return [f"/Game/Carla/Maps/{name}" for name in self._server.maps]

    def load_world(
        self, map_name: str, reset_settings: bool = True, map_layers: Any = None
    ) -> World:
        name = map_name.rsplit("/", 1)[-1]
        if name not in self._server.maps:
            raise RuntimeError(f"map not found: {map_name}")
        previous = self._server.world
        for actor in list(previous.actors.values()):
            actor.destroy()
        world = World(self._server, name)
        if not reset_settings:
            world._settings = previous.get_settings()
        self._server.world = world
        self._server.load_count += 1
        return world

    def reload_world(self, reset_settings: bool = True) -> World:
        name = self._server.world.get_map().name.rsplit("/", 1)[-1]
        return self.load_world(name, reset_settings)

    def get_trafficmanager(self, client_connection: int = 8000) -> TrafficManager:
        return self._server.traffic_manager(client_connection)

    def apply_batch(self, commands: Iterable[_Command]) -> None:
        for item in commands:
            item.run(self._server.world)

    def apply_batch_sync(
        self, commands: Iterable[_Command], due_tick_cue: bool = False
    ) -> List[Response]:
        responses = [item.run(self._server.world) for item in commands]
        if due_tick_cue:
            self._server.world.tick()
        return responses


def make_fake_carla(
    version: str = "0.9.16",
    two_wheelers: bool = True,
    maps: Sequence[str] = ("Town10HD_Opt",),
    with_ground_projection: bool = True,
    with_light_manager: bool = True,
    with_instance_segmentation: bool = True,
    client_version: Optional[str] = None,
    crosswalks: Optional[Sequence[Sequence[Sequence[float]]]] = None,
) -> types.SimpleNamespace:
    """Build a module-like object that stands in for ``carla``.

    ``version`` is what ``Client.get_server_version()`` reports, and
    ``client_version`` (the same by default) what ``get_client_version()``
    reports. ``two_wheelers=False`` removes bicycles and motorcycles from the
    blueprint library. ``crosswalks`` is a list of four-corner polygons in the
    world frame; the default is one rectangle at crosswalk 8 of Town10HD_Opt.
    The returned object exposes the server state as ``server`` (see
    :class:`FakeServer`), with the current world at ``server.world``.
    """
    server = FakeServer(
        version=version,
        client_version=client_version or version,
        maps=maps,
        library=_build_library(version, two_wheelers, with_instance_segmentation),
        crosswalks=crosswalks if crosswalks is not None else (CROSSWALK_8,),
        with_ground_projection=with_ground_projection,
        with_light_manager=with_light_manager,
    )
    client_class = type("Client", (_Client,), {"_server": server})
    return types.SimpleNamespace(
        __name__="carla",
        server=server,
        Client=client_class,
        World=World,
        Map=Map,
        Waypoint=Waypoint,
        Actor=Actor,
        Sensor=Sensor,
        WalkerAIController=WalkerAIController,
        ActorList=ActorList,
        ActorBlueprint=ActorBlueprint,
        ActorAttribute=ActorAttribute,
        BlueprintLibrary=BlueprintLibrary,
        Image=Image,
        Timestamp=Timestamp,
        Vector3D=Vector3D,
        Location=Location,
        Rotation=Rotation,
        Transform=Transform,
        BoundingBox=BoundingBox,
        WorldSettings=WorldSettings,
        WeatherParameters=WeatherParameters,
        WalkerControl=WalkerControl,
        VehicleControl=VehicleControl,
        AttachmentType=AttachmentType,
        ColorConverter=ColorConverter,
        VehicleLightState=VehicleLightState,
        LightGroup=LightGroup,
        LaneType=LaneType,
        MapLayer=MapLayer,
        LightManager=LightManager,
        TrafficManager=TrafficManager,
        command=command,
    )
