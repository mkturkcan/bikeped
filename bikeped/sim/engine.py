"""Engine profiles, server detection and capability probing.

Two CARLA generations are supported: CARLA 0.9.14 to 0.9.16 on Unreal Engine 4
(``ue4``) and CARLA 0.10.0 on Unreal Engine 5 (``ue5``). The rest of the package
asks :func:`probe` what a server offers instead of branching on version numbers,
so a feature missing from one build degrades the same way on any engine.

``carla`` is imported lazily through :func:`carla_module`. Tests replace it with
:func:`set_carla_module`.
"""

from __future__ import annotations

import importlib
import logging
import os
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional, Tuple

log = logging.getLogger(__name__)

_CARLA_OVERRIDE: Any = None


def set_carla_module(module: Any) -> None:
    """Use ``module`` in place of the real ``carla`` package (``None`` restores it)."""
    global _CARLA_OVERRIDE
    _CARLA_OVERRIDE = module


def carla_module() -> Any:
    """Return the ``carla`` module, importing it on first use.

    Raises ``ImportError`` with installation hints when the client is missing.
    """
    if _CARLA_OVERRIDE is not None:
        return _CARLA_OVERRIDE
    try:
        return importlib.import_module("carla")
    except ImportError as exc:
        raise ImportError(
            "The CARLA Python client is not installed. Install the wheel shipped with the "
            "server, e.g. PythonAPI/carla/dist/carla-0.9.16-*.whl for Unreal Engine 4 or "
            "carla-0.10.0-cp310-*.whl (Python 3.10) for Unreal Engine 5."
        ) from exc


@dataclass(frozen=True)
class EngineProfile:
    """Static facts about one CARLA generation."""

    name: str
    launcher: str
    default_map: str
    default_drive: str
    default_port: int
    launch_flags: Tuple[str, ...]
    supports_quality_flag: bool
    root_env: str
    wheel_hint: str
    search_paths: Tuple[str, ...] = ()


PROFILES: Dict[str, EngineProfile] = {
    "ue4": EngineProfile(
        name="ue4",
        launcher="CarlaUE4.sh",
        default_map="Town10HD_Opt",
        default_drive="kinematic",
        default_port=2000,
        launch_flags=("-RenderOffScreen", "-nosound"),
        supports_quality_flag=True,
        root_env="CARLA_UE4_ROOT",
        wheel_hint="PythonAPI/carla/dist/carla-0.9.16-cp312-cp312-manylinux_2_31_x86_64.whl",
        search_paths=("/opt/carla-simulator", "~/CARLA_0.9.16", "~/carla", "/data/CARLA_Latest"),
    ),
    "ue5": EngineProfile(
        name="ue5",
        launcher="CarlaUnreal.sh",
        default_map="Town10HD_Opt",
        default_drive="kinematic",
        default_port=2000,
        launch_flags=("-RenderOffScreen", "-nosound"),
        supports_quality_flag=False,
        root_env="CARLA_UE5_ROOT",
        wheel_hint="PythonAPI/carla/dist/carla-0.10.0-cp310-cp310-linux_x86_64.whl",
        search_paths=("~/Carla-0.10.0-Linux-Shipping", "/opt/Carla-0.10.0-Linux-Shipping",
                      "~/carlabins/Carla-0.10.0-Linux-Shipping"),
    ),
}


def engine_for_version(version: str) -> str:
    """Map a CARLA version string to ``ue4`` (0.9.x) or ``ue5`` (0.10.x and later)."""
    parts = version.strip().lstrip("v").split(".")
    try:
        major, minor = int(parts[0]), int(parts[1])
    except (IndexError, ValueError):
        # Development builds report a git hash; only 0.9.x builds do that on this API.
        return "ue4"
    return "ue4" if (major, minor) < (0, 10) else "ue5"


def detect_engine(client: Any) -> str:
    """Return ``ue4`` or ``ue5`` for the connected server and warn on a client mismatch."""
    server = str(client.get_server_version())
    engine = engine_for_version(server)
    client_version = str(getattr(client, "get_client_version", lambda: "")())
    if client_version and engine_for_version(client_version) != engine:
        log.warning(
            "CARLA client %s does not match server %s (%s); install %s",
            client_version, server, engine, PROFILES[engine].wheel_hint,
        )
    return engine


def attribute_text(blueprint: Any, name: str) -> Optional[str]:
    """Value of a blueprint attribute as lower-case text, or ``None`` when absent.

    The real client refuses ``as_str()`` on numeric attributes, so the typed
    accessors are tried in turn.
    """
    if not blueprint.has_attribute(name):
        return None
    attribute = blueprint.get_attribute(name)
    for getter in ("as_str", "as_int", "as_float", "as_bool"):
        fn = getattr(attribute, getter, None)
        if fn is None:
            continue
        try:
            return str(fn()).lower()
        except (RuntimeError, ValueError, TypeError):
            continue
    return str(attribute).lower()


_attr = attribute_text


def is_two_wheeler(blueprint: Any) -> bool:
    """True for bicycles, scooters and motorcycles in any CARLA build."""
    if _attr(blueprint, "number_of_wheels") == "2":
        return True
    return (_attr(blueprint, "base_type") or "").lower() in ("bicycle", "motorcycle")


@dataclass
class Capabilities:
    """What one running server offers to the renderer."""

    engine: str
    server_version: str
    client_version: str
    map: str
    maps: List[str] = field(default_factory=list)
    two_wheelers: List[str] = field(default_factory=list)
    bicycles: List[str] = field(default_factory=list)
    walkers: int = 0
    wheelchair_walkers: List[str] = field(default_factory=list)
    child_walkers: List[str] = field(default_factory=list)
    vehicles: int = 0
    depth_camera: bool = False
    instance_camera: bool = False
    ground_projection: bool = False
    light_manager: bool = False
    crosswalks: int = 0

    @property
    def has_rideable(self) -> bool:
        return bool(self.two_wheelers)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def probe(client: Any) -> Capabilities:
    """Query a connected server for maps, blueprints, sensors and world features."""
    engine = detect_engine(client)
    world = client.get_world()
    library = world.get_blueprint_library()
    caps = Capabilities(
        engine=engine,
        server_version=str(client.get_server_version()),
        client_version=str(getattr(client, "get_client_version", lambda: "")()),
        map=world.get_map().name.split("/")[-1],
        maps=sorted({m.split("/")[-1] for m in client.get_available_maps()}),
    )
    for bp in library.filter("vehicle.*"):
        if is_two_wheeler(bp):
            caps.two_wheelers.append(bp.id)
            if (_attr(bp, "base_type") or "").lower() == "bicycle":
                caps.bicycles.append(bp.id)
        else:
            caps.vehicles += 1
    for bp in library.filter("walker.pedestrian.*"):
        caps.walkers += 1
        if (_attr(bp, "can_use_wheelchair") or "").lower() == "true":
            caps.wheelchair_walkers.append(bp.id)
        if (_attr(bp, "age") or "").lower() == "child":
            caps.child_walkers.append(bp.id)
    ids = {bp.id for bp in library.filter("sensor.camera.*")}
    caps.depth_camera = "sensor.camera.depth" in ids
    caps.instance_camera = "sensor.camera.instance_segmentation" in ids
    caps.ground_projection = hasattr(world, "ground_projection")
    caps.light_manager = hasattr(world, "get_lightmanager")
    try:
        caps.crosswalks = len(world.get_map().get_crosswalks()) // 5
    except (AttributeError, RuntimeError):
        caps.crosswalks = 0
    return caps


def format_capabilities(caps: Capabilities) -> str:
    """Render capabilities as an aligned two-column table."""
    def yes(flag: bool) -> str:
        return "yes" if flag else "no"

    rows = [
        ("engine", caps.engine),
        ("server version", caps.server_version),
        ("client version", caps.client_version),
        ("current map", caps.map),
        ("maps", ", ".join(caps.maps)),
        ("two-wheelers", ", ".join(caps.two_wheelers) or "none (cyclists need --cyclist-proxy)"),
        ("bicycles", ", ".join(caps.bicycles) or "none"),
        ("walkers", f"{caps.walkers} ({len(caps.child_walkers)} child, "
                    f"{len(caps.wheelchair_walkers)} wheelchair-capable)"),
        ("four-wheeled vehicles", str(caps.vehicles)),
        ("depth camera", yes(caps.depth_camera)),
        ("instance segmentation", yes(caps.instance_camera)),
        ("ground projection", yes(caps.ground_projection)),
        ("light manager", yes(caps.light_manager)),
        ("crosswalks on map", str(caps.crosswalks)),
    ]
    width = max(len(k) for k, _ in rows)
    return "\n".join(f"{k.ljust(width)}  {v}" for k, v in rows)


def root_from_env(engine: str) -> Optional[str]:
    """Install root named by ``CARLA_UE4_ROOT`` or ``CARLA_UE5_ROOT``, if set."""
    return os.environ.get(PROFILES[engine].root_env) or None
