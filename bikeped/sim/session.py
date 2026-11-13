"""Connection to a CARLA server in synchronous mode with guaranteed cleanup.

:class:`Session` connects (with retries), loads the requested map only when it
is not already loaded, switches the world and the Traffic Manager to
synchronous mode with a fixed time step, tracks every actor spawned through it
and, on exit (including ``KeyboardInterrupt``), destroys those actors and
restores the previous world settings. :class:`SensorHub` returns, for each
``world.tick()``, the measurement of every sensor for exactly that frame.
"""

from __future__ import annotations

import logging
import queue
import time
from typing import Any, Callable, Dict, List, Optional

import numpy as np

from .engine import carla_module, detect_engine

log = logging.getLogger(__name__)


def image_to_bgr(image: Any) -> np.ndarray:
    """``carla.Image`` (BGRA bytes) to a contiguous ``uint8`` BGR array."""
    buf = np.frombuffer(image.raw_data, dtype=np.uint8).reshape(image.height, image.width, 4)
    return np.ascontiguousarray(buf[:, :, :3])


def depth_to_metres(image: Any) -> np.ndarray:
    """Decode a ``sensor.camera.depth`` image to ``float32`` metres (1000 m far plane)."""
    buf = np.frombuffer(image.raw_data, dtype=np.uint8).reshape(image.height, image.width, 4)
    b, g, r = (buf[:, :, i].astype(np.float32) for i in range(3))
    return ((r + g * 256.0 + b * 65536.0) / (256.0 ** 3 - 1.0) * 1000.0).astype(np.float32)


def instance_ids(image: Any) -> np.ndarray:
    """Decode ``sensor.camera.instance_segmentation`` to ``int32`` object ids (G + B * 256)."""
    buf = np.frombuffer(image.raw_data, dtype=np.uint8).reshape(image.height, image.width, 4)
    return buf[:, :, 1].astype(np.int32) + buf[:, :, 0].astype(np.int32) * 256


class SensorHub:
    """Frame-synchronised collection of sensor data.

    Each registered sensor gets its own queue; :meth:`collect` blocks until
    every sensor has delivered the measurement of the requested frame and
    discards older ones.
    """

    def __init__(self, timeout: float = 10.0) -> None:
        self.timeout = timeout
        self._queues: Dict[str, "queue.Queue[Any]"] = {}
        self._decoders: Dict[str, Callable[[Any], Any]] = {}

    def add(self, key: str, sensor: Any, decode: Callable[[Any], Any] = image_to_bgr) -> None:
        q: "queue.Queue[Any]" = queue.Queue()
        self._queues[key] = q
        self._decoders[key] = decode
        sensor.listen(q.put)

    @property
    def keys(self) -> List[str]:
        return list(self._queues)

    def collect(self, frame: int) -> Dict[str, Any]:
        """Decoded data of every sensor for ``frame``."""
        out: Dict[str, Any] = {}
        deadline = time.monotonic() + self.timeout
        for key, q in self._queues.items():
            while True:
                remaining = deadline - time.monotonic()
                try:
                    data = q.get(timeout=max(remaining, 0.001))
                except queue.Empty:
                    raise TimeoutError(
                        f"sensor '{key}' delivered nothing for frame {frame} within "
                        f"{self.timeout:.0f} s; the server may be overloaded (try --quality draft "
                        "or fewer views)") from None
                if data.frame == frame:
                    out[key] = self._decoders[key](data)
                    break
                if data.frame > frame:
                    raise RuntimeError(f"sensor '{key}' skipped frame {frame} (got {data.frame})")
        return out


class Session:
    """Synchronous-mode session on one CARLA server.

    Use as a context manager. ``fixed_delta`` is the simulation step in seconds
    (``1 / fps``). ``map_name`` is loaded only when the current map differs.
    """

    def __init__(self, host: str = "127.0.0.1", port: int = 2000, *, map_name: Optional[str] = None,
                 fixed_delta: float = 1.0 / 30.0, seed: int = 0, timeout: float = 30.0,
                 retries: int = 5, tm_port: Optional[int] = None) -> None:
        self.host, self.port = host, port
        self.map_name = map_name
        self.fixed_delta = fixed_delta
        self.seed = seed
        self.timeout = timeout
        self.retries = retries
        self.tm_port = tm_port if tm_port is not None else port + 6000
        self.carla: Any = None
        self.client: Any = None
        self.world: Any = None
        self.engine = ""
        self.actors: List[Any] = []
        self.sensors: List[Any] = []
        self._original_settings: Any = None
        self._tm: Any = None
        self.frame = 0
        #: Ids of map objects disabled for this session, enabled again on close.
        self.hidden_objects: set = set()

    def __enter__(self) -> "Session":
        try:
            self.open()
        except BaseException:
            self.close()
            raise
        return self

    def __exit__(self, *exc: Any) -> None:
        self.close()

    def open(self) -> None:
        self.carla = carla_module()
        last: Optional[BaseException] = None
        for attempt in range(self.retries):
            try:
                self.client = self.carla.Client(self.host, self.port)
                self.client.set_timeout(self.timeout)
                self.client.get_server_version()
                break
            except RuntimeError as exc:
                last = exc
                log.info("connection attempt %d to %s:%d failed: %s", attempt + 1,
                         self.host, self.port, exc)
                time.sleep(min(2.0 * (attempt + 1), 10.0))
        else:
            raise ConnectionError(f"no CARLA server at {self.host}:{self.port}: {last}")
        self.engine = detect_engine(self.client)
        self.world = self.client.get_world()
        if self.map_name:
            current = self.world.get_map().name.split("/")[-1]
            if current != self.map_name.split("/")[-1]:
                log.info("loading map %s (was %s)", self.map_name, current)
                self.client.set_timeout(max(self.timeout, 120.0))
                self.world = self.client.load_world(self.map_name)
                self.client.set_timeout(self.timeout)
        self._original_settings = self.world.get_settings()
        settings = self.world.get_settings()
        settings.synchronous_mode = True
        settings.fixed_delta_seconds = self.fixed_delta
        if hasattr(settings, "no_rendering_mode"):
            settings.no_rendering_mode = False
        self.world.apply_settings(settings)
        if hasattr(self.world, "set_pedestrians_seed"):
            self.world.set_pedestrians_seed(self.seed)
        self.frame = self.world.tick()

    def traffic_manager(self) -> Any:
        """Traffic Manager in synchronous mode and seeded, or ``None`` when unavailable."""
        if self._tm is None:
            try:
                tm = self.client.get_trafficmanager(self.tm_port)
                tm.set_synchronous_mode(True)
                if hasattr(tm, "set_random_device_seed"):
                    tm.set_random_device_seed(self.seed)
                self._tm = tm
            except RuntimeError as exc:
                log.warning("Traffic Manager unavailable: %s", exc)
                return None
        return self._tm

    def spawn(self, blueprint: Any, transform: Any, attach_to: Any = None) -> Any:
        """Spawn and track an actor; raises ``RuntimeError`` when the server refuses."""
        if attach_to is not None:
            actor = self.world.spawn_actor(blueprint, transform, attach_to=attach_to)
        else:
            actor = self.world.spawn_actor(blueprint, transform)
        self._track(actor)
        return actor

    def try_spawn(self, blueprint: Any, transform: Any, attach_to: Any = None) -> Any:
        """Like :meth:`spawn` but returns ``None`` on failure."""
        if attach_to is not None:
            actor = self.world.try_spawn_actor(blueprint, transform, attach_to=attach_to)
        else:
            actor = self.world.try_spawn_actor(blueprint, transform)
        if actor is not None:
            self._track(actor)
        return actor

    def _track(self, actor: Any) -> None:
        if str(getattr(actor, "type_id", "")).startswith("sensor."):
            self.sensors.append(actor)
        else:
            self.actors.append(actor)

    def destroy(self, actor: Any) -> None:
        """Destroy one tracked actor now and stop tracking it."""
        for group in (self.actors, self.sensors):
            if actor in group:
                group.remove(actor)
        if getattr(actor, "is_alive", True):
            actor.destroy()

    def tick(self) -> int:
        self.frame = self.world.tick()
        return self.frame

    def close(self) -> None:
        """Stop sensors, destroy tracked actors, restore settings. Safe to call twice."""
        for sensor in reversed(self.sensors):
            try:
                if getattr(sensor, "is_listening", False):
                    sensor.stop()
            except RuntimeError:
                pass
        for actor in reversed(self.sensors + self.actors):
            try:
                if getattr(actor, "is_alive", True):
                    actor.destroy()
            except RuntimeError as exc:
                log.debug("destroy %s failed: %s", getattr(actor, "id", "?"), exc)
        self.sensors, self.actors = [], []
        if self.hidden_objects and self.world is not None:
            try:
                self.world.enable_environment_objects(set(self.hidden_objects), True)
            except RuntimeError as exc:
                log.warning("could not restore hidden map objects: %s", exc)
            self.hidden_objects = set()
        if self._tm is not None:
            try:
                self._tm.set_synchronous_mode(False)
            except RuntimeError:
                pass
            self._tm = None
        if self.world is not None and self._original_settings is not None:
            try:
                self.world.apply_settings(self._original_settings)
            except RuntimeError as exc:
                log.warning("could not restore world settings: %s", exc)
            self._original_settings = None
