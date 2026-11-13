"""Background traffic and pedestrians around a scripted scenario.

Ambient actors make the rendered street look inhabited without getting in the
way of the scripted agents. The scripted corridor is the bounding box of all
scripted paths in the scenario frame plus a margin. Vehicles are spawned
outside it, drive under the Traffic Manager and are stopped when they
approach it. Walkers are spawned on the navigation mesh outside it, do not
cross the road, and are sent back to where they started if they wander in.

The corridor helpers are plain numpy and need no simulator.
"""

from __future__ import annotations

import logging
import random
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Set, Tuple

import numpy as np

from .engine import attribute_text

logger = logging.getLogger(__name__)

#: ``(xmin, ymin, xmax, ymax)`` in the scenario frame, metres.
Bounds = Tuple[float, float, float, float]

#: Number of ambient actors requested per level.
AMBIENT_LEVELS: Dict[str, Dict[str, int]] = {
    "none": {"vehicles": 0, "walkers": 0},
    "light": {"vehicles": 6, "walkers": 8},
    "busy": {"vehicles": 20, "walkers": 24},
}

#: Width of the band around the corridor in which approaching vehicles are stopped, metres.
HOLD_BACK_M = 12.0

#: Ambient vehicles drive this many percent below the speed limit.
CRUISE_SPEED_DIFFERENCE = 30.0

#: Look-ahead used to decide whether a vehicle is heading for the corridor, seconds.
APPROACH_HORIZON_S = 1.0

#: A vehicle counts as approaching when the look-ahead brings it this much closer, metres.
APPROACH_MIN_GAIN_M = 0.05

#: Walkers are spawned this far above the navigation mesh so they do not start in the ground.
SPAWN_LIFT_M = 0.5

#: Port of the Traffic Manager when the manager object cannot report its own.
DEFAULT_TM_PORT = 8000


def corridor_bounds(scenario: Dict[str, Any], margin: float = 6.0) -> Bounds:
    """Bounding box of every scripted path point, grown by ``margin`` metres.

    Returns ``(xmin, ymin, xmax, ymax)`` in the scenario frame.
    """
    points = np.array(
        [(p["x"], p["y"]) for agent in scenario.get("agents", []) for p in agent["path"]],
        dtype=np.float64,
    )
    if points.size == 0:
        raise ValueError("the scenario has no path points to build a corridor from")
    low = points.min(axis=0) - margin
    high = points.max(axis=0) + margin
    return float(low[0]), float(low[1]), float(high[0]), float(high[1])


def expand_bounds(bounds: Bounds, distance: float) -> Bounds:
    """Grow ``bounds`` by ``distance`` metres on every side."""
    xmin, ymin, xmax, ymax = bounds
    return xmin - distance, ymin - distance, xmax + distance, ymax + distance


def in_corridor(x: float, y: float, bounds: Bounds) -> bool:
    """Whether the scenario-frame point ``(x, y)`` lies inside ``bounds``."""
    xmin, ymin, xmax, ymax = bounds
    return bool(xmin <= x <= xmax and ymin <= y <= ymax)


def distance_to_bounds(x: float, y: float, bounds: Bounds) -> float:
    """Distance in metres from ``(x, y)`` to ``bounds``, zero inside."""
    xmin, ymin, xmax, ymax = bounds
    dx = max(xmin - x, 0.0, x - xmax)
    dy = max(ymin - y, 0.0, y - ymax)
    return float(np.hypot(dx, dy))


def _attribute_text(blueprint: Any, name: str) -> Optional[str]:
    """Value of a blueprint attribute as text, or ``None`` when it does not exist."""
    if not blueprint.has_attribute(name):
        return None
    return attribute_text(blueprint, name)


def _is_four_wheeled(blueprint: Any) -> bool:
    """Whether a vehicle blueprint is a car, van or truck rather than a two-wheeler."""
    base_type = _attribute_text(blueprint, "base_type")
    if base_type is not None and base_type.lower() in ("bicycle", "motorcycle"):
        return False
    wheels = _attribute_text(blueprint, "number_of_wheels")
    if wheels is None:
        return True
    try:
        return int(wheels) == 4
    except ValueError:
        return True


def _set_if_present(blueprint: Any, name: str, value: str) -> None:
    if blueprint.has_attribute(name):
        blueprint.set_attribute(name, value)


@dataclass
class _Walker:
    """An ambient walker, its AI controller and the place it started from."""

    actor: Any
    home: Any
    speed: float
    controller: Optional[Any] = None
    returning: bool = False


class Ambient:
    """Ambient vehicles and walkers for one render.

    Every actor is spawned through the session, which tracks and destroys it.
    Spawning is best effort: a map without spawn points, a missing Traffic
    Manager or a full navigation mesh leaves the scene with fewer ambient
    actors and a log message, and never stops the render.

    Args:
        session: the running ``Session``.
        site: the ``Site`` that maps between the world and the scenario frame.
        scenario: the scripted scenario; its paths define the corridor.
        level: a key of :data:`AMBIENT_LEVELS`.
        seed: seed of the blueprint, colour and spawn point choices.
        night: switch vehicle lights on.
    """

    def __init__(
        self,
        session: Any,
        site: Any,
        scenario: Dict[str, Any],
        level: str = "none",
        seed: int = 0,
        night: bool = False,
    ) -> None:
        if level not in AMBIENT_LEVELS:
            raise ValueError(
                f"unknown ambient level {level!r}, expected one of: {', '.join(AMBIENT_LEVELS)}"
            )
        self.session = session
        self.site = site
        self.level = level
        self.seed = int(seed)
        self.night = night
        self.bounds: Bounds = corridor_bounds(scenario)
        self.hold_bounds: Bounds = expand_bounds(self.bounds, HOLD_BACK_M)
        self._rng = random.Random(self.seed)
        self._tm: Optional[Any] = None
        self._tm_port = DEFAULT_TM_PORT
        self._vehicles: List[Any] = []
        self._walkers: List[_Walker] = []
        self._held: Set[int] = set()
        self._braked: Set[int] = set()
        self._ticks = 0
        self._controllers_started = False

    @property
    def counts(self) -> Dict[str, int]:
        """Number of ambient vehicles and walkers alive, and of vehicles held back."""
        return {
            "vehicles": len(self._vehicles),
            "walkers": len(self._walkers),
            "held": len(self._held | self._braked),
        }

    def spawn(self) -> None:
        """Spawn the vehicles and walkers of the chosen level."""
        budget = AMBIENT_LEVELS[self.level]
        if budget["vehicles"]:
            try:
                self._spawn_vehicles(budget["vehicles"])
            except Exception:
                logger.warning("ambient vehicles could not be spawned", exc_info=True)
        if budget["walkers"]:
            try:
                self._spawn_walkers(budget["walkers"])
            except Exception:
                logger.warning("ambient walkers could not be spawned", exc_info=True)
        if budget["vehicles"] or budget["walkers"]:
            logger.info("ambient %s: %d vehicles, %d walkers", self.level,
                        len(self._vehicles), len(self._walkers))

    def tick(self) -> None:
        """Update the ambient actors; call once before every ``world.tick()``."""
        # A walker controller can only be started once the server has ticked
        # after the spawn, so the first call leaves the controllers alone.
        if self._ticks >= 1 and not self._controllers_started:
            self._start_controllers()
        self._ticks += 1
        self._hold_vehicles()
        self._turn_back_walkers()

    def stop(self) -> None:
        """Stop the walker controllers. The session destroys the actors."""
        if self._controllers_started:
            for walker in self._walkers:
                if walker.controller is None:
                    continue
                try:
                    walker.controller.stop()
                except RuntimeError:
                    logger.debug("walker controller was already gone", exc_info=True)
        self._vehicles.clear()
        self._walkers.clear()
        self._held.clear()
        self._braked.clear()

    def _scenario_xy(self, world_x: float, world_y: float) -> Tuple[float, float]:
        x, y = self.site.to_scenario(world_x, world_y)
        return float(x), float(y)

    def _outside(self, location: Any, bounds: Bounds) -> bool:
        """Whether a world-frame location lies outside ``bounds``."""
        return not in_corridor(*self._scenario_xy(location.x, location.y), bounds)

    def _spawn_vehicles(self, count: int) -> None:
        world = self.session.world
        tm = self.session.traffic_manager()
        if tm is None:
            logger.info("no Traffic Manager on this server, ambient vehicles are left out")
            return
        self._tm = tm
        if hasattr(tm, "get_port"):
            self._tm_port = int(tm.get_port())

        blueprints = sorted(
            (bp for bp in world.get_blueprint_library().filter("vehicle.*")
             if _is_four_wheeled(bp)),
            key=lambda bp: bp.id,
        )
        points = [
            point for point in world.get_map().get_spawn_points()
            if self._outside(point.location, self.hold_bounds)
        ]
        if not blueprints or not points:
            logger.info("no vehicle blueprint or free spawn point for ambient traffic")
            return
        self._rng.shuffle(points)

        for point in points:
            if len(self._vehicles) >= count:
                break
            blueprint = self._rng.choice(blueprints)
            _set_if_present(blueprint, "role_name", "ambient")
            if blueprint.has_attribute("color"):
                colours = getattr(blueprint.get_attribute("color"), "recommended_values", None)
                if colours:
                    blueprint.set_attribute("color", self._rng.choice(list(colours)))
            vehicle = self.session.try_spawn(blueprint, point)
            if vehicle is None:
                continue
            if hasattr(vehicle, "set_autopilot"):
                vehicle.set_autopilot(True, self._tm_port)
            if hasattr(tm, "vehicle_percentage_speed_difference"):
                tm.vehicle_percentage_speed_difference(vehicle, CRUISE_SPEED_DIFFERENCE)
            if hasattr(tm, "update_vehicle_lights"):
                tm.update_vehicle_lights(vehicle, True)
            if self.night:
                self._lights_on(vehicle)
            self._vehicles.append(vehicle)

    def _lights_on(self, vehicle: Any) -> None:
        light_state = getattr(self.session.carla, "VehicleLightState", None)
        if light_state is None or not hasattr(vehicle, "set_light_state"):
            return
        try:
            vehicle.set_light_state(light_state(light_state.Position | light_state.LowBeam))
        except (AttributeError, TypeError, RuntimeError):
            logger.debug("could not switch the lights of %s on", vehicle, exc_info=True)

    def _spawn_walkers(self, count: int) -> None:
        world = self.session.world
        carla = self.session.carla
        if not hasattr(world, "get_random_location_from_navigation"):
            logger.info("no pedestrian navigation on this server, ambient walkers are left out")
            return
        if hasattr(world, "set_pedestrians_seed"):
            world.set_pedestrians_seed(self.seed)
        if hasattr(world, "set_pedestrians_cross_factor"):
            world.set_pedestrians_cross_factor(0.0)

        library = world.get_blueprint_library()
        blueprints = sorted(library.filter("walker.pedestrian.*"), key=lambda bp: bp.id)
        controllers = list(library.filter("controller.ai.walker"))
        if not blueprints:
            logger.info("no walker blueprint for ambient pedestrians")
            return

        for _ in range(count):
            home = self._navigation_point()
            if home is None:
                continue
            blueprint = self._rng.choice(blueprints)
            _set_if_present(blueprint, "is_invincible", "false")
            _set_if_present(blueprint, "role_name", "ambient")
            lifted = carla.Location(x=home.x, y=home.y, z=home.z + SPAWN_LIFT_M)
            actor = self.session.try_spawn(blueprint, carla.Transform(lifted))
            if actor is None:
                continue
            walker = _Walker(actor=actor, home=home, speed=self._walker_speed(blueprint))
            if controllers:
                walker.controller = self.session.try_spawn(
                    controllers[0], carla.Transform(), attach_to=actor
                )
            self._walkers.append(walker)

    def _navigation_point(self, attempts: int = 20) -> Optional[Any]:
        """A random sidewalk location outside the corridor, or ``None``."""
        world = self.session.world
        for _ in range(attempts):
            location = world.get_random_location_from_navigation()
            if location is not None and self._outside(location, self.bounds):
                return location
        return None

    def _walker_speed(self, blueprint: Any) -> float:
        """Walking speed in m/s: the blueprint's own value when it lists one."""
        if blueprint.has_attribute("speed"):
            values = getattr(blueprint.get_attribute("speed"), "recommended_values", None)
            if values and len(values) > 1:
                try:
                    return float(values[1])
                except (TypeError, ValueError):
                    pass
        return self._rng.uniform(1.1, 1.5)

    def _start_controllers(self) -> None:
        self._controllers_started = True
        for walker in self._walkers:
            if walker.controller is None:
                continue
            try:
                walker.controller.start()
                destination = self._navigation_point()
                if destination is not None:
                    walker.controller.go_to_location(destination)
                walker.controller.set_max_speed(walker.speed)
            except RuntimeError:
                logger.debug("walker controller could not be started", exc_info=True)

    def _hold_vehicles(self) -> None:
        for vehicle in list(self._vehicles):
            try:
                location = vehicle.get_location()
                x, y = self._scenario_xy(location.x, location.y)
                stopped = vehicle.id in self._held or vehicle.id in self._braked
                if not in_corridor(x, y, self.hold_bounds):
                    if stopped:
                        self._release(vehicle)
                elif in_corridor(x, y, self.bounds):
                    self._brake(vehicle)
                elif not stopped and self._approaching(vehicle, location, x, y):
                    self._hold(vehicle)
            except RuntimeError:
                logger.debug("ambient vehicle %s is gone", vehicle, exc_info=True)
                self._vehicles.remove(vehicle)

    def _approaching(self, vehicle: Any, location: Any, x: float, y: float) -> bool:
        """Whether the vehicle's velocity carries it towards the corridor."""
        velocity = vehicle.get_velocity()
        ahead_x, ahead_y = self._scenario_xy(
            location.x + velocity.x * APPROACH_HORIZON_S,
            location.y + velocity.y * APPROACH_HORIZON_S,
        )
        gain = distance_to_bounds(x, y, self.bounds) - distance_to_bounds(
            ahead_x, ahead_y, self.bounds
        )
        return gain > APPROACH_MIN_GAIN_M

    def _hold(self, vehicle: Any) -> None:
        """Let the Traffic Manager bring the vehicle to a stop, or brake it directly."""
        if self._tm is not None and hasattr(self._tm, "vehicle_percentage_speed_difference"):
            self._tm.vehicle_percentage_speed_difference(vehicle, 100.0)
            self._held.add(vehicle.id)
        else:
            self._brake(vehicle)

    def _brake(self, vehicle: Any) -> None:
        """Take the vehicle off autopilot and apply the brakes fully."""
        if vehicle.id in self._braked:
            return
        carla = self.session.carla
        if hasattr(vehicle, "set_autopilot"):
            vehicle.set_autopilot(False, self._tm_port)
        vehicle.apply_control(carla.VehicleControl(throttle=0.0, brake=1.0, hand_brake=True))
        self._braked.add(vehicle.id)

    def _release(self, vehicle: Any) -> None:
        if vehicle.id in self._braked:
            self._braked.discard(vehicle.id)
            if hasattr(vehicle, "set_autopilot"):
                vehicle.set_autopilot(True, self._tm_port)
        if self._tm is not None and hasattr(self._tm, "vehicle_percentage_speed_difference"):
            self._tm.vehicle_percentage_speed_difference(vehicle, CRUISE_SPEED_DIFFERENCE)
        self._held.discard(vehicle.id)

    def _turn_back_walkers(self) -> None:
        for walker in list(self._walkers):
            try:
                inside = not self._outside(walker.actor.get_location(), self.bounds)
                if inside and not walker.returning:
                    walker.returning = True
                    if walker.controller is not None and self._controllers_started:
                        walker.controller.go_to_location(walker.home)
                elif not inside:
                    walker.returning = False
            except RuntimeError:
                logger.debug("ambient walker %s is gone", walker.actor, exc_info=True)
                self._walkers.remove(walker)
