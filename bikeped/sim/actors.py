"""Spawning and driving the scripted scenario agents.

Each scenario agent becomes one CARLA actor (a walker for ``person`` agents, a
two-wheeled vehicle for ``bike`` agents, or a walker standing in for a cyclist
when the engine has no rideable two-wheeler and ``cyclist_proxy='walker'``).

Two drive modes are available:

``kinematic``
    The actor is placed on the scripted pose every tick. Height comes from a
    ground profile sampled with ``world.ground_projection`` before any actor
    exists (so the ray never hits an agent), yaw from the smoothed heading and,
    for two-wheelers, roll from the lean angle. Walkers keep physics and also
    receive a ``WalkerControl`` so their walk cycle plays.

``velocity``
    Walkers are steered with ``WalkerControl`` (scripted velocity plus a
    proportional correction on position error) and riders with target linear
    and angular velocities plus a light throttle and steer so wheels, pedals and
    handlebar move. An actor is re-placed on its scripted pose only when the
    position error exceeds a tolerance.

Positions reported to the decision pipeline and the ground truth always come
from the script, never from the simulated actor.
"""

from __future__ import annotations

import logging
import math
import random
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

from .engine import Capabilities, attribute_text, is_two_wheeler
from .motion import Trajectory
from .site import Site
from .types import AgentState

log = logging.getLogger(__name__)

WALKER_MODELS = ("adult", "child", "wheelchair", "runner")
RIDER_MODELS = ("bicycle", "road_bike", "ebike", "escooter", "cargo_bike", "moped")

#: Preferred blueprints per two-wheeler hint, in order; the first one present is used.
RIDER_BLUEPRINTS: Dict[str, Tuple[str, ...]] = {
    "bicycle": ("vehicle.bh.crossbike", "vehicle.gazelle.omafiets", "vehicle.diamondback.century"),
    "road_bike": ("vehicle.diamondback.century", "vehicle.bh.crossbike"),
    "ebike": ("vehicle.bh.crossbike", "vehicle.gazelle.omafiets"),
    "cargo_bike": ("vehicle.gazelle.omafiets", "vehicle.bh.crossbike"),
    "escooter": ("vehicle.vespa.zx125", "vehicle.bh.crossbike"),
    "moped": ("vehicle.vespa.zx125", "vehicle.yamaha.yzf", "vehicle.kawasaki.ninja"),
}

DRIVE_MODES = ("kinematic", "velocity")


def _attr(bp: Any, name: str) -> str:
    return attribute_text(bp, name) or ""


def choose_walker(library: Any, model: str, rng: random.Random) -> Tuple[Any, bool]:
    """Pick a walker blueprint for a hint; returns (blueprint, use_wheelchair)."""
    walkers = sorted(library.filter("walker.pedestrian.*"), key=lambda b: b.id)
    if not walkers:
        raise RuntimeError("the server has no walker blueprints")
    adults = [b for b in walkers if _attr(b, "age") not in ("child",)]
    pool = adults or walkers
    wheelchair = False
    if model == "child":
        pool = [b for b in walkers if _attr(b, "age") == "child"] or pool
    elif model == "wheelchair":
        capable = [b for b in walkers if _attr(b, "can_use_wheelchair") == "true"]
        if capable:
            pool, wheelchair = capable, True
    return rng.choice(pool), wheelchair


def choose_rider(library: Any, model: str, rng: random.Random) -> Optional[Any]:
    """Pick a two-wheeler blueprint for a hint, or ``None`` when the engine has none."""
    rideable = sorted((b for b in library.filter("vehicle.*") if is_two_wheeler(b)),
                      key=lambda b: b.id)
    if not rideable:
        return None
    by_id = {b.id: b for b in rideable}
    for bp_id in RIDER_BLUEPRINTS.get(model or "bicycle", RIDER_BLUEPRINTS["bicycle"]):
        if bp_id in by_id:
            return by_id[bp_id]
    bicycles = [b for b in rideable if _attr(b, "base_type") == "bicycle"]
    return rng.choice(bicycles or rideable)


def needs_rideable(scenario: Dict[str, Any]) -> bool:
    return any(a["type"] == "bike" for a in scenario["agents"])


@dataclass
class Agent:
    """One scripted agent and the actor that renders it."""

    trajectory: Trajectory
    kind: str
    model: str
    actor: Any = None
    bp: Any = None
    blueprint: str = ""
    walker: bool = True
    half_height: float = 0.9
    visible: bool = False
    ground: List[Tuple[float, float, float]] = field(default_factory=list)


class Cast:
    """All scripted actors of one scenario.

    ``drive`` is the drive mode for riders and walkers (see module docstring).
    ``cyclist_proxy`` is ``None`` or ``'walker'``.
    """

    def __init__(self, session: Any, site: Site, scenario: Dict[str, Any],
                 trajectories: Sequence[Trajectory], caps: Capabilities, *,
                 drive: str = "kinematic",
                 seed: int = 0, cyclist_proxy: Optional[str] = None,
                 walker_drive: Optional[str] = None) -> None:
        if drive not in DRIVE_MODES:
            raise ValueError(f"drive must be one of {DRIVE_MODES}")
        self.session = session
        self.site = site
        self.scenario = scenario
        self.caps = caps
        self.drive = drive
        self.walker_drive = walker_drive or drive
        self.rng = random.Random(seed)
        self.cyclist_proxy = cyclist_proxy
        self.agents: List[Agent] = []
        self.dt = session.fixed_delta
        self.time = 0.0
        if needs_rideable(scenario) and not caps.has_rideable and cyclist_proxy != "walker":
            raise RuntimeError(
                f"{caps.engine} server {caps.server_version} has no rideable two-wheeler "
                "blueprint, so cyclists cannot be rendered. Pass --cyclist-proxy walker to "
                "render them as walkers.")
        defs = {int(a["id"]): a for a in scenario["agents"]}
        for traj in trajectories:
            a = defs[traj.id]
            model = str(a.get("model") or ("bicycle" if a["type"] == "bike" else "adult"))
            self.agents.append(Agent(trajectory=traj, kind=a["type"], model=model))

    def sample_ground(self, step: float = 0.5) -> None:
        """Sample ground heights along every path before any agent is spawned."""
        world = self.session.world
        carla = self.session.carla
        for agent in self.agents:
            traj = agent.trajectory
            pts: List[Tuple[float, float, float]] = []
            t = traj.start
            while t <= traj.end + 1e-9:
                x, y = traj.position(t)
                pts.append((x, y, self._project(world, carla, x, y)))
                t += step / max(1.0, math.hypot(*traj.velocity(t)) or 1.0)
            agent.ground = pts

    def _project(self, world: Any, carla: Any, x: float, y: float) -> float:
        X, Y, Z = self.site.to_world(x, y)
        if not self.caps.ground_projection:
            return Z
        try:
            hit = world.ground_projection(carla.Location(x=X, y=Y, z=Z + 2.0), 6.0)
        except RuntimeError:
            hit = None
        if hit is None:
            return Z
        return float(hit.location.z)

    def ground_z(self, agent: Agent, x: float, y: float) -> float:
        if not agent.ground:
            return self.site.ground_z
        best = min(agent.ground, key=lambda p: (p[0] - x) ** 2 + (p[1] - y) ** 2)
        return best[2]

    def spawn(self) -> None:
        """Choose the blueprint of every agent; actors appear when their window starts.

        Actors are spawned on their first active tick and destroyed after their
        last one. Parking inactive actors under the map is not an option: walkers
        keep falling with physics disabled and the engine deletes them at the
        kill height.
        """
        library = self.session.world.get_blueprint_library()
        for agent in self.agents:
            bp = None
            if agent.kind == "bike" and self.caps.has_rideable:
                bp = choose_rider(library, agent.model, self.rng)
            if bp is None:
                walker_model = agent.model if agent.model in WALKER_MODELS else "adult"
                bp, wheelchair = choose_walker(library, walker_model, self.rng)
                if wheelchair and bp.has_attribute("use_wheelchair"):
                    bp.set_attribute("use_wheelchair", "true")
                if bp.has_attribute("is_invincible"):
                    bp.set_attribute("is_invincible", "false")
                agent.walker = True
                if agent.kind == "bike":
                    agent.model = f"walker-proxy:{agent.model}"
            else:
                agent.walker = False
            if bp.has_attribute("role_name"):
                bp.set_attribute("role_name", f"scripted_{agent.trajectory.id}")
            agent.bp, agent.blueprint = bp, bp.id

    def _pose_transform(self, agent: Agent, t: float) -> Any:
        carla = self.session.carla
        pose = agent.trajectory.sample(t)
        gz = self.ground_z(agent, pose.x, pose.y)
        X, Y, _ = self.site.to_world(pose.x, pose.y)
        yaw = self.site.heading_to_yaw(pose.heading)
        if agent.walker:
            z = gz + agent.half_height + 0.02
            rot = carla.Rotation(pitch=0.0, yaw=yaw, roll=0.0)
        else:
            z = gz + 0.02
            # Seen from above the world is drawn left-handed, so a counter-clockwise
            # (left) turn in the scenario is a negative yaw rate and a negative roll.
            rot = carla.Rotation(pitch=0.0, yaw=yaw, roll=-math.degrees(pose.lean))
        return carla.Transform(carla.Location(x=X, y=Y, z=z), rot)

    def _show(self, agent: Agent, t: float) -> None:
        target = self._pose_transform(agent, t)
        if agent.actor is None:
            carla = self.session.carla
            actor = None
            for attempt in range(6):
                loc = target.location
                tf = carla.Transform(carla.Location(x=loc.x, y=loc.y, z=loc.z + 0.3 * attempt
                                                    + (0.2 if agent.walker else 0.0)),
                                     target.rotation)
                actor = self.session.try_spawn(agent.bp, tf)
                if actor is not None:
                    break
            if actor is None:
                raise RuntimeError(f"could not spawn {agent.blueprint} for agent "
                                   f"{agent.trajectory.id}")
            agent.actor = actor
            ext = actor.bounding_box.extent
            if agent.walker:
                agent.half_height = float(ext.z) if ext.z > 0.3 else 0.9
            target = self._pose_transform(agent, t)
        physics = agent.walker or self.drive == "velocity"
        agent.actor.set_simulate_physics(physics)
        if hasattr(agent.actor, "set_enable_gravity"):
            agent.actor.set_enable_gravity(physics)
        agent.actor.set_transform(target)
        agent.visible = True

    def _hide(self, agent: Agent) -> None:
        try:
            self.session.destroy(agent.actor)
        except RuntimeError as exc:
            log.debug("destroying agent %d failed: %s", agent.trajectory.id, exc)
        agent.actor = None
        agent.visible = False

    def drive_tick(self, t: float) -> None:
        """Command every actor for the step that ends at ``t``."""
        self.time = t
        for agent in self.agents:
            traj = agent.trajectory
            active = traj.active(t)
            if not active:
                if agent.visible:
                    self._hide(agent)
                continue
            if not agent.visible:
                self._show(agent, t)
                continue
            mode = self.walker_drive if agent.walker else self.drive
            if mode == "kinematic":
                self._drive_kinematic(agent, t)
            else:
                self._drive_velocity(agent, t)

    def _scripted_world_velocity(self, agent: Agent, t: float) -> Tuple[float, float, float]:
        vx, vy = agent.trajectory.velocity(t)
        r, c = self.site.road_axis, self.site.cross_axis
        return vx * r[0] + vy * c[0], vx * r[1] + vy * c[1], math.hypot(vx, vy)

    def _drive_kinematic(self, agent: Agent, t: float) -> None:
        carla = self.session.carla
        tf = self._pose_transform(agent, t)
        agent.actor.set_transform(tf)
        if agent.walker:
            wx, wy, speed = self._scripted_world_velocity(agent, t)
            if speed > 1e-3:
                direction = carla.Vector3D(x=wx / speed, y=wy / speed, z=0.0)
            else:
                direction = tf.get_forward_vector()
            agent.actor.apply_control(carla.WalkerControl(direction, float(speed), False))

    def _drive_velocity(self, agent: Agent, t: float) -> None:
        carla = self.session.carla
        target = self._pose_transform(agent, t)
        current = agent.actor.get_transform()
        ex = target.location.x - current.location.x
        ey = target.location.y - current.location.y
        error = math.hypot(ex, ey)
        tolerance = 0.5 if agent.walker else 0.8
        if error > tolerance:
            agent.actor.set_transform(target)
            return
        wx, wy, speed = self._scripted_world_velocity(agent, t)
        gain = 1.5
        cx, cy = wx + gain * ex, wy + gain * ey
        if agent.walker:
            norm = math.hypot(cx, cy)
            if norm < 1e-3:
                agent.actor.apply_control(carla.WalkerControl(target.get_forward_vector(), 0.0,
                                                              False))
                return
            agent.actor.apply_control(carla.WalkerControl(
                carla.Vector3D(x=cx / norm, y=cy / norm, z=0.0), float(norm), False))
            return
        pose = agent.trajectory.sample(t)
        yaw_err = (target.rotation.yaw - current.rotation.yaw + 180.0) % 360.0 - 180.0
        yaw_rate = -math.degrees(pose.yaw_rate) + 4.0 * yaw_err
        vz = agent.actor.get_velocity().z if hasattr(agent.actor, "get_velocity") else 0.0
        agent.actor.set_target_velocity(carla.Vector3D(x=cx, y=cy, z=min(vz, 0.0)))
        agent.actor.set_target_angular_velocity(carla.Vector3D(x=0.0, y=0.0, z=yaw_rate))
        steer = max(-1.0, min(1.0, yaw_err / 25.0 - pose.yaw_rate * 0.3))
        throttle = 0.35 if speed > 0.3 else 0.0
        agent.actor.apply_control(carla.VehicleControl(throttle=throttle, steer=steer,
                                                       brake=0.0 if speed > 0.3 else 0.5))

    def states(self, t: float, scripted: Dict[int, Dict[str, Any]]) -> List[AgentState]:
        """``AgentState`` of every agent active at ``t`` from the actual actor pose."""
        import decision_testbench as tb

        out: List[AgentState] = []
        for agent in self.agents:
            pos = scripted.get(agent.trajectory.id)
            if pos is None or agent.actor is None:
                continue
            tf = agent.actor.get_transform()
            bb = agent.actor.bounding_box
            ext = (float(bb.extent.x), float(bb.extent.y), float(bb.extent.z))
            centre = tf.transform(bb.location)
            loc = (float(centre.x), float(centre.y), float(centre.z))
            if agent.kind == "bike" and not agent.walker and min(ext[0], ext[2]) < 0.3:
                obj = tb.OBJECT_TYPES["cyclist"]
                ext = (obj.length / 2.0, obj.width / 2.0, obj.height / 2.0)
                gz = self.ground_z(agent, pos["x"], pos["y"])
                loc = (float(tf.location.x), float(tf.location.y), gz + obj.height / 2.0)
            rot = (float(tf.rotation.pitch), float(tf.rotation.yaw), float(tf.rotation.roll))
            sx, sy = self.site.to_scenario(tf.location.x, tf.location.y)
            vx, vy = agent.trajectory.velocity(t)
            out.append(AgentState(
                id=agent.trajectory.id, kind=agent.kind, sx=float(pos["x"]), sy=float(pos["y"]),
                heading=float(pos["heading"]), speed=math.hypot(vx, vy), location=loc,
                rotation=rot, extent=ext, model=agent.model, blueprint=agent.blueprint,
                track_error_m=math.hypot(sx - pos["x"], sy - pos["y"]),
                instance_id=int(agent.actor.id)))
        return out
