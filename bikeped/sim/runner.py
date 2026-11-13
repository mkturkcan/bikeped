"""Render one scenario end to end.

:func:`render_scenario` builds the trajectories, connects a :class:`Session`,
places the site, applies weather, spawns the cast and the camera rigs, warms
up, and then for every tick drives the actors, steps the world, collects the
frame-synchronised images, reads the simulated agent states, runs the decision
pipeline on the scripted positions exactly as the testbench does, attaches the
ground-truth danger and hands a :class:`FrameRecord` to the recorder.
"""

from __future__ import annotations

import logging
import math
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, Optional, Sequence

import numpy as np

from .actors import Cast
from .engine import PROFILES, probe
from .environment import apply_weather, camera_overrides, is_night, settle_ticks
from .motion import scripted_agents, trajectories
from .rigs import make_rig
from .session import SensorHub, Session
from .site import Site
from .types import FrameRecord, RunInfo

log = logging.getLogger(__name__)

ROOT = Path(__file__).resolve().parents[2]


@dataclass
class RenderOptions:
    """Everything ``render`` accepts besides the scenario itself."""

    host: str = "127.0.0.1"
    port: int = 2000
    engine: str = "auto"
    map: Optional[str] = None
    crosswalk: int = 4
    crosswalks_file: Optional[str] = None
    rigs: Sequence[str] = ("pole",)
    weather: Optional[str] = None
    drive: Optional[str] = None
    ambient: str = "none"
    quality: str = "standard"
    layout: str = "single"
    overlay: bool = True
    save_frames: str = "none"
    out: str = "renders"
    seed: int = 0
    fps: int = 30
    pipeline_config: str = "default"
    cyclist_proxy: Optional[str] = None
    ground_truth: bool = True
    warmup_s: float = 1.0
    max_frames: Optional[int] = None
    extra: Dict[str, Any] = field(default_factory=dict)


def resolve_scenario(ref: str, suite: Optional[str] = None) -> Dict[str, Any]:
    """Look a scenario up by slug, name, ``suite:index`` or conformance index."""
    try:
        from bikeped.scenarios import get_scenario
    except ImportError:
        get_scenario = None
    if get_scenario is not None:
        key = f"{suite}:{ref}" if suite and ref.isdigit() else ref
        try:
            return get_scenario(key)
        except (KeyError, ValueError, IndexError):
            if not ref.isdigit():
                raise
    import decision_testbench as tb

    if ref.split(":")[-1].isdigit():
        return tb.SCENARIOS[int(ref.split(":")[-1])]
    for sc in tb.SCENARIOS:
        if sc["name"] == ref:
            return sc
    raise KeyError(f"unknown scenario '{ref}'")


def scenario_slug(scenario: Dict[str, Any]) -> str:
    if scenario.get("id"):
        return str(scenario["id"])
    return "".join(c if c.isalnum() else "-" for c in scenario["name"].lower()).strip("-")


def default_recorder_factory() -> Callable[..., Any]:
    from .recorder import Recorder

    return Recorder


def danger_for(scenario: Dict[str, Any], t: float) -> Optional[Dict[str, Any]]:
    """Ground-truth danger at ``t`` as a JSON-friendly dict."""
    try:
        from .groundtruth import danger_record
        return danger_record(scenario, t)
    except ImportError:
        import decision_testbench as tb

        d = tb.compute_ground_truth_danger(scenario, t)
        return {k: (float(v) if isinstance(v, (float, np.floating)) else v)
                for k, v in vars(d).items()}


def clear_near_field(world: Any, carla: Any, cam: Any, radius_m: float = 7.5) -> list:
    """Hide street furniture poles standing right in front of the pole camera.

    The camera is mounted on a pole of its own; a lamp post a few metres in
    front of it would cover a large part of the image. Returns the ids of the
    disabled objects so they can be enabled again.
    """
    label = getattr(getattr(carla, "CityObjectLabel", None), "Poles", None)
    if label is None or not hasattr(world, "get_environment_objects"):
        return []
    px, py, _ = cam.position
    fx, fy = cam.forward[0], cam.forward[1]
    ids = []
    for obj in world.get_environment_objects(label):
        loc = obj.transform.location
        dx, dy = loc.x - px, loc.y - py
        if math.hypot(dx, dy) < radius_m and dx * fx + dy * fy > -0.5:
            ids.append(obj.id)
    if ids:
        world.enable_environment_objects(set(ids), False)
    return ids


def _load_site(session: Session, opts: RenderOptions) -> Site:
    path = Path(opts.crosswalks_file) if opts.crosswalks_file else ROOT / "crosswalks.json"
    current = session.world.get_map().name.split("/")[-1]
    json_map = PROFILES["ue4"].default_map
    if path.exists() and current == json_map and not opts.extra.get("live_crosswalks"):
        site = Site.from_json(path, opts.crosswalk)
    else:
        site = Site.from_world(session.world, opts.crosswalk)
    return site


def render_scenario(scenario: Dict[str, Any], opts: RenderOptions,
                    recorder_factory: Optional[Callable[..., Any]] = None) -> Dict[str, Any]:
    """Render ``scenario`` on a running server; returns the recorder's summary."""
    import decision_testbench as tb

    fps = opts.fps
    dt = 1.0 / fps
    trajs = trajectories(scenario, fps)
    duration = tb.get_scenario_duration(scenario)
    n_frames = int(math.floor(duration * fps + 1e-9)) + 1
    if opts.max_frames is not None:
        n_frames = min(n_frames, opts.max_frames)
    weather = opts.weather or scenario.get("render", {}).get("weather") or "clear_noon"
    recorder_factory = recorder_factory or default_recorder_factory()

    with Session(opts.host, opts.port, map_name=opts.map, fixed_delta=dt,
                 seed=opts.seed) as session:
        if opts.engine != "auto" and opts.engine != session.engine:
            raise RuntimeError(f"--engine {opts.engine} requested but the server is "
                               f"{session.engine}")
        caps = probe(session.client)
        drive = opts.drive or PROFILES[session.engine].default_drive
        site = _load_site(session, opts)
        cast = Cast(session, site, scenario, trajs, caps, drive=drive, seed=opts.seed,
                    cyclist_proxy=opts.cyclist_proxy, walker_drive=opts.extra.get("walker_drive"))
        cast.sample_ground()
        view_yaw = site.heading_to_yaw(math.pi / 2.0)
        apply_weather(session.world, weather, view_yaw)
        ambient = None
        if opts.ambient != "none":
            from .ambient import Ambient
            ambient = Ambient(session, site, scenario, level=opts.ambient, seed=opts.seed,
                              night=is_night(weather))
            ambient.spawn()
        cast.spawn()
        hub = SensorHub(timeout=30.0)
        rigs = [make_rig(name, site, scenario, opts.quality,
                         ground_truth=opts.ground_truth and name == "pole",
                         trajectories=trajs, fps=fps) for name in opts.rigs]
        for rig in rigs:
            rig.post_overrides = camera_overrides(weather)
            rig.spawn(session, hub, caps)
        if "pole" in opts.rigs:
            pole_cam = next(r for r in rigs if r.name == "pole").camera(0.0)
            session.hidden_objects.update(
                clear_near_field(session.world, session.carla, pole_cam))
        warm = max(int(opts.warmup_s * fps), settle_ticks(weather, fps))
        for _ in range(warm):
            if ambient is not None:
                ambient.tick()
            frame_id = session.tick()
            hub.collect(frame_id)

        run = RunInfo(
            scenario_name=scenario["name"], scenario_id=scenario_slug(scenario),
            suite=str(scenario.get("suite", "conformance")), fps=fps, duration_s=duration,
            engine=session.engine, server_version=caps.server_version,
            client_version=caps.client_version, map=caps.map, crosswalk_id=opts.crosswalk,
            weather=weather, quality=opts.quality, drive=drive, ambient=opts.ambient,
            seed=opts.seed, views=list(opts.rigs), site=site.to_dict(),
            pipeline_config=opts.pipeline_config,
            extra={"cyclist_proxy": opts.cyclist_proxy, **opts.extra})
        recorder = recorder_factory(Path(opts.out), run, views=list(opts.rigs),
                                    overlay=opts.overlay,
                                    layout=opts.layout, save_frames=opts.save_frames,
                                    scenario=scenario)
        pipeline = tb.make_pipeline(tb.PIPELINE_CONFIGS[opts.pipeline_config], fps,
                                    profile=tb.get_braking_profile(scenario))
        cam_x, cam_y = site.pole_xy
        started = time.monotonic()
        try:
            for index in range(n_frames):
                t = index * dt
                cast.drive_tick(t)
                for rig in rigs:
                    rig.move(session, t)
                if ambient is not None:
                    ambient.tick()
                frame_id = session.tick()
                data = hub.collect(frame_id)
                agents_now = scripted_agents(scenario, t)
                state = tb.run_pipeline_on_agents(pipeline, agents_now, t, cam_x=cam_x,
                                                  cam_y=cam_y)
                images: Dict[str, np.ndarray] = {}
                aux: Dict[str, np.ndarray] = {}
                cameras = {}
                for rig in rigs:
                    img, extra_aux = rig.frame(data, t)
                    images[rig.name] = img
                    aux.update(extra_aux)
                    cameras[rig.name] = rig.camera(t)
                frame = FrameRecord(
                    index=index, t=t,
                    agents=cast.states(t, {a["id"]: a for a in agents_now}),
                    state=state, states={opts.pipeline_config: state},
                    alert_pair=_alert_pair(pipeline, state), danger=danger_for(scenario, t),
                    cameras=cameras, images=images, aux=aux)
                recorder.write(frame)
            summary = recorder.close()
        except BaseException:
            try:
                recorder.close()
            except Exception as exc:  # noqa: BLE001
                log.debug("recorder close after failure: %s", exc)
            raise
        finally:
            if ambient is not None:
                ambient.stop()
        summary["seconds"] = round(time.monotonic() - started, 1)
        summary["run"] = run.to_dict()
        return summary


def _alert_pair(pipeline: Any, state: str) -> Optional[tuple]:
    pair = getattr(pipeline, "alert_pair", None) or getattr(pipeline, "last_alert_pair", None)
    if state == "alert" and pair:
        return tuple(int(v) for v in pair[:2])
    return None
