"""The full render loop against the fake UE4 and UE5 servers."""

from __future__ import annotations

import fake_carla
import pytest

import decision_testbench as tb
from bikeped.sim import engine
from bikeped.sim.runner import RenderOptions, render_scenario, resolve_scenario

SCENARIO = tb.SCENARIOS[1]


class StubRecorder:
    """Collects frame records instead of writing videos."""

    instances: list = []

    def __init__(self, out_dir, run, *, views, fail_at=None, **kwargs):
        self.run, self.views, self.frames = run, views, []
        self.fail_at = fail_at
        self.closed = False
        StubRecorder.instances.append(self)

    def write(self, frame):
        if self.fail_at is not None and frame.index == self.fail_at:
            raise RuntimeError("recorder failure")
        self.frames.append(frame)

    def close(self):
        self.closed = True
        return {"run_dir": "", "videos": [], "frames": len(self.frames)}


def failing_recorder(*args, **kwargs):
    return StubRecorder(*args, fail_at=3, **kwargs)


def fake(version="0.9.16", two_wheelers=True):
    carla = fake_carla.make_fake_carla(version=version, two_wheelers=two_wheelers)
    engine.set_carla_module(carla)
    return carla


@pytest.fixture(autouse=True)
def reset():
    StubRecorder.instances = []
    yield
    engine.set_carla_module(None)


def options(**kw):
    base = dict(quality="draft", rigs=("pole",), warmup_s=0.1, out="unused")
    base.update(kw)
    return RenderOptions(**base)


def reference_states(scenario, fps=30):
    pipeline = tb.make_pipeline(tb.PIPELINE_CONFIGS["default"], fps,
                                profile=tb.get_braking_profile(scenario))
    out = []
    n = int(tb.get_scenario_duration(scenario) * fps + 1e-9) + 1
    for k in range(n):
        t = k / fps
        agents = [p for p in (tb.interpolate_agent(a, t) for a in scenario["agents"]) if p]
        out.append(tb.run_pipeline_on_agents(pipeline, agents, t, cam_x=tb.FisheyeCamera.cam_x,
                                             cam_y=tb.FisheyeCamera.cam_y))
    return out


def assert_clean(carla):
    world = carla.server.world
    assert not [a for a in world.get_actors() if a.is_alive and a.type_id != "spectator"]
    assert world.get_settings().synchronous_mode is False


def test_render_on_ue4_matches_the_testbench():
    carla = fake()
    summary = render_scenario(SCENARIO, options(), recorder_factory=StubRecorder)
    rec = StubRecorder.instances[0]
    assert summary["frames"] == len(rec.frames) == len(reference_states(SCENARIO))
    assert [f.state for f in rec.frames] == reference_states(SCENARIO)
    frame = rec.frames[60]
    assert frame.images["pole"].shape == (540, 960, 3)
    bike = next(a for a in frame.agents if a.kind == "bike")
    assert bike.extent[0] > 0.5 and bike.blueprint.startswith("vehicle.")
    assert bike.instance_id is not None
    assert rec.closed
    assert_clean(carla)


def test_cleanup_when_the_recorder_fails():
    carla = fake()
    with pytest.raises(RuntimeError, match="recorder failure"):
        render_scenario(SCENARIO, options(), recorder_factory=failing_recorder)
    assert_clean(carla)


def test_ue5_refuses_cyclists_without_proxy():
    carla = fake("0.10.0", two_wheelers=False)
    with pytest.raises(RuntimeError, match="cyclist-proxy"):
        render_scenario(SCENARIO, options(), recorder_factory=StubRecorder)
    assert_clean(carla)


def test_ue5_walker_proxy():
    carla = fake("0.10.0", two_wheelers=False)
    render_scenario(SCENARIO, options(cyclist_proxy="walker", max_frames=20),
                    recorder_factory=StubRecorder)
    frame = StubRecorder.instances[0].frames[-1]
    bike = next(a for a in frame.agents if a.kind == "bike")
    assert bike.model.startswith("walker-proxy:")
    assert bike.blueprint.startswith("walker.")
    assert StubRecorder.instances[0].run.engine == "ue5"
    assert_clean(carla)


@pytest.mark.parametrize("drive", ["kinematic", "velocity"])
def test_all_views_and_drive_modes(drive):
    carla = fake()
    render_scenario(SCENARIO, options(rigs=("pole", "fisheye", "chase", "orbit", "street", "bev"),
                                      drive=drive, max_frames=5),
                    recorder_factory=StubRecorder)
    frame = StubRecorder.instances[0].frames[-1]
    assert set(frame.images) == {"pole", "fisheye", "chase", "orbit", "street", "bev"}
    assert "pole/depth" in frame.aux
    assert_clean(carla)


def test_resolve_scenario_by_index_and_name():
    assert resolve_scenario("conformance:1")["name"] == SCENARIO["name"]
    assert resolve_scenario(SCENARIO["name"])["name"] == SCENARIO["name"]
