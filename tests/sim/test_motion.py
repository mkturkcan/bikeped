"""Rendering trajectories: exact scripted positions, smooth headings, bounded lean."""

from __future__ import annotations

import math

import numpy as np
import pytest

import decision_testbench as tb
from bikeped.sim.motion import MAX_LEAN_RAD, Trajectory, scripted_agents, wrap_angle


@pytest.mark.parametrize("index", range(len(tb.SCENARIOS)))
def test_positions_match_interpolate_agent(index):
    scenario = tb.SCENARIOS[index]
    fps = 30
    for agent in scenario["agents"]:
        traj = Trajectory(agent, fps)
        for k in range(int(tb.get_scenario_duration(scenario) * fps) + 1):
            t = k / fps
            ref = tb.interpolate_agent(agent, t)
            assert traj.active(t) == (ref is not None)
            if ref is not None:
                assert traj.position(t) == pytest.approx((ref["x"], ref["y"]), abs=1e-9)


def test_heading_is_continuous_through_wrap_around():
    agent = {"type": "bike", "id": 1, "path": [
        {"x": 0, "y": 0, "t": 0}, {"x": -10, "y": 0.5, "t": 2}, {"x": -20, "y": -0.5, "t": 4}]}
    traj = Trajectory(agent, 30)
    previous = traj.heading(0.0)[0]
    for k in range(1, 121):
        h, _ = traj.heading(k / 30)
        assert abs(wrap_angle(h - previous)) < 0.1
        previous = h
    assert abs(wrap_angle(traj.heading(3.9)[0] - math.pi)) < 0.2


def test_heading_turn_is_rate_limited():
    agent = {"type": "bike", "id": 1, "path": [
        {"x": 0, "y": 0, "t": 0}, {"x": 10, "y": 0, "t": 1}, {"x": 10, "y": 10, "t": 2}]}
    traj = Trajectory(agent, 30)
    rates = [abs(traj.heading(k / 30)[1]) for k in range(61)]
    assert max(rates) <= traj.max_turn_rate + 1e-9
    assert traj.heading(2.0)[0] < math.pi / 2


def test_lean_is_clamped_and_follows_the_turn():
    agent = {"type": "bike", "id": 1, "path": [
        {"x": 0, "y": 0, "t": 0}, {"x": 12, "y": 0, "t": 1}, {"x": 12, "y": 12, "t": 2}]}
    traj = Trajectory(agent, 30)
    leans = [traj.sample(k / 30).lean for k in range(61)]
    assert max(abs(v) for v in leans) <= MAX_LEAN_RAD + 1e-12
    assert max(leans) > 0.05


def test_walkers_do_not_lean():
    agent = tb.SCENARIOS[1]["agents"][0]
    assert all(Trajectory(agent).sample(t).lean == 0.0 for t in np.linspace(0, 7, 20))


def test_scripted_agents_matches_testbench():
    scenario = tb.SCENARIOS[1]
    got = scripted_agents(scenario, 2.5)
    assert [a["id"] for a in got] == [1, 2]
    assert got[1]["x"] == pytest.approx(tb.interpolate_agent(scenario["agents"][1], 2.5)["x"])
