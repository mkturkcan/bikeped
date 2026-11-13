"""Path builders: monotone time, the speeds they claim, the geometry they promise."""

from __future__ import annotations

import math

import numpy as np
import pytest

from bikeped.scenarios.builders import (
    Path,
    catmull_rom,
    path_accelerations,
    path_speeds,
    platoon,
    straight,
)


def _times(path):
    return [p['t'] for p in path]


def test_straight_speed_and_time():
    p = straight(34, 1, -6, 1, 6.0)
    assert np.allclose(path_speeds(p), 6.0, atol=0.05)
    assert all(b > a for a, b in zip(_times(p), _times(p)[1:]))


def test_arc_constant_speed_and_radius():
    p = Path.start(20, 0, heading=180).arc(5, -90, speed=4.0).build()
    assert np.allclose(path_speeds(p), 4.0, rtol=0.03)
    cx, cy = 20, 5
    for q in p:
        assert math.hypot(q['x'] - cx, q['y'] - cy) == pytest.approx(5, abs=0.01)
    assert p[-1]['x'] == pytest.approx(15, abs=0.01)


def test_brake_to_stops_at_target():
    p = Path.start(30, 0, heading=180, speed=6.0).brake_to(x=21).build()
    assert p[-1]['x'] == pytest.approx(21, abs=0.01)
    assert path_speeds(p)[-1] < 1.0
    assert path_accelerations(p).max() == pytest.approx(2.0, rel=0.1)


def test_accelerate_reaches_speed():
    p = Path.start(0, 0).accelerate(5.0, 1.5).ride(5).build()
    assert path_speeds(p)[-1] == pytest.approx(5.0, rel=0.02)
    assert path_accelerations(p).max() <= 1.6


def test_wait_appear_and_lane_change():
    p = Path.start(5, -6).appear(2.0).wait(1.0).to(5, 0, 1.2).build()
    assert p[0]['t'] == 2.0 and p[1]['t'] == 3.0
    lc = Path.start(30, 0, heading=180).lane_change(2.0, 12, 5.0).build()
    assert lc[-1]['y'] == pytest.approx(-2.0, abs=0.01)
    assert all(b > a for a, b in zip(_times(lc), _times(lc)[1:]))


def test_spline_and_zigzag_monotone():
    p = Path.start(0, 0).through([(5, 2), (10, -1), (15, 0)], 3.0).build()
    assert np.allclose(path_speeds(p), 3.0, rtol=0.05)
    z = Path.start(30, 0, heading=180).zigzag(20, 1.0, 10, 5.0).build()
    assert max(abs(q['y']) for q in z) == pytest.approx(1.0, abs=0.05)
    assert catmull_rom([(0, 0), (1, 1)]).shape[1] == 2


def test_platoon_deterministic():
    base = straight(30, 0, -5, 0, 5.0)
    assert platoon(base, 4, 1.0, jitter_m=0.3, seed=2) == platoon(base, 4, 1.0, jitter_m=0.3,
                                                                  seed=2)
    riders = platoon(base, 3, 1.5)
    assert [r[0]['t'] for r in riders] == [0.0, 1.5, 3.0]
