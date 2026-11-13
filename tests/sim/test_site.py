"""Site geometry: the scenario frame mapped onto a CARLA crosswalk and back."""

from __future__ import annotations

import json
import math
from pathlib import Path

import fake_carla
import numpy as np
import pytest

from bikeped.sim.site import Site

ROOT = Path(__file__).resolve().parents[2]
CROSSWALKS = ROOT / "crosswalks.json"


def xy(value) -> np.ndarray:
    """First two components of a point returned by the site, as floats."""
    return np.asarray(value, dtype=np.float64).ravel()[:2]


@pytest.fixture(scope="module")
def entry() -> dict:
    return json.loads(CROSSWALKS.read_text())[8]


@pytest.fixture(scope="module")
def site(entry) -> Site:
    return Site.from_crosswalk(entry)


def world_direction(site: Site, heading: float) -> np.ndarray:
    """World-frame unit vector of a scenario heading, taken from ``to_world`` alone."""
    start = xy(site.to_world(3.0, -2.0))
    end = xy(site.to_world(3.0 + math.cos(heading), -2.0 + math.sin(heading)))
    return end - start


def test_crosswalk_centre_is_scenario_5_0(site, entry):
    assert xy(site.to_world(5.0, 0.0)) == pytest.approx(entry["center"][:2], abs=1e-6)


def test_round_trip(site):
    rng = np.random.default_rng(0)
    for x, y in rng.uniform(-40.0, 40.0, size=(50, 2)):
        world = xy(site.to_world(x, y))
        assert xy(site.to_scenario(world[0], world[1])) == pytest.approx([x, y], abs=1e-9)


def test_mapping_preserves_distances(site):
    rng = np.random.default_rng(1)
    points = rng.uniform(-30.0, 30.0, size=(20, 2))
    world = np.array([xy(site.to_world(x, y)) for x, y in points])
    for i in range(len(points) - 1):
        scenario_distance = np.linalg.norm(points[i + 1] - points[i])
        world_distance = np.linalg.norm(world[i + 1] - world[i])
        assert world_distance == pytest.approx(scenario_distance, abs=1e-9)


def test_pedestrian_axis_runs_along_the_crosswalk(site, entry):
    corners = np.array(entry["corners"], dtype=np.float64)[:, :2]
    edges = [corners[(i + 1) % 4] - corners[i] for i in range(4)]
    long_edge = max(edges, key=np.linalg.norm)
    long_edge = long_edge / np.linalg.norm(long_edge)
    across = world_direction(site, math.pi / 2.0)
    along = world_direction(site, 0.0)
    assert abs(float(across @ long_edge)) > 0.99
    assert abs(float(along @ long_edge)) < 0.15


def test_heading_to_yaw_matches_to_world(site):
    """A CARLA yaw is clockwise seen from above; its forward vector is (cos, sin) in x, y."""
    for heading in np.linspace(-math.pi, math.pi, 17):
        yaw = math.radians(float(site.heading_to_yaw(float(heading))))
        forward = np.array([math.cos(yaw), math.sin(yaw)])
        assert forward == pytest.approx(world_direction(site, float(heading)), abs=1e-9)


def test_heading_to_yaw_agrees_with_unreal_rotation(site):
    carla = fake_carla.make_fake_carla()
    yaw = float(site.heading_to_yaw(math.pi / 2.0))
    forward = carla.Rotation(yaw=yaw).get_forward_vector()
    expected = world_direction(site, math.pi / 2.0)
    assert (forward.x, forward.y) == pytest.approx(tuple(expected), abs=1e-9)
    assert forward.z == pytest.approx(0.0, abs=1e-12)


def test_quarter_turn_keeps_its_sense(site):
    """Turning from +x to +y in the scenario is the same turn for every start heading."""
    def turn(heading: float) -> float:
        a = world_direction(site, heading)
        b = world_direction(site, heading + math.pi / 2.0)
        return float(a[0] * b[1] - a[1] * b[0])

    reference = turn(0.0)
    assert abs(reference) == pytest.approx(1.0, abs=1e-9)
    for heading in (0.3, 1.2, 2.5, -2.0):
        assert turn(heading) == pytest.approx(reference, abs=1e-9)


def test_from_json_matches_from_crosswalk(site):
    other = Site.from_json(CROSSWALKS, 8)
    for point in ((5.0, 0.0), (34.0, 2.0), (5.0, -6.0)):
        assert xy(other.to_world(*point)) == pytest.approx(xy(site.to_world(*point)), abs=1e-9)


def test_from_world_reads_the_live_crosswalk(site):
    carla = fake_carla.make_fake_carla()
    world = carla.Client("127.0.0.1", 2000).get_world()
    live = Site.from_world(world, 0)
    assert xy(live.to_world(5.0, 0.0)) == pytest.approx(xy(site.to_world(5.0, 0.0)), abs=0.05)
    back = xy(live.to_scenario(*xy(live.to_world(12.0, -3.0))))
    assert back == pytest.approx([12.0, -3.0], abs=1e-9)


def test_scene_is_not_mirrored(site):
    """Seen from above (world x right, y down), scenario +x to +y is counter-clockwise."""
    road = world_direction(site, 0.0)
    cross = world_direction(site, math.pi / 2.0)
    assert float(road[0] * cross[1] - road[1] * cross[0]) == pytest.approx(-1.0, abs=1e-9)
