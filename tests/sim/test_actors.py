"""Blueprint choice for scenario model hints."""

from __future__ import annotations

import random

import fake_carla
import pytest

from bikeped.sim.actors import RIDER_MODELS, choose_rider, choose_walker


@pytest.fixture(scope="module")
def library():
    return fake_carla.make_fake_carla().Client("h", 1).get_world().get_blueprint_library()


def test_same_seed_same_cast(library):
    a = [choose_walker(library, "adult", random.Random(7))[0].id for _ in range(3)]
    b = [choose_walker(library, "adult", random.Random(7))[0].id for _ in range(3)]
    assert a == b


def test_child_and_wheelchair_hints(library):
    bp, wheelchair = choose_walker(library, "child", random.Random(0))
    assert bp.get_attribute("age").as_str() == "child" and not wheelchair
    bp, wheelchair = choose_walker(library, "wheelchair", random.Random(0))
    assert wheelchair


@pytest.mark.parametrize("model", RIDER_MODELS)
def test_every_rider_hint_resolves_to_a_two_wheeler(library, model):
    bp = choose_rider(library, model, random.Random(0))
    assert bp.get_attribute("number_of_wheels").as_int() == 2


def test_road_bike_and_moped(library):
    assert choose_rider(library, "road_bike", random.Random(0)).id == "vehicle.diamondback.century"
    assert choose_rider(library, "moped", random.Random(0)).id == "vehicle.vespa.zx125"


def test_no_rider_on_ue5():
    lib = fake_carla.make_fake_carla(version="0.10.0", two_wheelers=False).Client(
        "h", 1).get_world().get_blueprint_library()
    assert choose_rider(lib, "bicycle", random.Random(0)) is None
