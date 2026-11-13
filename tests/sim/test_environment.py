"""Weather presets."""

from __future__ import annotations

import fake_carla
import pytest

from bikeped.sim import engine
from bikeped.sim.environment import (
    WEATHER_PRESETS,
    apply_weather,
    is_night,
    settle_ticks,
    weather_values,
)

SPEC_PRESETS = {"clear_noon", "golden_hour", "overcast", "light_rain", "heavy_rain", "dusk",
                "clear_night", "wet_night", "fog"}


def test_all_spec_presets_exist():
    assert set(WEATHER_PRESETS) == SPEC_PRESETS


def test_sun_azimuth_follows_the_view():
    a = weather_values("clear_noon", 0.0)["sun_azimuth_angle"]
    b = weather_values("clear_noon", 90.0)["sun_azimuth_angle"]
    assert (b - a) % 360.0 == pytest.approx(90.0)


def test_unknown_preset():
    with pytest.raises(KeyError):
        weather_values("blizzard", 0.0)


def test_night_and_settle():
    assert is_night("wet_night") and not is_night("clear_noon")
    assert settle_ticks("wet_night", 30) > settle_ticks("clear_noon", 30) > 0


def test_apply_weather_sets_known_fields_only():
    carla = fake_carla.make_fake_carla()
    engine.set_carla_module(carla)
    try:
        world = carla.Client("127.0.0.1", 2000).get_world()
        applied = apply_weather(world, "wet_night", 45.0)
        assert world.get_weather().wetness == pytest.approx(90.0)
        assert "night" not in applied and "sun_offset_deg" not in applied
    finally:
        engine.set_carla_module(None)
