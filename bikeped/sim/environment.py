"""Weather and lighting presets.

Each preset is a plain dict of ``carla.WeatherParameters`` fields. Fields a
server does not know are skipped, so the same table serves CARLA 0.9.x and
0.10.0. The sun azimuth is not stored as an absolute angle: ``sun_offset_deg``
is measured from the pole camera's viewing direction (CARLA gives the
azimuth the light travels from, hence the negative offsets), so the pole view is lit
from behind or from the side whatever the orientation of the crosswalk.
"""

from __future__ import annotations

import logging
from typing import Any, Dict

from .engine import carla_module

log = logging.getLogger(__name__)

WEATHER_PRESETS: Dict[str, Dict[str, Any]] = {
    "clear_noon": dict(cloudiness=10.0, precipitation=0.0, precipitation_deposits=0.0,
                       wind_intensity=10.0, sun_altitude_angle=60.0, sun_offset_deg=-30.0,
                       fog_density=0.0, wetness=0.0, night=False),
    "golden_hour": dict(cloudiness=20.0, precipitation=0.0, precipitation_deposits=0.0,
                        wind_intensity=5.0, sun_altitude_angle=16.0, sun_offset_deg=-100.0,
                        fog_density=3.0, fog_distance=60.0, wetness=0.0,
                        mie_scattering_scale=0.06, night=False),
    "overcast": dict(cloudiness=90.0, precipitation=0.0, precipitation_deposits=10.0,
                     wind_intensity=30.0, sun_altitude_angle=45.0, sun_offset_deg=-30.0,
                     fog_density=5.0, fog_distance=80.0, wetness=10.0, night=False),
    "light_rain": dict(cloudiness=80.0, precipitation=30.0, precipitation_deposits=40.0,
                       wind_intensity=30.0, sun_altitude_angle=35.0, sun_offset_deg=-30.0,
                       fog_density=8.0, fog_distance=60.0, wetness=60.0, night=False),
    "heavy_rain": dict(cloudiness=100.0, precipitation=90.0, precipitation_deposits=85.0,
                       wind_intensity=70.0, sun_altitude_angle=30.0, sun_offset_deg=-30.0,
                       fog_density=20.0, fog_distance=30.0, wetness=100.0, night=False),
    "dusk": dict(cloudiness=40.0, precipitation=0.0, precipitation_deposits=0.0,
                 wind_intensity=10.0, sun_altitude_angle=-3.0, sun_offset_deg=-60.0,
                 fog_density=5.0, fog_distance=60.0, wetness=0.0, night=True,
                 exposure_compensation=0.5),
    "clear_night": dict(cloudiness=10.0, precipitation=0.0, precipitation_deposits=0.0,
                        wind_intensity=5.0, sun_altitude_angle=-40.0, sun_offset_deg=-30.0,
                        fog_density=0.0, wetness=0.0, night=True, exposure_compensation=1.0),
    "wet_night": dict(cloudiness=70.0, precipitation=10.0, precipitation_deposits=70.0,
                      wind_intensity=20.0, sun_altitude_angle=-40.0, sun_offset_deg=-30.0,
                      fog_density=6.0, fog_distance=50.0, wetness=90.0, night=True,
                      exposure_compensation=1.3),
    "fog": dict(cloudiness=60.0, precipitation=0.0, precipitation_deposits=10.0,
                wind_intensity=5.0, sun_altitude_angle=25.0, sun_offset_deg=-30.0,
                fog_density=45.0, fog_distance=8.0, fog_falloff=1.0, wetness=20.0,
                night=False),
}

_META_KEYS = ("sun_offset_deg", "night", "exposure_compensation")


def camera_overrides(preset: str) -> Dict[str, float]:
    """Camera attributes a preset needs, such as extra exposure for night scenes."""
    value = WEATHER_PRESETS[preset].get("exposure_compensation")
    return {} if value is None else {"exposure_compensation": float(value)}


def is_night(preset: str) -> bool:
    return bool(WEATHER_PRESETS[preset]["night"])


def settle_ticks(preset: str, fps: int) -> int:
    """Ticks to run after a weather change so auto exposure and wet surfaces settle."""
    preset_values = WEATHER_PRESETS[preset]
    seconds = 2.0 + (1.5 if preset_values.get("wetness", 0.0) > 30.0 or preset_values["night"]
                     else 0.0)
    return int(round(seconds * fps))


def weather_values(preset: str, view_yaw_deg: float) -> Dict[str, float]:
    """``WeatherParameters`` fields for ``preset`` with an absolute sun azimuth."""
    if preset not in WEATHER_PRESETS:
        raise KeyError(f"unknown weather preset '{preset}', choose from "
                       f"{', '.join(WEATHER_PRESETS)}")
    values = {k: float(v) for k, v in WEATHER_PRESETS[preset].items() if k not in _META_KEYS}
    values["sun_azimuth_angle"] = (view_yaw_deg + WEATHER_PRESETS[preset]["sun_offset_deg"]) % 360.0
    return values


def apply_weather(world: Any, preset: str, view_yaw_deg: float) -> Dict[str, float]:
    """Apply a preset field by field; returns the fields that were set."""
    weather = world.get_weather()
    applied: Dict[str, float] = {}
    for key, value in weather_values(preset, view_yaw_deg).items():
        if hasattr(weather, key):
            setattr(weather, key, value)
            applied[key] = value
    world.set_weather(weather)
    set_street_lights(world, is_night(preset))
    return applied


def set_street_lights(world: Any, on: bool) -> bool:
    """Switch street lights through the light manager when the server has one."""
    if not hasattr(world, "get_lightmanager"):
        return False
    try:
        carla = carla_module()
        manager = world.get_lightmanager()
        group = getattr(getattr(carla, "LightGroup", None), "Street", None)
        lights = manager.get_all_lights(group) if group is not None else manager.get_all_lights()
        if on:
            manager.turn_on(lights)
        else:
            manager.turn_off(lights)
        return True
    except (RuntimeError, AttributeError, TypeError) as exc:
        log.warning("street lights unavailable: %s", exc)
        return False
