"""Validation of scenario dicts.

:func:`validate_scenario` checks one scenario against the testbench format and the library
metadata; :func:`validate_suites` also checks that ids and names are unique across scenarios.
Both return a list of human-readable problems, empty when everything is fine.

Speed and acceleration limits are per agent model and deliberately generous: they catch typos
(a pedestrian at 15 m/s, a waypoint with the wrong time) rather than judge realism. Speeds are
segment speeds of the piecewise-linear path in m/s; accelerations are velocity changes between
consecutive segments in m/s^2, so they include turning.
"""

from __future__ import annotations

import math
from typing import Iterable

from .builders import path_accelerations, path_speeds

STATES = ('idle', 'safe', 'warning', 'alert')
VEHICLES = ('bicycle', 'ebike', 'aashto')
AGENT_TYPES = ('person', 'bike')
WEATHER = ('clear_noon', 'golden_hour', 'overcast', 'light_rain', 'heavy_rain', 'dusk',
           'clear_night', 'wet_night', 'fog')
STATIC_SUITES = ('conformance', 'extended', 'stress')

PERSON_MODELS = ('adult', 'child', 'wheelchair', 'runner')
BIKE_MODELS = ('bicycle', 'road_bike', 'ebike', 'escooter', 'cargo_bike', 'moped')

# model -> (max speed m/s, max acceleration m/s^2). ``None`` is an agent without a model hint.
LIMITS: dict[tuple[str, str | None], tuple[float, float]] = {
    ('person', None): (5.5, 8.0),
    ('person', 'adult'): (3.0, 8.0),
    ('person', 'child'): (3.5, 8.0),
    ('person', 'runner'): (5.5, 8.0),
    ('person', 'wheelchair'): (2.0, 4.0),
    ('bike', None): (14.0, 12.0),
    ('bike', 'bicycle'): (10.0, 8.0),
    ('bike', 'road_bike'): (14.0, 8.0),
    ('bike', 'ebike'): (13.0, 10.0),
    ('bike', 'escooter'): (9.0, 8.0),
    ('bike', 'cargo_bike'): (7.0, 6.0),
    ('bike', 'moped'): (16.0, 10.0),
}

# Conformance paths are coarse (1 s waypoints) and some start at full speed mid-path, so the
# acceleration check uses a looser bound for them; it still flags corrupted waypoints.
COARSE_ACCEL_FACTOR = 3.0


def _known_families() -> set[str]:
    from .conformance import FAMILIES as CONF
    from .extended import FAMILIES as EXT
    from .generators import FAMILIES as GEN

    return set(CONF) | set(EXT) | set(GEN) | {'dense'}


def _finite(v) -> bool:
    return isinstance(v, (int, float)) and not isinstance(v, bool) and math.isfinite(v)


def validate_scenario(s: dict) -> list[str]:
    """Problems found in one scenario (empty list when valid)."""
    problems: list[str] = []
    label = s.get('id') or s.get('name') or '<unnamed>'

    def bad(msg: str) -> None:
        problems.append(f'{label}: {msg}')

    if not isinstance(s.get('name'), str) or not s.get('name'):
        bad('missing name')
    if s.get('expected_dominant') not in STATES:
        bad(f"expected_dominant {s.get('expected_dominant')!r} not one of {STATES}")
    if 'vehicle' in s and s['vehicle'] not in VEHICLES:
        bad(f"unknown vehicle {s['vehicle']!r}")
    if 'suite' in s and not isinstance(s['suite'], str):
        bad('suite must be a string')
    if 'family' in s and s['family'] not in _known_families():
        bad(f"unknown family {s['family']!r}")
    if 'tags' in s and not all(isinstance(t, str) for t in s['tags']):
        bad('tags must be strings')
    weather = (s.get('render') or {}).get('weather')
    if weather is not None and weather not in WEATHER:
        bad(f'unknown weather preset {weather!r}')
    agents = s.get('agents')
    if not isinstance(agents, list) or not agents:
        bad('needs a non-empty agents list')
        return problems
    coarse = s.get('suite') == 'conformance'
    ids = [a.get('id') for a in agents]
    if len(set(ids)) != len(ids):
        bad(f'duplicate agent ids {ids}')
    for a in agents:
        who = f"agent {a.get('id')}"
        if a.get('type') not in AGENT_TYPES:
            bad(f"{who}: type {a.get('type')!r} not one of {AGENT_TYPES}")
            continue
        model = a.get('model')
        allowed = PERSON_MODELS if a['type'] == 'person' else BIKE_MODELS
        if model is not None and model not in allowed:
            bad(f'{who}: model {model!r} not valid for a {a["type"]}')
            continue
        path = a.get('path')
        if not isinstance(path, list) or len(path) < 2:
            bad(f'{who}: path needs at least two waypoints')
            continue
        if not all(isinstance(p, dict) and all(_finite(p.get(k)) for k in 'xyt') for p in path):
            bad(f'{who}: every waypoint needs finite x, y, t')
            continue
        ts = [p['t'] for p in path]
        if any(t1 <= t0 for t0, t1 in zip(ts, ts[1:])):
            bad(f'{who}: waypoint times must be strictly increasing')
            continue
        if ts[0] < 0:
            bad(f'{who}: negative start time')
        vmax, amax = LIMITS[(a['type'], model)]
        speeds = path_speeds(path)
        if speeds.max() > vmax + 1e-6:
            bad(f'{who}: speed {speeds.max():.2f} m/s exceeds {vmax} m/s for '
                f'{model or a["type"]}')
        acc = path_accelerations(path)
        limit = amax * (COARSE_ACCEL_FACTOR if coarse else 1.0)
        if len(acc) and acc.max() > limit + 1e-6:
            bad(f'{who}: acceleration {acc.max():.2f} m/s^2 exceeds {limit} m/s^2')
    return problems


def validate_suites(scenarios: Iterable[dict]) -> list[str]:
    """Validate every scenario and check that ids and names are unique among them."""
    problems: list[str] = []
    seen_ids: dict[str, str] = {}
    seen_names: dict[str, str] = {}
    for s in scenarios:
        problems.extend(validate_scenario(s))
        where = f"{s.get('suite', '?')}"
        sid = s.get('id')
        if not sid:
            problems.append(f"{s.get('name')}: missing id")
        elif sid in seen_ids:
            problems.append(f'duplicate id {sid!r} in {seen_ids[sid]} and {where}')
        else:
            seen_ids[sid] = where
        name = s.get('name')
        if name in seen_names:
            problems.append(f'duplicate name {name!r} in {seen_names[name]} and {where}')
        else:
            seen_names[name] = where
    return problems
