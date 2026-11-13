"""Seeded parametric scenario families.

``generate(family, n, seed)`` returns ``n`` scenarios of one family in the usual frame (metres,
``x`` along the road, crosswalk centred on ``(5, 0)``). Sweep families spread their parameter
evenly over the range in :data:`FAMILIES` and use the seed only for small jitter (start times
within 0.2 s, speeds within 5 %), so the sweep values themselves stay readable. The same family,
``n`` and seed always give identical output.

``expected_dominant`` of each generated scenario is computed from the testbench ground truth with
:func:`bikeped.scenarios.evaluate.label_scenario`.
"""

from __future__ import annotations

import math

import numpy as np

from .builders import Path, group
from .extended import bike, ped

FAMILIES = {
    'crossing-angle': 'Pedestrian crossing angle from 30 to 150 degrees to the road',
    'cyclist-speed': 'Cyclist speed from 3 to 11 m/s, arriving as the pedestrian is mid-lane',
    'time-gap': 'Cyclist arrives -3 to +3 s relative to the pedestrian at the conflict point',
    'lateral-offset': 'Cyclist passes 0 to 5 m to the side of the pedestrian',
    'group-size': 'Groups of 1 to n pedestrians crossing together into one cyclist',
    'mixed-traffic': 'Random mix of 1-4 pedestrians and 1-3 cyclists at random times',
}

PED_SPEED = 1.4
BIKE_SPEED = 6.0
T_CONFLICT = 5.0


def _sweep(lo: float, hi: float, n: int) -> list[float]:
    return [lo] if n == 1 else [float(v) for v in np.linspace(lo, hi, n)]


def _cross_through(point: tuple[float, float], angle_deg: float, speed: float, t_at: float,
                   half_len: float = 7.0) -> list[dict]:
    """Pedestrian path crossing through ``point`` at ``angle_deg`` to the road at time ``t_at``."""
    a = math.radians(angle_deg)
    ux, uy = math.cos(a), math.sin(a)
    t0 = t_at - half_len / speed
    # Paths may not start before t = 0; a late conflict time trims the start of the walk instead.
    lead = max(0.0, -t0) * speed
    p = Path.start(point[0] - ux * (half_len - lead), point[1] - uy * (half_len - lead),
                   max(t0, 0.0))
    return p.to(point[0] + ux * half_len, point[1] + uy * half_len, speed).build()


def _ride(speed: float, t_at: float, y: float = 0.0, x_at: float = 5.0) -> list[dict]:
    """Cyclist towards ``-x`` reaching ``(x_at, y)`` at ``t_at``, starting at most 30 m out."""
    x0 = min(x_at + speed * t_at, x_at + 30.0)
    t0 = t_at - (x0 - x_at) / speed
    return Path.start(x0, y, t0).to(x_at - 14.0, y, speed).build()


def _finish(scenarios: list[dict], family: str, seed: int) -> list[dict]:
    from .evaluate import label_scenario

    for i, s in enumerate(scenarios):
        s['id'] = f'{family}-s{seed}-{i:03d}'
        s['name'] = f"{s['name']} [{family} s{seed} #{i}]"
        s['suite'] = f'generated-{family}'
        s['family'] = family
        s.setdefault('tags', []).append('generated')
        s['expected_dominant'] = label_scenario(s)
    return scenarios


def _base(name: str, description: str, agents: list[dict], **extra) -> dict:
    s = {'name': name, 'expected_dominant': 'idle', 'agents': agents,
         'description': description, 'tags': []}
    s.update(extra)
    return s


def generate(family: str, n: int = 20, seed: int = 0) -> list[dict]:
    """``n`` scenarios of ``family`` (a key of :data:`FAMILIES`), deterministic for ``seed``."""
    if family not in FAMILIES:
        raise KeyError(f'unknown family {family!r}; known: {", ".join(FAMILIES)}')
    if n < 1:
        raise ValueError('n must be at least 1')
    rng = np.random.default_rng(seed)

    def jt() -> float:
        return float(rng.uniform(-0.2, 0.2))

    def js(v: float) -> float:
        return v * float(rng.uniform(0.95, 1.05))

    out: list[dict] = []
    if family == 'crossing-angle':
        for ang in _sweep(30.0, 150.0, n):
            out.append(_base(
                f'Crossing at {ang:.0f} deg',
                f'A pedestrian crosses at {ang:.0f} degrees to the road and reaches the lane '
                f'centre as a {BIKE_SPEED:.0f} m/s cyclist arrives.',
                [ped(1, _cross_through((5.0, 0.0), ang, js(PED_SPEED), T_CONFLICT + jt())),
                 bike(2, _ride(js(BIKE_SPEED), T_CONFLICT + jt()))]))
    elif family == 'cyclist-speed':
        for v in _sweep(3.0, 11.0, n):
            out.append(_base(
                f'Cyclist at {v:.1f} m/s',
                f'A cyclist at {v:.1f} m/s reaches the crosswalk as the pedestrian is mid-lane.',
                [ped(1, _cross_through((5.0, 0.0), 90.0, js(PED_SPEED), T_CONFLICT + jt())),
                 bike(2, _ride(js(v), T_CONFLICT + jt()),
                      model='ebike' if v > 8.5 else 'bicycle')],
                vehicle='ebike' if v > 8.5 else 'bicycle'))
    elif family == 'time-gap':
        for gap in _sweep(-3.0, 3.0, n):
            out.append(_base(
                f'Time gap {gap:+.1f} s',
                f'The cyclist reaches the conflict point {abs(gap):.1f} s '
                f'{"after" if gap >= 0 else "before"} the pedestrian.',
                [ped(1, _cross_through((5.0, 0.0), 90.0, js(PED_SPEED), T_CONFLICT)),
                 bike(2, _ride(js(BIKE_SPEED), T_CONFLICT + gap + 0.5 * jt()))]))
    elif family == 'lateral-offset':
        for off in _sweep(0.0, 5.0, n):
            out.append(_base(
                f'Lateral offset {off:.1f} m',
                f'The cyclist passes {off:.1f} m to the side of the pedestrian at the moment '
                f'of crossing.',
                [ped(1, _cross_through((5.0, 0.0), 90.0, js(PED_SPEED), T_CONFLICT + jt())),
                 bike(2, _ride(js(BIKE_SPEED), T_CONFLICT, y=off))]))
    elif family == 'group-size':
        for k in range(1, n + 1):
            lead = _cross_through((5.0, 0.0), 90.0, PED_SPEED, T_CONFLICT)
            offsets = [(float(rng.uniform(-1.2, 1.2)), -0.8 * (i // 2), 0.15 * i + abs(0.1 * jt()))
                       for i in range(k)]
            out.append(_base(
                f'Group of {k}',
                f'{k} pedestrian{"s" if k > 1 else ""} cross together into the path of one '
                f'cyclist.',
                [*[ped(i + 1, p) for i, p in enumerate(group(lead, offsets))],
                 bike(k + 1, _ride(js(BIKE_SPEED), T_CONFLICT + jt()))]))
    else:
        for _ in range(n):
            n_p, n_b = int(rng.integers(1, 5)), int(rng.integers(1, 4))
            agents = []
            for i in range(n_p):
                agents.append(ped(i + 1, _cross_through(
                    (float(rng.uniform(2, 8)), float(rng.uniform(-2, 2))),
                    float(rng.uniform(55, 125)) + (180.0 if rng.random() < 0.5 else 0.0),
                    float(rng.uniform(1.1, 1.6)), float(rng.uniform(4, 9)))))
            for j in range(n_b):
                agents.append(bike(n_p + j + 1, _ride(float(rng.uniform(4, 8)),
                                                      float(rng.uniform(3, 10)),
                                                      y=float(rng.uniform(-2, 2)))))
            out.append(_base(
                f'Mixed traffic {n_p}P {n_b}B',
                f'{n_p} pedestrian{"s" if n_p > 1 else ""} and {n_b} '
                f'cyclist{"s" if n_b > 1 else ""} at random times, angles and speeds.',
                agents))
    return _finish(out, family, seed)
