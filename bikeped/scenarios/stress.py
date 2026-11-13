"""The ``stress`` suite: dense scenes for throughput and tracker-load testing.

Each scene puts many pedestrians and cyclists in view at once (12 to 24 agents) in the usual
scenario frame (metres, ``x`` along the road, crosswalk centred on ``(5, 0)``). Positions and
start times are jittered with fixed seeds, so the suite is identical on every run.
``expected_dominant`` is the ground-truth label from
:func:`bikeped.scenarios.evaluate.ground_truth_label`.
"""

from __future__ import annotations

import numpy as np

from .builders import Path
from .extended import bike, ped


def _scene(name: str, expected: str, description: str, agents: list[dict],
           tags: tuple[str, ...] = (), weather: str | None = None) -> dict:
    from .conformance import slugify

    s = {'name': name, 'expected_dominant': expected, 'agents': agents, 'id': slugify(name),
         'suite': 'stress', 'family': 'dense', 'description': description,
         'tags': ['dense', *tags]}
    if weather:
        s['render'] = {'weather': weather}
    return s


def _crowd(rng: np.random.Generator, n: int, first_id: int, t_span: float,
           x_lo: float = 3.5, x_hi: float = 6.5, both_ways: bool = True) -> list[dict]:
    out = []
    for i in range(n):
        x = rng.uniform(x_lo, x_hi)
        t0 = rng.uniform(0, t_span)
        v = rng.uniform(1.1, 1.6)
        if both_ways and i % 2:
            path = Path.start(x, 6.5 + rng.uniform(0, 1.0), t0).to(x, -6.5, v).build()
        else:
            path = Path.start(x, -6.5 - rng.uniform(0, 1.0), t0).to(x, 6.5, v).build()
        out.append(ped(first_id + i, path))
    return out


def _riders(rng: np.random.Generator, n: int, first_id: int, t_span: float,
            speed: tuple[float, float] = (4.5, 7.0), both_ways: bool = True) -> list[dict]:
    out = []
    for i in range(n):
        v = rng.uniform(*speed)
        t0 = rng.uniform(0, t_span)
        if both_ways and i % 3 == 2:
            path = Path.start(-24, rng.uniform(-2.5, -0.5), t0, heading=0).ride(54, v).build()
        else:
            path = Path.start(34, rng.uniform(0.3, 2.5), t0, heading=180).ride(54, v).build()
        out.append(bike(first_id + i, path))
    return out


def stress_suite() -> list[dict]:
    """Eight dense scenes, deterministic."""
    scenes = []
    rng = np.random.default_rng(101)
    scenes.append(_scene(
        'Rush Hour Crosswalk', 'alert',
        'Twelve pedestrians cross in both directions over 8 s while six cyclists ride through '
        'from both ends.',
        [*_crowd(rng, 12, 1, 8.0), *_riders(rng, 6, 13, 8.0)], ('rush-hour',)))
    rng = np.random.default_rng(102)
    scenes.append(_scene(
        'Stadium Exit', 'alert',
        'Twenty people leave an event and flood the crosswalk in one direction as three '
        'cyclists arrive.',
        [*_crowd(rng, 20, 1, 6.0, x_lo=2.5, x_hi=7.5, both_ways=False),
         *_riders(rng, 3, 21, 6.0, both_ways=False)], ('crowd',)))
    rng = np.random.default_rng(103)
    scenes.append(_scene(
        'Bike Commute Peak', 'alert',
        'Sixteen cyclists in a steady stream at commuter speeds and four pedestrians picking '
        'their way across.',
        [*_crowd(rng, 4, 1, 9.0), *_riders(rng, 16, 5, 9.0)], ('commute', 'multi-cyclist')))
    rng = np.random.default_rng(104)
    scenes.append(_scene(
        'Wet Night Mixed Traffic', 'alert',
        'Ten pedestrians and eight riders on a wet night; some riders go as fast as e-bikes.',
        [*_crowd(rng, 10, 1, 8.0), *_riders(rng, 8, 11, 8.0, speed=(5.0, 10.0))],
        ('night', 'rain', 'ebike', 'known-hard'), weather='wet_night'))
    rng = np.random.default_rng(105)
    tour = [ped(1 + i, Path.start(-6 + 1.5 * i, -7.5 - rng.uniform(0, 0.5), rng.uniform(0, 1))
                .to(26 + 1.5 * i, -7.5, 1.0).build()) for i in range(10)]
    scenes.append(_scene(
        'Sidewalk Tour Group', 'warning',
        'A tour group of ten walks along the sidewalk while six cyclists pass on the road; '
        'nobody crosses.',
        [*tour, *_riders(rng, 6, 11, 8.0, both_ways=False)],
        ('sidewalk', 'negative', 'known-hard')))
    rng = np.random.default_rng(106)
    scenes.append(_scene(
        'Crosswalk Without Cyclists', 'safe',
        'Twenty-four pedestrians cross in both directions and no cyclist appears.',
        _crowd(rng, 24, 1, 10.0), ('no-cyclist', 'negative')))
    rng = np.random.default_rng(107)
    scenes.append(_scene(
        'Group Ride Through Crowd', 'alert',
        'A group ride of ten passes in two lines through a crosswalk with eight pedestrians on '
        'it.',
        [*_crowd(rng, 8, 1, 5.0),
         *[bike(9 + i, Path.start(34 + 3.0 * (i // 2), 0.5 + 1.0 * (i % 2), 0.0, heading=180)
                .ride(60, 6.5).build()) for i in range(10)]], ('group-ride', 'platoon')))
    rng = np.random.default_rng(108)
    scenes.append(_scene(
        'Market Day Shared Street', 'alert',
        'A shared street with fourteen pedestrians wandering across at random angles and six '
        'slow cyclists.',
        [*[ped(1 + i, Path.start(rng.uniform(-6, 16), -7.0, rng.uniform(0, 6))
               .to(rng.uniform(-6, 16), 7.0, rng.uniform(0.9, 1.4)).build())
           for i in range(14)],
         *_riders(rng, 6, 15, 8.0, speed=(3.5, 5.0))], ('shared-space',)))
    return scenes
