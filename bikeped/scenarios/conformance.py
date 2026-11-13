"""Metadata for the 24 conformance scenarios of the paper.

The scenario content lives in ``decision_testbench.SCENARIOS`` and is not copied here. This module
only adds the library metadata (slug, family, description, tags, per-agent model hints) and
:func:`conformance_suite` returns decorated shallow copies, leaving the originals untouched.
"""

from __future__ import annotations

import re

FAMILIES = {
    'safe-crossing': 'Safe crossings',
    'standard-approach': 'Standard approaches',
    'high-speed': 'High-speed and accelerating',
    'accessibility': 'Accessibility',
    'multi-agent': 'Multi-agent',
    'edge-case': 'Edge cases',
    'non-linear': 'Non-linear trajectories',
}

# name -> (family, description, tags, {agent id: model})
META: dict[str, tuple[str, str, list[str], dict[int, str]]] = {
    'Safe Crossing': (
        'safe-crossing',
        'A pedestrian crosses the full crosswalk at about 2.7 m/s with no cyclist anywhere in the '
        'scene.', ['baseline', 'no-cyclist'], {}),
    'Bike Approaches Ped': (
        'standard-approach',
        'A cyclist rides in along the road from 34 m at 8 m/s, slows to about 2 m/s and passes '
        'through the crosswalk just behind a pedestrian walking across.', ['core'], {}),
    'High-Speed Near Miss': (
        'high-speed',
        'A cyclist at about 7 m/s crosses the crosswalk while a slow pedestrian is in the middle '
        'of it, so the danger window is short.', ['near-miss'], {}),
    'Multiple VRUs': (
        'multi-agent',
        'Three pedestrians cross in both directions while one cyclist rides along the road and a '
        'second comes in diagonally from the far side.', ['multi-pedestrian', 'multi-cyclist'],
        {}),
    'Occluded Emergence': (
        'edge-case',
        'The cyclist is first seen 4 m from the crosswalk, as if appearing from behind an '
        'obstacle, and cuts diagonally across the path of a slow pedestrian.',
        ['occlusion', 'late-detection'], {}),
    'Parallel Paths': (
        'safe-crossing',
        'A cyclist rides across the road 3 m to the side of a crossing pedestrian, at the same '
        'speed and heading; the gap stays at 3 m.', ['parallel', 'negative'], {}),
    'Bikes Only': (
        'safe-crossing',
        'Two cyclists ride through the crosswalk area and nobody is on foot.',
        ['no-pedestrian', 'negative'], {}),
    'Fast Approach': (
        'high-speed',
        'An e-bike at 12 m/s appears 20 m out and rides straight through the crosswalk while a '
        'pedestrian is on it.', ['ebike'], {2: 'ebike'}),
    'Speed Variant': (
        'standard-approach',
        'A slow pedestrian crosses while a cyclist at 3 m/s and then a second at 10 m/s ride '
        'through the crosswalk on either side of the centre line.', ['multi-cyclist'], {}),
    'Counter-Flow on Crosswalk': (
        'standard-approach',
        'A cyclist rides along the crosswalk itself, head-on towards a pedestrian walking the '
        'other way.', ['counter-flow', 'head-on'], {}),
    'Multi-Speed Converging': (
        'multi-agent',
        'Two pedestrians cross in opposite directions while three cyclists at different speeds '
        'converge on the crosswalk from different angles.', ['multi-pedestrian', 'multi-cyclist'],
        {}),
    'Wheelchair User': (
        'accessibility',
        'A wheelchair user crosses at 0.6 m/s and is still on the crosswalk when a cyclist rides '
        'through it at speed.', ['wheelchair', 'slow-crossing'], {1: 'wheelchair'}),
    'Stopped Pedestrian': (
        'edge-case',
        'A pedestrian stops in the middle of the crosswalk for 4 s to look at a phone. A cyclist '
        'rides through the stopping point.', ['stationary', 'distracted'], {}),
    'Cross-Street Cyclist': (
        'standard-approach',
        'A cyclist coming from the cross street rides along the crosswalk at 6 m/s, through a '
        'pedestrian who is crossing in the same direction.', ['cross-street', 'red-light'], {}),
    'E-Scooter': (
        'high-speed',
        'An e-scooter at about 7 m/s rides along the road through the crosswalk while a '
        'pedestrian is crossing.', ['escooter', 'micromobility'], {2: 'escooter'}),
    'Child Pedestrian': (
        'accessibility',
        'A child crosses at 0.9 m/s, so is exposed on the crosswalk for longer, and a cyclist '
        'rides through at up to 8 m/s.', ['child', 'slow-crossing'], {1: 'child'}),
    'Overtake from Behind': (
        'standard-approach',
        'A cyclist rides along the crosswalk in the same direction as a pedestrian and passes '
        'through them from behind at 4 m/s.', ['same-direction', 'overtake'], {}),
    'Cyclist Abort': (
        'non-linear',
        'A cyclist heads for the crosswalk, then turns away 3 m short of it and leaves, so the '
        'threat resolves.', ['turn-away', 'resolving'], {}),
    'Dense Pedestrian Group': (
        'multi-agent',
        'Eight pedestrians cross together as a tight group and a cyclist rides into it.',
        ['group', 'tracker-load'], {}),
    'Diagonal Jaywalker': (
        'edge-case',
        'A pedestrian crosses diagonally across the road beside the marked crosswalk while a '
        'cyclist comes along the road.', ['jaywalking', 'diagonal'], {}),
    'Swerving Cyclist': (
        'non-linear',
        'A cyclist riding along the far side of the road swerves across it towards the '
        'pedestrian on the crosswalk.', ['swerve'], {}),
    'Late Turn Toward Ped': (
        'non-linear',
        'A cyclist rides parallel to the kerb, then turns late across the crosswalk into the '
        'pedestrian.', ['turn'], {}),
    'U-Turn Cyclist': (
        'non-linear',
        'A cyclist passes a pedestrian standing on the crosswalk, makes a U-turn and comes back '
        'through them.', ['u-turn', 'stationary'], {}),
    'Accelerating E-Bike': (
        'high-speed',
        'An e-bike accelerates from 3 m/s to 8 m/s on the way to the crosswalk while a pedestrian '
        'is crossing.', ['ebike', 'accelerating'], {2: 'ebike'}),
}


def slugify(name: str) -> str:
    """Lower-case slug of a display name: ``'High-Speed Near Miss'`` -> ``'high-speed-near-miss'``.
    """
    return re.sub(r'[^a-z0-9]+', '-', name.lower()).strip('-')


def conformance_suite() -> list[dict]:
    """The paper's 24 scenarios in paper order, as decorated shallow copies.

    Each copy shares the waypoint lists with ``decision_testbench.SCENARIOS``; the agent dicts are
    copied so that the ``model`` hints do not touch the originals.
    """
    import decision_testbench as tb

    out = []
    for original in tb.SCENARIOS:
        family, description, tags, models = META[original['name']]
        s = dict(original)
        s['agents'] = []
        for agent in original['agents']:
            a = dict(agent)
            if agent['id'] in models:
                a['model'] = models[agent['id']]
            s['agents'].append(a)
        s.update(id=slugify(original['name']), suite='conformance', family=family,
                 description=description, tags=list(tags))
        out.append(s)
    return out
