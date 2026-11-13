"""The ``extended`` suite: hand-designed urban conflicts and deliberate negatives.

Same frame as the conformance scenarios: metres, ``x`` along the road, ``y`` across it, the
crosswalk centred on ``(5, 0)`` and about 3 m wide, kerbs near ``y = -6.5`` and ``y = +6.5``.
Through cyclists ride towards ``-x`` (heading 180 degrees) a metre or two either side of the
centre line, as in the paper's scenarios.

Speeds follow published ranges: walking 1.2-1.5 m/s, older adults 0.8-1.0, wheelchairs 1.0-1.3,
children running 2-3, runners 3-4, commuter cyclists 4-7, fast road cyclists 8-11, e-bikes up to
about 11, e-scooters 5-7, cargo bikes 3.5-5.5; comfortable cyclist braking about 2 m/s^2 and hard
braking 4-6 m/s^2.

``expected_dominant`` is the label the testbench ground truth implies
(:func:`bikeped.scenarios.evaluate.ground_truth_label`), checked by running each scenario.
Scenarios the default pipeline gets wrong carry the ``known-hard`` tag: its dominant state differs
from ``expected_dominant`` (a false alarm or a miss), or its first alert comes more than
:data:`~bikeped.scenarios.evaluate.LATE_ALERT_S` after a danger interval begins.
"""

from __future__ import annotations

from .builders import Path, group, platoon, shift

FAMILIES = {
    'turning': 'Turning conflicts',
    'occlusion': 'Occlusion and late emergence',
    'yielding': 'Yielding and braking cyclists',
    'counter-flow': 'Wrong-way and counter-flow riders',
    'shared-space': 'Shared space and sidewalk riding',
    'groups': 'Groups and platoons',
    'vulnerable': 'Vulnerable road users',
    'micromobility': 'Micromobility',
    'pedestrian-behaviour': 'Pedestrian behaviour',
    'near-miss': 'Near-miss geometry',
    'signal-platoon': 'Signal-phase platoons',
    'negative': 'True negatives',
}


def ped(aid: int, path: list[dict], model: str = 'adult', label: str | None = None) -> dict:
    """Pedestrian agent dict with a model hint."""
    a = {'type': 'person', 'id': aid, 'path': path, 'model': model}
    if label:
        a['label'] = label
    return a


def bike(aid: int, path: list[dict], model: str = 'bicycle', label: str | None = None) -> dict:
    """Cyclist agent dict with a model hint."""
    a = {'type': 'bike', 'id': aid, 'path': path, 'model': model}
    if label:
        a['label'] = label
    return a


def across(speed: float = 1.4, t0: float = 0.0, x: float = 5.0, y0: float = -6.5,
           y1: float = 6.5) -> list[dict]:
    """Pedestrian walking straight across the road at ``x`` from ``y0`` to ``y1``."""
    return Path.start(x, y0, t0).to(x, y1, speed).build()


def along(speed: float, t_at: float, y: float = 1.0, x_at: float = 5.0, x_end: float = -8.0,
          max_start: float = 34.0) -> list[dict]:
    """Cyclist riding towards ``-x`` at ``speed`` that reaches ``x_at`` at time ``t_at``.

    It starts at ``t = 0`` far enough back to arrive on time, or later if that would put it more
    than ``max_start`` metres down the road.
    """
    x0 = x_at + speed * t_at
    t0 = 0.0
    if x0 > max_start:
        t0 = t_at - (max_start - x_at) / speed
        x0 = max_start
    return Path.start(x0, y, t0).to(x_end, y, speed).build()


def scenario(name: str, family: str, expected: str, description: str, agents: list[dict],
             tags: tuple[str, ...] = (), vehicle: str | None = None,
             weather: str | None = None) -> dict:
    """Assemble one scenario dict with library metadata."""
    from .conformance import slugify

    s: dict = {'name': name, 'expected_dominant': expected}
    if vehicle:
        s['vehicle'] = vehicle
    s['agents'] = agents
    s.update(id=slugify(name), suite='extended', family=family, description=description,
             tags=list(tags))
    if weather:
        s['render'] = {'weather': weather}
    return s


def _turning() -> list[dict]:
    return [
        scenario(
            'Right Hook Across Crosswalk', 'turning', 'alert',
            'A cyclist at 5 m/s turns right off the main road through the far end of the '
            'crosswalk, where a pedestrian has just stepped off the kerb.',
            [ped(1, Path.start(5, 7.0).wait(1.5).to(5, -6.5, 1.35).build()),
             bike(2, Path.start(28, 2.5, heading=180).ride(18, 5.0).arc(5, -90).ride(6).build())],
            ('right-hook', 'turn')),
        scenario(
            'Left Hook Across Crosswalk', 'turning', 'alert',
            'A cyclist turns left across the oncoming lane and sweeps through the near half of '
            'the crosswalk while a pedestrian is walking out from that kerb.',
            [ped(1, across(1.4, 0.5, y0=-7.0)),
             bike(2, Path.start(30, -1.0, heading=180).ride(19, 5.5).arc(6, 90).ride(6)
                  .build())],
            ('left-hook', 'turn')),
        scenario(
            'Turn In From Cross Street', 'turning', 'alert',
            'A cyclist comes up the cross street, turns right onto the main road and rides '
            'through the middle of the crosswalk as a pedestrian reaches it.',
            [ped(1, across(1.4, 0.0, y0=-6.0)),
             bike(2, Path.start(2, -16, heading=90).ride(12, 4.5).arc(4, -90).ride(14).build())],
            ('cross-street', 'turn')),
        scenario(
            'Wide Turn Clears Pedestrian', 'turning', 'warning',
            'A cyclist swings a wide left turn that stays more than 6 m from a pedestrian walking '
            'on the far half of the crosswalk.',
            [ped(1, Path.start(5, 1.0).to(5, 7.0, 1.3).build()),
             bike(2, Path.start(30, 0.5, heading=180).ride(16, 5.0).arc(12, 90).ride(5)
                  .build())],
            ('turn', 'negative', 'known-hard')),
    ]


def _occlusion() -> list[dict]:
    return [
        scenario(
            'Steps Out From Behind Van', 'occlusion', 'alert',
            'A pedestrian hidden by a parked van is first detected stepping into the road 3.5 m '
            'from the kerb line, 1.5 s before a 6 m/s cyclist reaches the crossing point.',
            [ped(1, Path.start(5, -3.5).appear(3.0).to(5, 6.5, 1.4).build()),
             bike(2, along(6.0, 4.6, y=-1.5))],
            ('occlusion', 'late-detection', 'night'), weather='wet_night'),
        scenario(
            'Bus Stop Crowd Emergence', 'occlusion', 'alert',
            'Two people wait at a bus stop; a third steps out from behind them at a brisk 1.8 m/s '
            'straight into the path of a cyclist.',
            [ped(1, Path.start(6.2, -7.0).wait(9.0).build(), label='waiting'),
             ped(2, Path.start(4.2, -7.2).wait(9.0).build(), label='waiting'),
             ped(3, Path.start(5.0, -6.0).appear(2.5).to(5.0, 6.5, 1.8).build()),
             bike(4, along(6.5, 5.3, y=-2.0))],
            ('occlusion', 'bus-stop', 'group')),
        scenario(
            'Cyclist Emerges From Behind Bus', 'occlusion', 'alert',
            'A cyclist overtaking a stopped bus is first seen 8 m from the crosswalk at 6 m/s '
            'while a pedestrian is crossing in front of the bus.',
            [ped(1, across(1.4, 0.0)),
             bike(2, Path.start(13, 1.2, 3.3, heading=180).ride(20, 6.0).build())],
            ('occlusion', 'late-detection', 'fog'), weather='fog'),
    ]


def _yielding() -> list[dict]:
    return [
        scenario(
            'Cyclist Stops Before Crosswalk', 'yielding', 'alert',
            'A cyclist at 5.5 m/s brakes comfortably at 2 m/s^2 to a stop 4 m short of the '
            'crosswalk, waits for the pedestrian to clear, then rides on.',
            [ped(1, across(1.4, 1.0)),
             bike(2, Path.start(34, 1.0, heading=180).ride(17.4, 5.5).brake(0, 2.0).wait(4.5)
                  .accelerate(5.0, 1.5).ride(10).build())],
            ('yield', 'braking')),
        scenario(
            'Rolling Yield', 'yielding', 'warning',
            'A cyclist slows from 6 m/s to 2 m/s and rolls through the crosswalk after the '
            'pedestrian has passed the bike lane.',
            [ped(1, across(1.4, 0.0)),
             bike(2, Path.start(32, -1.5, heading=180).ride(10, 6.0).brake(2.0, 2.0)
                  .ride(10).accelerate(5.0, 1.5).ride(8).build())],
            ('yield', 'braking', 'known-hard')),
        scenario(
            'Track Stand Then Go', 'yielding', 'warning',
            'A cyclist balances in place 5 m before the crosswalk while a pedestrian crosses, '
            'then sets off once the pedestrian is on the far side.',
            [ped(1, across(1.4, 0.0)),
             bike(2, Path.start(10, 1.5, heading=180).wait(8.5).accelerate(4.5, 1.5).ride(12)
                  .build())],
            ('yield', 'stationary-bike')),
        scenario(
            'Late Hard Braking', 'yielding', 'alert',
            'A cyclist at 6.5 m/s notices the pedestrian late and brakes hard at 5 m/s^2. The '
            'bike stops about a metre from them.',
            [ped(1, across(1.4, 0.0)),
             bike(2, Path.start(30, 0.8, heading=180).ride(19.4, 6.5).brake(0, 5.0).wait(3.0)
                  .build())],
            ('braking', 'hard-brake')),
    ]


def _counter_flow() -> list[dict]:
    return [
        scenario(
            'Wrong-Way Rider', 'counter-flow', 'alert',
            'A cyclist rides against traffic towards +x at 5 m/s; the pedestrian checks the '
            'usual direction and steps out into the rider.',
            [ped(1, across(1.4, 0.0)),
             bike(2, Path.start(-20, -1.5, heading=0).ride(42, 5.0).build())],
            ('wrong-way',)),
        scenario(
            'Contraflow Lane Pass', 'counter-flow', 'warning',
            'A wrong-way cyclist stays in the far contraflow lane and crosses the crosswalk '
            'well ahead of the pedestrian, who is still at the opposite kerb.',
            [ped(1, Path.start(5, 7.0).wait(3.0).to(5, -6.5, 1.3).build()),
             bike(2, Path.start(-24, -4.5, heading=0).ride(50, 5.0).build())],
            ('wrong-way', 'negative', 'known-hard')),
    ]


def _shared_space() -> list[dict]:
    return [
        scenario(
            'Sidewalk Rider Head-On', 'shared-space', 'alert',
            'A cyclist rides along the sidewalk at 4.5 m/s towards a pedestrian walking the '
            'other way on the same line.',
            [ped(1, Path.start(-6, -8.0).to(18, -8.0, 1.4).build()),
             bike(2, Path.start(30, -8.2, heading=180).ride(36, 4.5).build())],
            ('sidewalk', 'head-on'), weather='light_rain'),
        scenario(
            'Shared Plaza Slalom', 'shared-space', 'alert',
            'In a shared plaza a cyclist at 4 m/s weaves between two walkers and cuts close in '
            'front of a third.',
            [ped(1, Path.start(9, -4).to(9, 4, 1.2).build()),
             ped(2, Path.start(15, 3).to(15, -4, 1.3).build()),
             ped(3, Path.start(4.5, -5).to(4.5, 4, 1.3).build()),
             bike(4, Path.start(28, 0, heading=180).through(
                 [(20, 1.5), (15, -0.8), (11, 1.2), (7, 0.2), (2, -0.5), (-6, 0)], 4.0).build())],
            ('shared-space', 'multi-pedestrian', 'known-hard')),
    ]


def _groups() -> list[dict]:
    lead = across(1.0, 0.0, y0=-7.0, y1=7.0)
    kids = group(lead, [(0, 0, 0), (0.7, -0.3, 0.2), (-0.6, -0.6, 0.4), (0.3, -1.2, 0.7),
                        (-0.4, -1.6, 0.9), (0.8, -2.0, 1.2)])
    return [
        scenario(
            'School Group Crossing', 'groups', 'alert',
            'Two adults walk a line of six children across at 1.0 m/s; a cyclist at 5 m/s reaches '
            'the crosswalk while the group is strung across the lane.',
            [ped(1, shift(lead, dx=-1.0), label='teacher'),
             *[ped(i + 2, p, model='child') for i, p in enumerate(kids)],
             ped(8, shift(lead, dx=1.0, dy=-2.6, dt=1.5), label='teacher'),
             bike(9, along(5.0, 7.0, y=0.5))],
            ('group', 'child', 'tracker-load')),
        scenario(
            'Cyclist Platoon', 'groups', 'alert',
            'Four commuters ride in single file 1.2 s apart at 5.5 m/s into a pedestrian who is '
            'halfway across.',
            [ped(1, across(1.3, 0.0)),
             *[bike(i + 2, p) for i, p in enumerate(
                 platoon(along(5.5, 4.5, y=0.8), 4, headway_s=1.2, jitter_m=0.3, seed=3))]],
            ('platoon', 'multi-cyclist')),
        scenario(
            'Two Abreast', 'groups', 'alert',
            'Two cyclists ride side by side at 6 m/s and fill the lane as a pedestrian crosses.',
            [ped(1, across(1.4, 0.0)),
             bike(2, along(6.0, 5.0, y=0.4)),
             bike(3, along(6.0, 5.1, y=1.6))],
            ('side-by-side', 'multi-cyclist')),
        scenario(
            'Pedestrian Couple', 'groups', 'alert',
            'A couple walks across side by side at 1.3 m/s and a cyclist at 6 m/s rides through '
            'their line.',
            [ped(1, across(1.3, 0.0, x=4.6)),
             ped(2, across(1.3, 0.1, x=5.4)),
             bike(3, along(6.0, 5.2, y=0.3))],
            ('couple', 'multi-pedestrian')),
        scenario(
            'Crowd With Straggler', 'groups', 'alert',
            'Five people cross together and clear the lane; a straggler follows 3.5 s later and '
            'meets the cyclist the rest of the group avoided.',
            [*[ped(i + 1, p) for i, p in enumerate(group(across(1.4, 0.0), [
                (-0.6, 0, 0), (0.5, -0.4, 0.2), (0.0, -0.9, 0.3), (-0.4, -1.5, 0.5),
                (0.6, -1.8, 0.6)]))],
             ped(6, across(1.3, 3.8), label='straggler'),
             bike(7, along(6.0, 8.8, y=0.6))],
            ('group', 'straggler', 'tracker-load')),
    ]


def _vulnerable() -> list[dict]:
    return [
        scenario(
            'Older Adult Slow Start', 'vulnerable', 'alert',
            'An older pedestrian waits 2 s at the kerb, then crosses at 0.85 m/s and is in the '
            'lane when a 5.5 m/s cyclist arrives.',
            [ped(1, Path.start(5, -6.5).wait(2.0).to(5, 6.5, 0.85).build(), label='older adult'),
             bike(2, along(5.5, 9.5, y=0.0))],
            ('older-adult', 'slow-crossing', 'known-hard'), weather='dusk'),
        scenario(
            'Wheelchair Mid-Crossing Pause', 'vulnerable', 'alert',
            'A wheelchair user crossing at 1.1 m/s stops for 3 s in the middle of the road; a '
            'cyclist rides through the stopping point.',
            [ped(1, Path.start(5, -6.5).to(5, -0.5, 1.1).wait(3.0).to(5, 6.5, 1.1).build(),
                 model='wheelchair'),
             bike(2, along(5.0, 8.0, y=-0.3))],
            ('wheelchair', 'stationary')),
        scenario(
            'Child Darts Back', 'vulnerable', 'alert',
            'A child halfway across turns and runs back at 2.5 m/s for a dropped toy, into the '
            'lane the cyclist had counted on being empty.',
            [ped(1, Path.start(5, -6.5).to(5, 2.0, 1.2).wait(0.5).to(5, -4.0, 2.5).wait(1.0)
                 .to(5, 6.5, 1.4).build(), model='child'),
             bike(2, along(6.0, 9.6, y=-1.5))],
            ('child', 'reversal')),
        scenario(
            'Guide Dog Pace', 'vulnerable', 'alert',
            'A visually impaired pedestrian with a guide dog crosses at a steady 1.0 m/s and '
            'cannot see the approaching 6 m/s cyclist.',
            [ped(1, across(1.0, 0.0), label='with guide dog'),
             bike(2, along(6.0, 7.0, y=0.5))],
            ('visually-impaired', 'slow-crossing', 'known-hard')),
    ]


def _micromobility() -> list[dict]:
    return [
        scenario(
            'E-Scooter Weaving', 'micromobility', 'alert',
            'An e-scooter rider at 6 m/s weaves a metre either side of the lane line and reaches '
            'the crosswalk while a pedestrian is in it.',
            [ped(1, across(1.4, 0.0)),
             bike(2, Path.start(34, 0.8, heading=180).zigzag(40, 1.0, 16.0, 6.0).build(),
                  model='escooter')],
            ('escooter', 'weaving'), vehicle='ebike'),
        scenario(
            'E-Bike In The Door Zone', 'micromobility', 'alert',
            'A fast e-bike at 10 m/s rides close to parked cars; a pedestrian steps out from '
            'between two of them a car length ahead.',
            [ped(1, Path.start(5, 5.5).appear(1.8).to(5, -6.5, 1.5).build()),
             bike(2, along(10.0, 3.2, y=3.6), model='ebike')],
            ('ebike', 'door-zone', 'occlusion', 'night'), vehicle='ebike',
            weather='clear_night'),
        scenario(
            'Cargo Bike Long Stop', 'micromobility', 'alert',
            'A loaded cargo bike at 5 m/s brakes at only 1.5 m/s^2 and needs most of the approach '
            'to stop. It comes to rest on the crosswalk next to the pedestrian.',
            [ped(1, across(1.3, 0.0)),
             bike(2, Path.start(30, 0.5, heading=180).ride(16.5, 5.0).brake(0, 1.5).wait(3)
                  .build(), model='cargo_bike')],
            ('cargo-bike', 'braking'), weather='light_rain'),
        scenario(
            'Delivery Rider Cuts Corner', 'micromobility', 'alert',
            'A delivery rider on an e-bike at 7 m/s cuts the corner from the cross street, '
            'diagonally through the crosswalk.',
            [ped(1, across(1.4, 0.5)),
             bike(2, Path.start(-2, -16, heading=80).through(
                 [(0, -8), (4, -1.5), (12, 2.0), (26, 3.0)], 7.0).build(), model='ebike')],
            ('delivery', 'ebike', 'corner-cut'), vehicle='ebike'),
    ]


def _pedestrian_behaviour() -> list[dict]:
    return [
        scenario(
            'Hesitation At The Kerb', 'pedestrian-behaviour', 'alert',
            'A pedestrian steps off the kerb, stops for 1.5 s unsure of the cyclist, then goes '
            'anyway as the cyclist arrives.',
            [ped(1, Path.start(5, -6.5).to(5, -4.0, 1.2).wait(1.5).to(5, 6.5, 1.5).build()),
             bike(2, along(5.5, 6.6, y=-1.2))],
            ('hesitation',)),
        scenario(
            'Turns Back Mid-Crossing', 'pedestrian-behaviour', 'alert',
            'A pedestrian reaches the centre line, changes their mind and walks back towards the '
            'kerb they came from, into the cyclist\'s lane.',
            [ped(1, Path.start(5, -6.5).to(5, 0.5, 1.4).wait(0.6).to(5, -6.5, 1.4).build()),
             bike(2, along(6.0, 7.4, y=-1.5))],
            ('reversal',)),
        scenario(
            'Diagonal Runner', 'pedestrian-behaviour', 'alert',
            'A runner crosses diagonally at 3.5 m/s, outside the crosswalk lines, and meets a '
            '6 m/s cyclist in the lane.',
            [ped(1, Path.start(-3, -7).to(13, 7, 3.5).build(), model='runner'),
             bike(2, along(6.0, 3.6, y=-0.5, x_at=4.0))],
            ('runner', 'diagonal')),
        scenario(
            'Phone-Distracted Diagonal', 'pedestrian-behaviour', 'alert',
            'Looking at a phone, a pedestrian drifts diagonally across at 0.9 m/s and spends a '
            'long time in the lane.',
            [ped(1, Path.start(2, -7).to(10, 7, 0.9).build(), label='on phone'),
             bike(2, along(5.5, 9.5, y=0.0, x_at=6.0))],
            ('distracted', 'diagonal', 'slow-crossing', 'known-hard')),
        scenario(
            'Mid-Block Crossing', 'pedestrian-behaviour', 'alert',
            'A pedestrian crosses 10 m upstream of the crosswalk, between gaps in traffic, and '
            'the cyclist meets them before reaching the marked crossing.',
            [ped(1, across(1.5, 0.0, x=15.0)),
             bike(2, along(6.5, 4.5, y=0.5, x_at=15.0))],
            ('jaywalking', 'mid-block', 'known-hard')),
    ]


def _near_miss() -> list[dict]:
    return [
        scenario(
            'Passes In Front', 'near-miss', 'alert',
            'The cyclist crosses the pedestrian\'s line about a second before the pedestrian '
            'reaches it, a margin of a few metres.',
            [ped(1, across(1.4, 0.0)),
             bike(2, along(6.0, 3.6, y=-0.5))],
            ('margin', 'front')),
        scenario(
            'Passes Behind', 'near-miss', 'warning',
            'The cyclist rides through the crosswalk 2.5 m behind a pedestrian who has already '
            'crossed the lane.',
            [ped(1, across(1.4, 0.0)),
             bike(2, along(6.0, 8.0, y=-1.5))],
            ('margin', 'behind', 'known-hard')),
        scenario(
            'Tangential Pass', 'near-miss', 'alert',
            'A cyclist passes 2 m to the side of a pedestrian standing at the edge of the lane, '
            'without changing course.',
            [ped(1, Path.start(5, -4.0).wait(7.0).build()),
             bike(2, along(6.0, 3.5, y=-2.0))],
            ('tangential', 'stationary')),
        scenario(
            'Parallel Never Converges', 'near-miss', 'warning',
            'A cyclist rides along the road while a pedestrian walks the same way on the '
            'sidewalk 7 m to the side; the paths never meet.',
            [ped(1, Path.start(20, -7.5).to(2, -7.5, 1.4).build()),
             bike(2, Path.start(30, -0.5, heading=180).ride(36, 5.0).build())],
            ('parallel', 'negative', 'known-hard')),
    ]


def _signal_platoon() -> list[dict]:
    riders = []
    for i in range(5):
        x0 = 30 + 1.8 * (i // 2)
        y0 = 0.4 + 1.1 * (i % 2)
        riders.append(bike(i + 2, Path.start(x0, y0, heading=180).wait(1.0 + 0.3 * i)
                           .accelerate(5.5, 1.5).ride(34).build()))
    return [
        scenario(
            'Green Wave Burst', 'signal-platoon', 'alert',
            'Five cyclists waiting at a light set off together when it turns green and meet a '
            'pedestrian who started crossing late.',
            [ped(1, Path.start(5, -6.5, 4.0).to(5, 6.5, 1.5).build(), label='late starter'),
             *riders],
            ('platoon', 'signal', 'multi-cyclist')),
    ]


def _negatives() -> list[dict]:
    return [
        scenario(
            'Far Lane Moving Away', 'negative', 'warning',
            'A cyclist just past the crosswalk rides away along the far lane while a pedestrian '
            'starts crossing behind them.',
            [ped(1, across(1.4, 1.0)),
             bike(2, Path.start(6, -3.0, heading=0).ride(28, 6.0).build())],
            ('receding',)),
        scenario(
            'Fast Pass After Clearing', 'negative', 'warning',
            'A road cyclist at 9 m/s goes through the crosswalk a second after the pedestrian '
            'has stepped onto the far kerb.',
            [ped(1, across(1.5, 0.0, y1=7.0)),
             bike(2, along(9.0, 10.0, y=1.0), model='road_bike')],
            ('fast', 'cleared', 'known-hard')),
        scenario(
            'Waits At The Kerb', 'negative', 'warning',
            'A pedestrian stands on the kerb the whole time and never steps out while a cyclist '
            'passes at 7 m/s.',
            [ped(1, Path.start(5, -7.0).wait(8.0).build()),
             bike(2, along(7.0, 4.0, y=1.0))],
            ('kerb', 'stationary', 'known-hard')),
    ]


def extended_suite() -> list[dict]:
    """All extended scenarios, grouped by family."""
    return [*_turning(), *_occlusion(), *_yielding(), *_counter_flow(), *_shared_space(),
            *_groups(), *_vulnerable(), *_micromobility(), *_pedestrian_behaviour(),
            *_near_miss(), *_signal_platoon(), *_negatives()]
