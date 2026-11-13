"""Catalog export: ``catalog.json``, one SVG thumbnail per scenario, simulator data, markdown.

Thumbnails are 320x200 top-down sketches in the scenario frame: the road runs left to right
(``+x`` to the right), ``+y`` points up, the crosswalk is the striped band at ``x = 5``. The view
is the window ``x in [-12, 36]``, ``y in [-15, 15]`` (metres), widened at the same aspect ratio
when a scenario's paths leave it, so most thumbnails share one scale. Pedestrian paths are cyan,
cyclist paths amber; dash length grows with the agent's mean speed, a filled dot marks a start
in view and a hollow dot an agent that emerges within 15 m of the crosswalk after the start,
a ring marks an agent that stands still.
"""

from __future__ import annotations

import json
import math
from pathlib import Path as FsPath
from typing import Iterable

import numpy as np

from .builders import path_speeds
from .registry import get_suite, list_suites

W, H = 320.0, 200.0
VIEW = (-12.0, 36.0, -15.0, 15.0)
ROAD_HALF = 6.5
CROSSWALK_X = (3.5, 6.5)

BG = '#111317'
ROAD = '#181b20'
KERB = '#2b2f37'
LANE = '#30343c'
STRIPE = '#262a32'
PED = '#6fe3ff'
BIKE = '#ffb547'


def _duration(s: dict) -> float:
    import decision_testbench as tb

    return float(tb.get_scenario_duration(s))


def _window(s: dict) -> tuple[float, float, float, float]:
    """Smallest window containing VIEW and every path point (1.5 m margin), aspect 320:200."""
    x0, x1, y0, y1 = VIEW
    pts = [(p['x'], p['y']) for a in s['agents'] for p in a['path']]
    xs = [p[0] for p in pts]
    ys = [p[1] for p in pts]
    x0, x1 = min(x0, min(xs) - 1.5), max(x1, max(xs) + 1.5)
    y0, y1 = min(y0, min(ys) - 1.5), max(y1, max(ys) + 1.5)
    w, h = x1 - x0, y1 - y0
    aspect = W / H
    if w / h < aspect:
        cx, w = (x0 + x1) / 2, h * aspect
        x0, x1 = cx - w / 2, cx + w / 2
    else:
        cy, h = (y0 + y1) / 2, w / aspect
        y0, y1 = cy - h / 2, cy + h / 2
    return x0, x1, y0, y1


def _f(v: float) -> str:
    return f'{v:.1f}'.rstrip('0').rstrip('.')


def thumbnail_svg(s: dict) -> str:
    """SVG source of the thumbnail for scenario ``s`` (320x200 viewBox, no text)."""
    x0, x1, y0, y1 = _window(s)
    k = W / (x1 - x0)

    def px(x: float) -> float:
        return (x - x0) * k

    def py(y: float) -> float:
        return (y1 - y) * k

    out = ['<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 320 200" width="320" '
           'height="200">',
           f'<rect width="320" height="200" fill="{BG}"/>',
           f'<rect x="0" y="{_f(py(ROAD_HALF))}" width="320" '
           f'height="{_f(2 * ROAD_HALF * k)}" fill="{ROAD}"/>']
    for yk in (ROAD_HALF, -ROAD_HALF):
        out.append(f'<line x1="0" y1="{_f(py(yk))}" x2="320" y2="{_f(py(yk))}" '
                   f'stroke="{KERB}" stroke-width="1.5"/>')
    out.append(f'<line x1="0" y1="{_f(py(0))}" x2="320" y2="{_f(py(0))}" stroke="{LANE}" '
               f'stroke-width="1" stroke-dasharray="8 8"/>')
    sx0, sx1 = px(CROSSWALK_X[0]), px(CROSSWALK_X[1])
    y = -ROAD_HALF + 0.3
    while y + 0.6 <= ROAD_HALF:
        out.append(f'<rect x="{_f(sx0)}" y="{_f(py(y + 0.6))}" width="{_f(sx1 - sx0)}" '
                   f'height="{_f(0.6 * k)}" fill="{STRIPE}"/>')
        y += 1.2
    # Pedestrians are drawn last so that they stay visible where a bike path crosses them.
    for agent in sorted(s['agents'], key=lambda a: a['type'] == 'person'):
        out.extend(_agent_svg(agent, px, py))
    out.append('</svg>')
    return '\n'.join(out) + '\n'


def _agent_svg(agent: dict, px, py) -> list[str]:
    color = PED if agent['type'] == 'person' else BIKE
    path = agent['path']
    pts = [(px(p['x']), py(p['y'])) for p in path]
    length = sum(math.hypot(b[0] - a[0], b[1] - a[1]) for a, b in zip(pts, pts[1:]))
    sx, sy = pts[0]
    # Hollow start dot: the agent is first seen close to the crosswalk, i.e. it emerges from
    # behind something rather than riding or walking into view from the edge of the scene.
    late = path[0]['t'] > 0.5 and math.hypot(path[0]['x'] - 5.0, path[0]['y']) < 15.0
    if length < 3.0:
        return [f'<circle cx="{_f(sx)}" cy="{_f(sy)}" r="5" fill="none" stroke="{color}" '
                f'stroke-width="2"/>',
                f'<circle cx="{_f(sx)}" cy="{_f(sy)}" r="2" fill="{color}"/>']
    speeds = path_speeds(path)
    moving = speeds[speeds > 0.2]
    v = float(np.mean(moving)) if len(moving) else 0.0
    dash = min(2.0 + 1.6 * v, 18.0)
    d = 'M' + ' L'.join(f'{_f(x)} {_f(y)}' for x, y in pts)
    parts = [f'<path d="{d}" fill="none" stroke="{color}" stroke-width="2" '
             f'stroke-linecap="round" stroke-linejoin="round" '
             f'stroke-dasharray="{_f(dash)} 4" stroke-opacity="0.95"/>']
    if late:
        parts.append(f'<circle cx="{_f(sx)}" cy="{_f(sy)}" r="3.2" fill="{BG}" stroke="{color}" '
                     f'stroke-width="1.6"/>')
    else:
        parts.append(f'<circle cx="{_f(sx)}" cy="{_f(sy)}" r="3.2" fill="{color}"/>')
    ex, ey = pts[-1]
    j = len(pts) - 2
    while j > 0 and math.hypot(ex - pts[j][0], ey - pts[j][1]) < 4.0:
        j -= 1
    ang = math.atan2(ey - pts[j][1], ex - pts[j][0])
    size = 7.0
    tip = (ex, ey)
    left = (ex - size * math.cos(ang - 0.45), ey - size * math.sin(ang - 0.45))
    right = (ex - size * math.cos(ang + 0.45), ey - size * math.sin(ang + 0.45))
    parts.append(f'<path d="M{_f(tip[0])} {_f(tip[1])} L{_f(left[0])} {_f(left[1])} '
                 f'L{_f(right[0])} {_f(right[1])} Z" fill="{color}"/>')
    return parts


def scenario_record(s: dict) -> dict:
    """JSON-ready metadata and paths of one scenario, as written to ``catalog.json``."""
    return {
        'id': s['id'], 'name': s['name'], 'suite': s.get('suite'), 'family': s.get('family'),
        'description': s.get('description', ''), 'tags': list(s.get('tags', [])),
        'expected_dominant': s['expected_dominant'], 'vehicle': s.get('vehicle', 'bicycle'),
        'render': s.get('render', {}), 'duration_s': round(_duration(s), 3),
        'n_agents': len(s['agents']),
        'n_pedestrians': sum(a['type'] == 'person' for a in s['agents']),
        'n_cyclists': sum(a['type'] == 'bike' for a in s['agents']),
        'thumbnail': f"{s['id']}.svg",
        'agents': [{k: a[k] for k in ('type', 'id', 'model', 'label', 'path') if k in a}
                   for a in s['agents']],
    }


def _family_names() -> dict[str, str]:
    from .conformance import FAMILIES as CONF
    from .extended import FAMILIES as EXT

    return {**CONF, **EXT, 'dense': 'Dense scenes'}


def export_catalog(out_dir: str | FsPath, suites: Iterable[str] | None = None) -> dict:
    """Write ``catalog.json`` and ``<id>.svg`` for every scenario of ``suites`` into ``out_dir``.

    Returns ``{'catalog': path, 'thumbnails': n, 'scenarios': n}``.
    """
    out = FsPath(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    names = list(suites) if suites is not None else list_suites()
    records = []
    for name in names:
        for s in get_suite(name):
            (out / f"{s['id']}.svg").write_text(thumbnail_svg(s), encoding='utf-8')
            records.append(scenario_record(s))
    catalog = {
        'frame': 'metres; x along the road, y across it; crosswalk centre at (5, 0)',
        'suites': {n: [r['id'] for r in records if r['suite'] == n] for n in names},
        'families': _family_names(),
        'scenarios': records,
    }
    path = out / 'catalog.json'
    path.write_text(json.dumps(catalog, indent=1) + '\n', encoding='utf-8')
    return {'catalog': str(path), 'thumbnails': len(records), 'scenarios': len(records)}


def export_simulator(sim_dir: str | FsPath, suites: Iterable[str] = ('extended', 'stress')
                     ) -> str:
    """Write ``scenarios.extended.js`` for the browser simulator in ``sim_dir``.

    The file defines ``window.BIKEPED_SUITES = {suite: [scenario, ...]}`` with the simulator's
    own scenario keys (``name``, ``expected``, ``challenge``, ``agents``) plus ``id``, ``family``
    and ``vehicle``, so the page loads it with a plain ``<script src>`` even from ``file://``.
    """
    data = {}
    for name in suites:
        data[name] = [{
            'id': s['id'], 'name': s['name'], 'expected': s['expected_dominant'],
            'family': s.get('family', ''), 'challenge': s.get('description', ''),
            'vehicle': s.get('vehicle', 'bicycle'),
            'agents': [{'type': a['type'], 'id': a['id'], 'path': a['path']}
                       for a in s['agents']],
        } for s in get_suite(name)]
    path = FsPath(sim_dir) / 'scenarios.extended.js'
    path.write_text('// Extended scenario suites for index.html, written by '
                    '"python -m bikeped.scenarios export --simulator".\n'
                    'window.BIKEPED_SUITES = ' + json.dumps(data, separators=(',', ':')) + ';\n',
                    encoding='utf-8')
    return str(path)


def catalog_markdown(suite: str, thumb_prefix: str = '../assets/scenarios/') -> str:
    """Markdown table of a suite: thumbnail, name, family, agents, duration, expected state."""
    fam = _family_names()
    rows = ['| | Scenario | Family | Agents | Duration | Expected |',
            '|---|---|---|---|---|---|']
    for s in get_suite(suite):
        rows.append(
            f"| ![{s['name']}]({thumb_prefix}{s['id']}.svg) | "
            f"**{s['name']}**<br>`{s['id']}`<br>{s.get('description', '')} | "
            f"{fam.get(s.get('family', ''), s.get('family', ''))} | {len(s['agents'])} | "
            f"{_duration(s):.1f} s | `{s['expected_dominant']}` |")
    return '\n'.join(rows) + '\n'
