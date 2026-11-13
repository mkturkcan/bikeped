"""Run scenarios through the testbench decision pipeline and summarise the outcome.

Everything is computed by ``decision_testbench.simulate_scenario`` with ideal perception (the
noise-free path, tracker on, no latency). This module only aggregates its per-frame ``states`` and
ground-truth ``dangers`` into per-scenario figures; the metrics themselves (danger labels,
false positives, sensitivity) are the testbench's.

The *dominant state* of a run is the most severe state (``alert`` > ``warning`` > ``safe`` >
``idle``) that the pipeline holds for at least :data:`MIN_STATE_S` seconds in total. The same
rule applied to the ground truth gives :func:`ground_truth_label`: ``idle`` without pedestrians,
``safe`` without cyclists, ``alert`` with at least :data:`MIN_STATE_S` of danger, else
``warning``.
"""

from __future__ import annotations

import json
import math
from typing import Iterable, Sequence

import numpy as np

SEVERITY = ('idle', 'safe', 'warning', 'alert')
MIN_STATE_S = 0.3
LEAD_WINDOW_S = 3.0
LATE_ALERT_S = 0.5
FPS = 30


def _tb():
    import decision_testbench as tb

    return tb


def dominant_state(states: Sequence[str], fps: int = FPS, min_s: float = MIN_STATE_S) -> str:
    """Most severe state held for at least ``min_s`` seconds in total (``idle`` if none)."""
    states = np.asarray(states)
    for state in reversed(SEVERITY):
        if np.sum(states == state) / fps >= min_s:
            return state
    return 'idle'


def ground_truth_label(scenario: dict, dangers: np.ndarray, fps: int = FPS,
                       min_s: float = MIN_STATE_S) -> str:
    """State the ground truth calls for: see the module docstring for the rule."""
    types = {a['type'] for a in scenario['agents']}
    if 'person' not in types:
        return 'idle'
    if 'bike' not in types:
        return 'safe'
    return 'alert' if np.sum(dangers) / fps >= min_s else 'warning'


def label_scenario(scenario: dict, fps: int = FPS) -> str:
    """:func:`ground_truth_label` from ``compute_ground_truth_danger`` alone (no pipeline run).

    The frame times match ``simulate_scenario``: ``t = k / fps`` up to the scenario duration.
    """
    tb = _tb()
    duration = tb.get_scenario_duration(scenario)
    n = int(math.floor(duration * fps + 1e-9)) + 1
    dangers = np.array([tb.compute_ground_truth_danger(scenario, k / fps).is_danger
                        for k in range(n)], dtype=bool)
    return ground_truth_label(scenario, dangers, fps)


def _runs(mask: np.ndarray) -> list[tuple[int, int]]:
    """Contiguous ``True`` runs of ``mask`` as ``(start, end_exclusive)`` index pairs."""
    padded = np.concatenate([[False], mask.astype(bool), [False]])
    edges = np.flatnonzero(np.diff(padded.astype(int)))
    return list(zip(edges[::2].tolist(), edges[1::2].tolist()))


def evaluate_scenario(scenario: dict, config: str = 'default', fps: int = FPS) -> dict:
    """Simulate one scenario and summarise it.

    Returned keys: ``id``, ``name``, ``suite``, ``family``, ``duration_s``, ``agents``,
    ``danger_s`` and ``actionable_s`` (ground-truth time), ``alert_s``, ``false_alert_s`` (alert
    on frames the ground truth calls safe), ``danger_intervals`` (one dict per interval with
    ``start``, ``end``, ``alerted`` and ``lead_s``: seconds from alert onset to danger onset,
    negative when the alert came late), ``sensitivity``, ``specificity``, ``dominant``,
    ``expected``, ``ground_truth`` and ``match``.
    """
    tb = _tb()
    params = tb.PIPELINE_CONFIGS[config]
    r = tb.simulate_scenario(scenario, params, fps=fps)
    times, states, dangers = r['times'], r['states'], r['dangers']
    dt = 1.0 / fps
    is_alert = states == 'alert'
    actionable = np.array([a.actionable for a in r['assessments']], dtype=bool)
    intervals = []
    for i0, i1 in _runs(dangers):
        lo = max(0, i0 - int(round(LEAD_WINDOW_S * fps)))
        hits = np.flatnonzero(is_alert[lo:i1])
        if len(hits):
            first = lo + int(hits[0])
            while first > 0 and is_alert[first - 1]:
                first -= 1
            lead = float(times[i0] - times[first])
            intervals.append({'start': round(float(times[i0]), 3),
                              'end': round(float(times[i1 - 1]) + dt, 3),
                              'alerted': True, 'lead_s': round(lead, 3)})
        else:
            intervals.append({'start': round(float(times[i0]), 3),
                              'end': round(float(times[i1 - 1]) + dt, 3),
                              'alerted': False, 'lead_s': None})
    dominant = dominant_state(states, fps)
    expected = scenario.get('expected_dominant')
    return {
        'id': scenario.get('id', ''),
        'name': scenario['name'],
        'suite': scenario.get('suite', ''),
        'family': scenario.get('family', ''),
        'duration_s': round(float(tb.get_scenario_duration(scenario)), 3),
        'agents': len(scenario['agents']),
        'danger_s': round(float(np.sum(dangers)) * dt, 3),
        'actionable_s': round(float(np.sum(actionable)) * dt, 3),
        'alert_s': round(float(np.sum(is_alert)) * dt, 3),
        'false_alert_s': round(r['false_pos'] * dt, 3),
        'danger_intervals': intervals,
        'sensitivity': round(float(r['sensitivity']), 4),
        'specificity': round(float(r['specificity']), 4),
        'dominant': dominant,
        'expected': expected,
        'ground_truth': ground_truth_label(scenario, dangers, fps),
        'match': dominant == expected,
        'tags': list(scenario.get('tags', [])),
    }


def is_known_hard(row: dict) -> bool:
    """True when the pipeline's dominant state is wrong or an alert is late or missing.

    Late means the alert began more than :data:`LATE_ALERT_S` seconds after the danger interval.
    """
    if not row['match']:
        return True
    return any(not d['alerted'] or d['lead_s'] < -LATE_ALERT_S for d in row['danger_intervals'])


def evaluate_suite(scenarios: Iterable[dict], config: str = 'default',
                   fps: int = FPS) -> list[dict]:
    """:func:`evaluate_scenario` for each scenario."""
    return [evaluate_scenario(s, config, fps) for s in scenarios]


def _fmt_lead(iv: list[dict]) -> str:
    if not iv:
        return '-'
    parts = []
    for d in iv:
        parts.append('miss' if not d['alerted'] else f"{d['lead_s']:+.1f}")
    return ','.join(parts)


def format_table(rows: Sequence[dict]) -> str:
    """Plain-text table of :func:`evaluate_scenario` rows with a totals line."""
    head = (f"{'scenario':<34} {'dur':>5} {'ag':>3} {'danger':>6} {'alert':>6} {'false':>6} "
            f"{'lead s (per interval)':<22} {'dominant':>8} {'expected':>8}  ok")
    lines = [head, '-' * len(head)]
    for r in rows:
        lines.append(
            f"{r['id'][:34]:<34} {r['duration_s']:>5.1f} {r['agents']:>3} {r['danger_s']:>6.2f} "
            f"{r['alert_s']:>6.2f} {r['false_alert_s']:>6.2f} "
            f"{_fmt_lead(r['danger_intervals'])[:22]:<22} {r['dominant']:>8} "
            f"{str(r['expected']):>8}  {'yes' if r['match'] else 'NO'}")
    n_iv = sum(len(r['danger_intervals']) for r in rows)
    n_ok = sum(1 for r in rows for d in r['danger_intervals']
               if d['alerted'] and d['lead_s'] >= -1.0 / FPS - 1e-9)
    lines.append('-' * len(head))
    lines.append(f"{len(rows)} scenarios, dominant state matches expected in "
                 f"{sum(r['match'] for r in rows)}; {n_ok} of {n_iv} danger intervals alerted "
                 f"no later than one frame after onset; false-alert time "
                 f"{sum(r['false_alert_s'] for r in rows):.2f} s")
    return '\n'.join(lines)


def to_json(rows: Sequence[dict]) -> str:
    """JSON text of evaluation rows (non-finite numbers become ``null``)."""
    def clean(v):
        if isinstance(v, float) and not math.isfinite(v):
            return None
        if isinstance(v, dict):
            return {k: clean(x) for k, x in v.items()}
        if isinstance(v, list):
            return [clean(x) for x in v]
        return v

    return json.dumps([clean(r) for r in rows], indent=2)
