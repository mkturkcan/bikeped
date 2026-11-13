"""Catalog export, evaluator, CLI, and the pinned default-pipeline outcome on the paper's 24."""

from __future__ import annotations

import json
import xml.etree.ElementTree as ET

import pytest

import decision_testbench as tb
from bikeped.scenarios import get_suite
from bikeped.scenarios.catalog import catalog_markdown, export_catalog, export_simulator
from bikeped.scenarios.cli import main
from bikeped.scenarios.evaluate import dominant_state, evaluate_suite, format_table, is_known_hard

# Dominant state of the default pipeline (ideal perception) on each conformance scenario,
# as decision_testbench produces it. A change here means the pipeline's behaviour changed.
PINNED_DOMINANT = {
    'Safe Crossing': 'safe', 'Bike Approaches Ped': 'alert', 'High-Speed Near Miss': 'alert',
    'Multiple VRUs': 'alert', 'Occluded Emergence': 'alert', 'Parallel Paths': 'warning',
    'Bikes Only': 'idle', 'Fast Approach': 'alert', 'Speed Variant': 'alert',
    'Counter-Flow on Crosswalk': 'alert', 'Multi-Speed Converging': 'alert',
    'Wheelchair User': 'alert', 'Stopped Pedestrian': 'alert', 'Cross-Street Cyclist': 'alert',
    'E-Scooter': 'alert', 'Child Pedestrian': 'alert', 'Overtake from Behind': 'alert',
    'Cyclist Abort': 'alert', 'Dense Pedestrian Group': 'alert', 'Diagonal Jaywalker': 'alert',
    'Swerving Cyclist': 'alert', 'Late Turn Toward Ped': 'alert', 'U-Turn Cyclist': 'alert',
    'Accelerating E-Bike': 'alert',
}


def test_export_catalog(tmp_path):
    info = export_catalog(tmp_path, suites=['conformance', 'stress'])
    catalog = json.loads((tmp_path / 'catalog.json').read_text())
    assert len(catalog['scenarios']) == info['scenarios'] == 32
    for rec in catalog['scenarios']:
        root = ET.parse(tmp_path / rec['thumbnail']).getroot()
        assert root.get('viewBox') == '0 0 320 200'
        assert rec['n_agents'] == len(rec['agents'])


def test_export_simulator(tmp_path):
    path = export_simulator(tmp_path)
    text = open(path).read()
    data = json.loads(text.split('=', 1)[1].strip().rstrip(';'))
    assert set(data) == {'extended', 'stress'}
    assert data['extended'][0]['agents'][0]['path']


def test_markdown():
    md = catalog_markdown('conformance')
    assert md.count('\n') == 26 and 'fast-approach.svg' in md


def test_pinned_conformance_dominant():
    got = {s['name']: dominant_state(tb.simulate_scenario(s, tb.PIPELINE_CONFIGS['default'])
                                     ['states']) for s in tb.SCENARIOS}
    assert got == PINNED_DOMINANT


@pytest.fixture(scope='module')
def extended_rows():
    return evaluate_suite(get_suite('extended'))


def test_extended_expected_is_ground_truth(extended_rows):
    for r in extended_rows:
        assert r['expected'] == r['ground_truth'], r['id']


def test_known_hard_tags_match_evaluation(extended_rows):
    for r in extended_rows:
        assert ('known-hard' in r['tags']) == is_known_hard(r), r['id']
    assert 'scenarios' in format_table(extended_rows)


def test_cli_smoke(tmp_path, capsys):
    assert main(['list', '--suite', 'conformance']) == 0
    assert main(['show', 'conformance:0']) == 0
    assert main(['validate', '--suite', 'stress']) == 0
    out = tmp_path / 'g.json'
    assert main(['generate', '--family', 'time-gap', '--n', '3', '--seed', '0',
                 '--out', str(out)]) == 0
    assert len(json.loads(out.read_text())) == 3
    capsys.readouterr()
