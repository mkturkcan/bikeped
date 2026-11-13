"""Registry integrity: the conformance suite is the paper's 24 scenarios, untouched."""

from __future__ import annotations

import pytest

import decision_testbench as tb
from bikeped.scenarios import all_scenarios, get_scenario, get_suite, list_suites
from bikeped.scenarios.conformance import slugify

CORE_KEYS = ('name', 'expected_dominant', 'vehicle')


def test_suites_listed():
    assert list_suites()[:3] == ['conformance', 'extended', 'stress']


def test_conformance_is_testbench_in_paper_order():
    suite = get_suite('conformance')
    assert len(suite) == 24 == len(tb.SCENARIOS)
    for ours, orig in zip(suite, tb.SCENARIOS):
        for k in CORE_KEYS:
            assert ours.get(k) == orig.get(k)
        assert len(ours['agents']) == len(orig['agents'])
        for a, b in zip(ours['agents'], orig['agents']):
            assert (a['type'], a['id'], a['path']) == (b['type'], b['id'], b['path'])
            assert 'model' not in b
        assert ours['suite'] == 'conformance'
        assert ours['id'] == slugify(orig['name'])


def test_originals_not_decorated():
    get_suite('conformance')
    for s in tb.SCENARIOS:
        assert not {'id', 'suite', 'family', 'tags', 'description'} & set(s)


def test_ids_and_names_unique():
    scenarios = all_scenarios()
    assert len({s['id'] for s in scenarios}) == len(scenarios)
    assert len({s['name'] for s in scenarios}) == len(scenarios)


def test_suite_sizes():
    assert len(get_suite('extended')) >= 36
    assert len(get_suite('stress')) >= 8


@pytest.mark.parametrize('key', ['fast-approach', 'Fast Approach', 'conformance:7'])
def test_lookup(key):
    assert get_scenario(key)['name'] == 'Fast Approach'


def test_lookup_unknown():
    with pytest.raises(KeyError):
        get_scenario('no-such-scenario')
    with pytest.raises(KeyError):
        get_scenario('conformance:99')


def test_model_hints():
    assert get_scenario('wheelchair-user')['agents'][0]['model'] == 'wheelchair'
    assert get_scenario('e-scooter')['agents'][1]['model'] == 'escooter'
