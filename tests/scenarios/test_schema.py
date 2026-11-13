"""Every scenario in every suite validates; the validator catches broken input."""

from __future__ import annotations

import copy

import pytest

from bikeped.scenarios import get_scenario, get_suite, list_suites
from bikeped.scenarios.schema import validate_scenario, validate_suites


@pytest.mark.parametrize('suite', list_suites())
def test_suite_validates(suite):
    assert validate_suites(get_suite(suite)) == []


def _broken(mutate):
    s = copy.deepcopy(get_scenario('bike-approaches-ped'))
    mutate(s)
    return validate_scenario(s)


def test_detects_time_order():
    assert _broken(lambda s: s['agents'][0]['path'].reverse())


def test_detects_duplicate_agent_ids():
    assert _broken(lambda s: s['agents'][1].update(id=1))


def test_detects_implausible_pedestrian_speed():
    def fast(s):
        s['agents'][0]['model'] = 'adult'
        s['agents'][0]['path'][-1]['x'] = 60
    assert any('speed' in p for p in _broken(fast))


def test_detects_bad_metadata():
    assert _broken(lambda s: s.update(expected_dominant='panic'))
    assert _broken(lambda s: s.update(render={'weather': 'snowstorm'}))
    assert _broken(lambda s: s['agents'][0]['path'][1].update(x=float('nan')))
