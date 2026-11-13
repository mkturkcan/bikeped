"""Generators are deterministic and produce valid scenarios."""

from __future__ import annotations

import pytest

from bikeped.scenarios.generators import FAMILIES, generate
from bikeped.scenarios.schema import validate_suites


@pytest.mark.parametrize('family', list(FAMILIES))
def test_deterministic_and_valid(family):
    a = generate(family, 5, seed=3)
    assert a == generate(family, 5, seed=3)
    assert len(a) == 5
    assert validate_suites(a) == []
    assert all(s['suite'] == f'generated-{family}' for s in a)


def test_seed_changes_output():
    assert generate('mixed-traffic', 3, 0) != generate('mixed-traffic', 3, 1)


def test_unknown_family():
    with pytest.raises(KeyError):
        generate('no-such-family', 3, 0)
