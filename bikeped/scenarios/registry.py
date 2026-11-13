"""Scenario registry: named suites and lookup by slug, name or ``suite:index``.

Suites are built lazily on first use and cached. ``conformance`` is the paper's 24 scenarios from
``decision_testbench.SCENARIOS``; ``extended`` and ``stress`` are defined in this package.
Callers receive the cached dicts, so treat them as read-only (copy before editing).
"""

from __future__ import annotations

from functools import lru_cache
from typing import Callable

SUITE_ORDER = ('conformance', 'extended', 'stress')


def _builders() -> dict[str, Callable[[], list[dict]]]:
    from .conformance import conformance_suite
    from .extended import extended_suite
    from .stress import stress_suite

    return {'conformance': conformance_suite, 'extended': extended_suite,
            'stress': stress_suite}


def list_suites() -> list[str]:
    """Names of the registered suites, in display order."""
    return list(SUITE_ORDER)


@lru_cache(maxsize=None)
def _suite(name: str) -> tuple[dict, ...]:
    builders = _builders()
    if name not in builders:
        raise KeyError(f'unknown suite {name!r}; known suites: {", ".join(SUITE_ORDER)}')
    return tuple(builders[name]())


def get_suite(name: str) -> list[dict]:
    """Scenarios of suite ``name`` in their defined order (for ``conformance``, paper order)."""
    return list(_suite(name))


def all_scenarios() -> list[dict]:
    """Every scenario of every registered suite."""
    return [s for name in SUITE_ORDER for s in _suite(name)]


def get_scenario(key: str) -> dict:
    """Look up one scenario.

    ``key`` is a slug (``'fast-approach'``), an exact display name (``'Fast Approach'``) or
    ``'suite:index'`` with a zero-based index (``'conformance:7'`` is Fast Approach).
    """
    if ':' in key:
        suite, _, idx = key.partition(':')
        if idx.lstrip('-').isdigit():
            scenarios = _suite(suite)
            i = int(idx)
            if not 0 <= i < len(scenarios):
                raise KeyError(f'{key!r}: suite {suite!r} has {len(scenarios)} scenarios')
            return scenarios[i]
    for s in all_scenarios():
        if s['id'] == key or s['name'] == key:
            return s
    raise KeyError(f'no scenario with id or name {key!r}')
