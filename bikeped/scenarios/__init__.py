"""Scenario library for the bike-pedestrian decision pipeline.

Scenarios use the format of ``decision_testbench.SCENARIOS``: agents with piecewise-linear
waypoint paths in metres and seconds, ``x`` along the road, ``y`` across it, crosswalk centre at
``(5, 0)``. The library adds metadata (``id``, ``suite``, ``family``, ``description``, ``tags``,
``render`` and per-agent ``model``) that the testbench ignores.

Suites: ``conformance`` (the paper's 24, unchanged), ``extended`` (hand-designed urban conflicts
and negatives) and ``stress`` (dense scenes). :mod:`bikeped.scenarios.generators` produces
parametric suites on demand. Command line: ``python -m bikeped.scenarios --help``.
"""

from __future__ import annotations

from .registry import all_scenarios, get_scenario, get_suite, list_suites

__all__ = ['all_scenarios', 'get_scenario', 'get_suite', 'list_suites']
