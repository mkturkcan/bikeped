"""bikeped: simulation and scenario tooling for the bike-pedestrian warning system.

The decision logic and the offline testbench live in the top-level modules
``decision_pipeline`` and ``decision_testbench``. This package adds two things
on top of them:

``bikeped.scenarios``
    The scenario library: the 24 conformance scenarios from the paper plus the
    extended suites, path builders, parametric generators and a catalog export.

``bikeped.sim``
    The CARLA renderer for Unreal Engine 4 (CARLA 0.9.x) and Unreal Engine 5
    (CARLA 0.10.x): camera rigs including a calibrated fisheye, weather
    presets, ground truth, overlays and video output.

Run the commands from the repository root so that the top-level modules are
importable.
"""

__version__ = "2.0.0"
