# Changelog

Changes to the repository, newest first. The project has no tagged releases; entries are grouped by date from the commit history.

## Unreleased

- Documentation site rebuilt with MkDocs Material from `docs_src/`, with pages on the system, scenarios, simulators, experiments, deployment and reference.
- `bikeped.scenarios`: scenario registry with the paper's 24 conformance scenarios unchanged, an `extended` suite of 41 scenarios, a `stress` suite of 8 dense scenes, six seeded generator families and a command line (`python -m bikeped.scenarios`).
- `bikeped.sim`: CARLA renderer for Unreal Engine 4 (CARLA 0.9.14 to 0.9.16) and Unreal Engine 5 (CARLA 0.10.0) with six camera views, the deployed fisheye rendered from cube faces, weather and ambient traffic presets, per-frame ground truth and COCO/YOLO export. It replaces the single-file runner; `carla_scenario.py` forwards its old options to `python -m bikeped.sim render`.
- Browser simulator restyled; it loads extra scenario suites from `simulator/scenarios.extended.js` when the file is present.
- Copies of the calibration scripts and tools in the repository root removed. They live in `calibration/` and `tools/`.

## 2026-05-06

- Documentation and calibration updates.
- Documentation fixes.

## 2026-05-01

- Repository refactored; `bridge_starter/` deployment kit added.
- README and documentation updates.

## 2026-04-17

- Code, testbench and scripts added.

## 2026-04-16

- Initial commit.
