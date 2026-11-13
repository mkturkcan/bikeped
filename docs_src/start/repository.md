# Repository layout

This page lists what is in the repository, one line per entry, and says which files produce the paper's numbers and which are copies.

```text
bikeped/
├── README.md
├── requirements.txt                 numpy, scipy, matplotlib, pyyaml, opencv-python; optional extras listed as comments
├── pyproject.toml
├── config.yaml                      runtime parameters of the deployed system
├── camera_calibration.json          calibrated fisheye intrinsics, read by every script
├── crosswalks.json                  crosswalk geometry extracted from the CARLA map Town10HD_Opt
│
├── decision_pipeline.py             three-stage decision module, shared with the live system
├── decision_testbench.py            offline testbench: scenarios, camera model, tracker, optimizer, Monte Carlo
├── sim_visualizer.py                renders schematic scenario MP4s through the fisheye model
├── generate_figures.py              publication figures
├── run_experiments.py               reproduces every table and figure in one command
├── run_height_pitch_sweep.py        camera height and pitch placement sweep
├── crosswalk_analysis.py            road-width and deployment geometry analysis
├── carla_scenario.py                old CARLA command line, forwards to python -m bikeped.sim render
│
├── bikeped/                         Python package
│   ├── scenarios/                   scenario registry: conformance, extended and stress suites, generators, CLI
│   └── sim/                         CARLA renderer for Unreal Engine 4 and 5, CLI
│
├── calibration/                     fisheye calibration
│   ├── capture_calibration.py       capture checkerboard frames from the camera
│   ├── calibrate_fisheye.py         perspective remap, corner detection, bundle adjustment
│   └── compare_fisheye_models.py    fit four projection models to the same frames
│
├── tools/
│   ├── build_docs.py                builds this site into docs/
│   ├── eval_models.py               evaluates detectors on fisheye-augmented COCO, writes eval_cache.json
│   ├── latency_report.py            per-step latency statistics from latency_data.npy
│   └── carla_find_crosswalks.py     extracts crosswalks.json from a running CARLA map
│
├── simulator/                       browser simulator (static HTML and JS)
├── bridge_starter/                  deployment starter kit
│   ├── bridge.py                    live bridge with a USER CONFIG block
│   ├── decision_pipeline.py         copy of the top-level decision_pipeline.py
│   └── requirements.txt
│
├── tests/                           pytest suite
├── docs_src/                        sources of this site
├── mkdocs.yml                       site configuration
└── docs/                            built site, served by GitHub Pages
```

## Sources of truth

Every number in the paper comes from four files: `decision_pipeline.py` (the decision logic), `decision_testbench.py` (scenarios, sensing model and metrics), `config.yaml` and `camera_calibration.json`. The rest of the repository imports from them. If you change one of the four, the paper's tables no longer reproduce.

`bridge_starter/decision_pipeline.py` is identical to the top-level `decision_pipeline.py` (`cmp bridge_starter/decision_pipeline.py decision_pipeline.py` prints nothing), kept so that the starter kit can be copied out of the repository on its own. Edit the top-level file and copy it over.

The calibration scripts live only in `calibration/` and the utilities only in `tools/`, so run them with their directory: `python calibration/calibrate_fisheye.py`, `python tools/latency_report.py`.

## Files that scripts write

Scripts read and write these files by relative path, so run them from the repository root. The files that one step writes and a later step reads are:

| File | Written by | Read by |
|---|---|---|
| `camera_calibration.json` | `calibration/calibrate_fisheye.py` | testbench, visualiser, bridge |
| `eval_cache.json` | `tools/eval_models.py` | `decision_testbench.py --eval-cache`, `generate_figures.py` |
| `height_pitch_sweep.json` | `run_height_pitch_sweep.py` | `generate_figures.py` |
| `latency_data.npy` | the bridge with latency profiling on | `tools/latency_report.py` |

## Related

- [Install](install.md)
- [Command line reference](../reference/cli.md)
- [Configuration reference](../reference/config.md)
