# Start here

bikeped is a collision warning system for crossings shared by pedestrians and cyclists. This page says what the project is, what the repository holds, and where each topic is documented.

## What the project is

A level fisheye camera on a pole faces across one crosswalk. A Jetson Orin next to it runs an object detector, a ground-plane tracker and a three-stage decision pipeline at 30 FPS, and drives a warning light. Each frame ends in one of four states: <span class="state state-idle">idle</span> when no pedestrian is present, <span class="state state-safe">safe</span> when a pedestrian is present and no cyclist has been seen in the last 58 frames (about 2 s), <span class="state state-warning">warning</span> when both are present, and <span class="state state-alert">alert</span> when a cyclist and a pedestrian are closing on each other.

The repository is the companion code of the paper [A Real-Time Bike-Pedestrian Safety System with Wide-Angle Perception and Evaluation Testbed for Urban Intersections](https://arxiv.org/abs/2604.17046). The decision logic in it is the file that runs on the device. Around that file sits an offline testbench that replays 24 scripted conformance scenarios through the same logic and produces every number in the paper, so a change to a threshold can be evaluated on a laptop before it reaches an intersection.

## What is in the repository

| Path | Contents |
|---|---|
| `decision_pipeline.py` | The three-stage decision logic, imported by both the deployed bridge and the testbench. |
| `decision_testbench.py` | Offline scenario runner: fisheye camera model, detection failure and localisation error models, ground-truth danger labelling, optimizer, Monte Carlo, latency sweep. |
| `bikeped/scenarios/` | Scenario library: the 24 conformance scenarios and the suites added next to them. |
| `bikeped/sim/` | CARLA renderer for Unreal Engine 4 and Unreal Engine 5 builds. |
| `simulator/index.html` | Browser simulator in a single HTML file. |
| `sim_visualizer.py` | Schematic 3D renderer that writes one MP4 per scenario. |
| `calibration/` | Fisheye intrinsic calibration from checkerboard frames. |
| `bridge_starter/` | Deployment starter kit: one script with a USER CONFIG block. |
| `run_experiments.py` | Runs every experiment in the paper in sequence. |
| `config.yaml`, `camera_calibration.json` | Runtime parameters and calibrated intrinsics of the deployed system. |

The [repository layout](repository.md) page lists every file.

## First steps

<div class="grid-cards" markdown>

- [Install](install.md)

    Python 3.10 or newer, one requirements file, and the optional pieces for CARLA, the bridge kit and the docs.

- [Quickstart](quickstart.md)

    Run the testbench, compare decision rules, render a scenario and call the pipeline from Python.

- [Repository layout](repository.md)

    An annotated tree, and which files the paper numbers come from.

</div>

## Understand the system

<div class="grid-cards" markdown>

- [How it works](../system/index.md)

    The path of one frame from the camera to the warning light.

- [Perception](../system/perception.md)

    The fisheye model, ground-plane localisation and the bounding-box error model.

- [Decision pipeline](../system/decision.md)

    The three stages, their parameters and the decision-rule baselines.

- [Safety model](../system/safety-model.md)

    Braking profiles and how a frame is labelled dangerous.

- [Uncertainty](../system/uncertainty.md)

    The detection failure and localisation noise models used by the testbench.

</div>

## Run simulations

<div class="grid-cards" markdown>

- [Simulators](../simulation/index.md)

    The three simulators and when to use which.

- [Scenarios](../scenarios/index.md)

    The conformance suite, the extended suites, and how to write or generate your own.

- [Browser simulator](../simulation/browser.md)

    The decision pipeline running live in a web page, with nothing to install.

- [Testbench](../simulation/testbench.md)

    The offline runner behind the tables in the paper.

- [CARLA](../simulation/carla/index.md)

    Photorealistic renders of the same scenarios through the calibrated lens model, on UE4 and UE5.

</div>

## Reproduce the paper

<div class="grid-cards" markdown>

- [Experiments](../experiments/index.md)

    One command for everything, and one command per table and figure.

- [Optimizer](../experiments/optimizer.md)

    How the selected parameters were found by differential evolution.

- [Latency](../experiments/latency.md)

    The camera latency sweep and the per-step latency report.

- [Placement](../experiments/placement.md)

    Camera height, pitch and road width.

- [Dataset](../dataset.md)

    The scenarios, per-frame labels and videos published on Hugging Face.

</div>

## Deploy

<div class="grid-cards" markdown>

- [Calibration](../calibration/index.md)

    Capture checkerboard frames and fit the lens model.

- [Bridge starter kit](../deployment/index.md)

    A self-contained live system to adapt to a new intersection.

- [Tuning](../deployment/tuning.md)

    The proximity floor and rider suppression for your detector and camera geometry.

- [Hardware](../deployment/hardware.md)

    The camera, the compute and the indicator light.

</div>

## Look something up

<div class="grid-cards" markdown>

- [Command line](../reference/cli.md)

    Every script and module entry point with its flags.

- [Configuration](../reference/config.md)

    Every key of `config.yaml` and `camera_calibration.json`.

- [File formats](../reference/formats.md)

    Scenario dictionaries, render manifests, ground-truth files and the catalog.

- [Python API](../api/index.md)

    Module reference built from the docstrings.

- [FAQ](../faq.md)

    Short answers to questions that come up when running or adapting the system.

- [Changelog](../changelog.md)

    What changed, by date.

- [Citing](../citing.md)

    The BibTeX entry and the abstract.

</div>
