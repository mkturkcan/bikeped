# Simulation

The repository has three simulators that replay the same scripted scenarios through the same decision logic. This page says what each one models, what it needs to run, and which one to reach for.

All three read scenarios in the [scenario format](../reference/formats.md) and evaluate them with `DecisionPipeline` from `decision_pipeline.py`, so a scenario behaves the same way in each. They differ in how much of the perception chain sits between the scripted positions and the pipeline.

<div class="grid-cards" markdown>

- [Browser simulator](browser.md)

    A single HTML file. Plays a scenario in a bird's-eye view while the decision flowchart lights up, with sliders for camera placement and detector uncertainty.

- [Offline testbench](testbench.md)

    `decision_testbench.py` scores every scenario against ground-truth danger labels and prints the tables reported in the paper. `sim_visualizer.py` renders the same runs as schematic fisheye videos.

- [CARLA renderer](carla/index.md)

    `python -m bikeped.sim` places the scripted agents at a crosswalk in a CARLA town and records photorealistic video with per-frame ground truth.

</div>

## Comparison

| | Browser simulator | Offline testbench | CARLA renderer |
|---|---|---|---|
| Entry point | `simulator/index.html` | `python decision_testbench.py`, `python sim_visualizer.py` | `python -m bikeped.sim render` |
| Needs | A browser | Python 3.10+ and `requirements.txt` | A CARLA server (UE4 or UE5 build), the `carla` wheel and a GPU |
| Perception model | Adjustable detector recall, latency and tracker settings | Calibrated fisheye projection, bounding-box localisation error, size-aware detection drops, track persistence | Rendered images. The scripted positions drive the pipeline, the renderer supplies pixels and boxes |
| Ground truth | Expected dominant state per scenario | Per-frame danger assessment from the braking model | Per-frame danger assessment plus 2D and 3D boxes per camera view |
| Output | On-screen only | Metric tables on stdout, `latency_sweep.json`, figures in `paper/`, MP4s from `sim_visualizer.py` | A run directory with `manifest.json`, videos, ground-truth JSONL and stills |
| Deterministic | Yes, unless detection drops are enabled | Yes. Monte Carlo trials are seeded by trial index | Scripted agents are; rendering and ambient traffic depend on the server |
| Typical use | Explaining the logic, trying a camera position, comparing decision rules by eye | Reproducing the paper, tuning parameters, regression checks | Producing video and labelled frames for detector work and demos |

## Which one to use

Start in the browser when the question is "what does the pipeline do in this situation". It runs with no install, and switching between the seven pipeline variants takes one click.

Use the testbench when the answer has to be a number. It is the only simulator that scores alerts against the ground-truth danger model described in [Safety model](../system/safety-model.md), and every table in the paper comes from it. [Reproduce the paper](../experiments/index.md) lists the commands.

Use CARLA when pixels matter: checking how a scenario looks through the deployed lens, exporting labelled frames, or producing video. The renderer does not replace the testbench for metrics. It records the decision state and the danger label next to each frame so the two can be compared afterwards.

## Shared pieces

| Piece | Where it lives | Used by |
|---|---|---|
| Decision logic | `decision_pipeline.py` | Testbench and CARLA renderer import it. The browser simulator carries a JavaScript port |
| Conformance scenarios | `decision_testbench.SCENARIOS` | All three. The browser copy mirrors the Python list |
| Pipeline variants | `decision_testbench.PIPELINE_CONFIGS` | Testbench (`--compare`), `sim_visualizer.py --pipeline`, browser pipeline selector |
| Lens model | `camera_calibration.json`, `FisheyeCamera` | Testbench, `sim_visualizer.py`, the CARLA fisheye view |

## Related

- [Scenario catalog](../scenarios/index.md)
- [Decision pipeline](../system/decision.md)
- [Command-line reference](../reference/cli.md)
