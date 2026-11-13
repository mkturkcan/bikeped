# How the system works

bikeped watches one crosswalk through a pole-mounted fisheye camera and sets a
warning light from the ground positions of the pedestrians and cyclists it
sees. This page follows one frame from the sensor to the light and names the
page that covers each step.

<figure class="diagram">
--8<-- "assets/figures/system-overview.svg"
<figcaption>The live path. Every stage runs once per frame on the edge device.</figcaption>
</figure>

## One frame, step by step

| Step | What happens | Where it is configured |
|---|---|---|
| Capture | The camera delivers a 3840 × 2160 frame holding a fisheye image circle that is taller than the sensor. | `video` in `config.yaml` |
| Crop | The frame is padded to 3840 × 3840 and centre-cropped to 3500 × 3500, which puts the 1750 px image circle in the middle of a square frame. | `preprocessing.pad_size`, `preprocessing.crop_size` |
| Detect | A YOLO detector runs on the square frame at 1280 px input size. In the bridge, a `person` box that overlaps a bike box is relabelled as the rider. | `detection` in `config.yaml`, `RIDER_*` in `bridge.py` |
| Locate | The bottom-centre pixel of each box is sent back through the lens model onto the ground plane, giving `(x, y)` in metres relative to the camera. The bridge precomputes this as a lookup table at start-up. | `camera` in `config.yaml`, `camera_calibration.json` |
| Track | A ground-plane tracker matches detections to tracks by metric distance and carries a track through missed detections with constant-velocity prediction. | `GroundPlaneTracker` |
| Decide | `DecisionPipeline` evaluates three stages and returns one of four states. | `decision` in `config.yaml` |
| Act | The state sets the light colour, optionally plays a sound, and the tracks are published over MQTT. | `feedback`, `mqtt` in `config.yaml` |

The geometry of the Locate step is on the [perception](perception.md) page and
the Decide step on the [decision pipeline](decision.md) page.

## The four states

| State | Condition | Light | Sound in the bridge |
|---|---|---|---|
| <span class="state state-idle">idle</span> | No pedestrian is tracked. | Blue | |
| <span class="state state-safe">safe</span> | A pedestrian is present and no cyclist has been seen within the memory window. | Green | Safe cue, if configured |
| <span class="state state-warning">warning</span> | A pedestrian and a recent cyclist are present, and no pair is closing. | Orange | |
| <span class="state state-alert">alert</span> | A cyclist is closing on a pedestrian inside the distance band. | Red | Alert cue, if configured |

The colour mapping is `apply_feedback` in
[`bridge_starter/bridge.py`](https://github.com/mkturkcan/bikeped/blob/main/bridge_starter/bridge.py).
The starter kit ships a `SimulatedLight` that prints each colour change, so the
whole path runs without indicator hardware. `sim_visualizer.py` draws the
states in the same four colours.

## Live and offline paths share the decision code

`decision_pipeline.py` has no dependency beyond the standard library and is
imported unchanged by the bridge and by `decision_testbench.py`. The testbench
replaces the first three steps with scripted agent paths and then puts back the
imperfections of a real detector: it projects each agent's 3D box through the
same fisheye model to get the bounding-box localisation error, drops detections
according to measured recall, and runs the same kind of ground-plane tracker
before calling the pipeline. Ground truth for each frame comes from a kinematic
[safety model](safety-model.md) that is independent of the pipeline.

That arrangement is why a parameter change in `config.yaml` can be evaluated
on the 24 conformance scenarios before it reaches an intersection. See
[the testbench](../simulation/testbench.md) for how to run it.

## In this section

<div class="grid-cards" markdown>

- [Perception geometry](perception.md)

    The fisheye lens model, the sensor crop, ground-plane back-projection and the bounding-box error.

- [Decision pipeline](decision.md)

    The three stages as coded, every parameter, and the baseline decision rules.

- [Safety model](safety-model.md)

    Braking profiles, pedestrian reaction time and how a frame is labelled dangerous.

- [Detection and localisation uncertainty](uncertainty.md)

    Missed-detection models, position uncertainty and the tracker used offline.

</div>
