# Tuning for a new site

The shipped decision parameters were found by the optimizer for the reference camera and the conformance scenarios. This page covers the parameters to revisit for a different camera height, lens or traffic mix, and how to check a change in the testbench before it reaches the street.

Constant names are those of the `USER CONFIG` block in `bridge_starter/bridge.py`. The `decision` block of `config.yaml` has the same parameters under the names listed in the [configuration reference](../reference/config.md#decision).

## Parameters

| Parameter | Default | Unit | Effect of raising it |
|---|---|---|---|
| `DEC_PROX_MIN_M` | `1.9` | m | More rider self-pairs are ignored. Real pairs closer than this never alert. |
| `DEC_PROX_MAX_M` | `24.8` | m | Pairs that are farther apart become eligible, so alerts start earlier and last longer. |
| `DEC_BIKE_MEMORY_FRAMES` | `58` | frames | Longer detection gaps are bridged. The state also stays at `warning` longer after a cyclist has left. |
| `DEC_PROX_SPEED_MIN` | `0.147` | m | Slower cyclists stop producing alerts. |
| `DEC_LOOKBACK_FRAMES` | `2` | frames | The closing check compares against an older position. |
| `DEC_MAX_RANGE_M` | `25.0` | m | Agents farther from the camera take part in stage 3. |
| `DEC_REACTION_S`, `DEC_DECEL_MS2` | `0.84`, `1.96` | s, m/s² | A longer reaction time lengthens the stopping distance. A higher deceleration shortens it. |
| `RIDER_X_OVERLAP_FRAC` | `0.6` | | More `person` boxes near a bike are treated as riders. |
| `RIDER_Y_OVERLAP_FRAC` | `0.3` | | Same, along the vertical image axis. |

## Rider-on-bike suppression

A detector usually reports a `person` box for the cyclist's torso on top of the `bicycle` or `motorcycle` box. On the ground plane the two land 0.3 to 0.7 m apart, so without a filter every cyclist would form a closing pair with their own body. The kit handles this in two layers.

The first layer is `find_rider_indices` in `bridge.py`. A `person` box is a rider when it overlaps a bicycle or motorcycle box horizontally and its bottom edge reaches the upper part of the bike box:

```python
abs(person_cx - bike_cx) < RIDER_X_OVERLAP_FRAC * max(person_w, bike_w)
and person_bottom >= bike_top - RIDER_Y_OVERLAP_FRAC * bike_height
```

The main loop relabels such boxes as class 1 before tracking, so the rider counts as a cyclist.

The second layer is the proximity floor. Stage 3 skips any pair closer than `DEC_PROX_MIN_M`, which catches rider boxes that the overlap test missed.

The kit's README lists the symptoms that call for a change:

| Symptom | Likely cause | Fix |
|---|---|---|
| A cyclist trips the alert alone at a low camera height | The bottom of the person box and the bottom of the bike box project to different ground points under barrel distortion | Raise `DEC_PROX_MIN_M` to 2.5 to 3.0 m |
| A walking adult next to a parked bike trips the alert | A real pedestrian closer than 1.9 m to a bike | Lower `DEC_PROX_MIN_M`, then check whether a pedestrian is being relabelled as a rider |
| A group of cyclists with pedestrians nearby never triggers | The overlap test relabels real pedestrians standing next to bikes | Reduce `RIDER_X_OVERLAP_FRAC` (for example 0.4) or `RIDER_Y_OVERLAP_FRAC` (for example 0.15) |

The README expects 2.5 to 3.0 m to be needed with a low pole-mounted camera or a narrower lens, because the error of the bottom-centre projection grows. `python decision_testbench.py --error-map` prints that error for your camera geometry, see [Fisheye model and ground-plane localisation](../system/perception.md).

## Cyclist memory

The memory is held in seconds: `DEC_BIKE_MEMORY_FRAMES / DEC_FPS`, which is 1.93 s with the defaults. Raise it if your detector loses cyclists for longer stretches than that. The cost is a longer `warning` phase after the last cyclist has gone, since stage 2 only returns `safe` once the memory is empty.

The lookback counts track samples and not seconds. At a lower frame rate the same `DEC_LOOKBACK_FRAMES` spans more time, and the displacement threshold `DEC_PROX_SPEED_MIN` corresponds to a lower speed. With the defaults at 30 FPS, 0.147 m over 2 frames is about 2.2 m/s. At 15 FPS the same numbers mean 1.1 m/s.

## Upper bound and stopping distance

With `DEC_SPEED_ADAPTIVE`, the upper distance bound for a cyclist is the larger of `DEC_PROX_MAX_M` and the cyclist's stopping distance

```text
d_stop = v · DEC_REACTION_S + v² / (2 · DEC_DECEL_MS2)
```

where `v` is the speed estimated from the track over up to 4 samples.

| Speed | Stopping distance with the defaults |
|---|---|
| 5 m/s (18 km/h) | 10.6 m |
| 8 m/s (28.8 km/h) | 23.0 m |
| 10 m/s (36 km/h) | 33.9 m |

The stopping distance passes 24.8 m at about 8.35 m/s (30 km/h). Below that speed the extension has no effect unless you lower `DEC_PROX_MAX_M`.

`DEC_MAX_RANGE_M` is a separate limit. A cyclist or pedestrian farther than 25 m from the camera, measured on the ground, is skipped whatever the pair distance is. Raise it only if the localisation error at that range is acceptable for your mount.

The defaults are braking values for conventional bicycles. The testbench also has an e-bike profile with a deceleration of 6.0 m/s², used by two conformance scenarios. See [Braking profiles and danger labelling](../system/safety-model.md).

## Latency compensation

`camera.latency_frames` in `config.yaml` tells the live bridge how many frames of measured latency to compensate with a first-order kinematic prediction. The file's comment gives the rule: at 30 FPS one frame is 33 ms, so use the measured latency in milliseconds divided by 33.

The kit contains the predictor as `LatencyCompensator`. It keeps a smoothed per-track velocity and moves each position forward by `delay_frames` frames. `main()` constructs it with `delay_frames=0`, which turns it off. Change that argument to enable it.

The kit's [latency log](hardware.md#latency-logging) covers the time from `cap.read` to the end of the frame. Delay inside the camera before the frame arrives is not in it and has to be measured separately. The testbench can show what a given delay costs: `python decision_testbench.py --latency-sweep` sweeps camera latency from 0 to 500 ms, see [Latency experiments](../experiments/latency.md).

## Track lifetime

`config.yaml` lists `decision.max_object_age: 10` and `DecisionParams` carries the field, but `DecisionPipeline` has no such argument and no code in this repository reads it.

In the kit, track lifetime is set by the arguments of `GroundPlaneTracker`, which `main()` constructs with defaults:

| Argument | Default | Unit | Meaning |
|---|---|---|---|
| `match_radius` | `3.0` | m | Largest distance between a detection and the track it can update. |
| `min_hits` | `2` | detections | Matched detections before a track is reported. |
| `max_lost` | `300` | frames | Frames a track survives without a detection. It keeps moving at its last velocity and is still reported. |
| `alpha` | `0.5` | | Smoothing factor of the velocity estimate. |

A track is also dropped when its speed exceeds the limit for its class: 6 m/s for persons, 17 m/s for bicycles and motorcycles, 28 m/s for cars and 20 m/s for buses and trucks. If stale tracks hold the state at `warning` or `alert` after people have left the scene, lower `max_lost`. 300 frames is 10 s at 30 FPS.

## Check a change offline

Run a candidate through the conformance scenarios before deploying it. This script builds the camera the same way `decision_testbench.py` does and evaluates one parameter set with fisheye localisation error enabled:

```python title="check_candidate.py"
import yaml
from decision_testbench import (DecisionParams, FisheyeCamera,
                                load_camera_calibration, run_all_scenarios)

calib = load_camera_calibration()
with open('config.yaml') as f:
    cam_cfg = yaml.safe_load(f)['camera']

cam = FisheyeCamera(
    height=cam_cfg['height'], pitch_deg=cam_cfg['pitch_deg'],
    fov_deg=calib['fov_deg'], radius_px=int(calib['radius_px']),
    frame_width=calib['frame_width'], frame_height=calib['frame_height'],
    pad_size=calib['pad_size'], cx_override=calib['cx'], cy_override=calib['cy'],
    k1=calib['k1'], k2=calib['k2'],
)

candidate = DecisionParams(prox_min=2.5)
run_all_scenarios(candidate, cam=cam, add_bbox_noise=True)
```

Save it in the repository root and run `python check_candidate.py`. It prints the per-scenario table and these totals:

```text
  SCENARIO RESULTS -- N=58, d=[2.5,24.8]m, v_min=0.147, k=2
  ...
  TOTAL                        1210  369  106 4419 91.9% 92.3%  8.0%

  Alert fatigue (mean fraction of time in alert): 31.4%
```

Compared with the shipped parameters from `python decision_testbench.py --compare`:

| `prox_min` | TP | FP | FN | TN | Sensitivity | Specificity | Time in alert |
|---|---|---|---|---|---|---|---|
| 1.9 m (default) | 1228 | 370 | 88 | 4418 | 93.3% | 92.3% | 32.3% |
| 2.5 m | 1210 | 369 | 106 | 4419 | 91.9% | 92.3% | 31.4% |

Raising the floor to 2.5 m costs 18 danger frames on this suite and leaves the false positives where they were. The scripted scenarios have no rider boxes, so the suite measures the cost of a higher floor. The benefit shows up only on your own footage.

Two further steps are worth the time. Add scenarios that reflect your site, such as its road width or typical cyclist speeds, as described in [Authoring scenarios](../scenarios/authoring.md). To search the parameter space instead of testing single values, run [the optimizer](../experiments/optimizer.md). [The testbench](../simulation/testbench.md) explains every column of the table.

## Related

- [The decision pipeline](../system/decision.md)
- [Bridge starter kit](index.md)
- [Hardware and I/O](hardware.md)
