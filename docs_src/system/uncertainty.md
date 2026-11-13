# Detection and localisation uncertainty

The testbench starts from exact scripted positions and degrades them before the
decision pipeline sees them: a bounding-box localisation error, missed
detections, and camera latency. This page documents the first two and the
tracker that sits between them and the pipeline. Latency has its own page under
[experiments](../experiments/latency.md).

## Order of operations in one simulated frame

`simulate_scenario` does the following for every agent at every frame:

1. If a recall value and a random generator were passed (the Monte Carlo
   mode), decide whether the detection is missed. The recall is looked up from
   the agent's projected bounding-box area.
2. If the error model is on, replace the true position with the position that
   the bounding-box bottom-centre back-projects to. See
   [perception](perception.md#bounding-box-localisation-error).
3. Push the frame's detections through the latency buffer.
4. Update the ground-plane tracker, which fills in missed detections by
   prediction.
5. Convert to camera-relative coordinates and call the decision pipeline.

`--compare` and the default run use steps 2, 4 and 5 only, so their output is
the same on every run. Missed detections enter with `--monte-carlo`.

## Detector performance parameters

`DetectionParams` is a pair of average precision and average recall.
`SizedDetectionParams.for_area(bbox_area_px)` returns the pair for a box of a
given pixel area, measured at detector input scale (the area in the crop times
`(1280 / 3840)^2`):

- with an `ar_curve` loaded from an evaluation cache, recall is interpolated
  linearly over the curve's bin centres and precision is scaled by the same
  ratio;
- with per-size values only, it uses the COCO bins: small below 32² = 1024 px²,
  medium below 96² = 9216 px², large above;
- with neither, it returns the overall values.

`UncertaintyParams` bundles the per-class parameters with the constants of the
position model. The defaults are commented in the code as coming from the
`fisheye_1280` model evaluated on fisheye-augmented COCO.

| Field | Default | Unit | Meaning |
|---|---|---|---|
| `pedestrian` | AP 0.678, AR 0.755 | | Used for `person` agents. |
| `cyclist` | AP 0.657, AR 0.747 | | Used for `bike` agents. |
| `vehicle` | AP 0.706, AR 0.765 | | Used for every other class. |
| `bev_angular_error_deg` | 5.0 | degrees | Angular error assumed for the ground projection. |
| `kf_std_weight_pos` | 0.05 | | Position noise weight of the tracker's filter. |
| `kf_std_weight_vel` | 0.00625 | | Velocity noise weight. |
| `kf_max_lost` | 30 | frames | Frames a track may stay lost. |
| `kf_iou_threshold` | 0.2 | | Association threshold. |
| `fps` | 30 | frames/s | Frame rate. |
| `latency_ms` | 19.0 | ms | Nominal processing latency. |

### Loading measured values

`tools/eval_models.py` evaluates one or more YOLO models and caches per-class
results in `eval_cache.json`. Pass the cache to the testbench to replace the
defaults:

```bash
python tools/eval_models.py --models yolo11x.pt --data cocoaug.yaml
python decision_testbench.py --monte-carlo --eval-cache eval_cache.json --eval-model yolo11x
```

`load_uncertainty_from_cache` reads the `merged` groups `pedestrian`,
`cyclist` and `vehicle` of the chosen model, with keys `ap5095`, `recall`,
`per_size` and `ar_curve`. Without `--eval-model` it takes the first model in
the file. If the file does not exist it returns `None` and the defaults stay.

## Detection failure models

A failure model decides, per agent and per frame, whether a detection is
missed, given the recall for that agent. `FAILURE_MODELS` maps the
`--failure-model` choices to classes.

| `--failure-model` | Class | Behaviour |
|---|---|---|
| `iid` (default) | `IIDFailureModel` | Each frame is missed independently with probability `1 - AR`. |
| `gilbert` | `GilbertFailureModel` | A two-state Markov chain per agent produces runs of consecutive misses. |

The Gilbert model has a detected state and a missed state. With mean burst
length `B` (`--burst-length`, default 3.0 frames):

```text
r = 1 / B                  # P(missed -> detected), recovery
p = (1 - AR) * r / AR      # P(detected -> missed), miss onset
```

The stationary miss rate `p / (p + r)` equals `1 - AR`, so both models miss the
same share of frames on average and differ in how the misses cluster. A burst
length of 1.0 is close to the independent model. The chain state is kept per
agent id and cleared between scenarios.

```bash
python decision_testbench.py --monte-carlo --mc-trials 50
python decision_testbench.py --monte-carlo --failure-model gilbert --burst-length 3
```

Bursts matter because the tracker and the cyclist memory both bridge short
gaps. Independent misses at the same average rate are mostly bridged; a long
run of misses on the cyclist is what delays an alert.

## Position uncertainty

`compute_position_uncertainty(cam, gx, gy, obj, unc)` estimates the standard
deviation of a ground position at a given location from three terms and
combines them as a root sum of squares:

| Key | Term |
|---|---|
| `sigma_yolo` | Box localisation: box size times `(1 - AP) / (1 + AP)`, converted to metres with `distance / f`. |
| `sigma_bev` | Ground projection: `distance * tan(bev_angular_error_deg)`. |
| `sigma_kf` | Tracker filter: `kf_std_weight_pos` times the box height, converted to metres. |
| `sigma_total` | Root sum of squares of the three. |
| `vel_std_world` | Velocity standard deviation in m/s. |

The box size comes from `projected_bbox_area`, which projects the eight corners
of the object's 3D box through the fisheye model and returns 0 when no corner
is on the sensor.

## The ground-plane tracker

`GroundPlaneTracker(track_buffer=30, match_radius=3.0)` mirrors the tracker of
the deployed bridge. Per frame it:

1. moves every existing track forward by its velocity;
2. matches detections to tracks of the same type by greedy nearest neighbour
   within `match_radius` metres;
3. starts a new track for each unmatched detection;
4. deletes tracks that have gone more than `track_buffer` frames without a
   detection;
5. returns all live tracks, including the ones it only predicted.

Velocity and acceleration are exponential moving averages with weight 0.5.
`simulate_scenario` builds the tracker with `track_buffer=fps`, one second of
persistence. `forecast(agents, n_frames, order)` extrapolates positions with
constant velocity (`order=1`) or constant acceleration (`order=2`) and is used
for latency compensation.

## Effect of the localisation error

`--no-bbox-noise` feeds the pipeline the exact scripted positions. Totals over
the 24 conformance scenarios with and without the error model:

```bash
python decision_testbench.py --compare
python decision_testbench.py --no-bbox-noise --compare
```

| Configuration | Sensitivity, with error | Specificity, with error | Sensitivity, exact positions | Specificity, exact positions |
|---|---|---|---|---|
| `default` | 93.3% | 92.3% | 91.9% | 92.5% |
| `conservative` | 77.1% | 92.6% | 75.9% | 91.9% |
| `proximity_only` | 56.1% | 97.4% | 54.1% | 97.0% |
| `speed_adaptive` | 93.1% | 91.8% | 92.2% | 91.8% |
| `baseline_distance_only` | 57.8% | 70.2% | 54.7% | 71.0% |
| `baseline_naive_closing` | 93.4% | 90.4% | 91.9% | 90.8% |
| `baseline_ttc` | 84.3% | 96.4% | 84.0% | 96.5% |

For the default configuration the frame counts move from 1209 true positives
and 361 false positives with exact positions to 1228 and 370 with the error
model. The shift is small against the gaps between configurations, and the
ranking of the decision rules is the same in both columns.

## Related

- [Perception geometry](perception.md) for the error model itself
- [Safety model](safety-model.md) for the definitions of sensitivity and specificity used here
- [The testbench](../simulation/testbench.md) for the full command line
- [`decision_testbench` API](../api/decision_testbench.md)
