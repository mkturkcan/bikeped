# Offline testbench

`decision_testbench.py` replays scripted scenarios through a model of the camera, the detector and the tracker, feeds the result to the decision pipeline, and scores every frame against a ground-truth danger label. This page describes that loop, the metrics it prints, the command-line flags, and the companion video renderer `sim_visualizer.py`.

```console
$ python decision_testbench.py
Loaded calibration: FOV=197.90 cx=1752.7 cy=1804.5 f=1013.3 R=1750 frame=3500x3500 pad=3840 pitch=0 k1=0 k2=0
Running all scenarios with Default pipeline...

  SCENARIO RESULTS -- N=58, d=[1.9,24.8]m, v_min=0.147, k=2
  Scenario                       TP   FP   FN   TN   Sens   Spec  SevFN    Lat WarnBgt RxnAdq Fatigue       Exp
  Safe Crossing                   0    0    0  196  0.0% 100.0%  0.0%    N/A     N/A    N/A   0.0%      safe
  Bike Approaches Ped            86   25   14  101 86.0% 80.2% 26.3%   0.47    3.5s   100%  49.1%     alert
  High-Speed Near Miss           97    0   17   82 85.1% 100.0% 18.0%   0.47    3.3s   100%  49.5%     alert
...
  TOTAL                        1209  361  107 4427 91.9% 92.5%  8.4%

  Alert fatigue (mean fraction of time in alert): 31.9%
```

The excerpt drops the rule lines and 21 scenario rows. Run from the repository root so that `config.yaml` and `camera_calibration.json` are found.

## What happens in one frame

`simulate_scenario` steps a scenario at 30 frames per second from `t = 0` to the last waypoint. At each step it does the following, in this order:

1. Interpolates every agent's scripted position at time `t`. An agent outside its path's time span is absent.
2. Drops detections at random, in Monte Carlo runs only. The recall comes from the projected bounding-box area of the agent on the fisheye sensor, so small, distant agents are missed more often. The drop pattern is independent per frame (`iid`) or bursty (`gilbert`).
3. Replaces the true position with the position the deployed system would compute, when localisation error is on. `compute_bbox_error` projects the agent's 3D box through the calibrated lens, takes the bottom centre of the resulting 2D box and projects it back to the ground.
4. Delays the detections by `camera_delay_frames` through a ring buffer, when a latency is being simulated.
5. Passes them through `GroundPlaneTracker`, which keeps a track alive for up to one second of missed detections using constant-velocity prediction, and optionally forecasts positions forward to compensate for the delay.
6. Converts positions to camera-relative coordinates and calls `DecisionPipeline.evaluate`, which returns `idle`, `safe`, `warning` or `alert`.
7. Computes the ground-truth danger assessment for time `t` from the scripted positions.

Steps 2 to 5 are described in [Perception](../system/perception.md) and [Detection and localisation uncertainty](../system/uncertainty.md), step 6 in [Decision pipeline](../system/decision.md), and step 7 in [Safety model](../system/safety-model.md).

## Metrics

Ground truth has two tiers. A frame is actionable when there is danger and a warning could still help. A danger frame that is no longer actionable counts neither for nor against the system.

| Column | Definition |
|---|---|
| `TP` | Frames in `alert` that are actionable danger frames |
| `FN` | Actionable danger frames without an alert |
| `FP` | Frames in `alert` with no danger of either tier |
| `TN` | Frames without an alert and without danger |
| `Sens` | `TP` divided by the number of actionable frames |
| `Spec` | `TN` divided by the number of safe frames |
| `SevFN` | Severity summed over missed actionable frames, divided by severity summed over all actionable frames |
| `Lat` | Seconds from the first actionable frame to the first alert at or after it |
| `WarnBgt` | Smallest warning budget over the alert onsets of the scenario. The budget is the time from an onset to the closest approach |
| `RxnAdq` | Share of alert onsets with a budget above 1.87 s, the reaction time assumed for a distracted pedestrian |
| `Fatigue` | Fraction of all frames spent in `alert` |
| `Exp` | The scenario's `expected_dominant` state, printed for reference |

The `TOTAL` row pools frames over scenarios for the four counts, sensitivity and specificity. Its `SevFN` is the mean of the per-scenario rates over scenarios that contain actionable frames. A scenario with no actionable frames, such as Safe Crossing, shows 0.0% sensitivity and does not enter the pooled figure.

`simulate_scenario` returns the same quantities as a dictionary (`sensitivity`, `specificity`, `sev_weighted_fn_rate`, `alert_latency_s`, `min_warning_budget_s`, `mean_warning_budget_s`, `reaction_adequacy`, `alert_fraction`, `peak_severity`, `peak_bike_speed_ms`) together with the per-frame arrays `times`, `states`, `dangers` and `assessments`.

## Modes

One mode runs per invocation. If several mode flags are given, the first in this table wins.

| Flag | What it runs | Localisation error |
|---|---|---|
| `--error-map` | Summary of bounding-box localisation error across the field of view | n/a |
| `--compare` | Every entry of `PIPELINE_CONFIGS` over all scenarios | on |
| `--monte-carlo` | Seeded trials with random detection drops, for every entry of `PIPELINE_CONFIGS` | on |
| `--latency-sweep` | The `default` configuration at camera delays from 0 to 495 ms | on |
| `--optimize` | Differential-evolution search, then tables for the result and for `default` | on |
| none | The `default` configuration over all scenarios | off |

The run without flags is the one case that leaves localisation error off, which is why its totals above match the `default` block of `--no-bbox-noise --compare` and not that of `--compare`.

## Options

| Option | Default | Applies to | Meaning |
|---|---|---|---|
| `--no-bbox-noise` | off | compare, Monte Carlo, sweep, optimize | Feed true positions to the pipeline |
| `--bbox-noise` | on | same | Accepted for symmetry. Localisation error is already on |
| `--mc-trials N` | 50 | Monte Carlo | Number of trials. Trial `i` uses seed `i` |
| `--failure-model {iid,gilbert}` | `iid` | Monte Carlo | Independent drops or bursts of consecutive drops |
| `--burst-length F` | 3.0 | Monte Carlo with `gilbert` | Mean burst length in frames |
| `--w-fn`, `--w-fp`, `--w-lat` | 5.0, 1.0, 2.0 | optimize | Cost weights, see [Optimizer](../experiments/optimizer.md) |
| `--eval-cache PATH` | none | all | Load measured recall and precision from an `eval_cache.json` written by `tools/eval_models.py` |
| `--eval-model NAME` | first entry | all | Which model in the cache to use |

```bash
python decision_testbench.py --compare
python decision_testbench.py --no-bbox-noise --compare
python decision_testbench.py --monte-carlo --mc-trials 50
python decision_testbench.py --monte-carlo --failure-model gilbert --burst-length 5
python decision_testbench.py --latency-sweep
python decision_testbench.py --error-map
```

```text
--8<-- "_generated/cli/decision_testbench.txt"
```

The totals each mode should print are listed under [Reproduce the paper](../experiments/index.md).

## Scenario videos with `sim_visualizer.py`

`sim_visualizer.py` renders the scenarios as the pole camera would see them: agents drawn as 3D boxes and projected through the fisheye model onto a ground grid. It runs the same pipeline as the testbench and writes one MP4 per scenario into the working directory.

```bash
python sim_visualizer.py --no-display                  # all scenarios, headless
python sim_visualizer.py --scenarios 1 5 7             # by index, with a preview window
python sim_visualizer.py --scenarios 7 --errors --no-display
```

| Option | Default | Meaning |
|---|---|---|
| `--scenarios I [I ...]` | all | Zero-based indices into `SCENARIOS` |
| `--pipeline NAME` | `optimized` | Key of `PIPELINE_CONFIGS` |
| `--fps N` | 30 | Simulation and video frame rate |
| `--two-cam` | off | Also render a side-by-side view with a second camera across the road |
| `--height M`, `--pitch DEG` | 3.6576 m, automatic | Mounting geometry, see [Camera placement](../experiments/placement.md) |
| `--errors` | off | Run the pipeline on error-perturbed positions, draw the error vectors, and save a JSON log of observations |
| `--no-save` | off | Do not write the MP4 |
| `--no-display` | off | Do not open the OpenCV preview window. Required on a machine without a display |

Each frame is 1080 by 1080 pixels and shows:

- the ground grid at 2 m spacing and the crosswalk, as the lens distorts them;
- the part of the image circle that falls outside the 3840 by 2160 sensor, dimmed;
- each agent with its trail, and a line between every cyclist and pedestrian closer than `prox_max`;
- the scenario name, the time, and a badge with the current decision state;
- during danger, the severity, cyclist speed, time to collision and stopping distance from the ground-truth model;
- a 300 by 300 pixel top-down inset in the lower right corner.

Files are named `sim_<index>_<scenario name>.mp4`, for example `sim_01_bike_approaches_ped.mp4`. With `--errors` the name gains `_errors` and a sidecar `..._observations.json` holds the per-frame true and observed positions.

## Related

- [Simulators compared](index.md)
- [Scenario format](../reference/formats.md)
- [API: decision_testbench](../api/decision_testbench.md)
