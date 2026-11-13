# Quickstart

A tour of the repository in five commands: evaluate the pipeline on the conformance scenarios, compare decision rules, render one scenario, open the browser simulator and call the pipeline from Python. It assumes the [core install](install.md) and a shell in the repository root.

## Run the conformance scenarios

```console
$ python decision_testbench.py
Loaded calibration: FOV=197.90 cx=1752.7 cy=1804.5 f=1013.3 R=1750 frame=3500x3500 pad=3840 pitch=0 k1=0 k2=0
Running all scenarios with Default pipeline...

========================================================================================================================
  SCENARIO RESULTS -- N=58, d=[1.9,24.8]m, v_min=0.147, k=2
========================================================================================================================
  Scenario                       TP   FP   FN   TN   Sens   Spec  SevFN    Lat WarnBgt RxnAdq Fatigue       Exp
  --------------------------------------------------------------------------------------------------------------------
  Safe Crossing                   0    0    0  196  0.0% 100.0%  0.0%    N/A     N/A    N/A   0.0%      safe
  Bike Approaches Ped            86   25   14  101 86.0% 80.2% 26.3%   0.47    3.5s   100%  49.1%     alert
  High-Speed Near Miss           97    0   17   82 85.1% 100.0% 18.0%   0.47    3.3s   100%  49.5%     alert
  Multiple VRUs                 113   29    0  114 100.0% 79.7%  0.0%   0.00    3.9s   100%  55.5%     alert
  ...
  Accelerating E-Bike            40    0    6  123 87.0% 100.0%  8.9%   0.20    3.3s   100%  40.7%     alert
  --------------------------------------------------------------------------------------------------------------------
  TOTAL                        1209  361  107 4427 91.9% 92.5%  8.4%

  Alert fatigue (mean fraction of time in alert): 31.9%
```

The run takes a few seconds. The header line repeats the parameters in use: a cyclist memory of `N=58` frames, a distance window of 1.9 to 24.8 m, a minimum cyclist displacement of 0.147 m and a lookback of `k=2` frames. These are the values in the `decision` block of `config.yaml`.

Each row is one scenario replayed at 30 FPS. `TP`, `FP`, `FN` and `TN` count frames: a true positive is a frame where the pipeline says alert and the [ground-truth safety model](../system/safety-model.md) labels the frame dangerous. `Sens` and `Spec` are the sensitivity and specificity from those counts, `SevFN` is the false-negative rate weighted by severity, `Fatigue` is the fraction of the scenario spent in alert, and `Exp` is the state the scenario author expects to dominate. The [testbench page](../simulation/testbench.md) defines the remaining columns.

## Compare decision rules

`--compare` runs the same 24 scenarios under each named configuration and adds the fisheye localisation error that the plain run leaves out.

```bash
python decision_testbench.py --compare
```

It prints one table like the one above per configuration. The totals of that run:

| Configuration | Sensitivity | Specificity | SevFN | Time in alert |
|---|---|---|---|---|
| `default` | 93.3% | 92.3% | 7.6% | 32.3% |
| `optimized` | 93.3% | 92.3% | 7.6% | 32.3% |
| `speed_adaptive` | 93.1% | 91.8% | 9.4% | 33.7% |
| `conservative` | 77.1% | 92.6% | 31.0% | 28.9% |
| `proximity_only` | 56.1% | 97.4% | 61.5% | 19.6% |
| `baseline_naive_closing` | 93.4% | 90.4% | 7.5% | 33.8% |
| `baseline_ttc` | 84.3% | 96.4% | 14.7% | 27.0% |
| `baseline_distance_only` | 57.8% | 70.2% | 58.5% | 41.1% |

The `default` row is the headline result of the paper: 93.3% sensitivity and 92.3% specificity with localisation error. The plain run above reports 91.9% and 92.5% because it evaluates exact positions. `python decision_testbench.py --no-bbox-noise --compare` reproduces that ablation for every configuration. The three `baseline_*` rows swap the stage 3 rule for a simpler one, described under [decision rules](../system/decision.md).

## Render a scenario

`sim_visualizer.py` draws what the fisheye camera would see: agents projected through the lens model onto a ground-plane wireframe, with the decision state overlaid. Scenario indices are zero-based, so 7 is Fast Approach.

```console
$ python sim_visualizer.py --no-display --scenarios 7
Rendering 1 scenarios, single-cam

  Rendering: Fast Approach (1-cam, 6.5s)
    Saved: sim_07_fast_approach.mp4 (196 frames)

Done.
```

The MP4 is written to the current directory. Without `--no-display` the frames also appear in a window while they render.

## Open the browser simulator

The [browser simulator](../simulation/browser.md) runs the same decision logic in a web page. Open `simulator/index.html` from a checkout, pick a scenario and a pipeline variant, and watch the state change as the agents move. Nothing needs to be installed.

## Call the pipeline from Python

`DecisionPipeline` is a plain class with no dependencies beyond the standard library. Feed it one position per tracked object per frame, in ground-plane metres relative to the camera, then ask for the state.

```python title="example.py"
from decision_pipeline import DecisionPipeline

pipeline = DecisionPipeline(bike_memory_frames=58, prox_min=1.9, prox_max=24.8,
                            prox_speed_min=0.147, lookback_frames=2)

fps = 30
for frame in range(91):
    t = frame / fps
    ped = {'track_id': 1, 'x': 5.0, 'y': -3.0 + 1.4 * t}      # walking across the road
    bike = {'track_id': 2, 'x': 23.0 - 6.0 * t, 'y': 0.5}      # riding toward the crosswalk
    pipeline.update_track(1, ped['x'], ped['y'], t)
    pipeline.update_track(2, bike['x'], bike['y'], t)
    state, pair = pipeline.evaluate([ped], [bike], bike_in_scene=True, sim_time=t)
    if frame % 15 == 0:
        print(f"t={t:4.1f}s  bike x={bike['x']:5.1f} m  state={state:7s}  pair={pair}")
```

```console
$ python example.py
t= 0.0s  bike x= 23.0 m  state=warning  pair=None
t= 0.5s  bike x= 20.0 m  state=alert    pair=(2, 1)
t= 1.0s  bike x= 17.0 m  state=alert    pair=(2, 1)
t= 1.5s  bike x= 14.0 m  state=alert    pair=(2, 1)
t= 2.0s  bike x= 11.0 m  state=alert    pair=(2, 1)
t= 2.5s  bike x=  8.0 m  state=alert    pair=(2, 1)
t= 3.0s  bike x=  5.0 m  state=warning  pair=None
```

The first frame is a warning because the closing check needs two positions per track. From the second frame on, the pair distance shrinks and the cyclist moves more than `prox_speed_min` per lookback, so the state is alert and `pair` names the cyclist and pedestrian track ids. In the last line the two are 0.7 m apart, closer than `prox_min`, and that pair is skipped. The floor exists so that a rider detected as a separate person does not pair with their own bike.

The constructor defaults are older values than the selected parameters, so pass the `decision` block of `config.yaml` explicitly as above. With an empty pedestrian list `evaluate` returns `('idle', None)`, and with a pedestrian and no cyclist in memory it returns `('safe', None)`.

## Related

- [How the system works](../system/index.md)
- [Decision pipeline](../system/decision.md)
- [Reproduce the paper](../experiments/index.md)
- [Command line reference](../reference/cli.md)
