# Conformance suite

The conformance suite is the set of 24 scripted scenarios behind every result in the paper. This page lists them, explains how a run is scored, and shows what the default pipeline does on each one.

The scenarios live in `decision_testbench.SCENARIOS` and are not edited. `bikeped.scenarios` exposes the same list, in the same order, as the suite `conformance`.

## The 24 scenarios

The table is read from `decision_testbench.SCENARIOS`. Duration is the last waypoint time plus 0.5 s (`get_scenario_duration`), rounded to 0.1 s.

| # | Scenario | Expected state | Pedestrians | Cyclists | Duration (s) | Braking profile |
|---:|---|---|---:|---:|---:|---|
| 0 | Safe Crossing | <span class="state state-safe">safe</span> | 1 | 0 | 6.5 | bicycle |
| 1 | Bike Approaches Ped | <span class="state state-alert">alert</span> | 1 | 1 | 7.5 | bicycle |
| 2 | High-Speed Near Miss | <span class="state state-alert">alert</span> | 1 | 1 | 6.5 | bicycle |
| 3 | Multiple VRUs | <span class="state state-alert">alert</span> | 3 | 2 | 8.5 | bicycle |
| 4 | Occluded Emergence | <span class="state state-alert">alert</span> | 1 | 1 | 6.5 | bicycle |
| 5 | Parallel Paths | <span class="state state-warning">warning</span> | 1 | 1 | 8.5 | bicycle |
| 6 | Bikes Only | <span class="state state-idle">idle</span> | 0 | 2 | 5.5 | bicycle |
| 7 | Fast Approach | <span class="state state-alert">alert</span> | 1 | 1 | 6.5 | ebike |
| 8 | Speed Variant | <span class="state state-alert">alert</span> | 1 | 2 | 10.5 | bicycle |
| 9 | Counter-Flow on Crosswalk | <span class="state state-alert">alert</span> | 1 | 1 | 6.5 | bicycle |
| 10 | Multi-Speed Converging | <span class="state state-alert">alert</span> | 2 | 3 | 8.5 | bicycle |
| 11 | Wheelchair User | <span class="state state-alert">alert</span> | 1 | 1 | 27.2 | bicycle |
| 12 | Stopped Pedestrian | <span class="state state-alert">alert</span> | 1 | 1 | 13.5 | bicycle |
| 13 | Cross-Street Cyclist | <span class="state state-alert">alert</span> | 1 | 1 | 8.5 | bicycle |
| 14 | E-Scooter | <span class="state state-alert">alert</span> | 1 | 1 | 7.5 | ebike |
| 15 | Child Pedestrian | <span class="state state-alert">alert</span> | 1 | 1 | 13.8 | bicycle |
| 16 | Overtake from Behind | <span class="state state-alert">alert</span> | 1 | 1 | 12.5 | bicycle |
| 17 | Cyclist Abort | <span class="state state-safe">safe</span> | 1 | 1 | 8.0 | bicycle |
| 18 | Dense Pedestrian Group | <span class="state state-alert">alert</span> | 8 | 1 | 10.4 | bicycle |
| 19 | Diagonal Jaywalker | <span class="state state-alert">alert</span> | 1 | 1 | 9.5 | bicycle |
| 20 | Swerving Cyclist | <span class="state state-alert">alert</span> | 1 | 1 | 8.0 | bicycle |
| 21 | Late Turn Toward Ped | <span class="state state-alert">alert</span> | 1 | 1 | 7.5 | bicycle |
| 22 | U-Turn Cyclist | <span class="state state-alert">alert</span> | 1 | 1 | 8.5 | bicycle |
| 23 | Accelerating E-Bike | <span class="state state-alert">alert</span> | 1 | 1 | 7.5 | ebike |

Twenty scenarios expect `alert`, two expect `safe`, one expects `warning` and one expects `idle`.

Twenty-one of the 24 contain at least one frame that the [ground-truth danger model](../system/safety-model.md) labels as dangerous. The three that contain none are Safe Crossing, Parallel Paths and Bikes Only. In two scenarios, Occluded Emergence and Fast Approach, every danger frame is too late for a warning to help the pedestrian, so they have danger frames and no actionable ones (see [Scoring](#scoring)).

### Families

`bikeped.scenarios` tags each conformance scenario with a family.

| Family | Count | Scenarios (index) |
|---|---:|---|
| `safe-crossing` | 3 | Safe Crossing (0), Parallel Paths (5), Bikes Only (6) |
| `standard-approach` | 5 | Bike Approaches Ped (1), Speed Variant (8), Counter-Flow on Crosswalk (9), Cross-Street Cyclist (13), Overtake from Behind (16) |
| `high-speed` | 4 | High-Speed Near Miss (2), Fast Approach (7), E-Scooter (14), Accelerating E-Bike (23) |
| `multi-agent` | 3 | Multiple VRUs (3), Multi-Speed Converging (10), Dense Pedestrian Group (18) |
| `edge-case` | 3 | Occluded Emergence (4), Stopped Pedestrian (12), Diagonal Jaywalker (19) |
| `accessibility` | 2 | Wheelchair User (11), Child Pedestrian (15) |
| `non-linear` | 4 | Cyclist Abort (17), Swerving Cyclist (20), Late Turn Toward Ped (21), U-Turn Cyclist (22) |

## Braking profiles

Three scenarios set `'vehicle': 'ebike'`: Fast Approach, E-Scooter and Accelerating E-Bike. The other 21 use the default `bicycle` profile.

| Profile | Reaction time (s) | Deceleration (m/s²) |
|---|---:|---:|
| `bicycle` | 0.84 | 1.96 |
| `ebike` | 0.84 | 6.0 |

The profile is applied twice. The danger model uses it to decide whether the cyclist can still stop, and `make_pipeline` passes the same values to the pipeline's speed-adaptive stopping distance. The sources of the values are listed in [Safety model](../system/safety-model.md).

## Scoring

`simulate_scenario` steps a scenario at 30 FPS from `t = 0` to its duration. Each frame gets a pipeline state and a ground-truth label with two tiers. A frame is a danger frame when the cyclist is on a collision course and cannot stop in time. It is also actionable when the time to collision is still longer than the pedestrian reaction time, so a warning can change the outcome.

| Column | Definition |
|---|---|
| `TP` | Frames in `alert` that are actionable. |
| `FN` | Actionable frames without `alert`. |
| `FP` | Frames in `alert` with no danger of either tier. |
| `TN` | Frames without `alert` and without danger. |
| `Sens` | `TP` divided by the number of actionable frames. |
| `Spec` | `TN` divided by the number of frames without danger. |
| `SevFN` | Severity-weighted miss rate: severity summed over missed actionable frames, divided by severity summed over all actionable frames. |
| `Lat` | Seconds from the first actionable frame to the first `alert` at or after it. |
| `WarnBgt` | Smallest time to closest approach, measured at each transition into `alert`. |
| `RxnAdq` | Share of those transitions that leave more than 1.87 s, the reaction time assumed for a distracted pedestrian. |
| `Fatigue` | Fraction of all frames spent in `alert`. |
| `Exp` | The scenario's `expected_dominant`. The testbench prints it and does not score against it. |

Danger frames that are not actionable count on neither side: an alert during one is not a false positive, and silence during one is not a false negative.

The `TOTAL` row pools the frame counts over all scenarios before dividing. Its `SevFN` is the mean over the scenarios that have actionable frames.

## Results with the default pipeline

```console
$ python decision_testbench.py
Loaded calibration: FOV=197.90 cx=1752.7 cy=1804.5 f=1013.3 R=1750 frame=3500x3500 pad=3840 pitch=0 k1=0 k2=0
Running all scenarios with Default pipeline...

  SCENARIO RESULTS -- N=58, d=[1.9,24.8]m, v_min=0.147, k=2
  Scenario                       TP   FP   FN   TN   Sens   Spec  SevFN    Lat WarnBgt RxnAdq Fatigue       Exp
  --------------------------------------------------------------------------------------------------------------------
  Safe Crossing                   0    0    0  196  0.0% 100.0%  0.0%    N/A     N/A    N/A   0.0%      safe
  Bike Approaches Ped            86   25   14  101 86.0% 80.2% 26.3%   0.47    3.5s   100%  49.1%     alert
  High-Speed Near Miss           97    0   17   82 85.1% 100.0% 18.0%   0.47    3.3s   100%  49.5%     alert
  Multiple VRUs                 113   29    0  114 100.0% 79.7%  0.0%   0.00    3.9s   100%  55.5%     alert
  Occluded Emergence              0    0    0  164  0.0% 100.0%  0.0%    N/A    1.0s     0%  10.7%     alert
  Parallel Paths                  0    0    0  256  0.0% 100.0%  0.0%    N/A     N/A    N/A   0.0%   warning
  Bikes Only                      0    0    0  166  0.0% 100.0%  0.0%    N/A     N/A    N/A   0.0%      idle
  Fast Approach                   0    0    0  147  0.0% 100.0%  0.0%    N/A    1.6s     0%  23.0%     alert
  Speed Variant                  54   83    0  179 100.0% 68.3%  0.0%   0.00    4.3s   100%  43.4%     alert
  Counter-Flow on Crosswalk      19    0    2  114 90.5% 100.0%  9.5%   0.07    2.6s   100%  35.7%     alert
  Multi-Speed Converging         72   81    0  103 100.0% 56.0%  0.0%   0.00    4.2s   100%  59.8%     alert
  Wheelchair User                83    0   15  651 84.7% 100.0% 26.9%   0.50    4.9s   100%  14.6%     alert
  Stopped Pedestrian             33    0    1  307 97.1% 100.0%  2.9%   0.03    3.2s   100%  20.2%     alert
  Cross-Street Cyclist           51    0    1  143 98.1% 100.0%  1.9%   0.03    3.7s   100%  39.5%     alert
  E-Scooter                      15    0    1  154 93.8% 100.0%  6.2%   0.03    2.3s   100%  28.8%     alert
  Child Pedestrian               99   22   31  263 76.2% 92.3% 26.2%   0.47    4.5s   100%  29.2%     alert
  Overtake from Behind           21    0    2  287 91.3% 100.0%  8.7%   0.07    2.9s   100%  17.6%     alert
  Cyclist Abort                  69    6    0  166 100.0% 96.5%  0.0%   0.00    2.4s   100%  31.1%      safe
  Dense Pedestrian Group         83   27    0  203 100.0% 88.3%  0.0%   0.00    3.5s   100%  35.1%     alert
  Diagonal Jaywalker             82   30   12  162 87.2% 84.4% 12.2%   0.13    4.0s   100%  39.2%     alert
  Swerving Cyclist               83   26    2  130 97.6% 83.3%  4.1%   0.07    3.5s   100%  45.2%     alert
  Late Turn Toward Ped           88   32    0  106 100.0% 76.8%  0.0%   0.00    3.8s   100%  53.1%     alert
  U-Turn Cyclist                 21    0    3  110 87.5% 100.0%  7.6%   0.03    2.2s   100%  45.7%     alert
  Accelerating E-Bike            40    0    6  123 87.0% 100.0%  8.9%   0.20    3.3s   100%  40.7%     alert
  --------------------------------------------------------------------------------------------------------------------
  TOTAL                        1209  361  107 4427 91.9% 92.5%  8.4%

  Alert fatigue (mean fraction of time in alert): 31.9%
```

Pooled over the suite, the default parameters alert on 91.9% of actionable frames and stay silent on 92.5% of frames without danger.

This invocation feeds the pipeline the scripted positions as they are. The paper's comparison table adds the fisheye localisation error, which `python decision_testbench.py --compare` applies unless `--no-bbox-noise` is given. Both runs are described in [Testbench](../simulation/testbench.md) and [Reproducing the paper](../experiments/index.md).

A few rows need the scoring rules to read correctly. Occluded Emergence and Fast Approach show no true positives and no misses because they have no actionable frames. The pipeline does alert in both (10.7% and 23.0% of frames), 1.0 s and 1.6 s before closest approach, which is less than the 1.87 s a distracted pedestrian needs. Cyclist Abort expects `safe` as its dominant state and still contains 69 actionable frames before the cyclist turns away, all of which are caught.

## Dominant state

`python -m bikeped.scenarios evaluate --suite conformance` scores the dominant state instead of frames: the most severe state (alert > warning > safe > idle) that the pipeline holds for at least 0.3 s in total over the run. Under that rule the default pipeline matches 23 of the 24 paper labels. The one miss is Cyclist Abort, where the pipeline alerts and the paper's label is safe.

```console
$ python -m bikeped.scenarios evaluate --suite conformance
scenario                             dur  ag danger  alert  false lead s (per interval)  dominant expected  ok
--------------------------------------------------------------------------------------------------------------
safe-crossing                        6.5   1   0.00   0.00   0.00 -                          safe     safe  yes
bike-approaches-ped                  7.5   2   3.33   3.67   0.83 -0.5,+1.6                 alert    alert  yes
...
cyclist-abort                        8.0   2   2.30   2.50   0.20 +0.1                      alert     safe  NO
...
--------------------------------------------------------------------------------------------------------------
24 scenarios, dominant state matches expected in 23; 18 of 27 danger intervals alerted no later than one frame after onset; false-alert time 12.27 s
```

## Catalog

--8<-- "_generated/scenarios/conformance.md"

The same data is available from `python -m bikeped.scenarios list --suite conformance` and, for one scenario, `python -m bikeped.scenarios show ID`.
