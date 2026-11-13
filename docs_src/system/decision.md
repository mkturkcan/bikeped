# Decision pipeline

The decision logic is one class, `DecisionPipeline` in
[`decision_pipeline.py`](https://github.com/mkturkcan/bikeped/blob/main/decision_pipeline.py),
imported by both the live bridge and the offline testbench. This page describes
the three stages as they are coded, every parameter with its unit, and the
alternative Stage 3 rules that the paper uses as baselines.

<figure class="diagram">
--8<-- "assets/figures/decision-flow.svg"
<figcaption>Each frame passes through the stages in order and leaves at the first one that fails.</figcaption>
</figure>

## Inputs and outputs

The pipeline works on ground-plane positions in metres, relative to the camera.
Per frame the caller records every tracked object with
`update_track(track_id, x, y, t)` and then calls
`evaluate(persons, bikes, bike_in_scene, sim_time)`, where `persons` and `bikes`
are lists of `{'track_id', 'x', 'y'}` dicts. The return value is
`(state, alert_pair)`: the state is one of `idle`, `safe`, `warning`, `alert`,
and `alert_pair` is `(bike_track_id, ped_track_id)` for an alert and `None`
otherwise.

The object keeps two pieces of state between frames: the cyclist memory buffer
and a history of the last 60 positions per track. Call `reset()` between
scenarios when you run it offline.

```python
from decision_pipeline import DecisionPipeline

pipe = DecisionPipeline(bike_memory_frames=58, prox_min=1.9, prox_max=24.8,
                        prox_speed_min=0.147, lookback_frames=2)

ped = {'track_id': 1, 'x': 0.0, 'y': 8.0}
print(pipe.evaluate([], [], bike_in_scene=False, sim_time=0.0))
print(pipe.evaluate([ped], [], bike_in_scene=False, sim_time=0.0))

# A cyclist rides toward the standing pedestrian at 6 m/s.
for frame in range(4):
    t = frame / 30
    bike = {'track_id': 2, 'x': 12.0 - 6.0 * t, 'y': 8.0}
    pipe.update_track(1, ped['x'], ped['y'], t)
    pipe.update_track(2, bike['x'], bike['y'], t)
    state, pair = pipe.evaluate([ped], [bike], bike_in_scene=True, sim_time=t)
    print(f't={t:.3f} s  {state:<7}  pair={pair}')
```

```text
('idle', None)
('safe', None)
t=0.000 s  warning  pair=None
t=0.033 s  alert    pair=(2, 1)
t=0.067 s  alert    pair=(2, 1)
t=0.100 s  alert    pair=(2, 1)
```

The first frame with both agents returns `warning` because Stage 3 needs two
history points per track before it can tell whether a pair is closing.

## The three stages

### Stage 1: pedestrian presence

If `persons` is empty the result is <span class="state state-idle">idle</span>.
Cyclists alone never raise the state.

### Stage 2: cyclist memory

Every frame with `bike_in_scene=True` appends `sim_time` to a buffer, and
entries older than `bike_memory_frames / fps` seconds are dropped. If the buffer
is empty the result is <span class="state state-safe">safe</span>. The memory
keeps the system out of `safe` for about two seconds (58 frames at 30 FPS)
after a cyclist was last detected, which covers missed detections and short
occlusions.

Setting `bike_memory_frames` to 0 turns the memory off. Stage 2 then looks only
at the `bikes` list of the current frame.

### Stage 3: pairwise closing check

For every bike and every pedestrian that are both within `max_range` of the
camera:

1. The current distance `d_cur` between them must lie in `[prox_min, d_max]`.
2. Both tracks need at least two history points.
3. The lookback is `min(lookback_frames, len(bike_hist) - 1, len(ped_hist) - 1)`
   frames. `d_prev` is the distance between the two agents at that earlier
   frame, using the earlier position of both.
4. The pair raises <span class="state state-alert">alert</span> when
   `d_cur < d_prev` and the bike has moved more than `prox_speed_min` metres
   over the lookback.

The first pair that passes ends the evaluation. If no pair passes, the result
is <span class="state state-warning">warning</span>: a pedestrian and a recent
cyclist are both present and no pair passes the closing check.

`prox_speed_min` is a displacement over the lookback window. With the deployed
values, 0.147 m over 2 frames at 30 FPS is about 2.2 m/s, so a cyclist who is
stopped or walking the bike does not trigger an alert.

### Speed-adaptive upper bound

With `speed_adaptive=True` the upper distance bound grows with cyclist speed:

```text
d_max  = max(prox_max, d_stop)
d_stop = v * cyclist_reaction_s + v^2 / (2 * cyclist_decel_ms2)
```

`v` is estimated from the track history over the last four intervals. With the
default braking values (0.84 s, 1.96 m/s²) `stopping_distance` returns 14.22 m
at 6 m/s, 23.05 m at 8 m/s and 33.91 m at 10 m/s, so against the deployed
`prox_max` of 24.8 m the bound only expands for cyclists faster than 8 m/s.
`max_range` still applies: agents farther than 25 m from the camera are
skipped whatever `d_max` is. The braking values come from the
[safety model](safety-model.md).

## Parameters

| Parameter | Constructor default | Deployed (`config.yaml`) | Unit | Meaning |
|---|---|---|---|---|
| `bike_memory_frames` | 77 | 58 | frames | Length of the cyclist memory. 0 disables it. |
| `prox_min` | 0.5 | 1.9 | m | Lower bound on bike to pedestrian distance. Rejects a rider's own body detected as a person. |
| `prox_max` | 20.7 | 24.8 | m | Upper bound on bike to pedestrian distance before speed adaptation. |
| `prox_speed_min` | 0.098 | 0.147 | m per lookback | Minimum bike displacement over the lookback window. |
| `lookback_frames` | 1 | 2 | frames | How far back the closing comparison looks. |
| `max_range` | 25.0 | 25 | m | Agents farther than this from the camera are ignored. |
| `speed_adaptive` | `True` | `true` | | Expand `d_max` to the stopping distance. |
| `cyclist_reaction_s` | 0.84 | 0.84 | s | Cyclist perception-reaction time. |
| `cyclist_decel_ms2` | 1.96 | 1.96 | m/s² | Cyclist braking deceleration. |
| `fps` | 30 | | frames/s | Converts `bike_memory_frames` to seconds. |
| `decision_rule` | `'pairwise'` | | | Stage 3 rule, see below. |
| `ttc_threshold_s` | 3.0 | | s | Threshold for the `ttc` rule only. |

The deployed column is the `decision:` block of `config.yaml`, and the same
values are the `default` entry of `PIPELINE_CONFIGS` in the testbench and the
`DEC_*` constants of the bridge starter kit. `config.yaml` and the testbench's
`DecisionParams` also carry `max_object_age` (10 frames), which `make_pipeline`
does not pass to `DecisionPipeline`.

## Stage 3 rules

`decision_rule` swaps the test applied to a pair that is inside the distance
band. Stages 1 and 2 are the same for all four.

| Rule | Alerts when |
|---|---|
| `pairwise` | `d_cur < d_prev`, with `d_prev` computed from the earlier positions of both agents, and the bike displacement exceeds `prox_speed_min`. |
| `distance_only` | The pair is inside `[prox_min, d_max]`. The rule does not use track history. |
| `naive_closing` | `d_cur` is smaller than the distance from the bike's earlier position to the pedestrian's current position, and the bike displacement exceeds `prox_speed_min`. |
| `ttc` | The closing speed `(d_prev - d_cur) / dt` is above 0.1 m/s and `d_cur / closing speed` is below `ttc_threshold_s`. |

## Named configurations

`PIPELINE_CONFIGS` in `decision_testbench.py` holds the eight configurations
that `--compare` runs.

| Name | N (frames) | d_min (m) | d_max (m) | v_min (m) | k (frames) | Speed-adaptive | Rule |
|---|---|---|---|---|---|---|---|
| `default` | 58 | 1.9 | 24.8 | 0.147 | 2 | yes | `pairwise` |
| `conservative` | 60 | 1.5 | 15.0 | 0.05 | 3 | no | `pairwise` |
| `proximity_only` | 0 | 2.0 | 10.0 | 0.1 | 3 | no | `pairwise` |
| `speed_adaptive` | 60 | 1.0 | 20.0 | 0.05 | 2 | yes | `pairwise` |
| `optimized` | 58 | 1.9 | 24.8 | 0.147 | 2 | yes | `pairwise` |
| `baseline_distance_only` | 58 | 1.9 | 10.0 | 0.0 | 2 | no | `distance_only` |
| `baseline_naive_closing` | 58 | 1.9 | 24.8 | 0.147 | 2 | yes | `naive_closing` |
| `baseline_ttc` | 58 | 1.9 | 24.8 | 0.0 | 2 | yes | `ttc` |

## Comparison on the conformance scenarios

```console
$ python decision_testbench.py --compare
Loaded calibration: FOV=197.90 cx=1752.7 cy=1804.5 f=1013.3 R=1750 frame=3500x3500 pad=3840 pitch=0 k1=0 k2=0
...
--- default ---
  SCENARIO RESULTS -- N=58, d=[1.9,24.8]m, v_min=0.147, k=2
  Scenario                       TP   FP   FN   TN   Sens   Spec  SevFN    Lat WarnBgt RxnAdq Fatigue       Exp
  Safe Crossing                   0    0    0  196  0.0% 100.0%  0.0%    N/A     N/A    N/A   0.0%      safe
  Bike Approaches Ped            90   24   10  102 90.0% 81.0% 18.8%   0.33    3.7s   100%  50.4%     alert
  High-Speed Near Miss          104    1   10   81 91.2% 98.8% 11.6%   0.33    3.5s   100%  53.6%     alert
  ...
  TOTAL                        1228  370   88 4418 93.3% 92.3%  7.6%

  Alert fatigue (mean fraction of time in alert): 32.3%
```

The `TOTAL` rows of that run, one per configuration. Counts are frames at
30 FPS summed over the 24 scenarios, with the fisheye localisation error
applied (the default).

| Configuration | TP | FP | FN | TN | Sensitivity | Specificity | Severity-weighted FN | Time in alert |
|---|---|---|---|---|---|---|---|---|
| `default` | 1228 | 370 | 88 | 4418 | 93.3% | 92.3% | 7.6% | 32.3% |
| `conservative` | 1015 | 354 | 301 | 4434 | 77.1% | 92.6% | 31.0% | 28.9% |
| `proximity_only` | 738 | 125 | 578 | 4663 | 56.1% | 97.4% | 61.5% | 19.6% |
| `speed_adaptive` | 1225 | 392 | 91 | 4396 | 93.1% | 91.8% | 9.4% | 33.7% |
| `optimized` | 1228 | 370 | 88 | 4418 | 93.3% | 92.3% | 7.6% | 32.3% |
| `baseline_distance_only` | 761 | 1429 | 555 | 3359 | 57.8% | 70.2% | 58.5% | 41.1% |
| `baseline_naive_closing` | 1229 | 460 | 87 | 4328 | 93.4% | 90.4% | 7.5% | 33.8% |
| `baseline_ttc` | 1109 | 173 | 207 | 4615 | 84.3% | 96.4% | 14.7% | 27.0% |

Sensitivity is counted over actionable danger frames and specificity over
frames with no danger; both terms are defined in the
[safety model](safety-model.md#how-the-metrics-use-the-labels). The naive
closing rule reaches the same sensitivity as the pairwise rule with 90 more
false-positive frames, because comparing against the pedestrian's current
position reads a pedestrian walking toward the bike's old position as the bike
closing. Distance alone alerts on 1429 safe frames.

## Related

- [Safety model](safety-model.md) for the braking values and the ground-truth labels behind the table above
- [Detection and localisation uncertainty](uncertainty.md) for what the testbench does to positions before they reach the pipeline
- [Tuning for a new site](../deployment/tuning.md) for choosing `prox_min` on a different camera geometry
- [`decision_pipeline` API](../api/decision_pipeline.md)
