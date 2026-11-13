# Safety model

The testbench needs a ground truth for "this frame is dangerous" that does not
depend on the decision pipeline. It gets one from a kinematic model built on
cyclist braking profiles, a pedestrian reaction time and a conflict zone around
the pedestrian. This page lists the constants with the sources cited in the
code and states the labelling rule exactly.

## Cyclist braking profiles

A profile is a perception-reaction time and a braking deceleration
(`CyclistBrakingProfile`). The stopping distance is

```text
d_stop = v * prt_s + v^2 / (2 * decel_ms2)
```

| Key | PRT (s) | Deceleration (m/s²) | Source given in the code |
|---|---|---|---|
| `bicycle` | 0.84 | 1.96 | Martin & Bigazzi 2025, 85th percentile, field measurements, n = 302 ([doi](https://doi.org/10.1061/JTEPBS.TEENG-8805)) |
| `ebike` | 0.84 | 6.0 | PRT as above; deceleration from Naude et al. 2025, mean maximum in track tests, n = 127 ([doi](https://doi.org/10.1016/j.treng.2025.100404)) |
| `aashto` | 2.5 | 3.4 | AASHTO, A Policy on Geometric Design of Highways and Streets, 7th ed. (2018) |

`bicycle` is the default. A scenario selects another profile with its
`vehicle` key, and the profile then drives both the ground-truth labels and the
speed-adaptive bound of the pipeline under test, because `make_pipeline` passes
the profile's values to `DecisionPipeline`. The E-Scooter and Accelerating
E-Bike conformance scenarios use `ebike`. The `aashto` profile is there for
reference against design guidance.

Stopping distances returned by `stopping_distance` for each profile:

| Profile | 4 m/s | 6 m/s | 8 m/s | 10 m/s |
|---|---|---|---|---|
| `bicycle` | 7.44 m | 14.22 m | 23.05 m | 33.91 m |
| `ebike` | 4.69 m | 8.04 m | 12.05 m | 16.73 m |
| `aashto` | 12.35 m | 20.29 m | 29.41 m | 39.71 m |

## Pedestrian and conflict-zone constants

| Constant | Value | Unit | Meaning |
|---|---|---|---|
| `PEDESTRIAN_REACTION_S` | 1.87 | s | Reaction time of a distracted pedestrian (Fugger et al. 2000, combined average). |
| `PEDESTRIAN_CLEAR_SPEED_MS` | 1.3 | m/s | Walking speed to clear the conflict zone. |
| `CONFLICT_ZONE_HALF_WIDTH_M` | 1.0 | m | Half-width of the zone the pedestrian occupies. |
| `SWERVE_FRICTION_MU` | 0.4 | | Tyre-road friction assumed for a swerve. |
| `MAX_SEVERITY_SPEED_MS` | 12.0 | m/s | Cyclist speed at which severity reaches 1. |
| `ACTIONABLE_TTC_LOWER` | 1.87 | s | Shortest time to collision at which a warning can still help. |

## Swerve time

A cyclist can also avoid the pedestrian by moving sideways out of the conflict
zone. The code models that as reaction time plus the time to cover the
half-width `w` under constant lateral acceleration `a_lat = mu * g`:

```text
a_lat      = 0.4 * 9.81           = 3.92 m/s^2
t_maneuver = sqrt(2 * w / a_lat)  = 0.71 s
t_swerve   = 0.84 + 0.71          = 1.55 s
```

The result does not depend on forward speed. It is kept as
`CYCLIST_SWERVE_TIME_S` for comparison, and the actionable bound uses the
pedestrian's 1.87 s instead: the system exists to warn the pedestrian, so the
labels are judged against the pedestrian's response window.

## Labelling a frame

`compute_ground_truth_danger(scenario, t)` evaluates every bike and pedestrian
pair that is within 25 m of each other at time `t` and returns one
`DangerAssessment`.

For each pair it computes:

- the bike speed `v` from the scripted path and the stopping distance `sd`
  under the scenario's braking profile;
- from `time_to_collision`, which scans the next 5 s of both paths in 0.05 s
  steps: the minimum future distance `d_min`, the time `ttc` until the distance
  first drops below the 1.0 m conflict radius, and the current closing speed
  `v_close`;
- the gap to the conflict zone, `max(d_cur - 1.0, 0)`;
- a time-to-collision threshold, 1.87 s plus the time the pedestrian needs to
  walk 2.0 m at their current speed. A pedestrian slower than 0.1 m/s cannot
  clear the zone and gets 30 s.

The pair is dangerous when the paths converge and the cyclist either cannot
stop in time or is about to arrive:

```python
is_danger = (
    v_close > 0.3                      # closing, m/s
    and d_min < convergence_m          # 5.0 m: the paths actually converge
    and (sd > gap_to_conflict * stop_margin    # stop_margin = 0.8
         or ttc < ttc_threshold)
)
actionable = is_danger and ttc >= ACTIONABLE_TTC_LOWER
severity   = min((v / MAX_SEVERITY_SPEED_MS) ** 2, 1.0)
```

The convergence test keeps parallel paths out: two agents can close on each
other for a while and still pass more than 5 m apart. Severity scales with the
square of cyclist speed, as kinetic energy does.

When several pairs are dangerous the function returns the one with the highest
severity. When none is, it returns the pair with the smallest `d_min`.

## The `DangerAssessment` record

| Field | Unit | Meaning |
|---|---|---|
| `is_danger` | | The frame is physically threatening. |
| `actionable` | | Dangerous, and the time to collision is at least 1.87 s, so a warning can still change the outcome. |
| `severity` | 0 to 1 | Squared cyclist speed relative to 12 m/s. |
| `ttc` | s | Time until the pair is within 1.0 m; infinite if that does not happen within 5 s. |
| `d_min_future` | m | Minimum distance within the 5 s lookahead. |
| `closing_speed` | m/s | Positive when the distance is shrinking. |
| `bike_speed` | m/s | Cyclist ground speed. |
| `stopping_dist` | m | Stopping distance at that speed. |
| `warning_budget` | s | Time from this frame to closest approach. |
| `distance` | m | Current distance between the pair. |

## How the metrics use the labels

Dangerous frames that are no longer actionable are called imminent in the code.
The two tiers are scored differently:

| Metric | Definition |
|---|---|
| True positive | Alert on an actionable frame. |
| False negative | No alert on an actionable frame. |
| False positive | Alert on a frame with no danger of either tier. |
| True negative | No alert on a frame with no danger. |
| Sensitivity | True positives over actionable frames. |
| Specificity | True negatives over frames with no danger. |
| Severity-weighted FN | Summed severity of missed actionable frames over summed severity of all actionable frames. |
| Reaction adequacy | Share of alert onsets that leave more than 1.87 s before closest approach. |

An alert during an imminent frame counts neither for nor against the system.
The alert is correct, and the pedestrian could no longer have used it.

## Related

- [Decision pipeline](decision.md) for where the braking values enter the speed-adaptive bound
- [Scenario format](../reference/formats.md) for the `vehicle` key
- [The testbench](../simulation/testbench.md) for running the scenarios and reading the result table
