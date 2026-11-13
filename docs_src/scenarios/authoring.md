# Authoring scenarios

This page shows how to write a scenario by hand, what each field means, how paths are interpolated, and how to check the result and run it through the testbench.

## A complete scenario

A pedestrian crosses while a cyclist approaches, slows down and stops short of the crosswalk.

```python title="late_braker.py"
scenario = {
    'name': 'Late Braker',            # unique display name
    'expected_dominant': 'alert',     # most severe state held for >= 0.3 s
    'vehicle': 'bicycle',             # braking profile (optional)

    # Metadata: used by bikeped.scenarios, ignored by the testbench.
    'id': 'late-braker',
    'suite': 'local',
    'family': 'standard',
    'description': 'A cyclist brakes late and stops two metres from the crosswalk.',
    'tags': ['braking'],

    'agents': [
        # Pedestrian: 12 m across the road in 8 s, so 1.5 m/s.
        {'type': 'person', 'id': 1,
         'path': [{'x': 5, 'y': -6, 't': 0}, {'x': 5, 'y': 6, 't': 8}]},

        # Cyclist: 6 m/s for 3 s, slows over the next 3 s, then waits at x = 7.
        {'type': 'bike', 'id': 2,
         'path': [{'x': 33, 'y': 0.5, 't': 0}, {'x': 15, 'y': 0.5, 't': 3},
                  {'x': 7, 'y': 0.5, 't': 6}, {'x': 7, 'y': 0.5, 't': 8}]},
    ],
}
```

Coordinates are metres in the [scenario frame](index.md#coordinate-frame), with `x` along the road, `y` across it and the crosswalk centre at `(5, 0)`, and times are seconds.

## Scenario fields

| Key | Required | Values | Meaning |
|---|---|---|---|
| `name` | yes | string | Display name, unique across suites. Results are reported under it. |
| `expected_dominant` | yes | `idle`, `safe`, `warning`, `alert` | The dominant state the run should produce: the most severe state (alert > warning > safe > idle) that the pipeline holds for at least 0.3 s in total over the run. |
| `vehicle` | no | `bicycle`, `ebike`, `aashto` | Braking profile for ground-truth labelling and for the pipeline's stopping distance. Defaults to `bicycle`; an unknown value also falls back to `bicycle`. |
| `agents` | yes | list | One entry per pedestrian or cyclist. |

Each entry of `agents` has three required keys. The optional appearance keys are listed under [Metadata](#metadata).

| Key | Values | Meaning |
|---|---|---|
| `type` | `person`, `bike` | Which side of the pipeline the agent feeds. Anything that rides counts as `bike`. |
| `id` | integer | Unique within the scenario. The pipeline uses it as the track id. |
| `path` | list of `{'x', 'y', 't'}` | Waypoints in time order. |

## Path rules

The testbench reads paths through two functions in `decision_testbench.py`. `interpolate_agent` gives an agent's position and heading at a time `t`, and `get_scenario_duration` decides when the run ends. Their behaviour comes down to the following points.

- Position is linear between consecutive waypoints. Speed is constant on a segment and changes at the waypoint, so a smooth acceleration takes several short segments.
- An agent exists only between its first and its last waypoint time, both included. Outside that interval it is absent from the frame, as if it had not been detected. Start a path late to make an agent enter mid-scenario; the cyclist in `Fast Approach` starts at `t = 2`.
- Waypoint times must increase. Two waypoints with the same `t` make a segment of zero duration, and interpolation divides by that duration.
- To hold an agent still, repeat its position with a later time, as the cyclist above does between `t = 6` and `t = 8`.
- Heading is the direction of the current segment, in radians, 0 along `+x`. A stationary segment reports 0.
- The scenario runs until the latest last-waypoint time plus 0.5 s. `Late Braker` runs for 8.5 s, which is 256 frames at 30 FPS.
- Keep the encounter within `max_range` of the camera (25 m from `(5, -13)` by default). Stage 3 of the pipeline skips agents beyond it.

## Metadata

The scenario library, the renderer and these docs read a few extra keys. The testbench ignores them.

| Key | Meaning |
|---|---|
| `id` | Slug, unique across all suites. Used on the command line and in file names. |
| `suite` | Name of the suite the scenario belongs to. |
| `family` | Family key used for grouping and filtering. |
| `description` | One or two plain sentences. |
| `tags` | List of free-form labels. |
| `render` | Optional rendering hints, for example `{'weather': 'wet_night'}`. |

Agents accept two optional keys: `model`, a rendering hint for the CARLA renderer (`adult`, `child`, `wheelchair`, `runner`, `bicycle`, `road_bike`, `ebike`, `escooter`, `cargo_bike`, `moped`), and `label`, a short text shown next to the agent.

## Validate

```bash
python -m bikeped.scenarios validate
```

The command checks every registered scenario and exits with a non-zero status if one fails.

## Run it through the testbench

`simulate_scenario` takes the dict directly. Append this to `late_braker.py` and run it from the repository root:

```python title="late_braker.py (continued)"
from collections import Counter

from decision_testbench import (
    PIPELINE_CONFIGS, FisheyeCamera, get_scenario_duration, simulate_scenario,
)

result = simulate_scenario(scenario, PIPELINE_CONFIGS['default'], cam=FisheyeCamera())

print('duration', get_scenario_duration(scenario), 's,', len(result['times']), 'frames')
print('states  ', dict(Counter(result['states'].tolist())))
print('TP/FP/FN/TN', result['true_pos'], result['false_pos'],
      result['false_neg'], result['true_neg'])
print(f"sensitivity {result['sensitivity']:.1%}  specificity {result['specificity']:.1%}")
print(f"alert latency {result['alert_latency_s']:.2f} s  "
      f"min warning budget {result['min_warning_budget_s']:.1f} s")
```

```console
$ python late_braker.py
duration 8.5 s, 256 frames
states   {'warning': 110, 'alert': 146}
TP/FP/FN/TN 85 61 0 110
sensitivity 100.0%  specificity 64.3%
alert latency 0.00 s  min warning budget 4.8 s
```

The pipeline is in `alert` for 146 of the 256 frames. Of those, 85 are actionable danger frames and none are missed. The other 61 fall on frames that the danger model labels safe, which is where the 64.3% specificity comes from. The metrics are defined in [Conformance suite](conformance.md#scoring).

`cam=FisheyeCamera()` places the camera at its default position `(5, -13)`, so the pipeline sees camera-relative positions as it does in `python decision_testbench.py`. Without `cam`, positions are passed in scenario coordinates and `max_range` is measured from the origin. To add the fisheye localisation error, pass `add_bbox_noise=True` together with the camera; the error model is described in [Perception](../system/perception.md).

## Related

- [Scenario format reference](../reference/formats.md)
- [Generators](generators.md)
- [Safety model](../system/safety-model.md)
