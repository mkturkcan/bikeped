# Scenarios

A scenario is a scripted encounter between pedestrians and cyclists at one crosswalk, written as timed waypoints. This page covers what a scenario contains, the coordinate frame, the suites that ship with the repo, and how to reach them from Python and from the command line.

## Scenario dict

A scenario is a plain Python dict: a name, the state the warning light should show for most of the run, an optional braking profile, and a list of agents that each follow a path. This is `Fast Approach` from the paper's conformance set, as it appears in `decision_testbench.SCENARIOS`:

```python
{
    'name': 'Fast Approach',
    'expected_dominant': 'alert',
    'vehicle': 'ebike',
    'agents': [
        {'type': 'person', 'id': 1,
         'path': [{'x': 5, 'y': -4, 't': 0}, {'x': 5, 'y': -2, 't': 1.5},
                  {'x': 5, 'y': 0, 't': 3}, {'x': 5, 'y': 2, 't': 4.5},
                  {'x': 5, 'y': 4, 't': 6}]},
        {'type': 'bike', 'id': 2,
         'path': [{'x': 25, 'y': 0.5, 't': 2}, {'x': 13, 'y': 0.2, 't': 3},
                  {'x': 1, 'y': -0.1, 't': 4}]},
    ],
}
```

The pedestrian walks across the road in six seconds. The cyclist appears at `t = 2` and covers 24 m in two seconds. The testbench steps the scenario at 30 FPS, interpolates every agent at each frame, feeds the positions to the [decision pipeline](../system/decision.md) and labels the frame with the [ground-truth danger model](../system/safety-model.md). The field-by-field reference is in [Authoring](authoring.md) and [File formats](../reference/formats.md).

## Coordinate frame

<figure class="diagram">
--8<-- "assets/figures/scenario-frame.svg"
<figcaption>The scenario frame seen from above, with one illustrative cyclist path and one pedestrian path.</figcaption>
</figure>

Coordinates are metres on the ground plane and times are seconds. The table lists the two axes and the two fixed points of the frame.

| Axis or point | Meaning |
|---|---|
| `x` | Along the road. Cyclists travel along it. |
| `y` | Across the road. Pedestrians cross along it. |
| `(5, 0)` | Centre of the crosswalk. The visualizer and the browser simulator draw its stripes from `y = -12` to `y = 12`. |
| `(5, -13)` | Default camera position (`FisheyeCamera.cam_x`, `cam_y`). The camera looks along `+y`, across the road. |
| `t` | Seconds from the start of the scenario. |

Headings are derived, not authored: `interpolate_agent` returns the direction of the current path segment in radians, with 0 pointing along `+x`.

The pipeline receives positions relative to the camera, as it does on the deployed system. `max_range` (25 m in `config.yaml`) is therefore a distance from the camera at `(5, -13)`, and an agent on the road at `x = 34` is outside it until it gets closer.

## Suites

A suite is an ordered list of scenarios with a name. `list_suites()` returns the names that are registered.

The conformance suite is the paper's set of 24, kept in `decision_testbench.SCENARIOS` and never edited, because every number in the paper comes from it. See [Conformance suite](conformance.md). Two more suites are registered next to it:

| Suite | Scenarios | Contents |
|---|---:|---|
| `conformance` | 24 | The paper's scenarios, unchanged, in paper order |
| `extended` | 41 | Hand-designed urban conflicts and deliberate negatives in twelve families |
| `stress` | 8 | Dense scenes with 12 to 24 agents for throughput and tracker load |

Both are described in [Extended suites](extended.md). [Generators](generators.md) produce further suites on demand from six parametric families.

## Python API

```python
from bikeped.scenarios import list_suites, get_suite, get_scenario, all_scenarios

list_suites()                        # suite names
conformance = get_suite('conformance')   # the paper's 24, in paper order

get_scenario('fast-approach')        # by id (slug)
get_scenario('Fast Approach')        # by exact name
get_scenario('conformance:7')        # by suite and index

every = all_scenarios()              # all suites, concatenated
```

Each returned dict is a normal scenario with extra metadata keys (`id`, `suite`, `family`, `description`, `tags`). The testbench ignores those keys, so a registry scenario can go straight into `simulate_scenario`:

```python
from decision_testbench import PIPELINE_CONFIGS, FisheyeCamera, simulate_scenario
from bikeped.scenarios import get_scenario

result = simulate_scenario(get_scenario('fast-approach'),
                           PIPELINE_CONFIGS['default'], cam=FisheyeCamera())
```

## Command line

| Command | What it does |
|---|---|
| `python -m bikeped.scenarios list` | Lists scenarios. `--suite` and `--family` filter, `--json` prints machine-readable output. |
| `python -m bikeped.scenarios show ID` | Prints one scenario: metadata, agents and waypoints. |
| `python -m bikeped.scenarios validate` | Checks every scenario for structural errors. `--suite` limits the check. |
| `python -m bikeped.scenarios evaluate` | Runs scenarios through the decision pipeline and reports the metrics. Takes `--suite`, `--config` and `--json`. |
| `python -m bikeped.scenarios generate` | Samples a new suite from a family and writes it to a JSON file. See [Generators](generators.md). |
| `python -m bikeped.scenarios export` | Writes `catalog.json` and one SVG thumbnail per scenario. |

```text
--8<-- "_generated/cli/bikeped_scenarios.txt"
```

## In this section

<div class="grid-cards" markdown>

- [Conformance suite](conformance.md)

    The paper's 24 scenarios, how they are scored and what the default pipeline does on each.

- [Extended suites](extended.md)

    Scenarios beyond the paper's set, grouped by what they stress.

- [Authoring](authoring.md)

    Write a scenario by hand, validate it and run it through the testbench.

- [Generators](generators.md)

    Sample a suite from a scenario family, reproducible by seed.

</div>
