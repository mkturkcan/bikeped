# Generators

`python -m bikeped.scenarios generate` writes a seeded parametric suite to a JSON file. This page lists the families, explains what the seed controls, and shows how to load and evaluate the result.

## Command

```bash
python -m bikeped.scenarios generate --family time-gap --n 20 --seed 0 --out time-gap.json
```

| Flag | Default | Meaning |
|---|---|---|
| `--family NAME` | required | One of the six families below |
| `--n N` | `20` | Number of scenarios |
| `--seed SEED` | `0` | Seed of the jitter |
| `--out PATH` | `-` (stdout) | Output JSON file |

## Families

| Family | What is swept |
|---|---|
| `crossing-angle` | Pedestrian crossing angle from 30 to 150 degrees to the road |
| `cyclist-speed` | Cyclist speed from 3 to 11 m/s, arriving as the pedestrian is mid-lane |
| `time-gap` | Cyclist arrives -3 to +3 s relative to the pedestrian at the conflict point |
| `lateral-offset` | Cyclist passes 0 to 5 m to the side of the pedestrian |
| `group-size` | Groups of 1 to n pedestrians crossing together into one cyclist |
| `mixed-traffic` | Random mix of 1 to 4 pedestrians and 1 to 3 cyclists at random times |

The first five are sweeps: the parameter is spread evenly over its range across the `--n` scenarios, so scenario `k` of a 20-scenario `time-gap` suite always has the same gap. The seed only adds small jitter, start times within 0.2 s and speeds within 5 %. `mixed-traffic` draws its agents from the seed. In every family the same family, `n` and seed give identical output.

`expected_dominant` is not chosen by hand: each generated scenario is labelled from the testbench ground truth (`bikeped.scenarios.evaluate.label_scenario`).

## Output

The file is a JSON list of scenarios in the usual format. Each one carries metadata that identifies how it was made:

```console
$ python -m bikeped.scenarios generate --family time-gap --n 3 --seed 0 --out tg.json
wrote 3 scenarios to tg.json
```

| Scenario | `id` | `expected_dominant` |
|---|---|---|
| Time gap -3.0 s [time-gap s0 #0] | `time-gap-s0-000` | alert |
| Time gap +0.0 s [time-gap s0 #1] | `time-gap-s0-001` | alert |
| Time gap +3.0 s [time-gap s0 #2] | `time-gap-s0-002` | alert |

Every scenario in the file has `suite` set to `generated-<family>`, `family` set to the family, the tag `generated` and a one-sentence `description`, for example "The cyclist reaches the conflict point 3.0 s before the pedestrian."

## Using a generated suite

The file loads with `json` and runs through the testbench like any other scenario list:

```python
import json
from decision_testbench import FisheyeCamera, PIPELINE_CONFIGS, simulate_scenario

suite = json.load(open('tg.json'))
cam = FisheyeCamera()
for scenario in suite:
    result = simulate_scenario(scenario, PIPELINE_CONFIGS['default'], cam=cam)
```

From Python, `bikeped.scenarios.generators.generate(family, n, seed)` returns the same list without writing a file.

## Related

- [Authoring](authoring.md)
- [Extended suites](extended.md)
- [File formats](../reference/formats.md#suite-file)
