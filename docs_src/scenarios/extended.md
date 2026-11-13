# Extended suites

The conformance suite is fixed by the paper. Scenarios written after it live in separate suites of `bikeped.scenarios`, so the published numbers stay reproducible while the library grows. This page describes the `extended` and `stress` suites and lists their scenarios.

## Working with a suite

```bash
python -m bikeped.scenarios list --suite extended
python -m bikeped.scenarios evaluate --suite extended
```

```python
from bikeped.scenarios import get_suite

for scenario in get_suite('extended'):
    print(scenario['id'], scenario['family'], scenario['expected_dominant'])
```

Scenarios in these suites use the conformance format, so they run in the testbench, in the [browser simulator](../simulation/browser.md) and in the [CARLA renderer](../simulation/carla/index.md) without conversion. Per-agent `model` hints (child, wheelchair, e-scooter, cargo bike and so on) change only how the renderer draws an agent.

## extended

Forty-one hand-designed urban conflicts and deliberate negatives in twelve families. Speeds follow published ranges: walking 1.2 to 1.5 m/s, older adults 0.8 to 1.0, wheelchairs 1.0 to 1.3, commuter cyclists 4 to 7, fast road cyclists 8 to 11, e-scooters 5 to 7, cargo bikes 3.5 to 5.5.

| Family | Scenarios | Covers |
|---|---:|---|
| `turning` | 4 | Turning conflicts |
| `occlusion` | 3 | Occlusion and late emergence |
| `yielding` | 4 | Yielding and braking cyclists |
| `counter-flow` | 2 | Wrong-way and counter-flow riders |
| `shared-space` | 2 | Shared space and sidewalk riding |
| `groups` | 5 | Groups and platoons |
| `vulnerable` | 4 | Vulnerable road users |
| `micromobility` | 4 | Micromobility |
| `pedestrian-behaviour` | 5 | Pedestrian behaviour |
| `near-miss` | 4 | Near-miss geometry |
| `signal-platoon` | 1 | Signal-phase platoons |
| `negative` | 3 | True negatives |

The dominant state of a run is the most severe state (alert > warning > safe > idle) that the pipeline holds for at least 0.3 s in total over the run. `expected_dominant` of an extended or stress scenario is the same rule applied to the per-frame labels of `compute_ground_truth_danger` rather than a judgement, and a test in `tests/scenarios/` keeps the two in agreement. Scenarios that the default pipeline gets wrong keep their label and carry the tag `known-hard`: either the dominant state differs from the expected one, or the first alert comes more than 0.5 s after a danger interval begins. Fourteen scenarios carry it across the library: seven false alarms, five late alerts and two in the stress suite. They are the useful ones when you change the pipeline.

The ground truth is conservative. It labels a comfortable yield, a 2 m tangential pass and the whole ±3 s time-gap sweep of the generators as danger, so a `yes` in the table below means the pipeline agrees with a cautious reference rather than with a human judgement of the scene.

```console
$ python -m bikeped.scenarios evaluate --suite extended
scenario                             dur  ag danger  alert  false lead s (per interval)  dominant expected  ok
--------------------------------------------------------------------------------------------------------------
right-hook-across-crosswalk         12.0   2   2.50   3.67   1.27 +1.3                      alert    alert  yes
left-hook-across-crosswalk          10.6   2   2.63   3.47   0.83 +0.7                      alert    alert  yes
turn-in-from-cross-street            9.4   2   3.83   3.33   0.00 -0.0                      alert    alert  yes
wide-turn-clears-pedestrian          8.5   2   0.00   3.13   3.13 -                         alert  warning  NO
...
--------------------------------------------------------------------------------------------------------------
41 scenarios, dominant state matches expected in 34; 23 of 33 danger intervals alerted no later than one frame after onset; false-alert time 33.93 s
```

`evaluate` runs `decision_testbench.simulate_scenario` with ideal perception. The columns are seconds of ground-truth danger, of alert and of false alert, the lead of the first alert over each danger interval (negative means late), and the dominant state against the expected one.

--8<-- "_generated/scenarios/extended.md"

## stress

Eight dense scenes with 12 to 24 agents in view at once, for throughput and tracker-load testing. Positions and start times are jittered with fixed seeds, so the suite is the same on every run. Two of them carry `known-hard`.

```console
$ python -m bikeped.scenarios evaluate --suite stress
...
sidewalk-tour-group                 33.5  16   0.00   5.63   5.63 -                         alert  warning  NO
crosswalk-without-cyclists          21.8  24   0.00   0.00   0.00 -                          safe     safe  yes
...
8 scenarios, dominant state matches expected in 7; 4 of 7 danger intervals alerted no later than one frame after onset; false-alert time 8.37 s
```

--8<-- "_generated/scenarios/stress.md"

## Related

- [Conformance suite](conformance.md)
- [Generators](generators.md)
- [Scenario format reference](../reference/formats.md)
