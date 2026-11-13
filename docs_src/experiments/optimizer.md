# Optimizer

The five decision parameters shipped in `config.yaml` were chosen by a differential-evolution search over the 24 conformance scenarios. This page describes the search space, the cost function and how to run the search yourself.

```bash
python decision_testbench.py --optimize
python decision_testbench.py --optimize --w-fn 8 --w-fp 1 --w-lat 2   # weight misses more heavily
python decision_testbench.py --optimize --no-bbox-noise              # search without localisation error
```

The README puts one run at about 30 minutes. The search uses every CPU core (`workers=-1`).

## Search space

`optimize_parameters` in `decision_testbench.py` searches five values. The other fields of `DecisionParams` stay at their defaults during the search: pairwise closing rule, speed-adaptive upper bound on, `max_range` 25 m.

| Parameter | Bounds | Unit | Type | Meaning |
|---|---|---|---|---|
| `bike_memory_frames` | 0 to 120 | frames | integer | How long a cyclist sighting keeps stage 2 open |
| `prox_min` | 0.5 to 5.0 | m | real | Lower bound of the pair distance that can alert |
| `prox_max` | 5.0 to 25.0 | m | real | Upper bound of the pair distance, before speed adaptation |
| `prox_speed_min` | 0.01 to 0.5 | m | real | Minimum cyclist displacement over the lookback window |
| `lookback_frames` | 1 to 10 | frames | integer | Lookback `k` of the closing check |

A candidate with `prox_min >= prox_max` is rejected with a cost of `1e6`. [Decision pipeline](../system/decision.md) explains what each parameter does at run time.

## Cost function

For each candidate, `optimization_cost` runs all 24 scenarios once, deterministically, and combines six terms. Lower is better.

```text
cost = w_fn  * sev_fn
     + w_fp  * fp_rate
     + w_lat * avg_latency
     + 3.0   * rxn_inadequacy
     + 1.0   * fatigue_penalty
     + sens_penalty
```

| Term | Definition | Weight |
|---|---|---|
| `sev_fn` | Severity-weighted false-negative rate, averaged over scenarios that contain actionable danger frames | `--w-fn`, default 5.0 |
| `fp_rate` | Alert frames on safe frames divided by all safe frames, pooled over scenarios | `--w-fp`, default 1.0 |
| `avg_latency` | Mean time in seconds from the first actionable danger frame to the first alert. Counts as 5.0 if no scenario alerts | `--w-lat`, default 2.0 |
| `rxn_inadequacy` | One minus the mean reaction adequacy: the share of alert onsets that leave the pedestrian less than 1.87 s | 3.0, fixed |
| `fatigue_penalty` | Mean fraction of time in alert above 0.20, zero below that | 1.0, fixed |
| `sens_penalty` | `20 * max(0, 0.90 - sensitivity)` on pooled frame-level sensitivity | fixed |

The severity weighting makes a missed fast cyclist cost more than a missed slow one. The last three terms keep the search away from two degenerate answers: alerting so late that the warning is useless, and alerting all the time. The metric definitions are on the [testbench page](../simulation/testbench.md), and severity is defined under [Safety model](../system/safety-model.md).

## Solver settings

The search calls `scipy.optimize.differential_evolution` with these settings:

| Setting | Value |
|---|---|
| Strategy | `best1bin` |
| Population | 30 vectors, supplied explicitly |
| Generations | at most 100 (`maxiter`) |
| Mutation | dithered in (0.5, 1.5) |
| Recombination | 0.9 |
| Tolerance | `1e-4` |
| Seed | 42 |
| Integer dimensions | `bike_memory_frames`, `lookback_frames` |
| Parallelism | all cores, deferred updating |

The initial population is seeded around the shipped values instead of sampled uniformly: the vector `[58, 1.9, 24.8, 0.147, 2]`, ten perturbations within 10% of each parameter's range, and nineteen within 40%, all clipped to the bounds.

## Output

The run prints a header, one line per generation from SciPy, and the result:

```text
======================================================================
  OPTIMIZING DECISION PARAMETERS
  Weights: w_fn=5.0, w_fp=1.0, w_lat=2.0
  BBox noise: enabled
======================================================================
```

It ends with `Optimal parameters:` in the form `N=<memory>, d=[<min>,<max>]m, v_min=<speed>, k=<lookback>`, the final cost and whether SciPy reported convergence. Two per-scenario tables follow, one for the parameters it found and one for `PIPELINE_CONFIGS['default']`, so the two can be compared row by row.

## From the result to `config.yaml`

The optimizer does not write any file. The selected values are recorded in three places, which must agree:

```yaml title="config.yaml"
decision:
  bike_memory_frames: 58
  prox_min: 1.9
  prox_max: 24.8
  prox_speed_min: 0.147
  lookback_frames: 2
```

- `config.yaml`, read by the deployed system.
- `DecisionParams` defaults and `PIPELINE_CONFIGS['default']` (also `'optimized'`) in `decision_testbench.py`, used by every experiment.
- The seed vector `current_best` inside `optimize_parameters`.

The bridge starter kit ships the same values in its `DEC_*` settings; see [Tuning](../deployment/tuning.md).

!!! note "Constructor defaults differ"
    `DecisionPipeline()` called without arguments uses different defaults (`bike_memory_frames=77`, `prox_min=0.5`, `prox_max=20.7`, `prox_speed_min=0.098`, `lookback_frames=1`). Build pipelines with `make_pipeline(PIPELINE_CONFIGS['default'])` or pass the `config.yaml` values explicitly to get the configuration evaluated in the paper.

If you re-run the search with a different scenario suite or different weights, update all three places together, then run `python decision_testbench.py --compare` to confirm the `default` row.

## Related

- [Reproduce the paper](index.md)
- [Configuration reference](../reference/config.md)
- [Scenario authoring](../scenarios/authoring.md)
