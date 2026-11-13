# Reproduce the paper

Every table and figure in [arXiv 2604.17046](https://arxiv.org/abs/2604.17046) comes from scripts in the repository root. This page gives the single command that runs them all, the command behind each result, and the output to compare against.

## Before you run anything

```bash
pip install -r requirements.txt
mkdir -p paper
```

Run every command from the repository root. The scripts open `config.yaml` and `camera_calibration.json` by relative path, and the testbench falls back to built-in lens defaults with the message `No camera_calibration.json found, using defaults` if it cannot find the calibration file.

Figures are saved into `paper/`, and no script creates that directory. With it missing, `generate_figures.py` stops with `FileNotFoundError: [Errno 2] No such file or directory: 'paper/fig_deployment.eps'`.

## One command

```bash
python run_experiments.py                   # every experiment, about 45 minutes
python run_experiments.py --skip-optimizer  # without the optimizer, about 10 minutes
```

`run_experiments.py` runs the steps below in order through the shell as `python ...`, so `python` must resolve to the interpreter that has the requirements installed. It prints a banner before each step and exits with status 1 at the first step that fails.

| # | Command | What it reproduces |
|---|---|---|
| 1 | `python decision_testbench.py --compare` | Table 2: pipeline comparison with fisheye localisation error |
| 2 | `python decision_testbench.py --latency-sweep` | Figure 4: camera latency sweep, 0 to 500 ms |
| 3 | `python decision_testbench.py --monte-carlo --mc-trials 50` | Monte Carlo with size-aware detection drops, one camera, 50 trials |
| 4 | `python decision_testbench.py --no-bbox-noise --compare` | Ablation: pipeline comparison without localisation error |
| 5 | `python decision_testbench.py --no-bbox-noise --latency-sweep` | Ablation: latency sweep without localisation error |
| 6 | `python decision_testbench.py --optimize` | Parameter search by differential evolution. Skipped with `--skip-optimizer` |
| 7 | `python decision_testbench.py --error-map` | Bounding-box localisation error summary |
| 8 | `python generate_figures.py` | Publication figures as EPS and PDF |
| 9 | `python sim_visualizer.py --no-display` | One schematic MP4 per conformance scenario |

Three inputs are outside this script. `run_height_pitch_sweep.py` (Figure 8b, 30 to 60 minutes) writes `height_pitch_sweep.json`, `tools/eval_models.py` writes `eval_cache.json`, and `crosswalk_analysis.py` prints the road-width tables. Run them first if you need the figures that depend on them; see [Camera placement](placement.md).

## Table 2: pipeline comparison

```console
$ python decision_testbench.py --compare
Loaded calibration: FOV=197.90 cx=1752.7 cy=1804.5 f=1013.3 R=1750 frame=3500x3500 pad=3840 pitch=0 k1=0 k2=0

======================================================================
  PIPELINE COMPARISON
======================================================================

--- default ---
...
  TOTAL                        1228  370   88 4418 93.3% 92.3%  7.6%

  Alert fatigue (mean fraction of time in alert): 32.3%
```

The command prints one per-scenario table for each entry of `PIPELINE_CONFIGS`. The `TOTAL` rows are:

| Configuration | Sensitivity | Specificity | Severity-weighted FN | Time in alert |
|---|---|---|---|---|
| `default` | 93.3% | 92.3% | 7.6% | 32.3% |
| `conservative` | 77.1% | 92.6% | 31.0% | 28.9% |
| `proximity_only` | 56.1% | 97.4% | 61.5% | 19.6% |
| `speed_adaptive` | 93.1% | 91.8% | 9.4% | 33.7% |
| `optimized` | 93.3% | 92.3% | 7.6% | 32.3% |
| `baseline_distance_only` | 57.8% | 70.2% | 58.5% | 41.1% |
| `baseline_naive_closing` | 93.4% | 90.4% | 7.5% | 33.8% |
| `baseline_ttc` | 84.3% | 96.4% | 14.7% | 27.0% |

`default` and `optimized` hold the same parameters, so their rows match. The 93.3% sensitivity and 92.3% specificity of `default` are the figures quoted in the paper's abstract. The run has no random component: repeating it gives the same counts.

With `--no-bbox-noise` the same eight totals are:

| Configuration | Sensitivity | Specificity | Severity-weighted FN | Time in alert |
|---|---|---|---|---|
| `default` | 91.9% | 92.5% | 8.4% | 31.9% |
| `conservative` | 75.9% | 91.9% | 31.9% | 29.2% |
| `proximity_only` | 54.1% | 97.0% | 61.7% | 19.4% |
| `speed_adaptive` | 92.2% | 91.8% | 9.6% | 33.5% |
| `optimized` | 91.9% | 92.5% | 8.4% | 31.9% |
| `baseline_distance_only` | 54.7% | 71.0% | 61.2% | 39.8% |
| `baseline_naive_closing` | 91.9% | 90.8% | 8.4% | 33.2% |
| `baseline_ttc` | 84.0% | 96.5% | 14.8% | 27.1% |

The metric definitions are on the [testbench page](../simulation/testbench.md), and the configurations are described under [Decision pipeline](../system/decision.md).

## Commands by result

| Result | Command | Writes |
|---|---|---|
| Table 2 | `python decision_testbench.py --compare` | stdout |
| Table 2 ablation | `python decision_testbench.py --no-bbox-noise --compare` | stdout |
| Figure 4 | `python decision_testbench.py --latency-sweep` | stdout, `latency_sweep.json`, `paper/fig_camera_latency.pdf` and `.eps` |
| Monte Carlo | `python decision_testbench.py --monte-carlo --mc-trials 50` | stdout |
| Optimizer (about 30 minutes) | `python decision_testbench.py --optimize` | stdout |
| Localisation error summary | `python decision_testbench.py --error-map` | stdout |
| Figure 8b (30 to 60 minutes) | `python run_height_pitch_sweep.py` | stdout, `height_pitch_sweep.json` |
| Road-width analysis | `python crosswalk_analysis.py` | stdout |
| Publication figures | `python generate_figures.py` | `paper/fig_*.eps` and `.pdf` |
| Scenario videos | `python sim_visualizer.py --no-display` | `sim_NN_<scenario>.mp4` in the working directory |

Details for the longer runs are on their own pages: [Optimizer](optimizer.md), [Latency](latency.md) and [Camera placement](placement.md).

## Figures from `generate_figures.py`

Each figure is saved as `.eps` and `.pdf` under `paper/`.

| File stem | Content | Input |
|---|---|---|
| `fig_height_pitch` | Sensitivity heatmap over camera height and pitch, one camera | `height_pitch_sweep.json`. Skipped with a message if the file is missing |
| `fig_ar_curve` | Average recall against bounding-box area, with distance annotations | `eval_cache.json`. Skipped with a message if the file is missing |
| `fig_deployment` | One camera against two across road widths of 30, 40, 50 and 60 ft | Values stored in the script |
| `fig_pipeline` | Sensitivity against specificity for four pipeline configurations | Values stored in the script |
| `fig_combined` | The pipeline scatter next to the height and pitch heatmap | `height_pitch_sweep.json` |
| `fig_warning_budget` | Warning budget per danger scenario with reaction-time thresholds | Runs the testbench when the figure is drawn |

`fig_deployment` and `fig_pipeline` plot numbers written into `generate_figures.py`. They do not change when you re-run the testbench, so compare the tables above with the stdout of the commands when you check a result.

## Related

- [Offline testbench](../simulation/testbench.md)
- [Dataset](../dataset.md)
- [Command-line reference](../reference/cli.md)
