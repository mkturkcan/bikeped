# Latency

A warning computed from an old frame describes where the cyclist was. This page covers the simulated camera-latency sweep behind Figure 4, the `camera.latency_frames` compensation setting, and the per-step timing report for a running system.

## Latency sweep

```bash
mkdir -p paper
python decision_testbench.py --latency-sweep
python decision_testbench.py --no-bbox-noise --latency-sweep   # ablation
```

The sweep runs the `default` configuration over all 24 scenarios at camera delays from 0 to 495 ms in steps of 33 ms, one frame at 30 frames per second. With a delay of `n` frames the pipeline receives the detections from `n` frames earlier, while ground truth is evaluated at the current time. Each delay is run three times:

| Variant | `forecast_order` | What the tracker does with the delayed detections |
|---|---|---|
| Raw observations | 0 | Nothing. The pipeline sees stale positions |
| First-order prediction | 1 | Extrapolates each track `n` frames ahead at constant velocity |
| Second-order prediction | 2 | Extrapolates with constant acceleration, `x + v n + a n² / 2` |

```console
$ python decision_testbench.py --latency-sweep
Loaded calibration: FOV=197.90 cx=1752.7 cy=1804.5 f=1013.3 R=1750 frame=3500x3500 pad=3840 pitch=0 k1=0 k2=0

================================================================================
  CAMERA LATENCY SWEEP (0-500 ms, step=33 ms)
================================================================================

  --- Raw observations (no prediction) ---
     Delay  Frames     Sens    SevFN   AvgLat   RxnAdq  WarnBgt
  --------------------------------------------------------------
       0 ms       0   93.3%    7.6%    0.08s     88%    3.24s
      33 ms       1   92.7%    9.2%    0.10s     88%    3.18s
      66 ms       2   91.9%   11.1%    0.12s     88%    3.16s
      99 ms       3   91.1%   12.9%    0.15s     88%    3.12s
...
     495 ms      15   79.6%   36.6%    0.45s     77%    2.73s
```

`AvgLat` is the mean alert latency, `RxnAdq` the mean reaction adequacy and `WarnBgt` the mean warning budget over the scenarios where each is defined. The definitions are on the [testbench page](../simulation/testbench.md).

Selected rows from the three tables of that run:

| Delay | Sensitivity, raw | first order | second order | Warning budget, raw | first order | second order |
|---|---|---|---|---|---|---|
| 0 ms | 93.3% | 93.3% | 93.3% | 3.24 s | 3.24 s | 3.24 s |
| 33 ms | 92.7% | 92.6% | 92.5% | 3.18 s | 3.06 s | 3.01 s |
| 99 ms | 91.1% | 91.5% | 90.3% | 3.12 s | 2.64 s | 2.48 s |
| 198 ms | 88.7% | 89.3% | 86.1% | 3.01 s | 2.44 s | 2.11 s |
| 297 ms | 85.8% | 87.1% | 83.1% | 2.91 s | 2.22 s | 1.97 s |
| 396 ms | 82.7% | 85.0% | 81.4% | 2.83 s | 2.08 s | 1.81 s |
| 495 ms | 79.6% | 82.6% | 79.8% | 2.73 s | 1.95 s | 1.74 s |

Without prediction, sensitivity falls by 13.7 points across 15 frames of delay. Constant-velocity prediction recovers 3.0 of those points at 495 ms and nothing at one frame. It also shortens the mean warning budget, from 2.73 s to 1.95 s at 495 ms, and reaction adequacy drops from 77% to 52% there. Second-order prediction scores below first-order at every non-zero delay in this run, and below the raw observations from 99 ms to 396 ms.

### Files written

| File | Content |
|---|---|
| `paper/fig_camera_latency.pdf`, `.eps` | Four panels against latency: sensitivity, severity-weighted false negatives, alert latency and warning budget, each with the three variants. The latency panel marks 0.9 s and 1.3 s cyclist reaction times; the budget panel marks 1.87 s and 0.84 s pedestrian reaction times |
| `latency_sweep.json` | `delays_ms`, `delays_frames`, `fps`, and for each of `raw`, `first_order`, `second_order` the lists `sensitivity`, `sev_weighted_fn`, `alert_latency_s`, `reaction_adequacy`, `mean_warning_budget_s` |

Both land relative to the working directory. The figure needs `paper/` to exist.

## Compensation in the deployed system

```yaml title="config.yaml"
camera:
  latency_frames: 0
```

| Key | Default | Unit | Meaning |
|---|---|---|---|
| `camera.latency_frames` | 0 | frames | Frames of measured camera latency to compensate by first-order kinematic prediction. 0 disables it. At 30 fps one frame is 33 ms, so set it to the measured latency in milliseconds divided by 33 |

The testbench does not read this key. In simulation the delay and the prediction order are arguments of `simulate_scenario` (`camera_delay_frames`, `forecast_order`), and the sweep sets them for each row.

The sweep is the evidence for what to put here. At one or two frames of latency the first-order rows are within 0.2 points of the raw rows on sensitivity and lower on reaction adequacy (83% against 88% at one frame), so compensation buys little at short delays.

## Measured latency of a running system

The live system can time every stage of every frame and save the result as a NumPy array. `tools/latency_report.py` turns that array into a report.

```bash
python tools/latency_report.py                     # reads latency_data.npy
python tools/latency_report.py run1/latency_data.npy
python tools/latency_report.py latency_data.npy --csv
```

Recording is controlled by the `latency` block of `config.yaml`, and by `LATENCY_ENABLED` and its neighbours in the [bridge starter kit](../deployment/index.md):

| Key | Default | Meaning |
|---|---|---|
| `latency.enabled` | `false` | Collect per-frame timings |
| `latency.save_path` | `latency_data.npy` | Output file |
| `latency.save_interval_frames` | 500 | Write a checkpoint every N processed frames |
| `latency.warmup_frames` | 30 | Discard the first N frames while the model warms up |

No recording ships with the repository, so the report has no sample output here. Without the file the tool prints `Error: latency_data.npy not found.` and exits with status 1.

### Input format

A two-dimensional array with one row per processed frame and ten columns, in milliseconds. An array with a different column count is rejected with a `ValueError`.

| Column | Name | Stage |
|---|---|---|
| 0 | `frame_acquire` | Reading the frame from the camera |
| 1 | `preprocess` | Pad and crop to the fisheye circle |
| 2 | `yolo_preprocess` | Detector input preparation |
| 3 | `yolo_inference` | Detector forward pass |
| 4 | `yolo_postprocess` | Detector output decoding |
| 5 | `bev_tracking` | Ground-plane projection and tracking |
| 6 | `decision` | Decision pipeline and feedback |
| 7 | `viz_submit` | Frame copy and hand-off to the drawing thread |
| 8 | `total` | All steps |
| 9 | `pipeline_only` | Total without acquisition, preprocessing and visualisation |

### Report

The report has four parts:

1. A header with the number of frames, the total wall time and the mean frame rate, computed as frames divided by the sum of `total`.
2. A table with mean, median, standard deviation, minimum, maximum, 95th and 99th percentile in milliseconds for each stage, plus `Detection total` (the three detector columns added per frame), `Total (all steps)` and `Pipeline only`.
3. Each stage's share of the mean total time.
4. The pipeline-only mean with the frame rate it would sustain (`1000 / mean`), its 95th and 99th percentiles, and the same percentiles for the visualisation hand-off, which runs on a separate thread and does not block the pipeline.

`--csv` also writes the raw array next to the input as `latency_data.csv`, with the column names as the header row.

## Related

- [Reproduce the paper](index.md)
- [Configuration reference](../reference/config.md)
- [Hardware](../deployment/hardware.md)
