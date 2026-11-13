# Camera placement

Where the camera sits decides how large a cyclist appears on the sensor, and with it how often the detector finds them. This page covers the two scripts that study mounting height, pitch, road width and a second camera, and the visualizer options for looking at a placement.

## What the analysis models

Both scripts use the Monte Carlo loop in `crosswalk_analysis.run_mc`. It differs from the main testbench in four ways that matter when you read its numbers:

- It builds its own scenario for a given road width (`make_scenario`): a pedestrian walks along the crosswalk at `x = 5` and a cyclist rides toward it along the road.
- Camera 1 stands on a pole at the kerb, at `(5, -width / 2)`, facing across the road. With two cameras, the second stands at `(5, +width / 2)` and faces back.
- A detection is dropped at random with the size-aware recall for its projected bounding-box area. With two cameras the recalls combine as `1 - (1 - AR1) * (1 - AR2)`.
- Positions enter the pipeline without localisation error and without the tracker, and sensitivity is counted over all danger frames. The main testbench counts actionable frames only.

Each trial is seeded with its index, and the pipeline is `PIPELINE_CONFIGS['optimized']`.

| Constant | Value | Meaning |
|---|---|---|
| `ROAD_WIDTHS_FT` | 30, 40, 50, 60 | Road widths in the sweep, in feet |
| `CAM_HEIGHT_DEFAULT` | 5.5 m | Mounting height unless swept |
| `CAM_HEIGHT_MIN`, `CAM_HEIGHT_MAX` | 2.5 m, 7.5 m | Range of traffic-signal mounting heights considered |
| `CAM_PITCH_MIN`, `CAM_PITCH_MAX` | -60°, 0° | 0 is horizontal, negative looks down |
| `PED_SPEED_MS` | 1.3 m/s | Pedestrian walking speed |
| `BIKE_SPEED_MS` | 8.3 m/s | Cyclist speed, about 30 km/h |
| Lens field of view | 220° | Set in `make_cam_at`. The calibrated lens in `camera_calibration.json` is 197.9° |

## The recall curve comes from `eval_cache.json`

Both scripts load size-dependent recall from `eval_cache.json`, the file written by `tools/eval_models.py`. The file is not in the repository. Without it they print a warning and use a flat recall of 0.747 at every object size, and then height and pitch have no effect on the result:

```console
$ python crosswalk_analysis.py
  Warning: eval_cache.json not found, using flat AR defaults
...
  Cyclist AR profile (camera at y=-13, h=5.5m)
  Bike from CW   3D dist   BBox area      AR
  ------------------------------------------
          20m     24.5m         673  0.747
          10m     17.3m        1336  0.747
           0m     14.1m        1633  0.747
...
  1-CAMERA: height x pitch -> sensitivity
    h\p    -15d    -20d    -25d    -30d    -35d    -40d    -45d    auto
  -----------------------------------------------------------------------
   2.5m  52.0%  52.0%  52.0%  52.0%  52.0%  52.0%  52.0%  52.0%
   3.0m  52.0%  52.0%  52.0%  52.0%  52.0%  52.0%  52.0%  52.0%
```

Every cell of the height and pitch grid reads 52.0% in that run. Produce the cache first if you want the placement study to say anything about placement; the detector evaluation is described under [Detection and localisation uncertainty](../system/uncertainty.md).

The second camera does show up with flat recall, because fusion is modelled separately from object size. The same run printed:

| Road width | One camera | Two cameras |
|---|---|---|
| 30 ft | 50.8% ± 6.1% | 80.0% ± 3.8% |
| 40 ft | 53.7% ± 6.2% | 84.7% ± 3.8% |
| 50 ft | 55.3% ± 6.6% | 87.1% ± 3.8% |
| 60 ft | 55.1% ± 6.8% | 87.0% ± 3.9% |

These are sensitivities with flat recall, 20 trials per row. They are not the paper's numbers, which use the measured recall curve.

## `crosswalk_analysis.py`

```bash
python crosswalk_analysis.py
```

The script takes no arguments and prints five sections:

1. Cyclist recall profile: 3D distance, projected box area and recall for a cyclist 0 to 20 m from the crosswalk, with the camera at `y = -13` and 5.5 m height.
2. Road-width sweep at the default height, one and two cameras, sensitivity and severity-weighted false-negative rate as mean ± standard deviation over 20 trials.
3. Joint height and pitch grid on the 40 ft road: ten heights from 2.5 to 7.5 m against pitches from -15° to -45° in 5° steps plus `auto`, for one and for two cameras, with the best cell of each.
4. Best height and pitch per road width.
5. A compact table of the road-width results in the layout used in the paper.

The run is long. A 110-second time box covered sections 1 to 3.

## `run_height_pitch_sweep.py`

```bash
python run_height_pitch_sweep.py
```

This is the dense grid behind Figure 8b. It takes no arguments.

| Setting | Value |
|---|---|
| Heights | 1.5 to 7.5 m in 0.5 m steps, 13 values |
| Pitches | 0° to -90° in 10° steps, 10 values |
| Road width | 40 ft |
| Cameras | 1 |
| Trials per cell | 20 |
| Total | 130 cells, 2600 Monte Carlo runs |
| Run time | 30 to 60 minutes, by the script's own estimate |

It prints the grid row by row as it goes and saves `height_pitch_sweep.json` in the working directory:

```json
{
  "heights": [1.5, 2.0, "..."],
  "pitches": [0, -10, "..."],
  "road_ft": 40,
  "n_trials": 20,
  "sensitivity_pct": [["13 rows of 10 values, in percent"]]
}
```

`generate_figures.py` reads that file for `fig_height_pitch` and `fig_combined`, and skips both figures when it is absent.

## Looking at a placement

`sim_visualizer.py` renders any conformance scenario from a chosen mounting position, which is the quickest way to see what a height or pitch does to the view.

```bash
python sim_visualizer.py --scenarios 1 --no-display                          # deployed geometry
python sim_visualizer.py --scenarios 1 --height 5.5 --pitch -30 --no-display # higher, tilted down
python sim_visualizer.py --scenarios 1 --two-cam --no-display                # adds the split view
```

| Option | Default | Effect |
|---|---|---|
| `--height` | 3.6576 m | Camera height for both poles. The help text says 5.5; the code falls back to the 12 ft deployment height in `CAM_POLE` |
| `--pitch` | automatic | Pitch in degrees, negative looks down. Left unset, the camera is aimed at the ground at the distance of its lateral offset: `-atan2(height, abs(cam_y))` |
| `--two-cam` | off | Renders each scenario twice: the single-camera video, then a side-by-side video with a second camera at `(5, 13)` facing back across the road |

The pole camera is at `(5, -13)` with heading 90°. Output names carry the placement: `sim_01_bike_approaches_ped_h5.5_p-30.mp4`, with `_2cam` inserted for the split view. Each camera view is 1080 by 1080 pixels.

The browser simulator has the same idea as sliders: camera position, height, field of view and heading, with presets for one pole, two poles, a central mount and an overhead mount. See [Browser simulator](../simulation/browser.md).

## Related

- [Perception](../system/perception.md)
- [Hardware](../deployment/hardware.md)
- [Reproduce the paper](index.md)
