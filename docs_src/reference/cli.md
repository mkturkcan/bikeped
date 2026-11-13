# Command line

Every script and module entry point in the repository, with what it does, a usage example and its help text. Run the commands from the repository root: the scripts open `config.yaml` and `camera_calibration.json` by relative path.

| Command | Purpose |
|---|---|
| [`python decision_testbench.py`](#decision_testbenchpy) | Evaluate the decision pipeline on the conformance scenarios. |
| [`python run_experiments.py`](#run_experimentspy) | Run every experiment in the paper in sequence. |
| [`python run_height_pitch_sweep.py`](#run_height_pitch_sweeppy) | Monte Carlo sweep over camera height and pitch. |
| [`python crosswalk_analysis.py`](#crosswalk_analysispy) | Performance against road width for one and two cameras. |
| [`python generate_figures.py`](#generate_figurespy) | Write the publication figures. |
| [`python sim_visualizer.py`](#sim_visualizerpy) | Render schematic scenario videos. |
| [`python -m bikeped.scenarios`](#python-m-bikepedscenarios) | List, inspect, validate, evaluate, generate and export scenarios. |
| [`python -m bikeped.sim`](#python-m-bikepedsim) | Probe a CARLA server, start one, and render scenarios. |
| [`python calibration/calibrate_fisheye.py`](#calibrationcalibrate_fisheyepy) | Fit the lens model to checkerboard frames. |
| [`python calibration/capture_calibration.py`](#calibrationcapture_calibrationpy) | Capture checkerboard frames from a live camera. |
| [`python calibration/compare_fisheye_models.py`](#calibrationcompare_fisheye_modelspy) | Fit four projection models and compare their error. |
| [`python tools/eval_models.py`](#toolseval_modelspy) | Evaluate detector checkpoints and cache per-class AP and AR. |
| [`python tools/latency_report.py`](#toolslatency_reportpy) | Per-step latency statistics from a recorded run. |
| [`python tools/carla_find_crosswalks.py`](#toolscarla_find_crosswalkspy) | Extract crosswalk geometry from a CARLA map. |
| [`python tools/build_docs.py`](#toolsbuild_docspy) | Build this site. |
| [`python bridge.py`](#bridge_starterbridgepy) | Run the live starter-kit bridge (from `bridge_starter/`). |

## Testbench and experiments

### decision_testbench.py

Runs the scripted scenarios through the decision pipeline and prints one row of frame counts and rates per scenario. The mode flags are checked in this order, and the first one present decides what runs: `--error-map`, `--compare`, `--monte-carlo`, `--latency-sweep`, `--optimize`. With no mode flag, the `default` configuration runs once over all scenarios.

```bash
python decision_testbench.py --compare                    # every named configuration, with localisation error
python decision_testbench.py --no-bbox-noise --compare    # the same without localisation error
python decision_testbench.py --monte-carlo --mc-trials 50 --failure-model gilbert --burst-length 3
python decision_testbench.py --optimize --w-fn 5 --w-fp 1 --w-lat 2
```

`--no-bbox-noise` applies to the five modes. The plain run without a mode flag never adds localisation error, which is why its totals equal those of `--no-bbox-noise --compare` for the `default` configuration. `--failure-model` and `--burst-length` are read by `--monte-carlo` only, and the three `--w-*` weights by `--optimize` only.

```text
--8<-- "_generated/cli/decision_testbench.txt"
```

### run_experiments.py

Calls the testbench once per experiment, then `generate_figures.py` and `sim_visualizer.py --no-display`, and stops at the first step that fails. The script's docstring puts the full run at about 45 minutes and the run with `--skip-optimizer` at about 10.

```bash
python run_experiments.py --skip-optimizer
```

```text
--8<-- "_generated/cli/run_experiments.txt"
```

### run_height_pitch_sweep.py

Takes no flags. It sweeps 13 camera heights (1.5 to 7.5 m) against 10 pitches (0 to -90 degrees) on a 40 ft road with 20 Monte Carlo trials per cell, and writes `height_pitch_sweep.json` to the current directory. If `eval_cache.json` is present, its measured recall replaces the flat defaults.

```bash
python run_height_pitch_sweep.py
```

### crosswalk_analysis.py

Takes no flags. It evaluates the pipeline as a function of road width for one-camera and two-camera deployments and prints the result.

```bash
python crosswalk_analysis.py
```

### generate_figures.py

Takes no flags. It writes six figures as EPS and PDF into `paper/`: `fig_height_pitch`, `fig_ar_curve`, `fig_deployment`, `fig_pipeline`, `fig_combined` and `fig_warning_budget`. The height-pitch heatmap needs `height_pitch_sweep.json` and the recall curve needs `eval_cache.json`; each is skipped with a message when its input is missing. The script does not create the output directory:

```bash
mkdir -p paper
python generate_figures.py
```

!!! warning "Three scripts do not parse arguments"
    `run_height_pitch_sweep.py`, `crosswalk_analysis.py` and `generate_figures.py` ignore their command line. Passing `--help` starts the run.

### sim_visualizer.py

Renders the scenarios as a schematic 3D view through the fisheye model, with the pipeline state drawn on each frame. Each scenario becomes `sim_<index>_<name>.mp4` in the current directory. `--scenarios` takes zero-based indices into the conformance list, and `--pipeline` takes the name of a testbench configuration.

```console
$ python sim_visualizer.py --no-display --scenarios 7
Rendering 1 scenarios, single-cam

  Rendering: Fast Approach (1-cam, 6.5s)
    Saved: sim_07_fast_approach.mp4 (196 frames)

Done.
```

```text
--8<-- "_generated/cli/sim_visualizer.txt"
```

## Scenario library

### python -m bikeped.scenarios

```text
python -m bikeped.scenarios list      [--suite NAME] [--family NAME] [--json]
python -m bikeped.scenarios show      ID
python -m bikeped.scenarios validate  [--suite NAME]
python -m bikeped.scenarios evaluate  [--suite NAME] [--config default] [--json]
python -m bikeped.scenarios generate  --family NAME [--n 20] [--seed 0] [--out suite.json]
python -m bikeped.scenarios export    --out docs_src/assets/scenarios [--simulator simulator]
```

| Subcommand | Purpose |
|---|---|
| `list` | Print the scenarios of one suite or of all suites, optionally filtered by family. |
| `show` | Print one scenario. `ID` is a slug, an exact name, or `suite:index` such as `conformance:7`. |
| `validate` | Check format, speeds and unique ids; exits with status 1 on problems. |
| `evaluate` | Run each scenario through `simulate_scenario` with ideal perception and a named pipeline configuration, and print danger, alert and false-alert time, alert lead and dominant state. |
| `generate` | Draw `--n` scenarios from a parametric family with a fixed seed and write them to a JSON file. |
| `export` | Write `catalog.json` and one SVG thumbnail per scenario; with `--simulator DIR`, also `DIR/scenarios.extended.js` for the browser simulator. |

```text
--8<-- "_generated/cli/bikeped_scenarios.txt"
```

See [scenarios](../scenarios/index.md) and [generators](../scenarios/generators.md).

## CARLA renderer

### python -m bikeped.sim

```text
python -m bikeped.sim [-v] probe    [--host 127.0.0.1] [--port 2000] [--json]
python -m bikeped.sim [-v] server   --engine ue4|ue5 [--root PATH] [--port 2000] [--quality Epic|Low] [--windowed]
python -m bikeped.sim [-v] presets
python -m bikeped.sim [-v] render   (--scenario ID [--suite NAME] | --suite NAME --all) [options]
python -m bikeped.sim [-v] export   RUN_DIR --format coco|yolo [--view pole] [--out PATH]
```

`render` options and their defaults:

| Option | Default | Meaning |
|---|---|---|
| `--scenario ID` | | Slug, exact name, `suite:index`, or an index into `--suite` (conformance when no suite is given) |
| `--suite NAME`, `--all` | | Render every scenario of a suite |
| `--engine auto\|ue4\|ue5` | `auto` | Engine you expect; `auto` accepts whatever the server runs |
| `--host`, `--port` | `127.0.0.1`, `2000` | Server address |
| `--launch`, `--root PATH` | off | Start a managed server for this run from `--root` (or `CARLA_UE4_ROOT` / `CARLA_UE5_ROOT`) and stop it afterwards |
| `--map NAME` | current map | Map to load first; a fresh server has `Town10HD_Opt` |
| `--crosswalk N` | `4` | Crosswalk index: `crosswalks.json` on `Town10HD_Opt`, live map polygons elsewhere |
| `--crosswalks-file PATH` | `crosswalks.json` | Alternative crosswalk file |
| `--rig LIST` | `pole` | Comma list of views: `pole`, `fisheye`, `chase`, `orbit`, `street`, `bev` |
| `--weather NAME` | scenario hint, else `clear_noon` | One of the nine weather presets |
| `--drive velocity\|kinematic` | `kinematic` | How actors follow the script |
| `--ambient none\|light\|busy` | `none` | Background traffic and pedestrians |
| `--quality draft\|standard\|cinematic` | `standard` | Image size and post-processing |
| `--layout single\|split\|quad` | `single` | Arrangement of `video/composite.mp4` |
| `--no-overlay` | off | Skip the HUD videos |
| `--save-frames none\|jpg\|png` | `none` | Also write every frame as an image |
| `--out DIR` | `renders` | Parent folder of the run directory |
| `--seed N`, `--fps N` | `0`, `30` | Random seed; frame rate and simulation step |
| `--config NAME` | `default` | Key of `decision_testbench.PIPELINE_CONFIGS` for the decision state |
| `--cyclist-proxy walker` | off | Render cyclists as walkers on servers without a rideable two-wheeler |
| `--max-frames N` | all | Stop after N frames |

| Subcommand | Purpose |
|---|---|
| `probe` | Connect to a server and print the engine, the versions, the maps and what the build can spawn. |
| `server` | Start a CARLA server. `CARLA_UE4_ROOT` and `CARLA_UE5_ROOT` give the install location when `--root` is omitted. |
| `presets` | List the weather, rig and quality presets. |
| `render` | Render one scenario, a suite or everything, and write videos, a manifest and ground truth. |
| `export` | Convert the ground truth of a finished run to COCO or YOLO format. |

```text
--8<-- "_generated/cli/bikeped_sim.txt"
```

See [CARLA](../simulation/carla/index.md) and [render output](../simulation/carla/output.md).

## Calibration

### calibration/calibrate_fisheye.py

Fits the equidistant lens model to checkerboard observations by bundle adjustment and writes `camera_calibration.json`. One of `--images` or `--video` is required.

```bash
python calibration/calibrate_fisheye.py --images calibration_frames/*.jpg --fit-distortion
```

```text
--8<-- "_generated/cli/calibrate_fisheye.txt"
```

### calibration/capture_calibration.py

Opens a camera, shows a preview and saves frames for the calibration.

```bash
python calibration/capture_calibration.py --camera 0 --outdir calibration_frames
```

```text
--8<-- "_generated/cli/capture_calibration.txt"
```

### calibration/compare_fisheye_models.py

Calibrates the same frames with the equidistant, equisolid, stereographic and orthographic models and reports the RMS reprojection error of each.

```bash
python calibration/compare_fisheye_models.py --images calibration_frames/frame_*.png
```

```text
--8<-- "_generated/cli/compare_fisheye_models.txt"
```

## Tools

### tools/eval_models.py

Evaluates YOLO checkpoints on a dataset and caches per-class AP and AR in `eval_cache.json`, which the testbench reads through `--eval-cache`.

```bash
python tools/eval_models.py --models yolo11x.pt --data cocoaug.yaml
python decision_testbench.py --compare --eval-cache eval_cache.json
```

```text
--8<-- "_generated/cli/eval_models.txt"
```

### tools/latency_report.py

Reads the per-frame latency array written by the bridge and prints statistics per pipeline step. The script reads its arguments by hand: the first argument that does not start with a dash is the path, and `--csv` is the only flag.

| Argument | Default | Meaning |
|---|---|---|
| `PATH` | `latency_data.npy` | Latency array to load. |
| `--csv` | off | Also write the statistics to a `.csv` file next to the input. |

```bash
python tools/latency_report.py latency_data.npy --csv
```

### tools/carla_find_crosswalks.py

Connects to a running CARLA server and lists the crosswalks of the current map. `crosswalks.json` in the repository root was exported with it.

| Flag | Default | Meaning |
|---|---|---|
| `--host` | `localhost` | Server address. |
| `--port` | `2000` | Server port. |
| `--map NAME` | current map | Load this map first. |
| `--goto N` | none | Move the spectator to crosswalk `N`. |
| `--interactive` | off | Mark world points interactively. |
| `--export PATH` | none | Write the crosswalk data to a JSON file. |

```bash
python tools/carla_find_crosswalks.py --map Town10HD_Opt --export crosswalks.json
```

### tools/build_docs.py

Builds this site from `docs_src/` into `docs/` and copies the browser simulator next to it.

| Flag | Meaning |
|---|---|
| `--serve` | Serve a live preview instead of writing the output directory. |
| `--open` | Build, then open the result in the default browser. |
| `--strict` | Treat warnings as errors. |
| `--out DIR` | Write the site to `DIR` instead of `docs/`. |

```bash
python tools/build_docs.py --strict --out /tmp/bikeped-site
```

## Bridge starter kit

### bridge_starter/bridge.py

The live system of the starter kit. Settings live in the USER CONFIG block at the top of the file, and three flags override them for a single run. Run it from inside `bridge_starter/`.

| Flag | Meaning |
|---|---|
| `--video PATH` | Read this video file instead of the webcam. |
| `--no-mqtt` | Disable MQTT publishing. |
| `--no-window` | Run without a display window. |

```bash
cd bridge_starter
python bridge.py --video clip.mp4 --no-mqtt
```

See [the starter kit](../deployment/index.md).
