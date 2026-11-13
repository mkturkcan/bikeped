# File formats

This page documents the data files the repo reads and writes: scenarios and suites, the scenario catalog, the output of a CARLA render, and the files produced by the calibration, evaluation and latency tools.

## Scenario

A scenario is a Python dict, or the equivalent JSON object. The testbench reads four keys and ignores the rest.

```python
{
    'name': 'Fast Approach',
    'expected_dominant': 'alert',
    'vehicle': 'ebike',
    'agents': [
        {'type': 'person', 'id': 1,
         'path': [{'x': 5, 'y': -4, 't': 0}, {'x': 5, 'y': 4, 't': 6}]},
        {'type': 'bike', 'id': 2,
         'path': [{'x': 25, 'y': 0.5, 't': 2}, {'x': 1, 'y': -0.1, 't': 4}]},
    ],
}
```

| Key | Type | Required | Meaning |
|---|---|---|---|
| `name` | string | yes | Unique display name. |
| `expected_dominant` | `idle`, `safe`, `warning`, `alert` | yes | Expected dominant state: the most severe state (alert > warning > safe > idle) that the pipeline holds for at least 0.3 s in total over the run. |
| `vehicle` | `bicycle`, `ebike`, `aashto` | no | Braking profile. Defaults to `bicycle`. |
| `agents` | list | yes | Pedestrians and cyclists. |
| `agents[].type` | `person`, `bike` | yes | Agent class seen by the decision pipeline. |
| `agents[].id` | integer | yes | Track id, unique within the scenario. |
| `agents[].path` | list | yes | Waypoints in time order. |
| `agents[].path[].x` | metres | yes | Position along the road. |
| `agents[].path[].y` | metres | yes | Position across the road. |
| `agents[].path[].t` | seconds | yes | Time of the waypoint. Positions between waypoints are linear in time. |

The crosswalk centre is at `(5, 0)`. The frame is drawn in [Scenarios](../scenarios/index.md#coordinate-frame) and the path rules are listed in [Authoring](../scenarios/authoring.md#path-rules).

### Metadata keys

`bikeped.scenarios`, the CARLA renderer and the docs read these. The testbench ignores them.

| Key | Type | Meaning |
|---|---|---|
| `id` | string | Slug, unique across all suites. |
| `suite` | string | Suite name. |
| `family` | string | Family key. |
| `description` | string | One or two plain sentences. |
| `tags` | list of strings | Free-form labels. |
| `render` | object | Optional rendering hints, such as `{'weather': 'wet_night'}`. |
| `agents[].model` | string | Optional appearance hint for the renderer. |
| `agents[].label` | string | Optional short label. |

## Suite file

A suite file is a JSON file written by `python -m bikeped.scenarios generate --out suite.json`. The scenarios inside it use the format above. See [Generators](../scenarios/generators.md).

## Scenario catalog

The catalog is the machine-readable index of every registered scenario. `python -m bikeped.scenarios export --out docs_src/assets/scenarios` writes it as `catalog.json`, next to one SVG thumbnail per scenario named `<id>.svg`. The catalog tables in [Conformance suite](../scenarios/conformance.md) and [Extended suites](../scenarios/extended.md) are written by `tools/build_docs.py` from the registry itself and use the same thumbnails.

## Render output

One run of `python -m bikeped.sim render` writes one directory. Its name is `<engine>_<scenario-id>_<weather>`.

```text
renders/<run_id>/
  manifest.json                 run metadata, per-frame records, camera models
  video/<view>.mp4              clean view, H.264
  video/<view>_overlay.mp4      view with the HUD
  video/composite.mp4           chosen layout
  frames/<view>/000000.jpg      only with --save-frames
  gt/<view>.jsonl               one line per frame
  stills/                       first frame, first alert frame, closest approach
```

The directory is described from the user's side in [CARLA output](../simulation/carla/output.md).

### manifest.json

The manifest holds the run metadata, the camera model of every view and one record per frame. The exporter in `bikeped/sim/export.py` reads these top-level keys:

| Key | Meaning |
|---|---|
| `run_id` | Name of the run. Falls back to the directory name when absent. |
| `run` | Run metadata, described below. |
| `views` | Names of the rendered views. |
| `cameras` | Camera model per view name. |
| `videos` | Relative path of the clean video per view name. The default is `video/<view>.mp4`. |
| `gt` | Relative path of the ground-truth file per view name. The default is `gt/<view>.jsonl`. |
| `frames_dir`, `frame_ext` | Relative directory of the saved frames per view name and their file extension, present when frames were saved. |

Run metadata (`RunInfo` in `bikeped/sim/types.py`):

| Field | Type | Meaning |
|---|---|---|
| `scenario_name`, `scenario_id`, `suite` | string | Which scenario was rendered. |
| `fps` | integer | Simulation and video frame rate. Default 30. |
| `duration_s` | float | Scenario duration in seconds. |
| `engine` | `ue4`, `ue5` | Unreal generation of the CARLA server. |
| `server_version`, `client_version` | string | CARLA versions reported at connect time. |
| `map` | string | CARLA map name. |
| `crosswalk_id` | integer or null | Entry of `crosswalks.json` the scenario was placed on. |
| `weather`, `quality`, `drive`, `ambient` | string | Presets and modes the run used. |
| `seed` | integer | Random seed. |
| `views` | list of strings | Names of the rendered views. |
| `site` | object | Site geometry derived from the crosswalk. |
| `pipeline_config` | string | Key of `PIPELINE_CONFIGS` used for the decision states. |
| `extra` | object | Anything else the runner recorded. |

Frame record (`FrameRecord.to_dict`):

| Field | Type | Meaning |
|---|---|---|
| `index` | integer | Frame number. The first frame is 0. |
| `t` | float | Scenario time in seconds, rounded to four decimals. |
| `state` | string | Decision state of the frame: `idle`, `safe`, `warning` or `alert`. |
| `states` | object | Decision state per named pipeline configuration. |
| `alert_pair` | `[bike_id, ped_id]` or null | The pair that triggered the alert. |
| `danger` | object or null | Ground-truth danger assessment of the frame. |
| `agents` | list | One agent record per agent present in the frame. |

Images and auxiliary depth or instance buffers are not written to the manifest.

Agent record (`AgentState`):

| Field | Type | Meaning |
|---|---|---|
| `id` | integer | Agent id from the scenario. |
| `kind` | `person`, `bike` | Agent class. |
| `sx`, `sy` | metres | Scripted position in the scenario frame. The decision pipeline and the danger model work on these values. |
| `heading` | radians | Scripted heading in the scenario frame. |
| `speed` | m/s | Scripted speed. |
| `location` | `[x, y, z]` metres | Centre of the actor's 3D box in the CARLA world frame. |
| `rotation` | `[pitch, yaw, roll]` degrees | Orientation in the CARLA world frame. |
| `extent` | `[forward, right, up]` metres | Half sizes of the box. |
| `role` | `scripted`, `ambient` | Whether the agent comes from the scenario or from ambient traffic. |
| `model` | string | Appearance hint from the scenario. |
| `blueprint` | string | CARLA blueprint that was spawned. |
| `track_error_m` | metres | Ground-plane distance between the scripted and the simulated position. |

Camera model (`CameraModel`):

| Field | Type | Meaning |
|---|---|---|
| `name` | string | View name. |
| `kind` | `pinhole`, `fisheye` | Projection. |
| `width`, `height` | pixels | Image size. |
| `position` | `[x, y, z]` metres | Camera position in the world frame. |
| `forward`, `right`, `up` | unit vectors | Camera basis in the world frame. Image `u` grows with `right`, image `v` grows against `up`. |
| `fov_deg` | degrees | Horizontal field of view for a pinhole view, full field of view of the image circle for a fisheye view. |
| `cx`, `cy` | pixels or null | Principal point. Null means the image centre. |
| `f_px_per_rad` | pixels per radian | Fisheye focal length. |
| `k1`, `k2` | number | Fisheye polynomial terms: `r = f (θ + k1 θ³ + k2 θ⁵)`. |
| `circle_radius_px` | pixels | Radius of the fisheye image circle. |

### gt/&lt;view&gt;.jsonl

The file has one JSON object per line and one line per frame. Line `k` belongs to frame `k` of the clean video of the same view.

| Field | Meaning |
|---|---|
| `index` | Frame number, the same as in the manifest. |
| `t` | Scenario time in seconds. |
| `width`, `height` | Image size in pixels that the boxes refer to. When absent, the size of the view's camera model applies. |
| `objects` | One entry per agent, with the fields below. |

| Object field | Meaning |
|---|---|
| `id`, `kind`, `role` | As in the agent record. |
| `bbox` | Axis-aligned 2D box `[x1, y1, x2, y2]` in pixels, origin at the top-left corner. |
| `corners_image` | The eight projected corners of the 3D box. |
| `distance_m` | Distance from the camera in metres. |
| `truncation` | Fraction of the box that falls outside the image. |
| `visible_fraction` | Fraction of the agent that is not occluded, available when depth or instance buffers were rendered. |

To turn one view into a detection dataset, run `python -m bikeped.sim export RUN_DIR --format coco|yolo`. The exported classes are `person` and `bicycle`; a scenario agent of type `bike` becomes `bicycle`. See [Ground truth](../simulation/carla/ground-truth.md).

## Tool output files

| File | Written by | Read by | Contents |
|---|---|---|---|
| `camera_calibration.json` | `calibration/calibrate_fisheye.py` | testbench, visualizer, renderer | Calibrated fisheye intrinsics. Every key is listed in [Configuration](config.md). |
| `eval_cache.json` | `tools/eval_models.py` | `decision_testbench.py --eval-cache` | One entry per model name. Under `merged`, the classes `pedestrian`, `cyclist` and `vehicle` each carry `ap5095` and `recall`, and optionally `per_size` and `ar_curve`. |
| `latency_data.npy` | the bridge, when latency profiling is on | `tools/latency_report.py` | A 2D array with one row per processed frame and ten columns: `frame_acquire`, `preprocess`, `yolo_preprocess`, `yolo_inference`, `yolo_postprocess`, `bev_tracking`, `decision`, `viz_submit`, `total`, `pipeline_only`. |

How the detector statistics enter the testbench is covered in [Uncertainty models](../system/uncertainty.md), and the latency columns in [Latency](../experiments/latency.md).
