# Run output

Each render writes one directory. This page describes its files and the structure of `manifest.json`.

## Directory layout

```text
renders/<run_id>/
  manifest.json                 run metadata, camera models, per-frame records
  video/<view>.mp4              clean view, H.264
  video/<view>_overlay.mp4      view with the heads-up display
  video/composite.mp4           the layout chosen with --layout
  frames/<view>/000000.jpg      individual frames, only with --save-frames
  gt/<view>.jsonl               one line per frame: boxes per agent
  stills/                       first frame, first alert frame, closest approach
```

The run id is `<engine>_<scenario-id>_<weather>`, for example `ue4_fast-approach_clear_noon`. `--out` changes the parent directory, which defaults to `renders`.

## Videos

| File | Content |
|---|---|
| `video/<view>.mp4` | The view as rendered, without any drawing on top. Use it as detector input |
| `video/<view>_overlay.mp4` | The same frames with the heads-up display: decision state, agent boxes and the danger indicator. Not written with `--no-overlay` |
| `video/composite.mp4` | The views arranged as `single`, `split` or `quad` |

Videos are encoded at the frame rate given by `--fps`, with `ffmpeg` (libx264, yuv420p) through a pipe. When no `ffmpeg` executable is found, the recorder falls back to OpenCV's `mp4v` codec and logs a warning; those files play in most players but are not H.264.

## Frames

`--save-frames jpg` or `--save-frames png` additionally writes every frame of every view to `frames/<view>/`, numbered from `000000`. The default is `none`. The COCO and YOLO exporters copy saved frames when they exist and otherwise decode the frames from the clean video.

## Stills

`stills/` holds a few frames picked from the run: the first frame, the first frame in which the pipeline is in <span class="state state-alert">alert</span>, and the frame of closest approach between a cyclist and a pedestrian.

## manifest.json

The manifest is one JSON object with these top-level keys:

| Key | Contents |
|---|---|
| `run_id` | `<engine>_<scenario-id>_<weather>` |
| `run` | Run metadata (`RunInfo`, below) |
| `scenario` | The scenario dict that was rendered, with its metadata |
| `layout`, `overlay`, `views` | Composite layout, whether HUD videos were written, rendered views |
| `cameras` | The `CameraModel` of each view as a dict, see [Cameras](cameras.md#camera-model) |
| `videos` | Video files by name, relative to the run directory |
| `frames_dir`, `frame_ext` | Folder and extension of saved frames, or `null` without `--save-frames` |
| `gt` | Ground-truth file per view, `gt/<view>.jsonl` |
| `stills` | Still images, relative to the run directory |
| `timeline` | Per-run summary, below |
| `complete`, `error` | `true` and `null` when the run finished; otherwise the error message |
| `frame_count`, `frames` | Number of frames and one record per frame |

Run metadata is a `RunInfo`:

| Field | Default | Meaning |
|---|---|---|
| `scenario_name` | | Display name of the scenario |
| `scenario_id` | | Scenario slug |
| `suite` | `conformance` | Suite the scenario belongs to |
| `fps` | `30` | Frames per second |
| `duration_s` | `0.0` | Scenario duration in seconds |
| `engine` | `ue4` | `ue4` or `ue5` |
| `server_version`, `client_version` | | CARLA versions of the server and the Python client |
| `map` | | CARLA map name |
| `crosswalk_id` | | Entry of `crosswalks.json` the scenario was placed on |
| `weather` | `clear_noon` | Weather preset |
| `quality` | `standard` | Quality preset |
| `drive` | | Drive mode; the CLI default is `kinematic` |
| `ambient` | `none` | Ambient traffic level |
| `seed` | `0` | Random seed |
| `views` | | List of rendered views |
| `site` | | Geometry of the site: how the scenario frame maps to the world |
| `pipeline_config` | `default` | Key of `decision_testbench.PIPELINE_CONFIGS` used for the decision state |
| `extra` | | Further options, for example `cyclist_proxy` (`null` or `"walker"`) |

Each frame record has these keys:

| Key | Meaning |
|---|---|
| `index` | Frame number, from 0 |
| `t` | Scenario time in seconds, rounded to four decimals |
| `state` | Decision state of the configured pipeline: `idle`, `safe`, `warning` or `alert` |
| `states` | Decision state per pipeline variant, keyed by name |
| `alert_pair` | `[bike_id, ped_id]` of the pair that raised the alert, or `null` |
| `danger` | Ground-truth danger assessment, see [Ground truth](ground-truth.md#danger-label) |
| `agents` | List of agent states, see [Ground truth](ground-truth.md#per-frame-records) |

Images and auxiliary buffers (depth, instance ids) are kept in memory during the run and are not written to the manifest.

The `timeline` summary is computed from the frame records when the run closes:

| Key | Meaning |
|---|---|
| `frames` | Number of frames |
| `state_counts` | Frames per decision state |
| `alert_intervals`, `warning_intervals` | Time intervals in which the pipeline was in that state |
| `danger_intervals`, `actionable_intervals` | Time intervals labelled dangerous, and those in which a warning is still actionable |
| `first_alert_t`, `first_danger_t` | Time of the first alert and of the first dangerous frame |
| `lead_time_s` | `first_danger_t - first_alert_t` |
| `closest_approach` | Frame, time, bike and person ids and distance in metres of the closest bike-pedestrian pair |

```python
import json

manifest = json.load(open("renders/ue4_fast-approach_clear_noon/manifest.json"))
```

## Return value of the recorder

Code that drives `Recorder` directly gets the same information from `close()`: a dictionary with `run_dir`, `manifest`, `videos`, `stills` and the number of `frames` written.

## Related

- [Ground truth](ground-truth.md)
- [File formats](../../reference/formats.md)
- [Dataset](../../dataset.md)
