# Ground truth

Every rendered frame comes with the state of each agent, its box in each view, and the danger label of the safety model. This page lists those records and shows how to export them for detector training.

## Per-frame records

The runner builds one `FrameRecord` per tick. It holds the frame index, the time `t` in seconds, the decision state, the alerting pair if there is one, the danger assessment, and one `AgentState` per agent.

| `AgentState` field | Unit | Meaning |
|---|---|---|
| `id` | | Agent id from the scenario |
| `kind` | | `person` or `bike` |
| `sx`, `sy` | m | Scripted position in the scenario frame |
| `heading` | | Scripted heading in the scenario frame |
| `speed` | m/s | Scripted ground speed |
| `location` | m | Centre of the simulated actor's 3D box, CARLA world frame |
| `rotation` | degrees | `(pitch, yaw, roll)` of the box |
| `extent` | m | Half sizes of the box along the actor's forward, right and up axes |
| `role` | | `scripted` or `ambient` |
| `model` | | Rendering hint from the scenario |
| `blueprint` | | CARLA blueprint id that was spawned |
| `track_error_m` | m | Ground-plane distance between scripted and simulated position |

## Boxes per view

`project_box(agent, camera)` projects the eight corners of an agent's oriented 3D box into a view and returns `None` when the agent is outside it. `frame_ground_truth(frame, view)` does this for every agent of a frame.

| Field | Meaning |
|---|---|
| `id`, `kind`, `role` | Copied from the agent |
| `bbox` | `[x1, y1, x2, y2]` in pixels, the 2D box around the projected 3D box |
| `corners_image` | The eight projected corners, an 8 × 2 array |
| `distance_m` | Distance from the camera to the agent |
| `truncation` | Fraction of the box that falls outside the image |
| `visible_fraction` | Fraction of the agent that is not occluded, when the view has a depth or instance buffer |

Pinhole boxes are clipped against the near plane edge by edge. An agent that straddles the camera plane keeps a correct box, where dropping the corners behind the camera would shrink it. Fisheye boxes go through the lens model, so they follow the distortion of the image.

```python
from bikeped.sim.groundtruth import frame_ground_truth

for obj in frame_ground_truth(frame, "pole"):
    print(obj["id"], obj["kind"], obj["bbox"], round(obj["distance_m"], 1))
```

The same records are written to `gt/<view>.jsonl`, one JSON object per frame:

```json
{"index": 0, "t": 0.0, "width": 1920, "height": 1080, "objects": [{"id": 1, "kind": "person", "role": "scripted", "bbox": [...], "corners_image": [...], "distance_m": ..., "truncation": ..., "visible_fraction": ...}]}
```

`visible_fraction` is `null` when the view has no depth or instance buffer. The runner attaches those buffers to the `pole` view only, when the server offers depth and instance cameras, so the other views carry boxes without a visibility estimate.

## Danger label

`FrameRecord.danger` is the result of `decision_testbench.compute_ground_truth_danger(scenario, t)`, the kinematic safety model that labels every number in the paper. It is evaluated on the scripted paths.

| Field | Unit | Meaning |
|---|---|---|
| `is_danger` | | The cyclist is on a collision course and cannot stop in time, or time to collision is low |
| `actionable` | | `is_danger`, and a warning can still change the outcome |
| `severity` | 0 to 1 | Grows with the square of the cyclist's speed |
| `ttc` | s | Time to collision, infinite when there is none |
| `d_min_future` | m | Minimum future distance within the lookahead |
| `closing_speed` | m/s | Positive when approaching |
| `bike_speed` | m/s | Cyclist ground speed |
| `stopping_dist` | m | Distance the cyclist needs to stop |
| `warning_budget` | s | Warning time left before closest approach |
| `distance` | m | Current distance of the pair |

[Safety model](../../system/safety-model.md) explains how these are derived.

## Export to COCO or YOLO

```bash
python -m bikeped.sim export renders/ue4_fast-approach_clear_noon --format coco
python -m bikeped.sim export renders/ue4_fast-approach_clear_noon --format yolo --view fisheye
```

| Flag | Default | Meaning |
|---|---|---|
| `RUN_DIR` | | A finished run directory |
| `--format` | | `coco` or `yolo` |
| `--view` | `pole` | View whose boxes and frames are exported |
| `--out` | inside `RUN_DIR` | Annotation file (COCO) or dataset folder (YOLO) |

The exporters turn one view of a run into a detection dataset with the classes `person` and `bicycle`. They read `manifest.json` and `gt/<view>.jsonl`, so they need no server.

| | COCO | YOLO |
|---|---|---|
| Output | One annotation file and an `images` folder next to it | A folder with `images/`, `labels/` and `data.yaml` |
| Box | `[x, y, width, height]` in pixels, origin at the top-left corner | `class cx cy w h`, centre and size divided by the image width and height |
| Classes | `person` is id 1, `bicycle` is id 2 | `person` is 0, `bicycle` is 1 |

Images are copied from the saved frames when the run has them. Without saved frames they are decoded from the clean video of the view, whose frame `k` belongs to line `k` of the ground-truth file. YOLO file names carry the run id, so several runs can be exported into one folder, and a frame without boxes gets an empty label file and counts as a negative.

The Python functions take filters that the defaults leave off:

```python
from bikeped.sim.export import export_coco, export_yolo

run = "renders/ue4_fast-approach_clear_noon"
export_coco(run, "dataset/annotations.json", view="pole", min_visible=0.3, every=5)
export_yolo(run, "dataset_yolo", view="fisheye", min_area=64.0)
```

| Argument | Default | Meaning |
|---|---|---|
| `min_visible` | `0.0` | Drop boxes whose `visible_fraction` is known and below this value. Boxes without a visibility estimate are kept |
| `min_area` | `0.0` | Drop boxes smaller than this area in square pixels |
| `every` | `1` | Export every n-th frame |

## Related

- [Run output](output.md)
- [Cameras](cameras.md)
- [File formats](../../reference/formats.md)
