# Configuration reference

Two files hold the parameters of a site: `config.yaml` for runtime settings and `camera_calibration.json` for the calibrated lens. This page lists every key of both, its shipped value, and which code reads it.

## Who reads what

| Reader | From `config.yaml` | From `camera_calibration.json` |
|---|---|---|
| `decision_testbench.py` | `camera.height`, `camera.pitch_deg` | `fov_deg`, `radius_px`, `frame_width`, `frame_height`, `pad_size`, `cx`, `cy`, `k1`, `k2` |
| `sim_visualizer.py --errors` | `camera.height`, `camera.pitch_deg` | `fov_deg`, `radius_px`, `frame_width`, `frame_height`, `pad_size`, `cx`, `cy` |
| `generate_figures.py` | | loads the file through `load_camera_calibration()` |
| `calibration/calibrate_fisheye.py` | | writes the file |
| `bikeped/sim/fisheye.py` (CARLA fisheye view) | `camera.fov`, `camera.radius`, `camera.cx`, `camera.cy`, `camera.k1`, `camera.k2`, `camera.pitch_deg` | `fov_deg`, `radius_px`, `cx`, `cy`, `k1`, `k2`, which override the `config.yaml` values |

No other code in this repository opens `config.yaml`. The remaining keys configure the live bridge of the reference deployment (`mqtt_bridge.py`), which is not included here. The [starter kit](../deployment/index.md) does not read `config.yaml` either. It holds the same settings as constants in its `USER CONFIG` block, and the tables below name the matching constant where there is one.

Two consequences are worth knowing before you edit the file:

- Changing the `decision` block does not change testbench results. The testbench takes its parameters from `DecisionParams` and `PIPELINE_CONFIGS` in `decision_testbench.py`, whose `default` entry carries the same values as the shipped `decision` block.
- If `config.yaml` is missing, the testbench uses a camera height of 3.048 m and a pitch of 0. If `camera_calibration.json` is missing, it prints `No camera_calibration.json found, using defaults` and uses the defaults of `FisheyeCamera`.

## config.yaml

### video

| Key | Default | Unit | Meaning |
|---|---|---|---|
| `use_webcam` | `true` | | Read the live camera instead of a file. Kit: `USE_WEBCAM`. |
| `file_path` | `"06022025_19.avi"` | | Video file used when `use_webcam` is false. Kit: `VIDEO_FILE`. |
| `process_every_k_frames` | `1` | frames | Frame subsampling factor of the live bridge. The kit processes every frame. |

### preprocessing

| Key | Default | Unit | Meaning |
|---|---|---|---|
| `pad_size` | `3840` | px | The frame is centred on a black square of this side. Kit: `INPUT_PAD_SIZE`. |
| `crop_size` | `3500` | px | Side of the centre crop taken from the padded square. All intrinsics refer to this cropped frame. Kit: `INPUT_CROP_SIZE`. |

### camera

| Key | Default | Unit | Meaning |
|---|---|---|---|
| `height` | `3.6576` | m | Mounting height above the ground (12 ft). Read by the testbench. Kit: `CAMERA_HEIGHT_M`. |
| `fov` | `197.90` | deg | Field of view across the image circle. Kit: `CAMERA_FOV_DEG`. |
| `radius` | `1750` | px | Radius of the image circle. Kit: `CAMERA_RADIUS_PX`. |
| `pitch_deg` | `0` | deg | Tilt of the optical axis. Negative looks down. Read by the testbench. Kit: `CAMERA_PITCH_DEG`. |
| `cx`, `cy` | `1752.69`, `1804.54` | px | Optical centre in the cropped frame. Kit: `CAMERA_CX`, `CAMERA_CY`. |
| `projection` | `"linear"` | | Projection model, see [projection names](#projection-names). Kit: `CAMERA_PROJECTION`. |
| `k1`, `k2` | `0.0`, `0.0` | | Radial terms of `r = f (θ + k1 θ³ + k2 θ⁵)`, used with `linear` only. Set them from a calibration run with `--fit-distortion`. Kit: `CAMERA_K1`, `CAMERA_K2`. |
| `latency_frames` | `0` | frames | Frames of measured camera latency to compensate by first-order kinematic prediction. `0` disables it. At 30 FPS one frame is 33 ms, so the value is the measured latency in milliseconds divided by 33. |

The intrinsics appear twice, here and in `camera_calibration.json`. The testbench reads them from the JSON file. After a calibration run, `calibrate_fisheye.py` prints the `camera.fov` and `camera.radius` values to copy into this block, and `camera.cx` and `camera.cy` when the optical centre is more than 5 px from the frame centre.

### detection

| Key | Default | Unit | Meaning |
|---|---|---|---|
| `model_path` | `"yolo11x.engine"` | | Detector weights or TensorRT engine. The kit downloads `yolo11xxl.pt` instead, see `YOLO_HF_FILE`. |
| `confidence` | `0.1` | | Detection confidence threshold. Kit: `YOLO_CONF`. |
| `input_size` | `1280` | px | Inference size. Kit: `YOLO_IMGSZ`. |
| `valid_classes` | `[0, 1, 2, 3, 5, 7, 9, 31, 36, 37, 16]` | | COCO class IDs kept. The code in this repository names 0 person, 1 bicycle, 2 car, 3 motorcycle, 5 bus, 7 truck, 9 traffic light and 16 dog. Kit: `YOLO_CLASSES`, which keeps the first six. |
| `tracker` | `"bytetrack.yaml"` | | Tracker configuration of the live bridge. The kit tracks on the ground plane with its own `GroundPlaneTracker`. |

### decision

| Key | Default | Unit | Meaning |
|---|---|---|---|
| `bike_memory_frames` | `58` | frames | Stage 2 stays open while a cyclist was seen within this many frames. Kit: `DEC_BIKE_MEMORY_FRAMES`. |
| `prox_min` | `1.9` | m | Lower bound on the bike to pedestrian distance that can alert. Kit: `DEC_PROX_MIN_M`. |
| `prox_max` | `24.8` | m | Upper bound on that distance. Kit: `DEC_PROX_MAX_M`. |
| `prox_speed_min` | `0.147` | m | Minimum cyclist displacement over the lookback for an alert. Kit: `DEC_PROX_SPEED_MIN`. |
| `lookback_frames` | `2` | frames | Lookback `k` of the closing check. Kit: `DEC_LOOKBACK_FRAMES`. |
| `max_range` | `25` | m | Agents farther than this from the camera are skipped in stage 3. Kit: `DEC_MAX_RANGE_M`. |
| `max_object_age` | `10` | frames | Listed in the file and in `DecisionParams`. `DecisionPipeline` has no such argument and no code in this repository reads the key. |
| `speed_adaptive` | `true` | | Extend the upper bound to the cyclist's stopping distance when that is larger. Kit: `DEC_SPEED_ADAPTIVE`. |
| `cyclist_reaction_s` | `0.84` | s | Perception-reaction time in the stopping distance. Kit: `DEC_REACTION_S`. |
| `cyclist_decel_ms2` | `1.96` | m/s² | Deceleration in the stopping distance. Kit: `DEC_DECEL_MS2`. |

The two braking values are the 85th-percentile field measurements of Martin and Bigazzi (2025) cited in `decision_pipeline.py`. [The decision pipeline](../system/decision.md) describes how each parameter enters the three stages.

The constructor defaults of `DecisionPipeline` are a different set (`bike_memory_frames=77`, `prox_min=0.5`, `prox_max=20.7`, `prox_speed_min=0.098`, `lookback_frames=1`). The testbench and the kit pass every parameter explicitly, so those defaults only apply if you construct the class yourself without arguments.

### tracking

| Key | Default | Unit | Meaning |
|---|---|---|---|
| `history_frames` | `5` | frames | Read by the live bridge only. The file gives no description. |
| `min_track_points` | `4` | points | Read by the live bridge only. The file gives no description. |
| `speed_window` | `4` | samples | Read by the live bridge only. `DecisionPipeline` estimates cyclist speed over a fixed window of 4 samples. |

### feedback

| Key | Default | Unit | Meaning |
|---|---|---|---|
| `blinking` | `true` | | Read by the live bridge only. The file gives no description. |
| `go_signal` | `true` | | Read by the live bridge only. The file gives no description. |
| `sound_threshold` | `60` | | Read by the live bridge only. The kit throttles sounds with `SOUND_THROTTLE_FRAMES`. |

### mqtt

| Key | Default | Unit | Meaning |
|---|---|---|---|
| `enabled` | `true` | | Publish track messages. Kit: `MQTT_ENABLED`. |
| `broker` | `"test.mosquitto.org"` | | Broker host name. Kit: `MQTT_BROKER`. |
| `port` | `1883` | | Broker port. Kit: `MQTT_PORT`. |
| `topic` | `"bikeped/intersection/tracks"` | | Topic. Kit: `MQTT_TOPIC`. |
| `username`, `password` | `null`, `null` | | Credentials. Kit: `MQTT_USERNAME`, `MQTT_PASSWORD`. |
| `client_id` | `"bikeped_tracker"` | | Client identifier. Kit: `MQTT_CLIENT_ID`. |
| `qos` | `0` | | Quality of service. Kit: `MQTT_QOS`. |
| `publish_interval` | `1` | | Read by the live bridge only. The kit publishes every frame. |

### latency

| Key | Default | Unit | Meaning |
|---|---|---|---|
| `enabled` | `false` | | Collect per-frame latency data. Kit: `LATENCY_ENABLED`, which defaults to `True`. |
| `save_path` | `"latency_data.npy"` | | Output array. Kit: `LATENCY_SAVE_PATH`. |
| `save_interval_frames` | `500` | frames | Checkpoint period. Kit: `LATENCY_SAVE_INTERVAL`. |
| `warmup_frames` | `30` | frames | Frames discarded at start while the model warms up. Kit: `LATENCY_WARMUP_FRAMES`. |

### Top level

| Key | Default | Unit | Meaning |
|---|---|---|---|
| `debug` | `false` | | Debug switch of the live bridge. Kit: `DEBUG`. |
| `verbose` | `false` | | Verbose logging. Kit: `VERBOSE`. |

## camera_calibration.json

Written by [`calibrate_fisheye.py`](../calibration/index.md). The shipped file is the calibration of the reference camera.

```json title="camera_calibration.json"
{
  "model": "equidistant",
  "cx": 1752.69,
  "cy": 1804.54,
  "focal_length": 1013.31,
  "fov_deg": 197.9,
  "radius_px": 1750.0,
  "k1": 0.0,
  "k2": 0.0,
  "rms_error_px": 4.6334,
  "num_frames": 42,
  "num_points": 1470,
  "frame_width": 3500,
  "frame_height": 3500,
  "pad_size": 3840,
  "crop_size": 3500
}
```

| Key | Shipped value | Unit | Meaning |
|---|---|---|---|
| `model` | `"equidistant"` | | Projection model that was fitted. `calibrate_fisheye.py` always writes `equidistant`. |
| `cx`, `cy` | `1752.69`, `1804.54` | px | Optical centre in the cropped frame. |
| `focal_length` | `1013.31` | px/rad | Equidistant focal length `f` in `r = f θ`. |
| `fov_deg` | `197.9` | deg | Field of view derived from the focal length: `2 · radius_px · 180 / (f · π)`. |
| `radius_px` | `1750.0` | px | Half the side of the cropped frame. |
| `k1`, `k2` | `0.0`, `0.0` | | Radial correction terms. Zero unless the run used `--fit-distortion`. |
| `rms_error_px` | `4.6334` | px | RMS reprojection error of the fit. |
| `num_frames` | `42` | frames | Checkerboard frames used. |
| `num_points` | `1470` | corners | Corner observations used. |
| `frame_width`, `frame_height` | `3500`, `3500` | px | Size of the cropped frame the intrinsics refer to. |
| `pad_size`, `crop_size` | `3840`, `3500` | px | Preprocessing the calibration frames went through. |

`FisheyeCamera` in the testbench does not read `focal_length`. It computes the focal length from `fov_deg` and `radius_px`, which gives the same 1013.3 px for the shipped file. The quality fields (`rms_error_px`, `num_frames`, `num_points`) and `model` are a record of the calibration run and are not used at runtime here.

The calibration script also writes a sibling `camera_calibration.yaml` with three blocks: `calibration` (the values above without the frame and preprocessing sizes), `preprocessing` (`pad_size`, `crop_size`) and `config_yaml_updates` (the `camera.fov` and `camera.radius` values for `config.yaml`).

## Projection names

`camera.projection` and the kit's `CAMERA_PROJECTION` use the names in the first column. The calibration tools use the model names in the second.

| Config name | Model | Formula |
|---|---|---|
| `linear` | equidistant | `r = f · θ` |
| `equalarea` | equisolid | `r = 2f · sin(θ / 2)` |
| `stereographic` | stereographic | `r = 2f · tan(θ / 2)` |
| `orthographic` | orthographic | `r = f · sin(θ)` |

`θ` is the angle between the incoming ray and the optical axis, and `r` is the distance from the optical centre in pixels. `calibrate_fisheye.py` fits the equidistant model only. `compare_fisheye_models.py` fits all four on the same frames so you can check which one your lens follows.

## Related

- [Fisheye calibration](../calibration/index.md)
- [Bridge starter kit](../deployment/index.md)
- [Fisheye model and ground-plane localisation](../system/perception.md)
