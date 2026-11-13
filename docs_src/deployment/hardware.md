# Hardware and I/O

This page lists what the repository says about the camera, the compute unit, the indicator light, sound, MQTT and latency logging, and the interfaces you implement to connect your own hardware. Where the repository does not specify a part, the page says so.

## Camera

| Property | Value | Source |
|---|---|---|
| Lens | Fisheye, 197.9° field of view, equidistant projection | `camera_calibration.json` |
| Capture resolution | 3840 × 2160 | `calibration/capture_calibration.py` |
| Image circle | 3500 px diameter (radius 1750 px) | `config.yaml`, `camera_calibration.json` |
| Mounting height | 3.6576 m (12 ft) | `config.yaml` `camera.height` |
| Pitch | 0°, optical axis level | `config.yaml` `camera.pitch_deg` |
| Capture backend on Linux | V4L2 with the MJPG pixel format | `calibration/capture_calibration.py` |

The image circle is taller than the 16:9 sensor, so its top and bottom are cut off. The software pads each frame to 3840 × 3840 and takes the centre 3500 × 3500 crop. In that cropped frame the sensor covers rows 670 to 2829, and the testbench treats a projected point outside those rows as not visible.

The camera and lens make and model are not specified in this repository. Any fisheye works as long as you [calibrate](../calibration/index.md) it and one of the four [projection models](../reference/config.md#projection-names) describes it.

## Compute

The README describes the system as a 30 FPS response system designed for Jetson Orin devices, and the docstring of `decision_pipeline.py` names the Jetson AGX Orin as the target, with no heavy imports and no allocations in the hot path. The Jetson module variant, power supply, enclosure and networking are not specified here.

Detector settings of the reference deployment, from the `detection` block of `config.yaml`:

| Setting | Value |
|---|---|
| Model | `yolo11x.engine`, a TensorRT engine |
| Input size | 1280 px |
| Confidence threshold | 0.1 |
| Classes kept | `[0, 1, 2, 3, 5, 7, 9, 31, 36, 37, 16]` (COCO IDs) |
| Tracker | `bytetrack.yaml` |

The starter kit differs in three places. It downloads `yolo11xxl.pt` from `mehmetkeremturkcan/YOLO11xxl` on Hugging Face, keeps classes `[0, 1, 2, 3, 5, 7]`, and tracks on the ground plane with its own nearest-neighbour tracker.

On first run the kit exports the weights to a TensorRT engine with half precision at the 1280 px input size on GPU 0. When the export fails it keeps running on the `.pt` weights and prints a warning that inference will be 2 to 5 times slower. Set `YOLO_DEVICE = "cpu"` to run without CUDA.

The live system's extra dependencies are listed in `requirements.txt` as TensorRT, ultralytics, paho-mqtt, busylight and play_sounds.

## Indicator light

The reference deployment drove a USB busylight. The kit ships `SimulatedLight`, which prints each colour change to stdout, so the loop runs without any light attached.

| State | Colour | RGB |
|---|---|---|
| <span class="state state-idle">idle</span> | Blue | `(0, 0, 255)` |
| <span class="state state-safe">safe</span> | Green | `(0, 255, 0)` |
| <span class="state state-warning">warning</span> | Orange | `(255, 165, 0)` |
| <span class="state state-alert">alert</span> | Red | `(255, 0, 0)` |

To drive real hardware, replace the module-level `light` object in `bridge.py` with a class that has one method:

```python
class MyHardwareLight:
    def set_color(self, rgb):  # rgb is a 3-tuple in 0..255
        ...
    on = set_color             # busylight-compatible alias


light = MyHardwareLight()
```

`apply_feedback` calls `light.set_color` on every frame with the colour of the current state, at the frame rate of the loop. `SimulatedLight` filters repeats itself. A hardware class should do the same if writing to the device is slow.

## Sound

Audio is optional. With the `play-sounds` package installed, the kit plays `ALERT_SOUND` when the state is `alert` and `SAFE_SOUND` when it is `safe`. Playback does not block the loop. Each sound is throttled to one play per `SOUND_THROTTLE_FRAMES` frames, 60 by default, which is 2 s at 30 FPS.

If the package is missing, the kit prints one notice at start-up and runs silently. If a WAV file is missing, it prints one warning for that file and continues.

## MQTT

The kit publishes one JSON message per frame with the confirmed tracks in ground coordinates. [Bridge starter kit](index.md#mqtt-message) documents the fields. Connection settings are the `MQTT_*` constants. If the `paho-mqtt` package is not installed or the broker cannot be reached, the kit prints the reason and continues without publishing.

## Latency logging

With `LATENCY_ENABLED` the kit records nine timings in milliseconds for every processed frame after the first `LATENCY_WARMUP_FRAMES` frames. It saves them to `LATENCY_SAVE_PATH` every `LATENCY_SAVE_INTERVAL` records and once more at exit.

| Column | Content |
|---|---|
| 0 | Frame acquisition (`cap.read`) |
| 1 | Preprocessing (the kit does none, so this is close to zero) |
| 2, 3, 4 | Detector preprocess, inference and postprocess, as reported by ultralytics |
| 5 | Ground lookup and tracking |
| 6 | Decision pipeline |
| 7 | Feedback, MQTT publish and display |
| 8 | Total for the frame |

`tools/latency_report.py` reads the ten-column layout of the deployed bridge, which has a `pipeline_only` column after the total. It rejects the kit's nine-column file:

```text
ValueError: Expected 10 columns, got shape (50, 9). The .npy file may have been recorded with an older format.
```

Append the tenth column first. The report describes it as the pipeline without acquisition, preprocessing and visualisation, which is the sum of columns 2 to 6:

```python
import numpy as np

d = np.load("latency_data.npy")          # (N, 9) from bridge.py
pipeline_only = d[:, 2:7].sum(axis=1)    # detector, tracking, decision
np.save("latency_data_10col.npy", np.column_stack([d, pipeline_only]))
```

```bash
python tools/latency_report.py latency_data_10col.npy
python tools/latency_report.py latency_data_10col.npy --csv
```

The report prints mean, median, standard deviation, minimum, maximum, P95 and P99 per step, each step's share of the total, and the frame rate the pipeline-only time would support. `--csv` also writes the rows to a CSV file next to the input.

## Related

- [Bridge starter kit](index.md)
- [Tuning for a new site](tuning.md)
- [Latency experiments](../experiments/latency.md)
- [Configuration reference](../reference/config.md)
