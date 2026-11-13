# Fisheye calibration

Ground positions come from one pixel per detection, so the lens model has to match the camera. This page covers capturing checkerboard frames, fitting the equidistant model with `calibrate_fisheye.py` and comparing projection models. It ends with the files and scripts that read the result.

The three scripts live in `calibration/`. The commands below are run from that folder, and output paths are relative to the current directory.

## The shipped calibration

`camera_calibration.json` at the repository root is the calibration of the reference camera.

| Quantity | Value |
|---|---|
| Model | equidistant, `r = f · θ` |
| Optical centre | (1752.69, 1804.54) px in the 3500 × 3500 frame |
| Focal length | 1013.31 px |
| Field of view | 197.9° |
| Image circle radius | 1750 px |
| `k1`, `k2` | 0, 0 |
| RMS reprojection error | 4.6334 px |
| Data | 42 frames, 1470 corners |

[Configuration reference](../reference/config.md#camera_calibrationjson) describes every key of the file.

## Why a custom solver

OpenCV's fisheye calibration fits the Kannala-Brandt model with polynomial distortion coefficients. The module docstring of `calibrate_fisheye.py` notes that this does not fit a 200° spherical lens on a 16:9 sensor, so the script fits the equidistant projection directly with its own bundle adjustment. The parameters are the optical centre `cx`, `cy`, the focal length `f` and, on request, the correction terms `k1` and `k2` in

```text
r = f · (θ + k1·θ³ + k2·θ⁵)
```

## 1. Capture frames

```bash
python calibration/capture_calibration.py
python calibration/capture_calibration.py --camera 0 --outdir my_cal_frames
```

The script opens the camera at 3840 × 2160 (V4L2 with MJPG on Linux) and shows a preview. Press ++r++ to start recording one frame per second and ++r++ again to stop. ++q++ or ++esc++ quits. Frames are written as `frame_0000.png`, `frame_0001.png` and so on, and numbering continues after frames that are already in the folder.

| Flag | Default | Meaning |
|---|---|---|
| `--camera` | `0` | Camera index. |
| `--outdir` | `calibration_frames` | Output directory. |
| `--display-width` | `1200` | Width of the preview window in pixels. |

Move the checkerboard through the whole image circle while recording, including the edge. The lens model matters most there. Calibration needs at least 3 frames with a detected board.

## 2. Calibrate

```bash
python calibration/calibrate_fisheye.py --images "calibration_frames/*.png"
python calibration/calibrate_fisheye.py --video calibration.avi --interval 30
python calibration/calibrate_fisheye.py --images "calibration_frames/*.png" --fit-distortion
python calibration/calibrate_fisheye.py --video calibration.avi --interactive
```

One of `--images` or `--video` is required.

| Flag | Default | Meaning |
|---|---|---|
| `--images` | | Image paths or glob patterns showing the checkerboard. |
| `--video` | | Video file with checkerboard frames. |
| `--interactive` | off | Step through the video and pick frames by hand. Requires `--video`. |
| `--board-width` | `9` | Inner corners along the board width. |
| `--board-height` | `6` | Inner corners along the board height. |
| `--square-size` | `0.025` | Square size in metres. |
| `--pad-size` | `3840` | Preprocessing pad size in pixels. |
| `--crop-size` | `3500` | Preprocessing crop size in pixels. |
| `--interval` | `30` | Frame sampling interval in video mode. |
| `--max-frames` | `50` | Maximum number of frames taken from a video. |
| `--initial-fov` | `200` | Initial field-of-view guess in degrees. |
| `--fit-distortion` | off | Also fit `k1` and `k2`. |
| `--output` | `camera_calibration.json` | Output JSON. A `.yaml` with the same stem is written next to it. |
| `--visualize` | off | Show the reprojection of each frame. ++q++ stops the review. |
| `--save-viz` | | Directory for reprojection images. |

In interactive mode the keys are ++s++ to keep the current frame (only when a board is detected), ++space++ for the next frame, ++d++ and ++a++ to jump 10 frames forward or back, and ++q++ to finish.

### What the script does

1. Preprocessing. Each frame is centred on a `pad-size` black square and the centre `crop-size` square is cut out, the same pad and crop the live system applies. With the defaults a 3840 × 2160 frame becomes 3500 × 3500, and every intrinsic refers to that cropped frame.
2. Corner detection. `findChessboardCorners` needs straight lines between corners, which a fisheye image does not have. The script remaps the fisheye frame to a 1500 px perspective view with a 160° field of view using the initial focal-length guess, detects the board there, maps the corners back to fisheye pixels and refines them with `cornerSubPix` on the original image. The remap only has to be good enough to straighten the lines.
3. Initial poses. Each frame's board pose is initialised with PnP and refined on its own.
4. Bundle adjustment. All board poses and the shared intrinsics are optimised together in two stages: Levenberg-Marquardt with a linear loss, then a bounded trust-region solve with a `soft_l1` loss to limit the influence of outlier corners.
5. Output. The field of view is derived from the fitted focal length as `2 · radius · 180 / (f · π)` with `radius = crop-size / 2`.

The report lists the optical centre and its shift from the frame centre, the focal length, the derived field of view, the RMS reprojection error and a per-frame RMS. Frames are tagged `OK` below 2 px, `HIGH` below 5 px and `BAD` above. Drop the `BAD` frames and run again if a few of them dominate the error.

### Output files

| File | Content |
|---|---|
| `camera_calibration.json` | Flat intrinsics plus `frame_width`, `frame_height`, `pad_size`, `crop_size`. Read by the testbench, the visualizer and the figure script. |
| `camera_calibration.yaml` | The same calibration in a `calibration` block, the `preprocessing` block, and `config_yaml_updates` with the `camera.fov` and `camera.radius` values for `config.yaml`. |

The script ends by printing the lines to copy into `config.yaml`. It adds `camera.cx` and `camera.cy` when the optical centre is more than 5 px away from the frame centre.

## 3. Compare projection models

`calibrate_fisheye.py` assumes an equidistant lens. To check that assumption for another lens, fit all four supported models on the same frames:

```bash
python calibration/compare_fisheye_models.py --images "calibration_frames/frame_*.png"
```

| Flag | Default | Meaning |
|---|---|---|
| `--images` | required | Image paths or glob patterns. |
| `--board-width` | `7` | Inner corners along the board width. |
| `--board-height` | `5` | Inner corners along the board height. |
| `--square-size` | `0.025` | Square size in metres. |
| `--pad-size` | `3840` | Preprocessing pad size in pixels. |
| `--crop-size` | `3500` | Preprocessing crop size in pixels. |
| `--initial-fov` | `200` | Initial field-of-view guess in degrees. |
| `--output` | `model_comparison.json` | Output JSON with one entry per model. |

The board defaults differ from `calibrate_fisheye.py` (7 × 5 here, 9 × 6 there), so pass `--board-width` and `--board-height` explicitly when you use both scripts on one set of frames.

The script prints one row per model with the RMS error, optical centre, focal length and field of view, then names the model with the lowest RMS error. The models are equidistant, equisolid, stereographic and orthographic. [Projection names](../reference/config.md#projection-names) maps them to the names `config.yaml` uses.

## 4. Use the result

| Consumer | What to update |
|---|---|
| Testbench, visualizer, figures | Nothing. They load `camera_calibration.json` from the repository root. Copy your output file there. |
| `config.yaml` | `camera.fov`, `camera.radius`, `camera.cx`, `camera.cy`, and `camera.k1`, `camera.k2` after `--fit-distortion`. Also set `camera.height` and `camera.pitch_deg` for your mount. The testbench reads those two from this file. |
| Starter kit | The `CAMERA_*` constants in the `USER CONFIG` block of `bridge_starter/bridge.py`. |

After replacing the calibration, print the localisation error of the new geometry:

```bash
python decision_testbench.py --error-map
```

The first line of the output confirms which intrinsics were loaded:

```text
Loaded calibration: FOV=197.90 cx=1752.7 cy=1804.5 f=1013.3 R=1750 frame=3500x3500 pad=3840 pitch=0 k1=0 k2=0
```

## Related

- [Fisheye model and ground-plane localisation](../system/perception.md)
- [Configuration reference](../reference/config.md)
- [Bridge starter kit](../deployment/index.md)
- API: [calibrate_fisheye](../api/calibrate_fisheye.md), [capture_calibration](../api/capture_calibration.md), [compare_fisheye_models](../api/compare_fisheye_models.md)
