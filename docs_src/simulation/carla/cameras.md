# Cameras

A render can record several views of the same run. This page lists the views, explains how the fisheye view is built from cube faces so that it matches the deployed lens, and documents the camera model that ties pixels to world coordinates.

## Views

Select views with `--rig`, a comma-separated list. Each view is written to `video/<view>.mp4` in the run directory.

| View | Kind | What it shows |
|---|---|---|
| `pole` | pinhole | Camera on the pole at the testbench camera position (12 ft), aimed so the crosswalk and every scripted path fit in the frame |
| `fisheye` | fisheye | The deployed lens on the pole, rendered as cube faces and resampled through the calibrated model |
| `chase` | pinhole | Follows the first cyclist (or pedestrian, if there is no cyclist) from behind and above |
| `orbit` | pinhole | Slow eye-level arc around the conflict point with a longer lens |
| `street` | pinhole | Static eye-level tripod on the kerb |
| `bev` | pinhole | Straight down from 60 m with a narrow lens, image up along scenario `+y` |

`pole`, `fisheye`, `street` and `bev` are static. `chase` and `orbit` move their sensor before every tick. The default is `--rig pole`. The schematic top-down panel used by the `split` and `quad` layouts is drawn separately from the scenario coordinates and is not a CARLA view.

```bash
python -m bikeped.sim render --scenario fast-approach --rig pole,fisheye,bev
```

## The fisheye view

The calibrated field of view is 197.9°, so the image circle reaches 98.95° from the optical axis. The deployed camera is level (`camera.pitch_deg` is 0) and faces across the road, so the image circle covers more than a hemisphere in front of it. A CARLA RGB camera is a pinhole and cannot cover that, so the renderer captures 90° cube faces from the pole position and resamples them through the lens model.

<figure class="diagram">
--8<-- "assets/figures/cube-remap.svg"
<figcaption>Cube faces rendered by CARLA are resampled into the frame the deployed lens would produce.</figcaption>
</figure>

1. `deployed_fisheye_model(position, yaw_deg)` builds a `CameraModel` of kind `fisheye` from `camera_calibration.json` and `config.yaml`: an equidistant projection with the `k1` and `k2` terms and `f = radius / (fov / 2)` (the focal length the testbench uses), a 3500 px image circle on a 3840 × 2160 sensor, and the pitch from `camera.pitch_deg`. A testbench heading `h` corresponds to CARLA yaw `-h`, so the testbench default heading of 90° is `yaw_deg=-90`.
2. `FisheyeRemap(model, face_size=1024, supersample=1)` works out which cube faces the image circle needs. For the deployed lens `remap.faces` is `('down', 'front', 'right', 'left', 'up')`; the face behind the camera receives no pixels and is not rendered.
3. `remap.face_camera(face)` returns a 90° pinhole `CameraModel` for a face, and the rig spawns one CARLA camera from each.
4. On every tick `remap.remap(face_images)` returns the BGR fisheye frame. Pixels outside the image circle are black.

The lens parameters come from the calibration file that the deployed system reads:

| Parameter | Value | Unit | Source |
|---|---|---|---|
| Projection | equidistant, `r = f (θ + k1 θ³ + k2 θ⁵)` | | `camera_calibration.json` |
| `focal_length` | 1013.31 | px per radian | `camera_calibration.json` |
| `fov_deg` | 197.9 | degrees | `camera_calibration.json` |
| `radius_px` | 1750 | px | `camera_calibration.json` |
| `k1`, `k2` | 0.0, 0.0 | | `camera_calibration.json` |
| Camera height | 3.6576 | m | `config.yaml`, `camera.height` |

The fisheye `CameraModel` has to agree with `decision_testbench.FisheyeCamera.world_to_pixel` to sub-pixel accuracy. That agreement is what lets a rendered fisheye frame stand in for a frame from the deployed camera: a ground point lands on the same pixel in both.

## Camera model

Every view is described by a `CameraModel` (`bikeped/sim/types.py`). The models of a run are stored in its manifest, so boxes can be re-projected later without a server.

| Field | Default | Meaning |
|---|---|---|
| `name` | | View name, for example `pole` |
| `kind` | | `pinhole` or `fisheye` |
| `width`, `height` | | Image size in pixels |
| `position` | | Camera position in the CARLA world frame, metres |
| `forward`, `right`, `up` | | Orthonormal basis vectors in the world frame |
| `fov_deg` | `90.0` | Pinhole: horizontal field of view. Fisheye: full field of view of the image circle |
| `cx`, `cy` | image centre | Principal point in pixels |
| `f_px_per_rad` | | Fisheye focal length in pixels per radian |
| `k1`, `k2` | `0.0` | Odd polynomial terms of the fisheye model |
| `circle_radius_px` | | Radius of the fisheye image circle |

A world point `P` has camera coordinates `(f, r, u) = (forward · d, right · d, up · d)` with `d = P - position`. Image `u` grows with `r` and image `v` grows with `-u`. For a pinhole view the focal length in pixels is `width / (2 tan(fov / 2))`. The world frame is CARLA's: left-handed, `x` forward, `y` right, `z` up, with rotations given as `(pitch, yaw, roll)` in degrees.

`bikeped.sim.projection` converts between the two descriptions and projects points:

```python
from bikeped.sim.projection import carla_rotation_to_basis, project_points

forward, right, up = carla_rotation_to_basis(pitch=-10.0, yaw=90.0, roll=0.0)
uv, valid, depth = project_points(points_world, camera)   # (N, 2), (N,) bool, (N,)
```

## Layout and overlay

Each view is saved twice: clean as `video/<view>.mp4`, and with the heads-up display as `video/<view>_overlay.mp4`. `--no-overlay` skips the second file. `--layout` chooses how the views and the top-down panel are arranged in `video/composite.mp4`:

| Layout | Composite |
|---|---|
| `single` | The first view alone |
| `split` | The first view on the left and the top-down panel in a square on the right |
| `quad` | A 2 × 2 grid with up to three views and the top-down panel in the last cell. Without the panel, a fourth view takes that cell |

The first view is the primary one. Views are scaled to fit their cell and centred on the background. They are never stretched, so a fisheye frame keeps its aspect ratio next to a 16:9 view.

## Quality presets

`--quality` takes `draft`, `standard` (the default) or `cinematic` and trades render time for image quality:

| Preset | Pinhole views | Fisheye cube face | Fisheye output | Post-processing |
|---|---|---|---|---|
| `draft` | 960 × 540 | 512 px | 0.25 × the 3840 × 2160 sensor | Motion blur, bloom and lens flare off |
| `standard` | 1920 × 1080 | 1024 px | 0.5 × sensor (1920 × 1080) | Light motion blur and bloom |
| `cinematic` | 2560 × 1440 | 1536 px | 0.5 × sensor | Stronger motion blur and bloom, lens flare, chromatic aberration |

The values are `QUALITY_PRESETS` in `bikeped/sim/rigs.py`; `python -m bikeped.sim presets` prints the sizes. The fisheye view renders one camera per cube face, so it is the view that gains most from `draft` while a setup is being checked.

## Related

- [Ground truth](ground-truth.md)
- [Run output](output.md)
- [Perception model](../../system/perception.md)
