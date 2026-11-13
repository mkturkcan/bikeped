# Perception geometry

A detection becomes a ground position by sending the bottom-centre pixel of its
bounding box back through the fisheye lens model onto the ground plane. This
page covers the lens model, the sensor crop, the back-projection, and the
localisation error that the bounding-box shortcut introduces. The code is
`FisheyeCamera` and `compute_bbox_error` in
[`decision_testbench.py`](https://github.com/mkturkcan/bikeped/blob/main/decision_testbench.py).

<figure class="diagram">
--8<-- "assets/figures/fisheye-geometry.svg"
<figcaption>Left: for a level camera and a ray in the vertical plane of the optical axis, the ray at angle θ meets the ground at d = h / tan θ. Right: the same ray lands at radius r = f θ from the principal point; only rows 670 to 2829 of the crop are on the sensor.</figcaption>
</figure>

## Lens model

The deployed lens follows the equidistant model. The angle θ between a ray and
the optical axis maps to an image radius

```text
r = f * theta                               # k1 = k2 = 0
r = f * (theta + k1*theta^3 + k2*theta^5)   # with radial correction
```

`FisheyeCamera` derives the focal length from the field of view and the image
circle radius, `f = 2R * 180 / (fov_deg * pi)`, which gives 1013.3 px for
R = 1750 px and a 197.9° field of view. `config.yaml` names four projection
models; the bridge's `fisheye_pixel_to_ground` handles all of them, and the
testbench implements the equidistant one with the optional `k1`, `k2` terms.

| `camera.projection` | Model | Formula |
|---|---|---|
| `linear` | equidistant | `r = f * theta` |
| `equalarea` | equisolid | `r = 2f * sin(theta/2)` |
| `stereographic` | stereographic | `r = 2f * tan(theta/2)` |
| `orthographic` | orthographic | `r = f * sin(theta)` |

The intrinsics come from `camera_calibration.json`, written by the
[calibration tools](../calibration/index.md):

| Key | Value | Unit |
|---|---|---|
| `cx`, `cy` | 1752.69, 1804.54 | px in the 3500 × 3500 crop |
| `focal_length` | 1013.31 | px per radian |
| `fov_deg` | 197.9 | degrees |
| `radius_px` | 1750.0 | px |
| `k1`, `k2` | 0.0, 0.0 | |
| `rms_error_px` | 4.6334 | px, over 1470 points in 42 frames |

## Sensor crop

The lens draws a 3500 px image circle, and the sensor is 3840 × 2160, so the
top and bottom of the circle fall off the sensor. Preprocessing pads the frame
to 3840 × 3840 and centre-crops 3500 × 3500. In that crop the sensor covers
rows 670 to 2829 and every column. `pixel_on_sensor` encodes this band, and a
projected point counts as visible only if it is inside the field of view and
inside the band.

## Camera pose in the testbench

| Field | Value | Meaning |
|---|---|---|
| `cam_x`, `cam_y` | 5.0, −13.0 | Position in the scenario frame, in metres. The crosswalk centre is at (5, 0). |
| `height` | 3.6576 | Metres, read from `camera.height` (12 feet). |
| `heading_deg` | 90 | Chosen automatically: 90° when `cam_y < 0`, 270° otherwise. |
| `pitch_deg` | 0 | Level optical axis, read from `camera.pitch_deg`. Negative values tilt down. |

The camera frame is x right, y down, z along the optical axis. All positions
handed to the decision pipeline are relative to the camera.

## Forward and inverse projection

`world_to_pixel` rotates a world point into the camera frame, takes
`theta = arccos(z / |d|)` and `phi = atan2(y, x)`, applies the lens model and
returns `u = cx + r cos(phi)`, `v = cy + r sin(phi)`. A point is valid when
`theta` is below half the field of view and it lies in front of the camera
plane.

`pixel_to_ground` inverts this. It converts the pixel radius to `theta` (in
closed form, or with six Newton iterations when `k1` or `k2` is non-zero),
builds the camera-frame ray `(sin θ cos φ, sin θ sin φ, cos θ)`, rotates it
into the world frame and intersects it with the plane `z = 0`. Rays that do not
point downward return NaN and `valid = False`.

## Bounding-box localisation error

A detector returns a box, and the system takes the bottom-centre of that box as
the point where the object touches the ground. Under fisheye distortion that
pixel is not the projection of the true ground point. `compute_bbox_error`
measures the difference:

1. Build the eight corners of the object's 3D box at ground position
   `(gx, gy)` with a given heading (`object_3d_corners`).
2. Project the corners and keep the ones on the sensor.
3. Take the tightest axis-aligned box around them and its bottom-centre pixel.
4. Back-project that pixel to the ground. The error is the distance between
   this estimate and `(gx, gy)`.

The object sizes are in `OBJECT_TYPES`:

| Type | Length (m) | Width (m) | Height (m) | COCO class | AP | AR |
|---|---|---|---|---|---|---|
| `pedestrian` | 0.3 | 0.5 | 1.7 | 0 | 0.678 | 0.755 |
| `cyclist` | 1.8 | 0.6 | 1.7 | 1 | 0.634 | 0.740 |
| `car` | 4.5 | 1.8 | 1.5 | 2 | 0.550 | 0.629 |
| `motorcycle` | 2.2 | 0.8 | 1.4 | 3 | 0.680 | 0.754 |
| `bus` | 12.0 | 2.5 | 3.0 | 5 | 0.820 | 0.860 |
| `truck` | 7.0 | 2.5 | 2.8 | 7 | 0.746 | 0.806 |

`--error-map` prints the error at fixed distances and bearings from the camera
for the pedestrian, cyclist and car types:

```console
$ python decision_testbench.py --error-map
Camera: pos=(5.0,-13.0), h=3.6576m, heading=90.0deg, pitch=0.0deg, FOV=197.9deg, R=1750px
...
  PEDESTRIAN (0.3m x 0.5m x 1.7m)
  Heading = 0deg from camera axis
  Dist (m)   Angle       gx       gy   Err (m)   Rel Err      BBox WxH  Corners
       3.0      0deg     5.00   -10.00    0.2479     8.26%        96x389        8
       3.0     80deg     2.05   -12.48    2.1337    71.12%       162x128        4
      10.0      0deg     5.00    -3.00    0.2493     2.49%        31x172        8
      10.0     80deg    -4.85   -11.26    0.3449     3.45%       108x242        8
      25.0      0deg     5.00    12.00    0.2497     1.00%         12x70        8
...
  CYCLIST (1.8m x 0.6m x 1.7m)
  Heading = 0deg from camera axis
      10.0      0deg     5.00    -3.00    0.2730     2.73%       185x175        8
      10.0     30deg     0.00    -4.34    0.5676     5.68%       199x199        8
      20.0     30deg    -5.00     4.32    0.5832     2.92%         98x97        8
```

Straight ahead of the camera a pedestrian is misplaced by about 0.25 m at every
distance from 3 m to 25 m, so the relative error falls from 8.26% to 1.00%.
The error grows toward the edge of the field of view, and it jumps where the
sensor band clips the box: at 3 m and 80° only four of the eight corners are on
the sensor and the estimate is 2.13 m off. A cyclist is longer than a
pedestrian, so the error depends more on bearing and heading, reaching 0.57 m
at 10 m and 30°.

The error is a function of position, heading and object size, with no random
part. When the testbench runs with the error model on, which is the default,
it replaces each agent's true position with the estimate before tracking.
`compute_error_grid` evaluates the same function on a 0.5 m grid out to 25 m
for the error-map figure. The [uncertainty](uncertainty.md) page shows how the
results change with the model off.

## Related

- [Calibration](../calibration/index.md) for producing `camera_calibration.json`
- [Configuration reference](../reference/config.md) for every `camera` key
- [Camera placement](../experiments/placement.md) for the height and pitch sweep
