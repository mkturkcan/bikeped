"""
Decision Testbench -- Python implementation with fisheye BBox error model and parameter optimizer.

Reimplements the index.html decision simulator in Python, adds rigorous 3D bounding box
projection error analysis under fisheye distortion, and optimizes decision parameters
against scripted scenarios using differential evolution.

Usage:
    python decision_testbench.py                  # run all analyses
    python decision_testbench.py --optimize        # optimize decision parameters
    python decision_testbench.py --error-map       # generate BBox error heatmap
"""

import numpy as np
from dataclasses import dataclass, field
from typing import List, Dict, Tuple, Optional
from scipy.optimize import differential_evolution
from collections import deque
import argparse
import json
from decision_pipeline import DecisionPipeline, stopping_distance as dp_stopping_distance

# ================================================================
# 1. FISHEYE CAMERA MODEL
# ================================================================

def load_camera_calibration(path='camera_calibration.json'):
    """Load calibrated intrinsics from JSON produced by calibrate_fisheye.py."""
    try:
        import json
        with open(path) as f:
            return json.load(f)
    except FileNotFoundError:
        return None


@dataclass
class FisheyeCamera:
    """Equidistant fisheye camera model with forward and inverse projection.

    Camera position matches the pole preset from index.html:
      (x=5, y=-13, h=5.5), looking across the crosswalk.

    Intrinsics (fov_deg, cx, cy) default to values from camera_calibration.json
    if present, otherwise fall back to manual estimates.

    pitch_deg: downward tilt in degrees (negative = looking down).
              If None, auto-computed from geometry to aim at ground
              at distance |cam_y| from the camera.
    heading_deg: horizontal direction (degrees). If None, auto-computed
              from cam_y: 90 deg if cam_y < 0, 270 deg if cam_y > 0.
    """
    height: float = 3.6576  # 12 feet (deployment height)
    fov_deg: float = 197.90
    radius_px: int = 1750
    frame_width: int = 3500       # fisheye crop (full circle)
    frame_height: int = 3500
    kappa: float = 1.0
    cam_x: float = 5.0
    cam_y: float = -13.0
    pad_size: int = 3840          # padded frame before crop
    yolo_imgsz: int = 1280        # YOLO inference resolution
    sensor_width: int = 3840      # physical sensor resolution
    sensor_height: int = 2160     # physical sensor resolution
    pitch_deg: Optional[float] = 0.0  # default: level (matches config.yaml); None = auto-aim at ground
    heading_deg: Optional[float] = None
    cx_override: Optional[float] = None
    cy_override: Optional[float] = None
    # Optional Kannala-Brandt-style radial distortion on top of the equidistant model:
    #   r = f * (theta + k1*theta^3 + k2*theta^5)
    # When both are 0 (default) this reduces to pure equidistant and the inverse
    # is closed-form; otherwise pixel_to_ground uses Newton iteration.
    k1: float = 0.0
    k2: float = 0.0

    def __post_init__(self):
        # Use calibrated optical center if provided, else frame center
        if self.cx_override is not None:
            self.xcenter = self.cx_override
        else:
            self.xcenter = (self.frame_width - 1) / 2.0
        if self.cy_override is not None:
            self.ycenter = self.cy_override
        else:
            self.ycenter = (self.frame_height - 1) / 2.0
        D = 2.0 * self.radius_px
        self.focal_length = D * 180.0 / (self.fov_deg * np.pi)

        # Sensor visibility band in the crop frame.
        # Physical model: 3500x3500 fisheye -> pad to 3840x3840 -> sensor captures 3840x2160
        # In the crop, the sensor-visible rows are offset by the padding and crop.
        y_pad = (self.pad_size - self.sensor_height) // 2   # padding above sensor in padded frame
        crop_start = (self.pad_size - self.frame_height) // 2  # crop offset in padded frame
        self.sensor_v_min = y_pad - crop_start               # first visible row in crop
        self.sensor_v_max = y_pad + self.sensor_height - crop_start  # last+1 visible row
        # Horizontal: sensor is wider than crop, so all columns are visible
        x_pad = (self.pad_size - self.sensor_width) // 2
        self.sensor_u_min = x_pad - crop_start
        self.sensor_u_max = x_pad + self.sensor_width - crop_start

        # Auto-compute heading from camera side
        if self.heading_deg is None:
            self.heading_deg = 90.0 if self.cam_y < 0 else 270.0
        # Auto-compute pitch to aim at ground at |cam_y| (used by height/pitch sweep)
        if self.pitch_deg is None:
            self.pitch_deg = -np.rad2deg(np.arctan2(self.height, abs(self.cam_y)))

        # Cache pitch trig for pixel_to_ground
        pitch_rad_simple = np.deg2rad(self.pitch_deg)
        self._cos_pitch = np.cos(pitch_rad_simple)
        self._sin_pitch = np.sin(pitch_rad_simple)

        # Build rotation matrix once (camera frame convention:
        # x_cam=right, y_cam=down, z_cam=forward/optical axis)
        heading_rad = np.deg2rad(self.heading_deg)
        pitch_rad = np.deg2rad(self.pitch_deg)
        ch, sh = np.cos(heading_rad), np.sin(heading_rad)
        cp, sp = np.cos(pitch_rad), np.sin(pitch_rad)

        right = np.array([sh, -ch, 0.0])
        forward = np.array([ch * cp, sh * cp, sp])
        down = np.cross(right, forward)
        if down[2] > 0:
            down = -down

        self._R = np.array([right, down, forward])
        self._cam_pos = np.array([self.cam_x, self.cam_y, self.height])

    def pixel_on_sensor(self, u: float, v: float) -> bool:
        """Check whether a pixel in the 3500x3500 crop is within the physical sensor region."""
        return (self.sensor_u_min <= u < self.sensor_u_max and
                self.sensor_v_min <= v < self.sensor_v_max)

    def dist_to(self, gx: float, gy: float) -> float:
        """3D distance from camera to an object at ground position (gx, gy, 0)."""
        dx = gx - self.cam_x
        dy = gy - self.cam_y
        return np.sqrt(dx*dx + dy*dy + self.height*self.height)

    def world_to_pixel(self, points: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Project 3D world points to pixel coordinates.

        Args:
            points: (..., 3) world coordinates [x, y, z].
        Returns:
            pixels: (..., 2) pixel coordinates [u, v].
            valid:  (...) boolean.
        """
        pts = np.asarray(points, dtype=np.float64)
        d_world = pts - self._cam_pos

        if d_world.ndim == 1:
            d_cam = self._R @ d_world
        else:
            orig_shape = d_world.shape
            flat = d_world.reshape(-1, 3)
            d_cam = (self._R @ flat.T).T
            d_cam = d_cam.reshape(orig_shape)

        z_cam = d_cam[..., 2]
        norm = np.linalg.norm(d_cam, axis=-1)
        norm = np.maximum(norm, 1e-12)
        theta = np.arccos(np.clip(z_cam / norm, -1.0, 1.0))
        phi = np.arctan2(d_cam[..., 1], d_cam[..., 0])

        if self.k1 != 0.0 or self.k2 != 0.0:
            theta2 = theta * theta
            r = self.focal_length * theta * (1.0 + theta2 * (self.k1 + self.k2 * theta2))
        else:
            r = self.focal_length * theta
        u = self.xcenter + r * np.cos(phi)
        v = self.ycenter + r * np.sin(phi)

        half_fov = np.deg2rad(self.fov_deg / 2.0)
        valid = (theta < half_fov) & (z_cam > 0.01)

        pixels = np.stack([u, v], axis=-1)
        return pixels, valid

    def projected_bbox_area(self, gx: float, gy: float,
                            obj: 'ObjectType', heading_rad: float = 0.0) -> float:
        """Compute the bounding box pixel area by projecting 8 3D corners.

        Replicates the physical model: the fisheye lens projects onto
        the 3500x3500 crop, but the sensor only captures 3840x2160,
        clipping the top and bottom of the circle. A corner is visible
        only if it is within the fisheye FOV AND within the sensor band.

        Returns 0.0 if no corner is visible on the sensor.
        """
        corners = object_3d_corners(gx, gy, obj, heading_rad)
        pixels, valid = self.world_to_pixel(corners)
        # Additionally mask by sensor visibility
        for i in range(len(valid)):
            if valid[i]:
                valid[i] = self.pixel_on_sensor(pixels[i, 0], pixels[i, 1])
        if not np.any(valid):
            return 0.0
        vis = pixels[valid]
        u_min, v_min = vis.min(axis=0)
        u_max, v_max = vis.max(axis=0)
        area_crop = (u_max - u_min) * (v_max - v_min)
        # Scale: YOLO letterboxes pad_size → yolo_imgsz
        linear_scale = self.yolo_imgsz / self.pad_size
        return area_crop * (linear_scale ** 2)

    def pixel_to_ground(self, pixels: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Inverse-project pixel coordinates to the ground plane z = 0.

        Uses the full rotation matrix (heading + pitch), consistent with
        world_to_pixel.  The camera-frame ray is computed from the
        equidistant model and transformed to world coordinates via R^T.

        Args:
            pixels: (..., 2) pixel coordinates [u, v].

        Returns:
            ground: (..., 3) world coordinates [x, y, z=0].
            valid:  (...) boolean mask.
        """
        px = np.asarray(pixels, dtype=np.float64)
        du = px[..., 0] - self.xcenter
        dv = px[..., 1] - self.ycenter
        r = np.sqrt(du**2 + dv**2)
        phi = np.arctan2(dv, du)
        r_norm = r / self.focal_length
        if self.k1 != 0.0 or self.k2 != 0.0:
            # Invert r/f = theta + k1*theta^3 + k2*theta^5 by Newton iteration.
            theta = r_norm.copy()
            for _ in range(6):
                t2 = theta * theta
                f_val = theta * (1.0 + t2 * (self.k1 + self.k2 * t2)) - r_norm
                fp_val = 1.0 + t2 * (3.0 * self.k1 + 5.0 * self.k2 * t2)
                theta = theta - f_val / fp_val
        else:
            theta = r_norm

        # Camera-frame ray direction (x_cam=right, y_cam=down, z_cam=forward)
        sin_t = np.sin(theta)
        cos_t = np.cos(theta)
        ray_cam = np.stack([
            sin_t * np.cos(phi),   # x_cam (right)
            sin_t * np.sin(phi),   # y_cam (down)
            cos_t,                  # z_cam (forward)
        ], axis=-1)

        # Rotate to world frame via R^T (inverse of world_to_pixel's R)
        Rt = self._R.T
        if ray_cam.ndim == 1:
            ray_world = Rt @ ray_cam
        else:
            orig_shape = ray_cam.shape
            flat = ray_cam.reshape(-1, 3)
            ray_world = (Rt @ flat.T).T
            ray_world = ray_world.reshape(orig_shape)

        # Intersect with ground plane z = 0
        ray_z = ray_world[..., 2]
        valid = ray_z < -1e-6

        t = np.where(valid, -self.height / ray_z, 0.0)

        ground = np.zeros(px.shape[:-1] + (3,))
        ground[..., 0] = self.cam_x + t * ray_world[..., 0]
        ground[..., 1] = self.cam_y + t * ray_world[..., 1]
        ground[..., 2] = 0.0
        ground[~valid] = np.nan
        return ground, valid


# ================================================================
# 2. 3D BOUNDING BOX PROJECTION AND ERROR MODEL
# ================================================================

@dataclass
class ObjectType:
    """Physical dimensions and detection characteristics of an object class."""
    name: str
    length: float       # along object forward axis (m)
    width: float        # lateral (m)
    height: float       # vertical (m)
    class_id: int       # COCO class id
    ap: float = 0.5     # average precision (adjustable)
    ar: float = 0.7     # average recall (adjustable)


# Default object types with typical dimensions
# Defaults from fisheye_1280 model eval on fisheye-augmented COCO dataset.
# Per-class AP/AR from eval_cache.json (merged groups).
OBJECT_TYPES = {
    'pedestrian': ObjectType('pedestrian', length=0.3, width=0.5, height=1.7, class_id=0,
                             ap=0.678, ar=0.755),
    'cyclist':    ObjectType('cyclist',    length=1.8, width=0.6, height=1.7, class_id=1,
                             ap=0.634, ar=0.740),
    'car':        ObjectType('car',        length=4.5, width=1.8, height=1.5, class_id=2,
                             ap=0.550, ar=0.629),
    'motorcycle': ObjectType('motorcycle', length=2.2, width=0.8, height=1.4, class_id=3,
                             ap=0.680, ar=0.754),
    'bus':        ObjectType('bus',        length=12.0, width=2.5, height=3.0, class_id=5,
                             ap=0.820, ar=0.860),
    'truck':      ObjectType('truck',      length=7.0, width=2.5, height=2.8, class_id=7,
                             ap=0.746, ar=0.806),
}


def object_3d_corners(gx: float, gy: float, obj: ObjectType,
                      heading_rad: float = 0.0) -> np.ndarray:
    """Compute the 8 corners of a 3D bounding box on the ground plane.

    Args:
        gx, gy: ground-plane position (object center).
        obj: object type with dimensions.
        heading_rad: yaw angle (radians, 0 = facing +x).

    Returns:
        corners: (8, 3) array of 3D world coordinates.
    """
    hl = obj.length / 2.0
    hw = obj.width / 2.0
    # Local corners: [front-left, front-right, back-left, back-right] x [bottom, top]
    local = np.array([
        [ hl,  hw, 0.0],
        [ hl, -hw, 0.0],
        [-hl,  hw, 0.0],
        [-hl, -hw, 0.0],
        [ hl,  hw, obj.height],
        [ hl, -hw, obj.height],
        [-hl,  hw, obj.height],
        [-hl, -hw, obj.height],
    ])
    # Rotate around z-axis by heading
    c, s = np.cos(heading_rad), np.sin(heading_rad)
    rot = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])
    corners = local @ rot.T
    corners[:, 0] += gx
    corners[:, 1] += gy
    return corners


def compute_bbox_error(cam: FisheyeCamera, gx: float, gy: float,
                       obj: ObjectType, heading_rad: float = 0.0) -> Dict:
    """Compute the localization error from using the bottom-center of the
    tightest 2D bounding box around a fisheye-projected 3D object.

    The detector is assumed to find the tightest axis-aligned rectangle
    enclosing all visible projected corners of the 3D bounding box.

    The error is the metric displacement between the true ground position and
    the position obtained by inverse-projecting the bbox bottom-center pixel.

    Returns a dict with error metrics, or None if the object is not visible.
    """
    corners_3d = object_3d_corners(gx, gy, obj, heading_rad)
    pixels, valid = cam.world_to_pixel(corners_3d)

    # Mask by sensor visibility (top/bottom of fisheye circle clipped by sensor)
    for i in range(len(valid)):
        if valid[i]:
            valid[i] = cam.pixel_on_sensor(pixels[i, 0], pixels[i, 1])

    if not np.any(valid):
        return None

    vis_pixels = pixels[valid]
    u_min, v_min = vis_pixels.min(axis=0)
    u_max, v_max = vis_pixels.max(axis=0)

    # Bottom-center of the detected bounding box
    bc_u = (u_min + u_max) / 2.0
    bc_v = v_max  # bottom = max v (image y increases downward)

    # Inverse-project bottom-center to ground
    bc_pixel = np.array([bc_u, bc_v])
    ground_est, gvalid = cam.pixel_to_ground(bc_pixel)

    if not gvalid:
        return None

    est_x, est_y = ground_est[0], ground_est[1]

    # True ground-contact point projected to pixel, then back to ground
    true_contact = np.array([gx, gy, 0.0])
    true_px, true_valid = cam.world_to_pixel(true_contact)

    if true_valid:
        ideal_ground, ivalid = cam.pixel_to_ground(true_px)
        ideal_err_x = ideal_ground[0] - gx if ivalid else np.nan
        ideal_err_y = ideal_ground[1] - gy if ivalid else np.nan
    else:
        ideal_err_x = ideal_err_y = np.nan

    err_x = est_x - gx
    err_y = est_y - gy
    err_dist = np.sqrt(err_x**2 + err_y**2)
    true_dist = np.sqrt((gx - cam.cam_x)**2 + (gy - cam.cam_y)**2)

    # Pixel shift: how far is the bbox bottom-center from the true contact projection
    if true_valid:
        px_shift_u = bc_u - true_px[0]
        px_shift_v = bc_v - true_px[1]
    else:
        px_shift_u = px_shift_v = np.nan

    return {
        'gx': gx, 'gy': gy,
        'true_dist': true_dist,
        'est_x': est_x, 'est_y': est_y,
        'err_x': err_x, 'err_y': err_y,
        'err_dist': err_dist,
        'err_relative': err_dist / max(true_dist, 0.01),
        'bbox_width_px': u_max - u_min,
        'bbox_height_px': v_max - v_min,
        'px_shift_u': px_shift_u,
        'px_shift_v': px_shift_v,
        'ideal_err_x': ideal_err_x,
        'ideal_err_y': ideal_err_y,
        'n_visible_corners': int(np.sum(valid)),
    }


def compute_error_grid(cam: FisheyeCamera, obj: ObjectType,
                       max_range: float = 25.0, grid_step: float = 0.5,
                       heading_rad: float = 0.0) -> Dict:
    """Compute BBox localization error over a grid of positions within max_range.

    The grid is in camera-relative coordinates (forward / lateral).
    World positions are computed from the camera's position and heading.

    Returns arrays for plotting: X, Y grids and error magnitudes.
    """
    fwd_rad = np.deg2rad(cam.heading_deg)
    fwd_x, fwd_y = np.cos(fwd_rad), np.sin(fwd_rad)
    right_x, right_y = -fwd_y, fwd_x

    xs = np.arange(0.5, max_range + grid_step, grid_step)
    ys = np.arange(-max_range, max_range + grid_step, grid_step)
    X, Y = np.meshgrid(xs, ys)

    err_dist = np.full_like(X, np.nan)
    err_x = np.full_like(X, np.nan)
    err_y = np.full_like(X, np.nan)
    err_rel = np.full_like(X, np.nan)

    for i in range(X.shape[0]):
        for j in range(X.shape[1]):
            along, across = X[i, j], Y[i, j]
            if along**2 + across**2 > max_range**2:
                continue
            gx = cam.cam_x + along * fwd_x + across * right_x
            gy = cam.cam_y + along * fwd_y + across * right_y
            result = compute_bbox_error(cam, gx, gy, obj, heading_rad)
            if result is not None:
                err_dist[i, j] = result['err_dist']
                err_x[i, j] = result['err_x']
                err_y[i, j] = result['err_y']
                err_rel[i, j] = result['err_relative']

    return {
        'X': X, 'Y': Y,
        'err_dist': err_dist,
        'err_x': err_x,
        'err_y': err_y,
        'err_relative': err_rel,
        'object_type': obj.name,
    }


def print_error_summary(cam: FisheyeCamera):
    """Print a table of BBox errors at selected distances and angles for each object type.

    Distances and angles are measured from the camera position along the
    camera's forward (heading) direction, matching how mqtt_bridge produces
    camera-relative BEV coordinates.
    """
    distances = [3, 5, 8, 10, 15, 20, 25]
    angles_deg = [0, 30, 60, 80]  # angle from camera forward axis
    headings_deg = [0, 45, 90]

    # Camera forward direction on the ground plane
    fwd_rad = np.deg2rad(cam.heading_deg)
    fwd_x = np.cos(fwd_rad)
    fwd_y = np.sin(fwd_rad)
    # Perpendicular (right of forward)
    right_x = -fwd_y
    right_y = fwd_x

    print("=" * 100)
    print("BOUNDING BOX LOCALIZATION ERROR ANALYSIS")
    print(f"Camera: pos=({cam.cam_x},{cam.cam_y}), h={cam.height}m, "
          f"heading={cam.heading_deg}deg, pitch={cam.pitch_deg:.1f}deg, "
          f"FOV={cam.fov_deg}deg, R={cam.radius_px}px")
    print("=" * 100)

    for obj_name, obj in OBJECT_TYPES.items():
        if obj_name not in ('pedestrian', 'cyclist', 'car'):
            continue
        print(f"\n{'-' * 100}")
        print(f"  {obj.name.upper()} ({obj.length}m x {obj.width}m x {obj.height}m)")
        print(f"  AP={obj.ap:.3f}  AR={obj.ar:.3f}")
        print(f"{'-' * 100}")

        for heading_deg in headings_deg:
            heading_rad = np.deg2rad(heading_deg)
            print(f"\n  Heading = {heading_deg}deg from camera axis")
            print(f"  {'Dist (m)':>8}  {'Angle':>6}  {'gx':>7}  {'gy':>7}  "
                  f"{'Err (m)':>8}  {'Rel Err':>8}  {'BBox WxH':>12}  {'Corners':>7}")

            for dist in distances:
                for angle_deg in angles_deg:
                    angle_rad = np.deg2rad(angle_deg)
                    # Place point relative to camera along its heading
                    along = dist * np.cos(angle_rad)
                    across = dist * np.sin(angle_rad)
                    gx = cam.cam_x + along * fwd_x + across * right_x
                    gy = cam.cam_y + along * fwd_y + across * right_y
                    cam_dist = np.sqrt((gx - cam.cam_x)**2 + (gy - cam.cam_y)**2)
                    if cam_dist > 25:
                        continue
                    result = compute_bbox_error(cam, gx, gy, obj, heading_rad)
                    if result is None:
                        print(f"  {dist:>8.1f}  {angle_deg:>5}deg  {gx:>7.2f}  {gy:>7.2f}  "
                              f"{'N/A':>8}  {'N/A':>8}  {'N/A':>12}  {'N/A':>7}")
                    else:
                        bwh = f"{result['bbox_width_px']:.0f}x{result['bbox_height_px']:.0f}"
                        print(f"  {dist:>8.1f}  {angle_deg:>5}deg  {gx:>7.2f}  {gy:>7.2f}  "
                              f"{result['err_dist']:>8.4f}  {result['err_relative']*100:>7.2f}%  "
                              f"{bwh:>12}  {result['n_visible_corners']:>7}")


# ================================================================
# 3. DETECTION UNCERTAINTY MODEL
# ================================================================

@dataclass
class DetectionParams:
    """Per-class detection performance parameters."""
    ap: float       # average precision (AP@50:95)
    ar: float       # average recall


# COCO size thresholds (bbox area in pixels^2)
SIZE_AREA_SMALL = 32 ** 2    # < 1024
SIZE_AREA_LARGE = 96 ** 2    # >= 9216


class SizedDetectionParams:
    """Detection params with continuous AR interpolation over bbox area.

    When an ar_curve is provided (from eval_cache.json), AR is linearly
    interpolated over log-spaced area bins. Otherwise falls back to the
    3 coarse COCO bins.
    """

    def __init__(self, overall: DetectionParams,
                 small: Optional[DetectionParams] = None,
                 medium: Optional[DetectionParams] = None,
                 large: Optional[DetectionParams] = None,
                 ar_curve: Optional[Dict] = None):
        self.overall = overall
        self.small = small
        self.medium = medium
        self.large = large
        # ar_curve: {'bin_centers': [...], 'ar': [...]}
        self._ar_curve_centers = None
        self._ar_curve_values = None
        if ar_curve is not None and 'bin_centers' in ar_curve and 'ar' in ar_curve:
            self._ar_curve_centers = np.array(ar_curve['bin_centers'])
            self._ar_curve_values = np.array(ar_curve['ar'])

    def for_area(self, bbox_area_px: float) -> DetectionParams:
        """Return AP/AR for a given bbox pixel area, using interpolation."""
        if self._ar_curve_centers is not None:
            ar = float(np.interp(bbox_area_px,
                                 self._ar_curve_centers,
                                 self._ar_curve_values))
            # Use overall AP scaled by the AR ratio (AP and AR correlate)
            ap = self.overall.ap * (ar / max(self.overall.ar, 1e-6))
            ap = min(ap, 1.0)
            return DetectionParams(ap=ap, ar=ar)
        # Fallback to coarse bins
        if bbox_area_px < SIZE_AREA_SMALL and self.small is not None:
            return self.small
        elif bbox_area_px < SIZE_AREA_LARGE and self.medium is not None:
            return self.medium
        elif bbox_area_px >= SIZE_AREA_LARGE and self.large is not None:
            return self.large
        return self.overall


def _make_sized(overall_ap, overall_ar, per_size=None, ar_curve=None):
    """Helper to build SizedDetectionParams from eval cache data."""
    overall = DetectionParams(ap=overall_ap, ar=overall_ar)
    small = medium = large = None
    if per_size is not None:
        small = DetectionParams(
            ap=per_size.get('small', {}).get('ap', overall_ap),
            ar=per_size.get('small', {}).get('ar', overall_ar))
        medium = DetectionParams(
            ap=per_size.get('medium', {}).get('ap', overall_ap),
            ar=per_size.get('medium', {}).get('ar', overall_ar))
        large = DetectionParams(
            ap=per_size.get('large', {}).get('ap', overall_ap),
            ar=per_size.get('large', {}).get('ar', overall_ar))
    return SizedDetectionParams(overall=overall, small=small, medium=medium,
                                large=large, ar_curve=ar_curve)


@dataclass
class UncertaintyParams:
    """Full uncertainty model parameters."""
    # Defaults from fisheye_1280 model eval on fisheye-augmented COCO dataset.
    # Without ar_curve data from eval_cache.json, falls back to overall AR.
    # Run with --eval-cache eval_cache.json for continuous AR interpolation.
    pedestrian: SizedDetectionParams = field(default_factory=lambda: _make_sized(0.678, 0.755))
    cyclist: SizedDetectionParams = field(default_factory=lambda: _make_sized(0.657, 0.747))
    vehicle: SizedDetectionParams = field(default_factory=lambda: _make_sized(0.706, 0.765))
    bev_angular_error_deg: float = 5.0
    kf_std_weight_pos: float = 0.05      # 1/20
    kf_std_weight_vel: float = 0.00625   # 1/160
    kf_max_lost: int = 30
    kf_iou_threshold: float = 0.2
    fps: int = 30
    latency_ms: float = 19.0

    def for_class(self, class_name: str) -> SizedDetectionParams:
        if class_name in ('pedestrian', 'person'):
            return self.pedestrian
        elif class_name in ('cyclist', 'bike', 'bicycle'):
            return self.cyclist
        else:
            return self.vehicle


# ── Detection failure models ──
#
# These model the temporal correlation of detection failures. The i.i.d.
# model (default) drops each detection independently with P(miss) = 1 - AR.
# The Gilbert model uses a two-state Markov chain to produce correlated
# (bursty) misses, which is more realistic for video object detection.

class DetectionFailureModel:
    """Base class for detection failure models."""

    def should_drop(self, agent_id: int, ar: float, rng) -> bool:
        """Return True if this agent should be dropped (missed) this frame.

        Args:
            agent_id: unique identifier for the tracked object.
            ar: the per-frame average recall (detection probability) for
                this object at its current size/distance.
            rng: numpy random generator.
        """
        raise NotImplementedError


class IIDFailureModel(DetectionFailureModel):
    """Independent identically distributed detection failures.

    Each frame, each agent is dropped with probability 1 - AR,
    independently of all other frames. This is the simplest model
    and the default.
    """

    def should_drop(self, agent_id, ar, rng):
        return rng.random() > ar


class GilbertFailureModel(DetectionFailureModel):
    """Gilbert (two-state Markov) model for correlated detection failures.

    States: Detected (D) and Missed (M).
    Transitions:
        P(M at t+1 | D at t) = p   (miss onset)
        P(D at t+1 | M at t) = r   (recovery)

    The stationary miss rate is p / (p + r).
    The mean burst length (consecutive misses) is 1 / r.

    The AR from the size-aware model is used to set the stationary miss
    rate: p / (p + r) = 1 - AR. Given a user-specified mean burst length
    B = 1/r, the parameters are:

        r = 1 / B
        p = (1 - AR) * r / AR

    Reference: Gilbert, E.N. (1960). "Capacity of a burst-noise channel."
    Bell System Technical Journal.
    """

    def __init__(self, mean_burst_length: float = 3.0):
        """
        Args:
            mean_burst_length: expected number of consecutive missed frames
                once a miss begins. Typical values: 2-5 for a decent detector.
                Setting to 1.0 recovers approximately i.i.d. behavior.
        """
        self.mean_burst = mean_burst_length
        self._states = {}  # agent_id -> bool (True = currently detected)

    def should_drop(self, agent_id, ar, rng):
        if ar >= 1.0:
            return False
        if ar <= 0.0:
            return True

        r = 1.0 / self.mean_burst  # recovery probability
        p = (1.0 - ar) * r / max(ar, 1e-6)  # miss onset probability

        was_detected = self._states.get(agent_id, True)

        if was_detected:
            # Transition D -> M with probability p
            missed = rng.random() < p
        else:
            # Transition M -> D with probability r
            missed = rng.random() >= r

        self._states[agent_id] = not missed
        return missed

    def reset(self):
        """Reset all agent states (call between scenarios)."""
        self._states.clear()


# Available failure models, selectable via --failure-model CLI flag.
FAILURE_MODELS = {
    'iid': IIDFailureModel,
    'gilbert': GilbertFailureModel,
}


def load_uncertainty_from_cache(cache_path: str,
                                model_name: str = None) -> Optional['UncertaintyParams']:
    """Load AP/AR values from eval_cache.json into UncertaintyParams.

    Args:
        cache_path: path to eval_cache.json produced by eval_models.py.
        model_name: which model entry to use. If None, uses the first one.

    Returns:
        UncertaintyParams with real AP/AR values, or None if cache not found.
    """
    try:
        with open(cache_path, 'r') as f:
            cache = json.load(f)
    except FileNotFoundError:
        return None

    if model_name is None:
        model_name = next(iter(cache))
    if model_name not in cache:
        print(f'Warning: model "{model_name}" not in cache, available: {list(cache.keys())}')
        return None

    data = cache[model_name]
    mg = data.get('merged', {})

    ped = mg.get('pedestrian', {})
    cyc = mg.get('cyclist', {})
    veh = mg.get('vehicle', {})

    params = UncertaintyParams(
        pedestrian=_make_sized(
            ped.get('ap5095', 0.678), ped.get('recall', 0.755),
            ped.get('per_size'), ped.get('ar_curve')),
        cyclist=_make_sized(
            cyc.get('ap5095', 0.657), cyc.get('recall', 0.747),
            cyc.get('per_size'), cyc.get('ar_curve')),
        vehicle=_make_sized(
            veh.get('ap5095', 0.706), veh.get('recall', 0.765),
            veh.get('per_size'), veh.get('ar_curve')),
    )

    print(f'  Loaded from cache ({model_name}):')
    for name, sp in [('pedestrian', params.pedestrian),
                     ('cyclist', params.cyclist),
                     ('vehicle', params.vehicle)]:
        line = f'    {name:>12}: AP={sp.overall.ap:.3f} AR={sp.overall.ar:.3f}'
        if sp._ar_curve_centers is not None:
            line += f'  [AR curve: {len(sp._ar_curve_centers)} bins]'
        elif sp.small:
            line += f'  | S:{sp.small.ar:.3f} M:{sp.medium.ar:.3f} L:{sp.large.ar:.3f}'
        print(line)

    return params


def compute_position_uncertainty(cam: FisheyeCamera, gx: float, gy: float,
                                 obj: ObjectType, unc: UncertaintyParams) -> Dict:
    """Compute position uncertainty for an object at a given location."""
    dist = cam.dist_to(gx, gy)
    if dist < 0.1:
        dist = 0.1

    sized = unc.for_class(obj.name)

    # Compute bbox area by projecting 8 3D corners through the fisheye model.
    # This accounts for camera position, heading, pitch, and fisheye distortion.
    bbox_area = cam.projected_bbox_area(gx, gy, obj)
    det = sized.for_area(bbox_area)

    # Approximate bbox pixel dimensions for localization error estimation
    f_px = cam.focal_length
    bbox_w = max(np.sqrt(bbox_area * 9.0) * (obj.width / max(obj.height, 0.1)), 1.0)
    bbox_h = max(np.sqrt(bbox_area * 9.0) * (obj.height / max(obj.width, 0.1)), 1.0)
    px_to_world = dist / max(f_px, 1.0)

    # YOLO localization error
    dx_px = bbox_w * (1 - det.ap) / (1 + det.ap)
    dy_px = bbox_h * (1 - det.ap) / (1 + det.ap)
    sigma_yolo = np.sqrt(dx_px**2 + dy_px**2) * px_to_world

    # BEV projection error
    sigma_bev = dist * np.tan(np.deg2rad(unc.bev_angular_error_deg))

    # Kalman filter position uncertainty at K=0
    qp = (unc.kf_std_weight_pos * bbox_h) ** 2
    sigma_kf = np.sqrt(qp) * px_to_world

    # Combined RSS
    sigma_total = np.sqrt(sigma_yolo**2 + sigma_bev**2 + sigma_kf**2)

    # Velocity uncertainty
    vel_std_px = bbox_h * np.sqrt(unc.kf_std_weight_vel * unc.kf_std_weight_pos)
    vel_std_world = vel_std_px * px_to_world * unc.fps

    return {
        'dist': dist,
        'sigma_yolo': sigma_yolo,
        'sigma_bev': sigma_bev,
        'sigma_kf': sigma_kf,
        'sigma_total': sigma_total,
        'vel_std_world': vel_std_world,
        'bbox_w_px': bbox_w,
        'bbox_h_px': bbox_h,
    }


# ================================================================
# 4. DECISION PIPELINE (uses shared decision_pipeline.py module)
# ================================================================

@dataclass
class DecisionParams:
    """Parameters for constructing a DecisionPipeline instance."""
    bike_memory_frames: int = 58
    prox_min: float = 1.9
    prox_max: float = 24.8
    prox_speed_min: float = 0.147
    lookback_frames: int = 2
    max_range: float = 25.0
    max_object_age: int = 10
    speed_adaptive: bool = True
    decision_rule: str = 'pairwise'
    ttc_threshold_s: float = 3.0


PIPELINE_CONFIGS = {
    # ── Pipeline parameter variants (all use pairwise closing) ──
    'default': DecisionParams(
        bike_memory_frames=58, prox_min=1.9, prox_max=24.8,
        prox_speed_min=0.147, lookback_frames=2, speed_adaptive=True),
    'conservative': DecisionParams(
        bike_memory_frames=60, prox_min=1.5, prox_max=15.0,
        prox_speed_min=0.05, lookback_frames=3, speed_adaptive=False),
    'proximity_only': DecisionParams(
        bike_memory_frames=0, prox_min=2.0, prox_max=10.0,
        prox_speed_min=0.1, lookback_frames=3, speed_adaptive=False),
    'speed_adaptive': DecisionParams(
        bike_memory_frames=60, prox_min=1.0, prox_max=20.0,
        prox_speed_min=0.05, lookback_frames=2, speed_adaptive=True),
    'optimized': DecisionParams(
        bike_memory_frames=58, prox_min=1.9, prox_max=24.8,
        prox_speed_min=0.147, lookback_frames=2, speed_adaptive=True),
    # ── Decision rule baselines (use optimized params, vary the rule) ──
    'baseline_distance_only': DecisionParams(
        bike_memory_frames=58, prox_min=1.9, prox_max=10.0,
        prox_speed_min=0.0, lookback_frames=2, speed_adaptive=False,
        decision_rule='distance_only'),
    'baseline_naive_closing': DecisionParams(
        bike_memory_frames=58, prox_min=1.9, prox_max=24.8,
        prox_speed_min=0.147, lookback_frames=2, speed_adaptive=True,
        decision_rule='naive_closing'),
    'baseline_ttc': DecisionParams(
        bike_memory_frames=58, prox_min=1.9, prox_max=24.8,
        prox_speed_min=0.0, lookback_frames=2, speed_adaptive=True,
        decision_rule='ttc', ttc_threshold_s=3.0),
}


def make_pipeline(params: DecisionParams, fps: int = 30,
                  profile: Optional['CyclistBrakingProfile'] = None) -> DecisionPipeline:
    """Create a DecisionPipeline from DecisionParams."""
    if profile is None:
        profile = DEFAULT_BRAKING_PROFILE
    return DecisionPipeline(
        bike_memory_frames=params.bike_memory_frames,
        prox_min=params.prox_min,
        prox_max=params.prox_max,
        prox_speed_min=params.prox_speed_min,
        lookback_frames=params.lookback_frames,
        max_range=params.max_range,
        speed_adaptive=params.speed_adaptive,
        cyclist_reaction_s=profile.prt_s,
        cyclist_decel_ms2=profile.decel_ms2,
        fps=fps,
        decision_rule=params.decision_rule,
        ttc_threshold_s=params.ttc_threshold_s,
    )


def run_pipeline_on_agents(pipeline: DecisionPipeline, agents: List[Dict],
                           sim_time: float,
                           cam_x: float = 0.0, cam_y: float = 0.0) -> str:
    """Adapter: split agents by type, feed the shared pipeline, return state.

    Coordinates are transformed to camera-relative before entering the
    pipeline, matching the mqtt_bridge deployment where the fisheye
    inverse-projection produces camera-centred BEV coordinates.
    """
    persons = []
    bikes = []
    bike_in_scene = False
    for a in agents:
        rx = a['x'] - cam_x
        ry = a['y'] - cam_y
        entry = {'track_id': a['id'], 'x': rx, 'y': ry}
        pipeline.update_track(a['id'], rx, ry, sim_time)
        if a['type'] == 'person':
            persons.append(entry)
        elif a['type'] == 'bike':
            bikes.append(entry)
            bike_in_scene = True

    state, _ = pipeline.evaluate(persons, bikes, bike_in_scene, sim_time)
    return state


# ================================================================
# 5. TEST SCENARIOS
# ================================================================
#
# Each scenario may specify a 'vehicle' field: 'bicycle' (default),
# 'ebike', or 'aashto'. This selects the braking profile used for
# ground-truth danger labeling and the speed-adaptive stopping
# distance in the decision pipeline.

SCENARIOS = [
    {
        'name': 'Safe Crossing',
        'expected_dominant': 'safe',
        'agents': [
            {'type': 'person', 'id': 1,
             'path': [{'x': 5, 'y': -8, 't': 0}, {'x': 5, 'y': -4, 't': 1.5},
                      {'x': 5, 'y': 0, 't': 3}, {'x': 5, 'y': 4, 't': 4.5},
                      {'x': 5, 'y': 8, 't': 6}]},
        ],
    },
    {
        'name': 'Bike Approaches Ped',
        'expected_dominant': 'alert',
        'agents': [
            {'type': 'person', 'id': 1,
             'path': [{'x': 5, 'y': -6, 't': 0}, {'x': 5, 'y': -3, 't': 1.5},
                      {'x': 5, 'y': 0, 't': 3}, {'x': 5, 'y': 3, 't': 4.5},
                      {'x': 5, 'y': 6, 't': 6}, {'x': 5, 'y': 8, 't': 7}]},
            {'type': 'bike', 'id': 2,
             'path': [{'x': 34, 'y': 2, 't': 0}, {'x': 26, 'y': 1.5, 't': 1},
                      {'x': 18, 'y': 1, 't': 2}, {'x': 14, 'y': 0.5, 't': 3},
                      {'x': 10, 'y': 0, 't': 4}, {'x': 7, 'y': -0.2, 't': 5},
                      {'x': 5.2, 'y': -0.4, 't': 6}, {'x': 3, 'y': -0.6, 't': 7}]},
        ],
    },
    {
        'name': 'High-Speed Near Miss',
        'expected_dominant': 'alert',
        'agents': [
            {'type': 'person', 'id': 1,
             'path': [{'x': 5, 'y': -3, 't': 0}, {'x': 5, 'y': -2, 't': 1},
                      {'x': 5, 'y': -1, 't': 2}, {'x': 5, 'y': 0, 't': 3},
                      {'x': 5, 'y': 1, 't': 4}, {'x': 5, 'y': 2, 't': 5},
                      {'x': 5, 'y': 3, 't': 6}]},
            {'type': 'bike', 'id': 2,
             'path': [{'x': 34, 'y': 1.6, 't': 0}, {'x': 27, 'y': 1.3, 't': 1},
                      {'x': 20, 'y': 1, 't': 2}, {'x': 16, 'y': 0.8, 't': 2.7},
                      {'x': 12, 'y': 0.5, 't': 3.4}, {'x': 8, 'y': 0.2, 't': 4.1},
                      {'x': 5.5, 'y': 0, 't': 4.6}, {'x': 3, 'y': -0.2, 't': 5.1},
                      {'x': -1, 'y': -0.5, 't': 5.8}]},
        ],
    },
    {
        'name': 'Multiple VRUs',
        'expected_dominant': 'alert',
        'agents': [
            {'type': 'person', 'id': 1,
             'path': [{'x': 5, 'y': -8, 't': 0}, {'x': 5, 'y': -4, 't': 2},
                      {'x': 5, 'y': 0, 't': 4}, {'x': 5, 'y': 4, 't': 6},
                      {'x': 5, 'y': 8, 't': 8}]},
            {'type': 'person', 'id': 2,
             'path': [{'x': 5, 'y': 7, 't': 0.5}, {'x': 5, 'y': 3, 't': 2},
                      {'x': 5, 'y': -1, 't': 3.5}, {'x': 5, 'y': -5, 't': 5},
                      {'x': 5, 'y': -9, 't': 7}]},
            {'type': 'person', 'id': 3,
             'path': [{'x': 5, 'y': -4, 't': 1}, {'x': 5, 'y': 0, 't': 3},
                      {'x': 5, 'y': 4, 't': 5}, {'x': 5, 'y': 8, 't': 7}]},
            {'type': 'bike', 'id': 4,
             'path': [{'x': 34, 'y': -2, 't': 0}, {'x': 28, 'y': -1.5, 't': 1},
                      {'x': 22, 'y': -1, 't': 2}, {'x': 17, 'y': -0.5, 't': 3.2},
                      {'x': 12, 'y': 0, 't': 4.4}, {'x': 7, 'y': 0.3, 't': 5.6},
                      {'x': 2, 'y': 0.5, 't': 6.8}]},
            {'type': 'bike', 'id': 5,
             'path': [{'x': 27, 'y': 12, 't': 0.5}, {'x': 21, 'y': 9, 't': 1.5},
                      {'x': 15, 'y': 6, 't': 2.5}, {'x': 12, 'y': 4, 't': 3.5},
                      {'x': 9, 'y': 2, 't': 4.5}, {'x': 6, 'y': 0, 't': 5.5},
                      {'x': 3, 'y': -2, 't': 6.5}]},
        ],
    },
    {
        'name': 'Occluded Emergence',
        'expected_dominant': 'alert',
        'agents': [
            {'type': 'person', 'id': 1,
             'path': [{'x': 5, 'y': -3, 't': 0}, {'x': 5, 'y': -2, 't': 1},
                      {'x': 5, 'y': -1, 't': 2}, {'x': 5, 'y': 0, 't': 3},
                      {'x': 5, 'y': 1, 't': 4}, {'x': 5, 'y': 2, 't': 5},
                      {'x': 5, 'y': 3, 't': 6}]},
            # Occluded: bike only visible from 9m (emerges from behind obstacle)
            # but was approaching from further away
            {'type': 'bike', 'id': 2,
             'path': [{'x': 9, 'y': 4, 't': 3}, {'x': 7.5, 'y': 2.5, 't': 3.5},
                      {'x': 6, 'y': 1, 't': 4}, {'x': 4.8, 'y': -0.2, 't': 4.5},
                      {'x': 3.5, 'y': -1.5, 't': 5}]},
        ],
    },
    {
        'name': 'Parallel Paths',
        'expected_dominant': 'warning',  # should NOT alert: constant offset, not closing
        'agents': [
            {'type': 'person', 'id': 1,
             'path': [{'x': 5, 'y': -6, 't': 0}, {'x': 5, 'y': -3, 't': 1.5},
                      {'x': 5, 'y': 0, 't': 3}, {'x': 5, 'y': 3, 't': 4.5},
                      {'x': 5, 'y': 6, 't': 6}, {'x': 5, 'y': 9, 't': 7.5}]},
            {'type': 'bike', 'id': 2,
             'path': [{'x': 8, 'y': -6, 't': 2}, {'x': 8, 'y': -3, 't': 3.5},
                      {'x': 8, 'y': 0, 't': 5}, {'x': 8, 'y': 3, 't': 6.5},
                      {'x': 8, 'y': 6, 't': 8}]},
        ],
    },
    {
        'name': 'Bikes Only',
        'expected_dominant': 'idle',
        'agents': [
            {'type': 'bike', 'id': 1,
             'path': [{'x': 18, 'y': -2, 't': 0}, {'x': 14, 'y': -1.5, 't': 1},
                      {'x': 10, 'y': -1, 't': 2}, {'x': 6, 'y': -0.5, 't': 3},
                      {'x': 2, 'y': 0, 't': 4}]},
            {'type': 'bike', 'id': 2,
             'path': [{'x': 15, 'y': 5, 't': 1}, {'x': 12, 'y': 4, 't': 2},
                      {'x': 9, 'y': 3, 't': 3}, {'x': 6, 'y': 2, 't': 4},
                      {'x': 3, 'y': 1, 't': 5}]},
        ],
    },
    {
        'name': 'Fast Approach',
        'expected_dominant': 'alert',
        'vehicle': 'ebike',
        'agents': [
            {'type': 'person', 'id': 1,
             'path': [{'x': 5, 'y': -4, 't': 0}, {'x': 5, 'y': -2, 't': 1.5},
                      {'x': 5, 'y': 0, 't': 3}, {'x': 5, 'y': 2, 't': 4.5},
                      {'x': 5, 'y': 4, 't': 6}]},
            {'type': 'bike', 'id': 2,
             'path': [{'x': 25, 'y': 0.5, 't': 2}, {'x': 13, 'y': 0.2, 't': 3},
                      {'x': 1, 'y': -0.1, 't': 4}]},
        ],
    },
    {
        'name': 'Speed Variant',
        'expected_dominant': 'alert',
        'agents': [
            {'type': 'person', 'id': 1,
             'path': [{'x': 5, 'y': -3, 't': 0}, {'x': 5, 'y': -1, 't': 2},
                      {'x': 5, 'y': 1, 't': 4}, {'x': 5, 'y': 3, 't': 6},
                      {'x': 5, 'y': 5, 't': 8}, {'x': 5, 'y': 7, 't': 10}]},
            {'type': 'bike', 'id': 2,
             'path': [{'x': 20, 'y': 2, 't': 2}, {'x': 17, 'y': 1.7, 't': 3},
                      {'x': 14, 'y': 1.4, 't': 4}, {'x': 11, 'y': 1.1, 't': 5},
                      {'x': 8, 'y': 0.8, 't': 6}, {'x': 5, 'y': 0.5, 't': 7},
                      {'x': 2, 'y': 0.2, 't': 8}]},
            {'type': 'bike', 'id': 3,
             'path': [{'x': 20, 'y': -2, 't': 5}, {'x': 14, 'y': -1.5, 't': 5.6},
                      {'x': 8, 'y': -1, 't': 6.2}, {'x': 2, 'y': -0.5, 't': 6.8}]},
        ],
    },
    {
        'name': 'Counter-Flow on Crosswalk',
        'expected_dominant': 'alert',
        'agents': [
            {'type': 'person', 'id': 1,
             'path': [{'x': 5, 'y': -6, 't': 0}, {'x': 5, 'y': -3, 't': 2},
                      {'x': 5, 'y': 0, 't': 4}, {'x': 5, 'y': 3, 't': 6}]},
            {'type': 'bike', 'id': 2,
             'path': [{'x': 5.5, 'y': 12, 't': 2}, {'x': 5.3, 'y': 8, 't': 3},
                      {'x': 5.1, 'y': 4, 't': 4}, {'x': 5, 'y': 0, 't': 5},
                      {'x': 4.9, 'y': -4, 't': 6}]},
        ],
    },
    {
        'name': 'Multi-Speed Converging',
        'expected_dominant': 'alert',
        'agents': [
            {'type': 'person', 'id': 1,
             'path': [{'x': 5, 'y': -8, 't': 0}, {'x': 5, 'y': -4, 't': 2},
                      {'x': 5, 'y': 0, 't': 4}, {'x': 5, 'y': 4, 't': 6},
                      {'x': 5, 'y': 8, 't': 8}]},
            {'type': 'person', 'id': 2,
             'path': [{'x': 5, 'y': 6, 't': 1}, {'x': 5, 'y': 2, 't': 3},
                      {'x': 5, 'y': -2, 't': 5}, {'x': 5, 'y': -6, 't': 7}]},
            {'type': 'bike', 'id': 3,
             'path': [{'x': 20, 'y': 3, 't': 2}, {'x': 16, 'y': 2.5, 't': 3.5},
                      {'x': 12, 'y': 2, 't': 5}, {'x': 8, 'y': 1.5, 't': 6.5},
                      {'x': 4, 'y': 1, 't': 8}]},
            {'type': 'bike', 'id': 4,
             'path': [{'x': 22, 'y': -6, 't': 4}, {'x': 16, 'y': -4, 't': 4.8},
                      {'x': 10, 'y': -2, 't': 5.6}, {'x': 4, 'y': 0, 't': 6.4}]},
            {'type': 'bike', 'id': 5,
             'path': [{'x': 18, 'y': 8, 't': 5}, {'x': 13, 'y': 5, 't': 5.7},
                      {'x': 8, 'y': 2, 't': 6.4}, {'x': 3, 'y': -1, 't': 7.1}]},
        ],
    },
    # ── New scenarios: accessibility, edge cases, real-world failure modes ──
    {
        'name': 'Wheelchair User',
        'expected_dominant': 'alert',
        'agents': [
            # Wheelchair user at 0.6 m/s (half walking speed), 2x exposure time
            {'type': 'person', 'id': 1,
             'path': [{'x': 5, 'y': -8, 't': 0}, {'x': 5, 'y': -4, 't': 6.7},
                      {'x': 5, 'y': 0, 't': 13.3}, {'x': 5, 'y': 4, 't': 20},
                      {'x': 5, 'y': 8, 't': 26.7}]},
            {'type': 'bike', 'id': 2,
             'path': [{'x': 34, 'y': 2, 't': 6}, {'x': 26, 'y': 1.5, 't': 7},
                      {'x': 18, 'y': 1, 't': 8}, {'x': 14, 'y': 0.5, 't': 9},
                      {'x': 10, 'y': 0, 't': 10}, {'x': 7, 'y': -0.2, 't': 11},
                      {'x': 5.2, 'y': -0.4, 't': 12}, {'x': 3, 'y': -0.6, 't': 13}]},
        ],
    },
    {
        'name': 'Stopped Pedestrian',
        'expected_dominant': 'alert',
        'agents': [
            # Pedestrian walks to mid-crosswalk, stops for 4s (phone), resumes
            {'type': 'person', 'id': 1,
             'path': [{'x': 5, 'y': -6, 't': 0}, {'x': 5, 'y': -2, 't': 3},
                      {'x': 5, 'y': 0, 't': 4.5},
                      {'x': 5, 'y': 0, 't': 8.5},   # stationary 4 seconds
                      {'x': 5, 'y': 3, 't': 10.8},
                      {'x': 5, 'y': 6, 't': 13}]},
            {'type': 'bike', 'id': 2,
             'path': [{'x': 20, 'y': 0.5, 't': 5}, {'x': 15, 'y': 0.3, 't': 6},
                      {'x': 10, 'y': 0.1, 't': 7}, {'x': 6, 'y': -0.1, 't': 8},
                      {'x': 3, 'y': -0.2, 't': 9}]},
        ],
    },
    {
        'name': 'Cross-Street Cyclist',
        'expected_dominant': 'alert',
        'agents': [
            # Pedestrian on crosswalk
            {'type': 'person', 'id': 1,
             'path': [{'x': 5, 'y': -5, 't': 0}, {'x': 5, 'y': -2, 't': 2.3},
                      {'x': 5, 'y': 0, 't': 3.8}, {'x': 5, 'y': 3, 't': 6},
                      {'x': 5, 'y': 6, 't': 8}]},
            # Cyclist runs red from cross street: approaches along y-axis
            {'type': 'bike', 'id': 2,
             'path': [{'x': 5.5, 'y': -20, 't': 2}, {'x': 5.3, 'y': -14, 't': 3},
                      {'x': 5.1, 'y': -8, 't': 4}, {'x': 5, 'y': -2, 't': 5},
                      {'x': 4.9, 'y': 4, 't': 6}, {'x': 4.8, 'y': 10, 't': 7}]},
        ],
    },
    {
        'name': 'E-Scooter',
        'expected_dominant': 'alert',
        'vehicle': 'ebike',
        'agents': [
            # Pedestrian crossing
            {'type': 'person', 'id': 1,
             'path': [{'x': 5, 'y': -5, 't': 0}, {'x': 5, 'y': -2, 't': 2.3},
                      {'x': 5, 'y': 1, 't': 4.6}, {'x': 5, 'y': 4, 't': 7}]},
            # E-scooter at 25 km/h (6.9 m/s), detected as bike
            {'type': 'bike', 'id': 2,
             'path': [{'x': 22, 'y': 0.3, 't': 2}, {'x': 15, 'y': 0.1, 't': 3},
                      {'x': 8, 'y': -0.1, 't': 4}, {'x': 1, 'y': -0.3, 't': 5}]},
        ],
    },
    {
        'name': 'Child Pedestrian',
        'expected_dominant': 'alert',
        'agents': [
            # Child at 0.9 m/s, 1.0m tall (smaller projected area)
            {'type': 'person', 'id': 1,
             'path': [{'x': 5, 'y': -6, 't': 0}, {'x': 5, 'y': -3, 't': 3.3},
                      {'x': 5, 'y': 0, 't': 6.7}, {'x': 5, 'y': 3, 't': 10},
                      {'x': 5, 'y': 6, 't': 13.3}]},
            {'type': 'bike', 'id': 2,
             'path': [{'x': 34, 'y': 1, 't': 2}, {'x': 26, 'y': 0.8, 't': 3},
                      {'x': 18, 'y': 0.5, 't': 4}, {'x': 14, 'y': 0.3, 't': 5},
                      {'x': 10, 'y': 0.1, 't': 6}, {'x': 7, 'y': -0.1, 't': 7},
                      {'x': 5, 'y': -0.2, 't': 8}, {'x': 3, 'y': -0.3, 't': 9}]},
        ],
    },
    {
        'name': 'Overtake from Behind',
        'expected_dominant': 'alert',
        'agents': [
            # Pedestrian walks along crosswalk
            {'type': 'person', 'id': 1,
             'path': [{'x': 5, 'y': -8, 't': 0}, {'x': 5, 'y': -4, 't': 3},
                      {'x': 5, 'y': 0, 't': 6}, {'x': 5, 'y': 4, 't': 9},
                      {'x': 5, 'y': 8, 't': 12}]},
            # Cyclist overtakes from behind on the same path, zero lateral offset
            {'type': 'bike', 'id': 2,
             'path': [{'x': 5.2, 'y': -12, 't': 3}, {'x': 5.1, 'y': -6, 't': 4.5},
                      {'x': 5, 'y': 0, 't': 6}, {'x': 4.9, 'y': 6, 't': 7.5},
                      {'x': 4.8, 'y': 12, 't': 9}]},
        ],
    },
    {
        'name': 'Cyclist Abort',
        'expected_dominant': 'safe',  # threat resolves, alert should clear
        'agents': [
            {'type': 'person', 'id': 1,
             'path': [{'x': 5, 'y': -4, 't': 0}, {'x': 5, 'y': -2, 't': 1.5},
                      {'x': 5, 'y': 0, 't': 3}, {'x': 5, 'y': 2, 't': 4.5},
                      {'x': 5, 'y': 4, 't': 6}, {'x': 5, 'y': 6, 't': 7.5}]},
            # Cyclist approaches then turns away at 8m
            {'type': 'bike', 'id': 2,
             'path': [{'x': 20, 'y': 1, 't': 2}, {'x': 15, 'y': 0.5, 't': 3},
                      {'x': 10, 'y': 0, 't': 4}, {'x': 8, 'y': -0.5, 't': 4.5},
                      {'x': 9, 'y': -3, 't': 5.5}, {'x': 12, 'y': -6, 't': 7}]},
        ],
    },
    {
        'name': 'Dense Pedestrian Group',
        'expected_dominant': 'alert',
        'agents': [
            # 8 pedestrians in a cluster crossing together
            {'type': 'person', 'id': 1,
             'path': [{'x': 4.5, 'y': -6, 't': 0}, {'x': 4.5, 'y': 0, 't': 4.6},
                      {'x': 4.5, 'y': 6, 't': 9.2}]},
            {'type': 'person', 'id': 2,
             'path': [{'x': 5.5, 'y': -6, 't': 0.2}, {'x': 5.5, 'y': 0, 't': 4.8},
                      {'x': 5.5, 'y': 6, 't': 9.4}]},
            {'type': 'person', 'id': 3,
             'path': [{'x': 4.8, 'y': -5.5, 't': 0.4}, {'x': 4.8, 'y': 0.5, 't': 5},
                      {'x': 4.8, 'y': 6.5, 't': 9.6}]},
            {'type': 'person', 'id': 4,
             'path': [{'x': 5.2, 'y': -5.8, 't': 0.1}, {'x': 5.2, 'y': 0.2, 't': 4.7},
                      {'x': 5.2, 'y': 6.2, 't': 9.3}]},
            {'type': 'person', 'id': 5,
             'path': [{'x': 4.6, 'y': -6.2, 't': 0.3}, {'x': 4.6, 'y': -0.2, 't': 4.9},
                      {'x': 4.6, 'y': 5.8, 't': 9.5}]},
            {'type': 'person', 'id': 6,
             'path': [{'x': 5.4, 'y': -5.6, 't': 0.5}, {'x': 5.4, 'y': 0.4, 't': 5.1},
                      {'x': 5.4, 'y': 6.4, 't': 9.7}]},
            {'type': 'person', 'id': 7,
             'path': [{'x': 5.0, 'y': -6.4, 't': 0.6}, {'x': 5.0, 'y': -0.4, 't': 5.2},
                      {'x': 5.0, 'y': 5.6, 't': 9.8}]},
            {'type': 'person', 'id': 8,
             'path': [{'x': 4.9, 'y': -5.4, 't': 0.7}, {'x': 4.9, 'y': 0.6, 't': 5.3},
                      {'x': 4.9, 'y': 6.6, 't': 9.9}]},
            # Cyclist approaches the group
            {'type': 'bike', 'id': 9,
             'path': [{'x': 20, 'y': 0.5, 't': 3}, {'x': 15, 'y': 0.3, 't': 4.2},
                      {'x': 10, 'y': 0.1, 't': 5.4}, {'x': 6, 'y': -0.1, 't': 6.6},
                      {'x': 3, 'y': -0.2, 't': 7.5}]},
        ],
    },
    {
        'name': 'Diagonal Jaywalker',
        'expected_dominant': 'alert',
        'agents': [
            # Pedestrian crosses diagonally, not on crosswalk
            {'type': 'person', 'id': 1,
             'path': [{'x': 2, 'y': -6, 't': 0}, {'x': 4, 'y': -2, 't': 3},
                      {'x': 6, 'y': 2, 't': 6}, {'x': 8, 'y': 6, 't': 9}]},
            {'type': 'bike', 'id': 2,
             'path': [{'x': 34, 'y': 1, 't': 0}, {'x': 26, 'y': 0.8, 't': 1},
                      {'x': 18, 'y': 0.5, 't': 2}, {'x': 14, 'y': 0.3, 't': 3},
                      {'x': 10, 'y': 0, 't': 4}, {'x': 6, 'y': -0.2, 't': 5},
                      {'x': 3, 'y': -0.4, 't': 6}]},
        ],
    },
    # ── Non-linear trajectory scenarios (stress-test forecasting) ──
    {
        'name': 'Swerving Cyclist',
        'expected_dominant': 'alert',
        'agents': [
            # Pedestrian mid-crosswalk
            {'type': 'person', 'id': 1,
             'path': [{'x': 5, 'y': -4, 't': 0}, {'x': 5, 'y': -1, 't': 2.5},
                      {'x': 5, 'y': 2, 't': 5}, {'x': 5, 'y': 5, 't': 7.5}]},
            # Cyclist approaching straight, then swerves toward pedestrian
            {'type': 'bike', 'id': 2,
             'path': [{'x': 30, 'y': 6, 't': 0}, {'x': 24, 'y': 6, 't': 1},
                      {'x': 18, 'y': 5.5, 't': 2}, {'x': 14, 'y': 4, 't': 3},
                      {'x': 10, 'y': 2, 't': 4}, {'x': 7, 'y': 0, 't': 5},
                      {'x': 5.2, 'y': -0.5, 't': 6}]},
        ],
    },
    {
        'name': 'Late Turn Toward Ped',
        'expected_dominant': 'alert',
        'agents': [
            # Pedestrian on crosswalk
            {'type': 'person', 'id': 1,
             'path': [{'x': 5, 'y': -3, 't': 0}, {'x': 5, 'y': 0, 't': 2.3},
                      {'x': 5, 'y': 3, 't': 4.6}, {'x': 5, 'y': 6, 't': 7}]},
            # Cyclist riding parallel, then turns sharply toward crosswalk
            {'type': 'bike', 'id': 2,
             'path': [{'x': 30, 'y': 1, 't': 0}, {'x': 22, 'y': 1, 't': 1.5},
                      {'x': 14, 'y': 1, 't': 3},
                      {'x': 10, 'y': 0.5, 't': 4},     # begins turn
                      {'x': 7, 'y': -0.5, 't': 5},
                      {'x': 5, 'y': -1.5, 't': 6},
                      {'x': 4, 'y': -3, 't': 7}]},
        ],
    },
    {
        'name': 'U-Turn Cyclist',
        'expected_dominant': 'alert',
        'agents': [
            # Pedestrian stationary at crosswalk
            {'type': 'person', 'id': 1,
             'path': [{'x': 5, 'y': 0, 't': 0}, {'x': 5, 'y': 0, 't': 8}]},
            # Cyclist rides past, U-turns, comes back toward pedestrian
            {'type': 'bike', 'id': 2,
             'path': [{'x': 20, 'y': 3, 't': 0}, {'x': 14, 'y': 2, 't': 1},
                      {'x': 8, 'y': 1, 't': 2}, {'x': 3, 'y': 0.5, 't': 3},
                      {'x': 0, 'y': 1, 't': 4},        # passes, starts turn
                      {'x': -1, 'y': 3, 't': 5},
                      {'x': 1, 'y': 4, 't': 5.7},       # U-turn apex
                      {'x': 3, 'y': 2, 't': 6.4},
                      {'x': 5, 'y': 0.5, 't': 7.2},     # approaches ped again
                      {'x': 7, 'y': -1, 't': 8}]},
        ],
    },
    {
        'name': 'Accelerating E-Bike',
        'expected_dominant': 'alert',
        'vehicle': 'ebike',
        'agents': [
            # Pedestrian crossing
            {'type': 'person', 'id': 1,
             'path': [{'x': 5, 'y': -5, 't': 0}, {'x': 5, 'y': -2, 't': 2.3},
                      {'x': 5, 'y': 1, 't': 4.6}, {'x': 5, 'y': 4, 't': 7}]},
            # E-bike starts slow, accelerates hard toward crossing
            {'type': 'bike', 'id': 2,
             'path': [{'x': 30, 'y': 0.5, 't': 0},
                      {'x': 27, 'y': 0.4, 't': 1},      # 3 m/s
                      {'x': 23, 'y': 0.3, 't': 2},      # 4 m/s
                      {'x': 17, 'y': 0.2, 't': 3},      # 6 m/s
                      {'x': 9, 'y': 0.1, 't': 4},       # 8 m/s (throttle)
                      {'x': 1, 'y': 0, 't': 5}]},       # 8 m/s
        ],
    },
]


def interpolate_agent(agent_def: Dict, t: float) -> Optional[Dict]:
    """Interpolate agent position and heading at time t from waypoint path.

    Heading is computed from the velocity vector of the current path segment,
    in radians (0 = facing +x direction).
    """
    path = agent_def['path']
    if t < path[0]['t'] or t > path[-1]['t']:
        return None
    for i in range(len(path) - 1):
        if path[i]['t'] <= t <= path[i + 1]['t']:
            frac = (t - path[i]['t']) / (path[i + 1]['t'] - path[i]['t'])
            dx = path[i + 1]['x'] - path[i]['x']
            dy = path[i + 1]['y'] - path[i]['y']
            x = path[i]['x'] + dx * frac
            y = path[i]['y'] + dy * frac
            heading = np.arctan2(dy, dx)
            return {'id': agent_def['id'], 'type': agent_def['type'],
                    'x': x, 'y': y, 'heading': float(heading)}
    return None


def get_scenario_duration(scenario: Dict) -> float:
    max_t = 0.0
    for ag in scenario['agents']:
        last_t = ag['path'][-1]['t']
        if last_t > max_t:
            max_t = last_t
    return max_t + 0.5


# ================================================================
# 5b. KINEMATIC SAFETY MODEL
# ================================================================

# ── Cyclist braking profiles ──
#
# Each profile defines the cyclist perception-reaction time (PRT) and
# comfortable braking deceleration used in the stopping distance formula
# d_stop = v * PRT + v^2 / (2 * a).
#
# Three profiles are defined for comparative analysis:
#
#   bicycle   - Conventional bicycle, field-measured 85th-percentile values
#               from Martin & Bigazzi (2025).
#               PRT = 0.84 s, decel = 1.96 m/s^2
#               https://doi.org/10.1061/JTEPBS.TEENG-8805
#
#   ebike     - E-bicycle, emergency braking on dry pavement at 25 km/h.
#               PRT uses the same Martin & Bigazzi value (no e-bike-specific
#               PRT study exists); decel from Naude et al. (2025) controlled
#               track tests (mean maximum, n=127).
#               PRT = 0.84 s, decel = 6.0 m/s^2
#               https://doi.org/10.1016/j.treng.2025.100404
#
@dataclass
class CyclistBrakingProfile:
    """Cyclist perception-reaction time and braking deceleration."""
    name: str
    prt_s: float          # perception-reaction time (seconds)
    decel_ms2: float      # comfortable braking deceleration (m/s^2)
    source: str           # citation key or short description

#
# Two profiles are defined based on field-measured values:
#
#   bicycle   - Conventional bicycle, field-measured 85th-percentile values
#               from Martin & Bigazzi (2025).
#               PRT = 0.84 s, decel = 1.96 m/s^2
#               https://doi.org/10.1061/JTEPBS.TEENG-8805
#
#   ebike     - E-bicycle, emergency braking on dry pavement at 25 km/h.
#               PRT uses the same Martin & Bigazzi value (no e-bike-specific
#               PRT study exists); decel from Naude et al. (2025) controlled
#               track tests (mean maximum, n=127).
#               PRT = 0.84 s, decel = 6.0 m/s^2
#               https://doi.org/10.1016/j.treng.2025.100404
#
# A third profile uses AASHTO design guidance values for reference:
#
#   aashto    - PRT = 2.5 s, decel = 3.4 m/s^2 from AASHTO, A Policy on
#               Geometric Design of Highways and Streets, 7th ed. (2018).

BRAKING_PROFILES = {
    'bicycle': CyclistBrakingProfile(
        name='Conventional bicycle',
        prt_s=0.84,
        decel_ms2=1.96,
        source='Martin & Bigazzi 2025 (85th pctl, field, n=302)'),
    'ebike': CyclistBrakingProfile(
        name='E-bicycle',
        prt_s=0.84,
        decel_ms2=6.0,
        source='PRT: Martin & Bigazzi 2025; decel: Naude et al. 2025 (mean max, track, n=127)'),
    'aashto': CyclistBrakingProfile(
        name='AASHTO design',
        prt_s=2.5,
        decel_ms2=3.4,
        source='AASHTO Policy on Geometric Design of Highways and Streets, 7th ed. 2018'),
}

# Default profile for all pipeline and ground-truth calculations
DEFAULT_BRAKING_PROFILE = BRAKING_PROFILES['bicycle']

BIKE_DECEL_MS2 = DEFAULT_BRAKING_PROFILE.decel_ms2
CYCLIST_REACTION_S = DEFAULT_BRAKING_PROFILE.prt_s

PEDESTRIAN_REACTION_S = 1.87     # distracted pedestrian PRT (Fugger et al. 2000, combined avg)
PEDESTRIAN_CLEAR_SPEED_MS = 1.3  # walking speed to clear conflict zone
CONFLICT_ZONE_HALF_WIDTH_M = 1.0 # half-width of pedestrian occupied zone

# Cyclist swerve avoidance model.
#
# The cyclist can avoid by swerving laterally out of the pedestrian's
# conflict zone (half-width w = 1.0 m). The swerve requires:
#
#   t_swerve = t_react + t_maneuver
#
# where t_react is the cyclist PRT (0.84 s) and t_maneuver is the time
# to move laterally by w meters under constant lateral acceleration:
#
#   w = 0.5 * a_lat * t_maneuver^2
#   => t_maneuver = sqrt(2 * w / a_lat)
#
# The maximum lateral acceleration is limited by tire-road friction:
#
#   a_lat = mu * g
#
# Using a conservative friction coefficient mu = 0.4 (wet or worn tires
# on asphalt; dry bicycle tires achieve 0.5-0.7):
#
#   a_lat = 0.4 * 9.81 = 3.92 m/s^2
#   t_maneuver = sqrt(2 * 1.0 / 3.92) = 0.71 s
#   t_swerve = 0.84 + 0.71 = 1.55 s
#
# This is independent of forward speed (lateral dynamics decouple from
# longitudinal speed for small angles), making swerve faster than
# braking at all speeds above ~5 km/h.
SWERVE_FRICTION_MU = 0.4
SWERVE_LATERAL_ACCEL = SWERVE_FRICTION_MU * 9.81  # 3.92 m/s^2
CYCLIST_SWERVE_TIME_S = DEFAULT_BRAKING_PROFILE.prt_s + np.sqrt(
    2.0 * CONFLICT_ZONE_HALF_WIDTH_M / SWERVE_LATERAL_ACCEL)  # 1.55 s

# Actionable lower bound: the distracted pedestrian PRT. Below this TTC,
# the pedestrian cannot perceive and begin reacting to the alert.
# The cyclist could potentially swerve in ~1.55s (see CYCLIST_SWERVE_TIME_S),
# but the system's primary function is to warn the pedestrian, so we
# evaluate conservatively against the pedestrian's response window.
ACTIONABLE_TTC_LOWER = PEDESTRIAN_REACTION_S


def estimate_closing_speed(bike_def: Dict, ped_def: Dict,
                           t: float, dt: float = 0.05) -> float:
    """Estimate closing speed between bike and pedestrian (m/s).

    Positive value means distance is decreasing.
    """
    bp0 = interpolate_agent(bike_def, t)
    pp0 = interpolate_agent(ped_def, t)
    bp1 = interpolate_agent(bike_def, t + dt)
    pp1 = interpolate_agent(ped_def, t + dt)
    if not all([bp0, pp0, bp1, pp1]):
        return 0.0
    d0 = np.sqrt((bp0['x'] - pp0['x'])**2 + (bp0['y'] - pp0['y'])**2)
    d1 = np.sqrt((bp1['x'] - pp1['x'])**2 + (bp1['y'] - pp1['y'])**2)
    return (d0 - d1) / dt  # positive = closing


def estimate_agent_speed(agent_def: Dict, t: float, dt: float = 0.05) -> float:
    """Estimate agent ground speed (m/s) from trajectory at time t."""
    p0 = interpolate_agent(agent_def, t)
    p1 = interpolate_agent(agent_def, t + dt)
    if not p0 or not p1:
        return 0.0
    dx = p1['x'] - p0['x']
    dy = p1['y'] - p0['y']
    return np.sqrt(dx**2 + dy**2) / dt


# Aliases for readability
estimate_bike_speed = estimate_agent_speed
estimate_ped_speed = estimate_agent_speed


def stopping_distance(speed_ms: float, decel: float = BIKE_DECEL_MS2,
                      reaction_s: float = CYCLIST_REACTION_S) -> float:
    """Compute stopping distance = reaction_distance + braking_distance."""
    return speed_ms * reaction_s + speed_ms**2 / (2.0 * decel)


def time_to_collision(bike_def: Dict, ped_def: Dict, t: float,
                      lookahead: float = 5.0, dt: float = 0.05,
                      conflict_radius: float = CONFLICT_ZONE_HALF_WIDTH_M
                      ) -> Tuple[float, float, float]:
    """Estimate time to collision, minimum future distance, and closing speed.

    Returns:
        ttc: seconds until distance < conflict_radius (inf if no collision).
        d_min: minimum distance within lookahead window.
        v_close: current closing speed (m/s, positive = approaching).
    """
    v_close = estimate_closing_speed(bike_def, ped_def, t)
    d_min = np.inf
    ttc = np.inf

    for t_f in np.arange(t, t + lookahead, dt):
        bp = interpolate_agent(bike_def, t_f)
        pp = interpolate_agent(ped_def, t_f)
        if bp is None or pp is None:
            break
        d = np.sqrt((bp['x'] - pp['x'])**2 + (bp['y'] - pp['y'])**2)
        if d < d_min:
            d_min = d
        if d < conflict_radius and ttc == np.inf:
            ttc = t_f - t

    return ttc, d_min, v_close


@dataclass
class DangerAssessment:
    """Per-frame danger assessment for a bike-pedestrian pair.

    Two-tier danger labeling:
      is_danger:   True if the situation is physically threatening (collision
                   trajectory + insufficient stopping distance or low TTC).
      actionable:  True if is_danger AND the system's warning can still change
                   the outcome (TTC > pedestrian PRT). Frames with is_danger
                   but not actionable are "imminent" — the alert is correct
                   but the pedestrian cannot react in time.

    Sensitivity is computed over actionable frames only.
    Specificity treats all danger frames (actionable + imminent) as
    expected-alert, so alerting during imminent danger is not penalized.
    """
    is_danger: bool
    actionable: bool         # True if warning can still help the pedestrian
    severity: float          # 0-1, based on kinetic energy relative to max
    ttc: float               # time to collision (seconds, inf = no collision)
    d_min_future: float      # minimum future distance within lookahead
    closing_speed: float     # m/s, positive = approaching
    bike_speed: float        # m/s
    stopping_dist: float     # meters required to stop
    warning_budget: float    # seconds of warning before closest approach
    distance: float          # current distance


MAX_SEVERITY_SPEED_MS = 12.0   # ~43 km/h, speed at which severity = 1.0


def get_braking_profile(scenario: Dict) -> 'CyclistBrakingProfile':
    """Return the braking profile for a scenario based on its vehicle tag."""
    key = scenario.get('vehicle', 'bicycle')
    return BRAKING_PROFILES.get(key, DEFAULT_BRAKING_PROFILE)


def compute_ground_truth_danger(scenario: Dict, t: float,
                                convergence_m: float = 5.0,
                                stop_margin: float = 0.8,
                                ttc_threshold_s: Optional[float] = None) -> DangerAssessment:
    """Compute danger using a kinematic safety model.

    Danger is defined by the relationship between stopping distance and
    current distance: if the cyclist cannot stop before reaching the
    pedestrian's conflict zone, the situation is dangerous.

    The severity is proportional to the square of the cyclist's speed,
    normalized to MAX_SEVERITY_SPEED_MS, reflecting the kinetic energy
    of a potential impact.
    """
    null_assessment = DangerAssessment(
        is_danger=False, actionable=False, severity=0.0, ttc=np.inf,
        d_min_future=np.inf, closing_speed=0.0, bike_speed=0.0,
        stopping_dist=0.0, warning_budget=np.inf, distance=np.inf)

    bike_defs = [a for a in scenario['agents'] if a['type'] == 'bike']
    ped_defs = [a for a in scenario['agents'] if a['type'] == 'person']

    if not bike_defs or not ped_defs:
        return null_assessment

    profile = get_braking_profile(scenario)

    worst = null_assessment
    for bike_def in bike_defs:
        bp = interpolate_agent(bike_def, t)
        if bp is None:
            continue
        v_bike = estimate_bike_speed(bike_def, t)
        sd = stopping_distance(v_bike, decel=profile.decel_ms2,
                               reaction_s=profile.prt_s)

        for ped_def in ped_defs:
            pp = interpolate_agent(ped_def, t)
            if pp is None:
                continue

            d_cur = np.sqrt((bp['x'] - pp['x'])**2 + (bp['y'] - pp['y'])**2)
            if d_cur > 25.0:
                continue

            ttc, d_min, v_close = time_to_collision(bike_def, ped_def, t)

            # Estimate actual pedestrian speed for clearance time
            v_ped = estimate_ped_speed(ped_def, t)
            if v_ped > 0.1:
                clearance_time = (2.0 * CONFLICT_ZONE_HALF_WIDTH_M) / v_ped
            else:
                # Stationary or near-stationary: cannot clear the zone
                clearance_time = 30.0  # effectively infinite

            ttc_threshold = PEDESTRIAN_REACTION_S + clearance_time

            gap_to_conflict = max(d_cur - CONFLICT_ZONE_HALF_WIDTH_M, 0.0)

            # Convergence check: the trajectory must bring the pair within
            # a meaningful conflict distance. Parallel paths with large CPA
            # are excluded even if closing speed is positive.
            trajectory_converges = d_min < convergence_m

            if ttc_threshold_s is not None:
                ttc_threshold = ttc_threshold_s

            # Danger condition requires ALL of:
            #   (1) trajectory converges: d_min_future < convergence threshold
            #   (2) bike is closing: v_close > 0.3 m/s
            #   AND at least one of:
            #   (a) stopping distance exceeds gap to conflict zone, OR
            #   (b) TTC < pedestrian reaction + clearance time
            is_danger = False
            if v_close > 0.3 and trajectory_converges:
                if sd > gap_to_conflict * stop_margin:
                    is_danger = True
                if ttc < ttc_threshold:
                    is_danger = True

            # Actionable: the system's warning can still change the outcome.
            # Either the pedestrian reacts (needs PRT to begin moving) or
            # the cyclist swerves (needs PRT + lateral maneuver time).
            # Below ACTIONABLE_TTC_LOWER (~1.55s), neither party can avoid
            # the collision. The alert is still correct (not a false positive)
            # but the system should not be penalized for missing these frames.
            actionable = is_danger and (ttc >= ACTIONABLE_TTC_LOWER)

            severity = min((v_bike / MAX_SEVERITY_SPEED_MS) ** 2, 1.0)

            # Warning budget: how many seconds from now until closest approach
            warning_budget = np.inf
            if d_min < np.inf:
                # Find time of closest approach
                for t_f in np.arange(t, t + 5.0, 0.05):
                    bf = interpolate_agent(bike_def, t_f)
                    pf = interpolate_agent(ped_def, t_f)
                    if bf and pf:
                        d = np.sqrt((bf['x'] - pf['x'])**2 + (bf['y'] - pf['y'])**2)
                        if abs(d - d_min) < 0.05:
                            warning_budget = t_f - t
                            break

            assessment = DangerAssessment(
                is_danger=is_danger, actionable=actionable,
                severity=severity, ttc=ttc,
                d_min_future=d_min, closing_speed=v_close,
                bike_speed=v_bike, stopping_dist=sd,
                warning_budget=warning_budget, distance=d_cur)

            # Keep worst case
            if is_danger and (not worst.is_danger or severity > worst.severity):
                worst = assessment
            elif not worst.is_danger and d_min < worst.d_min_future:
                worst = assessment

    return worst


# ================================================================
# 6. BYTETRACK SIMULATOR
# ================================================================

class TrackedObject:
    """Single tracked object with Kalman-style position prediction.

    Maintains EMA-smoothed velocity and acceleration estimates for
    both first-order (constant velocity) and second-order (constant
    acceleration) prediction.
    """
    __slots__ = ('track_id', 'obj_type', 'x', 'y', 'vx', 'vy',
                 'ax', 'ay', 'frames_since_seen', 'heading')

    def __init__(self, track_id, obj_type, x, y, heading=0.0):
        self.track_id = track_id
        self.obj_type = obj_type
        self.x = x
        self.y = y
        self.vx = 0.0
        self.vy = 0.0
        self.ax = 0.0
        self.ay = 0.0
        self.frames_since_seen = 0
        self.heading = heading

    def update(self, x, y, heading=0.0):
        """Update with a new detection (matched)."""
        new_vx = x - self.x
        new_vy = y - self.y
        # Acceleration: change in velocity (EMA smoothed)
        new_ax = new_vx - self.vx
        new_ay = new_vy - self.vy
        self.ax = 0.5 * self.ax + 0.5 * new_ax
        self.ay = 0.5 * self.ay + 0.5 * new_ay
        # Velocity (EMA smoothed)
        self.vx = 0.5 * self.vx + 0.5 * new_vx
        self.vy = 0.5 * self.vy + 0.5 * new_vy
        self.x = x
        self.y = y
        self.heading = heading
        self.frames_since_seen = 0

    def predict(self):
        """Predict position for a frame with no detection (constant velocity)."""
        self.x += self.vx
        self.y += self.vy
        self.frames_since_seen += 1


class GroundPlaneTracker:
    """Ground-plane nearest-neighbor tracker with constant-velocity prediction.

    Matches detections to existing tracks by metric distance in the BEV
    ground plane, consistent with the deployed GroundPlaneTracker in
    mqtt_bridge.py. Key behaviours:
      - Tracks persist across missed detections for up to track_buffer frames
        using constant-velocity prediction.
      - Detections are matched to existing tracks by greedy nearest-neighbour
        within match_radius (meters).
      - Unmatched detections create new tracks.
      - Tracks exceeding track_buffer lost frames are removed.
    """

    def __init__(self, track_buffer: int = 30, match_radius: float = 3.0):
        self.track_buffer = track_buffer
        self.match_radius = match_radius
        self.tracks: Dict[int, TrackedObject] = {}

    def update(self, detections: List[Dict]) -> List[Dict]:
        """Process one frame of detections, return active tracked agents.

        Args:
            detections: list of {'id': int, 'type': str, 'x': float, 'y': float,
                        'heading': float} for agents detected in this frame.

        Returns:
            list of dicts in the same format, including tracks that were not
            detected but are still within the track_buffer timeout (predicted).
        """
        # Predict all existing tracks forward
        for trk in self.tracks.values():
            trk.predict()

        # Match detections to existing tracks (greedy nearest-neighbour)
        used_tracks = set()
        used_dets = set()

        # Build cost matrix
        det_list = list(detections)
        trk_list = list(self.tracks.values())

        if trk_list and det_list:
            costs = []
            for di, det in enumerate(det_list):
                for ti, trk in enumerate(trk_list):
                    if det['type'] == trk.obj_type:
                        dx = det['x'] - trk.x
                        dy = det['y'] - trk.y
                        dist = np.sqrt(dx * dx + dy * dy)
                        if dist < self.match_radius:
                            costs.append((dist, di, ti))

            # Greedy assignment by distance
            costs.sort()
            for _, di, ti in costs:
                if di not in used_dets and ti not in used_tracks:
                    trk_list[ti].update(
                        det_list[di]['x'], det_list[di]['y'],
                        det_list[di].get('heading', 0.0))
                    used_tracks.add(ti)
                    used_dets.add(di)

        # Unmatched detections → new tracks
        for di, det in enumerate(det_list):
            if di not in used_dets:
                tid = det['id']
                self.tracks[tid] = TrackedObject(
                    tid, det['type'], det['x'], det['y'],
                    det.get('heading', 0.0))

        # Remove tracks that have been lost too long
        to_remove = [tid for tid, trk in self.tracks.items()
                     if trk.frames_since_seen > self.track_buffer]
        for tid in to_remove:
            del self.tracks[tid]

        # Build output: all active tracks
        output = []
        for trk in self.tracks.values():
            output.append({
                'id': trk.track_id,
                'type': trk.obj_type,
                'x': trk.x,
                'y': trk.y,
                'vx': trk.vx,
                'vy': trk.vy,
                'ax': trk.ax,
                'ay': trk.ay,
                'heading': trk.heading,
            })

        return output

    def forecast(self, agents: List[Dict], n_frames: int,
                 order: int = 1) -> List[Dict]:
        """Predict agent positions n_frames into the future.

        Args:
            agents: tracked agent dicts with vx, vy, ax, ay.
            n_frames: number of frames to predict forward.
            order: 1 = constant velocity, 2 = constant acceleration.

        Both orders run in O(N) time for N tracks. The second-order
        forecast adds one multiply-accumulate per track per axis —
        negligible on the Orin.
        """
        forecasted = []
        for a in agents:
            vx = a.get('vx', 0.0)
            vy = a.get('vy', 0.0)
            n = n_frames
            if order >= 2:
                ax = a.get('ax', 0.0)
                ay = a.get('ay', 0.0)
                # x(t+n) = x + v*n + 0.5*a*n^2
                px = a['x'] + vx * n + 0.5 * ax * n * n
                py = a['y'] + vy * n + 0.5 * ay * n * n
            else:
                px = a['x'] + vx * n
                py = a['y'] + vy * n
            forecasted.append({
                'id': a['id'],
                'type': a['type'],
                'x': px,
                'y': py,
                'heading': a.get('heading', 0.0),
            })
        return forecasted


# ================================================================
# 7. SCENARIO SIMULATION
# ================================================================

def simulate_scenario(scenario: Dict, params: DecisionParams,
                      fps: int = 30,
                      cam: Optional[FisheyeCamera] = None,
                      add_bbox_noise: bool = False,
                      detection_ar: Optional[float] = None,
                      rng: Optional[np.random.Generator] = None,
                      use_tracker: bool = True,
                      camera_delay_frames: int = 0,
                      forecast_order: int = 0,
                      uncertainty: Optional['UncertaintyParams'] = None,
                      failure_model: Optional['DetectionFailureModel'] = None) -> Dict:
    """Run a scenario through the decision pipeline and record state history.

    Args:
        scenario: scenario definition with agents and paths.
        params: decision pipeline parameters.
        fps: simulation frame rate.
        cam: fisheye camera for BBox noise injection.
        add_bbox_noise: whether to perturb positions by fisheye BBox error.
        detection_ar: if set, stochastically drop detections with P(miss) = 1 - ar.
        rng: random generator for stochastic detection.
        use_tracker: if True, simulate ByteTrack persistence across missed
                     detections (default True).
        camera_delay_frames: number of frames of camera/pipeline latency.
                     The decision pipeline sees positions from this many frames
                     ago, while ground truth is evaluated at the current time.
                     At 30 fps, 1 frame = 33 ms.
        forecast_order: 0 = no forecast, 1 = constant velocity,
                     2 = constant acceleration. Applied when camera_delay_frames > 0.

    Returns dict with time series of states and comprehensive safety metrics.
    """
    profile = get_braking_profile(scenario)
    pipeline = make_pipeline(params, fps, profile=profile)
    duration = get_scenario_duration(scenario)
    dt = 1.0 / fps

    # Camera offset for coordinate transform (world → camera-relative)
    _cam_x = cam.cam_x if cam is not None else 0.0
    _cam_y = cam.cam_y if cam is not None else 0.0

    tracker = GroundPlaneTracker(track_buffer=fps) if use_tracker else None

    # Detection failure model (default: i.i.d.)
    _failure_model = failure_model if failure_model is not None else IIDFailureModel()
    if hasattr(_failure_model, 'reset'):
        _failure_model.reset()

    # Ring buffer for camera latency: stores detection lists
    delay_buffer = deque(maxlen=max(camera_delay_frames + 1, 1))

    times = []
    states = []
    assessments = []

    t = 0.0
    # Uncertainty params for size-aware detection drops (use provided or default)
    _unc = uncertainty if uncertainty is not None else UncertaintyParams()

    while t <= duration:
        detections = []
        for ag_def in scenario['agents']:
            pos = interpolate_agent(ag_def, t)
            if pos is not None:
                # Stochastic detection drop, size-aware when camera is available
                if detection_ar is not None and rng is not None:
                    ar = detection_ar
                    if cam is not None:
                        obj_name = 'pedestrian' if pos['type'] == 'person' else 'cyclist'
                        obj = OBJECT_TYPES.get(obj_name)
                        if obj:
                            hdg = pos.get('heading', 0.0)
                            bbox_area = cam.projected_bbox_area(pos['x'], pos['y'], obj, hdg)
                            if bbox_area > 0:
                                sized = _unc.for_class(obj_name)
                                ar = sized.for_area(bbox_area).ar
                    if _failure_model.should_drop(pos['id'], ar, rng):
                        continue  # missed detection

                if cam is not None and add_bbox_noise:
                    obj_name = 'pedestrian' if pos['type'] == 'person' else 'cyclist'
                    obj = OBJECT_TYPES.get(obj_name)
                    if obj:
                        err = compute_bbox_error(cam, pos['x'], pos['y'], obj,
                                                pos.get('heading', 0.0))
                        if err is not None:
                            pos['x'] = err['est_x']
                            pos['y'] = err['est_y']
                detections.append(pos)

        # Camera latency: push current detections, pop delayed ones
        delay_buffer.append(detections)
        if len(delay_buffer) > camera_delay_frames:
            delayed_detections = delay_buffer[0]
        else:
            delayed_detections = []  # not enough history yet

        # ByteTrack: persist tracks across missed detections
        if tracker is not None:
            agents = tracker.update(delayed_detections)
            # Forecast positions forward to compensate for camera delay
            if forecast_order > 0 and camera_delay_frames > 0:
                agents = tracker.forecast(agents, camera_delay_frames,
                                          order=forecast_order)
        else:
            agents = delayed_detections

        state = run_pipeline_on_agents(pipeline, agents, t,
                                       cam_x=_cam_x, cam_y=_cam_y)
        assessment = compute_ground_truth_danger(scenario, t)

        times.append(t)
        states.append(state)
        assessments.append(assessment)
        t += dt

    times = np.array(times)
    states = np.array(states)
    dangers = np.array([a.is_danger for a in assessments])
    actionable = np.array([a.actionable for a in assessments])
    severities = np.array([a.severity for a in assessments])

    n_total = len(times)
    is_alert = states == 'alert'

    # Two-tier metrics:
    # - Sensitivity uses only ACTIONABLE danger frames (system could have helped)
    # - Specificity: danger frames (actionable + imminent) are expected-alert,
    #   so alerting during any danger is not a false positive.
    #   FP = alert on safe frames only. TN = silent on safe frames.
    n_actionable = int(np.sum(actionable))
    n_safe = int(np.sum(~dangers))  # truly safe frames (no danger of any tier)

    true_pos = int(np.sum(is_alert & actionable))
    false_neg = int(np.sum(~is_alert & actionable))
    false_pos = int(np.sum(is_alert & ~dangers))  # alert on safe frames only
    true_neg = int(np.sum(~is_alert & ~dangers))

    # Severity-weighted false negative rate (over actionable frames only)
    sev_fn = float(np.sum(severities[~is_alert & actionable]))
    sev_total = float(np.sum(severities[actionable])) if n_actionable > 0 else 1.0
    sev_weighted_fn_rate = sev_fn / max(sev_total, 1e-6)

    # Alert latency: time from first actionable danger to first alert
    alert_latency = np.nan
    if n_actionable > 0:
        first_danger_idx = int(np.argmax(actionable))
        subsequent_alerts = is_alert[first_danger_idx:]
        if np.any(subsequent_alerts):
            latency_frames = int(np.argmax(subsequent_alerts))
            alert_latency = latency_frames * dt

    # Warning budget: time from first alert onset to closest approach.
    # Measured at each alert-onset transition (non-alert -> alert).
    # The warning_budget field in the assessment gives the time from the
    # current frame to closest approach, regardless of danger state.
    warning_budgets = []
    if np.any(is_alert):
        prev_alert = False
        for i, (a, s) in enumerate(zip(assessments, states)):
            is_a = (s == 'alert')
            if is_a and not prev_alert and a.warning_budget < np.inf:
                warning_budgets.append(a.warning_budget)
            prev_alert = is_a

    min_warning_budget = min(warning_budgets) if warning_budgets else np.nan
    mean_warning_budget = float(np.mean(warning_budgets)) if warning_budgets else np.nan

    # Reaction time adequacy: fraction of alert onsets with enough
    # warning budget for the pedestrian to react (1.87s distracted, Fugger et al.).
    adequate_warnings = sum(1 for w in warning_budgets if w > PEDESTRIAN_REACTION_S)
    reaction_adequacy = adequate_warnings / max(len(warning_budgets), 1)

    # Peak severity encountered
    peak_severity = float(np.max(severities)) if len(severities) > 0 else 0.0
    peak_bike_speed = max((a.bike_speed for a in assessments), default=0.0)

    # Alert fatigue: fraction of total simulation time spent in alert state
    alert_fraction = float(np.sum(is_alert)) / max(n_total, 1)

    return {
        'scenario': scenario['name'],
        'times': times,
        'states': states,
        'dangers': dangers,
        'assessments': assessments,
        'true_pos': true_pos,
        'false_pos': false_pos,
        'false_neg': false_neg,
        'true_neg': true_neg,
        'sensitivity': true_pos / max(n_actionable, 1),
        'specificity': true_neg / max(n_safe, 1),
        'sev_weighted_fn_rate': sev_weighted_fn_rate,
        'alert_latency_s': alert_latency,
        'min_warning_budget_s': min_warning_budget,
        'mean_warning_budget_s': mean_warning_budget,
        'reaction_adequacy': reaction_adequacy,
        'peak_severity': peak_severity,
        'peak_bike_speed_ms': peak_bike_speed,
        'alert_fraction': alert_fraction,
    }


def run_all_scenarios(params: DecisionParams, fps: int = 30,
                      cam: Optional[FisheyeCamera] = None,
                      add_bbox_noise: bool = False,
                      verbose: bool = True,
                      camera_delay_frames: int = 0,
                      forecast_order: int = 0,
                      uncertainty: Optional['UncertaintyParams'] = None) -> List[Dict]:
    """Run all scenarios and return results."""
    results = []
    for scenario in SCENARIOS:
        result = simulate_scenario(scenario, params, fps, cam, add_bbox_noise,
                                   camera_delay_frames=camera_delay_frames,
                                   forecast_order=forecast_order,
                                   uncertainty=uncertainty)
        results.append(result)

    if verbose:
        print(f"\n{'=' * 120}")
        print(f"  SCENARIO RESULTS -- {params_to_str(params)}")
        print(f"{'=' * 120}")
        print(f"  {'Scenario':<28} {'TP':>4} {'FP':>4} {'FN':>4} {'TN':>4} "
              f"{'Sens':>6} {'Spec':>6} {'SevFN':>6} {'Lat':>6} "
              f"{'WarnBgt':>7} {'RxnAdq':>6} {'Fatigue':>7}  {'Exp':>8}")
        print(f"  {'-' * 116}")
        for r in results:
            scn = next(s for s in SCENARIOS if s['name'] == r['scenario'])
            lat = f"{r['alert_latency_s']:.2f}" if not np.isnan(r['alert_latency_s']) else "N/A"
            wb = f"{r['min_warning_budget_s']:.1f}s" if not np.isnan(r['min_warning_budget_s']) else "N/A"
            ra = f"{r['reaction_adequacy']:.0%}" if not np.isnan(r['mean_warning_budget_s']) else "N/A"
            print(f"  {r['scenario']:<28} {r['true_pos']:>4} {r['false_pos']:>4} "
                  f"{r['false_neg']:>4} {r['true_neg']:>4} "
                  f"{r['sensitivity']:>5.1%} {r['specificity']:>5.1%} "
                  f"{r['sev_weighted_fn_rate']:>5.1%} "
                  f"{lat:>6} {wb:>7} {ra:>6} {r['alert_fraction']:>6.1%}"
                  f"  {scn['expected_dominant']:>8}")

        # Aggregate
        total_tp = sum(r['true_pos'] for r in results)
        total_fp = sum(r['false_pos'] for r in results)
        total_fn = sum(r['false_neg'] for r in results)
        total_tn = sum(r['true_neg'] for r in results)
        total_danger = total_tp + total_fn
        total_safe = total_tn + total_fp
        avg_sev_fn = np.mean([r['sev_weighted_fn_rate'] for r in results
                              if r['true_pos'] + r['false_neg'] > 0])
        avg_fatigue = np.mean([r['alert_fraction'] for r in results])
        print(f"  {'-' * 116}")
        print(f"  {'TOTAL':<28} {total_tp:>4} {total_fp:>4} {total_fn:>4} {total_tn:>4} "
              f"{total_tp / max(total_danger, 1):>5.1%} "
              f"{total_tn / max(total_safe, 1):>5.1%} "
              f"{avg_sev_fn:>5.1%}")
        print(f"\n  Alert fatigue (mean fraction of time in alert): {avg_fatigue:.1%}")

    return results


def run_monte_carlo(params: DecisionParams, n_trials: int = 50,
                    fps: int = 30, cam: Optional[FisheyeCamera] = None,
                    detection_ar: float = 0.747,
                    add_bbox_noise: bool = False,
                    uncertainty: Optional['UncertaintyParams'] = None,
                    failure_model: Optional['DetectionFailureModel'] = None) -> Dict:
    """Run Monte Carlo simulation with stochastic detection failures.

    Each trial randomly drops detections based on the average recall rate.
    When a camera is provided, the drop rate is size-aware: objects that
    are small on the sensor are dropped at the per-size AR from the eval
    cache, so distant objects are missed more frequently than close ones.
    """
    size_note = "size-aware" if cam is not None else "flat"
    print(f"\n{'=' * 80}")
    print(f"  MONTE CARLO SIMULATION ({n_trials} trials, AR={detection_ar:.3f} [{size_note}])")
    print(f"  Pipeline: {params_to_str(params)}")
    print(f"{'=' * 80}")

    # Per-scenario stats for reporting
    per_scenario = {s['name']: {'sens': [], 'spec': [], 'sev_fn': []}
                    for s in SCENARIOS}
    # Frame-level accumulators per trial (for proper aggregate metrics)
    trial_sens = []    # one frame-level sensitivity per trial
    trial_spec = []
    trial_sev_fn = []
    all_latencies = []

    for trial in range(n_trials):
        rng = np.random.default_rng(seed=trial)
        trial_tp = trial_fp = trial_fn = trial_tn = 0
        trial_sev_fn_num = 0.0
        trial_sev_total = 0.0
        for scenario in SCENARIOS:
            r = simulate_scenario(
                scenario, params, fps, cam,
                add_bbox_noise=add_bbox_noise,
                detection_ar=detection_ar, rng=rng,
                uncertainty=uncertainty,
                failure_model=failure_model)
            per_scenario[r['scenario']]['sens'].append(r['sensitivity'])
            per_scenario[r['scenario']]['spec'].append(r['specificity'])
            per_scenario[r['scenario']]['sev_fn'].append(r['sev_weighted_fn_rate'])
            # Accumulate frame counts for this trial
            trial_tp += r['true_pos']
            trial_fp += r['false_pos']
            trial_fn += r['false_neg']
            trial_tn += r['true_neg']
            trial_sev_fn_num += r['sev_weighted_fn_rate'] * (r['true_pos'] + r['false_neg'])
            trial_sev_total += r['true_pos'] + r['false_neg']
            if not np.isnan(r['alert_latency_s']):
                all_latencies.append(r['alert_latency_s'])

        # Frame-level aggregate for this trial
        trial_danger = trial_tp + trial_fn
        trial_safe = trial_tn + trial_fp
        trial_sens.append(trial_tp / max(trial_danger, 1))
        trial_spec.append(trial_tn / max(trial_safe, 1))
        trial_sev_fn.append(trial_sev_fn_num / max(trial_sev_total, 1e-6))

    print(f"\n  {'Scenario':<28} {'Sens (mean+/-std)':>18} {'Spec (mean+/-std)':>18} "
          f"{'SevFN (mean+/-std)':>19}")
    print(f"  {'-' * 87}")
    for name, data in per_scenario.items():
        s_arr = np.array(data['sens'])
        sp_arr = np.array(data['spec'])
        sf_arr = np.array(data['sev_fn'])
        print(f"  {name:<28} {np.mean(s_arr):>6.1%} +/- {np.std(s_arr):>5.1%}"
              f"   {np.mean(sp_arr):>6.1%} +/- {np.std(sp_arr):>5.1%}"
              f"   {np.mean(sf_arr):>6.1%} +/- {np.std(sf_arr):>5.1%}")

    ts = np.array(trial_sens)
    tsp = np.array(trial_spec)
    tsf = np.array(trial_sev_fn)
    print(f"\n  Frame-level aggregate over all scenarios and trials:")
    print(f"    Sensitivity:       {np.mean(ts):.1%} +/- {np.std(ts):.1%}")
    print(f"    Specificity:       {np.mean(tsp):.1%} +/- {np.std(tsp):.1%}")
    print(f"    Sev-weighted FN:   {np.mean(tsf):.1%} +/- {np.std(tsf):.1%}")
    if all_latencies:
        print(f"    Alert latency:     {np.mean(all_latencies):.2f}s +/- {np.std(all_latencies):.2f}s")
    print(f"    Worst-case sens:   {np.min(ts):.1%}")
    print(f"    Worst-case spec:   {np.min(tsp):.1%}")

    return {
        'mean_sens': float(np.mean(ts)),
        'std_sens': float(np.std(ts)),
        'mean_spec': float(np.mean(tsp)),
        'worst_sens': float(np.min(ts)),
    }


def params_to_str(p: DecisionParams) -> str:
    return (f"N={p.bike_memory_frames}, d=[{p.prox_min:.1f},{p.prox_max:.1f}]m, "
            f"v_min={p.prox_speed_min:.3f}, k={p.lookback_frames}")


# ================================================================
# 7. OPTIMIZER
# ================================================================

def optimization_cost(x: np.ndarray, fps: int = 30,
                      w_fn: float = 5.0, w_fp: float = 1.0,
                      w_lat: float = 2.0,
                      cam: Optional[FisheyeCamera] = None,
                      add_bbox_noise: bool = False) -> float:
    """Cost function for decision parameter optimization.

    The cost function balances five objectives:
      1. Severity-weighted false negative rate (missed high-speed threats cost more)
      2. False positive rate (alert fatigue erodes trust)
      3. Alert latency (earlier warnings save lives)
      4. Reaction adequacy penalty (alerts too late for pedestrian reaction)
      5. Alert fatigue penalty (systems that alert too often get ignored)

    Args:
        x: parameter vector [bike_memory_frames, prox_min, prox_max, prox_speed_min, lookback].
        w_fn: weight for severity-weighted false negatives.
        w_fp: weight for false positives.
        w_lat: weight for alert latency.

    Returns:
        Scalar cost (lower is better).
    """
    bike_mem = int(round(x[0]))
    prox_min = x[1]
    prox_max = x[2]
    prox_speed_min = x[3]
    lookback = int(round(x[4]))

    if prox_min >= prox_max:
        return 1e6

    params = DecisionParams(
        bike_memory_frames=bike_mem,
        prox_min=prox_min,
        prox_max=prox_max,
        prox_speed_min=prox_speed_min,
        lookback_frames=lookback,
    )

    results = run_all_scenarios(params, fps, cam, add_bbox_noise=add_bbox_noise,
                                verbose=False)

    # Severity-weighted false negative rate
    sev_fn_rates = [r['sev_weighted_fn_rate'] for r in results
                    if r['true_pos'] + r['false_neg'] > 0]
    sev_fn = np.mean(sev_fn_rates) if sev_fn_rates else 0.0

    # Standard false positive rate
    total_fp = sum(r['false_pos'] for r in results)
    total_safe = sum(r['true_neg'] + r['false_pos'] for r in results)
    fp_rate = total_fp / max(total_safe, 1)

    # Alert latency
    latencies = [r['alert_latency_s'] for r in results
                 if not np.isnan(r['alert_latency_s'])]
    avg_latency = np.mean(latencies) if latencies else 5.0

    # Reaction adequacy penalty: fraction of alerts that come too late
    rxn_rates = [r['reaction_adequacy'] for r in results
                 if not np.isnan(r['mean_warning_budget_s'])]
    rxn_inadequacy = 1.0 - (np.mean(rxn_rates) if rxn_rates else 0.0)

    # Alert fatigue: penalize if average alert fraction > 20% of scenario time
    avg_fatigue = np.mean([r['alert_fraction'] for r in results])
    fatigue_penalty = max(0.0, avg_fatigue - 0.20)

    # Direct sensitivity penalty: heavily penalize low sensitivity
    total_tp = sum(r['true_pos'] for r in results)
    total_fn = sum(r['false_neg'] for r in results)
    sens = total_tp / max(total_tp + total_fn, 1)
    sens_penalty = max(0.0, 0.90 - sens) * 20.0  # 20x cost per pp below 90%

    cost = (w_fn * sev_fn
            + w_fp * fp_rate
            + w_lat * avg_latency
            + 3.0 * rxn_inadequacy
            + 1.0 * fatigue_penalty
            + sens_penalty)
    return cost


def optimize_parameters(fps: int = 30, cam: Optional[FisheyeCamera] = None,
                        w_fn: float = 5.0, w_fp: float = 1.0,
                        w_lat: float = 2.0,
                        add_bbox_noise: bool = False) -> DecisionParams:
    """Find optimal decision parameters using differential evolution.

    Parameter bounds:
        bike_memory_frames: [0, 120]
        prox_min: [0.5, 5.0] m
        prox_max: [5.0, 25.0] m
        prox_speed_min: [0.01, 0.5] m/frame
        lookback_frames: [1, 10]
    """
    bounds = [
        (0, 120),       # bike_memory_frames
        (0.5, 5.0),     # prox_min
        (5.0, 25.0),    # prox_max
        (0.01, 0.5),    # prox_speed_min
        (1, 10),        # lookback_frames
    ]

    print("\n" + "=" * 70)
    print("  OPTIMIZING DECISION PARAMETERS")
    print(f"  Weights: w_fn={w_fn}, w_fp={w_fp}, w_lat={w_lat}")
    print(f"  BBox noise: {'enabled' if add_bbox_noise else 'disabled'}")
    print("=" * 70)

    # Seed the initial population with the current best params plus
    # variations, so the optimizer starts from a known-good region
    # rather than exploring blindly.
    current_best = [58, 1.9, 24.8, 0.147, 2]  # current default
    n_dims = len(bounds)
    init_pop = [current_best]
    rng_init = np.random.default_rng(42)
    # 10 tight perturbations around current best
    for _ in range(10):
        vec = []
        for d, (lo, hi) in enumerate(bounds):
            spread = (hi - lo) * 0.1
            val = current_best[d] + rng_init.uniform(-spread, spread)
            vec.append(np.clip(val, lo, hi))
        init_pop.append(vec)
    # 19 wider perturbations for diversity
    for _ in range(19):
        vec = []
        for d, (lo, hi) in enumerate(bounds):
            spread = (hi - lo) * 0.4
            val = current_best[d] + rng_init.uniform(-spread, spread)
            vec.append(np.clip(val, lo, hi))
        init_pop.append(vec)
    init_pop = np.array(init_pop)

    result = differential_evolution(
        optimization_cost,
        bounds=bounds,
        args=(fps, w_fn, w_fp, w_lat, cam, add_bbox_noise),
        seed=42,
        maxiter=100,
        popsize=30,
        tol=1e-4,
        mutation=(0.5, 1.5),
        recombination=0.9,
        strategy='best1bin',
        init=init_pop,
        disp=True,
        integrality=[True, False, False, False, True],
        workers=-1,  # use all CPU cores
        updating='deferred',  # required for parallel workers
    )

    optimal = DecisionParams(
        bike_memory_frames=int(round(result.x[0])),
        prox_min=result.x[1],
        prox_max=result.x[2],
        prox_speed_min=result.x[3],
        lookback_frames=int(round(result.x[4])),
    )

    print(f"\n  Optimal parameters: {params_to_str(optimal)}")
    print(f"  Cost: {result.fun:.4f}")
    print(f"  Converged: {result.success}")

    return optimal


# ================================================================
# 8. COMPARISON OF PIPELINE CONFIGURATIONS
# ================================================================

def compare_pipelines(cam: Optional[FisheyeCamera] = None,
                      add_bbox_noise: bool = False,
                      uncertainty: Optional['UncertaintyParams'] = None):
    """Run all scenarios under each named pipeline configuration and compare."""
    print("\n" + "=" * 70)
    print("  PIPELINE COMPARISON")
    print("=" * 70)

    for name, params in PIPELINE_CONFIGS.items():
        print(f"\n--- {name} ---")
        run_all_scenarios(params, cam=cam, add_bbox_noise=add_bbox_noise,
                          uncertainty=uncertainty)


# ================================================================
# 9. CAMERA LATENCY ANALYSIS
# ================================================================

def run_latency_sweep(params: DecisionParams, fps: int = 30,
                      cam: Optional[FisheyeCamera] = None,
                      max_delay_ms: int = 500, step_ms: int = 33,
                      save_figure: bool = True,
                      uncertainty: Optional['UncertaintyParams'] = None,
                      add_bbox_noise: bool = True):
    """Sweep camera latency from 0 to max_delay_ms and plot the effect.

    Each step is one frame (33 ms at 30 fps). At each latency setting, runs
    all scenarios deterministically and records aggregate sensitivity,
    specificity, and severity-weighted FN rate.
    """
    delays_ms = list(range(0, max_delay_ms + 1, step_ms))
    delays_frames = [round(d / 1000.0 * fps) for d in delays_ms]

    def _sweep(forecast_order, label):
        sens_l, sev_fn_l, lat_l, rxn_l, warn_l = [], [], [], [], []
        print(f"\n  --- {label} ---")
        print(f"  {'Delay':>8}  {'Frames':>6}  {'Sens':>7}  "
              f"{'SevFN':>7}  {'AvgLat':>7}  {'RxnAdq':>7}  {'WarnBgt':>7}")
        print(f"  {'-' * 62}")

        for delay_ms, delay_f in zip(delays_ms, delays_frames):
            results = run_all_scenarios(params, fps, cam, verbose=False,
                                        add_bbox_noise=add_bbox_noise,
                                        camera_delay_frames=delay_f,
                                        forecast_order=forecast_order,
                                        uncertainty=uncertainty)

            total_tp = sum(r['true_pos'] for r in results)
            total_fn = sum(r['false_neg'] for r in results)
            total_danger = total_tp + total_fn

            sens = total_tp / max(total_danger, 1)
            sev_fn_rates = [r['sev_weighted_fn_rate'] for r in results
                            if r['true_pos'] + r['false_neg'] > 0]
            sev_fn = np.mean(sev_fn_rates) if sev_fn_rates else 0.0
            lats = [r['alert_latency_s'] for r in results
                    if not np.isnan(r['alert_latency_s'])]
            avg_lat = np.mean(lats) if lats else float('nan')
            rxn_rates = [r['reaction_adequacy'] for r in results
                         if not np.isnan(r['mean_warning_budget_s'])]
            rxn_adq = np.mean(rxn_rates) if rxn_rates else float('nan')
            warn_budgets = [r['mean_warning_budget_s'] for r in results
                            if not np.isnan(r['mean_warning_budget_s'])]
            avg_warn = np.mean(warn_budgets) if warn_budgets else float('nan')

            sens_l.append(sens)
            sev_fn_l.append(sev_fn)
            lat_l.append(avg_lat)
            rxn_l.append(rxn_adq)
            warn_l.append(avg_warn)

            print(f"  {delay_ms:>6} ms  {delay_f:>6}  {sens:>6.1%}  "
                  f"{sev_fn:>6.1%}  {avg_lat:>6.2f}s  {rxn_adq:>6.0%}  "
                  f"{avg_warn:>6.2f}s")

        return sens_l, sev_fn_l, lat_l, rxn_l, warn_l

    print(f"\n{'=' * 80}")
    print(f"  CAMERA LATENCY SWEEP (0-{max_delay_ms} ms, step={step_ms} ms)")
    print(f"{'=' * 80}")

    sens_base, sev_base, lat_base, rxn_base, warn_base = \
        _sweep(0, "Raw observations (no prediction)")
    sens_fc1, sev_fc1, lat_fc1, rxn_fc1, warn_fc1 = \
        _sweep(1, "First-order kinematic prediction")
    sens_fc2, sev_fc2, lat_fc2, rxn_fc2, warn_fc2 = \
        _sweep(2, "Second-order kinematic prediction")

    if save_figure:
        try:
            import matplotlib
            matplotlib.use('Agg')
            import matplotlib.pyplot as plt

            fig, axes = plt.subplots(2, 2, figsize=(10, 7))

            def _triple(ax, base, fc1, fc2, ylabel, ylim=None):
                ax.plot(delays_ms, base, 'o-', color='#2196F3',
                        markersize=3, label='Raw observations')
                ax.plot(delays_ms, fc1, 's--', color='#FF9800',
                        markersize=3, label='1st-order (const. velocity)')
                ax.plot(delays_ms, fc2, '^:', color='#4CAF50',
                        markersize=3, label='2nd-order (const. accel.)')
                ax.set_ylabel(ylabel)
                ax.set_xlabel('Camera Latency (ms)')
                ax.grid(True, alpha=0.3)
                ax.legend(fontsize=6)
                if ylim:
                    ax.set_ylim(ylim)

            _triple(axes[0, 0],
                    [s * 100 for s in sens_base],
                    [s * 100 for s in sens_fc1],
                    [s * 100 for s in sens_fc2],
                    'Sensitivity (%)', (50, 100))
            axes[0, 0].set_title('(a) Sensitivity', fontsize=9)

            _triple(axes[0, 1],
                    [s * 100 for s in sev_base],
                    [s * 100 for s in sev_fc1],
                    [s * 100 for s in sev_fc2],
                    'Sev-Weighted FN (%)', (0, 30))
            axes[0, 1].set_title('(b) Severity-Weighted FN', fontsize=9)

            # (c) Alert Latency with literature thresholds
            all_lats = lat_base + lat_fc1 + lat_fc2
            lat_ymax = max(all_lats) * 1.8
            lat_ymax = max(lat_ymax, 1.5)
            _triple(axes[1, 0], lat_base, lat_fc1, lat_fc2,
                    'Alert Latency (s)', (0, lat_ymax))
            axes[1, 0].set_title('(c) Alert Latency', fontsize=9)
            axes[1, 0].axhline(0.9, color='#9C27B0', linewidth=0.8,
                               linestyle='--', alpha=0.6)
            axes[1, 0].annotate('Mean cyclist PRT (0.9 s)', xy=(500, 0.9),
                                fontsize=6, color='#9C27B0',
                                ha='right', va='bottom')
            axes[1, 0].axhspan(1.3, lat_ymax, color='#F44336', alpha=0.06)
            axes[1, 0].axhline(1.3, color='#F44336', linewidth=0.8,
                               linestyle=':', alpha=0.5)
            axes[1, 0].annotate('85th pctl PRT (1.3 s)', xy=(500, 1.3),
                                fontsize=6, color='#F44336',
                                ha='right', va='bottom')

            # (d) Warning Budget with literature thresholds
            all_warns = warn_base + warn_fc1 + warn_fc2
            warn_ymax = max(all_warns) * 1.3
            warn_ymax = max(warn_ymax, 2.2)
            _triple(axes[1, 1], warn_base, warn_fc1, warn_fc2,
                    'Warning Budget (s)', (0, warn_ymax))
            axes[1, 1].set_title('(d) Warning Budget', fontsize=9)
            axes[1, 1].axhline(1.87, color='#E65100', linewidth=0.8,
                               linestyle='--', alpha=0.6)
            axes[1, 1].annotate('Distracted ped. PRT (1.87 s)', xy=(500, 1.87),
                                fontsize=6, color='#E65100',
                                ha='right', va='bottom')
            axes[1, 1].axhline(0.84, color='#9C27B0', linewidth=0.8,
                               linestyle='--', alpha=0.6)
            axes[1, 1].annotate('Attentive ped. PRT (0.84 s)', xy=(500, 0.84),
                                fontsize=6, color='#9C27B0',
                                ha='right', va='bottom')
            axes[1, 1].axhspan(0, 0.84, color='#F44336', alpha=0.06)

            plt.tight_layout()

            for fmt in ('pdf', 'eps'):
                out_path = f'paper/fig_camera_latency.{fmt}'
                fig.savefig(out_path, bbox_inches='tight', dpi=300)
            plt.close(fig)
            print(f"\n  Figure saved to: paper/fig_camera_latency.pdf/.eps")
        except ImportError:
            print("  (matplotlib not available, skipping figure)")

    # Save data as JSON
    data = {
        'delays_ms': delays_ms,
        'delays_frames': delays_frames,
        'fps': fps,
        'raw': {
            'sensitivity': sens_base, 'sev_weighted_fn': sev_base,
            'alert_latency_s': lat_base, 'reaction_adequacy': rxn_base,
            'mean_warning_budget_s': warn_base,
        },
        'first_order': {
            'sensitivity': sens_fc1, 'sev_weighted_fn': sev_fc1,
            'alert_latency_s': lat_fc1, 'reaction_adequacy': rxn_fc1,
            'mean_warning_budget_s': warn_fc1,
        },
        'second_order': {
            'sensitivity': sens_fc2, 'sev_weighted_fn': sev_fc2,
            'alert_latency_s': lat_fc2, 'reaction_adequacy': rxn_fc2,
            'mean_warning_budget_s': warn_fc2,
        },
    }
    with open('latency_sweep.json', 'w') as f:
        json.dump(data, f, indent=2)
    print(f"  Data saved to: latency_sweep.json")

    return data


# ================================================================
# 10. MAIN
# ================================================================

def main():
    parser = argparse.ArgumentParser(description='Decision Testbench')
    parser.add_argument('--optimize', action='store_true',
                        help='Optimize decision parameters')
    parser.add_argument('--error-map', action='store_true',
                        help='Print BBox error analysis')
    parser.add_argument('--compare', action='store_true',
                        help='Compare pipeline configurations')
    parser.add_argument('--monte-carlo', action='store_true',
                        help='Run Monte Carlo with stochastic detection failures')
    parser.add_argument('--mc-trials', type=int, default=50,
                        help='Number of Monte Carlo trials (default: 50)')
    parser.add_argument('--failure-model', type=str, default='iid',
                        choices=list(FAILURE_MODELS.keys()),
                        help='Detection failure model: iid (default) or gilbert')
    parser.add_argument('--burst-length', type=float, default=3.0,
                        help='Mean burst length for Gilbert model (default: 3.0 frames)')
    parser.add_argument('--bbox-noise', action='store_true', default=True,
                        help='Include fisheye BBox localization error (default: on)')
    parser.add_argument('--no-bbox-noise', action='store_true',
                        help='Disable fisheye BBox localization error')
    parser.add_argument('--w-fn', type=float, default=5.0,
                        help='Severity-weighted FN weight (default: 5.0)')
    parser.add_argument('--w-fp', type=float, default=1.0,
                        help='False positive weight (default: 1.0)')
    parser.add_argument('--w-lat', type=float, default=2.0,
                        help='Latency weight (default: 2.0)')
    parser.add_argument('--latency-sweep', action='store_true',
                        help='Sweep camera latency 0–500 ms and generate figure')
    parser.add_argument('--eval-cache', type=str, default=None,
                        help='Path to eval_cache.json from eval_models.py')
    parser.add_argument('--eval-model', type=str, default=None,
                        help='Model name in eval cache to use (default: first)')
    args = parser.parse_args()
    # --no-bbox-noise overrides the default-on --bbox-noise
    if args.no_bbox_noise:
        args.bbox_noise = False

    # Construct detection failure model
    if args.failure_model == 'gilbert':
        fm = GilbertFailureModel(mean_burst_length=args.burst_length)
        print(f"Detection failure model: Gilbert (mean burst = {args.burst_length:.1f} frames)")
    else:
        fm = IIDFailureModel()

    # Load deployment config (pitch, height) from config.yaml — same source as mqtt_bridge
    try:
        import yaml
        with open('config.yaml') as f:
            _cfg = yaml.safe_load(f)
    except (FileNotFoundError, ImportError):
        _cfg = {}
    _cam_cfg = _cfg.get('camera', {})

    calib = load_camera_calibration()
    if calib:
        cam = FisheyeCamera(
            height=_cam_cfg.get('height', 3.048),
            fov_deg=calib.get('fov_deg', 197.90),
            radius_px=int(calib.get('radius_px', 1750)),
            frame_width=int(calib.get('frame_width', 3500)),
            frame_height=int(calib.get('frame_height', 3500)),
            pad_size=int(calib.get('pad_size', 3840)),
            pitch_deg=_cam_cfg.get('pitch_deg', 0.0),
            cx_override=calib.get('cx'),
            cy_override=calib.get('cy'),
            k1=float(calib.get('k1', 0.0)),
            k2=float(calib.get('k2', 0.0)),
        )
        print(f"Loaded calibration: FOV={cam.fov_deg:.2f} "
              f"cx={cam.xcenter:.1f} cy={cam.ycenter:.1f} f={cam.focal_length:.1f} "
              f"R={cam.radius_px} frame={cam.frame_width}x{cam.frame_height} "
              f"pad={cam.pad_size} pitch={cam.pitch_deg} "
              f"k1={cam.k1:g} k2={cam.k2:g}")
    else:
        cam = FisheyeCamera(
            height=_cam_cfg.get('height', 3.048),
            pitch_deg=_cam_cfg.get('pitch_deg', 0.0),
        )
        print("No camera_calibration.json found, using defaults")

    # Load real AP/AR from evaluation cache if provided
    unc = None
    if args.eval_cache:
        unc = load_uncertainty_from_cache(args.eval_cache, args.eval_model)
        if unc is not None:
            # Update the default OBJECT_TYPES with real values
            OBJECT_TYPES['pedestrian'].ap = unc.pedestrian.overall.ap
            OBJECT_TYPES['pedestrian'].ar = unc.pedestrian.overall.ar
            OBJECT_TYPES['cyclist'].ap = unc.cyclist.overall.ap
            OBJECT_TYPES['cyclist'].ar = unc.cyclist.overall.ar
            for vtype in ('car', 'motorcycle', 'bus', 'truck'):
                OBJECT_TYPES[vtype].ap = unc.vehicle.overall.ap
                OBJECT_TYPES[vtype].ar = unc.vehicle.overall.ar

    if args.error_map:
        print_error_summary(cam)
        return

    if args.compare:
        compare_pipelines(cam, add_bbox_noise=args.bbox_noise, uncertainty=unc)
        return

    if args.monte_carlo:
        for name, params in PIPELINE_CONFIGS.items():
            run_monte_carlo(params, n_trials=args.mc_trials, cam=cam,
                            add_bbox_noise=args.bbox_noise,
                            uncertainty=unc, failure_model=fm)
        return

    if args.latency_sweep:
        run_latency_sweep(PIPELINE_CONFIGS['default'], cam=cam,
                          uncertainty=unc, add_bbox_noise=args.bbox_noise)
        return

    if args.optimize:
        optimal = optimize_parameters(
            cam=cam, add_bbox_noise=args.bbox_noise,
            w_fn=args.w_fn, w_fp=args.w_fp, w_lat=args.w_lat)

        print("\n\nResults with OPTIMAL parameters:")
        run_all_scenarios(optimal, cam=cam, add_bbox_noise=args.bbox_noise,
                          uncertainty=unc)

        print("\n\nResults with DEFAULT parameters:")
        run_all_scenarios(PIPELINE_CONFIGS['default'], cam=cam,
                          add_bbox_noise=args.bbox_noise, uncertainty=unc)
        return

    # Default: run all scenarios with default pipeline
    print("Running all scenarios with Default pipeline...\n")
    run_all_scenarios(PIPELINE_CONFIGS['default'], cam=cam,
                      uncertainty=unc)


if __name__ == '__main__':
    main()
