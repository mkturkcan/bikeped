#!/usr/bin/env python3
"""
Fisheye Camera Calibration
==========================

Calibrates the equidistant fisheye model (r = f * theta) used by mqtt_bridge.py.
Standard OpenCV fisheye calibration uses Kannala-Brandt with polynomial distortion
coefficients, which does not fit this 200-degree spherical lens on a 16:9 sensor.

This script uses a custom bundle adjustment that directly fits the equidistant
projection model to checkerboard observations.

Parameters calibrated:
    cx, cy         : optical center in the preprocessed frame (pixels)
    focal_length   : equidistant projection focal length (pixels)
    k1, k2         : optional radial correction terms (r = f*(theta + k1*theta^3 + k2*theta^5))

Usage:
    # From a set of images showing a checkerboard:
    python calibrate_fisheye.py --images calibration_frames/*.jpg

    # From a video (extracts frames at an interval):
    python calibrate_fisheye.py --video calibration.avi --interval 30

    # With a known board size and square dimensions:
    python calibrate_fisheye.py --images *.jpg --board-width 9 --board-height 6 --square-size 0.025

    # Enable radial correction terms for higher accuracy:
    python calibrate_fisheye.py --images *.jpg --fit-distortion

    # Interactive mode: step through a video and press 's' to select frames:
    python calibrate_fisheye.py --video calibration.avi --interactive
"""

import argparse
import glob
import sys
from pathlib import Path
from typing import List, Optional, Tuple

import cv2
import numpy as np
import yaml
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation


# ---------------------------------------------------------------------------
# Preprocessing (matches mqtt_bridge.py exactly)
# ---------------------------------------------------------------------------

def preprocess_frame(frame, pad_size=3840, crop_size=3500):
    """Pad to square and center-crop, matching the mqtt_bridge pipeline."""
    h, w = frame.shape[:2]
    padded = np.zeros((pad_size, pad_size, 3), dtype=frame.dtype)

    if h > pad_size or w > pad_size:
        h_crop = min(h, pad_size)
        w_crop = min(w, pad_size)
        y_start = (h - h_crop) // 2
        x_start = (w - w_crop) // 2
        padded = frame[y_start:y_start + h_crop, x_start:x_start + w_crop].copy()
    else:
        y_offset = (pad_size - h) // 2
        x_offset = (pad_size - w) // 2
        padded[y_offset:y_offset + h, x_offset:x_offset + w] = frame

    center = pad_size // 2
    half = crop_size // 2
    y0 = max(0, center - half)
    y1 = min(pad_size, center + half)
    x0 = max(0, center - half)
    x1 = min(pad_size, center + half)

    return padded[y0:y1, x0:x1]


# ---------------------------------------------------------------------------
# Equidistant fisheye projection
# ---------------------------------------------------------------------------

def equidistant_project(pts_3d, rvec, tvec, cx, cy, f, k1=0.0, k2=0.0):
    """Project 3D points through the equidistant fisheye model.

    Model:
        theta  = angle between ray and optical axis
        r      = f * (theta + k1*theta^3 + k2*theta^5)
        u      = cx + r * cos(phi)
        v      = cy + r * sin(phi)

    Camera convention: x-right, y-down, z-forward (optical axis).

    Args:
        pts_3d:  (N, 3) object-frame coordinates.
        rvec:    (3,) Rodrigues rotation vector.
        tvec:    (3,) translation vector.
        cx, cy:  optical center (pixels).
        f:       equidistant focal length (pixels).
        k1, k2:  radial correction coefficients.

    Returns:
        (N, 2) pixel coordinates [u, v].
    """
    R = Rotation.from_rotvec(rvec).as_matrix()
    p_cam = (R @ pts_3d.T).T + tvec  # (N, 3) in camera frame

    x, y, z = p_cam[:, 0], p_cam[:, 1], p_cam[:, 2]
    norm = np.sqrt(x ** 2 + y ** 2 + z ** 2)
    norm = np.maximum(norm, 1e-12)

    theta = np.arccos(np.clip(z / norm, -1.0, 1.0))
    phi = np.arctan2(y, x)

    # Equidistant projection with optional polynomial correction
    r = f * (theta + k1 * theta ** 3 + k2 * theta ** 5)

    u = cx + r * np.cos(phi)
    v = cy + r * np.sin(phi)
    return np.column_stack([u, v])


# ---------------------------------------------------------------------------
# Fisheye → perspective remapping (required for checkerboard detection)
# ---------------------------------------------------------------------------

def build_perspective_remap(fish_h, fish_w, cx, cy, focal_length,
                            persp_size=1500, persp_fov_deg=160.0):
    """Build remap tables that convert a fisheye image to a perspective view.

    findChessboardCorners requires straight lines between corners, which
    does not hold under fisheye distortion.  Remapping to a perspective
    (rectilinear) view straightens those lines so detection works.

    The remap is built from the approximate equidistant model — it does not
    need to be perfectly accurate, only good enough that the perspective
    image has approximately straight lines.

    Returns:
        dict with map_x, map_y (for cv2.remap), f_persp, persp_size.
    """
    f_persp = persp_size / (2.0 * np.tan(np.deg2rad(persp_fov_deg / 2.0)))
    cx_p = (persp_size - 1) / 2.0
    cy_p = (persp_size - 1) / 2.0

    v_p, u_p = np.mgrid[0:persp_size, 0:persp_size].astype(np.float64)

    # Perspective pixel → ray direction (pinhole model)
    x_n = (u_p - cx_p) / f_persp
    y_n = (v_p - cy_p) / f_persp

    # Incidence angle and azimuth
    theta = np.arctan(np.sqrt(x_n ** 2 + y_n ** 2))
    phi = np.arctan2(y_n, x_n)

    # Equidistant projection: r = f * theta
    r_fish = focal_length * theta
    map_x = (cx + r_fish * np.cos(phi)).astype(np.float32)
    map_y = (cy + r_fish * np.sin(phi)).astype(np.float32)

    return {
        'map_x': map_x,
        'map_y': map_y,
        'f_persp': f_persp,
        'persp_size': persp_size,
        'cx_fish': cx,
        'cy_fish': cy,
        'f_fish': focal_length,
    }


def _persp_corners_to_fisheye(corners_persp, remap):
    """Map detected corner positions from perspective image back to fisheye."""
    pts = corners_persp.reshape(-1, 2).astype(np.float64)
    cx_p = (remap['persp_size'] - 1) / 2.0
    cy_p = (remap['persp_size'] - 1) / 2.0

    x_n = (pts[:, 0] - cx_p) / remap['f_persp']
    y_n = (pts[:, 1] - cy_p) / remap['f_persp']

    theta = np.arctan(np.sqrt(x_n ** 2 + y_n ** 2))
    phi = np.arctan2(y_n, x_n)

    r_fish = remap['f_fish'] * theta
    u_fish = remap['cx_fish'] + r_fish * np.cos(phi)
    v_fish = remap['cy_fish'] + r_fish * np.sin(phi)

    return np.column_stack([u_fish, v_fish]).reshape(-1, 1, 2).astype(np.float32)


# ---------------------------------------------------------------------------
# Checkerboard detection
# ---------------------------------------------------------------------------

def detect_checkerboard(image, board_size, remap):
    """Detect checkerboard corners in a fisheye image.

    Pipeline:
      1. Remap fisheye → perspective (straightens curved lines)
      2. Detect checkerboard in the perspective image
      3. Map corners back to fisheye pixel coordinates
      4. Refine with cornerSubPix on the original fisheye image

    Args:
        image:       Full-resolution fisheye image (BGR or grayscale).
        board_size:  (cols, rows) inner corner count.
        remap:       Dict from build_perspective_remap().

    Returns:
        (found, corners)  where corners is (N, 1, 2) in fisheye pixels.
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.ndim == 3 else image

    # 1. Remap fisheye → perspective
    persp = cv2.remap(gray, remap['map_x'], remap['map_y'],
                      cv2.INTER_LINEAR, borderValue=0)

    # 2. Detect in perspective space
    found, corners = _detect_corners(persp, board_size)

    if not found:
        return False, None

    # 3. Map corners back to fisheye coordinates
    corners_fish = _persp_corners_to_fisheye(corners, remap)

    # 4. Subpixel refinement on the original fisheye image
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER,
                100, 0.001)
    corners_fish = cv2.cornerSubPix(gray, corners_fish, (11, 11),
                                    (-1, -1), criteria)

    return True, corners_fish


def _detect_corners(gray, board_size):
    """Run corner detection with multiple fallback strategies."""
    # Strategy 1: Sector-based detector (OpenCV 4+)
    try:
        found, corners = cv2.findChessboardCornersSB(
            gray, board_size,
            cv2.CALIB_CB_NORMALIZE_IMAGE | cv2.CALIB_CB_EXHAUSTIVE)
        if found:
            return True, corners
    except cv2.error:
        pass

    # Strategy 2-3: Classic detector
    flags_list = [
        cv2.CALIB_CB_ADAPTIVE_THRESH | cv2.CALIB_CB_NORMALIZE_IMAGE,
        cv2.CALIB_CB_ADAPTIVE_THRESH | cv2.CALIB_CB_NORMALIZE_IMAGE
        | cv2.CALIB_CB_FILTER_QUADS,
    ]
    for flags in flags_list:
        found, corners = cv2.findChessboardCorners(gray, board_size, flags)
        if found:
            return True, corners

    # Strategy 4: CLAHE enhancement
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(gray)

    try:
        found, corners = cv2.findChessboardCornersSB(
            enhanced, board_size,
            cv2.CALIB_CB_NORMALIZE_IMAGE | cv2.CALIB_CB_EXHAUSTIVE)
        if found:
            return True, corners
    except cv2.error:
        pass

    for flags in flags_list:
        found, corners = cv2.findChessboardCorners(enhanced, board_size, flags)
        if found:
            return True, corners

    return False, None


# ---------------------------------------------------------------------------
# Extrinsic initialization via PnP on bearing vectors
# ---------------------------------------------------------------------------

def _init_extrinsics_pnp(obj_pts, img_pts, cx, cy, f, k1=0.0, k2=0.0):
    """Estimate initial R, t for one frame via PnP on undistorted bearings.

    We invert the equidistant model to get bearing vectors, then convert
    those to pinhole-normalised coordinates for OpenCV's solvePnP.
    Points behind the pinhole camera (theta >= pi/2) are excluded.
    """
    dx = img_pts[:, 0] - cx
    dy = img_pts[:, 1] - cy
    r = np.sqrt(dx ** 2 + dy ** 2)
    phi = np.arctan2(dy, dx)

    # Invert r = f*(theta + k1*theta^3 + k2*theta^5) via Newton for theta
    theta = r / f  # initial guess (ignoring k1, k2)
    for _ in range(20):
        g = f * (theta + k1 * theta ** 3 + k2 * theta ** 5) - r
        gp = f * (1 + 3 * k1 * theta ** 2 + 5 * k2 * theta ** 4)
        gp = np.maximum(gp, 1e-12)
        theta = theta - g / gp
        theta = np.clip(theta, 0, np.pi)

    # Keep only points in front of pinhole plane (theta < ~85 deg)
    mask = theta < (85.0 * np.pi / 180.0)
    if mask.sum() < 4:
        mask = theta < (np.pi / 2 - 0.01)
    if mask.sum() < 4:
        # Fallback: use all points anyway
        mask = np.ones(len(theta), dtype=bool)

    r_pinhole = np.tan(np.clip(theta[mask], 0, np.pi / 2 - 0.01))
    x_norm = r_pinhole * np.cos(phi[mask])
    y_norm = r_pinhole * np.sin(phi[mask])

    pts_2d = np.column_stack([x_norm, y_norm]).astype(np.float64)
    pts_3d = obj_pts[mask].astype(np.float64)
    K = np.eye(3, dtype=np.float64)

    success, rvec, tvec = cv2.solvePnP(
        pts_3d, pts_2d, K, None, flags=cv2.SOLVEPNP_SQPNP
    )
    if not success:
        success, rvec, tvec = cv2.solvePnP(
            pts_3d, pts_2d, K, None, flags=cv2.SOLVEPNP_ITERATIVE
        )
    if not success:
        rvec = np.zeros((3, 1), dtype=np.float64)
        tvec = np.array([[0.0], [0.0], [0.5]], dtype=np.float64)

    return rvec.ravel(), tvec.ravel()


# ---------------------------------------------------------------------------
# Bundle adjustment
# ---------------------------------------------------------------------------

def _build_residuals(params, obj_pts_list, img_pts_list, n_frames, fit_distortion):
    """Reprojection residuals for all frames.

    Parameter layout:
        [cx, cy, f, (k1, k2)?,  rvec0(3), tvec0(3), rvec1(3), tvec1(3), ...]
    """
    cx, cy, f = params[0], params[1], params[2]
    n_intrinsic = 5 if fit_distortion else 3
    k1 = params[3] if fit_distortion else 0.0
    k2 = params[4] if fit_distortion else 0.0

    residuals = []
    for i in range(n_frames):
        off = n_intrinsic + i * 6
        rv = params[off:off + 3]
        tv = params[off + 3:off + 6]
        proj = equidistant_project(obj_pts_list[i], rv, tv, cx, cy, f, k1, k2)
        residuals.append((proj - img_pts_list[i]).ravel())

    return np.concatenate(residuals)


def _build_jac_sparsity(n_frames, pts_per_frame, fit_distortion):
    """Analytical sparsity structure for the Jacobian (speeds up least_squares)."""
    n_intrinsic = 5 if fit_distortion else 3
    n_params = n_intrinsic + n_frames * 6
    n_residuals = sum(n * 2 for n in pts_per_frame)

    from scipy.sparse import lil_matrix
    J = lil_matrix((n_residuals, n_params), dtype=int)

    row = 0
    for i, n_pts in enumerate(pts_per_frame):
        n_res = n_pts * 2
        # Intrinsics affect every residual
        J[row:row + n_res, 0:n_intrinsic] = 1
        # Extrinsics for frame i
        off = n_intrinsic + i * 6
        J[row:row + n_res, off:off + 6] = 1
        row += n_res

    return J


def _refine_single_frame(obj_pts, img_pts, rv0, tv0, cx, cy, f, k1=0.0, k2=0.0):
    """Refine extrinsics for a single frame with intrinsics fixed (LM)."""

    def resid(p):
        return (equidistant_project(obj_pts, p[:3], p[3:], cx, cy, f, k1, k2)
                - img_pts).ravel()

    p0 = np.concatenate([rv0, tv0])
    res = least_squares(resid, p0, method='lm', max_nfev=500)
    return res.x[:3], res.x[3:]


def calibrate(obj_pts_list, img_pts_list, image_size,
              initial_cx=None, initial_cy=None, initial_f=None,
              fit_distortion=False):
    """Run full calibration via two-stage bundle adjustment.

    Stage 1 — Levenberg-Marquardt with linear loss (fast convergence).
    Stage 2 — Trust-region with soft_l1 loss (outlier robustness).

    Args:
        obj_pts_list:   list of (N, 3) checkerboard 3D points per frame.
        img_pts_list:   list of (N, 2) detected pixel coordinates per frame.
        image_size:     (width, height) of preprocessed frame.
        initial_cx/cy:  initial optical-center guess (default: frame center).
        initial_f:      initial focal-length guess (default: from 200 deg FOV).
        fit_distortion: if True, also fit k1, k2 correction coefficients.

    Returns:
        dict with calibrated parameters, per-frame errors, and diagnostics.
    """
    n_frames = len(obj_pts_list)
    w, h = image_size

    if initial_cx is None:
        initial_cx = (w - 1) / 2.0
    if initial_cy is None:
        initial_cy = (h - 1) / 2.0
    if initial_f is None:
        radius_guess = min(w, h) / 2.0
        initial_f = (2 * radius_guess) * 180.0 / (200.0 * np.pi)

    n_intrinsic = 5 if fit_distortion else 3
    k1_init, k2_init = 0.0, 0.0

    # --- Initialize per-frame extrinsics via PnP, then refine individually ---
    print("  Initializing per-frame extrinsics...")
    rvecs_init, tvecs_init = [], []
    for obj, img in zip(obj_pts_list, img_pts_list):
        rv, tv = _init_extrinsics_pnp(obj, img, initial_cx, initial_cy, initial_f)
        rv, tv = _refine_single_frame(obj, img, rv, tv, initial_cx, initial_cy, initial_f)
        rvecs_init.append(rv)
        tvecs_init.append(tv)

    # --- Pack parameter vector ---
    params0 = np.zeros(n_intrinsic + n_frames * 6)
    params0[0] = initial_cx
    params0[1] = initial_cy
    params0[2] = initial_f
    if fit_distortion:
        params0[3] = k1_init
        params0[4] = k2_init
    for i in range(n_frames):
        off = n_intrinsic + i * 6
        params0[off:off + 3] = rvecs_init[i]
        params0[off + 3:off + 6] = tvecs_init[i]

    pts_per_frame = [len(p) for p in img_pts_list]
    total_pts = sum(pts_per_frame)
    n_params = len(params0)
    jac_sparsity = _build_jac_sparsity(n_frames, pts_per_frame, fit_distortion)

    # --- Stage 1: Levenberg-Marquardt (linear loss, fast convergence) ---
    print(f"\n  Stage 1 — LM optimisation: {n_params} params "
          f"({n_frames} frames, {total_pts} corners)")

    result = least_squares(
        _build_residuals,
        params0,
        args=(obj_pts_list, img_pts_list, n_frames, fit_distortion),
        method='lm',
        ftol=1e-12,
        xtol=1e-12,
        gtol=1e-12,
        max_nfev=10000,
        verbose=1,
    )

    # --- Stage 2: Trust-region with soft_l1 for outlier robustness ---
    lower = np.full_like(result.x, -np.inf)
    upper = np.full_like(result.x, np.inf)
    lower[0], upper[0] = initial_cx - w * 0.2, initial_cx + w * 0.2
    lower[1], upper[1] = initial_cy - h * 0.2, initial_cy + h * 0.2
    lower[2], upper[2] = initial_f * 0.2, initial_f * 3.0
    if fit_distortion:
        lower[3], upper[3] = -2.0, 2.0
        lower[4], upper[4] = -2.0, 2.0

    print(f"  Stage 2 — TRF + soft_l1 refinement")

    result = least_squares(
        _build_residuals,
        result.x,
        args=(obj_pts_list, img_pts_list, n_frames, fit_distortion),
        method='trf',
        bounds=(lower, upper),
        jac_sparsity=jac_sparsity,
        loss='soft_l1',
        f_scale=1.0,
        ftol=1e-12,
        xtol=1e-12,
        gtol=1e-12,
        max_nfev=2000,
        verbose=1,
    )

    # --- Unpack results ---
    cx_opt, cy_opt, f_opt = result.x[0], result.x[1], result.x[2]
    k1_opt = result.x[3] if fit_distortion else 0.0
    k2_opt = result.x[4] if fit_distortion else 0.0

    opt_rvecs, opt_tvecs, per_frame_rms = [], [], []
    for i in range(n_frames):
        off = n_intrinsic + i * 6
        rv = result.x[off:off + 3]
        tv = result.x[off + 3:off + 6]
        opt_rvecs.append(rv)
        opt_tvecs.append(tv)
        proj = equidistant_project(obj_pts_list[i], rv, tv, cx_opt, cy_opt, f_opt, k1_opt, k2_opt)
        err = np.sqrt(np.mean((proj - img_pts_list[i]) ** 2))
        per_frame_rms.append(err)

    rms_total = np.sqrt(np.mean(result.fun ** 2))

    # Derive FOV from calibrated focal length
    radius_px = min(w, h) / 2.0
    fov_deg = (2 * radius_px) * 180.0 / (f_opt * np.pi)

    return {
        'cx': cx_opt,
        'cy': cy_opt,
        'focal_length': f_opt,
        'fov_deg': fov_deg,
        'radius_px': radius_px,
        'k1': k1_opt,
        'k2': k2_opt,
        'rms_error': rms_total,
        'per_frame_rms': per_frame_rms,
        'num_frames': n_frames,
        'num_points': total_pts,
        'rvecs': opt_rvecs,
        'tvecs': opt_tvecs,
        'optimizer_cost': result.cost,
        'optimizer_status': result.message,
    }


# ---------------------------------------------------------------------------
# Visualization helpers
# ---------------------------------------------------------------------------

def draw_reprojection(image, obj_pts, img_pts, rvec, tvec,
                      cx, cy, f, k1=0.0, k2=0.0, label=""):
    """Overlay detected (green) and reprojected (red) corners on an image."""
    vis = image.copy()
    proj = equidistant_project(obj_pts, rvec, tvec, cx, cy, f, k1, k2)

    for obs, rep in zip(img_pts, proj):
        ox, oy = int(round(obs[0])), int(round(obs[1]))
        rx, ry = int(round(rep[0])), int(round(rep[1]))
        cv2.circle(vis, (ox, oy), 6, (0, 255, 0), 2)
        cv2.drawMarker(vis, (rx, ry), (0, 0, 255), cv2.MARKER_CROSS, 12, 2)
        cv2.line(vis, (ox, oy), (rx, ry), (255, 128, 0), 1)

    if label:
        cv2.putText(vis, label, (20, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 255), 2)
    return vis


def draw_detection(image, corners, board_size):
    """Draw detected checkerboard corners on an image."""
    vis = image.copy()
    cv2.drawChessboardCorners(vis, board_size, corners, True)
    return vis


# ---------------------------------------------------------------------------
# Frame acquisition
# ---------------------------------------------------------------------------

def load_frames_from_images(patterns):
    """Load frames from image file paths / glob patterns."""
    paths = []
    for pat in patterns:
        paths.extend(sorted(glob.glob(pat)))
    frames = []
    for p in paths:
        img = cv2.imread(p)
        if img is not None:
            frames.append((Path(p).name, img))
        else:
            print(f"  Warning: could not read {p}")
    return frames


def load_frames_from_video(video_path, interval, max_frames):
    """Extract frames from a video at a regular interval."""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: cannot open video {video_path}")
        sys.exit(1)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frames = []
    idx = 0
    while len(frames) < max_frames:
        ret, frame = cap.read()
        if not ret:
            break
        if idx % interval == 0:
            frames.append((f"frame_{idx:06d}", frame))
        idx += 1
    cap.release()
    print(f"Extracted {len(frames)} frames from video "
          f"({total} total, interval={interval})")
    return frames


def interactive_select_frames(video_path, pad_size, crop_size, board_size,
                              remap):
    """Step through a video and let the user press 's' to select frames.

    Controls:
        s       - save current frame for calibration
        d / ->  - skip forward 10 frames
        a / <-  - skip backward 10 frames
        space   - advance 1 frame
        q / Esc - done selecting
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: cannot open video {video_path}")
        sys.exit(1)

    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f"\nInteractive selection — {total} frames in video")
    print("  [s] save frame   [space] next   [d/→] +10   [a/←] -10   [q] done\n")

    selected = []
    idx = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        prep = preprocess_frame(frame, pad_size, crop_size)
        found, corners = detect_checkerboard(prep, board_size, remap)

        # Build display image
        disp = prep.copy()
        if found:
            cv2.drawChessboardCorners(disp, board_size, corners, True)
            status = "DETECTED"
            colour = (0, 255, 0)
        else:
            status = "not found"
            colour = (0, 0, 255)

        text = f"Frame {idx}/{total}  |  Board: {status}  |  Saved: {len(selected)}"
        # Scale for display
        scale = min(1.0, 1200.0 / max(disp.shape[:2]))
        small = cv2.resize(disp, None, fx=scale, fy=scale)
        cv2.putText(small, text, (15, 35),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, colour, 2)
        cv2.imshow("Calibration - Select Frames", small)

        key = cv2.waitKey(0) & 0xFF
        if key == ord('q') or key == 27:
            break
        elif key == ord('s') and found:
            selected.append((f"frame_{idx:06d}", frame.copy()))
            print(f"  Saved frame {idx} ({len(selected)} total)")
        elif key == ord('d') or key == 83:  # right arrow
            skip = min(10, total - idx - 1)
            for _ in range(skip):
                cap.read()
                idx += 1
        elif key == ord('a') or key == 81:  # left arrow
            new_idx = max(0, idx - 10)
            cap.set(cv2.CAP_PROP_POS_FRAMES, new_idx)
            idx = new_idx - 1  # will be incremented below
        # space or any other key: advance one frame

        idx += 1

    cv2.destroyAllWindows()
    cap.release()
    print(f"\nSelected {len(selected)} frames")
    return selected


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Calibrate equidistant fisheye camera",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    src = parser.add_mutually_exclusive_group(required=True)
    src.add_argument('--images', nargs='+',
                     help='Image paths or glob patterns showing checkerboard')
    src.add_argument('--video', type=str,
                     help='Video file containing checkerboard frames')

    parser.add_argument('--interactive', action='store_true',
                        help='Interactively select frames from video (requires --video)')

    parser.add_argument('--board-width', type=int, default=9,
                        help='Inner corners along board width (default: 9)')
    parser.add_argument('--board-height', type=int, default=6,
                        help='Inner corners along board height (default: 6)')
    parser.add_argument('--square-size', type=float, default=0.025,
                        help='Checkerboard square size in metres (default: 0.025)')

    parser.add_argument('--pad-size', type=int, default=3840,
                        help='Preprocessing pad size (default: 3840)')
    parser.add_argument('--crop-size', type=int, default=3500,
                        help='Preprocessing crop size (default: 3500)')

    parser.add_argument('--interval', type=int, default=30,
                        help='Frame sampling interval for video mode (default: 30)')
    parser.add_argument('--max-frames', type=int, default=50,
                        help='Max frames to use from video (default: 50)')

    parser.add_argument('--initial-fov', type=float, default=200.0,
                        help='Initial FOV guess in degrees (default: 200)')
    parser.add_argument('--fit-distortion', action='store_true',
                        help='Also fit k1, k2 radial correction terms')

    parser.add_argument('--output', type=str, default='camera_calibration.json',
                        help='Output JSON file consumed by mqtt_bridge / decision_testbench '
                             '/ sim_visualizer (default: camera_calibration.json). '
                             'A sibling .yaml with the full bundle (preprocessing block, '
                             'config_yaml_updates) is written next to it.')
    parser.add_argument('--visualize', action='store_true',
                        help='Show reprojection visualizations per frame')
    parser.add_argument('--save-viz', type=str, default=None,
                        help='Directory to save reprojection images')

    args = parser.parse_args()
    board_size = (args.board_width, args.board_height)

    # --- Object points (Z = 0 planar target) ---
    objp = np.zeros((board_size[0] * board_size[1], 3), dtype=np.float64)
    objp[:, :2] = np.mgrid[0:board_size[0], 0:board_size[1]].T.reshape(-1, 2)
    objp *= args.square_size

    # --- Build perspective remap early (needed for interactive mode too) ---
    radius_guess_early = args.crop_size / 2.0
    initial_f_early = (2 * radius_guess_early) * 180.0 / (args.initial_fov * np.pi)
    remap_early = build_perspective_remap(
        args.crop_size, args.crop_size,
        cx=(args.crop_size - 1) / 2.0,
        cy=(args.crop_size - 1) / 2.0,
        focal_length=initial_f_early,
    )

    # --- Load frames ---
    if args.video and args.interactive:
        raw_frames = interactive_select_frames(
            args.video, args.pad_size, args.crop_size, board_size,
            remap_early)
    elif args.video:
        raw_frames = load_frames_from_video(args.video, args.interval, args.max_frames)
    else:
        raw_frames = load_frames_from_images(args.images)

    if not raw_frames:
        print("No frames to process.")
        sys.exit(1)

    print(f"Loaded {len(raw_frames)} raw frames")
    print(f"Board: {board_size[0]}x{board_size[1]}, "
          f"square size: {args.square_size} m\n")

    # Reuse the remap already built above
    remap = remap_early
    initial_f = initial_f_early

    # --- Detect checkerboard in preprocessed frames ---
    obj_pts_list: List[np.ndarray] = []
    img_pts_list: List[np.ndarray] = []
    prep_frames: List[np.ndarray] = []
    used_names: List[str] = []

    print("Detecting checkerboard corners...")
    for name, frame in raw_frames:
        prep = preprocess_frame(frame, args.pad_size, args.crop_size)
        found, corners = detect_checkerboard(prep, board_size, remap)

        if found:
            pts_2d = corners.reshape(-1, 2)
            obj_pts_list.append(objp.copy())
            img_pts_list.append(pts_2d)
            prep_frames.append(prep)
            used_names.append(name)
            print(f"  [{len(obj_pts_list):3d}] {name}: "
                  f"{len(pts_2d)} corners detected")
        else:
            print(f"  [ - ] {name}: board not found")

    n_good = len(obj_pts_list)
    if n_good < 3:
        print(f"\nOnly {n_good} usable frames — need at least 3 for calibration.")
        sys.exit(1)

    print(f"\n{n_good} frames with detected corners")

    # --- Run calibration ---
    result = calibrate(
        obj_pts_list, img_pts_list,
        image_size=(args.crop_size, args.crop_size),
        initial_f=initial_f,
        fit_distortion=args.fit_distortion,
    )

    # --- Print results ---
    cx_shift = result['cx'] - (args.crop_size - 1) / 2.0
    cy_shift = result['cy'] - (args.crop_size - 1) / 2.0

    print("\n" + "=" * 64)
    print("  CALIBRATION RESULTS")
    print("=" * 64)
    print(f"  Model:           equidistant  r = f * theta"
          + (" + k1*theta^3 + k2*theta^5" if args.fit_distortion else ""))
    print(f"  Optical center:  ({result['cx']:.2f}, {result['cy']:.2f})  "
          f"[shift from frame center: ({cx_shift:+.2f}, {cy_shift:+.2f})]")
    print(f"  Focal length:    {result['focal_length']:.2f} px")
    print(f"  Derived FOV:     {result['fov_deg']:.2f} deg")
    if args.fit_distortion:
        print(f"  k1:              {result['k1']:.6f}")
        print(f"  k2:              {result['k2']:.6f}")
    print(f"  RMS reproj err:  {result['rms_error']:.4f} px")
    print(f"  Frames / points: {result['num_frames']} / {result['num_points']}")
    print()

    print("  Per-frame RMS (pixels):")
    for name, err in zip(used_names, result['per_frame_rms']):
        tag = "OK" if err < 2.0 else ("HIGH" if err < 5.0 else "BAD")
        print(f"    {name:30s}  {err:7.3f}  [{tag}]")

    # --- Kappa diagnostic ---
    # If the previous kappa was 2.0 and the new focal length is ~2x the old,
    # that explains the scale factor.
    old_f = (2 * radius_guess_early) * 180.0 / (args.initial_fov * np.pi)
    ratio = result['focal_length'] / old_f
    print(f"\n  Focal-length ratio vs initial guess: {ratio:.3f}")
    if abs(ratio - 2.0) < 0.3:
        print("  --> This likely explains the kappa=2.0 scale factor in the "
              "deployed code.")
        print("      With the calibrated focal length, kappa should be 1.0.")

    # --- Save calibration ---
    calib_out = {
        'calibration': {
            'model': 'equidistant',
            'cx': round(float(result['cx']), 4),
            'cy': round(float(result['cy']), 4),
            'focal_length': round(float(result['focal_length']), 4),
            'fov_deg': round(float(result['fov_deg']), 4),
            'radius_px': round(float(result['radius_px']), 1),
            'k1': round(float(result['k1']), 8),
            'k2': round(float(result['k2']), 8),
            'rms_error_px': round(float(result['rms_error']), 6),
            'num_frames': result['num_frames'],
            'num_points': result['num_points'],
        },
        'preprocessing': {
            'pad_size': args.pad_size,
            'crop_size': args.crop_size,
        },
        'config_yaml_updates': {
            'camera': {
                'fov': round(float(result['fov_deg']), 2),
                'radius': round(float(result['radius_px'])),
            },
        },
    }

    # Canonical runtime file: camera_calibration.json (loaded by mqtt_bridge,
    # decision_testbench, sim_visualizer, generate_figures). The companion YAML
    # next to it carries the preprocessing block and config.yaml hints.
    import json
    json_out = calib_out['calibration'].copy()
    json_out['frame_width'] = args.crop_size
    json_out['frame_height'] = args.crop_size
    json_out['pad_size'] = args.pad_size
    json_out['crop_size'] = args.crop_size

    json_path = Path(args.output)
    yaml_path = json_path.with_suffix('.yaml')
    with open(json_path, 'w') as f:
        json.dump(json_out, f, indent=2)
    with open(yaml_path, 'w') as f:
        yaml.dump(calib_out, f, default_flow_style=False, sort_keys=False)
    print(f"  Saved to: {json_path}")
    print(f"  Saved to: {yaml_path}")

    # --- Visualizations ---
    if args.save_viz:
        Path(args.save_viz).mkdir(parents=True, exist_ok=True)

    if args.visualize or args.save_viz:
        for i, (name, prep) in enumerate(zip(used_names, prep_frames)):
            label = (f"{name}  |  RMS: {result['per_frame_rms'][i]:.3f} px")
            vis = draw_reprojection(
                prep, obj_pts_list[i], img_pts_list[i],
                result['rvecs'][i], result['tvecs'][i],
                result['cx'], result['cy'], result['focal_length'],
                result['k1'], result['k2'], label=label,
            )

            if args.save_viz:
                stem = Path(name).stem
                out_path = Path(args.save_viz) / f"reproj_{i:03d}_{stem}.png"
                scale = min(1.0, 1280.0 / max(vis.shape[:2]))
                small = cv2.resize(vis, None, fx=scale, fy=scale)
                cv2.imwrite(str(out_path), small)
                print(f"  Saved {out_path}")

            if args.visualize:
                scale = min(1.0, 900.0 / max(vis.shape[:2]))
                small = cv2.resize(vis, None, fx=scale, fy=scale)
                cv2.imshow(f"Reprojection [{i}]", small)
                key = cv2.waitKey(0) & 0xFF
                cv2.destroyAllWindows()
                if key == ord('q'):
                    break

    # --- Summary for config.yaml ---
    print("\n" + "-" * 64)
    print("  To update config.yaml:")
    print(f"    camera.fov:    {result['fov_deg']:.2f}")
    print(f"    camera.radius: {round(result['radius_px'])}")
    if abs(cx_shift) > 5 or abs(cy_shift) > 5:
        print(f"    # Add optical center offset to config:")
        print(f"    camera.cx: {result['cx']:.2f}")
        print(f"    camera.cy: {result['cy']:.2f}")
    if abs(ratio - 2.0) < 0.3:
        print(f"    # Remove kappa=2.0 from pixel_to_ground (set to 1.0)")
    print("-" * 64)


if __name__ == '__main__':
    main()
