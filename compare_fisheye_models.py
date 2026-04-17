#!/usr/bin/env python3
"""
Compare fisheye projection models
=================================

Runs calibration with each supported projection model on the same
checkerboard frames and reports RMS reprojection error for each.

Models tested:
    equidistant    r = f * theta
    equisolid      r = 2f * sin(theta/2)
    stereographic  r = 2f * tan(theta/2)
    orthographic   r = f * sin(theta)

Usage:
    python compare_fisheye_models.py --images calibration_frames/frame_*.png
    python compare_fisheye_models.py --images calibration_frames/frame_*.png --board-width 7 --board-height 5
"""

import argparse
import glob
import json
import sys
from pathlib import Path

import cv2
import numpy as np
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation

from calibrate_fisheye import (
    preprocess_frame,
    build_perspective_remap,
    detect_checkerboard,
    _init_extrinsics_pnp,
    _refine_single_frame,
)


# ---------------------------------------------------------------------------
# Multi-model projection
# ---------------------------------------------------------------------------

MODELS = ['equidistant', 'equisolid', 'stereographic', 'orthographic']


def project_model(pts_3d, rvec, tvec, cx, cy, f, model):
    """Project 3D points through a named fisheye model."""
    R = Rotation.from_rotvec(rvec).as_matrix()
    p_cam = (R @ pts_3d.T).T + tvec
    x, y, z = p_cam[:, 0], p_cam[:, 1], p_cam[:, 2]
    norm = np.maximum(np.sqrt(x ** 2 + y ** 2 + z ** 2), 1e-12)
    theta = np.arccos(np.clip(z / norm, -1.0, 1.0))
    phi = np.arctan2(y, x)

    if model == 'equidistant':
        r = f * theta
    elif model == 'equisolid':
        r = 2.0 * f * np.sin(theta / 2.0)
    elif model == 'stereographic':
        r = 2.0 * f * np.tan(theta / 2.0)
    elif model == 'orthographic':
        r = f * np.sin(theta)
    else:
        raise ValueError(f"Unknown model: {model}")

    u = cx + r * np.cos(phi)
    v = cy + r * np.sin(phi)
    return np.column_stack([u, v])


# ---------------------------------------------------------------------------
# Per-model calibration
# ---------------------------------------------------------------------------

def calibrate_model(obj_pts_list, img_pts_list, model, initial_cx, initial_cy,
                    initial_f):
    """Run LM bundle adjustment for a single projection model."""
    n = len(obj_pts_list)

    # Initialize extrinsics using equidistant model (good enough for all)
    rvecs, tvecs = [], []
    for obj, img in zip(obj_pts_list, img_pts_list):
        rv, tv = _init_extrinsics_pnp(obj, img, initial_cx, initial_cy,
                                       initial_f)
        rv, tv = _refine_single_frame(obj, img, rv, tv, initial_cx,
                                       initial_cy, initial_f)
        rvecs.append(rv)
        tvecs.append(tv)

    # Pack: [cx, cy, f, rv0, tv0, rv1, tv1, ...]
    p0 = np.zeros(3 + n * 6)
    p0[0], p0[1], p0[2] = initial_cx, initial_cy, initial_f
    for i in range(n):
        p0[3 + i * 6:3 + i * 6 + 3] = rvecs[i]
        p0[3 + i * 6 + 3:3 + i * 6 + 6] = tvecs[i]

    def residuals(params):
        cx, cy, f = params[0], params[1], params[2]
        res = []
        for i in range(n):
            rv = params[3 + i * 6:3 + i * 6 + 3]
            tv = params[3 + i * 6 + 3:3 + i * 6 + 6]
            proj = project_model(obj_pts_list[i], rv, tv, cx, cy, f, model)
            res.append((proj - img_pts_list[i]).ravel())
        return np.concatenate(res)

    result = least_squares(residuals, p0, method='lm',
                           ftol=1e-12, xtol=1e-12, gtol=1e-12,
                           max_nfev=10000, verbose=0)

    cx, cy, f = result.x[0], result.x[1], result.x[2]
    rms = np.sqrt(np.mean(result.fun ** 2))

    # Per-frame RMS
    per_frame = []
    for i in range(n):
        rv = result.x[3 + i * 6:3 + i * 6 + 3]
        tv = result.x[3 + i * 6 + 3:3 + i * 6 + 6]
        proj = project_model(obj_pts_list[i], rv, tv, cx, cy, f, model)
        err = np.sqrt(np.mean((proj - img_pts_list[i]) ** 2))
        per_frame.append(err)

    # Derive FOV
    radius_px = 1750.0
    if model == 'equidistant':
        fov = (2 * radius_px) * 180.0 / (f * np.pi)
    elif model == 'equisolid':
        fov = 4.0 * np.rad2deg(np.arcsin(radius_px / (2.0 * f)))
    elif model == 'stereographic':
        fov = 4.0 * np.rad2deg(np.arctan(radius_px / (2.0 * f)))
    elif model == 'orthographic':
        fov = 2.0 * np.rad2deg(np.arcsin(min(radius_px / f, 1.0)))

    return {
        'model': model,
        'cx': round(float(cx), 4),
        'cy': round(float(cy), 4),
        'focal_length': round(float(f), 4),
        'fov_deg': round(float(fov), 4),
        'rms_error_px': round(float(rms), 6),
        'per_frame_rms': [round(float(e), 4) for e in per_frame],
        'num_frames': n,
        'num_points': sum(len(p) for p in img_pts_list),
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Compare fisheye projection models",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument('--images', nargs='+', required=True,
                        help='Image paths or glob patterns')
    parser.add_argument('--board-width', type=int, default=7)
    parser.add_argument('--board-height', type=int, default=5)
    parser.add_argument('--square-size', type=float, default=0.025)
    parser.add_argument('--pad-size', type=int, default=3840)
    parser.add_argument('--crop-size', type=int, default=3500)
    parser.add_argument('--initial-fov', type=float, default=200.0)
    parser.add_argument('--output', type=str,
                        default='model_comparison.json',
                        help='Output JSON (default: model_comparison.json)')
    args = parser.parse_args()

    board_size = (args.board_width, args.board_height)
    objp = np.zeros((board_size[0] * board_size[1], 3), dtype=np.float64)
    objp[:, :2] = np.mgrid[0:board_size[0], 0:board_size[1]].T.reshape(-1, 2)
    objp *= args.square_size

    # Load and detect
    paths = []
    for pat in args.images:
        paths.extend(sorted(glob.glob(pat)))

    radius_guess = args.crop_size / 2.0
    initial_f = (2 * radius_guess) * 180.0 / (args.initial_fov * np.pi)
    remap = build_perspective_remap(
        args.crop_size, args.crop_size,
        (args.crop_size - 1) / 2.0, (args.crop_size - 1) / 2.0,
        initial_f)

    obj_list, img_list = [], []
    for p in paths:
        img = cv2.imread(p)
        if img is None:
            continue
        prep = preprocess_frame(img, args.pad_size, args.crop_size)
        found, corners = detect_checkerboard(prep, board_size, remap)
        if found:
            obj_list.append(objp.copy())
            img_list.append(corners.reshape(-1, 2))

    print(f"{len(obj_list)} frames with detected corners\n")
    if len(obj_list) < 3:
        print("Need at least 3 frames.")
        sys.exit(1)

    initial_cx = (args.crop_size - 1) / 2.0
    initial_cy = (args.crop_size - 1) / 2.0

    # Run each model
    results = []
    print(f"{'Model':15s}  {'RMS (px)':>9s}  {'cx':>8s}  {'cy':>8s}  "
          f"{'f':>8s}  {'FOV':>7s}")
    print("-" * 65)

    for model in MODELS:
        r = calibrate_model(obj_list, img_list, model,
                            initial_cx, initial_cy, initial_f)
        results.append(r)
        print(f"{model:15s}  {r['rms_error_px']:9.4f}  {r['cx']:8.1f}  "
              f"{r['cy']:8.1f}  {r['focal_length']:8.1f}  "
              f"{r['fov_deg']:7.2f}")

    # Sort by RMS
    best = min(results, key=lambda x: x['rms_error_px'])
    print(f"\nBest model: {best['model']} (RMS {best['rms_error_px']:.4f} px)")

    # Save
    with open(args.output, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"Saved to {args.output}")


if __name__ == '__main__':
    main()
