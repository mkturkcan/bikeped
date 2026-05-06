#!/usr/bin/env python3
"""
Calibration Frame Capture
=========================

Connects to the fisheye camera and shows a live preview.
Press 'r' to start recording frames (1 per second), 'r' again to stop.
Saved frames go to calibration_frames/ for use with calibrate_fisheye.py.

Controls:
    r           Start / stop recording (1 frame per second)
    q / Esc     Quit

Usage:
    python capture_calibration.py
    python capture_calibration.py --camera 0 --outdir my_cal_frames
"""

import argparse
import sys
import time
from pathlib import Path

import cv2


def open_camera(camera_id):
    """Open the camera matching mqtt_bridge.py's setup."""
    cap = None
    if sys.platform.startswith("linux"):
        cap = cv2.VideoCapture(camera_id, cv2.CAP_V4L2)
        if cap.isOpened():
            cap.set(cv2.CAP_PROP_FOURCC,
                    cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'))
    if cap is None or not cap.isOpened():
        cap = cv2.VideoCapture(camera_id)

    if not cap.isOpened():
        print(f"Error: cannot open camera {camera_id}")
        sys.exit(1)

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 3840)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 2160)

    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(f"Camera opened: {w}x{h}")
    return cap


def main():
    parser = argparse.ArgumentParser(
        description="Capture calibration frames from fisheye camera")
    parser.add_argument('--camera', type=int, default=0,
                        help='Camera index (default: 0)')
    parser.add_argument('--outdir', type=str, default='calibration_frames',
                        help='Output directory (default: calibration_frames)')
    parser.add_argument('--display-width', type=int, default=1200,
                        help='Preview window width (default: 1200)')
    args = parser.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    cap = open_camera(args.camera)

    recording = False
    n_saved = len(list(outdir.glob("frame_*.png")))
    last_save_time = 0.0

    if n_saved > 0:
        print(f"Found {n_saved} existing frames in {outdir}/")

    print("\n  [r] start/stop recording   [q] quit\n")

    while True:
        ret, frame = cap.read()
        if not ret:
            time.sleep(0.05)
            continue

        now = time.time()

        # Save 1 frame/sec while recording
        if recording and (now - last_save_time) >= 1.0:
            path = outdir / f"frame_{n_saved:04d}.png"
            cv2.imwrite(str(path), frame)
            n_saved += 1
            last_save_time = now
            print(f"  [{n_saved}] {path.name}")

        # Build display
        scale = args.display_width / max(frame.shape[1], 1)
        small = cv2.resize(frame, None, fx=scale, fy=scale)
        w = small.shape[1]

        if recording:
            cv2.rectangle(small, (0, 0), (w, 36), (0, 0, 180), -1)
            cv2.putText(small, f"RECORDING  |  {n_saved} frames saved",
                        (12, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                        (255, 255, 255), 2)
        else:
            cv2.rectangle(small, (0, 0), (w, 36), (50, 50, 50), -1)
            cv2.putText(small, f"PAUSED  |  {n_saved} frames  |  [r] record  [q] quit",
                        (12, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                        (200, 200, 200), 2)

        cv2.imshow("Calibration Capture", small)
        key = cv2.waitKey(1) & 0xFF

        if key == ord('q') or key == 27:
            break
        elif key == ord('r'):
            recording = not recording
            if recording:
                last_save_time = 0.0
                print("  Recording started (1 frame/sec)")
            else:
                print(f"  Recording stopped ({n_saved} frames total)")

    cap.release()
    cv2.destroyAllWindows()
    print(f"\n{n_saved} frames in {outdir}/")


if __name__ == '__main__':
    main()
