"""
Scenario visualizer for the bikeped decision testbench.

Renders simulated fisheye camera views of test scenarios showing what the
camera would actually see: agents projected through the fisheye model onto
a ground-plane wireframe, with decision state overlay.

Camera placement matches index.html presets:
  - Single pole: (x=5, y=-13, h=5.5), heading 90 deg (facing +y across road)
  - Cross-cam:   add second camera at (x=5, y=+13, h=5.5), heading 270 deg

Usage:
    python sim_visualizer.py                        # key scenarios, single cam
    python sim_visualizer.py --scenarios 1 5 7      # specific scenarios
    python sim_visualizer.py --two-cam              # two-camera split view
    python sim_visualizer.py --no-save              # display only
"""

import json
import cv2
import numpy as np
import argparse
from math import sqrt, pi, cos, sin, atan2
from decision_testbench import (
    SCENARIOS, PIPELINE_CONFIGS, OBJECT_TYPES,
    interpolate_agent, get_scenario_duration,
    compute_ground_truth_danger, make_pipeline, run_pipeline_on_agents,
    compute_bbox_error, get_braking_profile,
    load_camera_calibration, FisheyeCamera,
)

# Camera presets from index.html
CAM_POLE = {'x': 5.0, 'y': -13.0, 'h': 3.6576, 'heading_deg': 90.0}
CAM_CROSS = {'x': 5.0, 'y': 13.0, 'h': 3.6576, 'heading_deg': 270.0}

IMG_W = 640
IMG_H = 480

# Load calibration
_CALIB = load_camera_calibration()
FOV_DEG = _CALIB['fov_deg'] if _CALIB else 197.90
# Optical center offset from frame center, in calibration frame (3500x3500)
_CALIB_CX_OFFSET = _CALIB['cx'] - 1749.5 if _CALIB else 0.0
_CALIB_CY_OFFSET = _CALIB['cy'] - 1749.5 if _CALIB else 0.0

STATE_COLORS = {
    'idle':    (255, 100, 0),
    'safe':    (0, 200, 0),
    'warning': (0, 165, 255),
    'alert':   (0, 0, 255),
}
STATE_LABELS = {
    'idle': 'IDLE', 'safe': 'SAFE', 'warning': 'WARNING', 'alert': 'ALERT',
}


class FisheyeRenderer:
    """Thin wrapper around FisheyeCamera for rendering at arbitrary resolution.

    Delegates all projection math to FisheyeCamera (decision_testbench.py)
    so that the simulation visualiser uses the exact same model as the
    decision testbench and error analysis.

    The calibrated 3500x3500 camera is scaled to the render resolution
    (default 640x480) by adjusting radius_px and optical center proportionally.
    """

    def __init__(self, cam_x, cam_y, cam_h, heading_deg, fov_deg=197.90,
                 img_w=640, img_h=480, pitch_deg=None,
                 cx_offset=0.0, cy_offset=0.0):
        self.img_w = img_w
        self.img_h = img_h

        # Scale calibrated parameters from 3500x3500 to render resolution
        calib_size = 3500
        render_dim = min(img_w, img_h)
        scale = render_dim / calib_size
        scaled_radius = int(round(1750 * scale))

        # Optical center in render coords
        render_cx = (img_w - 1) / 2.0 + cx_offset
        render_cy = (img_h - 1) / 2.0 + cy_offset

        self._cam = FisheyeCamera(
            height=cam_h,
            fov_deg=fov_deg,
            radius_px=scaled_radius,
            frame_width=img_w,
            frame_height=img_h,
            kappa=1.0,
            cam_x=cam_x,
            cam_y=cam_y,
            pitch_deg=pitch_deg,
            heading_deg=heading_deg,
            cx_override=render_cx,
            cy_override=render_cy,
        )

        # Sensor visible band scaled to render resolution
        # In the 3500x3500 crop: rows 670-2829 are on the sensor
        self.sensor_v_min = int(round(self._cam.sensor_v_min * scale))
        self.sensor_v_max = int(round(self._cam.sensor_v_max * scale))

    def project(self, world_points):
        """Project world points (N,3) to pixel coordinates (N,2).
        Returns pixels and a valid mask."""
        pts = np.atleast_2d(world_points).astype(np.float64)
        pixels, valid = self._cam.world_to_pixel(pts)
        return pixels, valid

    def project_single(self, x, y, z):
        """Project a single world point. Returns (u, v, valid)."""
        pts, valid = self.project(np.array([[x, y, z]]))
        return int(pts[0, 0]), int(pts[0, 1]), bool(valid[0])


def draw_ground_grid(frame, renderer, spacing=2.0, extent=50.0):
    """Draw a ground-plane grid through the fisheye projection."""
    color = (50, 50, 50)
    step = 0.5  # finer interpolation for smooth curves under fisheye
    # Lines along x
    for y in np.arange(-extent, extent + spacing, spacing):
        pts_3d = np.array([[x, y, 0.0] for x in np.arange(-extent, extent + step, step)])
        pts_2d, valid = renderer.project(pts_3d)
        for i in range(len(pts_2d) - 1):
            if valid[i] and valid[i+1]:
                p1 = (int(pts_2d[i, 0]), int(pts_2d[i, 1]))
                p2 = (int(pts_2d[i+1, 0]), int(pts_2d[i+1, 1]))
                cv2.line(frame, p1, p2, color, 1)
    # Lines along y
    for x in np.arange(-extent, extent + spacing, spacing):
        pts_3d = np.array([[x, y, 0.0] for y in np.arange(-extent, extent + step, step)])
        pts_2d, valid = renderer.project(pts_3d)
        for i in range(len(pts_2d) - 1):
            if valid[i] and valid[i+1]:
                p1 = (int(pts_2d[i, 0]), int(pts_2d[i, 1]))
                p2 = (int(pts_2d[i+1, 0]), int(pts_2d[i+1, 1]))
                cv2.line(frame, p1, p2, color, 1)


def draw_crosswalk(frame, renderer):
    """Draw crosswalk stripes at x=5, spanning y=-12 to y=12."""
    cw_x = 5.0
    for stripe_y in np.arange(-12, 12, 2.0):
        corners = np.array([
            [cw_x - 1.0, stripe_y, 0.0],
            [cw_x + 1.0, stripe_y, 0.0],
            [cw_x + 1.0, stripe_y + 1.0, 0.0],
            [cw_x - 1.0, stripe_y + 1.0, 0.0],
        ])
        pts_2d, valid = renderer.project(corners)
        if np.all(valid):
            poly = pts_2d.astype(np.int32).reshape(-1, 1, 2)
            cv2.fillPoly(frame, [poly], (60, 60, 80))


def draw_agent_3d(frame, renderer, agent, trail):
    """Draw an agent as a 3D wireframe box projected through fisheye,
    oriented along the agent's heading direction."""
    from decision_testbench import OBJECT_TYPES, object_3d_corners

    x, y = agent['x'], agent['y']
    heading = agent.get('heading', 0.0)
    is_person = agent['type'] == 'person'

    if is_person:
        obj = OBJECT_TYPES['pedestrian']
        color = (80, 80, 255)
        label = f"P{agent['id']}"
    else:
        obj = OBJECT_TYPES['cyclist']
        color = (80, 255, 80)
        label = f"B{agent['id']}"

    corners = object_3d_corners(x, y, obj, heading)
    pts_2d, valid = renderer.project(corners)

    if not np.any(valid):
        return

    edges = [(0,1),(1,2),(2,3),(3,0),  # bottom
             (4,5),(5,6),(6,7),(7,4),  # top
             (0,4),(1,5),(2,6),(3,7)]  # verticals

    for i, j in edges:
        if valid[i] and valid[j]:
            p1 = (int(pts_2d[i, 0]), int(pts_2d[i, 1]))
            p2 = (int(pts_2d[j, 0]), int(pts_2d[j, 1]))
            cv2.line(frame, p1, p2, color, 1, cv2.LINE_AA)

    # Label at top center
    top_center = np.array([[x, y, obj.height + 0.3]])
    lbl_px, lbl_valid = renderer.project(top_center)
    if lbl_valid[0]:
        lp = (int(lbl_px[0, 0]) + 5, int(lbl_px[0, 1]))
        cv2.putText(frame, label, lp,
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)

    # Trail on ground
    if len(trail) >= 2:
        trail_pts = np.array([[p[0], p[1], 0.0] for p in trail[-30:]])
        t_2d, t_valid = renderer.project(trail_pts)
        for i in range(len(t_2d) - 1):
            if t_valid[i] and t_valid[i+1]:
                p1 = (int(t_2d[i, 0]), int(t_2d[i, 1]))
                p2 = (int(t_2d[i+1, 0]), int(t_2d[i+1, 1]))
                cv2.line(frame, p1, p2, (color[0]//3, color[1]//3, color[2]//3),
                         1, cv2.LINE_AA)


def draw_proximity_line(frame, renderer, bx, by, px, py):
    """Draw a ground-level line between bike and pedestrian."""
    pts_3d = np.array([[bx, by, 0.5], [px, py, 0.5]])
    pts_2d, valid = renderer.project(pts_3d)
    if np.all(valid):
        p1 = (int(pts_2d[0, 0]), int(pts_2d[0, 1]))
        p2 = (int(pts_2d[1, 0]), int(pts_2d[1, 1]))
        cv2.line(frame, p1, p2, (0, 0, 200), 1, cv2.LINE_AA)
        d = sqrt((bx - px)**2 + (by - py)**2)
        mid = ((p1[0] + p2[0]) // 2, (p1[1] + p2[1]) // 2)
        cv2.putText(frame, f"{d:.1f}m", (mid[0]+3, mid[1]-3),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 0, 200), 1)


def draw_hud(frame, scenario_name, t, duration, state, danger, cam_label=""):
    """Draw heads-up display overlay."""
    h, w = frame.shape[:2]
    state_color = STATE_COLORS.get(state, (128, 128, 128))

    # Scenario name + camera label
    title = scenario_name
    if cam_label:
        title += f" [{cam_label}]"
    cv2.putText(frame, title, (8, 18),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (220, 220, 220), 1)
    cv2.putText(frame, f"t={t:.2f}s / {duration:.1f}s", (8, 36),
                cv2.FONT_HERSHEY_SIMPLEX, 0.35, (140, 140, 140), 1)

    # State badge
    label = STATE_LABELS.get(state, state.upper())
    cv2.rectangle(frame, (w - 110, 6), (w - 6, 30), state_color, -1)
    cv2.putText(frame, label, (w - 105, 24),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

    # Danger info
    if danger.is_danger:
        cv2.putText(frame, f"DANGER sev={danger.severity:.2f} v={danger.bike_speed:.1f}m/s",
                    (8, h - 28), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 0, 255), 1)
        if danger.ttc < 100:
            cv2.putText(frame, f"TTC={danger.ttc:.1f}s d_stop={danger.stopping_dist:.1f}m",
                        (8, h - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 0, 255), 1)

    # Progress bar
    progress = t / max(duration, 0.01)
    bar_y = h - 4
    cv2.rectangle(frame, (6, bar_y), (w - 6, bar_y + 3), (60, 60, 60), -1)
    bw = int((w - 12) * progress)
    cv2.rectangle(frame, (6, bar_y), (6 + bw, bar_y + 3), state_color, -1)


def draw_bev_radar(agents, trail_history, cam_x, cam_y, heading_deg, state,
                   bev_w=200, bev_h=200, max_range=25.0):
    """Draw a bird's-eye-view radar showing agent positions relative to camera.

    The camera's forward direction (heading) points up in the BEV.
    This matches mqtt_bridge's BEV where the camera is at the origin
    and forward = +x (displayed as up).
    """
    bev = np.full((bev_h, bev_w, 3), 30, dtype=np.uint8)

    # Camera at bottom center
    cx = bev_w // 2
    cy = int(bev_h * 0.85)
    scale = bev_h * 0.75 / max_range

    # Heading rotation: transform world offset to camera-relative (fwd, right)
    h_rad = heading_deg * pi / 180.0
    cos_h, sin_h = cos(h_rad), sin(h_rad)

    def world_to_bev(wx, wy):
        """World position -> BEV pixel. Forward = up, right = right."""
        dx = wx - cam_x
        dy = wy - cam_y
        # Rotate to camera frame: fwd = along heading, right = perpendicular
        fwd = dx * cos_h + dy * sin_h
        right = dx * sin_h - dy * cos_h
        bx = int(cx + right * scale)
        by = int(cy - fwd * scale)
        return bx, by

    # Range rings
    for r_m in [5, 10, 15, 20, 25]:
        r_px = int(r_m * scale)
        cv2.circle(bev, (cx, cy), r_px, (50, 50, 50), 1)
        if r_m % 10 == 0:
            label_y = cy - r_px
            if 5 < label_y < bev_h - 5:
                cv2.putText(bev, f"{r_m}m", (cx + 3, label_y + 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.25, (80, 80, 80), 1)

    # Camera marker
    cv2.circle(bev, (cx, cy), 3, (255, 255, 255), -1)

    # Agents
    for a in agents:
        bev_x, bev_y = world_to_bev(a['x'], a['y'])

        if not (0 <= bev_x < bev_w and 0 <= bev_y < bev_h):
            continue

        is_person = a['type'] == 'person'
        color = (80, 80, 255) if is_person else (80, 255, 80)

        # Trail
        trail = trail_history.get(a['id'], [])
        if len(trail) >= 2:
            for i in range(max(0, len(trail) - 20), len(trail) - 1):
                tx1, ty1 = world_to_bev(trail[i][0], trail[i][1])
                tx2, ty2 = world_to_bev(trail[i+1][0], trail[i+1][1])
                if (0 <= tx1 < bev_w and 0 <= ty1 < bev_h and
                    0 <= tx2 < bev_w and 0 <= ty2 < bev_h):
                    cv2.line(bev, (tx1, ty1), (tx2, ty2),
                             (color[0]//3, color[1]//3, color[2]//3), 1)

        cv2.circle(bev, (bev_x, bev_y), 4, color, -1)
        label = f"P{a['id']}" if is_person else f"B{a['id']}"
        cv2.putText(bev, label, (bev_x + 5, bev_y - 3),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.25, color, 1)

    # State indicator
    state_color = STATE_COLORS.get(state, (128, 128, 128))
    cv2.rectangle(bev, (2, 2), (bev_w - 3, 12), state_color, -1)
    cv2.putText(bev, STATE_LABELS.get(state, state.upper()), (5, 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.28, (255, 255, 255), 1)

    return bev


def draw_sensor_boundary(frame, renderer):
    """Draw horizontal lines showing the physical sensor boundary.

    Pixels above the top line and below the bottom line fall outside
    the 3840x2160 sensor and would not be captured.
    """
    v_min = max(0, renderer.sensor_v_min)
    v_max = min(frame.shape[0] - 1, renderer.sensor_v_max)
    w = frame.shape[1]
    color = (0, 140, 140)
    cv2.line(frame, (0, v_min), (w, v_min), color, 1)
    cv2.line(frame, (0, v_max), (w, v_max), color, 1)
    # Dim the outside regions
    if v_min > 0:
        frame[:v_min] = frame[:v_min] // 3
    if v_max < frame.shape[0]:
        frame[v_max:] = frame[v_max:] // 3


def draw_error_vector(frame, renderer, true_x, true_y, obs_x, obs_y, err_m):
    """Draw an error vector from true to observed position on the ground plane."""
    pts_true = np.array([[true_x, true_y, 0.2]])
    pts_obs = np.array([[obs_x, obs_y, 0.2]])
    px_true, v_true = renderer.project(pts_true)
    px_obs, v_obs = renderer.project(pts_obs)
    if v_true[0] and v_obs[0]:
        p1 = (int(px_true[0, 0]), int(px_true[0, 1]))
        p2 = (int(px_obs[0, 0]), int(px_obs[0, 1]))
        # Yellow dashed error line
        cv2.line(frame, p1, p2, (0, 255, 255), 1, cv2.LINE_AA)
        # Observed position marker (hollow circle)
        cv2.circle(frame, p2, 5, (0, 255, 255), 1, cv2.LINE_AA)
        # Error label
        mid = ((p1[0] + p2[0]) // 2, (p1[1] + p2[1]) // 2)
        cv2.putText(frame, f"{err_m:.2f}m", (mid[0] + 3, mid[1] - 3),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.25, (0, 255, 255), 1)


def render_camera_frame(renderer, agents, trail_history, prox_max,
                        scenario_name, t, duration, state, danger,
                        cam_label="", cam_x=0.0, cam_y=0.0,
                        heading_deg=90.0, error_data=None):
    """Render one frame from a single camera's perspective with BEV radar.

    If error_data is provided (list of dicts with true_x/y, obs_x/y, err_m),
    draws error vectors from true to observed positions.
    """
    frame = np.full((IMG_H, IMG_W, 3), 20, dtype=np.uint8)

    draw_ground_grid(frame, renderer, spacing=2.0, extent=25.0)
    draw_crosswalk(frame, renderer)

    # Proximity lines
    persons = [a for a in agents if a['type'] == 'person']
    bikes = [a for a in agents if a['type'] == 'bike']
    for b in bikes:
        for p in persons:
            d = sqrt((b['x'] - p['x'])**2 + (b['y'] - p['y'])**2)
            if d < prox_max:
                draw_proximity_line(frame, renderer, b['x'], b['y'], p['x'], p['y'])

    # Agents (true positions)
    for a in agents:
        trail = trail_history.get(a['id'], [])
        draw_agent_3d(frame, renderer, a, trail)

    # Error vectors (true → observed)
    if error_data:
        for ed in error_data:
            draw_error_vector(frame, renderer,
                              ed['true_x'], ed['true_y'],
                              ed['obs_x'], ed['obs_y'], ed['err_m'])

    # Sensor boundary (dims regions outside the physical 3840x2160 sensor)
    draw_sensor_boundary(frame, renderer)

    draw_hud(frame, scenario_name, t, duration, state, danger, cam_label)

    # BEV radar overlay (bottom-right)
    bev_w, bev_h = 180, 180
    bev = draw_bev_radar(agents, trail_history, cam_x, cam_y, heading_deg,
                         state, bev_w, bev_h)
    margin = 6
    y1 = IMG_H - bev_h - margin
    x1 = IMG_W - bev_w - margin
    roi = frame[y1:y1+bev_h, x1:x1+bev_w]
    cv2.addWeighted(bev, 0.85, roi, 0.15, 0, dst=roi)
    frame[y1:y1+bev_h, x1:x1+bev_w] = roi

    return frame


def render_scenario(scenario, scenario_idx, params_name='optimized',
                    fps=30, save=True, display=True, two_cam=False,
                    cam_height=None, cam_pitch=None, show_errors=False):
    """Render a scenario from the fisheye camera perspective.

    If show_errors is True, computes and visualizes fisheye bbox
    localization error for each agent, runs the pipeline on the
    *observed* (error-perturbed) positions, and saves all per-frame
    observation data to a JSON sidecar file.
    """
    params = PIPELINE_CONFIGS[params_name]
    profile = get_braking_profile(scenario)
    pipeline = make_pipeline(params, fps, profile=profile)
    duration = get_scenario_duration(scenario)
    dt = 1.0 / fps

    h1 = cam_height if cam_height is not None else CAM_POLE['h']
    h2 = cam_height if cam_height is not None else CAM_CROSS['h']

    # Scale calibration cx/cy offset from 3500x3500 to render resolution
    cx_off = _CALIB_CX_OFFSET * (IMG_W / 3500.0)
    cy_off = _CALIB_CY_OFFSET * (IMG_H / 3500.0)

    renderer1 = FisheyeRenderer(
        CAM_POLE['x'], CAM_POLE['y'], h1,
        CAM_POLE['heading_deg'], FOV_DEG, IMG_W, IMG_H,
        pitch_deg=cam_pitch, cx_offset=cx_off, cy_offset=cy_off)
    renderer2 = None
    if two_cam:
        renderer2 = FisheyeRenderer(
            CAM_CROSS['x'], CAM_CROSS['y'], h2,
            CAM_CROSS['heading_deg'], FOV_DEG, IMG_W, IMG_H,
            pitch_deg=cam_pitch, cx_offset=cx_off, cy_offset=cy_off)

    # Full-resolution camera for bbox error computation (only when show_errors)
    errors_cam = None
    if show_errors:
        calib = load_camera_calibration()
        import yaml
        with open('config.yaml') as _f:
            _cfg = yaml.safe_load(_f)
        _cc = _cfg.get('camera', {})
        errors_cam = FisheyeCamera(
            height=_cc.get('height', CAM_POLE['h']),
            fov_deg=calib['fov_deg'] if calib else FOV_DEG,
            radius_px=int(calib['radius_px']) if calib else 1750,
            frame_width=int(calib['frame_width']) if calib else 3500,
            frame_height=int(calib['frame_height']) if calib else 3500,
            pad_size=int(calib.get('pad_size', 3840)) if calib else 3840,
            cam_x=CAM_POLE['x'], cam_y=CAM_POLE['y'],
            pitch_deg=_cc.get('pitch_deg', 0.0),
            cx_override=calib['cx'] if calib else None,
            cy_override=calib['cy'] if calib else None,
        )

    out_w = IMG_W * 2 if two_cam else IMG_W
    out_h = IMG_H
    suffix = "_2cam" if two_cam else ""
    if show_errors:
        suffix += "_errors"
    if cam_height is not None or cam_pitch is not None:
        suffix += f"_h{h1:.1f}"
        if cam_pitch is not None:
            suffix += f"_p{cam_pitch:.0f}"
    out_path = f"sim_{scenario_idx:02d}_{scenario['name'].replace(' ', '_').lower()}{suffix}.mp4"

    writer = None
    if save:
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        writer = cv2.VideoWriter(out_path, fourcc, fps, (out_w, out_h))

    trail_history = {}
    obs_trail_history = {}  # trails for observed (error-perturbed) positions
    observation_log = []    # per-frame data for JSON export
    t = 0.0
    frame_count = 0

    cam_config = "2-cam" if two_cam else "1-cam"
    err_label = "+errors" if show_errors else ""
    print(f"  Rendering: {scenario['name']} ({cam_config}{err_label}, {duration:.1f}s)")

    while t <= duration:
        agents = []
        for ag_def in scenario['agents']:
            pos = interpolate_agent(ag_def, t)
            if pos is not None:
                agents.append(pos)
                aid = pos['id']
                if aid not in trail_history:
                    trail_history[aid] = []
                trail_history[aid].append((pos['x'], pos['y']))

        # Compute bbox errors and build observed agents
        frame_errors = []
        observed_agents = []
        if show_errors and errors_cam is not None:
            for a in agents:
                obj_name = 'pedestrian' if a['type'] == 'person' else 'cyclist'
                obj = OBJECT_TYPES.get(obj_name)
                if obj is None:
                    observed_agents.append(a)
                    continue
                heading = a.get('heading', 0.0)
                err = compute_bbox_error(errors_cam, a['x'], a['y'], obj, heading)
                if err is not None:
                    obs_x, obs_y = err['est_x'], err['est_y']
                    err_m = err['err_dist']
                    frame_errors.append({
                        'true_x': a['x'], 'true_y': a['y'],
                        'obs_x': obs_x, 'obs_y': obs_y,
                        'err_m': err_m,
                        'id': a['id'], 'type': a['type'],
                    })
                    # Build observed agent with perturbed position
                    obs_a = dict(a)
                    obs_a['x'] = obs_x
                    obs_a['y'] = obs_y
                    observed_agents.append(obs_a)
                    # Track observed positions
                    aid = a['id']
                    if aid not in obs_trail_history:
                        obs_trail_history[aid] = []
                    obs_trail_history[aid].append((obs_x, obs_y))
                else:
                    observed_agents.append(a)
        else:
            observed_agents = agents

        # Run pipeline on observed positions (error-perturbed when --errors)
        pipeline_agents = observed_agents if show_errors else agents
        state = run_pipeline_on_agents(pipeline, pipeline_agents, t,
                                       cam_x=CAM_POLE['x'], cam_y=CAM_POLE['y'])
        danger = compute_ground_truth_danger(scenario, t)

        # Log observation data
        if show_errors:
            frame_record = {
                'frame': frame_count,
                't': round(t, 4),
                'state': state,
                'danger': danger.is_danger,
                'severity': round(danger.severity, 4),
                'agents': [],
            }
            for a in agents:
                agent_rec = {
                    'id': a['id'], 'type': a['type'],
                    'true_x': round(a['x'], 4), 'true_y': round(a['y'], 4),
                }
                # Find matching error data
                for ed in frame_errors:
                    if ed['id'] == a['id']:
                        agent_rec['obs_x'] = round(ed['obs_x'], 4)
                        agent_rec['obs_y'] = round(ed['obs_y'], 4)
                        agent_rec['err_m'] = round(ed['err_m'], 4)
                        break
                frame_record['agents'].append(agent_rec)
            observation_log.append(frame_record)

        f1 = render_camera_frame(
            renderer1, agents, trail_history, params.prox_max,
            scenario['name'], t, duration, state, danger,
            cam_label="CAM-1" if two_cam else "",
            cam_x=CAM_POLE['x'], cam_y=CAM_POLE['y'],
            heading_deg=CAM_POLE['heading_deg'],
            error_data=frame_errors if show_errors else None)

        if two_cam and renderer2 is not None:
            f2 = render_camera_frame(
                renderer2, agents, trail_history, params.prox_max,
                scenario['name'], t, duration, state, danger,
                cam_label="CAM-2",
                cam_x=CAM_CROSS['x'], cam_y=CAM_CROSS['y'],
                heading_deg=CAM_CROSS['heading_deg'],
                error_data=frame_errors if show_errors else None)
            combined = np.hstack([f1, f2])
        else:
            combined = f1

        if writer is not None:
            writer.write(combined)

        if display:
            cv2.imshow("Scenario Simulation", combined)
            if cv2.waitKey(1) == ord('q'):
                break

        t += dt
        frame_count += 1

    if writer is not None:
        writer.release()
        print(f"    Saved: {out_path} ({frame_count} frames)")

    # Save observation log
    if show_errors and observation_log:
        log_path = out_path.replace('.mp4', '_observations.json')
        with open(log_path, 'w') as f:
            json.dump({
                'scenario': scenario['name'],
                'scenario_idx': scenario_idx,
                'fps': fps,
                'n_frames': frame_count,
                'cam_x': CAM_POLE['x'], 'cam_y': CAM_POLE['y'],
                'frames': observation_log,
            }, f, indent=2)
        print(f"    Observations: {log_path}")

    return out_path


def main():
    parser = argparse.ArgumentParser(description='3D fisheye scenario visualizer')
    parser.add_argument('--scenarios', nargs='+', type=int, default=None,
                        help='Scenario indices (0-indexed)')
    parser.add_argument('--fps', type=int, default=30)
    parser.add_argument('--pipeline', type=str, default='optimized')
    parser.add_argument('--two-cam', action='store_true',
                        help='Render split-screen two-camera view')
    parser.add_argument('--height', type=float, default=None,
                        help='Camera height in meters (default: 5.5)')
    parser.add_argument('--pitch', type=float, default=None,
                        help='Camera pitch in degrees (negative=down, default: auto)')
    parser.add_argument('--errors', action='store_true',
                        help='Show bbox localization error vectors and save observation data')
    parser.add_argument('--no-save', action='store_true')
    parser.add_argument('--no-display', action='store_true')
    args = parser.parse_args()

    indices = args.scenarios if args.scenarios else list(range(len(SCENARIOS)))

    modes = [False]
    if args.two_cam:
        modes = [False, True]

    for two_cam in modes:
        cam_config = "two-cam" if two_cam else "single-cam"
        err_label = " +errors" if args.errors else ""
        print(f"Rendering {len(indices)} scenarios, {cam_config}{err_label}\n")

        for idx in indices:
            if 0 <= idx < len(SCENARIOS):
                render_scenario(SCENARIOS[idx], idx,
                                params_name=args.pipeline,
                                fps=args.fps,
                                save=not args.no_save,
                                display=not args.no_display,
                                two_cam=two_cam,
                                cam_height=args.height,
                                cam_pitch=args.pitch,
                                show_errors=args.errors)

    if not args.no_display:
        cv2.destroyAllWindows()
    print("\nDone.")


if __name__ == '__main__':
    main()
