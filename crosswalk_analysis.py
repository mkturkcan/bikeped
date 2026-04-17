"""
Crosswalk deployment analysis for the bikeped collision-warning system.

Evaluates decision pipeline performance as a function of road width
for one-camera and two-camera deployments with size-aware detection.

Geometry (matching index.html):
  - The road runs along the y-axis. Road width is measured along y.
  - The crosswalk crosses the road at x = 5m (along the y-axis).
  - Camera 1 is on a pole at (x=5, y=-road_half, h=5.5), heading 90 deg
    (facing +y, looking across the road).
  - Camera 2 is on a pole at (x=5, y=+road_half, h=5.5), heading 270 deg
    (facing -y, looking back across the road).
  - Cyclists approach along the x-axis toward the crosswalk.

  The distance from each camera to a cyclist at (bike_x, bike_y) is:
    dist = sqrt((bike_x - 5)^2 + (bike_y - cam_y)^2 + h^2)

  Two-camera fusion: P(detect) = 1 - (1-AR_cam1) * (1-AR_cam2).

Usage:
    python crosswalk_analysis.py
"""

import numpy as np
from math import sqrt, atan, atan2
from pathlib import Path
from decision_testbench import (
    FisheyeCamera, OBJECT_TYPES, UncertaintyParams, object_3d_corners,
    PIPELINE_CONFIGS, get_scenario_duration, interpolate_agent,
    compute_ground_truth_danger, make_pipeline, run_pipeline_on_agents,
    load_uncertainty_from_cache,
)

FEET_TO_METERS = 0.3048
CAM_X = 5.0             # cameras are at the crosswalk x-position
CROSSWALK_X = 5.0

ROAD_WIDTHS_FT = [30, 40, 50, 60]

# Camera height bounds (NYC traffic infrastructure)
# Pedestrian signal: 2.4-3.7m, traffic signal: 3.7-5.5m, mast arm: up to 7.6m
CAM_HEIGHT_MIN = 2.5    # meters (low pedestrian signal)
CAM_HEIGHT_MAX = 7.5    # meters (top of mast arm)
CAM_HEIGHT_DEFAULT = 5.5

# Pitch bounds: 0 = horizontal, negative = looking down
CAM_PITCH_MIN = -60.0   # degrees
CAM_PITCH_MAX = 0.0     # degrees

PED_SPEED_MS = 1.3
BIKE_SPEED_MS = 8.3     # ~30 km/h


def make_cam_at(cam_y, height=CAM_HEIGHT_DEFAULT, pitch_deg=None):
    """Create a FisheyeCamera positioned on a pole at the crosswalk edge."""
    return FisheyeCamera(
        height=height, fov_deg=220.0, cam_x=CAM_X, cam_y=cam_y,
        pitch_deg=pitch_deg)


def size_aware_ar(cam, obj_name, obj_x, obj_y, unc):
    """Get size-aware AR using full 8-corner fisheye projection."""
    obj = OBJECT_TYPES.get(obj_name)
    if obj is None:
        return 0.0
    area = cam.projected_bbox_area(obj_x, obj_y, obj)
    if area <= 0:
        return 0.0
    sized = unc.for_class(obj_name)
    return sized.for_area(area).ar


def make_scenario(road_width_m):
    """Create a bike-approaches-ped scenario for a given road width.

    Pedestrian walks along the crosswalk (y-axis) at x=5.
    Cyclist approaches along x toward the crosswalk center (y~0).
    """
    half = road_width_m / 2.0
    ped_time = road_width_m / PED_SPEED_MS

    bike_start_x = CROSSWALK_X + 20.0
    bike_end_x = CROSSWALK_X - 5.0
    bike_travel = bike_start_x - bike_end_x
    bike_time = bike_travel / BIKE_SPEED_MS
    bike_start_t = 1.0

    return {
        'name': f'Road {road_width_m:.1f}m',
        'expected_dominant': 'alert',
        'agents': [
            {'type': 'person', 'id': 1, 'path': [
                {'x': CROSSWALK_X, 'y': -half, 't': 0},
                {'x': CROSSWALK_X, 'y': 0, 't': ped_time / 2},
                {'x': CROSSWALK_X, 'y': half, 't': ped_time},
            ]},
            {'type': 'bike', 'id': 2, 'path': [
                {'x': bike_start_x, 'y': 0.5, 't': bike_start_t},
                {'x': CROSSWALK_X, 'y': 0.1, 't': bike_start_t + bike_time * 0.8},
                {'x': bike_end_x, 'y': -0.1, 't': bike_start_t + bike_time},
            ]},
        ],
    }


def run_mc(scenario, n_cameras, road_width_m, unc=None,
           cam_height=CAM_HEIGHT_DEFAULT, cam_pitch=None,
           n_trials=30, fps=30):
    """Run Monte Carlo for a scenario with 1 or 2 cameras."""
    if unc is None:
        unc = UncertaintyParams()
    params = PIPELINE_CONFIGS['optimized']
    duration = get_scenario_duration(scenario)
    dt = 1.0 / fps

    half = road_width_m / 2.0
    cam1 = make_cam_at(-half, height=cam_height, pitch_deg=cam_pitch)
    cam2 = make_cam_at(half, height=cam_height, pitch_deg=cam_pitch) if n_cameras == 2 else None

    all_sens = []
    all_sev_fn = []

    for trial in range(n_trials):
        rng = np.random.default_rng(seed=trial)
        pipeline = make_pipeline(params, fps)

        states_list = []
        dangers_list = []
        severities_list = []

        t = 0.0
        while t <= duration:
            agents = []
            for ag_def in scenario['agents']:
                pos = interpolate_agent(ag_def, t)
                if pos is not None:
                    obj_name = 'pedestrian' if pos['type'] == 'person' else 'cyclist'

                    ar1 = size_aware_ar(cam1, obj_name,
                                        pos['x'], pos['y'], unc)

                    if cam2 is not None:
                        ar2 = size_aware_ar(cam2, obj_name,
                                            pos['x'], pos['y'], unc)
                        ar = 1.0 - (1.0 - ar1) * (1.0 - ar2)
                    else:
                        ar = ar1

                    if rng.random() > ar:
                        continue
                    agents.append(pos)

            state = run_pipeline_on_agents(pipeline, agents, t,
                                           cam_x=cam1.cam_x, cam_y=cam1.cam_y)
            assessment = compute_ground_truth_danger(scenario, t)

            states_list.append(state)
            dangers_list.append(assessment.is_danger)
            severities_list.append(assessment.severity)
            t += dt

        dangers = np.array(dangers_list)
        states = np.array(states_list)
        severities = np.array(severities_list)
        is_alert = states == 'alert'

        n_danger = int(np.sum(dangers))
        tp = int(np.sum(is_alert & dangers))
        sens = tp / max(n_danger, 1)
        all_sens.append(sens)

        sev_total = float(np.sum(severities[dangers])) if n_danger > 0 else 1.0
        sev_missed = float(np.sum(severities[~is_alert & dangers])) if n_danger > 0 else 0.0
        all_sev_fn.append(sev_missed / max(sev_total, 1e-6))

    return {
        'mean_sens': float(np.mean(all_sens)),
        'std_sens': float(np.std(all_sens)),
        'mean_sev_fn': float(np.mean(all_sev_fn)),
        'std_sev_fn': float(np.std(all_sev_fn)),
    }


def run_height_sweep(road_width_ft, unc, heights, n_cameras=1, n_trials=20):
    """Sweep camera height and return sensitivity for each."""
    width_m = road_width_ft * FEET_TO_METERS
    scenario = make_scenario(width_m)
    results = []
    for h in heights:
        r = run_mc(scenario, n_cameras, width_m, unc=unc,
                   cam_height=h, n_trials=n_trials)
        results.append((h, r))
    return results


def main():
    # Load AR curves from eval cache if available
    cache_path = 'eval_cache.json'
    unc = load_uncertainty_from_cache(cache_path, 'best')
    if unc is None:
        print(f'  Warning: {cache_path} not found, using flat AR defaults')
        unc = UncertaintyParams()

    print("=" * 100)
    print("  CROSSWALK DEPLOYMENT ANALYSIS -- Size-Aware Monte Carlo")
    print("  Optimized pipeline, 20 trials per configuration")
    print("=" * 100)

    # ── 1. AR profile at default height ──
    cam_default = make_cam_at(-13.0)
    cyclist = OBJECT_TYPES['cyclist']
    print(f"\n  Cyclist AR profile (camera at y=-13, h={CAM_HEIGHT_DEFAULT}m)")
    print(f"  {'Bike from CW':>12}  {'3D dist':>8}  {'BBox area':>10}  {'AR':>6}")
    print(f"  {'-' * 42}")
    for offset in [20, 15, 10, 8, 5, 3, 1, 0]:
        bx = CROSSWALK_X + offset
        d3d = cam_default.dist_to(bx, 0.0)
        area = cam_default.projected_bbox_area(bx, 0.0, cyclist)
        ar = size_aware_ar(cam_default, 'cyclist', bx, 0.0, unc)
        print(f"  {offset:>10}m  {d3d:>7.1f}m  {area:>10.0f}  {ar:>5.3f}")

    # ── 2. Road width sweep at default height ──
    print(f"\n  Road width sweep (h={CAM_HEIGHT_DEFAULT}m)")
    print(f"  {'Road':>6}  {'Config':>10}  {'Sensitivity':>16}  {'Sev-weighted FN':>18}")
    print(f"  {'-' * 56}")

    width_results = []
    for width_ft in ROAD_WIDTHS_FT:
        width_m = width_ft * FEET_TO_METERS
        scenario = make_scenario(width_m)
        r1 = run_mc(scenario, 1, width_m, unc=unc, n_trials=20)
        r2 = run_mc(scenario, 2, width_m, unc=unc, n_trials=20)
        width_results.append({'width_ft': width_ft, 'one_cam': r1, 'two_cam': r2})

        print(f"  {width_ft:>4}ft  {'1 camera':>10}  "
              f"{r1['mean_sens']:>5.1%} +/- {r1['std_sens']:>4.1%}  "
              f"{r1['mean_sev_fn']:>6.1%} +/- {r1['std_sev_fn']:>4.1%}")
        print(f"  {'':>6}  {'2 cameras':>10}  "
              f"{r2['mean_sens']:>5.1%} +/- {r2['std_sens']:>4.1%}  "
              f"{r2['mean_sev_fn']:>6.1%} +/- {r2['std_sev_fn']:>4.1%}")

    # ── 3. Joint height + pitch optimization ──
    heights = [2.5, 3.0, 3.5, 4.0, 4.5, 5.0, 5.5, 6.0, 7.0, 7.5]
    pitches = [-15, -20, -25, -30, -35, -40, -45, None]  # None = auto

    test_road_ft = 40
    width_m = test_road_ft * FEET_TO_METERS
    scenario = make_scenario(width_m)

    print(f"\n{'=' * 100}")
    print(f"  JOINT HEIGHT + PITCH OPTIMIZATION (road: {test_road_ft}ft)")
    print(f"  Heights: {', '.join(f'{h}m' for h in heights)}")
    print(f"  Pitches: {', '.join(f'{p}' + 'deg' if p else 'auto' for p in pitches)}")
    print(f"{'=' * 100}")

    # 1-camera grid search
    print(f"\n  1-CAMERA: height x pitch -> sensitivity")
    header = f"  {'h\\p':>5}"
    for p in pitches:
        label = f"{p}d" if p is not None else "auto"
        header += f"  {label:>6}"
    print(header)
    print(f"  {'-' * (7 + 8 * len(pitches))}")

    best_1 = (0, None, 0.0)
    best_2 = (0, None, 0.0)

    for h in heights:
        row = f"  {h:>4.1f}m"
        for p in pitches:
            r = run_mc(scenario, 1, width_m, unc=unc,
                       cam_height=h, cam_pitch=p, n_trials=15)
            row += f"  {r['mean_sens']:>5.1%}"
            if r['mean_sens'] > best_1[2]:
                best_1 = (h, p, r['mean_sens'])
        print(row)

    p_label = f"{best_1[1]}deg" if best_1[1] is not None else "auto"
    print(f"\n  Best 1-cam: h={best_1[0]}m, pitch={p_label} -> {best_1[2]:.1%}")

    # 2-camera: test a subset (best heights from 1-cam neighborhood)
    print(f"\n  2-CAMERA: height x pitch -> sensitivity")
    print(header)
    print(f"  {'-' * (7 + 8 * len(pitches))}")

    for h in heights:
        row = f"  {h:>4.1f}m"
        for p in pitches:
            r = run_mc(scenario, 2, width_m, unc=unc,
                       cam_height=h, cam_pitch=p, n_trials=15)
            row += f"  {r['mean_sens']:>5.1%}"
            if r['mean_sens'] > best_2[2]:
                best_2 = (h, p, r['mean_sens'])
        print(row)

    p_label = f"{best_2[1]}deg" if best_2[1] is not None else "auto"
    print(f"\n  Best 2-cam: h={best_2[0]}m, pitch={p_label} -> {best_2[2]:.1%}")

    # ── 4. Best config across road widths ──
    print(f"\n{'=' * 80}")
    print(f"  OPTIMAL CONFIG PER ROAD WIDTH")
    print(f"{'=' * 80}")
    print(f"\n  {'Road':>6}  {'h':>5}  {'pitch':>6}  {'1-cam':>8}  "
          f"{'h':>5}  {'pitch':>6}  {'2-cam':>8}")
    print(f"  {'-' * 52}")

    # Use a coarser grid for the per-width sweep
    h_coarse = [2.5, 3.0, 3.5, 4.0, 5.0, 5.5, 7.5]
    p_coarse = [-20, -30, -40, None]

    for width_ft in ROAD_WIDTHS_FT:
        wm = width_ft * FEET_TO_METERS
        sc = make_scenario(wm)
        b1 = (0, None, 0.0)
        b2 = (0, None, 0.0)
        for h in h_coarse:
            for p in p_coarse:
                r1 = run_mc(sc, 1, wm, unc=unc, cam_height=h, cam_pitch=p,
                            n_trials=15)
                r2 = run_mc(sc, 2, wm, unc=unc, cam_height=h, cam_pitch=p,
                            n_trials=15)
                if r1['mean_sens'] > b1[2]:
                    b1 = (h, p, r1['mean_sens'])
                if r2['mean_sens'] > b2[2]:
                    b2 = (h, p, r2['mean_sens'])
        p1 = f"{b1[1]}d" if b1[1] is not None else "auto"
        p2 = f"{b2[1]}d" if b2[1] is not None else "auto"
        print(f"  {width_ft:>4}ft  {b1[0]:>4.1f}m  {p1:>5}  {b1[2]:>7.1%}  "
              f"{b2[0]:>4.1f}m  {p2:>5}  {b2[2]:>7.1%}")

    # ── 5. Summary tables ──
    print(f"\n{'=' * 60}")
    print(f"  PAPER TABLE (default h={CAM_HEIGHT_DEFAULT}m)")
    print(f"{'=' * 60}")
    print(f"  {'Width':>8}  {'1-cam':>8}  {'2-cam':>8}  {'1-cam':>10}  {'2-cam':>10}")
    print(f"  {'(ft)':>8}  {'Sens':>8}  {'Sens':>8}  {'SevFN':>10}  {'SevFN':>10}")
    print(f"  {'-' * 50}")
    for r in width_results:
        print(f"  {r['width_ft']:>8}  "
              f"{r['one_cam']['mean_sens']:>7.1%}  "
              f"{r['two_cam']['mean_sens']:>7.1%}  "
              f"{r['one_cam']['mean_sev_fn']:>9.1%}  "
              f"{r['two_cam']['mean_sev_fn']:>9.1%}")


if __name__ == '__main__':
    main()
