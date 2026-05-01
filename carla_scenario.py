#!/usr/bin/env python3
"""
CARLA scenario visualizer for the bikeped collision-warning system.

Spawns pedestrians and cyclists from the decision testbench scenarios
in a CARLA intersection environment, captures frames from a fixed
pole-mounted camera, and optionally saves them as a video.

Requires a running CARLA server (0.9.14+).

Usage:
    python carla_scenario.py                          # scenario 1, live view
    python carla_scenario.py --scenario 7             # Fast Approach
    python carla_scenario.py --scenario 1 --save      # save frames to disk
    python carla_scenario.py --all --save             # all scenarios, save frames
    python carla_scenario.py --list                   # list available scenarios

Camera: single 120-degree FOV pinhole camera at the pole position.
A future version will fuse multiple cameras to approximate the fisheye view.
"""

import argparse
import cv2
import json
import os
import sys
import time
import math
import queue
import numpy as np

try:
    import carla
except ImportError:
    print("Error: CARLA Python API not found.")
    print("Install it from your CARLA distribution:")
    print("  pip install carla")
    print("  # or add <CARLA_ROOT>/PythonAPI/carla to PYTHONPATH")
    sys.exit(1)

from decision_testbench import (
    SCENARIOS, PIPELINE_CONFIGS, FisheyeCamera,
    interpolate_agent, get_scenario_duration,
    compute_ground_truth_danger, make_pipeline, run_pipeline_on_agents,
)

# ── Camera configuration ──
CAM_HEIGHT = 3.6576  # 12 ft
CAM_FOV = 120        # pinhole approximation (single camera)
CAM_WIDTH = 1920
CAM_HEIGHT_PX = 1080

# ── Default map ──
CARLA_MAP = 'Town10HD_Opt'

# ── Crosswalk-derived placement ──
# These are computed by load_crosswalk() from crosswalks.json.
# scenario_to_carla() uses them to map testbench coordinates to CARLA.

# Transformation: scenario (x, y) → CARLA (X, Y)
#   In our scenarios: crosswalk at x=5, pedestrians walk along Y, bikes along X.
#   Camera at (5, -road_half_width) looking across the crosswalk (+Y).
#
#   From the crosswalk corners we derive:
#     _road_axis:  unit vector along the long edge (road direction = our +X)
#     _cross_axis: unit vector along the short edge (crosswalk direction = our +Y)
#     _cam_carla:  CARLA position of camera (midpoint of one short edge, elevated)
#     _origin:     CARLA position corresponding to scenario (0, 0)

_road_axis = np.array([1.0, 0.0])   # unit vector: scenario +X in CARLA coords
_cross_axis = np.array([0.0, 1.0])  # unit vector: scenario +Y in CARLA coords
_origin = np.array([0.0, 0.0])      # scenario (0,0) in CARLA world
_origin_z = 0.3
_cam_carla = np.array([0.0, 0.0])   # camera position in CARLA world
_cam_yaw = 0.0                      # camera yaw in CARLA degrees


def load_crosswalk(crosswalk_id, crosswalks_path='crosswalks.json'):
    """Load a crosswalk from JSON and compute the full coordinate mapping.

    The camera is placed at the midpoint of one short edge (the edge
    closest to negative Y in scenario space, i.e. "behind" the crosswalk
    from the pedestrian's perspective), looking toward the crosswalk center.

    Scenario coordinate mapping:
        +X = along the road (long edge of crosswalk)
        +Y = across the road (short edge / pedestrian crossing direction)
        crosswalk center ≈ scenario (5, 0)
        camera ≈ scenario (5, -road_half_width)
    """
    global _road_axis, _cross_axis, _origin, _origin_z, _cam_carla, _cam_yaw

    with open(crosswalks_path) as f:
        crosswalks = json.load(f)

    cw = crosswalks[crosswalk_id]
    corners = [np.array(c[:2]) for c in cw['corners']]

    # Identify short and long edges
    # Edges: 0-1, 1-2, 2-3, 3-0
    edges = []
    for i in range(4):
        j = (i + 1) % 4
        length = np.linalg.norm(corners[j] - corners[i])
        mid = (corners[i] + corners[j]) / 2
        edges.append({'i': i, 'j': j, 'length': length, 'mid': mid,
                       'vec': corners[j] - corners[i]})

    edges.sort(key=lambda e: e['length'])
    short_edges = edges[:2]  # ~2.5m, the two sidewalk-side edges
    long_edges = edges[2:]   # ~19m, the two road-side edges

    center = np.array(cw['center'][:2])

    # Step 1: Camera at midpoint of one short edge (on the sidewalk).
    short0_mid = short_edges[0]['mid']
    short1_mid = short_edges[1]['mid']
    # Pick arbitrarily (first one); user can try the other with --crosswalk
    cam_edge = short0_mid

    # Step 2: cross_axis (our +Y) = camera look direction = toward center.
    # This is the direction pedestrians walk (across the road).
    cam_to_center = center - cam_edge
    cross_axis = cam_to_center / np.linalg.norm(cam_to_center)

    # Step 3: road_axis (our +X) = perpendicular to cross_axis.
    # Bikes approach along this direction (along the road).
    # 90° CW rotation so +X is to the right when facing +Y.
    road_axis = np.array([cross_axis[1], -cross_axis[0]])

    # Road half-width = distance from cam_edge to center
    road_half = np.linalg.norm(cam_to_center)

    # Camera yaw in CARLA
    cam_yaw = math.degrees(math.atan2(cross_axis[1], cross_axis[0]))

    # Scenario mapping:
    #   scenario (5, 0) = crosswalk center
    #   scenario +X = road_axis (bikes approach along the road)
    #   scenario +Y = cross_axis (peds cross the road)
    #   camera at scenario (5, -road_half)
    origin = center - 5.0 * road_axis

    _road_axis = road_axis
    _cross_axis = cross_axis
    _origin = origin
    _origin_z = cw['center'][2] if len(cw['center']) > 2 else 0.0
    _cam_carla = cam_edge
    _cam_yaw = cam_yaw

    print(f"  Crosswalk {crosswalk_id}: center=({center[0]:.1f}, {center[1]:.1f}), "
          f"road_width={2*road_half:.1f}m")
    print(f"  Road axis (bike +X): ({road_axis[0]:.3f}, {road_axis[1]:.3f})")
    print(f"  Cross axis (ped +Y): ({cross_axis[0]:.3f}, {cross_axis[1]:.3f})")
    print(f"  Camera at CARLA ({cam_edge[0]:.1f}, {cam_edge[1]:.1f}), "
          f"yaw={cam_yaw:.1f}°, looking across {2*road_half:.0f}m road")

    return cw


def scenario_to_carla(x, y, z=0.0):
    """Convert scenario world coordinates to CARLA world coordinates."""
    carla_xy = _origin + x * _road_axis + y * _cross_axis
    return carla.Location(
        x=float(carla_xy[0]),
        y=float(carla_xy[1]),
        z=_origin_z + z,
    )


def heading_to_carla_rotation(heading_rad):
    """Convert scenario heading (radians, 0=+X) to CARLA rotation.

    Scenario heading is relative to the road axis; we rotate it into
    CARLA world coordinates using the road_axis direction.
    """
    # Road axis angle in CARLA world
    road_angle = math.atan2(_road_axis[1], _road_axis[0])
    yaw_world = math.degrees(road_angle + heading_rad)
    return carla.Rotation(pitch=0, yaw=yaw_world, roll=0)


class CarlaScenarioRunner:
    """Runs a testbench scenario in CARLA."""

    def __init__(self, host='localhost', port=2000, fps=30, animate_vehicles=False,
                 ped_model='random', bike_model='vehicle.diamondback.century',
                 use_wheelchair=False):
        self.client = carla.Client(host, port)
        self.client.set_timeout(120.0)
        self.fps = fps
        self.dt = 1.0 / fps
        self.animate_vehicles = animate_vehicles
        self.ped_model = ped_model
        self.bike_model = bike_model
        self.use_wheelchair = use_wheelchair
        self.world = None
        self.camera = None
        self.actors = []
        self.controllers = []
        self.image_queue = queue.Queue()

    def setup_world(self):
        """Load map and configure synchronous mode."""
        print(f"  Loading map: {CARLA_MAP}")
        self.world = self.client.load_world(CARLA_MAP)

        settings = self.world.get_settings()
        settings.synchronous_mode = True
        settings.fixed_delta_seconds = self.dt
        self.world.apply_settings(settings)

        # Set weather to clear
        weather = carla.WeatherParameters.ClearNoon
        self.world.set_weather(weather)

        # Tick once to apply settings
        self.world.tick()

    def spawn_camera(self):
        """Spawn a fixed RGB camera at the crosswalk-derived pole position."""
        bp_lib = self.world.get_blueprint_library()
        camera_bp = bp_lib.find('sensor.camera.rgb')
        camera_bp.set_attribute('image_size_x', str(CAM_WIDTH))
        camera_bp.set_attribute('image_size_y', str(CAM_HEIGHT_PX))
        camera_bp.set_attribute('fov', str(CAM_FOV))
        camera_bp.set_attribute('sensor_tick', '0.0')

        # Camera position: midpoint of short edge, elevated
        cam_location = carla.Location(
            x=float(_cam_carla[0]),
            y=float(_cam_carla[1]),
            z=_origin_z + CAM_HEIGHT,
        )
        cam_rotation = carla.Rotation(
            pitch=-10.0,  # slight downward tilt to see the crosswalk
            yaw=_cam_yaw,
            roll=0.0,
        )
        cam_transform = carla.Transform(cam_location, cam_rotation)

        self.camera = self.world.spawn_actor(camera_bp, cam_transform)
        self.camera.listen(self.image_queue.put)
        self.actors.append(self.camera)
        print(f"  Camera spawned at CARLA ({_cam_carla[0]:.1f}, {_cam_carla[1]:.1f}, "
              f"{_origin_z + CAM_HEIGHT:.1f}), yaw={_cam_yaw:.1f}°, FOV={CAM_FOV}")

    def _project_bbox(self, actor, cam_transform):
        """Project an actor's 3D bounding box into 2D camera pixels.

        Returns (x1, y1, x2, y2) or None if not visible.
        """
        bb = actor.bounding_box
        actor_transform = actor.get_transform()

        # 8 corners of the bounding box in local actor space
        extent = bb.extent  # half-sizes
        corners_local = [
            carla.Location(x=+extent.x, y=+extent.y, z=+extent.z),
            carla.Location(x=+extent.x, y=+extent.y, z=-extent.z),
            carla.Location(x=+extent.x, y=-extent.y, z=+extent.z),
            carla.Location(x=+extent.x, y=-extent.y, z=-extent.z),
            carla.Location(x=-extent.x, y=+extent.y, z=+extent.z),
            carla.Location(x=-extent.x, y=+extent.y, z=-extent.z),
            carla.Location(x=-extent.x, y=-extent.y, z=+extent.z),
            carla.Location(x=-extent.x, y=-extent.y, z=-extent.z),
        ]

        # Transform corners to world space, then to camera space
        # Build camera intrinsic matrix
        fov_rad = math.radians(CAM_FOV)
        focal = CAM_WIDTH / (2.0 * math.tan(fov_rad / 2.0))
        cx = CAM_WIDTH / 2.0
        cy = CAM_HEIGHT_PX / 2.0

        # Camera world transform
        cam_loc = cam_transform.location
        cam_fwd = cam_transform.get_forward_vector()
        cam_right = cam_transform.get_right_vector()
        cam_up = cam_transform.get_up_vector()

        u_vals = []
        v_vals = []

        for corner in corners_local:
            # Transform to world: apply bbox center offset, then actor transform
            world_pt = actor_transform.transform(
                bb.location + corner)

            # Vector from camera to point
            dx = world_pt.x - cam_loc.x
            dy = world_pt.y - cam_loc.y
            dz = world_pt.z - cam_loc.z

            # Project to camera frame
            fwd = dx * cam_fwd.x + dy * cam_fwd.y + dz * cam_fwd.z
            right = dx * cam_right.x + dy * cam_right.y + dz * cam_right.z
            up = dx * cam_up.x + dy * cam_up.y + dz * cam_up.z

            if fwd < 0.1:  # behind camera
                continue

            u = cx + focal * right / fwd
            v = cy - focal * up / fwd

            if 0 <= u < CAM_WIDTH and 0 <= v < CAM_HEIGHT_PX:
                u_vals.append(u)
                v_vals.append(v)

        if len(u_vals) < 2:
            return None

        return (min(u_vals), min(v_vals), max(u_vals), max(v_vals))

    def spawn_agents(self, scenario, t):
        """Spawn or update agents for the current timestep.

        Returns list of active agent dicts for pipeline evaluation.
        """
        # For simplicity, we use CARLA's teleport approach:
        # spawn actors once, then teleport them each frame.
        # This avoids the complexity of AI controllers for
        # scripted trajectories.
        pass  # Handled in run_scenario

    def run_scenario(self, scenario, scenario_idx, save_frames=False,
                     draw_bboxes=False, output_dir='carla_output'):
        """Run a single scenario in CARLA."""
        name = scenario['name']
        duration = get_scenario_duration(scenario)
        print(f"\n  Running: {name} ({duration:.1f}s)")

        if save_frames:
            scene_dir = os.path.join(output_dir,
                                     f"{scenario_idx:02d}_{name.replace(' ', '_').lower()}")
            os.makedirs(scene_dir, exist_ok=True)

        # Spawn actors for each agent
        bp_lib = self.world.get_blueprint_library()
        agent_actors = {}     # id -> actor
        agent_types = {}      # id -> 'person' or 'bike'

        import random as _rng

        for ag_def in scenario['agents']:
            pos = interpolate_agent(ag_def, ag_def['path'][0]['t'])
            if pos is None:
                continue

            if ag_def['type'] == 'person':
                if self.ped_model == 'random':
                    all_peds = bp_lib.filter('walker.pedestrian.*')
                    if self.use_wheelchair:
                        # Filter to only wheelchair-capable models
                        capable = [bp for bp in all_peds if bp.has_attribute('can_use_wheelchair')]
                        bp_candidates = [capable[_rng.randint(0, len(capable) - 1)]] if capable else [all_peds[_rng.randint(0, len(all_peds) - 1)]]
                    else:
                        bp_candidates = [all_peds[_rng.randint(0, len(all_peds) - 1)]]
                else:
                    bp_candidates = bp_lib.filter(self.ped_model)
                    if len(bp_candidates) == 0:
                        bp_candidates = bp_lib.filter('walker.pedestrian.*')
            else:
                if self.bike_model == 'random':
                    # Pool every bike-like blueprint; pick uniformly per agent.
                    pool = []
                    for pat in ('vehicle.diamondback.century',
                                'vehicle.bh.crossbike',
                                'vehicle.gazelle.omafiets',
                                'vehicle.vespa.zx125'):
                        pool.extend(bp_lib.filter(pat))
                    if not pool:
                        pool = list(bp_lib.filter('vehicle.*bicycle*'))
                    bp_candidates = ([pool[_rng.randint(0, len(pool) - 1)]]
                                     if pool else [])
                else:
                    bp_candidates = bp_lib.filter(self.bike_model)
                    if len(bp_candidates) == 0:
                        # Fallback chain
                        for fallback in ('vehicle.diamondback.century',
                                         'vehicle.bh.crossbike',
                                         'vehicle.*bicycle*',
                                         'vehicle.vespa.zx125'):
                            bp_candidates = bp_lib.filter(fallback)
                            if len(bp_candidates) > 0:
                                break

            if len(bp_candidates) == 0:
                print(f"    Warning: no blueprint found for {ag_def['type']}, skipping")
                continue

            bp = bp_candidates[0]
            # Enable wheelchair if requested and supported
            if self.use_wheelchair and ag_def['type'] == 'person':
                if bp.has_attribute('can_use_wheelchair'):
                    bp.set_attribute('use_wheelchair', 'True')
            loc = scenario_to_carla(pos['x'], pos['y'])
            # Pedestrians need extra z for physics settling; vehicles stay at ground
            if ag_def['type'] == 'person':
                loc.z += 1.0
            else:
                loc.z += 0.3
            heading = pos.get('heading', 0.0)
            rot = heading_to_carla_rotation(heading)
            transform = carla.Transform(loc, rot)

            actor = self.world.try_spawn_actor(bp, transform)
            if actor is None:
                loc.z += 0.5
                transform = carla.Transform(loc, rot)
                actor = self.world.try_spawn_actor(bp, transform)

            if actor is not None:
                agent_actors[ag_def['id']] = actor
                agent_types[ag_def['id']] = ag_def['type']
                self.actors.append(actor)
                # Disable physics for vehicles so teleportation doesn't bounce
                if ag_def['type'] != 'person':
                    # actor.set_simulate_physics(False)
                    pass
                label = 'Ped' if ag_def['type'] == 'person' else 'Bike'
                print(f"    Spawned {label} {ag_def['id']} ({bp.id})")
            else:
                print(f"    Failed to spawn agent {ag_def['id']}")

        # Run the scenario frame by frame
        pipeline = make_pipeline(PIPELINE_CONFIGS['default'], self.fps)
        annotations = []
        t = 0.0
        frame_num = 0

        while t <= duration:
            for ag_def in scenario['agents']:
                pos = interpolate_agent(ag_def, t)
                actor = agent_actors.get(ag_def['id'])
                if actor is None:
                    continue

                if pos is not None:
                    loc = scenario_to_carla(pos['x'], pos['y'])
                    is_person = agent_types[ag_def['id']] == 'person'
                    # Pedestrians get physics settling; bikes stay near ground
                    loc.z += 1.0 if is_person else 0.3
                    heading = pos.get('heading', 0.0)
                    rot = heading_to_carla_rotation(heading)

                    if is_person:
                        # Compute movement direction in CARLA world
                        pos_next = interpolate_agent(ag_def, t + self.dt)
                        if pos_next is not None:
                            dx = pos_next['x'] - pos['x']
                            dy = pos_next['y'] - pos['y']
                            speed = math.sqrt(dx*dx + dy*dy) / self.dt
                            # Direction in CARLA world coords
                            carla_next = scenario_to_carla(pos_next['x'], pos_next['y'])
                            carla_cur = scenario_to_carla(pos['x'], pos['y'])
                            dir_x = carla_next.x - carla_cur.x
                            dir_y = carla_next.y - carla_cur.y
                            dir_len = math.sqrt(dir_x**2 + dir_y**2)
                            if dir_len > 0.01:
                                dir_x /= dir_len
                                dir_y /= dir_len
                            direction = carla.Vector3D(x=dir_x, y=dir_y, z=0.0)
                            # Face the walking direction
                            walk_yaw = math.degrees(math.atan2(dir_y, dir_x))
                            rot = carla.Rotation(pitch=0, yaw=walk_yaw, roll=0)
                        else:
                            speed = 0.0
                            direction = carla.Vector3D(0, 0, 0)

                        # Teleport and apply animated walk control
                        actor.set_transform(carla.Transform(loc, rot))
                        control = carla.WalkerControl(
                            direction=direction,
                            speed=speed,
                        )
                        actor.apply_control(control)
                    else:
                        # Vehicles: teleport with correct facing direction
                        pos_next = interpolate_agent(ag_def, t + self.dt)
                        steer = 0.0
                        if pos_next is not None:
                            carla_next = scenario_to_carla(pos_next['x'], pos_next['y'])
                            carla_cur = scenario_to_carla(pos['x'], pos['y'])
                            dir_x = carla_next.x - carla_cur.x
                            dir_y = carla_next.y - carla_cur.y
                            dir_len = math.sqrt(dir_x**2 + dir_y**2)
                            if dir_len > 0.01:
                                ride_yaw = math.degrees(math.atan2(dir_y, dir_x))
                                # Steer proportional to yaw change
                                old_yaw = rot.yaw
                                yaw_diff = ride_yaw - old_yaw
                                # Normalize to [-180, 180]
                                yaw_diff = (yaw_diff + 180) % 360 - 180
                                steer = max(-1.0, min(1.0, yaw_diff / 30.0))
                                rot = carla.Rotation(pitch=0, yaw=ride_yaw, roll=0)
                        actor.set_transform(carla.Transform(loc, rot))
                        # Apply throttle to animate wheels/pedaling
                        if self.animate_vehicles:
                            control = carla.VehicleControl(
                                throttle=0.5,
                                steer=steer,
                                brake=0.0,
                            )
                            actor.apply_control(control)
                else:
                    actor.set_transform(carla.Transform(
                        carla.Location(x=-1000, y=-1000, z=-10)))

            # Step simulation
            self.world.tick()

            # Compute 2D bounding boxes if requested
            frame_bboxes = []
            if draw_bboxes:
                cam_transform = self.camera.get_transform()
                for ag_def in scenario['agents']:
                    actor = agent_actors.get(ag_def['id'])
                    if actor is None:
                        continue
                    pos = interpolate_agent(ag_def, t)
                    if pos is None:
                        continue

                    bbox_2d = self._project_bbox(actor, cam_transform)
                    if bbox_2d is not None:
                        x1, y1, x2, y2 = bbox_2d
                        label = 'pedestrian' if ag_def['type'] == 'person' else 'cyclist'
                        frame_bboxes.append({
                            'id': ag_def['id'],
                            'class': label,
                            'bbox': [int(x1), int(y1), int(x2), int(y2)],
                            'scenario_x': round(pos['x'], 3),
                            'scenario_y': round(pos['y'], 3),
                        })

                annotations.append({
                    'frame': frame_num,
                    't': round(t, 4),
                    'bboxes': frame_bboxes,
                })

            # Capture frame
            if save_frames:
                try:
                    image = self.image_queue.get(timeout=2.0)

                    if draw_bboxes and frame_bboxes:
                        # Convert to numpy for drawing
                        img_array = np.frombuffer(image.raw_data, dtype=np.uint8)
                        img_array = img_array.reshape((image.height, image.width, 4))[:, :, :3]
                        img_array = img_array.copy()

                        for bb in frame_bboxes:
                            x1, y1, x2, y2 = bb['bbox']
                            color = (80, 80, 255) if bb['class'] == 'pedestrian' else (80, 255, 80)
                            cv2.rectangle(img_array, (x1, y1), (x2, y2), color, 2)
                            lbl = f"{bb['class'][0].upper()}{bb['id']}"
                            cv2.putText(img_array, lbl, (x1, y1 - 5),
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

                        frame_path = os.path.join(scene_dir, f"frame_{frame_num:06d}.png")
                        cv2.imwrite(frame_path, img_array)
                    else:
                        frame_path = os.path.join(scene_dir, f"frame_{frame_num:06d}.png")
                        image.save_to_disk(frame_path)
                except queue.Empty:
                    print(f"    Warning: no image at frame {frame_num}")

            # Drain queue if not saving
            while not self.image_queue.empty():
                try:
                    self.image_queue.get_nowait()
                except queue.Empty:
                    break

            t += self.dt
            frame_num += 1

        print(f"    Completed: {frame_num} frames"
              + (f", saved to {scene_dir}" if save_frames else ""))

        # Save annotations JSON
        if draw_bboxes and annotations:
            ann_path = os.path.join(scene_dir, 'annotations.json')
            with open(ann_path, 'w') as f:
                json.dump({
                    'scenario': name,
                    'scenario_idx': scenario_idx,
                    'fps': self.fps,
                    'image_width': CAM_WIDTH,
                    'image_height': CAM_HEIGHT_PX,
                    'n_frames': frame_num,
                    'frames': annotations,
                }, f, indent=2)
            print(f"    Annotations: {ann_path}")

        # Clean up scenario actors (keep camera)
        for aid, actor in agent_actors.items():
            if actor.is_alive:
                actor.destroy()
            if actor in self.actors:
                self.actors.remove(actor)

        return frame_num

    def cleanup(self):
        """Destroy all actors and restore async mode."""
        for actor in self.actors:
            if actor.is_alive:
                actor.destroy()
        self.actors.clear()

        if self.world is not None:
            settings = self.world.get_settings()
            settings.synchronous_mode = False
            self.world.apply_settings(settings)


def frames_to_video(frame_dir, output_path, fps=30):
    """Convert a directory of frame PNGs to an MP4 video."""
    try:
        import cv2
    except ImportError:
        print(f"  cv2 not available, skipping video creation for {frame_dir}")
        return

    frames = sorted(f for f in os.listdir(frame_dir) if f.endswith('.png'))
    if not frames:
        return

    first = cv2.imread(os.path.join(frame_dir, frames[0]))
    h, w = first.shape[:2]
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    writer = cv2.VideoWriter(output_path, fourcc, fps, (w, h))

    for fname in frames:
        img = cv2.imread(os.path.join(frame_dir, fname))
        writer.write(img)
    writer.release()
    print(f"  Video saved: {output_path}")


def _set_map(map_name):
    global CARLA_MAP
    CARLA_MAP = map_name


def main():
    parser = argparse.ArgumentParser(
        description='Run testbench scenarios in CARLA')
    parser.add_argument('--scenario', type=int, default=1,
                        help='Scenario index (0-indexed, default: 1)')
    parser.add_argument('--all', action='store_true',
                        help='Run all scenarios')
    parser.add_argument('--save', action='store_true',
                        help='Save frames to disk')
    parser.add_argument('--bbox', action='store_true',
                        help='Draw 2D bounding boxes on saved frames and export annotations')
    parser.add_argument('--animate', action='store_true',
                        help='Animate vehicle wheels and pedaling (applies throttle with physics off)')
    parser.add_argument('--ped-model', type=str, default='random',
                        help='Pedestrian blueprint (default: random). '
                             'Use "random" or a specific ID like "walker.pedestrian.0037"')
    parser.add_argument('--wheelchair', action='store_true',
                        help='Use wheelchair pedestrian (walker.pedestrian.0028 with wheelchair attribute)')
    parser.add_argument('--bike-model', type=str, default='vehicle.diamondback.century',
                        help='Bike blueprint (default: vehicle.diamondback.century). '
                             'Use "random" to pick uniformly per bike from the full pool '
                             '(diamondback.century, bh.crossbike, gazelle.omafiets, vespa.zx125), '
                             'or a specific blueprint id.')
    parser.add_argument('--video', action='store_true',
                        help='Also create MP4 from saved frames')
    parser.add_argument('--output', type=str, default='carla_output',
                        help='Output directory (default: carla_output)')
    parser.add_argument('--host', type=str, default='localhost')
    parser.add_argument('--port', type=int, default=2000)
    parser.add_argument('--fps', type=int, default=30)
    parser.add_argument('--list', action='store_true',
                        help='List available scenarios and exit')
    parser.add_argument('--map', type=str, default='Town10HD_Opt',
                        help='CARLA map to load (default: Town10HD_Opt)')
    parser.add_argument('--crosswalk', type=int, default=8,
                        help='Crosswalk ID from crosswalks.json (default: 8)')
    parser.add_argument('--crosswalks-file', type=str, default='crosswalks.json',
                        help='Path to crosswalks.json')
    args = parser.parse_args()

    if args.list:
        print(f"Available scenarios ({len(SCENARIOS)} total):\n")
        for i, s in enumerate(SCENARIOS):
            n_agents = len(s['agents'])
            dur = get_scenario_duration(s)
            print(f"  {i:>2}: {s['name']:<28} {n_agents} agents, {dur:.1f}s")
        return

    _set_map(args.map)

    # Derive camera placement and coordinate mapping from crosswalk geometry
    print(f"Loading crosswalk {args.crosswalk} from {args.crosswalks_file}...")
    load_crosswalk(args.crosswalk, args.crosswalks_file)

    runner = CarlaScenarioRunner(host=args.host, port=args.port, fps=args.fps,
                                 animate_vehicles=args.animate,
                                 ped_model=args.ped_model,
                                 bike_model=args.bike_model,
                                 use_wheelchair=args.wheelchair)

    try:
        print("Connecting to CARLA...")
        runner.setup_world()
        runner.spawn_camera()

        if args.all:
            indices = range(len(SCENARIOS))
        else:
            indices = [args.scenario]

        for idx in indices:
            if 0 <= idx < len(SCENARIOS):
                runner.run_scenario(SCENARIOS[idx], idx,
                                    save_frames=args.save,
                                    draw_bboxes=args.bbox,
                                    output_dir=args.output)

                if args.save and args.video:
                    scene_dir = os.path.join(
                        args.output,
                        f"{idx:02d}_{SCENARIOS[idx]['name'].replace(' ', '_').lower()}")
                    video_path = os.path.join(
                        args.output,
                        f"carla_{idx:02d}_{SCENARIOS[idx]['name'].replace(' ', '_').lower()}.mp4")
                    frames_to_video(scene_dir, video_path, args.fps)

    except KeyboardInterrupt:
        print("\nInterrupted.")
    except Exception as e:
        print(f"\nError: {e}")
        raise
    finally:
        print("\nCleaning up...")
        runner.cleanup()

    print("Done.")


if __name__ == '__main__':
    main()
