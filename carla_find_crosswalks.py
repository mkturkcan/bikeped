#!/usr/bin/env python3
"""
CARLA crosswalk finder and world-point marker.

Connects to a running CARLA server, finds crosswalk locations,
and provides an interactive tool to mark world points.

Usage:
    # List all crosswalks in the current map
    python carla_find_crosswalks.py

    # List crosswalks in a specific map
    python carla_find_crosswalks.py --map Town10HD_Opt

    # Interactive mode: click in the spectator view to mark points
    python carla_find_crosswalks.py --interactive

    # Move spectator to a specific crosswalk and print coordinates
    python carla_find_crosswalks.py --goto 3

    # Export crosswalk data to JSON
    python carla_find_crosswalks.py --export crosswalks.json
"""

import argparse
import json
import sys
import math
import time

try:
    import carla
except ImportError:
    print("Error: CARLA Python API not found.")
    sys.exit(1)


def get_crosswalks(world):
    """Extract crosswalk locations from the CARLA map.

    Returns a list of dicts with center, corners, and dimensions.
    """
    carla_map = world.get_map()

    # Method 1: get_crosswalks() returns a list of Location vertices
    # defining crosswalk polygons. Each crosswalk is a sequence of
    # 5 points (4 corners + repeat of first to close the polygon).
    raw = carla_map.get_crosswalks()
    if not raw:
        return []

    # Group into crosswalks (every 5 points)
    crosswalks = []
    i = 0
    while i + 4 <= len(raw):
        corners = [(raw[i+j].x, raw[i+j].y, raw[i+j].z) for j in range(4)]

        # Center
        cx = sum(c[0] for c in corners) / 4
        cy = sum(c[1] for c in corners) / 4
        cz = sum(c[2] for c in corners) / 4

        # Approximate dimensions
        d01 = math.sqrt((corners[0][0]-corners[1][0])**2 +
                         (corners[0][1]-corners[1][1])**2)
        d12 = math.sqrt((corners[1][0]-corners[2][0])**2 +
                         (corners[1][1]-corners[2][1])**2)
        width = min(d01, d12)
        length = max(d01, d12)

        # Road direction (along the longer edge)
        if d01 > d12:
            dx = corners[1][0] - corners[0][0]
            dy = corners[1][1] - corners[0][1]
        else:
            dx = corners[2][0] - corners[1][0]
            dy = corners[2][1] - corners[1][1]
        road_heading = math.degrees(math.atan2(dy, dx))

        crosswalks.append({
            'id': len(crosswalks),
            'center': (cx, cy, cz),
            'corners': corners,
            'width_m': width,
            'length_m': length,
            'road_heading_deg': road_heading,
        })
        i += 5

    return crosswalks


def move_spectator(world, x, y, z, yaw=0, pitch=-30):
    """Move the CARLA spectator camera to a location."""
    spectator = world.get_spectator()
    transform = carla.Transform(
        carla.Location(x=x, y=y, z=z + 15),  # elevated view
        carla.Rotation(pitch=pitch, yaw=yaw),
    )
    spectator.set_transform(transform)


def print_crosswalks(crosswalks):
    """Print crosswalk table."""
    print(f"\n{'ID':>4}  {'Center X':>10}  {'Center Y':>10}  {'Z':>6}  "
          f"{'Width':>6}  {'Length':>7}  {'Heading':>8}")
    print("-" * 65)
    for cw in crosswalks:
        cx, cy, cz = cw['center']
        print(f"{cw['id']:>4}  {cx:>10.1f}  {cy:>10.1f}  {cz:>6.1f}  "
              f"{cw['width_m']:>5.1f}m  {cw['length_m']:>6.1f}m  "
              f"{cw['road_heading_deg']:>7.1f}°")


def suggest_scenario_origin(cw):
    """Suggest origin-x and origin-y for carla_scenario.py based on a crosswalk."""
    cx, cy, cz = cw['center']
    heading_rad = math.radians(cw['road_heading_deg'])

    # Our scenario has the crosswalk at x=5, camera at (5, -13).
    # The camera is perpendicular to the road, offset by road half-width.
    # We want the CARLA crosswalk center to map to approximately (5, 0)
    # in our scenario coordinates.

    # Scenario origin = crosswalk center offset by (-5, 0) in scenario space
    # But scenario space is rotated relative to CARLA by the road heading.
    cos_h = math.cos(heading_rad)
    sin_h = math.sin(heading_rad)

    # Offset of -5 in scenario X (perpendicular to road in CARLA)
    # Scenario X is perpendicular to road, Scenario Y is along road
    origin_x = cx - 5.0 * cos_h
    origin_y = cy - 5.0 * sin_h

    return origin_x, origin_y, cw['road_heading_deg']


def interactive_mode(world):
    """Let the user explore and print spectator coordinates."""
    print("\nInteractive mode. Move the spectator in CARLA and press Enter here")
    print("to print the current position. Type 'q' to quit.\n")

    points = []
    while True:
        try:
            cmd = input("  [Enter=print position, q=quit, s=save] > ").strip()
        except EOFError:
            break

        if cmd == 'q':
            break

        spectator = world.get_spectator()
        t = spectator.get_transform()
        loc = t.location
        rot = t.rotation

        if cmd == 's' and points:
            path = 'carla_marked_points.json'
            with open(path, 'w') as f:
                json.dump(points, f, indent=2)
            print(f"  Saved {len(points)} points to {path}")
            continue

        point = {
            'x': round(loc.x, 2),
            'y': round(loc.y, 2),
            'z': round(loc.z, 2),
            'yaw': round(rot.yaw, 1),
            'pitch': round(rot.pitch, 1),
        }
        points.append(point)
        print(f"  Point {len(points)}: x={loc.x:.2f}, y={loc.y:.2f}, z={loc.z:.2f}, "
              f"yaw={rot.yaw:.1f}, pitch={rot.pitch:.1f}")

    return points


def main():
    parser = argparse.ArgumentParser(description='Find CARLA crosswalks')
    parser.add_argument('--host', default='localhost')
    parser.add_argument('--port', type=int, default=2000)
    parser.add_argument('--map', type=str, default=None,
                        help='Load this map (default: use current map)')
    parser.add_argument('--goto', type=int, default=None,
                        help='Move spectator to crosswalk N')
    parser.add_argument('--interactive', action='store_true',
                        help='Interactive point marking mode')
    parser.add_argument('--export', type=str, default=None,
                        help='Export crosswalk data to JSON file')
    args = parser.parse_args()

    client = carla.Client(args.host, args.port)
    client.set_timeout(10.0)

    if args.map:
        print(f"Loading map: {args.map}")
        world = client.load_world(args.map)
        time.sleep(2)
    else:
        world = client.get_world()

    map_name = world.get_map().name
    print(f"Map: {map_name}")

    crosswalks = get_crosswalks(world)
    print(f"Found {len(crosswalks)} crosswalks")

    if crosswalks:
        print_crosswalks(crosswalks)

        # Print suggested carla_scenario.py commands for a few crosswalks
        print(f"\nSuggested carla_scenario.py commands:")
        for cw in crosswalks[:5]:
            ox, oy, heading = suggest_scenario_origin(cw)
            print(f"  # Crosswalk {cw['id']} ({cw['width_m']:.0f}m wide, "
                  f"heading {cw['road_heading_deg']:.0f}°)")
            print(f"  python carla_scenario.py --origin-x {ox:.1f} "
                  f"--origin-y {oy:.1f} --scenario 1 --save")

    if args.goto is not None and args.goto < len(crosswalks):
        cw = crosswalks[args.goto]
        cx, cy, cz = cw['center']
        print(f"\nMoving spectator to crosswalk {args.goto}...")
        move_spectator(world, cx, cy, cz, yaw=cw['road_heading_deg'])

    if args.export and crosswalks:
        with open(args.export, 'w') as f:
            json.dump(crosswalks, f, indent=2)
        print(f"\nExported to {args.export}")

    if args.interactive:
        interactive_mode(world)


if __name__ == '__main__':
    main()
