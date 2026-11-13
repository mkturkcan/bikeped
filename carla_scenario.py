#!/usr/bin/env python3
"""Render testbench scenarios in CARLA with the flags of the original script.

The renderer lives in ``bikeped.sim``. This script accepts the options of the
single-file tool it replaces and forwards them to
``python -m bikeped.sim render``, which also documents the full set of views,
weather presets and output files.

Requires a running CARLA server.

Usage:
    python carla_scenario.py                          # scenario 1
    python carla_scenario.py --scenario 7             # Fast Approach
    python carla_scenario.py --scenario 1 --save      # also save frames to disk
    python carla_scenario.py --all --save             # all scenarios, save frames
    python carla_scenario.py --list                   # list available scenarios
"""

from __future__ import annotations

import argparse
import sys
from typing import List, Optional, Sequence

COMPATIBILITY_NOTE = "Kept for compatibility, has no effect."


def build_parser() -> argparse.ArgumentParser:
    """Argument parser with the options of the original script."""
    parser = argparse.ArgumentParser(
        description="Run testbench scenarios in CARLA. The options are translated to "
                    "'python -m bikeped.sim render'; use that command for views, weather, "
                    "ambient traffic and the other rendering options.")
    parser.add_argument("--scenario", type=int, default=1,
                        help="Scenario index in the conformance suite (0-indexed, default: 1)")
    parser.add_argument("--all", action="store_true",
                        help="Run all conformance scenarios")
    parser.add_argument("--save", action="store_true",
                        help="Save the frames as JPEG files next to the videos")
    parser.add_argument("--bbox", action="store_true",
                        help="Draw the overlay with boxes and decision state on the output")
    parser.add_argument("--animate", action="store_true",
                        help=COMPATIBILITY_NOTE + " Riders and walkers are always animated.")
    parser.add_argument("--ped-model", type=str, default=None,
                        help=COMPATIBILITY_NOTE + " Models come from the scenario's agents.")
    parser.add_argument("--wheelchair", action="store_true",
                        help=COMPATIBILITY_NOTE + " Models come from the scenario's agents.")
    parser.add_argument("--bike-model", type=str, default=None,
                        help=COMPATIBILITY_NOTE + " Models come from the scenario's agents.")
    parser.add_argument("--video", action="store_true",
                        help=COMPATIBILITY_NOTE + " A video is always written.")
    parser.add_argument("--output", type=str, default="carla_output",
                        help="Output directory (default: carla_output)")
    parser.add_argument("--host", type=str, default="localhost")
    parser.add_argument("--port", type=int, default=2000)
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--list", action="store_true",
                        help="List available scenarios and exit")
    parser.add_argument("--map", type=str, default="Town10HD_Opt",
                        help="CARLA map to load (default: Town10HD_Opt)")
    parser.add_argument("--crosswalk", type=int, default=8,
                        help="Crosswalk ID from crosswalks.json (default: 8)")
    parser.add_argument("--crosswalks-file", type=str, default="crosswalks.json",
                        help="Path to crosswalks.json")
    return parser


def _render_args(args: argparse.Namespace) -> List[str]:
    argv = ["render"]
    if args.all:
        argv += ["--suite", "conformance", "--all"]
    else:
        argv += ["--scenario", f"conformance:{args.scenario}"]
    argv += [
        "--host", args.host,
        "--port", str(args.port),
        "--fps", str(args.fps),
        "--map", args.map,
        "--crosswalk", str(args.crosswalk),
    ]
    if args.crosswalks_file != "crosswalks.json":
        argv += ["--crosswalks-file", args.crosswalks_file]
    argv += ["--out", args.output, "--rig", "pole"]
    if args.save:
        argv += ["--save-frames", "jpg"]
    if not args.bbox:
        argv.append("--no-overlay")
    return argv


def _ignored_flags(args: argparse.Namespace) -> List[str]:
    """Options given on the command line that the renderer no longer uses."""
    given = {
        "--animate": args.animate,
        "--ped-model": args.ped_model is not None,
        "--wheelchair": args.wheelchair,
        "--bike-model": args.bike_model is not None,
        "--video": args.video,
    }
    return [flag for flag, present in given.items() if present]


def translate_args(argv: Optional[Sequence[str]] = None) -> List[str]:
    """Translate the original options into arguments for ``bikeped.sim``.

    Returns the argument list of the equivalent ``render`` command, starting
    with the word ``render``. ``--list`` and the options that no longer have
    an effect do not appear in it.
    """
    return _render_args(build_parser().parse_args(argv))


def format_scenario_list() -> str:
    """Index, name, agent count and duration of every conformance scenario."""
    from decision_testbench import SCENARIOS, get_scenario_duration

    lines = [f"Available scenarios ({len(SCENARIOS)} total):", ""]
    for index, scenario in enumerate(SCENARIOS):
        agents = len(scenario["agents"])
        duration = get_scenario_duration(scenario)
        lines.append(f"  {index:>2}: {scenario['name']:<28} {agents} agents, {duration:.1f}s")
    return "\n".join(lines)


def main(argv: Optional[Sequence[str]] = None) -> int:
    """Run the translated ``render`` command and return its exit code."""
    args = build_parser().parse_args(argv)
    if args.list:
        print(format_scenario_list())
        return 0
    ignored = _ignored_flags(args)
    if ignored:
        print(f"note: ignoring {', '.join(ignored)} (kept for compatibility, no effect)",
              file=sys.stderr)

    from bikeped.sim.cli import main as sim_main

    return int(sim_main(_render_args(args)) or 0)


if __name__ == "__main__":
    sys.exit(main())
