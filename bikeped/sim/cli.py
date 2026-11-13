"""Command line interface: ``python -m bikeped.sim {probe,server,presets,render,export}``."""

from __future__ import annotations

import argparse
import json
import logging
import sys
from typing import List, Optional

from .engine import PROFILES, format_capabilities, probe
from .environment import WEATHER_PRESETS
from .rigs import QUALITY_PRESETS, RIG_NAMES, describe_rigs


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m bikeped.sim",
        description="Render bikeped scenarios in CARLA (Unreal Engine 4: 0.9.14 to 0.9.16; "
                    "Unreal Engine 5: 0.10.0).")
    parser.add_argument("-v", "--verbose", action="store_true", help="log debug messages")
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("probe", help="print engine, version, maps and capabilities of a server")
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=2000)
    p.add_argument("--json", action="store_true", help="print JSON instead of a table")

    s = sub.add_parser("server", help="start a CARLA server and keep it running until Ctrl-C")
    s.add_argument("--engine", choices=sorted(PROFILES), required=True)
    s.add_argument("--root", help="install folder (default: CARLA_UE4_ROOT / CARLA_UE5_ROOT)")
    s.add_argument("--port", type=int, default=2000)
    s.add_argument("--quality", choices=("Epic", "Low"), default="Epic",
                   help="render quality level (Unreal Engine 4 only)")
    s.add_argument("--windowed", action="store_true", help="open a window instead of "
                   "rendering off screen")

    sub.add_parser("presets", help="list weather, rig and quality presets")

    r = sub.add_parser("render", help="render one scenario, a suite or all scenarios")
    r.add_argument("--scenario", help="slug, exact name, suite:index or conformance index")
    r.add_argument("--suite", help="suite of --scenario indices, or the suite to render "
                   "with --all")
    r.add_argument("--all", action="store_true", help="render every scenario of --suite")
    r.add_argument("--engine", choices=("auto", "ue4", "ue5"), default="auto",
                   help="expected engine; auto accepts whatever the server runs")
    r.add_argument("--host", default="127.0.0.1")
    r.add_argument("--port", type=int, default=2000)
    r.add_argument("--launch", action="store_true", help="start a managed server for this run "
                   "and stop it afterwards")
    r.add_argument("--root", help="install folder used with --launch")
    r.add_argument("--map", default=None, help="map to load if not loaded (default: keep the "
                   "current map, Town10HD_Opt on a fresh server)")
    r.add_argument("--crosswalk", type=int, default=4, help="crosswalk index (crosswalks.json on "
                   "Town10HD_Opt, live map polygons elsewhere)")
    r.add_argument("--crosswalks-file", default=None, help="alternative crosswalks.json")
    r.add_argument("--rig", default="pole", help="comma list of views: " + ",".join(RIG_NAMES))
    r.add_argument("--weather", choices=sorted(WEATHER_PRESETS), default=None,
                   help="weather preset (default: the scenario's hint, else clear_noon)")
    r.add_argument("--drive", choices=("velocity", "kinematic"), default=None,
                   help="how actors follow the script (default: kinematic)")
    r.add_argument("--ambient", choices=("none", "light", "busy"), default="none",
                   help="background traffic and pedestrians kept out of the scripted corridor")
    r.add_argument("--quality", choices=sorted(QUALITY_PRESETS), default="standard")
    r.add_argument("--layout", choices=("single", "split", "quad"), default="single")
    r.add_argument("--no-overlay", action="store_true", help="skip the HUD videos")
    r.add_argument("--save-frames", choices=("none", "jpg", "png"), default="none")
    r.add_argument("--out", default="renders", help="output folder")
    r.add_argument("--seed", type=int, default=0)
    r.add_argument("--fps", type=int, default=30)
    r.add_argument("--config", default="default", help="decision pipeline configuration")
    r.add_argument("--cyclist-proxy", choices=("walker",), default=None,
                   help="render cyclists as walkers on engines without a rideable two-wheeler")
    r.add_argument("--max-frames", type=int, default=None, help="stop after N frames")

    e = sub.add_parser("export", help="export ground truth of a run as COCO or YOLO")
    e.add_argument("run_dir")
    e.add_argument("--format", choices=("coco", "yolo"), required=True)
    e.add_argument("--view", default="pole")
    e.add_argument("--out", default=None, help="output path (default: inside RUN_DIR)")
    return parser


def _cmd_probe(args: argparse.Namespace) -> int:
    from .engine import carla_module

    carla = carla_module()
    client = carla.Client(args.host, args.port)
    client.set_timeout(20.0)
    caps = probe(client)
    print(json.dumps(caps.to_dict(), indent=2) if args.json else format_capabilities(caps))
    return 0


def _cmd_server(args: argparse.Namespace) -> int:
    import time

    from .server import managed_server

    with managed_server(args.engine, root=args.root, port=args.port, quality=args.quality,
                        windowed=args.windowed) as server:
        print(f"{args.engine} server {server.version or ''} ready on port {args.port} "
              f"(pid {server.pid}); Ctrl-C stops it")
        try:
            while server.running:
                time.sleep(1.0)
        except KeyboardInterrupt:
            pass
    return 0


def _cmd_presets(args: argparse.Namespace) -> int:
    print("weather")
    for name, v in WEATHER_PRESETS.items():
        print(f"  {name:<12} sun {v['sun_altitude_angle']:>5.0f} deg, clouds {v['cloudiness']:.0f}"
              f", rain {v['precipitation']:.0f}, fog {v['fog_density']:.0f}"
              f"{', street lights on' if v['night'] else ''}")
    print("rigs")
    for name, text in describe_rigs():
        print(f"  {name:<12} {text}")
    print("quality")
    for name, q in QUALITY_PRESETS.items():
        print(f"  {name:<12} {q['width']}x{q['height']}, fisheye faces {q['face']} px")
    return 0


def _render_options(args: argparse.Namespace):
    from .runner import RenderOptions

    return RenderOptions(
        host=args.host, port=args.port, engine=args.engine, map=args.map,
        crosswalk=args.crosswalk, crosswalks_file=args.crosswalks_file,
        rigs=tuple(v.strip() for v in args.rig.split(",") if v.strip()), weather=args.weather,
        drive=args.drive, ambient=args.ambient, quality=args.quality, layout=args.layout,
        overlay=not args.no_overlay, save_frames=args.save_frames, out=args.out, seed=args.seed,
        fps=args.fps, pipeline_config=args.config, cyclist_proxy=args.cyclist_proxy,
        max_frames=args.max_frames)


def _scenarios(args: argparse.Namespace) -> list:
    from .runner import resolve_scenario

    if args.all:
        try:
            from bikeped.scenarios import all_scenarios, get_suite
            return get_suite(args.suite) if args.suite else all_scenarios()
        except ImportError:
            import decision_testbench as tb
            return list(tb.SCENARIOS)
    if not args.scenario:
        raise SystemExit("render needs --scenario ID or --all")
    return [resolve_scenario(args.scenario, args.suite)]


def _cmd_render(args: argparse.Namespace) -> int:
    from contextlib import nullcontext

    from .runner import render_scenario

    opts = _render_options(args)
    for rig in opts.rigs:
        if rig not in RIG_NAMES:
            raise SystemExit(f"unknown rig '{rig}', choose from {', '.join(RIG_NAMES)}")
    scenarios = _scenarios(args)
    if args.launch:
        from .server import managed_server
        engine = "ue4" if args.engine == "auto" else args.engine
        ctx = managed_server(engine, root=args.root, port=args.port)
    else:
        ctx = nullcontext()
    failures = 0
    with ctx:
        for sc in scenarios:
            try:
                summary = render_scenario(sc, opts)
            except RuntimeError as exc:
                failures += 1
                print(f"{sc['name']}: {exc}", file=sys.stderr)
                continue
            print(f"{sc['name']}: {summary.get('frames')} frames in {summary.get('seconds')} s")
            print(f"  run dir   {summary.get('run_dir')}")
            for video in summary.get("videos", []):
                print(f"  video     {video}")
    return 1 if failures else 0


def _cmd_export(args: argparse.Namespace) -> int:
    from . import export

    if args.format == "coco":
        out = args.out or f"{args.run_dir}/coco_{args.view}.json"
        export.export_coco(args.run_dir, out, view=args.view)
    else:
        out = args.out or f"{args.run_dir}/yolo_{args.view}"
        export.export_yolo(args.run_dir, out, view=args.view)
    print(out)
    return 0


COMMANDS = {"probe": _cmd_probe, "server": _cmd_server, "presets": _cmd_presets,
            "render": _cmd_render, "export": _cmd_export}


def main(argv: Optional[List[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.INFO,
                        format="%(levelname)s %(name)s: %(message)s")
    return COMMANDS[args.command](args)
