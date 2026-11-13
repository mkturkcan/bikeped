"""Argument parsing of ``python -m bikeped.sim``."""

from __future__ import annotations

import pytest

from bikeped.sim.cli import _render_options, build_parser, main


def parse(*argv):
    return build_parser().parse_args(list(argv))


def test_probe():
    args = parse("probe", "--port", "2700", "--json")
    assert args.port == 2700 and args.json


def test_server():
    args = parse("server", "--engine", "ue5", "--root", "/x", "--port", "2100")
    assert (args.engine, args.root, args.port, args.quality) == ("ue5", "/x", 2100, "Epic")


def test_render_options():
    args = parse("render", "--scenario", "conformance:1", "--rig", "pole,fisheye",
                 "--weather", "golden_hour", "--drive", "velocity", "--ambient", "light",
                 "--quality", "cinematic", "--layout", "split", "--no-overlay",
                 "--save-frames", "jpg", "--seed", "3", "--fps", "25")
    opts = _render_options(args)
    assert opts.rigs == ("pole", "fisheye")
    assert (opts.weather, opts.drive, opts.ambient, opts.quality) == (
        "golden_hour", "velocity", "light", "cinematic")
    assert not opts.overlay and opts.save_frames == "jpg" and opts.seed == 3 and opts.fps == 25


def test_render_rejects_unknown_weather():
    with pytest.raises(SystemExit):
        parse("render", "--scenario", "1", "--weather", "snow")


def test_export():
    args = parse("export", "renders/run", "--format", "yolo", "--view", "pole")
    assert (args.run_dir, args.format, args.view) == ("renders/run", "yolo", "pole")


def test_presets_prints_tables(capsys):
    assert main(["presets"]) == 0
    out = capsys.readouterr().out
    assert "golden_hour" in out and "fisheye" in out and "cinematic" in out
