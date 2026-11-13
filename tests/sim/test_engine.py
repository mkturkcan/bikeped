"""Engine detection and capability probing against the fake servers."""

from __future__ import annotations

import logging

import fake_carla
import pytest

from bikeped.sim import engine


@pytest.fixture
def ue4():
    carla = fake_carla.make_fake_carla()
    engine.set_carla_module(carla)
    yield carla
    engine.set_carla_module(None)


@pytest.fixture
def ue5():
    carla = fake_carla.make_fake_carla(version="0.10.0", two_wheelers=False)
    engine.set_carla_module(carla)
    yield carla
    engine.set_carla_module(None)


@pytest.mark.parametrize("version, expected", [
    ("0.9.14", "ue4"), ("0.9.16", "ue4"), ("0.10.0", "ue5"), ("4fdea38", "ue4"),
])
def test_engine_for_version(version, expected):
    assert engine.engine_for_version(version) == expected


def test_detect_engine_ue4(ue4):
    assert engine.detect_engine(ue4.Client("127.0.0.1", 2000)) == "ue4"


def test_detect_engine_warns_on_mismatch(caplog):
    carla = fake_carla.make_fake_carla(version="0.10.0", client_version="0.9.16",
                                       two_wheelers=False)
    with caplog.at_level(logging.WARNING):
        assert engine.detect_engine(carla.Client("127.0.0.1", 2000)) == "ue5"
    assert "does not match" in caplog.text


def test_probe_ue4(ue4):
    caps = engine.probe(ue4.Client("127.0.0.1", 2000))
    assert caps.engine == "ue4"
    assert caps.has_rideable
    assert "vehicle.diamondback.century" in caps.bicycles
    assert caps.walkers >= 1
    assert caps.depth_camera and caps.instance_camera
    assert caps.ground_projection and caps.light_manager
    assert set(caps.to_dict()) >= {"engine", "maps", "two_wheelers"}
    assert "engine" in engine.format_capabilities(caps)


def test_probe_ue5_has_no_two_wheelers(ue5):
    caps = engine.probe(ue5.Client("127.0.0.1", 2000))
    assert caps.engine == "ue5"
    assert not caps.has_rideable
    assert caps.vehicles >= 1


def test_profiles():
    assert engine.PROFILES["ue4"].launcher == "CarlaUE4.sh"
    assert engine.PROFILES["ue5"].launcher == "CarlaUnreal.sh"
    assert engine.PROFILES["ue4"].supports_quality_flag
    assert not engine.PROFILES["ue5"].supports_quality_flag


def test_carla_module_override():
    sentinel = object()
    engine.set_carla_module(sentinel)
    try:
        assert engine.carla_module() is sentinel
    finally:
        engine.set_carla_module(None)


def test_server_command_per_engine(tmp_path):
    from bikeped.sim.server import build_command

    (tmp_path / "CarlaUE4.sh").write_text("")
    cmd = build_command("ue4", tmp_path, port=2100, quality="Low")
    assert cmd[0].endswith("CarlaUE4.sh")
    assert "-RenderOffScreen" in cmd and "-carla-rpc-port=2100" in cmd
    assert "-quality-level=Low" in cmd
    (tmp_path / "Linux").mkdir()
    (tmp_path / "Linux" / "CarlaUnreal.sh").write_text("")
    cmd = build_command("ue5", tmp_path, port=2100)
    assert cmd[0].endswith("CarlaUnreal.sh")
    assert not any(c.startswith("-quality-level") for c in cmd)
