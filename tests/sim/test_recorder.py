"""End-to-end tests of the recorder on synthetic frames; no simulator needed."""

from __future__ import annotations

import json

import cv2
import numpy as np
import pytest

import decision_testbench as tb
from bikeped.sim import recorder as recorder_module
from bikeped.sim.export import export_coco, export_yolo
from bikeped.sim.fisheye import FisheyeRemap, deployed_fisheye_model, synthetic_faces
from bikeped.sim.groundtruth import scenario_to_world, synthetic_frame
from bikeped.sim.overlay import build_timeline
from bikeped.sim.projection import carla_rotation_to_basis
from bikeped.sim.recorder import Recorder, VideoSink
from bikeped.sim.types import CameraModel, RunInfo

FPS = 10
SCENARIO = next(s for s in tb.SCENARIOS if s["name"] == "Bike Approaches Ped")


def _pole() -> CameraModel:
    f, r, u = carla_rotation_to_basis(-16.0, -90.0, 0.0)
    return CameraModel(name="pole", kind="pinhole", width=320, height=180,
                       position=scenario_to_world(5.0, -13.0, 3.6576), forward=f, right=r, up=u,
                       fov_deg=90.0)


@pytest.fixture(scope="module")
def frames():
    pole = _pole()
    fisheye = deployed_fisheye_model(scenario_to_world(5.0, -13.0, 3.6576), -90.0, scale=0.1)
    remap = FisheyeRemap(fisheye, face_size=96)
    fisheye_image = remap.remap(synthetic_faces(remap))
    timeline = build_timeline(SCENARIO, FPS)
    out = []
    for i in range(0, timeline["n"], 2):
        frame = synthetic_frame(SCENARIO, i, FPS, cameras={"pole": pole, "fisheye": fisheye},
                                states=timeline["states"])
        frame.index = len(out)
        frame.images["pole"] = np.full((pole.height, pole.width, 3), 90, dtype=np.uint8)
        frame.images["fisheye"] = fisheye_image
        out.append(frame)
    return out


def _decode(path) -> int:
    cap = cv2.VideoCapture(str(path))
    count = 0
    while True:
        ok, _ = cap.read()
        if not ok:
            break
        count += 1
    cap.release()
    return count


def _run(tmp_path, frames, **kwargs):
    run = RunInfo(scenario_name=SCENARIO["name"], scenario_id="bike-approaches-ped",
                  fps=FPS, engine="ue4", views=["pole", "fisheye"])
    with Recorder(tmp_path, run, views=["pole", "fisheye"], scenario=SCENARIO, **kwargs) as rec:
        for frame in frames:
            rec.write(frame)
    return rec.close()


def test_end_to_end_split(tmp_path, frames):
    result = _run(tmp_path, frames, layout="split", save_frames="jpg")
    run_dir = tmp_path / "ue4_bike-approaches-ped_clear_noon"
    assert result["run_dir"] == str(run_dir)
    manifest = json.loads((run_dir / "manifest.json").read_text())
    assert manifest["frame_count"] == len(frames) == result["frames"]
    assert set(manifest["cameras"]) == {"pole", "fisheye"}
    assert manifest["complete"] is True
    assert set(manifest["videos"]) == {"pole", "pole_overlay", "fisheye", "fisheye_overlay",
                                       "composite"}
    for rel in manifest["videos"].values():
        assert _decode(run_dir / rel) == len(frames)
    timeline = manifest["timeline"]
    assert sum(timeline["state_counts"].values()) == len(frames)
    assert timeline["danger_intervals"] and timeline["closest_approach"]["distance_m"] < 5.0
    if timeline["first_alert_t"] is not None and timeline["first_danger_t"] is not None:
        assert timeline["lead_time_s"] == pytest.approx(
            timeline["first_danger_t"] - timeline["first_alert_t"], abs=1e-3)
    stills = {p.name for p in (run_dir / "stills").iterdir()}
    assert {"first_frame.jpg", "first_frame.png", "closest_approach.png"} <= stills
    for still in (run_dir / "stills").iterdir():
        assert cv2.imread(str(still)) is not None
    lines = (run_dir / "gt" / "pole.jsonl").read_text().splitlines()
    assert len(lines) == len(frames)
    assert any(json.loads(line)["objects"] for line in lines)
    assert len(list((run_dir / "frames" / "pole").glob("*.jpg"))) == len(frames)
    composite = cv2.VideoCapture(str(run_dir / manifest["videos"]["composite"]))
    ok, image = composite.read()
    composite.release()
    assert ok and image.shape[0] == 180 and image.shape[1] > 320

    coco = export_coco(run_dir, tmp_path / "coco" / "pole.json", view="pole")
    assert coco["images"] == len(frames) and coco["annotations"] > 0
    yolo = export_yolo(run_dir, tmp_path / "yolo", view="pole")
    assert yolo["images"] == len(frames)


def test_quad_without_overlay(tmp_path, frames):
    result = _run(tmp_path, frames[:5], layout="quad", overlay=False)
    names = {p.split("/")[-1] for p in result["videos"]}
    assert names == {"pole.mp4", "fisheye.mp4", "composite.mp4"}


def test_close_is_idempotent_and_survives_errors(tmp_path, frames):
    run = RunInfo(scenario_name="x", scenario_id="x", fps=FPS)
    with pytest.raises(RuntimeError):
        with Recorder(tmp_path, run, views=["pole"]) as rec:
            rec.write(frames[0])
            raise RuntimeError("simulator lost")
    first = rec.close()
    assert rec.close() is first
    manifest = json.loads((tmp_path / "ue4_x_clear_noon" / "manifest.json").read_text())
    assert manifest["complete"] is False and "simulator lost" in manifest["error"]
    with pytest.raises(RuntimeError):
        rec.write(frames[1])


def test_odd_sizes_and_opencv_fallback(tmp_path, monkeypatch):
    monkeypatch.setattr(recorder_module, "find_ffmpeg", lambda: None)
    sink = VideoSink(tmp_path / "odd.mp4", 10, ffmpeg="")
    for _ in range(3):
        sink.write(np.zeros((51, 75, 3), dtype=np.uint8))
    sink.close()
    sink.close()
    assert sink.size == (76, 52)
    assert _decode(tmp_path / "odd.mp4") == 3


def test_rejects_bad_arguments(tmp_path):
    run = RunInfo(scenario_name="x", scenario_id="x")
    with pytest.raises(ValueError):
        Recorder(tmp_path, run, views=["pole"], layout="mosaic")
    with pytest.raises(ValueError):
        Recorder(tmp_path, run, views=["pole"], save_frames="tiff")
