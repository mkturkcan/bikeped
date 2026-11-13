"""COCO and YOLO export from a hand-built run directory."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional

import cv2
import numpy as np
import pytest
import yaml

from bikeped.sim.export import CLASS_NAMES, export_coco, export_yolo

WIDTH, HEIGHT, FRAMES = 320, 240, 5
RUN_ID = "ue4_fast-approach_clear_noon"


def obj(
    agent_id: int,
    kind: str,
    bbox: List[float],
    visible: Optional[float],
    role: str = "scripted",
) -> Dict[str, Any]:
    return {
        "id": agent_id,
        "kind": kind,
        "role": role,
        "bbox": bbox,
        "corners_image": [None] * 8,
        "distance_m": 12.5,
        "truncation": 0.0,
        "visible_fraction": visible,
    }


#: Six boxes over five frames: two bikes below 0.5 visibility, one 16 px^2 person.
OBJECTS: Dict[int, List[Dict[str, Any]]] = {
    0: [obj(1, "person", [10, 20, 60, 120], 1.0), obj(2, "bike", [100, 50, 220, 150], 0.4)],
    1: [obj(1, "person", [5, 5, 9, 9], None)],
    2: [],
    3: [obj(2, "bike", [200, 100, 300, 200], 0.05)],
    4: [
        obj(1, "person", [30, 40, 80, 160], 0.9),
        obj(3, "bike", [150, 60, 250, 140], 0.8, role="ambient"),
    ],
}


def frame_image(index: int) -> np.ndarray:
    image = np.empty((HEIGHT, WIDTH, 3), dtype=np.uint8)
    image[:] = (40 * index, 255 - 40 * index, 128)
    return image


def make_run(root: Path, *, saved_frames: bool, ext: str = "jpg") -> Path:
    """Write a five-frame run with either a clean video or saved frames."""
    run_dir = root / RUN_ID
    (run_dir / "gt").mkdir(parents=True)
    manifest: Dict[str, Any] = {
        "run_id": RUN_ID,
        "run": {"scenario_name": "Fast Approach", "engine": "ue4", "weather": "clear_noon"},
        "scenario": None,
        "layout": "single",
        "overlay": False,
        "views": ["pole"],
        "cameras": {"pole": {"name": "pole", "kind": "pinhole", "width": WIDTH, "height": HEIGHT}},
        "videos": {},
        "frame_ext": None,
        "gt": {"pole": "gt/pole.jsonl"},
        "stills": [],
        "timeline": {},
        "frame_count": FRAMES,
        "frames": [],
    }
    if saved_frames:
        frames_dir = run_dir / "frames" / "pole"
        frames_dir.mkdir(parents=True)
        for index in range(FRAMES):
            assert cv2.imwrite(str(frames_dir / f"{index:06d}.{ext}"), frame_image(index))
        manifest["frames_dir"] = {"pole": "frames/pole"}
        manifest["frame_ext"] = ext
    else:
        (run_dir / "video").mkdir()
        writer = cv2.VideoWriter(
            str(run_dir / "video" / "pole.mp4"),
            cv2.VideoWriter_fourcc(*"mp4v"),
            30.0,
            (WIDTH, HEIGHT),
        )
        assert writer.isOpened()
        for index in range(FRAMES):
            writer.write(frame_image(index))
        writer.release()
        manifest["videos"] = {"pole": "video/pole.mp4"}

    with open(run_dir / "gt" / "pole.jsonl", "w", encoding="utf-8") as fh:
        for index in range(FRAMES):
            record = {
                "index": index,
                "t": index / 30.0,
                "width": WIDTH,
                "height": HEIGHT,
                "objects": OBJECTS[index],
            }
            fh.write(json.dumps(record) + "\n")
    (run_dir / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    return run_dir


@pytest.fixture()
def video_run(tmp_path: Path) -> Path:
    return make_run(tmp_path / "runs", saved_frames=False)


@pytest.fixture()
def frames_run(tmp_path: Path) -> Path:
    return make_run(tmp_path / "runs", saved_frames=True)


def test_coco_from_video(video_run: Path, tmp_path: Path):
    out_path = tmp_path / "coco" / "annotations.json"
    result = export_coco(video_run, out_path)
    assert result == {"images": 5, "annotations": 6, "path": str(out_path)}

    data = json.loads(out_path.read_text())
    assert {"info", "images", "annotations", "categories"} <= set(data)
    assert [(c["id"], c["name"]) for c in data["categories"]] == [(1, "person"), (2, "bicycle")]
    assert [im["id"] for im in data["images"]] == [1, 2, 3, 4, 5]
    assert len({a["id"] for a in data["annotations"]}) == 6

    for image in data["images"]:
        assert (image["width"], image["height"]) == (WIDTH, HEIGHT)
        decoded = cv2.imread(str(out_path.parent / "images" / image["file_name"]))
        assert decoded.shape == (HEIGHT, WIDTH, 3)
        expected = frame_image(image["frame_index"])[0, 0].astype(float)
        assert np.abs(decoded.reshape(-1, 3).mean(axis=0) - expected).max() < 12.0

    first = data["annotations"][0]
    assert first["image_id"] == 1 and first["category_id"] == 1 and first["iscrowd"] == 0
    assert first["bbox"] == [10.0, 20.0, 50.0, 100.0]
    assert first["area"] == 5000.0
    assert first["agent_id"] == 1 and first["visible_fraction"] == 1.0
    bike = data["annotations"][1]
    assert bike["category_id"] == 2 and bike["bbox"] == [100.0, 50.0, 120.0, 100.0]
    assert sorted(a["image_id"] for a in data["annotations"]) == [1, 1, 2, 4, 5, 5]


def test_coco_copies_saved_frames(frames_run: Path, tmp_path: Path):
    out_path = tmp_path / "annotations.json"
    images_dir = tmp_path / "pictures"
    result = export_coco(frames_run, out_path, images_dir=images_dir)
    assert (result["images"], result["annotations"]) == (5, 6)
    data = json.loads(out_path.read_text())
    for image in data["images"]:
        source = frames_run / "frames" / "pole" / f"{image['frame_index']:06d}.jpg"
        assert (images_dir / image["file_name"]).read_bytes() == source.read_bytes()


def test_png_frames_are_kept_for_coco_and_transcoded_for_yolo(tmp_path: Path):
    run_dir = make_run(tmp_path / "runs", saved_frames=True, ext="png")
    manifest_path = run_dir / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["frame_ext"] = None
    manifest_path.write_text(json.dumps(manifest))

    out_path = tmp_path / "coco" / "annotations.json"
    export_coco(run_dir, out_path)
    names = [im["file_name"] for im in json.loads(out_path.read_text())["images"]]
    assert all(name.endswith(".png") for name in names)
    assert np.array_equal(cv2.imread(str(out_path.parent / "images" / names[1])), frame_image(1))

    export_yolo(run_dir, tmp_path / "yolo")
    jpg = cv2.imread(str(tmp_path / "yolo" / "images" / f"{RUN_ID}_pole_000001.jpg"))
    assert np.abs(jpg.astype(int) - frame_image(1).astype(int)).max() <= 3


def test_filters(video_run: Path, tmp_path: Path):
    def count(**kwargs) -> int:
        return export_coco(video_run, tmp_path / "f" / "a.json", **kwargs)["annotations"]

    assert count(min_visible=0.5) == 4
    assert count(min_area=100.0) == 5
    assert count(min_visible=0.5, min_area=100.0) == 3
    assert count(min_visible=0.95) == 2


def test_every_keeps_frames_aligned(video_run: Path, tmp_path: Path):
    out_path = tmp_path / "coco" / "annotations.json"
    result = export_coco(video_run, out_path, every=2)
    assert (result["images"], result["annotations"]) == (3, 4)
    data = json.loads(out_path.read_text())
    assert [im["frame_index"] for im in data["images"]] == [0, 2, 4]
    for image in data["images"]:
        decoded = cv2.imread(str(out_path.parent / "images" / image["file_name"]))
        expected = frame_image(image["frame_index"])[0, 0].astype(float)
        assert np.abs(decoded.reshape(-1, 3).mean(axis=0) - expected).max() < 12.0
    with pytest.raises(ValueError):
        export_coco(video_run, out_path, every=0)


@pytest.mark.parametrize("saved_frames", [False, True])
def test_yolo(tmp_path: Path, saved_frames: bool):
    run_dir = make_run(tmp_path / "runs", saved_frames=saved_frames)
    out_dir = tmp_path / "yolo"
    result = export_yolo(run_dir, out_dir)
    assert result == {"images": 5, "labels": 6, "path": str(out_dir)}

    stems = [f"{RUN_ID}_pole_{index:06d}" for index in range(FRAMES)]
    for stem in stems:
        assert cv2.imread(str(out_dir / "images" / f"{stem}.jpg")).shape == (HEIGHT, WIDTH, 3)
        assert (out_dir / "labels" / f"{stem}.txt").is_file()

    lines = (out_dir / "labels" / f"{stems[0]}.txt").read_text().splitlines()
    assert len(lines) == 2
    person = lines[0].split()
    assert person[0] == "0"
    assert [float(v) for v in person[1:]] == pytest.approx(
        [35 / WIDTH, 70 / HEIGHT, 50 / WIDTH, 100 / HEIGHT], abs=1e-6
    )
    assert lines[1].split()[0] == "1"
    assert (out_dir / "labels" / f"{stems[2]}.txt").read_text() == ""

    data = yaml.safe_load((out_dir / "data.yaml").read_text())
    assert data["names"] == {0: "person", 1: "bicycle"}
    assert tuple(data["names"].values()) == CLASS_NAMES
    assert data["train"] == "images" and data["val"] == "images"
    assert Path(data["path"]) == out_dir.resolve()


def test_yolo_filters(video_run: Path, tmp_path: Path):
    result = export_yolo(video_run, tmp_path / "yolo", min_visible=0.5, min_area=100.0)
    assert (result["images"], result["labels"]) == (5, 3)
    empty = tmp_path / "yolo" / "labels" / f"{RUN_ID}_pole_000003.txt"
    assert empty.read_text() == ""


def test_boxes_follow_a_resampled_video(video_run: Path, tmp_path: Path):
    gt_path = video_run / "gt" / "pole.jsonl"
    records = [json.loads(line) for line in gt_path.read_text().splitlines()]
    for record in records:
        record["width"], record["height"] = 2 * WIDTH, 2 * HEIGHT
        for item in record["objects"]:
            item["bbox"] = [2 * v for v in item["bbox"]]
    gt_path.write_text("".join(json.dumps(r) + "\n" for r in records))

    out_path = tmp_path / "coco" / "annotations.json"
    export_coco(video_run, out_path)
    data = json.loads(out_path.read_text())
    assert (data["images"][0]["width"], data["images"][0]["height"]) == (WIDTH, HEIGHT)
    assert data["annotations"][0]["bbox"] == [10.0, 20.0, 50.0, 100.0]


def test_missing_inputs_raise_clear_errors(video_run: Path, tmp_path: Path):
    with pytest.raises(FileNotFoundError, match="manifest.json"):
        export_coco(tmp_path / "nowhere", tmp_path / "a.json")
    with pytest.raises(ValueError, match="fisheye"):
        export_yolo(video_run, tmp_path / "yolo", view="fisheye")

    (video_run / "gt" / "pole.jsonl").unlink()
    with pytest.raises(FileNotFoundError, match="pole.jsonl"):
        export_coco(video_run, tmp_path / "a.json")


def test_missing_frames_and_video_raise(video_run: Path, tmp_path: Path):
    (video_run / "video" / "pole.mp4").unlink()
    with pytest.raises(FileNotFoundError, match="pole"):
        export_yolo(video_run, tmp_path / "yolo")
