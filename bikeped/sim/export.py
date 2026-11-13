"""Detection datasets from a rendered run.

A run directory written by ``bikeped.sim.recorder.Recorder`` holds the clean
video of every view, optionally the individual frames, and one ground-truth
file per view. The exporters here turn one view of a run into a COCO
annotation file or a YOLO dataset folder with the classes ``person`` and
``bicycle``.

Boxes are axis-aligned and given in pixels of the exported image, with the
origin at the top-left corner. Saved frames are copied as they are; when a
run has no saved frames the images are decoded from the clean video of the
view, whose frame ``k`` belongs to line ``k`` of the ground-truth file.
"""

from __future__ import annotations

import json
import logging
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Tuple, Union

import cv2
import numpy as np
import yaml

log = logging.getLogger(__name__)

PathLike = Union[str, Path]

#: Dataset classes in YOLO index order. The COCO category id is the index plus one.
CLASS_NAMES: Tuple[str, ...] = ("person", "bicycle")

#: Agent kind of the scenario format to its index in :data:`CLASS_NAMES`.
KIND_TO_CLASS: Dict[str, int] = {"person": 0, "bike": 1}

_SUPERCATEGORIES: Tuple[str, ...] = ("person", "vehicle")
_FRAME_EXTENSIONS: Tuple[str, ...] = ("jpg", "png")
_JPEG_QUALITY = 95


@dataclass(frozen=True)
class _Run:
    """One view of a run directory: manifest plus the ground-truth lines."""

    run_dir: Path
    run_id: str
    view: str
    manifest: Dict[str, Any]
    records: List[Dict[str, Any]]

    @property
    def stem(self) -> str:
        return f"{self.run_id}_{self.view}"


@dataclass(frozen=True)
class _Box:
    """Axis-aligned box in pixels of the exported image, top-left origin."""

    agent_id: int
    class_index: int
    x: float
    y: float
    w: float
    h: float
    truncation: Optional[float]
    visible_fraction: Optional[float]
    distance_m: Optional[float]


@dataclass(frozen=True)
class _Sample:
    """One frame to export.

    Exactly one of ``source`` (a saved frame file) and ``image`` (a frame
    decoded from the video) is set.
    """

    index: int
    t: float
    width: int
    height: int
    boxes: List[_Box]
    source: Optional[Path]
    image: Optional[np.ndarray]


def _load_run(run_dir: PathLike, view: str) -> _Run:
    """Read the manifest and the ground-truth file of ``view``."""
    root = Path(run_dir)
    manifest_path = root / "manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(f"no manifest.json in run directory {root}")
    with open(manifest_path, encoding="utf-8") as fh:
        manifest = json.load(fh)

    listed = manifest.get("gt") or {}
    gt_path = root / listed.get(view, f"gt/{view}.jsonl")
    known = set(manifest.get("views") or []) | set(listed)
    if view not in known and not gt_path.is_file():
        available = ", ".join(sorted(known)) or "none"
        raise ValueError(f"view {view!r} is not part of run {root}; available views: {available}")
    if not gt_path.is_file():
        raise FileNotFoundError(f"ground-truth file for view {view!r} is missing: {gt_path}")

    with open(gt_path, encoding="utf-8") as fh:
        records = [json.loads(line) for line in fh if line.strip()]
    run_id = str(manifest.get("run_id") or root.name)
    return _Run(root, run_id, view, manifest, records)


def _saved_frame(frames_dir: Path, index: int, ext: Optional[str]) -> Path:
    """Path of the saved frame ``index``, trying both formats when ``ext`` is unknown."""
    candidates = [ext] if ext else list(_FRAME_EXTENSIONS)
    for candidate in candidates:
        path = frames_dir / f"{index:06d}.{candidate}"
        if path.is_file():
            return path
    raise FileNotFoundError(f"saved frame {index:06d} is missing in {frames_dir}")


def _video_frames(
    path: Path, records: List[Dict[str, Any]], every: int
) -> Iterator[Tuple[Dict[str, Any], np.ndarray]]:
    """Decode ``path`` in step with ``records`` and yield every ``every``-th pair."""
    capture = cv2.VideoCapture(str(path))
    if not capture.isOpened():
        capture.release()
        raise OSError(f"cannot open video {path}")
    try:
        for k, record in enumerate(records):
            grabbed = capture.grab()
            if grabbed and k % every:
                continue
            ok, image = capture.retrieve() if grabbed else (False, None)
            if not ok:
                log.warning("video %s ends at frame %d of %d", path, k, len(records))
                return
            yield record, image
    finally:
        capture.release()


def _iter_frames(
    run: _Run, every: int
) -> Iterator[Tuple[Dict[str, Any], Optional[Path], Optional[np.ndarray]]]:
    """Yield ``(record, saved frame path, decoded frame)`` for every ``every``-th line."""
    frames_rel = (run.manifest.get("frames_dir") or {}).get(run.view)
    if frames_rel and (run.run_dir / frames_rel).is_dir():
        frames_dir = run.run_dir / frames_rel
        ext = run.manifest.get("frame_ext")
        for record in run.records[::every]:
            yield record, _saved_frame(frames_dir, int(record["index"]), ext), None
        return

    video_rel = (run.manifest.get("videos") or {}).get(run.view, f"video/{run.view}.mp4")
    video_path = run.run_dir / video_rel
    if not video_path.is_file():
        raise FileNotFoundError(
            f"run {run.run_dir} has neither saved frames nor a video for view {run.view!r} "
            f"(looked for {video_path})"
        )
    for record, image in _video_frames(video_path, run.records, every):
        yield record, None, image


def _optional_float(value: Any) -> Optional[float]:
    return None if value is None else float(value)


def _record_size(run: _Run, record: Dict[str, Any]) -> Tuple[int, int]:
    """Image size ``(width, height)`` that the boxes of ``record`` refer to."""
    camera = (run.manifest.get("cameras") or {}).get(run.view) or {}
    width = record.get("width") or camera.get("width")
    height = record.get("height") or camera.get("height")
    if not width or not height:
        raise ValueError(
            f"frame {record.get('index')} of view {run.view!r} has no image size in the "
            "ground truth or the manifest"
        )
    return int(width), int(height)


def _boxes(
    record: Dict[str, Any],
    gt_size: Tuple[int, int],
    image_size: Tuple[int, int],
    min_visible: float,
    min_area: float,
) -> List[_Box]:
    """Boxes of ``record`` that pass the filters, in pixels of the exported image."""
    width, height = image_size
    # Video encoding needs even dimensions, which can change a frame by one
    # pixel without moving its content. A larger difference means the video
    # was resampled, and the boxes are scaled with it.
    scale_x = width / gt_size[0] if abs(width - gt_size[0]) > 1 else 1.0
    scale_y = height / gt_size[1] if abs(height - gt_size[1]) > 1 else 1.0

    boxes: List[_Box] = []
    for obj in record.get("objects") or []:
        class_index = KIND_TO_CLASS.get(obj.get("kind"))
        bbox = obj.get("bbox")
        if class_index is None or not bbox:
            continue
        visible = _optional_float(obj.get("visible_fraction"))
        if visible is not None and visible < min_visible:
            continue
        x1 = min(max(float(bbox[0]) * scale_x, 0.0), float(width))
        y1 = min(max(float(bbox[1]) * scale_y, 0.0), float(height))
        x2 = min(max(float(bbox[2]) * scale_x, 0.0), float(width))
        y2 = min(max(float(bbox[3]) * scale_y, 0.0), float(height))
        w, h = x2 - x1, y2 - y1
        if w <= 0.0 or h <= 0.0 or w * h < min_area:
            continue
        boxes.append(
            _Box(
                agent_id=int(obj.get("id", -1)),
                class_index=class_index,
                x=x1,
                y=y1,
                w=w,
                h=h,
                truncation=_optional_float(obj.get("truncation")),
                visible_fraction=visible,
                distance_m=_optional_float(obj.get("distance_m")),
            )
        )
    return boxes


def _iter_samples(
    run: _Run, every: int, min_visible: float, min_area: float
) -> Iterator[_Sample]:
    """Yield the frames to export with their filtered boxes."""
    if every < 1:
        raise ValueError(f"every must be at least 1, got {every}")
    for record, source, image in _iter_frames(run, every):
        gt_size = _record_size(run, record)
        image_size = gt_size if image is None else (image.shape[1], image.shape[0])
        yield _Sample(
            index=int(record["index"]),
            t=float(record.get("t", 0.0)),
            width=image_size[0],
            height=image_size[1],
            boxes=_boxes(record, gt_size, image_size, min_visible, min_area),
            source=source,
            image=image,
        )


def _store_image(sample: _Sample, dest: Path) -> None:
    """Write the frame of ``sample`` to ``dest``, copying the file when formats match."""
    if sample.source is not None and sample.source.suffix.lower() == dest.suffix.lower():
        shutil.copyfile(sample.source, dest)
        return
    image = sample.image
    if image is None:
        image = cv2.imread(str(sample.source), cv2.IMREAD_COLOR)
        if image is None:
            raise OSError(f"cannot read frame {sample.source}")
    if not cv2.imwrite(str(dest), image, [cv2.IMWRITE_JPEG_QUALITY, _JPEG_QUALITY]):
        raise OSError(f"cannot write image {dest}")


def export_coco(
    run_dir: PathLike,
    out_path: PathLike,
    view: str = "pole",
    *,
    min_visible: float = 0.0,
    min_area: float = 0.0,
    every: int = 1,
    images_dir: Optional[PathLike] = None,
) -> Dict[str, Any]:
    """Write one view of a run as a COCO detection dataset.

    Parameters
    ----------
    run_dir:
        Run directory with ``manifest.json`` and ``gt/<view>.jsonl``.
    out_path:
        Annotation file to write. ``bbox`` is ``[x, y, width, height]`` in
        pixels; the categories are ``person`` (id 1) and ``bicycle`` (id 2).
    view:
        Name of the camera view to export.
    min_visible:
        Drop boxes whose ``visible_fraction`` is known and below this value.
        Boxes without a visibility estimate are kept.
    min_area:
        Drop boxes smaller than this area in square pixels.
    every:
        Export every ``every``-th frame.
    images_dir:
        Folder for the images; ``images`` next to ``out_path`` by default.
        ``file_name`` in the annotation file is relative to it.

    Returns
    -------
    dict
        ``{"images": n, "annotations": m, "path": out_path}``.
    """
    run = _load_run(run_dir, view)
    out_file = Path(out_path)
    images_root = Path(images_dir) if images_dir is not None else out_file.parent / "images"
    out_file.parent.mkdir(parents=True, exist_ok=True)
    images_root.mkdir(parents=True, exist_ok=True)

    images: List[Dict[str, Any]] = []
    annotations: List[Dict[str, Any]] = []
    for sample in _iter_samples(run, every, min_visible, min_area):
        suffix = sample.source.suffix.lower() if sample.source is not None else ".jpg"
        name = f"{run.stem}_{sample.index:06d}{suffix}"
        _store_image(sample, images_root / name)
        image_id = len(images) + 1
        images.append(
            {
                "id": image_id,
                "file_name": name,
                "width": sample.width,
                "height": sample.height,
                "frame_index": sample.index,
                "time_s": round(sample.t, 4),
            }
        )
        for box in sample.boxes:
            annotations.append(
                {
                    "id": len(annotations) + 1,
                    "image_id": image_id,
                    "category_id": box.class_index + 1,
                    "bbox": [round(v, 2) for v in (box.x, box.y, box.w, box.h)],
                    "area": round(box.w * box.h, 2),
                    "iscrowd": 0,
                    "agent_id": box.agent_id,
                    "truncation": box.truncation,
                    "visible_fraction": box.visible_fraction,
                    "distance_m": box.distance_m,
                }
            )

    info = run.manifest.get("run") or {}
    dataset = {
        "info": {
            "description": f"bikeped render {run.run_id}, view {view}",
            "version": "1.0",
            "run_id": run.run_id,
            "view": view,
            "scenario": info.get("scenario_name", ""),
            "engine": info.get("engine", ""),
            "map": info.get("map", ""),
            "weather": info.get("weather", ""),
        },
        "licenses": [],
        "images": images,
        "annotations": annotations,
        "categories": [
            {"id": i + 1, "name": name, "supercategory": _SUPERCATEGORIES[i]}
            for i, name in enumerate(CLASS_NAMES)
        ],
    }
    with open(out_file, "w", encoding="utf-8") as fh:
        json.dump(dataset, fh)
        fh.write("\n")
    return {"images": len(images), "annotations": len(annotations), "path": str(out_file)}


def export_yolo(
    run_dir: PathLike,
    out_dir: PathLike,
    view: str = "pole",
    *,
    min_visible: float = 0.0,
    min_area: float = 0.0,
    every: int = 1,
) -> Dict[str, Any]:
    """Write one view of a run as a YOLO detection dataset.

    The output folder gets ``images/<run_id>_<view>_<frame>.jpg``, a label
    file with the same stem under ``labels/`` and a ``data.yaml``. Each label
    line is ``class cx cy w h`` with the box centre and size divided by the
    image width and height; class 0 is ``person`` and class 1 is ``bicycle``.
    Frames without boxes get an empty label file, so they count as negatives.

    ``min_visible``, ``min_area`` and ``every`` filter as in
    :func:`export_coco`. Exports of several runs into the same folder add up,
    because the file names carry the run id.

    Returns
    -------
    dict
        ``{"images": n, "labels": m, "path": out_dir}``, where ``m`` counts
        the boxes written.
    """
    run = _load_run(run_dir, view)
    root = Path(out_dir)
    images_root = root / "images"
    labels_root = root / "labels"
    images_root.mkdir(parents=True, exist_ok=True)
    labels_root.mkdir(parents=True, exist_ok=True)

    n_images = 0
    n_labels = 0
    for sample in _iter_samples(run, every, min_visible, min_area):
        stem = f"{run.stem}_{sample.index:06d}"
        _store_image(sample, images_root / f"{stem}.jpg")
        lines = [
            "{:d} {:.6f} {:.6f} {:.6f} {:.6f}".format(
                box.class_index,
                (box.x + box.w / 2.0) / sample.width,
                (box.y + box.h / 2.0) / sample.height,
                box.w / sample.width,
                box.h / sample.height,
            )
            for box in sample.boxes
        ]
        with open(labels_root / f"{stem}.txt", "w", encoding="utf-8") as fh:
            fh.write("".join(line + "\n" for line in lines))
        n_images += 1
        n_labels += len(lines)

    data = {
        "path": str(root.resolve()),
        "train": "images",
        "val": "images",
        "names": dict(enumerate(CLASS_NAMES)),
    }
    with open(root / "data.yaml", "w", encoding="utf-8") as fh:
        yaml.safe_dump(data, fh, sort_keys=False)
    return {"images": n_images, "labels": n_labels, "path": str(root)}
