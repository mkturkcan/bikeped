"""
Model evaluation script for the bikeped collision-warning system.

Runs YOLO validation on the fisheye-augmented COCO dataset, extracts per-class
and per-size AP/AR metrics, merges related classes into the groups used by the
decision testbench (pedestrian / cyclist / vehicle), and caches the results.

Per-class-per-size AP/AR is computed from raw predictions and ground truth
labels using IoU matching, since ultralytics does not run pycocotools eval
for custom datasets.

Usage:
    # Run on the remote machine with GPU + dataset:
    python eval_models.py

    # Specify models explicitly:
    python eval_models.py --models best_640.pt best_1280.pt yolo11x.pt

    # Use existing cache without re-running val:
    python eval_models.py --cache-only

    # Recompute per-size metrics from saved predictions:
    python eval_models.py --recompute-sizes

The cache file (eval_cache.json) can be copied to the local machine and loaded
by decision_testbench.py to replace hardcoded AP/AR values.
"""

import json
import argparse
import numpy as np
from pathlib import Path
from collections import defaultdict


# COCO class IDs -> merged groups for the decision testbench
MERGE_GROUPS = {
    'pedestrian': [0],             # person
    'cyclist':    [1, 3],          # bicycle, motorcycle
    'vehicle':    [2, 5, 7],       # car, bus, truck
}

CLASSES_OF_INTEREST = {
    0: 'person', 1: 'bicycle', 2: 'car', 3: 'motorcycle',
    5: 'bus', 7: 'truck', 9: 'traffic light', 16: 'dog',
}

# COCO size thresholds (area in pixels^2)
SIZE_BINS = {
    'small':  (0, 32**2),        # < 1024 px^2
    'medium': (32**2, 96**2),    # 1024 - 9216 px^2
    'large':  (96**2, float('inf')),
}


# ================================================================
# Per-class-per-size AP/AR computation from raw predictions + labels
# ================================================================

def compute_iou_matrix(boxes_a, boxes_b):
    """Compute IoU between two sets of [x1, y1, x2, y2] boxes.

    Returns (len(boxes_a), len(boxes_b)) matrix.
    """
    x1 = np.maximum(boxes_a[:, 0:1], boxes_b[:, 0])
    y1 = np.maximum(boxes_a[:, 1:2], boxes_b[:, 1])
    x2 = np.minimum(boxes_a[:, 2:3], boxes_b[:, 2])
    y2 = np.minimum(boxes_a[:, 3:4], boxes_b[:, 3])

    inter = np.maximum(0, x2 - x1) * np.maximum(0, y2 - y1)
    area_a = (boxes_a[:, 2] - boxes_a[:, 0]) * (boxes_a[:, 3] - boxes_a[:, 1])
    area_b = (boxes_b[:, 2] - boxes_b[:, 0]) * (boxes_b[:, 3] - boxes_b[:, 1])
    union = area_a[:, None] + area_b[None, :] - inter
    return inter / np.maximum(union, 1e-6)


def compute_ap(recalls, precisions):
    """Compute AP using 101-point interpolation (COCO style)."""
    mrec = np.concatenate(([0.0], recalls, [1.0]))
    mpre = np.concatenate(([1.0], precisions, [0.0]))

    # Make precision monotonically decreasing
    for i in range(len(mpre) - 2, -1, -1):
        mpre[i] = max(mpre[i], mpre[i + 1])

    # 101-point interpolation
    recall_points = np.linspace(0, 1, 101)
    ap = 0.0
    for r in recall_points:
        idx = np.searchsorted(mrec, r)
        if idx < len(mpre):
            ap += mpre[idx]
    return ap / 101.0


def match_predictions(pred_boxes, pred_scores, gt_boxes, iou_threshold=0.5):
    """Match predictions to ground truth boxes at a given IoU threshold.

    Returns:
        tp: boolean array, True for each prediction that is a true positive.
        n_gt: number of ground truth boxes.
    """
    n_pred = len(pred_boxes)
    n_gt = len(gt_boxes)

    if n_pred == 0:
        return np.array([], dtype=bool), n_gt
    if n_gt == 0:
        return np.zeros(n_pred, dtype=bool), 0

    # Sort predictions by score descending
    order = np.argsort(-pred_scores)
    pred_boxes = pred_boxes[order]
    pred_scores = pred_scores[order]

    iou_mat = compute_iou_matrix(pred_boxes, gt_boxes)
    tp = np.zeros(n_pred, dtype=bool)
    gt_matched = np.zeros(n_gt, dtype=bool)

    for i in range(n_pred):
        best_iou = iou_threshold
        best_j = -1
        for j in range(n_gt):
            if gt_matched[j]:
                continue
            if iou_mat[i, j] > best_iou:
                best_iou = iou_mat[i, j]
                best_j = j
        if best_j >= 0:
            tp[i] = True
            gt_matched[best_j] = True

    return tp, n_gt


def compute_ar_curve(all_preds, all_gts, all_orig_shapes=None,
                     yolo_imgsz=1280, n_bins=20, iou_threshold=0.5):
    """Compute recall as a continuous function of bbox area for each class.

    Instead of 3 coarse COCO bins, this uses n_bins log-spaced area bins.
    Areas are normalized to the YOLO input resolution: for each image,
    GT box areas are scaled by (yolo_imgsz / max(h,w))^2 so that the
    curve reflects the pixel area the model actually sees during inference.

    Args:
        all_preds, all_gts: per-image predictions and ground truth.
        all_orig_shapes: list of (h, w) tuples for each image.
        yolo_imgsz: YOLO input size (default 1280).
        n_bins: number of area bins.
        iou_threshold: IoU threshold for matching.

    Returns:
        {class_id: {'bin_centers': [...], 'ar': [...], 'n_gt': [...]}}
    """
    bin_edges = np.logspace(np.log10(10), np.log10(500000), n_bins + 1)
    bin_centers = np.sqrt(bin_edges[:-1] * bin_edges[1:])  # geometric mean

    # Collect gt and matched-gt counts per class per bin
    # For each gt box, check if any prediction matched it at the given IoU
    class_bins = defaultdict(lambda: {
        'n_gt': np.zeros(n_bins, dtype=int),
        'n_detected': np.zeros(n_bins, dtype=int),
    })

    for img_idx, (preds, gts) in enumerate(zip(all_preds, all_gts)):
        gt_boxes = gts['boxes']
        gt_classes = gts['classes']
        pred_boxes = preds['boxes']
        pred_scores = preds['scores']
        pred_classes = preds['classes']

        # Compute area scale factor to YOLO input resolution
        area_scale = 1.0
        if all_orig_shapes is not None and img_idx < len(all_orig_shapes):
            h, w = all_orig_shapes[img_idx]
            max_dim = max(h, w)
            if max_dim > 0:
                area_scale = (yolo_imgsz / max_dim) ** 2

        for cls in set(int(c) for c in gt_classes):
            gt_mask = gt_classes == cls
            pred_mask = pred_classes == cls
            cls_gt = gt_boxes[gt_mask]
            cls_pred = pred_boxes[pred_mask]
            cls_scores = pred_scores[pred_mask]

            if len(cls_gt) == 0:
                continue

            # Match predictions to gt
            gt_matched = np.zeros(len(cls_gt), dtype=bool)
            if len(cls_pred) > 0:
                order = np.argsort(-cls_scores)
                cls_pred = cls_pred[order]
                iou_mat = compute_iou_matrix(cls_pred, cls_gt)
                for i in range(len(cls_pred)):
                    best_j = -1
                    best_iou = iou_threshold
                    for j in range(len(cls_gt)):
                        if gt_matched[j]:
                            continue
                        if iou_mat[i, j] > best_iou:
                            best_iou = iou_mat[i, j]
                            best_j = j
                    if best_j >= 0:
                        gt_matched[best_j] = True

            # Bin each gt box by area at YOLO input resolution
            for j in range(len(cls_gt)):
                box = cls_gt[j]
                area = (box[2] - box[0]) * (box[3] - box[1]) * area_scale
                bin_idx = np.searchsorted(bin_edges[1:], area)
                if bin_idx >= n_bins:
                    bin_idx = n_bins - 1
                class_bins[cls]['n_gt'][bin_idx] += 1
                if gt_matched[j]:
                    class_bins[cls]['n_detected'][bin_idx] += 1

    # Compute AR per bin
    results = {}
    for cls, data in class_bins.items():
        ar = np.zeros(n_bins)
        for i in range(n_bins):
            if data['n_gt'][i] > 0:
                ar[i] = data['n_detected'][i] / data['n_gt'][i]
        results[int(cls)] = {
            'bin_centers': bin_centers.tolist(),
            'ar': ar.tolist(),
            'n_gt': data['n_gt'].tolist(),
        }

    return results


def compute_class_size_metrics(all_preds, all_gts, iou_thresholds=None):
    """Compute per-class, per-size AP and AR.

    Args:
        all_preds: list of dicts per image, each with:
            'boxes': (N, 4) array [x1, y1, x2, y2]
            'scores': (N,) array
            'classes': (N,) int array
        all_gts: list of dicts per image, each with:
            'boxes': (M, 4) array [x1, y1, x2, y2]
            'classes': (M,) int array

    Returns:
        results: {class_id: {size_name: {'ap': float, 'ar': float, 'n_gt': int}}}
    """
    if iou_thresholds is None:
        iou_thresholds = np.arange(0.5, 1.0, 0.05)  # 0.50:0.05:0.95

    # Collect all predictions and gt per class per size
    # Structure: class_id -> size -> {'pred_boxes': [], 'pred_scores': [],
    #                                  'gt_boxes': [], 'image_indices': []}
    data = defaultdict(lambda: defaultdict(lambda: {
        'pred_boxes': [], 'pred_scores': [], 'gt_boxes_per_image': [],
    }))

    for preds, gts in zip(all_preds, all_gts):
        pred_boxes = preds['boxes']
        pred_scores = preds['scores']
        pred_classes = preds['classes']
        gt_boxes = gts['boxes']
        gt_classes = gts['classes']

        # Bin ground truth by class and size
        gt_by_class_size = defaultdict(lambda: defaultdict(list))
        for j in range(len(gt_boxes)):
            cls = int(gt_classes[j])
            box = gt_boxes[j]
            area = (box[2] - box[0]) * (box[3] - box[1])
            for size_name, (lo, hi) in SIZE_BINS.items():
                if lo <= area < hi:
                    gt_by_class_size[cls][size_name].append(box)
                    break

        # Bin predictions by class and size (use gt-matched size, but for
        # predictions we use their own box area to determine size)
        pred_by_class_size = defaultdict(lambda: defaultdict(
            lambda: {'boxes': [], 'scores': []}))
        for j in range(len(pred_boxes)):
            cls = int(pred_classes[j])
            box = pred_boxes[j]
            area = (box[2] - box[0]) * (box[3] - box[1])
            for size_name, (lo, hi) in SIZE_BINS.items():
                if lo <= area < hi:
                    pred_by_class_size[cls][size_name]['boxes'].append(box)
                    pred_by_class_size[cls][size_name]['scores'].append(
                        pred_scores[j])
                    break

        # Accumulate into per-class per-size data
        all_classes = set(gt_by_class_size.keys()) | set(pred_by_class_size.keys())
        for cls in all_classes:
            for size_name in SIZE_BINS:
                gt_list = gt_by_class_size[cls][size_name]
                pred_data = pred_by_class_size[cls][size_name]

                entry = data[cls][size_name]
                if pred_data['boxes']:
                    entry['pred_boxes'].append(np.array(pred_data['boxes']))
                    entry['pred_scores'].append(np.array(pred_data['scores']))
                else:
                    entry['pred_boxes'].append(np.empty((0, 4)))
                    entry['pred_scores'].append(np.empty(0))
                entry['gt_boxes_per_image'].append(
                    np.array(gt_list) if gt_list else np.empty((0, 4)))

    # Compute AP and AR for each class/size combination
    results = {}
    for cls in sorted(data.keys()):
        results[cls] = {}
        for size_name in SIZE_BINS:
            entry = data[cls][size_name]
            if not entry['gt_boxes_per_image']:
                results[cls][size_name] = {'ap': 0.0, 'ar': 0.0, 'n_gt': 0}
                continue

            n_gt_total = sum(len(g) for g in entry['gt_boxes_per_image'])
            if n_gt_total == 0 and all(
                    len(p) == 0 for p in entry['pred_boxes']):
                results[cls][size_name] = {'ap': 0.0, 'ar': 0.0, 'n_gt': 0}
                continue

            # Compute AP and AR averaged over IoU thresholds
            aps = []
            ars = []
            for iou_thr in iou_thresholds:
                all_tp = []
                all_scores = []
                total_gt = 0

                for img_preds, img_scores, img_gts in zip(
                        entry['pred_boxes'], entry['pred_scores'],
                        entry['gt_boxes_per_image']):

                    tp, n_gt = match_predictions(
                        img_preds, img_scores, img_gts, iou_thr)
                    all_tp.extend(tp)
                    all_scores.extend(
                        img_scores if len(img_scores) > 0 else [])
                    total_gt += n_gt

                if total_gt == 0:
                    aps.append(0.0)
                    ars.append(0.0)
                    continue

                all_tp = np.array(all_tp)
                all_scores = np.array(all_scores)

                if len(all_tp) == 0:
                    aps.append(0.0)
                    ars.append(0.0)
                    continue

                order = np.argsort(-all_scores)
                all_tp = all_tp[order]

                cum_tp = np.cumsum(all_tp)
                cum_fp = np.cumsum(~all_tp)
                recalls = cum_tp / total_gt
                precisions = cum_tp / (cum_tp + cum_fp)

                ap = compute_ap(recalls, precisions)
                ar = recalls[-1] if len(recalls) > 0 else 0.0
                aps.append(ap)
                ars.append(ar)

            results[cls][size_name] = {
                'ap': float(np.mean(aps)),
                'ar': float(np.mean(ars)),
                'n_gt': n_gt_total,
            }

    return results


def collect_predictions_and_gt(model_path, data_yaml, imgsz=None):
    """Run model prediction on validation set and collect raw boxes + labels.

    Returns (all_preds, all_gts) lists for compute_class_size_metrics.
    """
    from ultralytics import YOLO

    model = YOLO(str(model_path))
    val_kwargs = {'data': data_yaml, 'verbose': False, 'save_json': False,
                  'plots': False, 'batch': 1}
    if imgsz is not None:
        val_kwargs['imgsz'] = imgsz

    print(f'  Running validation for per-size metrics...')
    results = model.val(**val_kwargs)

    all_preds = []
    all_gts = []
    all_orig_shapes = []

    # We need to re-run prediction on the val set to get raw boxes.
    import yaml as _yaml
    with open(data_yaml, 'r') as f:
        data_cfg = _yaml.safe_load(f)

    # Resolve val path relative to dataset root, matching ultralytics behavior
    dataset_root = Path(data_cfg.get('path', ''))
    val_rel = data_cfg.get('val', data_cfg.get('test', ''))

    # Try: dataset_root / val_rel, then relative to yaml dir, then as-is
    yaml_dir = Path(data_yaml).resolve().parent
    candidates = [
        dataset_root / val_rel,
        yaml_dir / dataset_root / val_rel,
        Path(val_rel),
    ]
    val_path = None
    for c in candidates:
        if c.exists():
            val_path = str(c.resolve())
            break
    if val_path is None:
        raise FileNotFoundError(
            f'Cannot find val set. Tried: {[str(c) for c in candidates]}')

    print(f'  Val source: {val_path}')
    print(f'  Collecting raw predictions on val set...')
    pred_results = model.predict(
        source=val_path, imgsz=imgsz or 640, conf=0.001,
        verbose=False, stream=True, save=False)

    for r in pred_results:
        # Predictions
        boxes = r.boxes
        if boxes is not None and len(boxes) > 0:
            pred_xyxy = boxes.xyxy.cpu().numpy()
            pred_conf = boxes.conf.cpu().numpy()
            pred_cls = boxes.cls.cpu().numpy().astype(int)
        else:
            pred_xyxy = np.empty((0, 4))
            pred_conf = np.empty(0)
            pred_cls = np.empty(0, dtype=int)

        all_preds.append({
            'boxes': pred_xyxy,
            'scores': pred_conf,
            'classes': pred_cls,
        })

        # Ground truth from the label file (ultralytics YOLO format: cls cx cy w h)
        # Standard convention: .../images/foo.jpg -> .../labels/foo.txt
        img_path = Path(r.path)
        img_str = str(img_path)
        # Replace the last occurrence of /images/ or \images\ with /labels/
        for sep_img, sep_lbl in [('/images/', '/labels/'), ('\\images\\', '\\labels\\')]:
            idx = img_str.rfind(sep_img)
            if idx >= 0:
                img_str = img_str[:idx] + sep_lbl + img_str[idx + len(sep_img):]
                break
        label_path = Path(img_str).with_suffix('.txt')

        gt_boxes = []
        gt_classes = []
        if label_path.exists():
            h, w = r.orig_shape
            with open(label_path, 'r') as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) >= 5:
                        cls = int(parts[0])
                        cx, cy, bw, bh = float(parts[1]), float(parts[2]), \
                                         float(parts[3]), float(parts[4])
                        # Convert normalized xywh to pixel xyxy
                        x1 = (cx - bw / 2) * w
                        y1 = (cy - bh / 2) * h
                        x2 = (cx + bw / 2) * w
                        y2 = (cy + bh / 2) * h
                        gt_boxes.append([x1, y1, x2, y2])
                        gt_classes.append(cls)

        all_gts.append({
            'boxes': np.array(gt_boxes) if gt_boxes else np.empty((0, 4)),
            'classes': np.array(gt_classes, dtype=int) if gt_classes else np.empty(0, dtype=int),
        })
        all_orig_shapes.append(r.orig_shape)

    print(f'  Collected {len(all_preds)} images')
    return all_preds, all_gts, all_orig_shapes, results


# ================================================================
# Top-level evaluation
# ================================================================

def extract_metrics(results):
    """Extract per-class AP and AR from ultralytics validation results."""
    metrics = results.results_dict
    box = results.box

    overall = {
        'mAP50': float(metrics.get('metrics/mAP50(B)', 0)),
        'mAP50-95': float(metrics.get('metrics/mAP50-95(B)', 0)),
        'precision': float(metrics.get('metrics/precision(B)', 0)),
        'recall': float(metrics.get('metrics/recall(B)', 0)),
    }

    per_class = {}
    if hasattr(box, 'ap_class_index') and box.ap_class_index is not None:
        class_indices = box.ap_class_index.tolist() if hasattr(
            box.ap_class_index, 'tolist') else list(box.ap_class_index)
        ap50_all = box.ap50 if hasattr(box, 'ap50') else None
        ap_all = box.ap if hasattr(box, 'ap') else None
        p_all = box.p if hasattr(box, 'p') else None
        r_all = box.r if hasattr(box, 'r') else None

        for i, cls_id in enumerate(class_indices):
            cls_id = int(cls_id)
            entry = {'class_id': cls_id}
            if ap50_all is not None and i < len(ap50_all):
                entry['ap50'] = float(ap50_all[i])
            if ap_all is not None and i < len(ap_all):
                if ap_all.ndim == 2:
                    entry['ap5095'] = float(np.mean(ap_all[i]))
                    entry['ap_per_iou'] = [float(v) for v in ap_all[i]]
                else:
                    entry['ap5095'] = float(ap_all[i])
            if p_all is not None and i < len(p_all):
                entry['precision'] = float(p_all[i])
            if r_all is not None and i < len(r_all):
                entry['recall'] = float(r_all[i])
            per_class[cls_id] = entry

    # Merged group metrics
    merged = {}
    for group_name, class_ids in MERGE_GROUPS.items():
        group_entries = [per_class[c] for c in class_ids if c in per_class]
        if not group_entries:
            merged[group_name] = {
                'ap50': 0.0, 'ap5095': 0.0, 'precision': 0.0, 'recall': 0.0,
                'n_classes': 0, 'source_classes': class_ids,
            }
            continue
        merged[group_name] = {
            'ap50': float(np.mean([e.get('ap50', 0) for e in group_entries])),
            'ap5095': float(np.mean([e.get('ap5095', 0) for e in group_entries])),
            'precision': float(np.mean([e.get('precision', 0) for e in group_entries])),
            'recall': float(np.mean([e.get('recall', 0) for e in group_entries])),
            'n_classes': len(group_entries),
            'source_classes': class_ids,
        }

    return {
        'overall': overall,
        'per_class': {int(k): v for k, v in per_class.items()},
        'per_size': {},
        'merged': merged,
    }


def evaluate_model(model_path, data_yaml='cocoaug.yaml', imgsz=None,
                   compute_sizes=True):
    """Full evaluation pipeline for one model."""
    print(f'\n{"=" * 70}')
    print(f'  Evaluating: {model_path}')
    print(f'  Dataset: {data_yaml}')
    if imgsz:
        print(f'  Image size: {imgsz}')
    print(f'{"=" * 70}')

    if compute_sizes:
        all_preds, all_gts, all_orig_shapes, results = collect_predictions_and_gt(
            model_path, data_yaml, imgsz)
        metrics = extract_metrics(results)

        # Compute per-class-per-size metrics (coarse COCO bins)
        print(f'  Computing per-class per-size AP/AR...')
        per_class_size = compute_class_size_metrics(all_preds, all_gts)

        # Compute fine-grained AR curve (20 log-spaced area bins at YOLO input resolution)
        print(f'  Computing AR-vs-area curves (at imgsz={imgsz or 1280})...')
        ar_curves = compute_ar_curve(all_preds, all_gts,
                                     all_orig_shapes=all_orig_shapes,
                                     yolo_imgsz=imgsz or 1280)
        metrics['ar_curves'] = {int(k): v for k, v in ar_curves.items()}

        # Merged AR curves
        for group_name, class_ids in MERGE_GROUPS.items():
            group_n_gt = np.zeros(20, dtype=int)
            group_n_det = np.zeros(20, dtype=int)
            for cls in class_ids:
                if cls in ar_curves:
                    c = ar_curves[cls]
                    n_gt = np.array(c['n_gt'])
                    ar = np.array(c['ar'])
                    group_n_gt += n_gt
                    group_n_det += (ar * n_gt).astype(int)
            group_ar = np.where(group_n_gt > 0,
                                group_n_det / group_n_gt, 0.0)
            if class_ids[0] in ar_curves:
                bin_centers = ar_curves[class_ids[0]]['bin_centers']
            else:
                bin_centers = np.logspace(
                    np.log10(100), np.log10(500000), 21)
                bin_centers = np.sqrt(bin_centers[:-1] * bin_centers[1:]).tolist()
            metrics['merged'][group_name]['ar_curve'] = {
                'bin_centers': bin_centers,
                'ar': group_ar.tolist(),
                'n_gt': group_n_gt.tolist(),
            }

        # Store per-class-per-size
        metrics['per_class_per_size'] = {
            int(cls): {s: v for s, v in sizes.items()}
            for cls, sizes in per_class_size.items()
        }

        # Aggregate per-size across all classes
        per_size_agg = {}
        for size_name in SIZE_BINS:
            aps = []
            ars = []
            for cls in per_class_size:
                if size_name in per_class_size[cls]:
                    entry = per_class_size[cls][size_name]
                    if entry['n_gt'] > 0:
                        aps.append(entry['ap'])
                        ars.append(entry['ar'])
            per_size_agg[size_name] = {
                'ap': float(np.mean(aps)) if aps else 0.0,
                'ar': float(np.mean(ars)) if ars else 0.0,
            }
        metrics['per_size'] = per_size_agg

        # Add per-size for merged groups
        for group_name, class_ids in MERGE_GROUPS.items():
            group_per_size = {}
            for size_name in SIZE_BINS:
                aps = []
                ars = []
                n_gt = 0
                for cls in class_ids:
                    if cls in per_class_size and size_name in per_class_size[cls]:
                        entry = per_class_size[cls][size_name]
                        if entry['n_gt'] > 0:
                            aps.append(entry['ap'])
                            ars.append(entry['ar'])
                            n_gt += entry['n_gt']
                group_per_size[size_name] = {
                    'ap': float(np.mean(aps)) if aps else 0.0,
                    'ar': float(np.mean(ars)) if ars else 0.0,
                    'n_gt': n_gt,
                }
            metrics['merged'][group_name]['per_size'] = group_per_size
    else:
        from ultralytics import YOLO
        model = YOLO(str(model_path))
        val_kwargs = {'data': data_yaml, 'verbose': False}
        if imgsz is not None:
            val_kwargs['imgsz'] = imgsz
        results = model.val(**val_kwargs)
        metrics = extract_metrics(results)

    # Print summary
    print(f'\n  Overall: mAP50={metrics["overall"]["mAP50"]:.3f}  '
          f'mAP50-95={metrics["overall"]["mAP50-95"]:.3f}  '
          f'P={metrics["overall"]["precision"]:.3f}  '
          f'R={metrics["overall"]["recall"]:.3f}')

    if metrics['per_size']:
        print(f'\n  Per-size AP/AR (all classes):')
        for size, vals in metrics['per_size'].items():
            print(f'    {size:>8}: AP={vals["ap"]:.3f}  AR={vals["ar"]:.3f}')

    print(f'\n  Merged groups for decision testbench:')
    for group, vals in metrics['merged'].items():
        line = (f'    {group:>12}: AP50={vals["ap50"]:.3f}  '
                f'AP={vals["ap5095"]:.3f}  '
                f'P={vals["precision"]:.3f}  R={vals["recall"]:.3f}')
        if 'per_size' in vals:
            ps = vals['per_size']
            line += '  |'
            for s in ('small', 'medium', 'large'):
                if s in ps and ps[s]['n_gt'] > 0:
                    line += f'  {s[0].upper()}:AP={ps[s]["ap"]:.3f}/AR={ps[s]["ar"]:.3f}'
        print(line)

    print(f'\n  Per-class (classes of interest):')
    for cls_id, name in sorted(CLASSES_OF_INTEREST.items()):
        if cls_id in metrics['per_class']:
            e = metrics['per_class'][cls_id]
            line = (f'    {cls_id:>3} {name:>15}: AP50={e.get("ap50", 0):.3f}  '
                    f'AP={e.get("ap5095", 0):.3f}  '
                    f'P={e.get("precision", 0):.3f}  R={e.get("recall", 0):.3f}')
            # Add per-size if available
            if 'per_class_per_size' in metrics and cls_id in metrics['per_class_per_size']:
                pcs = metrics['per_class_per_size'][cls_id]
                line += '  |'
                for s in ('small', 'medium', 'large'):
                    if s in pcs and pcs[s]['n_gt'] > 0:
                        line += f'  {s[0].upper()}:AP={pcs[s]["ap"]:.3f}/AR={pcs[s]["ar"]:.3f}({pcs[s]["n_gt"]})'
            print(line)
        else:
            print(f'    {cls_id:>3} {name:>15}: (not in eval results)')

    return metrics


def save_cache(cache, path='eval_cache.json'):
    with open(path, 'w') as f:
        json.dump(cache, f, indent=2)
    print(f'\nCache saved to {path}')


def load_cache(path='eval_cache.json'):
    with open(path, 'r') as f:
        return json.load(f)


def print_cache_summary(cache):
    print(f'\n{"=" * 70}')
    print(f'  EVALUATION CACHE SUMMARY')
    print(f'{"=" * 70}')

    for model_name, data in cache.items():
        print(f'\n  --- {model_name} ---')
        ov = data.get('overall', {})
        print(f'  Overall: mAP50={ov.get("mAP50", 0):.3f}  '
              f'mAP50-95={ov.get("mAP50-95", 0):.3f}  '
              f'R={ov.get("recall", 0):.3f}')

        print(f'  Merged groups:')
        for group, vals in data.get('merged', {}).items():
            line = (f'    {group:>12}: AP50={vals["ap50"]:.3f}  '
                    f'AP={vals["ap5095"]:.3f}  '
                    f'P={vals["precision"]:.3f}  R={vals["recall"]:.3f}')
            ps = vals.get('per_size', {})
            if ps:
                line += '  |'
                for s in ('small', 'medium', 'large'):
                    if s in ps and ps[s].get('n_gt', 0) > 0:
                        line += f'  {s[0].upper()}:AP={ps[s]["ap"]:.3f}/AR={ps[s]["ar"]:.3f}'
            print(line)

        ps = data.get('per_size', {})
        if ps and any(ps.values()):
            print(f'  Per-size (all classes):')
            for size in ('small', 'medium', 'large'):
                if size in ps:
                    print(f'    {size:>8}: AP={ps[size]["ap"]:.3f}  '
                          f'AR={ps[size]["ar"]:.3f}')

    # Print testbench-ready snippet
    print(f'\n{"=" * 70}')
    print(f'  TESTBENCH-READY VALUES')
    print(f'{"=" * 70}')
    for model_name, data in cache.items():
        mg = data.get('merged', {})
        ped = mg.get('pedestrian', {})
        cyc = mg.get('cyclist', {})
        veh = mg.get('vehicle', {})
        print(f'\n  # {model_name}')
        print(f'  UncertaintyParams(')
        print(f'      pedestrian=DetectionParams(ap={ped.get("ap5095", 0):.3f}, ar={ped.get("recall", 0):.3f}),')
        print(f'      cyclist=DetectionParams(ap={cyc.get("ap5095", 0):.3f}, ar={cyc.get("recall", 0):.3f}),')
        print(f'      vehicle=DetectionParams(ap={veh.get("ap5095", 0):.3f}, ar={veh.get("recall", 0):.3f}),')
        # Per-size for merged groups
        for group_name in ('pedestrian', 'cyclist', 'vehicle'):
            gps = mg.get(group_name, {}).get('per_size', {})
            if gps:
                parts = []
                for s in ('small', 'medium', 'large'):
                    if s in gps and gps[s].get('n_gt', 0) > 0:
                        parts.append(f'{s}: AP={gps[s]["ap"]:.3f} AR={gps[s]["ar"]:.3f}')
                if parts:
                    print(f'      # {group_name} per-size: {", ".join(parts)}')
        print(f'  )')


def main():
    parser = argparse.ArgumentParser(
        description='Evaluate YOLO models and cache per-class AP/AR')
    parser.add_argument('--models', nargs='+', default=None,
                        help='Model paths to evaluate')
    parser.add_argument('--data', default='cocoaug.yaml',
                        help='Dataset config YAML (default: cocoaug.yaml)')
    parser.add_argument('--cache', default='eval_cache.json',
                        help='Cache file path (default: eval_cache.json)')
    parser.add_argument('--cache-only', action='store_true',
                        help='Only print cached results')
    parser.add_argument('--no-sizes', action='store_true',
                        help='Skip per-size computation (faster)')
    args = parser.parse_args()

    if args.cache_only:
        try:
            cache = load_cache(args.cache)
            print_cache_summary(cache)
        except FileNotFoundError:
            print(f'Error: {args.cache} not found.')
        return

    if args.models is None:
        models = [
            ('fisheye_1280', 'runs/detect/train142/weights/best.pt', 1280),
            ('fisheye_640', 'runs/detect/train139/weights/best.pt', 640),
            ('coco_original', 'yolo11x.pt', 640),
        ]
    else:
        models = []
        for m in args.models:
            name = Path(m).stem
            imgsz = None
            if '640' in m:
                imgsz = 640
            elif '1280' in m:
                imgsz = 1280
            models.append((name, m, imgsz))

    cache = {}
    for model_name, model_path, imgsz in models:
        if not Path(model_path).exists():
            print(f'\n  Skipping {model_name}: {model_path} not found')
            continue
        metrics = evaluate_model(model_path, args.data, imgsz,
                                 compute_sizes=not args.no_sizes)
        cache[model_name] = metrics

    if cache:
        save_cache(cache, args.cache)
        print_cache_summary(cache)


if __name__ == '__main__':
    main()
