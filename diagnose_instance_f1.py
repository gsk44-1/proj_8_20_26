"""Investigate F1 for identically expanded predicted and GT instances."""
from __future__ import annotations
import numpy as np
from scipy import ndimage as ndi
from geodesic_instances import components, geodesic_watershed, instance_score, WatershedConfig


def best_overlap_report(pred_instances, true_instances):
    """Per-GT best overlap without applying the 0.5 IoU threshold."""
    pred = np.asarray(pred_instances, dtype=np.int64)
    true = np.asarray(true_instances, dtype=np.int64)
    gt_ids = np.unique(true[true > 0]); pred_ids = np.unique(pred[pred > 0])
    results = []
    for gid in gt_ids:
        mask = (true == gid)
        candidates, counts = np.unique(pred[mask], return_counts=True)
        best_iou, best_pred, best_size, best_intersection = 0., 0, 0, 0
        for pid, overlap in zip(candidates, counts):
            if pid == 0: continue
            size = int((pred == pid).sum())
            iou = float(overlap / (mask.sum() + size - overlap))
            if iou > best_iou:
                best_iou, best_pred, best_size, best_intersection = iou, int(pid), size, int(overlap)
        results.append(dict(gt_id=int(gid), gt_pixels=int(mask.sum()),
                            best_pred_id=best_pred, pred_pixels=best_size,
                            intersection=best_intersection, best_iou=best_iou))
    return results


def diagnose_prediction(prob, target, radii=(0, 1, 2, 3, 5, 10, 25), iou_threshold=0.5):
    """Returns radius summary and per-ground-truth best-IoU rows; GT -100 is not a growth mask."""
    prob = np.asarray(prob)
    target = np.asarray(target)
    if prob.shape[0] != 3 or prob.shape[1:] != target.shape:
        raise ValueError('prob must be [3,H,W] and target [H,W]')
    seeds = (prob.argmax(0) == 1)
    gt_seeds = (target == 1)
    summary, detail = [], []
    for radius in radii:
        ws = WatershedConfig(radius=float(radius))
        grown = geodesic_watershed(seeds, config=ws)
        gt_labels = geodesic_watershed(gt_seeds, config=ws)
        score = instance_score(grown, gt_labels, iou_threshold=iou_threshold)
        matches = best_overlap_report(grown, gt_labels)
        max_ious = np.array([r['best_iou'] for r in matches])
        summary.append(dict(radius=float(radius), tp=score['tp'], fp=score['fp'],
                            fn=score['fn'], f1=score['instance_f1'],
                            pred_instances=score['pred_instances'],
                            gt_instances=score['gt_instances'],
                            best_iou_mean=float(max_ious.mean()) if len(max_ious) else np.nan,
                            best_iou_max=float(max_ious.max()) if len(max_ious) else np.nan,
                            n_best_iou_over_half=int((max_ious >= iou_threshold).sum()),
                            pred_foreground_pixels=int((grown > 0).sum()),
                            gt_foreground_pixels=int((gt_labels > 0).sum()),
                            ignored_pixels=int((target == -100).sum())))
        detail.extend(dict(radius=float(radius), **r) for r in matches)
    return summary, detail
