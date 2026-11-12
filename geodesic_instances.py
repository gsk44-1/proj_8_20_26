"""Radius-limited multi-source geodesic watershed for nuclear seed components.

Ground truth is scored as its original binary nuclear connected components.
Predictions produce seeds through the 3-class argmax nuclear channel, then
region-growing on a FIXED, normalized nonnuclear-fluorescence image.
"""
from __future__ import annotations
import heapq
from dataclasses import dataclass
import numpy as np
from scipy import ndimage as ndi
from scipy.optimize import linear_sum_assignment

STRUCTURE_8 = np.ones((3, 3), dtype=np.uint8)
NEIGHBORS = tuple((dy, dx, np.hypot(dy, dx)) for dy in (-1, 0, 1)
                  for dx in (-1, 0, 1) if (dy or dx))

@dataclass(frozen=True)
class WatershedConfig:
    radius: float = 25.
    cost_weight: float = 4.
    connectivity: int = 8
    percentile_low: float = 1.
    percentile_high: float = 99.
    # Preserve original seed pixels and do NOT merge overlapping grown labels.


def normalize_cost(image, low=1., high=99.):
    x = np.asarray(image, dtype=np.float64)
    if x.ndim != 2 or not np.isfinite(x).all():
        raise ValueError('Expected a finite 2D nonnuclear fluorescence image')
    lo, hi = np.percentile(x, [low, high])
    return np.clip((x - lo) / max(hi - lo, 1e-12), 0., 1.).astype(np.float32)


def components(mask):
    return ndi.label(np.asarray(mask, dtype=bool), STRUCTURE_8)


def geodesic_watershed(seed_mask, nonnuclear_image, config=WatershedConfig()):
    """Return uint32 instance IDs with geodesic competition and per-seed radius.

    Uses 8-neighbor Dijkstra edge weight = length * average(c(source),c(dest)),
    c=1+cost_weight*normalized_intensity. All original connected components
    remain fixed seeds. Each label's expansion is limited by Euclidean distance
    to ANY pixel in that seed (not to the centroid or another label's seed).
    """
    if config.radius < 0 or config.cost_weight < 0:
        raise ValueError('radius and cost_weight must be nonnegative')
    if config.connectivity not in (4, 8):
        raise ValueError('connectivity must be 4 or 8')
    seed_labels, n_seeds = components(seed_mask)
    h, w = seed_labels.shape
    if np.shape(nonnuclear_image) != (h, w):
        raise ValueError('Nonnuclear image and seeds must have matching 2D shape')
    if n_seeds == 0:
        return np.zeros((h, w), dtype=np.int32)
    norm = normalize_cost(nonnuclear_image, config.percentile_low, config.percentile_high)
    cost = 1. + config.cost_weight * norm
    owner = seed_labels.astype(np.int32)
    best = np.full((h, w), np.inf, dtype=np.float64)
    best[owner > 0] = 0.
    # Seed-specific distance maps in local bounding boxes; avoids allocating
    # an H*W distance map per seed while keeping exact distances.
    distance_regions = {}
    margin = int(np.ceil(config.radius)) + 1
    slices = ndi.find_objects(seed_labels)
    for j, sl in enumerate(slices, 1):
        if sl is None:
            continue
        ys, xs = sl
        y0, y1 = max(0, ys.start - margin), min(h, ys.stop + margin)
        x0, x1 = max(0, xs.start - margin), min(w, xs.stop + margin)
        local = seed_labels[y0:y1, x0:x1] == j
        distance_regions[j] = (y0, x0, ndi.distance_transform_edt(~local))
    heap = [(0., int(y), int(x), int(seed_labels[y, x]))
            for y, x in zip(*np.nonzero(seed_labels))]
    heapq.heapify(heap)
    neighbors = NEIGHBORS if config.connectivity == 8 else tuple(n for n in NEIGHBORS if n[2] == 1.)
    while heap:
        d, y, x, label = heapq.heappop(heap)
        if owner[y, x] != label or d > best[y, x] + 1e-12:
            continue
        for dy, dx, length in neighbors:
            yy, xx = y + dy, x + dx
            if yy < 0 or xx < 0 or yy >= h or xx >= w:
                continue
            # Original seed pixels cannot be overwritten by competitors.
            if seed_labels[yy, xx] and seed_labels[yy, xx] != label:
                continue
            y0, x0, distances = distance_regions[label]
            ry, rx = yy - y0, xx - x0
            if ry < 0 or rx < 0 or ry >= distances.shape[0] or rx >= distances.shape[1]:
                continue
            if distances[ry, rx] > config.radius:
                continue
            new_d = d + length * 0.5 * (float(cost[y, x]) + float(cost[yy, xx]))
            if new_d < best[yy, xx] - 1e-10:
                best[yy, xx], owner[yy, xx] = new_d, label
                heapq.heappush(heap, (new_d, yy, xx, label))
    return owner


def instance_score(pred_labels, true_labels, iou_threshold=0.5,
                   substantial_pixels=3, substantial_fraction=0.1):
    """IoU matching on INSTANCE IDs, including overlaps/merges/splits and PQ."""
    pred = np.asarray(pred_labels, dtype=np.int64)
    truth = np.asarray(true_labels, dtype=np.int64)
    if pred.shape != truth.shape or pred.ndim != 2 or pred.min() < 0 or truth.min() < 0:
        raise ValueError('Expected same-shaped 2D nonnegative instance label maps')
    # Relabel while ensuring background index zero.
    p_ids, p_rev = np.unique(pred, return_inverse=True)
    t_ids, t_rev = np.unique(truth, return_inverse=True)
    p_idx = p_rev + int(p_ids[0] != 0)
    t_idx = t_rev + int(t_ids[0] != 0)
    if p_ids[0] != 0:  # all foreground
        p_idx = p_rev + 1
    if t_ids[0] != 0:
        t_idx = t_rev + 1
    npred = len(p_ids) - int(p_ids[0] == 0)
    ntrue = len(t_ids) - int(t_ids[0] == 0)
    counts = np.bincount((t_idx.ravel() * (npred + 1) + p_idx.ravel()),
                         minlength=(ntrue+1)*(npred+1)).reshape(ntrue+1, npred+1)
    overlap = counts[1:, 1:].astype(np.float64)
    areas_t = counts[1:, :].sum(axis=1)
    areas_p = counts[:, 1:].sum(axis=0)
    union = areas_t[:, None] + areas_p[None, :] - overlap
    iou = np.divide(overlap, union, out=np.zeros_like(overlap), where=union > 0)
    matches = []
    if ntrue and npred:
        valid = iou >= iou_threshold
        bonus = min(ntrue, npred) + 1
        rr, cc = linear_sum_assignment(np.where(valid, -(bonus + iou), 0.))
        matches = [(r,c) for r,c in zip(rr,cc) if valid[r,c]]
    tp = len(matches)
    fp, fn = npred - tp, ntrue - tp
    f1 = 2*tp / (2*tp+fp+fn) if tp+fp+fn else np.nan
    pq = sum(iou[r,c] for r,c in matches) / (tp+0.5*fp+0.5*fn) if tp+fp+fn else np.nan
    if ntrue and npred:
        min_area = np.minimum(areas_t[:,None], areas_p[None,:])
        substantial = (overlap >= substantial_pixels) & (overlap >= substantial_fraction*min_area)
        merges = int(np.sum(substantial.sum(axis=0) >= 2))
        splits = int(np.sum(substantial.sum(axis=1) >= 2))
    else:
        merges = splits = 0
    return dict(tp=int(tp), fp=int(fp), fn=int(fn), gt_instances=int(ntrue),
                pred_instances=int(npred), instance_f1=float(f1), pq=float(pq),
                merge_count=merges, split_count=splits,
                merge_rate=merges/npred if npred else np.nan,
                split_rate=splits/ntrue if ntrue else np.nan)


def evaluate_prediction(prob, target, nonnuclear_image, config=WatershedConfig(),
                        ignore_index=-100, return_maps=False, iou_threshold=0.5):
    """Argmax predicted nuclear seeds -> watershed vs GT nuclear components.

    Ignore-index pixels are excluded from ground truth and *both* matched
    maps, so boundaries with nuclear/nonnuclear GT overlap are not penalized.
    """
    p = np.asarray(prob)
    y = np.asarray(target)
    if p.ndim != 3 or p.shape[0] != 3 or p.shape[1:] != y.shape:
        raise ValueError('prob must be [3,H,W], target [H,W]')
    valid = y != ignore_index
    initial_seeds = (p.argmax(0) == 1) & valid
    grown = geodesic_watershed(initial_seeds, nonnuclear_image, config)
    grown = np.where(valid, grown, 0)
    reference, _ = components((y == 1) & valid)
    stats = instance_score(grown, reference, iou_threshold=iou_threshold)
    stats['nuclear_dice_before_growth'] = _binary_dice(initial_seeds, y==1)
    stats['nuclear_dice_after_growth'] = _binary_dice(grown>0, y==1, valid)
    if return_maps:
        return stats, dict(seeds=initial_seeds, grown=grown, ground_truth=reference)
    return stats


def _binary_dice(a, b, valid=None):
    if valid is None:
        valid = np.ones_like(a, dtype=bool)
    a, b = (np.asarray(m, dtype=bool) & valid for m in (a,b))
    denom = a.sum() + b.sum()
    return float(2 * (a & b).sum() / denom) if denom else np.nan
