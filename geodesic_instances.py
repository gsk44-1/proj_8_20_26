"""Radius-limited seeded instance expansion for nuclear seed components.

Ground truth and predictions are BOTH converted to connected-component nuclear
seeds, then expanded using the same Euclidean nearest-seed rule and radius. The older
geodesic-cost fields are kept in the config for backward compatibility but are
not used by the current Euclidean watershed.
"""
from __future__ import annotations
from dataclasses import dataclass
import numpy as np
from scipy import ndimage as ndi
from scipy.optimize import linear_sum_assignment

STRUCTURE_8 = np.ones((3, 3), dtype=np.uint8)

@dataclass(frozen=True)
class WatershedConfig:
    radius: float = 25.
    cost_weight: float = 0.0
    connectivity: int = 8
    percentile_low: float = 1.
    percentile_high: float = 99.
    # cost_weight / percentile_* are currently ignored by the Euclidean method.


def normalize_cost(image, low=1., high=99.):
    """Backward-compatible helper; not used by Euclidean watershed."""
    x = np.asarray(image, dtype=np.float64)
    if x.ndim != 2 or not np.isfinite(x).all():
        raise ValueError('Expected a finite 2D image')
    lo, hi = np.percentile(x, [low, high])
    return np.clip((x - lo) / max(hi - lo, 1e-12), 0., 1.).astype(np.float32)


def components(mask):
    return ndi.label(np.asarray(mask, dtype=bool), STRUCTURE_8)


def geodesic_watershed(seed_mask, nonnuclear_image=None, config=WatershedConfig()):
    """Return uint32 instance IDs by Euclidean nearest-seed expansion.

    Each 8-connected seed component defines one marker. Every pixel within the
    specified Euclidean distance `config.radius` of *any* seed pixel is assigned
    to the nearest seed component; pixels farther than the radius remain 0.

    The `nonnuclear_image` argument is accepted only for backward compatibility
    with earlier geodesic experiments and is ignored.
    """
    if config.radius < 0:
        raise ValueError('radius must be nonnegative')
    if config.connectivity not in (4, 8):
        raise ValueError('connectivity must be 4 or 8')
    seed_labels, n_seeds = ndi.label(np.asarray(seed_mask, dtype=bool),
                                     STRUCTURE_8 if config.connectivity == 8 else ndi.generate_binary_structure(2, 1))
    h, w = seed_labels.shape
    if nonnuclear_image is not None and np.shape(nonnuclear_image) != (h, w):
        raise ValueError('Nonnuclear image and seeds must have matching 2D shape')
    if n_seeds == 0:
        return np.zeros((h, w), dtype=np.int32)

    # Distance to nearest seed pixel, plus its coordinates.
    distances, indices = ndi.distance_transform_edt(seed_labels == 0, return_indices=True)
    nearest_y, nearest_x = indices
    owner = seed_labels[nearest_y, nearest_x].astype(np.int32)
    owner[distances > config.radius] = 0
    # Preserve original seed labels exactly.
    owner[seed_labels > 0] = seed_labels[seed_labels > 0]
    return owner


def instance_score(pred_labels, true_labels, iou_threshold=0.5,
                   substantial_pixels=3, substantial_fraction=0.1):
    """IoU matching on INSTANCE IDs, including overlaps/merges/splits and PQ."""
    pred = np.asarray(pred_labels, dtype=np.int64)
    truth = np.asarray(true_labels, dtype=np.int64)
    if pred.shape != truth.shape or pred.ndim != 2 or pred.min() < 0 or truth.min() < 0:
        raise ValueError('Expected same-shaped 2D nonnegative instance label maps')
    p_ids, p_rev = np.unique(pred, return_inverse=True)
    t_ids, t_rev = np.unique(truth, return_inverse=True)
    p_idx = p_rev + int(p_ids[0] != 0)
    t_idx = t_rev + int(t_ids[0] != 0)
    if p_ids[0] != 0:
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
    """Score identically expanded prediction and GT nuclear components.

    Prediction seeds are `argmax(prob) == 1`; reference seeds are `target == 1`.
    Both use the same Euclidean nearest-component expansion and radius. GT
    `ignore_index` values are not used as barriers or masks during expansion.
    """
    p = np.asarray(prob)
    y = np.asarray(target)
    if p.ndim != 3 or p.shape[0] != 3 or p.shape[1:] != y.shape:
        raise ValueError('prob must be [3,H,W], target [H,W]')
    initial_seeds = (p.argmax(0) == 1)
    reference_seeds = (y == 1)
    grown = geodesic_watershed(initial_seeds, nonnuclear_image, config)
    reference = geodesic_watershed(reference_seeds, nonnuclear_image, config)
    stats = instance_score(grown, reference, iou_threshold=iou_threshold)
    stats['nuclear_dice_before_growth'] = _binary_dice(initial_seeds, reference_seeds)
    stats['nuclear_dice_after_growth'] = _binary_dice(grown > 0, reference > 0)
    if return_maps:
        return stats, dict(seeds=initial_seeds, grown=grown,
                           ground_truth_seeds=reference_seeds, ground_truth=reference)
    return stats


def _binary_dice(a, b, valid=None):
    if valid is None:
        valid = np.ones_like(a, dtype=bool)
    a, b = (np.asarray(m, dtype=bool) & valid for m in (a,b))
    denom = a.sum() + b.sum()
    return float(2 * (a & b).sum() / denom) if denom else np.nan
