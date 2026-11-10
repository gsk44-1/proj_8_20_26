"""2D semantic Dice and connected-components instance scoring (8-connectivity)."""
import numpy as np
from scipy import ndimage
from scipy.optimize import linear_sum_assignment

STRUCTURE_8 = np.ones((3, 3), dtype=np.uint8)


def components(mask):
    """Return (labels, count), treating diagonal touches as connected."""
    return ndimage.label(np.asarray(mask, dtype=bool), structure=STRUCTURE_8)


def dice_by_class(prediction, target, n_classes=3, ignore_index=-100):
    valid = target != ignore_index
    out = {}
    for k in range(n_classes):
        p = (prediction == k) & valid
        g = (target == k) & valid
        denom = int(p.sum() + g.sum())
        out[f'dice_{k}'] = float(2 * (p & g).sum() / denom) if denom else np.nan
    return out


def instance_stats(pred_semantic, gt_semantic, iou_threshold=0.5,
                   min_overlap_pixels=3, min_overlap_fraction=0.1, ignore_index=-100):
    """2D nuclei as 8-connected components; no watershed or size filtering.

    IoU matching is maximum-cardinality one-to-one at the given threshold,
    with IoU as the tie breaker. Merge: one predicted component substantially
    overlaps >=2 GT instances. Split: one GT instance substantially overlaps
    >=2 predicted components. Substantial overlap requires >=min_overlap_pixels
    AND intersection/min(areaGT, areaPred) >= min_overlap_fraction.
    Rates: merge_count / number of predicted instances; split_count / number
    of GT instances. All objects in a slice are included, including small ones.
    """
    gt = np.asarray(gt_semantic)
    pred = np.asarray(pred_semantic)
    if gt.shape != pred.shape or gt.ndim != 2:
        raise ValueError('Expected matching 2D semantic arrays')
    valid = gt != ignore_index
    true_labels, nt = components((gt == 1) & valid)
    pred_labels, npred = components((pred == 1) & valid)
    # Contingency table (rows GT incl background, cols predictions incl bg)
    counts = np.bincount((true_labels.ravel() * (npred + 1) + pred_labels.ravel()),
                         minlength=(nt + 1) * (npred + 1)).reshape(nt + 1, npred + 1)
    overlap = counts[1:, 1:].astype(np.float64)
    areas_t = np.bincount(true_labels.ravel(), minlength=nt+1)[1:]
    areas_p = np.bincount(pred_labels.ravel(), minlength=npred+1)[1:]
    union = areas_t[:, None] + areas_p[None, :] - overlap
    iou = np.divide(overlap, union, out=np.zeros_like(overlap), where=union > 0)
    if nt and npred:
        allowed = iou >= iou_threshold
        # High bonus prioritizes maximum number of valid matches before IoU.
        cost = np.where(allowed, -(min(nt, npred) + 1 + iou), 0.)
        rows, cols = linear_sum_assignment(cost)
        matches = [(int(r), int(c)) for r, c in zip(rows, cols) if allowed[r, c]]
    else:
        matches = []
    tp = len(matches)
    fp, fn = npred - tp, nt - tp
    instance_f1 = 2 * tp / (2 * tp + fp + fn) if (2 * tp + fp + fn) else np.nan
    sum_iou = sum(iou[r, c] for r, c in matches)
    pq = sum_iou / (tp + 0.5 * fp + 0.5 * fn) if (tp + fp + fn) else np.nan
    if nt and npred:
        min_area = np.minimum(areas_t[:, None], areas_p[None, :])
        substantial = (overlap >= min_overlap_pixels) & (overlap >= min_overlap_fraction * min_area)
        merges = int(np.sum(substantial.sum(axis=0) >= 2))
        splits = int(np.sum(substantial.sum(axis=1) >= 2))
    else:
        merges = splits = 0
    return {
        'gt_instances': int(nt), 'pred_instances': int(npred), 'tp': int(tp),
        'fp': int(fp), 'fn': int(fn), 'instance_f1': float(instance_f1),
        'pq': float(pq), 'merge_count': merges, 'split_count': splits,
        'merge_rate': merges / npred if npred else np.nan,
        'split_rate': splits / nt if nt else np.nan,
    }


def score_batch(probabilities, labels, **instance_kwargs):
    """CPU tensors / numpy arrays [B,3,H,W], [B,H,W]; one dict per slice."""
    probs = np.asarray(probabilities)
    y = np.asarray(labels)
    pred = probs.argmax(axis=1)
    results = []
    for a, b in zip(pred, y):
        results.append({**dice_by_class(a, b), **instance_stats(a, b, **instance_kwargs)})
    return results
