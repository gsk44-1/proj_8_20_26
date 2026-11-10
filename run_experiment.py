"""Orchestrates frozen UNet -> anchored fixed MM -> semantic/CC metrics.

This script deliberately does not train, threshold, watershed, or mutate the
original energy. Each sampled volume is independent; stats are 2D per slice.
"""
from dataclasses import asdict, dataclass, field
from pathlib import Path
import csv
import json
import math
import numpy as np
import torch
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from mm_fixed_compatible import Config, Energy
from synthetic_fluor_stains_three_classes_weighted import SyntheticFluorStains
from anchored_refinement import RefinementSettings, refine
from instance_metrics import score_batch, components
from unet_inference import load_unet, predict_volume, volume_labels_to_semantic


@dataclass
class RunConfig:
    checkpoint: str = '/content/drive/MyDrive/Colab Notebooks/proj_26/checkpoints_unet_baseline_v1_boosted_ce'
    output_dir: str = '/content/drive/MyDrive/Colab Notebooks/proj_26/anchored_mm_results'
    device: str = 'cuda' if torch.cuda.is_available() else 'cpu'
    seed: int = 100000   # disjoint from seed=0 indexed training samples
    n_volumes: int = 4  # increase to 50 for full validation
    volume_N: int = 192
    z_slices: int = 16
    z_ratio: float = 2
    n_seeds: int = 45
    class_probs: tuple = (0.4, 0.2, 0.4)
    domain: str = 'a'
    inference_batch_size: int = 4
    refine_batch_size: int = 2
    anchor_values: tuple = (0., 0.02, 0.1, 0.5)
    steps: int = 400
    checkpoints: tuple = (0, 25, 50, 100, 200, 300, 400)
    step_size: float = 0.02
    backtracking: bool = False
    # Initial values only, NOT verified hand-tuned optimum.
    energy: dict = field(default_factory=lambda: dict(
        lam=0.2, epsilon=0.5, well_cubic_weight=1.,
        barrier_weight=0.1, barrier_width=5, nuclear_band_width=2,
        barrier_length=13, barrier_longitudinal_sigma=3.5,
        barrier_orientations=8, boundary_weight=0.5,
        boundary_middle_width=3, boundary_outer_width=2,
        boundary_orientations=16, learn_strengths=False, learn_residuals=False))
    unet_kwargs: dict = field(default_factory=dict)
    min_overlap_pixels: int = 3
    min_overlap_fraction: float = 0.1
    iou_threshold: float = 0.5
    preview_volume: int = 0
    preview_slice: int = 8


def _save_csv(path, rows):
    if not rows:
        return
    with open(path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def summarize(rows):
    """Macro average of per-slice metrics, plus global TP/FP/FN instance F1."""
    groups = {}
    for row in rows:
        k = (row['anchor_mu'], row['iteration'])
        groups.setdefault(k, []).append(row)
    summary = []
    for (mu, iteration), rs in sorted(groups.items()):
        result = {'anchor_mu': mu, 'iteration': iteration, 'slices': len(rs)}
        for key in ('dice_0', 'dice_1', 'dice_2', 'instance_f1', 'pq', 'merge_rate', 'split_rate'):
            vals = np.asarray([r[key] for r in rs], dtype=float)
            result[key] = float(np.nanmean(vals)) if np.isfinite(vals).any() else float('nan')
        for key in ('tp', 'fp', 'fn', 'gt_instances', 'pred_instances', 'merge_count', 'split_count'):
            result[key] = int(sum(r[key] for r in rs))
        denom = 2 * result['tp'] + result['fp'] + result['fn']
        result['micro_instance_f1'] = 2 * result['tp'] / denom if denom else float('nan')
        summary.append(result)
    return summary


def plot_summary(summary, path):
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    for mu in sorted({r['anchor_mu'] for r in summary}):
        sub = sorted((r for r in summary if r['anchor_mu'] == mu), key=lambda r: r['iteration'])
        for ax, key in zip(axes, ('dice_1', 'micro_instance_f1', 'merge_rate')):
            ax.plot([r['iteration'] for r in sub], [r[key] for r in sub], marker='o', label=f'mu={mu:g}')
    for ax, name in zip(axes, ('Nuclear Dice', 'Instance F1 (micro)', 'Merge rate (macro)')):
        ax.set(title=name, xlabel='Iterations'); ax.grid(alpha=0.3)
    axes[0].legend(fontsize=8)
    fig.tight_layout(); fig.savefig(path, dpi=160); plt.close(fig)


def plot_preview(images, gt, raw, refined, path, slice_idx=0):
    """Show raw microscopy, semantic classes and 8-connected instance labels."""
    z = min(slice_idx, gt.shape[0]-1)
    raw_sem = raw[z].argmax(0).numpy()
    ref_sem = refined[z].argmax(0).numpy()
    truth = gt[z].numpy()
    fig, axes = plt.subplots(2, 4, figsize=(15, 8))
    a = images[z]
    axes[0, 0].imshow(a[0], cmap='gray'); axes[0,0].set_title('Input channel 0')
    for ax, im, name in zip(axes[0,1:], (truth, raw_sem, ref_sem), ('Ground truth','Raw UNet','Refined')):
        ax.imshow(im, vmin=0, vmax=2, cmap='viridis', interpolation='nearest'); ax.set_title(name)
    axes[1,0].imshow(a[1], cmap='gray'); axes[1,0].set_title('Input channel 1')
    for ax, im, name in zip(axes[1,1:], (truth, raw_sem, ref_sem), ('GT CC (8)','Raw CC (8)','Refined CC (8)')):
        labels, n = components(im == 1)
        ax.imshow(labels, cmap='nipy_spectral', interpolation='nearest'); ax.set_title(f'{name}: {n} instances')
    for ax in axes.flat: ax.axis('off')
    fig.tight_layout(); fig.savefig(path, dpi=160); plt.close(fig)


def run(cfg: RunConfig):
    out = Path(cfg.output_dir); out.mkdir(parents=True, exist_ok=True)
    (out / 'settings.json').write_text(json.dumps(asdict(cfg), indent=2))
    print(f'Device: {cfg.device}; source seed: {cfg.seed}; volumes: {cfg.n_volumes}')
    torch.manual_seed(cfg.seed); np.random.seed(cfg.seed)
    unet, ckpt_path = load_unet(cfg.checkpoint, cfg.device, cfg.unet_kwargs)
    print(f'Loaded frozen UNet checkpoint: {ckpt_path}')
    energy_cfg = Config(**cfg.energy)
    energy_cfg.validate()
    energy = Energy(energy_cfg).to(cfg.device).eval().requires_grad_(False)
    dataset = SyntheticFluorStains(samples_per_epoch=cfg.n_volumes, seed=cfg.seed,
                                  N=cfg.volume_N, z_slices=cfg.z_slices,
                                  z_ratio=cfg.z_ratio, n_seeds=cfg.n_seeds,
                                  class_probs=cfg.class_probs, domain=cfg.domain)
    # batch_size=1, num_workers=0 avoids re-generating 3D volumes per slice.
    # Set a different seed and avoid overlap with samples used for UNet training.
    loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0)
    rows = []
    for vol_idx, (images, labels) in enumerate(loader):
        images = images[0]
        labels = volume_labels_to_semantic(labels[0])
        raw = predict_volume(unet, images, cfg.device, cfg.inference_batch_size)
        # One raw baseline row per (mu, horizon), for easy paired comparisons.
        raw_scores = score_batch(raw.numpy(), labels.numpy(), iou_threshold=cfg.iou_threshold,
                                 min_overlap_pixels=cfg.min_overlap_pixels,
                                 min_overlap_fraction=cfg.min_overlap_fraction)
        for mu in cfg.anchor_values:
            snapshots = {t: [] for t in set(cfg.checkpoints) | {0, cfg.steps}}
            for batch in raw.split(cfg.refine_batch_size):
                settings = RefinementSettings(anchor_mu=mu, step_size=cfg.step_size,
                    steps=cfg.steps, checkpoints=cfg.checkpoints,
                    backtracking=cfg.backtracking)
                snap, _trace = refine(energy, batch.to(cfg.device), settings)
                for t, tensor in snap.items():
                    snapshots[t].append(tensor)
            for t in sorted(snapshots):
                refined = torch.cat(snapshots[t], dim=0)
                metrics = raw_scores if t == 0 else score_batch(refined.numpy(), labels.numpy(),
                    iou_threshold=cfg.iou_threshold, min_overlap_pixels=cfg.min_overlap_pixels,
                    min_overlap_fraction=cfg.min_overlap_fraction)
                for z, stats in enumerate(metrics):
                    rows.append({'volume': vol_idx, 'slice': z, 'anchor_mu': mu,
                                 'iteration': t, **stats})
                if vol_idx == cfg.preview_volume and t == cfg.steps:
                    plot_preview(images.numpy(), labels, raw, refined,
                                 out / f'preview_mu_{mu:g}.png', cfg.preview_slice)
            print(f'Volume {vol_idx+1}/{cfg.n_volumes}, anchor {mu:g}: done', flush=True)
        _save_csv(out / 'per_slice.csv', rows)
        interim = summarize(rows)
        _save_csv(out / 'summary.csv', interim)
        plot_summary(interim, out / 'curves.png')
    print(f'Completed. Output folder: {out}')
    return summarize(rows)


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--output-dir', required=True)
    parser.add_argument('--volumes', type=int, default=4)
    args = parser.parse_args()
    run(RunConfig(checkpoint=args.checkpoint, output_dir=args.output_dir, n_volumes=args.volumes))
