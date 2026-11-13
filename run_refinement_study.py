"""Paired corrupted-GT ablation evaluation with Euclidean seeded-expansion instance F1.

Run from notebook, or import build_validation_data and run_study.
"""
from __future__ import annotations
import csv
import json
from dataclasses import asdict, dataclass
from pathlib import Path
import numpy as np
import torch
from dataclasses import replace
from mm_fixed_compatible import Config, synthetic_logits, target_from_stain_labels
from anchored_refinement import anchor_energy
from template_energy import anchored_rollout
from refinement_ablations import ABLATIONS, make_energy, energy_breakdown
from geodesic_instances import WatershedConfig, evaluate_prediction

@dataclass
class StudyConfig:
    output_dir: str = 'refinement_results'
    device: str = 'cuda' if torch.cuda.is_available() else 'cpu'
    n_volumes: int = 2
    n_slices_per_volume: int = 4
    volume_seed: int = 320000
    corruption_seed: int = 840000
    image_size: int = 192
    z_slices: int = 16
    z_ratio: float = 2.
    n_seeds: int = 45
    class_probs: tuple = (0.4, 0.2, 0.4)
    domain: str = 'a'
    noise_levels: tuple = (0.4, 0.8, 1.2)
    blur_sigma: float = 0.75
    horizons: tuple = (0, 25, 50, 100, 200, 300, 400)
    step_size: float = 0.02
    anchor_mu: float = 0.4
    lam: float = 0.2
    epsilon: float = 0.5
    cubic_weight: float = 1.0
    template_dir: str = 'templates'
    orientations: int = 16
    fixed_strength: float = 0.05
    strength_cap: float = 0.3
    # Optional saved template model, same weights for all template-bearing ablations.
    template_checkpoint: str = ''
    watershed_radius: float = 25.
    watershed_cost_weight: float = 0.0
    iou_threshold: float = 0.5
    centroid_distance_threshold: float = 5.
    example_volumes: int = 3
    example_noise_level: float | None = None  # None uses first configured noise level
    save_previews: bool = True


def build_validation_data(config: StudyConfig, volume_loader=None):
    """Precompute paired validation examples identical for every ablation.

    A volume_loader(index) may return (images[Z,2,H,W],labels[Z,3,H,W])
    to bypass optional psfmodels/synthetic generator dependencies in tests.
    """
    if volume_loader is None:
        from synthetic_fluor_stains_three_classes_weighted import SyntheticFluorStains
        ds = SyntheticFluorStains(samples_per_epoch=config.n_volumes,
                                  seed=config.volume_seed, N=config.image_size,
                                  z_slices=config.z_slices, z_ratio=config.z_ratio,
                                  n_seeds=config.n_seeds, class_probs=config.class_probs,
                                  domain=config.domain)
        volume_loader = lambda i: ds[i]
    cfg = Config(lam=config.lam, epsilon=config.epsilon,
                 gt_blur_sigma=config.blur_sigma, perturb_strength=0.)
    examples = []
    for i in range(config.n_volumes):
        images, labels = volume_loader(i)
        images = np.asarray(images)
        labels = torch.as_tensor(labels).float()
        if images.ndim != 4 or images.shape[1] != 2:
            raise ValueError('Expected image [Z,2,H,W]')
        if labels.ndim != 4 or labels.shape[1] < 2:
            raise ValueError('Expected labels [Z,3,H,W]')
        target = target_from_stain_labels(labels)
        chosen = np.unique(np.linspace(0, len(target)-1,
                        min(config.n_slices_per_volume, len(target)), dtype=int))
        for z in chosen:
            for noise in config.noise_levels:
                # Stable seed based on sample and noise index, NOT ablation.
                noise_idx = list(config.noise_levels).index(noise)
                seed = config.corruption_seed + i*100000 + int(z)*100 + noise_idx
                local_cfg = replace(cfg, perturb_strength=float(noise))
                original = synthetic_logits(target[z][None], local_cfg,
                                            seed=seed).softmax(1)[0].cpu()
                examples.append(dict(volume=i, z=int(z), noise=float(noise),
                                     original=original, target=target[z].cpu(),
                                     nonnuclear=np.asarray(images[z, 1], dtype=np.float32).copy(),
                                     initial_seed=seed))
        print(f'Prepared volume {i+1}/{config.n_volumes}: {len(chosen)} slices', flush=True)
    return examples


def _write_csv(filename, rows):
    if not rows:
        return
    filename.parent.mkdir(parents=True, exist_ok=True)
    columns = list(dict.fromkeys(k for row in rows for k in row))
    with filename.open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)


def _as_numpy(p):
    return p.detach().float().cpu().numpy()[0]


def summarize_rows(rows):
    """Pooled instance F1, macro-over-slices F1, and mean Dice for each condition."""
    groups = {}
    for row in rows:
        groups.setdefault((row['ablation'], row['noise'], row['iteration']), []).append(row)
    result = []
    for (name, noise, step), sub in groups.items():
        tp, fp, fn = (sum(int(r[k]) for r in sub) for k in ('tp','fp','fn'))
        f1 = 2*tp / max(1, 2*tp+fp+fn)
        def mean(key):
            vals = np.array([r[key] for r in sub], dtype=float)
            return float(np.nanmean(vals)) if np.isfinite(vals).any() else float('nan')
        utp, ufp, ufn = (sum(int(r['unexpanded_' + k]) for r in sub) for k in ('tp','fp','fn'))
        ctp, cfp, cfn = (sum(int(r['centroid_' + k]) for r in sub) for k in ('tp','fp','fn'))
        result.append(dict(ablation=name, noise=noise, iteration=step,
                           n_slices=len(sub), micro_instance_f1=f1,
                           unexpanded_f1=2*utp/max(1,2*utp+ufp+ufn),
                           centroid_f1=2*ctp/max(1,2*ctp+cfp+cfn),
                           dice_macro=mean('dice_macro'),
                           dice_nuclear=mean('dice_nuclear'),
                           dice_background=mean('dice_background'),
                           dice_nonnuclear=mean('dice_nonnuclear'),
                           macro_instance_f1=mean('instance_f1'),
                           mean_pq=mean('pq'),
                           nuclear_dice_before_growth=mean('nuclear_dice_before_growth'),
                           nuclear_dice_after_growth=mean('nuclear_dice_after_growth'),
                           merges=sum(int(r['merge_count']) for r in sub),
                           splits=sum(int(r['split_count']) for r in sub),
                           tp=tp, fp=fp, fn=fn,
                           unexpanded_tp=utp, unexpanded_fp=ufp, unexpanded_fn=ufn,
                           centroid_tp=ctp, centroid_fp=cfp, centroid_fn=cfn))
    by_ref = {(r['ablation'],r['noise']):r['micro_instance_f1']
              for r in result if r['iteration']==0}
    for r in result:
        r['delta_f1'] = r['micro_instance_f1'] - by_ref[(r['ablation'],r['noise'])]
    return result


def run_study(examples, config: StudyConfig, selected=None):
    """Evaluate fixed paired examples over all requested horizons and ablations.

    Returns (rows, summary, traces) and writes CSV/JSON plus sample arrays.
    """
    import matplotlib
    matplotlib.use('Agg', force=True)
    import matplotlib.pyplot as plt
    out = Path(config.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    (out/'config.json').write_text(json.dumps(asdict(config), indent=2))
    selected = tuple(selected) if selected is not None else ABLATIONS
    horizons = sorted(set((0,) + tuple(int(x) for x in config.horizons)))
    if config.example_volumes < 0:
        raise ValueError('example_volumes must be >=0')
    if config.centroid_distance_threshold < 0:
        raise ValueError('centroid_distance_threshold must be nonnegative')
    ws = WatershedConfig(radius=config.watershed_radius,
                         cost_weight=config.watershed_cost_weight)
    rows, traces = [], []
    chosen_noise = (config.noise_levels[0] if config.example_noise_level is None
                    else config.example_noise_level)
    volume_ids = list(dict.fromkeys(e['volume'] for e in examples))[:config.example_volumes]
    # One slice per independent volume, and only one chosen noise level.
    chosen_examples = {}
    for idx, ex in enumerate(examples):
        if ex['volume'] in volume_ids and ex['noise'] == chosen_noise:
            chosen_examples.setdefault(ex['volume'], idx)
    chosen_indices = set(chosen_examples.values())
    for si, spec in enumerate(selected):
        print(f'[{si+1}/{len(selected)}] {spec.name}', flush=True)
        if spec.templates and not list(Path(config.template_dir).glob('*.png')):
            raise FileNotFoundError('Template ablations require original 21x21 PNG files in '
                                    f'{config.template_dir!r}. No templates were attached.')
        model = make_energy(spec, lam=config.lam, epsilon=config.epsilon,
                            well_cubic_weight=config.cubic_weight,
                            template_dir=config.template_dir, orientations=config.orientations,
                            fixed_strength=config.fixed_strength, strength_cap=config.strength_cap,
                            checkpoint=config.template_checkpoint if spec.templates and config.template_checkpoint else None,
                            device=config.device)
        example_trajectories = {}
        for ix, ex in enumerate(examples):
            original = ex['original'][None].to(config.device)
            current = original.clone()
            last = 0
            for step in horizons:
                if step > last and spec.name != 'No refinement (anchor only)':
                    current = anchored_rollout(model,current,original,
                        steps=step-last, mu=config.anchor_mu,
                        step_size=config.step_size, differentiable=False)
                last = step
                probs = _as_numpy(current)
                score, maps = evaluate_prediction(probs, ex['target'].numpy(),
                    ex['nonnuclear'], ws, return_maps=True,
                    iou_threshold=config.iou_threshold,
                    centroid_distance_threshold=config.centroid_distance_threshold)
                if ix in chosen_indices and config.save_previews:
                    example_trajectories.setdefault(ix, {})[step] = probs.argmax(0).copy()
                row = dict(ablation=spec.name, volume=ex['volume'],
                           slice=ex['z'], noise=ex['noise'], iteration=step, **score)
                rows.append(row)
                energy = energy_breakdown(model, current, original, config.anchor_mu)
                traces.append(dict(ablation=spec.name, volume=ex['volume'],
                              slice=ex['z'], noise=ex['noise'], iteration=step,
                              **energy, total=sum(energy.values())))
                if ix == 0 and config.save_previews and step in (0,horizons[-1]):
                    fig, axs = plt.subplots(1,4,figsize=(14,3.5),constrained_layout=True)
                    axs[0].imshow(maps['ground_truth'],cmap='nipy_spectral',interpolation='nearest')
                    axs[0].set_title('Expanded GT instances')
                    axs[1].imshow(probs.argmax(0)==1,cmap='gray',vmin=0,vmax=1)
                    axs[1].set_title('Predicted seeds')
                    axs[2].imshow(maps['ground_truth_seeds'], cmap='gray',vmin=0,vmax=1)
                    axs[2].set_title('GT nuclear seeds')
                    axs[3].imshow(maps['grown'], cmap='nipy_spectral',interpolation='nearest')
                    axs[3].set_title(f'Expanded instances F1={score["instance_f1"]:.3f}')
                    for ax in axs: ax.axis('off')
                    fig.suptitle(f'{spec.name} | noise={ex["noise"]} | t={step}')
                    safe = ''.join(c if c.isalnum() else '_' for c in spec.name)
                    fig.savefig(out/f'{safe}_t{step}_preview.png',dpi=130)
                    plt.close(fig)
        if config.save_previews and example_trajectories:
            plot_three_class_trajectories(examples, example_trajectories, horizons,
                                          out / ('three_class_' + safe_filename(spec.name)
                                                 + '_noise_' + str(chosen_noise).replace('.','p') + '.png'),
                                          spec.name, chosen_noise)
        del model
        if torch.cuda.is_available(): torch.cuda.empty_cache()
        _write_csv(out/'slice_metrics.csv',rows)
        _write_csv(out/'energy_traces.csv',traces)
    summary = summarize_rows(rows)
    _write_csv(out/'summary.csv',summary)
    write_final_ablation_tables(summary, horizons, out)
    return rows, summary, traces


def plot_study(summary, output_dir=None):
    """One figure per noise level; pooled expanded-instance F1 vs iteration."""
    import matplotlib.pyplot as plt
    figures = []
    for noise in sorted({x['noise'] for x in summary}):
        fig, ax = plt.subplots(figsize=(9,5.5))
        for name in dict.fromkeys(r['ablation'] for r in summary):
            items = sorted((r for r in summary if r['noise']==noise and r['ablation']==name),
                           key=lambda r:r['iteration'])
            if items:
                ax.plot([x['iteration'] for x in items],
                        [x['micro_instance_f1'] for x in items], marker='o',label=name)
        ax.set(xlabel='Refinement iteration', ylabel='Pooled expanded-instance F1',
               title=f'Spectral perturbation strength {noise:g}')
        ax.grid(alpha=.25)
        ax.legend(fontsize=8)
        fig.tight_layout()
        if output_dir:
            p = Path(output_dir) / f'f1_vs_steps_noise_{noise:g}.png'
            fig.savefig(p, dpi=160)
        figures.append(fig)
    return figures


def safe_filename(name):
    return ''.join(ch if ch.isalnum() else '_' for ch in name)


def plot_three_class_trajectories(examples, trajectories, horizons, output_file,
                                  ablation_name='', noise=None):
    """Three-class categorical maps for one slice each from up to 3 volumes.

    One row per volume, GT at left and predictions at all requested horizons.
    A fourth color (gray) marks GT ignored (-100) pixels; predictions retain
    all three classes. `trajectories` maps example-index -> step -> [H,W] labels.
    """
    import matplotlib.pyplot as plt
    from matplotlib.colors import ListedColormap, BoundaryNorm
    from matplotlib.patches import Patch
    ordered = sorted(trajectories, key=lambda k: examples[k]['volume'])
    if not ordered:
        return None
    horizons = sorted(set(horizons))
    cmap = ListedColormap(['#232a38', '#f59e42', '#56c0cb', '#b3b3b3'])
    norm = BoundaryNorm([-0.5,.5,1.5,2.5,3.5], cmap.N)
    fig, axes = plt.subplots(len(ordered), 1+len(horizons),
                             figsize=(2.45*(1+len(horizons)), 2.65*len(ordered)),
                             squeeze=False, constrained_layout=True)
    for i, idx in enumerate(ordered):
        ex = examples[idx]
        gt = ex['target'].numpy().astype(int).copy()
        gt[gt == -100] = 3
        axes[i, 0].imshow(gt, cmap=cmap, norm=norm, interpolation='nearest')
        axes[i, 0].set_title('Ground truth' if i==0 else '')
        axes[i, 0].set_ylabel(f"Volume {ex['volume']} · slice {ex['z']}")
        for j, step in enumerate(horizons, 1):
            if step in trajectories[idx]:
                axes[i, j].imshow(trajectories[idx][step], cmap=cmap, norm=norm,
                                  interpolation='nearest')
            axes[i, j].set_title(f't={step}' if i==0 else '')
        for ax in axes[i]:
            ax.set_xticks([]); ax.set_yticks([])
    fig.legend(handles=[Patch(facecolor=c,label=n) for c,n in
                        zip(cmap.colors, ['Background','Nuclear','Nonnuclear','GT ignored (-100)'])],
               loc='outside lower center', ncol=4, fontsize=9)
    fig.suptitle(f'{ablation_name} | noise={noise} | three-class segmentation', fontsize=13)
    output_file = Path(output_file)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_file, dpi=145)
    plt.close(fig)
    return output_file


def final_ablation_table(summary, final_horizon):
    """One row per (noise, ablation), evaluated at exactly the final horizon."""
    return [dict(ablation=r['ablation'], noise=r['noise'], iteration=r['iteration'],
                 n_slices=r['n_slices'], f1_unexpanded=r['unexpanded_f1'],
                 f1_expanded=r['micro_instance_f1'], f1_centroid=r['centroid_f1'],
                 dice=r['dice_macro'], splits=r['splits'], merges=r['merges'])
            for r in summary if r['iteration'] == final_horizon]


def write_final_ablation_tables(summary, horizons, output_dir):
    """Export final-horizon report as CSV and styled HTML grouped by noise."""
    import html
    final_horizon = max(horizons)
    rows = final_ablation_table(summary, final_horizon)
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    _write_csv(out/'final_ablation_table.csv', rows)
    headers = [('ablation','Ablation'), ('f1_unexpanded','F1 (raw)'),
               ('f1_expanded','F1 (expanded)'), ('f1_centroid','F1 (centroid)'),
               ('dice','Macro Dice'), ('splits','Splits ↓'), ('merges','Merges ↓')]
    fragments = ['<html><head><meta charset="utf-8"><style>',
                 'body{font-family:Arial,sans-serif;margin:24px;color:#202735}',
                 'table{border-collapse:collapse;margin:14px 0 30px;font-size:14px}',
                 'th,td{padding:9px 12px;border-bottom:1px solid #ddd;text-align:right}',
                 'th:first-child,td:first-child{text-align:left}',
                 'th{background:#eef2f7}strong{font-weight:800}',
                 '</style></head><body>',
                 '<h1>Ablation comparison at t=' + str(final_horizon) + '</h1>',
                 '<p>F1: pooled TP/FP/FN across slices. Dice: mean of per-slice macro',
                 ' Dice over three classes (GT -100 excluded). Splits/merges:',
                 ' totals on expanded instances. Bold = best within noise level.</p>']
    for noise in sorted({r['noise'] for r in rows}):
        group = [r for r in rows if r['noise'] == noise]
        fragments.append('<h2>Noise level ' + html.escape(str(noise)) + '</h2><table><thead><tr>')
        fragments.extend('<th>' + label + '</th>' for _,label in headers)
        fragments.append('</tr></thead><tbody>')
        best = {}
        for key,_ in headers[1:]:
            vals = [float(r[key]) for r in group if np.isfinite(r[key])]
            best[key] = (min(vals) if key in ('splits','merges') else max(vals)) if vals else float('nan')
        for r in group:
            fragments.append('<tr><td>' + html.escape(str(r['ablation'])) + '</td>')
            for key,_ in headers[1:]:
                val = r[key]
                display = str(val) if key in ('splits','merges') else f'{val:.4f}'
                if np.isfinite(val) and np.isclose(val, best[key], atol=1e-10):
                    display = '<strong>' + display + '</strong>'
                fragments.append('<td>' + display + '</td>')
            fragments.append('</tr>')
        fragments.append('</tbody></table>')
    fragments.append('</body></html>')
    file = out/'final_ablation_table.html'
    file.write_text(''.join(fragments), encoding='utf-8')
    return rows, file
