"""Train the exact fixed-compatible MM energy on real stained segmentation labels.

Requires mm_fixed_compatible.py beside this file.

Two-stage corruption, BOTH Gaussian blur followed by spectral logit noise:
  (1) Ground truth -> blurred one-hot -> noisy logits -> softmax -> p0.
  (2) Detached warm-up trajectory p_w -> blurred p_w -> noisy logits
      -> softmax -> three differentiable projected-gradient steps.

The second corruption is applied to the *current warm-up state*, NOT a fresh
copy of the clean target. Train/validation must use different dataloaders.
No morphological dilation/erosion in this first real-data experiment.

Example:
    from train_mm_real import RealTrainConfig, train_real
    model, history = train_real(train_dataloader, val_dataloader,
        energy_cfg=your_existing_mm_config,
        train_cfg=RealTrainConfig(device='cuda'))
"""
from __future__ import annotations

import copy
import json
from dataclasses import dataclass, asdict, replace
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import torch
import torch.nn.functional as F
from scipy.ndimage import gaussian_filter

from mm_fixed_compatible import (
    Config, Energy, projected_step, target_from_stain_labels,
    synthetic_logits, spectral_noise,
)


@dataclass
class RealTrainConfig:
    updates: int = 250
    learning_rate: float = 0.002
    step_size: float = 0.02
    warmup_steps: tuple[int, ...] = (0, 20, 80, 160, 300)
    differentiable_steps: int = 3
    # Independent blur and spectral intensities at the two corruption points.
    initial_blur_range: tuple[float, float] = (0.0, 1.5)
    initial_spectral_range: tuple[float, float] = (0.4, 1.2)
    restart_blur_range: tuple[float, float] = (0.0, 0.8)
    restart_spectral_range: tuple[float, float] = (0.0, 0.5)
    restart_corruption_probability: float = 1.0
    # Preserves the warm-up state's confidence when restart noise = 0.
    # 'log_prob' adds the spectral noise in log-probability/logit space.
    logit_floor: float = 1e-5
    crop_size: int | None = 128  # None to use whole images
    nuclear_loss_weight: float = 3.0
    nonnuclear_loss_weight: float = 2.0
    background_loss_weight: float = 1.0
    false_positive_weight: float = 0.25
    strength_drift_weight: float = 0.001
    residual_penalty_weight: float = 0.01
    stationarity_weight: float = 0.0
    learn_strengths: bool = True
    learn_residuals: bool = False
    gradient_clip_norm: float = 5.0
    eval_every: int = 25
    validation_examples: int = 8
    validation_horizons: tuple[int, ...] = (0, 50, 150, 300, 400)
    # The long-horizon evaluation uses the same fixed-step solver as training.
    # A separate Experiment.run can evaluate with line search afterwards.
    seed: int = 123
    device: str = 'cpu'
    output_dir: str = 'mm_real_training'
    labels_key: str = 'labels'
    max_val_batches: int | None = None
    preview_every: int = 0  # 0: only initial / best / final

    def validate(self):
        if self.updates < 1 or self.differentiable_steps < 1:
            raise ValueError('updates and differentiable_steps must be positive')
        if self.step_size <= 0 or self.learning_rate <= 0:
            raise ValueError('step_size and learning_rate must be positive')
        if not self.warmup_steps or min(self.warmup_steps) < 0:
            raise ValueError('warmup_steps must be nonnegative and nonempty')
        if not (0 <= self.restart_corruption_probability <= 1):
            raise ValueError('restart_corruption_probability must be in [0,1]')
        if self.crop_size is not None and self.crop_size < 16:
            raise ValueError('crop_size must be >=16 or None')
        if self.logit_floor <= 0 or self.logit_floor >= 1/3:
            raise ValueError('logit_floor should be in (0, 1/3)')
        if self.eval_every < 1 or self.validation_examples < 1:
            raise ValueError('eval_every and validation_examples must be positive')
        if not self.validation_horizons or min(self.validation_horizons) < 0:
            raise ValueError('validation_horizons must be nonnegative and nonempty')
        for name in ('initial_blur_range', 'initial_spectral_range',
                     'restart_blur_range', 'restart_spectral_range'):
            a, b = getattr(self, name)
            if a < 0 or b < a:
                raise ValueError(f'Invalid {name}: {(a,b)}')
        if not self.learn_strengths and not self.learn_residuals:
            raise ValueError('Enable learn_strengths and/or learn_residuals')


def labels_from_batch(batch: Any, key: str = 'labels') -> torch.Tensor:
    """Accept a dictionary, a direct tensor, or (image, labels) tuple.

    Class index targets [B,H,W] / [H,W] and stained label channels
    [B,3,H,W] / [B,Z,3,H,W] are supported. For the latter, nuclei are
    channel 0, nonnuclear channel 1, and overlaps are ignored (-100).
    """
    if isinstance(batch, dict):
        if key not in batch:
            raise KeyError(f'Dataloader batch has keys {list(batch)}; missing {key!r}')
        labels = batch[key]
    elif isinstance(batch, (tuple, list)):
        if len(batch) < 2:
            raise ValueError('Tuple/list batches should be (image, labels)')
        labels = batch[1]
    else:
        labels = batch
    if not torch.is_tensor(labels):
        labels = torch.as_tensor(labels)
    if labels.ndim == 2:
        labels = labels[None]
    if labels.ndim == 3:
        return labels.long()  # [B,H,W]
    if labels.ndim in (4, 5):
        return target_from_stain_labels(labels)
    raise ValueError(f'Unrecognized label shape: {tuple(labels.shape)}')


def spatial_crop(y: torch.Tensor, size: int | None, rng: np.random.Generator,
                 *, random_crop: bool) -> torch.Tensor:
    if size is None:
        return y
    h, w = y.shape[-2:]
    if min(h, w) < size:
        raise ValueError(f'Image {h}x{w} smaller than crop_size={size}; lower crop_size or use None')
    top = int(rng.integers(h - size + 1)) if random_crop else (h-size)//2
    left = int(rng.integers(w - size + 1)) if random_crop else (w-size)//2
    return y[..., top:top+size, left:left+size]


def _uniform_range(rng: np.random.Generator, interval: tuple[float, float]) -> float:
    lo, hi = interval
    return float(rng.uniform(lo, hi)) if lo != hi else float(lo)


def corrupt_initial_target(target: torch.Tensor, cfg: Config,
                           tc: RealTrainConfig, rng: np.random.Generator) -> torch.Tensor:
    """One-hot clean labels -> Gaussian blur -> spectral-noisy logits -> softmax.

    Directly reuses the original synthetic_logits implementation, including
    its spectrum and class-channel jitter. Blurred input only; target stays crisp.
    """
    blur = _uniform_range(rng, tc.initial_blur_range)
    noise = _uniform_range(rng, tc.initial_spectral_range)
    source_cfg = replace(cfg, gt_blur_sigma=blur, perturb_strength=noise)
    # Explicitly use independent RNG seed per call, not the same perturbation.
    seed = int(rng.integers(0, 2**31-1))
    return synthetic_logits(target, source_cfg, seed=seed).softmax(dim=1)


def corrupt_warmup_state(p: torch.Tensor, cfg: Config,
                         tc: RealTrainConfig, rng: np.random.Generator) -> torch.Tensor:
    """Gaussian blur *of current warm-up probabilities*, then spectral noise.

    We use log(p) as logits so the operation is identity when blur/noise = 0,
    unlike reconstructing hard GT logits after the warm-up. Only this
    restart perturbation is detached; gradients traverse subsequent steps.
    """
    p = p.detach()
    if float(rng.random()) > tc.restart_corruption_probability:
        return p
    sigma = _uniform_range(rng, tc.restart_blur_range)
    strength = _uniform_range(rng, tc.restart_spectral_range)
    B, C, H, W = p.shape
    arr = p.cpu().numpy().astype(np.float32, copy=False)
    if sigma > 0:
        arr = gaussian_filter(arr, sigma=(0, 0, sigma, sigma), mode='reflect')
    arr = np.maximum(arr, tc.logit_floor)
    arr /= np.sum(arr, axis=1, keepdims=True)
    logits = np.log(arr)
    if strength > 0:
        amps = np.asarray(cfg.spectral_amplitudes, dtype=np.float64)
        waves = np.asarray(cfg.spectral_wavelengths, dtype=np.float64)
        for b in range(B):
            if rng.random() >= cfg.perturb_probability:
                continue
            for c in range(C):
                jitter = rng.uniform(*cfg.spectral_amplitude_jitter, size=amps.shape)
                bands = np.column_stack([amps*jitter, waves])
                noise = spectral_noise((H,W), bands, cfg.spectral_bdwidth, rng)
                # As in synthetic_logits: perturbation is proportional to logit span.
                logits[b,c] += strength * 2.0 * cfg.base_logit_magnitude * noise
    return torch.from_numpy(logits.copy()).to(p.device).softmax(dim=1)


def masked_weighted_mse(p: torch.Tensor, target: torch.Tensor,
                        tc: RealTrainConfig) -> torch.Tensor:
    """Ignore overlapping stain labels (-100) entirely in supervised loss."""
    valid = (target != -100)
    safe = target.clamp(min=0, max=2)
    clean = F.one_hot(safe.long(), 3).permute(0,3,1,2).to(p.dtype)
    weights = p.new_tensor([tc.background_loss_weight, tc.nuclear_loss_weight,
                            tc.nonnuclear_loss_weight]).view(1,3,1,1)
    errors = (p-clean).square() * weights * valid[:,None]
    return errors.sum() / (valid.sum().clamp_min(1) * 3)


def background_fp_loss(p: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    bg = (target == 0).to(p.dtype)
    score = p[:,1].square()+p[:,2].square()
    return (score * bg).sum()/bg.sum().clamp_min(1)


def class_dice(p: torch.Tensor, target: torch.Tensor, label: int) -> float:
    valid = target != -100
    pred = (p.argmax(dim=1) == label) & valid
    true = (target == label) & valid
    den = int(pred.sum()+true.sum())
    return float(2*(pred & true).sum()/den) if den else float('nan')


def load_validation_examples(loader: Iterable, cfg: Config, tc: RealTrainConfig):
    """Cache a small, deterministic validation set with fixed perturbations."""
    rng = np.random.default_rng(tc.seed+100000)
    pairs = []
    for bidx, batch in enumerate(loader):
        if tc.max_val_batches is not None and bidx >= tc.max_val_batches:
            break
        y = labels_from_batch(batch, tc.labels_key).cpu()
        for j in range(y.shape[0]):
            yy = spatial_crop(y[j:j+1], tc.crop_size, rng, random_crop=False)
            p0 = corrupt_initial_target(yy, cfg, tc, rng)
            pairs.append((p0, yy))
            if len(pairs) >= tc.validation_examples:
                return pairs
    if not pairs:
        raise ValueError('Validation loader returned no usable examples')
    return pairs


def _iterate(model: Energy, p: torch.Tensor, n: int, dt: float) -> torch.Tensor:
    for _ in range(n):
        p = projected_step(model, p, dt, differentiable=False)
    return p


def evaluate_real(model: Energy, validation_pairs, tc: RealTrainConfig,
                  horizons: tuple[int,...] | None = None) -> dict:
    """Test on fixed initial corruptions, with NO restart noise at validation."""
    horizons = tuple(sorted(set(tc.validation_horizons if horizons is None else horizons)))
    aggregates = {str(t): [] for t in horizons}
    for p0, target in validation_pairs:
        p = p0.to(tc.device)
        y = target.to(tc.device)
        last = 0
        for t in horizons:
            p = _iterate(model, p, t-last, tc.step_size)
            last = t
            with torch.no_grad():
                valid_bg = y == 0
                hard = p.argmax(dim=1)
                fp = (((hard != 0)&valid_bg).sum().float() /
                      valid_bg.sum().clamp_min(1)).item()
                aggregates[str(t)].append({
                    'mse': float(masked_weighted_mse(p, y, tc)),
                    'dice_nuclear': class_dice(p, y, 1),
                    'dice_nonnuclear': class_dice(p, y, 2),
                    'background_fp_fraction': fp,
                })
    # Ignore NaN Dice when a class is absent in a crop.
    return {key: {metric: float(np.nanmean([r[metric] for r in records]))
                  for metric in records[0]}
            for key, records in aggregates.items()}


def save_preview(model: Energy, validation_pairs, tc: RealTrainConfig,
                 path: Path):
    import matplotlib.pyplot as plt
    p0, target = validation_pairs[0]
    p = p0.to(tc.device)
    horizons = tuple(sorted(set((0, *tc.validation_horizons))))
    states = {}
    last = 0
    for t in horizons:
        p = _iterate(model, p, t-last, tc.step_size)
        last = t
        states[t] = p[0].detach().cpu()
    ncols = len(horizons)+1
    fig, axes = plt.subplots(3, ncols, figsize=(2.25*ncols, 7), squeeze=False)
    clean = target[0].clamp_min(0)
    for j, (name, v) in enumerate([('GT', F.one_hot(clean,3).permute(2,0,1).float())] +
                                  [(str(t),states[t]) for t in horizons]):
        for row, img in enumerate((v.argmax(0), v[1], v[2])):
            axes[row,j].imshow(img.numpy(), vmin=0,
                               vmax=2 if row==0 else 1,
                               cmap='viridis' if row==0 else 'magma',
                               interpolation='nearest')
            axes[row,j].axis('off')
        axes[0,j].set_title('GT' if name=='GT' else f'Step {name}')
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def train_real(train_dataloader: Iterable, val_dataloader: Iterable,
               energy_cfg: Config | None = None,
               train_cfg: RealTrainConfig | None = None):
    """Train on real stain labels with re-corruption at every training update.

    The validation set and its initial spectral perturbations are fixed.
    The best model is selected by held-out MSE at the largest horizon <= 300,
    or the latest specified horizon when all are > 300.
    """
    tc = RealTrainConfig() if train_cfg is None else train_cfg
    tc.validate()
    cfg = Config() if energy_cfg is None else energy_cfg
    cfg = replace(cfg, learn_strengths=tc.learn_strengths,
                  learn_residuals=tc.learn_residuals)
    cfg.validate()
    torch.manual_seed(tc.seed)
    rng = np.random.default_rng(tc.seed)
    device = torch.device(tc.device)
    output = Path(tc.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    model = Energy(cfg).to(device)
    parameters = [v for v in model.parameters() if v.requires_grad]
    if not parameters:
        raise ValueError('No trainable parameters enabled')
    optimizer = torch.optim.Adam(parameters, lr=tc.learning_rate)
    original_strengths = {name: model.strength(name).detach().clone()
                          for name in ('separator', 'boundary')}
    validation_pairs = load_validation_examples(val_dataloader, cfg, tc)
    select_horizon = max((t for t in tc.validation_horizons if t <= 300),
                         default=max(tc.validation_horizons))
    hist = []
    initial = evaluate_real(model, validation_pairs, tc)
    best_val = initial[str(select_horizon)]['mse']
    best_state = copy.deepcopy(model.state_dict())

    def save_checkpoint(step: int):
        torch.save({'model_state': best_state, 'config': asdict(cfg),
                    'training_config': asdict(tc), 'update': step,
                    'validation_mse': best_val}, output/'best_model.pt')

    save_checkpoint(0)
    save_preview(model, validation_pairs, tc, output/'initial_preview.png')
    print('Initial validation:', initial)
    train_iterator = iter(train_dataloader)
    for step in range(1, tc.updates+1):
        try:
            batch = next(train_iterator)
        except StopIteration:
            train_iterator = iter(train_dataloader)
            batch = next(train_iterator)
        y = labels_from_batch(batch, tc.labels_key).cpu()
        # Independently crop each sample and perturb anew on every use.
        ys = [spatial_crop(y[j:j+1], tc.crop_size, rng, random_crop=True)
              for j in range(y.shape[0])]
        y = torch.cat(ys).to(device)
        p = corrupt_initial_target(y, cfg, tc, rng)
        warmup = int(rng.choice(tc.warmup_steps))
        p = _iterate(model, p, warmup, tc.step_size).detach()
        # Critical: perturb the *warm-up state itself* immediately before
        # the 3 (default) differentiable steps. BLUR FIRST, THEN SPECTRAL.
        p = corrupt_warmup_state(p, cfg, tc, rng)
        for _ in range(tc.differentiable_steps):
            p = projected_step(model, p, tc.step_size, differentiable=True)
        reconstruction = masked_weighted_mse(p, y, tc)
        fp = background_fp_loss(p, y)
        drift = sum((model.strength(name)-original_strengths[name]).square()
                    for name in ('separator','boundary'))
        residual = sum((v.square().mean() for name, v in model.named_parameters()
                        if name in ('separator_residual','boundary_residual')),
                       reconstruction.new_zeros(()))
        stability = (masked_weighted_mse(projected_step(
            model,
            F.one_hot(y.clamp(0,2),3).permute(0,3,1,2).float(),
            tc.step_size, differentiable=True), y, tc)
            if tc.stationarity_weight else reconstruction.new_zeros(()))
        loss = (reconstruction + tc.false_positive_weight*fp +
                tc.strength_drift_weight*drift +
                tc.residual_penalty_weight*residual +
                tc.stationarity_weight*stability)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(parameters, tc.gradient_clip_norm)
        if not torch.isfinite(loss) or not torch.isfinite(grad_norm):
            raise FloatingPointError(f'Nonfinite loss / gradient at update {step}')
        optimizer.step()
        with torch.no_grad():
            for name in ('separator_weight_parameter','boundary_weight_parameter'):
                value = getattr(model, name, None)
                if value is not None:
                    value.clamp_(min=0)
        record = {'update': step, 'warmup': warmup,
                  'loss': float(loss.detach()),
                  'mse': float(reconstruction.detach()),
                  'background_fp_loss': float(fp.detach()),
                  'separator_strength': float(model.strength('separator').detach()),
                  'boundary_strength': float(model.strength('boundary').detach())}
        if step == 1 or step % tc.eval_every == 0 or step == tc.updates:
            val = evaluate_real(model, validation_pairs, tc)
            record['validation'] = val
            score = val[str(select_horizon)]['mse']
            if score < best_val:
                best_val = score
                best_state = copy.deepcopy(model.state_dict())
                save_checkpoint(step)
            print(f'Update {step:4d}/{tc.updates}  '
                  f'loss={record["loss"]:.6f} val@{select_horizon}={score:.6f} '
                  f'S={record["separator_strength"]:.4g} '
                  f'B={record["boundary_strength"]:.4g}')
            if tc.preview_every and step % tc.preview_every == 0:
                save_preview(model, validation_pairs, tc,
                             output/f'preview_{step:04d}.png')
        hist.append(record)
    # Restore best model, not merely final parameter iterate.
    model.load_state_dict(best_state)
    final_validation = evaluate_real(model, validation_pairs, tc)
    save_preview(model, validation_pairs, tc, output/'best_preview.png')
    with (output/'history.json').open('w') as f:
        json.dump({'initial': initial, 'best': final_validation,
                   'history': hist, 'config': asdict(cfg),
                   'training_config': asdict(tc)}, f, indent=2)
    print('Best validation:', final_validation)
    print('Saved to:', output.resolve())
    return model, hist


def load_real_checkpoint(path: str, device: str = 'cpu'):
    checkpoint = torch.load(path, map_location=device, weights_only=False)
    cfg = Config(**checkpoint['config'])
    model = Energy(cfg).to(device)
    model.load_state_dict(checkpoint['model_state'])
    model.eval()
    return model, cfg
