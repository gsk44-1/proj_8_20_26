
"""Train the image-free three-class probability energy corrector.

Requires probability_hierarchical_energy.py in the same directory, and existing
`loader_train` / `loader_val` DataLoaders in a notebook, OR pass loaders to train().

Dataset convention inherited from the previous trainer:
  images: [B,Z,2,H,W] (ignored by this energy model)
  labels: [B,Z,3,H,W]: nuclear mask, nonnuclear mask, supervision boost.
The input to the corrector is synthetic probabilities generated from perturbed
three-class ground-truth logits using independent spectral-band noise.
Ground-truth blurring is optional and disabled by default. No GT or image is passed to the
energy itself; GT is used only to generate a synthetic starting state and loss.

Training uses a random detached burn-in plus TRACKED_STEPS differentiable
projected-gradient steps with shared energy parameters. The well substep is OFF.
Model.step() internally enables autograd even during detached rollouts; do not
wrap model calls inside torch.inference_mode().

Notebook:
    from train_probability_energy import TrainConfig, train
    model = train(loader_train, loader_val, TrainConfig())

Script: supply your DataLoaders in your own runner and call train(...).
"""
from __future__ import annotations

import csv
import math
import os
import random
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from scipy import fft
from scipy.ndimage import gaussian_filter

from probability_hierarchical_energy import ProbabilityHierarchicalEnergy


@dataclass
class TrainConfig:
    experiment: str = "prob_energy_noimage_v1"
    output_dir: str = "./prob_energy_runs"
    epochs: int = 5
    seed: int = 4
    slice_batch_size: int = 4
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    grad_clip: float = 7.0
    state_width: int = 32
    head_width: int = 24
    scale_weights: tuple[float, float, float] = (1.0, 0.5, 0.25)
    log_sigma: float = 5.0
    log_weight: float = 0.01
    initial_step: float = 0.1
    # Distribution over total rollout lengths: 3 tracked steps + 0..21 detached.
    tracked_steps: int = 3
    max_burnin_units: int = 7
    unroll_loss_weights: tuple[float, ...] = (0.0, 0.0, 1.0)
    # Synthetic logit initialization (class k: +magnitude, others: -magnitude).
    # Fixed span = 2*magnitude, so spectral noise is multiplied by this span.
    # With magnitude=1.5, clean classes have probabilities ~0.91 / 0.045 / 0.045.
    base_logit_magnitude: float = 1.5
    perturb_strength: float = 1.0
    perturb_probability: float = 1.0
    spectral_bdwidth: float = 0.9
    spectral_wavelengths: tuple[float, ...] = (250., 150., 100., 50., 25., 12.5, 1.)
    spectral_amplitudes: tuple[float, ...] = (15., 12., 8., 2., 1., 0.5, 0.3)
    spectral_amplitude_jitter: tuple[float, float] = (0.6, 1.5)
    # 0 = no Gaussian blur of the ground truth; >0 smooths class boundaries.
    gt_blur_sigma: float = 0.0
    # Supervision and task loss
    class_weights: tuple[float, float, float] = (1.0, 2.0, 1.0)
    dice_class_weights: tuple[float, float, float] = (0.2, 0.4, 0.4)
    ce_weight: float = 1.0
    dice_weight: float = 0.5
    supervision_boost: float = 2.0
    ignore_index: int = -100
    eps: float = 1e-7
    # Triple well intentionally excluded from training.
    use_triple_well: bool = False
    validation_depths: tuple[int, ...] = (0, 1, 4, 8, 12, 20, 40)
    validation_max_volume_batches: int = 2
    print_every: int = 5
    validate_every: int = 200
    checkpoint_every: int = 200
    resume: bool = False


def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def flatten_volumes(images, labels, *, shuffle=False):
    """Accept original [B,Z,C,H,W] or pre-sliced [B,C,H,W]."""
    if images.ndim == 5:
        b, z, c, h, w = images.shape
        images = images.reshape(b * z, c, h, w)
        labels = labels.reshape(b * z, *labels.shape[2:])
    elif images.ndim != 4 or labels.ndim != 4:
        raise ValueError("Expected volume [B,Z,C,H,W] or slice [B,C,H,W]")
    if shuffle:
        perm = torch.randperm(len(images))
        images, labels = images[perm], labels[perm]
    return images, labels


def targets_from_labels(labels, ignore_index=-100):
    """[B,3,H,W] = nuclear, nonnuclear, weight -> [B,H,W] labels."""
    nuc = labels[:, 0] > 0.5
    non = labels[:, 1] > 0.5
    target = torch.zeros_like(labels[:, 0], dtype=torch.long)
    target[nuc & ~non] = 1
    target[non & ~nuc] = 2
    target[nuc & non] = ignore_index
    return target, labels[:, 2].float().clamp_min(0)


def spectral_noise(shape, bands, bdwidth=0.9, rng=None):
    """2D spectral-band random field, centered and max-absolute normalized.

    Each amplitude-wavelength pair defines a Gaussian annulus in frequency
    space; this is a finite number of *bands*, not discrete Fourier modes.
    Frequencies are in cycles/pixel; wavelengths are measured in pixels.
    """
    rng = np.random.default_rng() if rng is None else rng
    h, w = shape
    white = rng.standard_normal((h, w))
    ky = fft.fftfreq(h)[:, None]
    kx = fft.rfftfreq(w)[None, :]
    k = np.hypot(kx, ky)
    filt = np.zeros_like(k)
    for amplitude, wavelength in bands:
        filt += amplitude * np.exp(
            -0.5 * ((k - 1.0 / wavelength) / (bdwidth / wavelength)) ** 2
        )
    noise = fft.irfftn(fft.rfftn(white) * filt, s=(h, w)).real
    noise -= noise.mean()
    noise /= max(float(np.max(np.abs(noise))), 1e-8)
    return noise.astype(np.float32)


def random_bands(rng, cfg: TrainConfig):
    """One independently jittered amplitude per predefined wavelength."""
    wavelengths = np.asarray(cfg.spectral_wavelengths, dtype=np.float64)
    amplitudes = np.asarray(cfg.spectral_amplitudes, dtype=np.float64)
    amplitudes = amplitudes * rng.uniform(
        cfg.spectral_amplitude_jitter[0],
        cfg.spectral_amplitude_jitter[1],
        size=len(amplitudes),
    )
    return np.column_stack((amplitudes, wavelengths))


@torch.no_grad()
def perturb_logits_stub(target, cfg: TrainConfig, *, generator=None):
    """Construct synthetic [B,3,H,W] logits from ground-truth class IDs.

    The historical name is preserved for notebook compatibility; this now
    implements the spectral perturbation, not the old Gaussian-noise stub.
    Every image and class gets independent bands and random Fourier phases.

    Baseline logits = -magnitude + 2*magnitude * one_hot.
    Perturbation = strength * (2*magnitude) * spectral_noise.
    When gt_blur_sigma > 0, optionally blur each spatial class mask first.
    Input logits are *never* passed to the energy model; only their softmax is.

    Note: this reference implementation uses SciPy/NumPy on CPU and moves the
    resulting tensor to the target device. It is suitable as a starting point
    and can be replaced with a GPU FFT implementation if data loading is slow.
    """
    if target.ndim != 3:
        raise ValueError("Expected target [B,H,W]")
    b, h, w = target.shape
    # An independently sampled seed for each batch, using the main torch RNG
    # (or a user-provided CPU torch.Generator). Deterministic when seed_all()
    # is called; stable validation seeds are handled by validate().
    rng = np.random.default_rng(
        int(torch.randint(0, 2**31 - 1, (1,), generator=generator).item())
    )
    safe = target.clamp(0, 2).detach().cpu().numpy().astype(np.int64)
    one_hot = np.eye(3, dtype=np.float32)[safe].transpose(0, 3, 1, 2)
    if cfg.gt_blur_sigma > 0:
        one_hot = gaussian_filter(
            one_hot, sigma=(0, 0, cfg.gt_blur_sigma, cfg.gt_blur_sigma),
            mode="reflect",
        )
    span = 2.0 * cfg.base_logit_magnitude
    logits = -cfg.base_logit_magnitude + span * one_hot

    if cfg.perturb_strength > 0 and cfg.perturb_probability > 0:
        for i in range(b):
            if rng.random() >= cfg.perturb_probability:
                continue
            for cls in range(3):
                bands = random_bands(rng, cfg)
                noise = spectral_noise((h, w), bands, cfg.spectral_bdwidth, rng)
                logits[i, cls] += cfg.perturb_strength * span * noise

    return torch.from_numpy(logits).to(device=target.device, dtype=torch.float32)


def make_initial_state(target, cfg, generator=None):
    return torch.softmax(perturb_logits_stub(target, cfg, generator=generator), dim=1)


def task_parts(prob, target, supervision, cfg):
    """Probability-space weighted CE + soft Dice; returns scalar & diagnostics.

    IMPORTANT: model outputs PROBABILITIES, not logits. Do not softmax again.
    Clamp only inside log(), preserving gradients of ordinary p where positive.
    """
    valid = target != cfg.ignore_index
    safe = target.masked_fill(~valid, 0)
    weights = prob.new_tensor(cfg.class_weights)
    sw = (1.0 + cfg.supervision_boost * supervision) * valid
    logp = prob.clamp_min(cfg.eps).log()
    ce_map = F.nll_loss(logp, safe, weight=weights, reduction="none")
    ce_denom = (sw * weights[safe]).sum((1, 2)).clamp_min(1e-8)
    ce = ((ce_map * sw).sum((1, 2)) / ce_denom)

    valid_f = valid.unsqueeze(1)
    onehot = F.one_hot(safe, 3).permute(0, 3, 1, 2).to(prob.dtype)
    pp, yy = prob * valid_f, onehot * valid_f
    intersection = (pp * yy).sum((2, 3))
    denominator = pp.sum((2, 3)) + yy.sum((2, 3))
    dice_per_class = (2 * intersection + 1e-6) / (denominator + 1e-6)
    dw = prob.new_tensor(cfg.dice_class_weights)
    dw = dw / dw.sum()
    dice_loss = 1.0 - (dice_per_class * dw).sum(dim=1)
    mask = valid.flatten(1).any(dim=1)
    if not mask.any():
        return None
    ce_mean = ce[mask].mean()
    dice_mean = dice_loss[mask].mean()
    return (cfg.ce_weight * ce_mean + cfg.dice_weight * dice_mean,
            ce_mean, dice_mean)


@torch.no_grad()
def hard_dice(prob, target, ignore_index=-100):
    valid = target != ignore_index
    pred = prob.argmax(dim=1)
    result = []
    for k in range(3):
        p = (pred == k) & valid
        y = (target == k) & valid
        denom = p.sum() + y.sum()
        result.append(float((2 * (p & y).sum() / denom).item()) if denom > 0 else float("nan"))
    return result


def run_tracked(model, start, target, supervision, cfg):
    """Exactly cfg.tracked_steps connected energy-gradient updates."""
    p = start.detach()
    losses, states = [], []
    for _ in range(cfg.tracked_steps):
        p = model.step(p, differentiable=True, use_triple_well=False)
        parts = task_parts(p, target, supervision, cfg)
        if parts is None:
            return None
        states.append(p)
        losses.append(parts)
    loss = sum(w * parts[0] for w, parts in zip(cfg.unroll_loss_weights, losses))
    return p, loss, states, losses


def train_batch(model, optimizer, target, supervision, cfg):
    """Random burn-in, optionally sampled from current autonomous dynamics."""
    model.train()
    p0 = make_initial_state(target, cfg).detach()
    units = random.randint(0, cfg.max_burnin_units)
    burnin = units * cfg.tracked_steps
    p = p0
    for _ in range(burnin):
        # Model internally re-enables grad w.r.t. p; result is detached.
        p = model.step(p, differentiable=False, use_triple_well=False)
    starting = p.detach()
    with torch.no_grad():
        baseline = task_parts(starting, target, supervision, cfg)
    tracked = run_tracked(model, starting, target, supervision, cfg)
    if tracked is None:
        return None
    final, loss, _, parts = tracked
    if not bool(torch.isfinite(loss)):
        raise FloatingPointError("Nonfinite training task loss")
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    grad_norm = nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
    if not bool(torch.isfinite(grad_norm)):
        raise FloatingPointError("Nonfinite parameter gradients")
    optimizer.step()
    with torch.no_grad():
        score = task_parts(final.detach(), target, supervision, cfg)
        d = hard_dice(final.detach(), target, cfg.ignore_index)
        step_rms = (final.detach() - starting).square().mean().sqrt().item()
    return {
        "loss": float(loss.detach()),
        "ce": float(score[1]), "soft_dice_loss": float(score[2]),
        "before_loss": float(baseline[0]) if baseline else float("nan"),
        "after_loss": float(score[0]),
        "step_rms": step_rms,
        "grad_norm": float(grad_norm),
        "burnin_steps": burnin,
        "tracked_steps": cfg.tracked_steps,
        "step_size": float(model.log_step.detach().exp()),
        "dice_bg": d[0], "dice_nuc": d[1], "dice_nonnuc": d[2],
        **{f"loss_step_{i+1}": float(a[0].detach()) for i, a in enumerate(parts)},
    }


def _append_csv(path, row):
    path.parent.mkdir(parents=True, exist_ok=True)
    exists = path.exists() and path.stat().st_size > 0
    with path.open("a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(row), extrasaction="ignore")
        if not exists:
            writer.writeheader()
        writer.writerow(row)


def _slice_iter(images, labels, cfg, device):
    for pos in range(0, len(images), cfg.slice_batch_size):
        # We don't need the image for this model; labels provide training GT.
        lb = labels[pos:pos+cfg.slice_batch_size].to(device, dtype=torch.float32)
        target, sup = targets_from_labels(lb, cfg.ignore_index)
        if (target != cfg.ignore_index).any():
            yield target, sup


def validate(model, loader_val, cfg, device):
    """Fixed-seed spectral perturbations, independent of training RNG.

    Validation uses synthetic perturbed GT, not real first-model logits.
    """
    model.eval()
    depths = sorted(set(cfg.validation_depths))
    if not depths:
        return []
    store = {d: {"n": 0, "loss": 0.0, "ce": 0.0,
                 "dice_bg": [], "dice_nuc": [], "dice_nonnuc": [],
                 "energy": 0.0} for d in depths}
    # Fork torch RNG; NumPy fields are seeded from that torch RNG.
    with torch.random.fork_rng(devices=list(range(torch.cuda.device_count())) if device.type == "cuda" else []):
        torch.manual_seed(cfg.seed + 100000)
        if device.type == "cuda":
            torch.cuda.manual_seed_all(cfg.seed + 100000)
        for vi, (images, labels) in enumerate(loader_val):
            if cfg.validation_max_volume_batches and vi >= cfg.validation_max_volume_batches:
                break
            images, labels = flatten_volumes(images, labels, shuffle=False)
            for target, sup in _slice_iter(images, labels, cfg, device):
                p = make_initial_state(target, cfg).detach()
                for d in range(depths[-1] + 1):
                    if d > 0:
                        p = model.step(p, differentiable=False, use_triple_well=False)
                    if d not in store:
                        continue
                    with torch.no_grad():
                        parts = task_parts(p, target, sup, cfg)
                        if parts is None:
                            continue
                        dice = hard_dice(p, target, cfg.ignore_index)
                        e = float(model.energy(p).mean()) / (p.shape[-2] * p.shape[-1])
                    s = store[d]
                    s["n"] += 1
                    s["loss"] += float(parts[0]); s["ce"] += float(parts[1])
                    s["energy"] += e
                    for k, name in enumerate(("dice_bg", "dice_nuc", "dice_nonnuc")):
                        if math.isfinite(dice[k]):
                            s[name].append(dice[k])
    rows = []
    for depth, s in store.items():
        if not s["n"]:
            continue
        rows.append({"depth": depth,
                     "task": s["loss"]/s["n"], "ce": s["ce"]/s["n"],
                     "energy_per_pixel": s["energy"]/s["n"],
                     **{key: (float(np.mean(s[key])) if s[key] else float("nan"))
                        for key in ("dice_bg", "dice_nuc", "dice_nonnuc")}})
    return rows


def _save_checkpoint(path, model, optimizer, cfg, epoch_next, step):
    ckpt = {
        "model": model.state_dict(), "optimizer": optimizer.state_dict(),
        "config": asdict(cfg), "epoch_next": epoch_next, "global_step": step,
        "torch_rng": torch.get_rng_state(), "numpy_rng": np.random.get_state(),
        "python_rng": random.getstate(),
    }
    if torch.cuda.is_available():
        ckpt["cuda_rng"] = torch.cuda.get_rng_state_all()
    temp = str(path) + ".tmp"
    torch.save(ckpt, temp)
    os.replace(temp, path)


def train(loader_train, loader_val, cfg: TrainConfig | None = None,
          *, device=None, model=None):
    cfg = TrainConfig() if cfg is None else cfg
    if cfg.use_triple_well:
        raise ValueError("Triple-well training disabled by design; set use_triple_well=False")
    if cfg.tracked_steps < 1 or len(cfg.unroll_loss_weights) != cfg.tracked_steps:
        raise ValueError("Unroll weight tuple must match positive tracked_steps")
    if abs(sum(cfg.unroll_loss_weights)-1) > 1e-5:
        raise ValueError("Unroll loss weights should sum to 1")
    if cfg.max_burnin_units < 0:
        raise ValueError("max_burnin_units must be >= 0")
    if (cfg.base_logit_magnitude <= 0 or cfg.perturb_strength < 0
            or not 0 <= cfg.perturb_probability <= 1
            or cfg.gt_blur_sigma < 0 or cfg.spectral_bdwidth <= 0):
        raise ValueError("Invalid input perturbation settings")
    if (not cfg.spectral_wavelengths
            or len(cfg.spectral_wavelengths) != len(cfg.spectral_amplitudes)
            or any(w <= 0 for w in cfg.spectral_wavelengths)
            or any(a < 0 for a in cfg.spectral_amplitudes)
            or not (0 <= cfg.spectral_amplitude_jitter[0]
                    <= cfg.spectral_amplitude_jitter[1])):
        raise ValueError("Invalid spectral bands or amplitude jitter")
    device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    seed_all(cfg.seed)
    out = Path(cfg.output_dir) / cfg.experiment
    out.mkdir(parents=True, exist_ok=True)
    ckpt_path = out / "latest.pt"
    model = model or ProbabilityHierarchicalEnergy(
        state_width=cfg.state_width, head_width=cfg.head_width,
        scale_weights=cfg.scale_weights, log_sigma=cfg.log_sigma,
        log_weight=cfg.log_weight, initial_step=cfg.initial_step)
    model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.learning_rate,
                                  weight_decay=cfg.weight_decay)
    epoch_start = 0
    global_step = 0
    if cfg.resume:
        saved = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        model.load_state_dict(saved["model"])
        optimizer.load_state_dict(saved["optimizer"])
        for state in optimizer.state.values():
            for key, val in state.items():
                if torch.is_tensor(val):
                    state[key] = val.to(device)
        epoch_start, global_step = saved["epoch_next"], saved["global_step"]
        torch.set_rng_state(saved["torch_rng"])
        np.random.set_state(saved["numpy_rng"])
        random.setstate(saved["python_rng"])
        if torch.cuda.is_available() and "cuda_rng" in saved:
            torch.cuda.set_rng_state_all(saved["cuda_rng"])
    elif ckpt_path.exists():
        raise FileExistsError(f"Existing checkpoint at {ckpt_path}; resume or change experiment name")

    print(f"Device: {device}; params: {sum(p.numel() for p in model.parameters()):,}")
    print("State: probabilities; image-free energy; triple well OFF")
    print(f"Input: spectral-band 3-logit perturbations -> softmax; "
          f"strength={cfg.perturb_strength}, GT blur={cfg.gt_blur_sigma}")
    print(f"Tracked steps: {cfg.tracked_steps}; burn-in units: 0..{cfg.max_burnin_units}")
    for epoch in range(epoch_start, cfg.epochs):
        t0 = time.perf_counter()
        losses = []
        for vi, (images, labels) in enumerate(loader_train):
            images, labels = flatten_volumes(images, labels, shuffle=True)
            for target, sup in _slice_iter(images, labels, cfg, device):
                row = train_batch(model, optimizer, target, sup, cfg)
                if row is None:
                    continue
                global_step += 1
                losses.append(row["loss"])
                logged = {"global_step": global_step, "epoch": epoch+1,
                          "volume_idx": vi+1, **row}
                _append_csv(out/"training.csv", logged)
                if cfg.print_every and global_step % cfg.print_every == 0:
                    print(f"Epoch {epoch+1} step {global_step} | loss {row['loss']:.4f} "
                          f"| before {row['before_loss']:.4f} -> {row['after_loss']:.4f} "
                          f"| nuc {row['dice_nuc']:.3f} non {row['dice_nonnuc']:.3f} "
                          f"| burnin {row['burnin_steps']} | eta {row['step_size']:.4g}")
                if cfg.validate_every and global_step % cfg.validate_every == 0:
                    for val in validate(model, loader_val, cfg, device):
                        _append_csv(out/"validation.csv", {
                            "global_step": global_step, "epoch": epoch+1, **val})
                        print(f"Validation depth {val['depth']:2d} | task {val['task']:.4f} "
                              f"| nuc {val['dice_nuc']:.3f} non {val['dice_nonnuc']:.3f} "
                              f"| E/pixel {val['energy_per_pixel']:.4g}")
                # Mid-epoch checkpoints restart the current epoch from its beginning;
                # full exact mid-volume recovery can be added if needed.
                if cfg.checkpoint_every and global_step % cfg.checkpoint_every == 0:
                    _save_checkpoint(ckpt_path, model, optimizer, cfg, epoch, global_step)
        _save_checkpoint(ckpt_path, model, optimizer, cfg, epoch+1, global_step)
        print(f"Epoch {epoch+1} finished | mean loss {np.mean(losses) if losses else float('nan'):.4f} "
              f"| {time.perf_counter()-t0:.1f}s")
    return model


if __name__ == "__main__":
    print("Import this module and call train(loader_train, loader_val, TrainConfig()).")
