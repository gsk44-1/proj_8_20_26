"""Multi-perturbation energy-ranking training for ProbabilityHierarchicalEnergy.

Requires train_probability_energy.py and probability_hierarchical_energy.py
in the same directory. No unrolling or state Hessians during training.
Validation still measures 0..N projected energy-gradient correction steps.
"""
from __future__ import annotations

import math
import random
import time
from dataclasses import dataclass, replace, asdict
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from probability_hierarchical_energy import ProbabilityHierarchicalEnergy
from train_probability_energy import (
    TrainConfig, seed_all, flatten_volumes, _slice_iter,
    perturb_logits_stub, validate, _append_csv, _save_checkpoint,
)


@dataclass
class RankingConfig(TrainConfig):
    experiment: str = "prob_energy_multinoise_rank_v1"
    noise_realizations: int = 4
    amplitude_levels: tuple[float, ...] = (0., 0.5, 1., 1.5, 2.)
    ranking_margin_scale: float = 0.5
    clean_comparison_weight: float = 0.5
    # Use the sharp one-hot GT as an additional lowest-energy reference.
    include_clean_reference: bool = True
    perturb_probability: float = 1.0
    gt_blur_sigma: float = 0.0
    # This optimizer NEVER changes log_step: ranking trains E, not the step size.


def corruption_distance(p: torch.Tensor, target: torch.Tensor,
                        ignore_index: int) -> torch.Tensor:
    """Per-image mean sum_k (p_k - onehot_k)^2 over valid pixels: [B]."""
    valid = target.ne(ignore_index)
    safe = target.masked_fill(~valid, 0)
    onehot = F.one_hot(safe.long(), num_classes=3).permute(0, 3, 1, 2).to(p.dtype)
    sq = (p - onehot).square().sum(dim=1)
    return (sq * valid).sum(dim=(1, 2)) / valid.sum(dim=(1, 2)).clamp_min(1)


def make_ranked_states(target: torch.Tensor, cfg: RankingConfig):
    """Create [L,B,3,H,W] states on one shared-noise amplitude ray.

    Each invocation draws a new independent spectral noise realization.
    In that invocation, ALL amplitude levels reuse exactly the same noise.
    """
    # The existing generator builds synthetic logits from class labels.
    # At strength=1 it gives base_logits + noise_delta; at strength=0 base.
    # The clean baseline needs no FFT work and may optionally be GT-blurred.
    with torch.no_grad():
        logits_1 = perturb_logits_stub(target, replace(cfg, perturb_strength=1.0))
        logits_0 = perturb_logits_stub(target, replace(cfg, perturb_strength=0.0))
        delta = logits_1 - logits_0
        amplitudes = torch.tensor(cfg.amplitude_levels, device=target.device,
                                  dtype=logits_0.dtype)
        logits = logits_0.unsqueeze(0) + amplitudes[:, None, None, None, None] * delta.unsqueeze(0)
        levels, batch, classes, height, width = logits.shape
        probs = logits.reshape(levels * batch, classes, height, width).softmax(1)
        return probs.reshape(levels, batch, classes, height, width)


def ranking_loss_for_states(model, states, target, cfg: RankingConfig):
    """Per-image margin ranking by *measured* distortion, not nominal amplitude.

    - adjacent ordered corruptions, plus optionally clean-vs-all constraints
    - energies normalized per pixel
    - returns differentiable scalar loss and detached diagnostics
    """
    L, B, C, H, W = states.shape
    valid_images = target.ne(cfg.ignore_index).flatten(1).any(1)
    if not valid_images.any():
        return None

    d = torch.stack([
        corruption_distance(states[i], target, cfg.ignore_index)
        for i in range(L)
    ])  # [L,B], detached by construction

    # model.energy([B,3,H,W]) returns [B], NOT a single summed batch energy.
    e = model.energy(states.reshape(L * B, C, H, W)).reshape(L, B) / (H * W)

    # Sort each image independently, since larger nominal amplitude is not
    # guaranteed to produce higher measured distortion.
    order = d.argsort(dim=0)
    e_sorted = e.gather(0, order)
    d_sorted = d.gather(0, order)

    gaps = d_sorted[1:] - d_sorted[:-1]
    energy_gaps = e_sorted[1:] - e_sorted[:-1]
    adjacent = F.relu(cfg.ranking_margin_scale * gaps - energy_gaps)
    adj_per_image = adjacent.mean(dim=0)

    # GT has zero distortion; rank it below all spectral corruptions.
    # It is a separate reference because strength=0 yields softmax probabilities
    # rather than an exact one-hot state.
    if cfg.include_clean_reference:
        safe = target.masked_fill(target.eq(cfg.ignore_index), 0)
        clean = F.one_hot(safe.long(), 3).permute(0, 3, 1, 2).to(states.dtype)
        e_clean = model.energy(clean) / (H * W)  # [B]
        clean_violations = F.relu(cfg.ranking_margin_scale * d - (e - e_clean[None]))
        clean_per_image = clean_violations.mean(dim=0)
    else:
        clean_per_image = torch.zeros_like(adj_per_image)

    loss_per_image = adj_per_image + cfg.clean_comparison_weight * clean_per_image
    loss = loss_per_image[valid_images].mean()

    with torch.no_grad():
        correct = ((energy_gaps > 0) | (gaps < 1e-8)).float().mean()
        diagnostics = dict(
            adjacent=float(adj_per_image[valid_images].mean()),
            clean=float(clean_per_image[valid_images].mean()),
            ordered_fraction=float(correct),
            mean_distance=float(d[:, valid_images].mean()),
            mean_energy_gap=float(energy_gaps.mean()),
        )
    return loss, diagnostics


def train_ranking_batch(model, optimizer, target, cfg: RankingConfig):
    """Accumulate first-order gradients over independent noise realizations.

    Backward separately for each noise realization to limit activation memory.
    No `model.step` and no create_graph=True calls occur here.
    """
    model.train()
    optimizer.zero_grad(set_to_none=True)
    stats = []
    for _ in range(cfg.noise_realizations):
        states = make_ranked_states(target, cfg)
        result = ranking_loss_for_states(model, states, target, cfg)
        if result is None:
            continue
        loss, metrics = result
        if not torch.isfinite(loss):
            raise FloatingPointError("Nonfinite ranking loss")
        (loss / cfg.noise_realizations).backward()
        stats.append(dict(loss=float(loss.detach()), **metrics))

    if not stats:
        return None
    grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
    if not torch.isfinite(grad_norm):
        raise FloatingPointError("Nonfinite gradients")
    optimizer.step()
    result = {k: float(np.mean([s[k] for s in stats])) for k in stats[0]}
    result["grad_norm"] = float(grad_norm)
    return result


def train_ranking(loader_train, loader_val, cfg: RankingConfig | None = None,
                  *, device=None, model=None):
    cfg = RankingConfig() if cfg is None else cfg
    if cfg.noise_realizations < 1 or len(cfg.amplitude_levels) < 2:
        raise ValueError("Require >=1 independent noises and >=2 amplitude levels")
    if any(a < 0 for a in cfg.amplitude_levels) or cfg.ranking_margin_scale <= 0:
        raise ValueError("Amplitudes must be nonnegative, ranking margin positive")
    if cfg.clean_comparison_weight < 0:
        raise ValueError("Clean reference weight must be nonnegative")
    if cfg.perturb_probability != 1:
        raise ValueError("Set perturb_probability=1 so every ray contains noise")

    device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    seed_all(cfg.seed)
    output = Path(cfg.output_dir) / cfg.experiment
    output.mkdir(parents=True, exist_ok=True)
    checkpoint = output / "latest.pt"

    if model is None:
        model = ProbabilityHierarchicalEnergy(
            state_width=cfg.state_width, head_width=cfg.head_width,
            scale_weights=cfg.scale_weights, log_sigma=cfg.log_sigma,
            log_weight=cfg.log_weight, initial_step=cfg.initial_step,
        )
    model = model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.learning_rate,
                                  weight_decay=cfg.weight_decay)
    start_epoch, step = 0, 0
    if cfg.resume:
        ckpt = torch.load(checkpoint, map_location="cpu", weights_only=False)
        if ckpt.get("config") != asdict(cfg):
            raise ValueError("Checkpoint configuration differs from current configuration")
        model.load_state_dict(ckpt["model"])
        optimizer.load_state_dict(ckpt["optimizer"])
        for group in optimizer.state.values():
            for key, value in group.items():
                if torch.is_tensor(value):
                    group[key] = value.to(device)
        start_epoch, step = ckpt["epoch_next"], ckpt["global_step"]
        torch.set_rng_state(ckpt["torch_rng"])
        np.random.set_state(ckpt["numpy_rng"])
        random.setstate(ckpt["python_rng"])
        if device.type == "cuda" and "cuda_rng" in ckpt:
            torch.cuda.set_rng_state_all(ckpt["cuda_rng"])
    elif checkpoint.exists():
        raise FileExistsError(f"Checkpoint exists: {checkpoint}")

    print(f"Training ranking energy on {device}, {sum(p.numel() for p in model.parameters()):,} parameters")
    print(f"{cfg.noise_realizations} noise realizations x {len(cfg.amplitude_levels)} amplitude levels")
    print("No gradient unrolling; the log_step parameter is not learned by ranking")
    for epoch in range(start_epoch, cfg.epochs):
        t0 = time.perf_counter()
        losses = []
        for vi, (images, labels) in enumerate(loader_train):
            images, labels = flatten_volumes(images, labels, shuffle=True)
            for target, _sup in _slice_iter(images, labels, cfg, device):
                result = train_ranking_batch(model, optimizer, target, cfg)
                if result is None:
                    continue
                step += 1
                losses.append(result["loss"])
                _append_csv(output / "training_ranking.csv",
                            dict(step=step, epoch=epoch + 1, volume_idx=vi + 1, **result))
                if cfg.print_every and step % cfg.print_every == 0:
                    print(f"Epoch {epoch+1} step {step} | rank loss {result['loss']:.5f} "
                          f"| ordering {result['ordered_fraction']:.1%} "
                          f"| d {result['mean_distance']:.3f} "
                          f"| energy gap {result['mean_energy_gap']:.4g}")
                if cfg.validate_every and step % cfg.validate_every == 0:
                    # Same diagnostic as the previous trainer: perform gradient
                    # descent on the learned energy, even though trained by ranking.
                    for row in validate(model, loader_val, cfg, device):
                        _append_csv(output / "validation.csv",
                                    dict(step=step, epoch=epoch+1, **row))
                        print(f"Validation depth {row['depth']:2d} "
                              f"| nuc {row['dice_nuc']:.3f} "
                              f"| non {row['dice_nonnuc']:.3f} "
                              f"| E/pixel {row['energy_per_pixel']:.4f}")
                if cfg.checkpoint_every and step % cfg.checkpoint_every == 0:
                    # Restart of current epoch may repeat its beginning.
                    _save_checkpoint(checkpoint, model, optimizer, cfg, epoch, step)
        _save_checkpoint(checkpoint, model, optimizer, cfg, epoch+1, step)
        print(f"Epoch {epoch+1} complete | mean rank loss "
              f"{np.mean(losses) if losses else float('nan'):.5f} "
              f"| {time.perf_counter()-t0:.1f}s")
    return model


if __name__ == "__main__":
    print("Import RankingConfig and train_ranking and supply your DataLoaders.")
