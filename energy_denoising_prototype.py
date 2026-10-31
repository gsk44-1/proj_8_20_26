"""One-step denoising supervision for a learned segmentation energy.

Inputs: image [B,2,H,W], integer segmentation targets [B,H,W] with
0=background, 1=nucleus, 2=nonnucleus, -100=ignored.

The *only* trainable model is SegmentationEnergyModel. RandomEnergyPerturber
is a frozen, randomly initialized energy CNN with the same layer pattern.
Its negative energy gradient creates perturbations resembling untrained
energy-descent updates. No auxiliary state channels are used.

Example:
    model = SegmentationEnergyModel(width=64).to(device)
    perturb = RandomEnergyPerturber(width=32).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-4)
    for images, targets in train_loader:
        info = train_one_batch(model, perturb, optimizer,
                               images.to(device), targets.to(device))
        print(info)
    model.eval()
    logits = model.infer(images.to(device), steps=20)
"""

import math
import random
from dataclasses import dataclass
from typing import Optional

import torch
from torch import nn
from torch.nn import functional as F


def make_energy_cnn(image_channels=2, state_channels=3, width=64):
    """Original-style energy CNN; dilation (1,3,7), scalar output density."""
    return nn.Sequential(
        nn.Conv2d(image_channels + state_channels, width, 3, padding=1),
        nn.SiLU(),
        nn.AvgPool2d(kernel_size=2, stride=2),
        nn.Conv2d(width, width, 3, padding=3),
        nn.SiLU(),
        nn.AvgPool2d(kernel_size=2, stride=2),
        nn.Conv2d(width, width, 3, padding=7, dilation=7),
        nn.SiLU(),
        nn.Conv2d(width, 1, 1),
    )


class SegmentationEnergyModel(nn.Module):
    """Energy on just three segmentation logits, trained via one-step updates."""
    def __init__(self, image_channels=2, classes=3, width=64,
                 initial_step=0.3, state_l2=1e-3):
        super().__init__()
        self.image_channels = image_channels
        self.classes = classes
        self.state_l2 = state_l2
        self.net = make_energy_cnn(image_channels, classes, width)
        self.log_step = nn.Parameter(torch.tensor(math.log(initial_step)))

    def energy(self, image, state):
        # Sum over spatial positions. Taking mean here would make the state
        # gradient vanish with image area unless step size is rescaled.
        density = self.net(torch.cat([image, state], dim=1))
        return density.sum() + 0.5 * self.state_l2 * state.square().sum()

    def state_gradient(self, image, state, *, create_graph):
        # Needed even inside an outer torch.no_grad() at inference time.
        with torch.enable_grad():
            if not state.requires_grad:
                state = state.detach().requires_grad_(True)
            energy = self.energy(image, state)
            grad, = torch.autograd.grad(energy, state, create_graph=create_graph)
        return grad

    def update(self, image, state, *, differentiable):
        grad = self.state_gradient(image, state, create_graph=differentiable)
        return state - self.log_step.exp() * grad

    def infer(self, image, *, steps=20, state=None):
        """Euler descent with constant memory; works under torch.no_grad()."""
        if state is None:
            state = image.new_zeros(image.shape[0], self.classes, *image.shape[-2:])
        else:
            state = state.detach()
        for _ in range(steps):
            state = self.update(image, state, differentiable=False).detach()
        return state


class RandomEnergyPerturber(nn.Module):
    """Frozen random energy whose gradient generates structured perturbations.

    Shares the same *convolution/dilation pattern* as the learned energy,
    but is smaller by default. The resulting perturbation is a potential
    gradient, not merely the output of an arbitrary random CNN.
    """
    def __init__(self, image_channels=2, classes=3, width=32):
        super().__init__()
        self.net = make_energy_cnn(image_channels, classes, width)
        self.reset_random_weights()

    def reset_random_weights(self):
        for module in self.net.modules():
            if hasattr(module, 'reset_parameters'):
                module.reset_parameters()
        for param in self.parameters():
            param.requires_grad_(False)

    def forward(self, image, state):
        with torch.enable_grad():
            reference = state.detach().requires_grad_(True)
            density = self.net(torch.cat([image.detach(), reference], dim=1))
            grad, = torch.autograd.grad(density.sum(), reference)
        # Negative gradient = one update from a randomly initialized energy.
        perturbation = -grad.detach()
        # Normalize each example to a controlled RMS, then apply sigma.
        rms = perturbation.square().mean(dim=(1, 2, 3), keepdim=True).sqrt()
        return perturbation / rms.clamp_min(1e-6)


@dataclass
class DenoisingConfig:
    gamma: float = 0.30            # fraction of error removed by desired step
    target_logit: float = 3.0      # true class 2, false classes -1
    max_rollout_steps: int = 12
    # Probabilities correspond to zero, truth, interpolation, model trajectory.
    source_probs: tuple = (0.25, 0.25, 0.25, 0.25)
    perturb_probability: float = 0.75
    random_energy_probability: float = 0.70  # else Gaussian perturbation
    sigma_min: float = 0.10
    sigma_max: float = 0.80
    ce_weight: float = 0.10
    ignore_index: int = -100


def make_target_logits(targets, classes=3, amplitude=3.0, ignore_index=-100):
    """Mean-centered finite target logits; no infinite one-hot logits."""
    safe = targets.masked_fill(targets == ignore_index, 0)
    one_hot = F.one_hot(safe.long(), num_classes=classes).permute(0, 3, 1, 2)
    logits = amplitude * (one_hot.to(torch.float32) - 1.0 / classes)
    return logits


def choose_training_state(model, image, target_logits, cfg):
    """Return a state whose history is detached (one-step training only)."""
    source = random.choices(range(4), weights=cfg.source_probs, k=1)[0]
    if source == 0:
        return torch.zeros_like(target_logits), 'zero'
    if source == 1:
        return target_logits.detach().clone(), 'target'
    if source == 2:
        t = torch.rand(image.shape[0], 1, 1, 1, device=image.device)
        return ((1.0 - t) * target_logits).detach(), 'interpolation'
    k = random.randint(1, cfg.max_rollout_steps)
    # No graph retained through this rollout, despite gradient-based inference.
    return model.infer(image, steps=k).detach(), f'rollout_{k}'


def train_one_batch(model, perturber, optimizer, image, targets,
                    cfg: Optional[DenoisingConfig] = None):
    """Exactly ONE differentiated energy-gradient step and one optimizer step."""
    cfg = cfg or DenoisingConfig()
    model.train()
    if not (targets != cfg.ignore_index).any().item():
        return None
    target_logits = make_target_logits(
        targets, classes=model.classes, amplitude=cfg.target_logit,
        ignore_index=cfg.ignore_index).to(device=image.device, dtype=image.dtype)
    state, source_name = choose_training_state(model, image, target_logits, cfg)

    if random.random() < cfg.perturb_probability:
        sigma = random.uniform(cfg.sigma_min, cfg.sigma_max)
        if random.random() < cfg.random_energy_probability:
            noise = perturber(image, state)
            perturb_kind = 'random_energy_gradient'
        else:
            noise = torch.randn_like(state)
            noise = noise / noise.square().mean(dim=(1, 2, 3), keepdim=True).sqrt().clamp_min(1e-6)
            perturb_kind = 'gaussian'
        state = (state + sigma * noise).detach()
    else:
        perturb_kind = 'none'

    desired = ((1.0 - cfg.gamma) * state + cfg.gamma * target_logits).detach()
    optimizer.zero_grad(set_to_none=True)
    updated = model.update(image, state, differentiable=True)

    # Mask ignored pixels; use per-element MSE normalized by valid pixels.
    valid = (targets != cfg.ignore_index).unsqueeze(1)
    correction_loss = ((updated - desired).square() * valid).sum()
    correction_loss = correction_loss / (valid.sum() * model.classes).clamp_min(1)
    ce = F.cross_entropy(updated, targets, ignore_index=cfg.ignore_index)
    loss = correction_loss + cfg.ce_weight * ce
    loss.backward()
    optimizer.step()
    return dict(loss=float(loss.detach()), correction_mse=float(correction_loss.detach()),
                ce=float(ce.detach()), source=source_name, perturb=perturb_kind,
                step_size=float(model.log_step.detach().exp()))


if __name__ == '__main__':
    torch.manual_seed(4)
    random.seed(4)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = SegmentationEnergyModel(width=8).to(device)
    perturber = RandomEnergyPerturber(width=8).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
    image = torch.randn(2, 2, 16, 16, device=device)
    targets = torch.randint(0, 3, (2, 16, 16), device=device)
    for source in range(4):
        cfg = DenoisingConfig(source_probs=tuple(float(i == source) for i in range(4)),
                              max_rollout_steps=3)
        info = train_one_batch(model, perturber, opt, image, targets, cfg)
        assert info is not None
        assert math.isfinite(info['loss'])
        print(info)
    with torch.no_grad():
        out = model.eval().infer(image, steps=4)
    assert out.shape == (2, 3, 16, 16)
    assert all(p.grad is None for p in perturber.parameters())
    print('Smoke test passed', out.shape)
