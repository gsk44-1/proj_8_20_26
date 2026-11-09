
"""Three-phase Modica--Mortola energy with two oriented filter families.

Filter families (each acts on all THREE probability channels):
  separator: nuclear | nonnuclear | nuclear, 180-degree rotations
  boundary:  background | nonnuclear | nuclear, 360-degree rotations

Widths: bridge_width is the central nonnuclear band's width; side_width
is the identical width of the two outer phase bands. Rotations use a common
angular step, so the asymmetric boundary bank has twice as many orientations.

Learning modes:
  A: train two independent nonnegative family strengths; filters fixed.
  B: train strengths and shared smooth geometric widths, longitudinal scale,
     and nonnuclear amplitude.
  C: train strengths and independent residuals on two canonical kernels,
     rotating each residual consistently with its family.

Splitting solver: explicit filter force, exact periodic implicit diffusion
(FFT), then relaxed projected fixed-point iterations for the implicit
multiwell step. Total-energy monotonicity is not guaranteed.

Run:
    python train_mm_learned_bridge.py --mode A --steps 100 --device cpu

Dependencies: torch, numpy, matplotlib (plots only).
"""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Optional

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F


# ============================================================
# Configuration
# ============================================================

@dataclass
class Config:
    mode: str = "A"

    size: int = 48
    batch_size: int = 2

    nuclei_min: int = 2
    nuclei_max: int = 4
    radius_min: float = 3.5
    radius_max: float = 6.0
    halo_width: float = 2.0

    corruption: str = "mixed"
    dilation_radius: int = 2
    erosion_radius: int = 1
    corruption_bridge_width: int = 2
    input_confidence: float = 0.90

    # Modica--Mortola
    epsilon: float = 0.5
    lam: float = 1.0
    well_cubic_weight: float = 1.0

    # Two trainable filter strengths
    separator_init_weight: float = 0.1
    boundary_init_weight: float = 0.1

    # Symmetric cross-sectional widths
    bridge_width: float = 3.0
    side_width: float = 2.0

    barrier_length: int = 13
    barrier_longitudinal_sigma: float = 3.5
    barrier_orientations: int = 8

    # Evolution
    outer_steps: int = 3
    dt: float = 0.05
    well_iterations: int = 3
    well_relaxation: float = 0.25

    # Training
    rollout_weights: tuple[float, ...] = (0.0, 0.0, 1.0)

    stationarity_weight: float = 0.0
    stationarity_steps: int = 1

    kernel_reg_weight: float = 0.01
    nuclear_loss_weight: float = 3.0

    steps: int = 200
    lr: float = 0.003
    grad_clip: float = 5.0

    eval_every: int = 20
    eval_rollout_steps: int = 15

    seed: int = 13
    device: str = "cpu"
    output_dir: str = "mm_bridge_run"

    def validate(self):
        if self.mode not in ("A", "B", "C"):
            raise ValueError("mode A/B/C required")

        if self.size < self.barrier_length or self.size < 24:
            raise ValueError("image too small")

        if (
            self.outer_steps < 1
            or len(self.rollout_weights) != self.outer_steps
        ):
            raise ValueError(
                "rollout_weights must have one entry per outer step"
            )

        if (
            not any(w > 0 for w in self.rollout_weights)
            or any(w < 0 for w in self.rollout_weights)
        ):
            raise ValueError(
                "rollout weights must be nonnegative with positive sum"
            )

        if self.dt <= 0 or self.epsilon <= 0 or self.lam < 0:
            raise ValueError("bad MM settings")

        if (
            self.well_iterations < 1
            or not (0 < self.well_relaxation <= 1)
        ):
            raise ValueError("bad implicit well settings")

        if not 1 / 3 < self.input_confidence < 1:
            raise ValueError("bad input confidence")

        if self.corruption not in (
            "dilation", "erosion", "bridge", "mixed"
        ):
            raise ValueError("bad corruption")

        if (
            self.barrier_orientations < 1
            or self.barrier_length % 2 != 1
        ):
            raise ValueError(
                "orientations >= 1 and odd length required"
            )

        if self.bridge_width <= 0 or self.side_width <= 0:
            raise ValueError(
                "bridge_width and side_width must be positive"
            )

        if (
            self.separator_init_weight < 0
            or self.boundary_init_weight < 0
        ):
            raise ValueError("filter strengths must be nonnegative")

        if self.corruption_bridge_width < 0:
            raise ValueError(
                "corruption_bridge_width must be nonnegative"
            )


# ============================================================
# Simplex and triple-well potential
# ============================================================

def simplex_project(v):
    """Euclidean projection onto the 3-class simplex at every pixel."""
    v_sort, _ = v.sort(dim=1, descending=True)
    cssv = v_sort.cumsum(dim=1) - 1

    seq = torch.arange(
        1, v.shape[1] + 1,
        device=v.device,
        dtype=v.dtype,
    ).view(1, -1, 1, 1)

    active = v_sort > cssv / seq
    rho = active.long().sum(
        dim=1, keepdim=True
    ).clamp_min(1)

    theta = cssv.gather(1, rho - 1) / rho
    return (v - theta).clamp_min(0)


def well_potential(p, gamma):
    a, b, c = p[:, 0], p[:, 1], p[:, 2]

    return (
        a * a * b * b
        + a * a * c * c
        + b * b * c * c
        + gamma * a * b * c
    )


def well_gradient(p, gamma):
    a = p[:, 0:1]
    b = p[:, 1:2]
    c = p[:, 2:3]

    return torch.cat(
        (
            2 * a * (b * b + c * c) + gamma * b * c,
            2 * b * (a * a + c * c) + gamma * a * c,
            2 * c * (a * a + b * b) + gamma * a * b,
        ),
        dim=1,
    )


def implicit_diffusion(p, dt, lam, eps):
    """Solve (I - 2 dt lam eps Delta) q = p, periodic boundary."""
    h, w = p.shape[-2:]

    fy = torch.fft.fftfreq(
        h, device=p.device, dtype=p.dtype
    )

    fx = torch.fft.rfftfreq(
        w, device=p.device, dtype=p.dtype
    )

    negative_delta = 4 * (
        torch.sin(math.pi * fy)[:, None] ** 2
        + torch.sin(math.pi * fx)[None, :] ** 2
    )

    denom = 1 + 2 * dt * lam * eps * negative_delta

    q = torch.fft.irfft2(
        torch.fft.rfft2(p, dim=(-2, -1)) / denom,
        s=(h, w),
        dim=(-2, -1),
    )

    return q


# ============================================================
# Construct three-phase oriented filters
# ============================================================

def _base_template_torch(
    cfg,
    family,
    dtype,
    device,
    bw=None,
    sw=None,
    sigma=None,
    amp=None,
):
    """Build a canonical three-channel phase-matching filter.

    Across the filter:

      separator: nuclear | nonnuclear | nuclear
      boundary:  background | nonnuclear | nuclear

    Each channel is spatially centered so a spatially uniform
    segmentation has zero response.
    """
    if family not in ("separator", "boundary"):
        raise ValueError("unknown family: " + str(family))

    n = cfg.barrier_length

    x = torch.arange(
        n, device=device, dtype=dtype
    ) - n // 2

    y = x

    bw = (
        torch.as_tensor(
            cfg.bridge_width, device=device, dtype=dtype
        )
        if bw is None else bw
    )

    sw = (
        torch.as_tensor(
            cfg.side_width, device=device, dtype=dtype
        )
        if sw is None else sw
    )

    sigma = (
        torch.as_tensor(
            cfg.barrier_longitudinal_sigma,
            device=device,
            dtype=dtype,
        )
        if sigma is None else sigma
    )

    amp = (
        torch.as_tensor(
            1.0, device=device, dtype=dtype
        )
        if amp is None else amp
    )

    sharp = 0.40

    # Smooth approximations to rectangular bands.
    middle = (
        torch.sigmoid((x + bw / 2) / sharp)
        * torch.sigmoid((bw / 2 - x) / sharp)
    )

    left = (
        torch.sigmoid((x + bw / 2 + sw) / sharp)
        * torch.sigmoid((-bw / 2 - x) / sharp)
    )

    right = (
        torch.sigmoid((x - bw / 2) / sharp)
        * torch.sigmoid((bw / 2 + sw - x) / sharp)
    )

    envelope = torch.exp(
        -0.5 * (y / sigma).square()
    )

    masks = torch.zeros(
        (3, n, n),
        dtype=dtype,
        device=device,
    )

    if family == "separator":
        masks[1] = (
            (left + right)[None, :]
            * envelope[:, None]
        )
    else:
        masks[0] = left[None, :] * envelope[:, None]
        masks[1] = right[None, :] * envelope[:, None]

    masks[2] = (
        amp * middle[None, :] * envelope[:, None]
    )

    # Expected phase versus competing phases.
    base = masks - masks.sum(
        dim=0, keepdim=True
    ) / 3

    # Remove channelwise DC response.
    return base - base.mean(
        dim=(-2, -1), keepdim=True
    )


def _rotate_bank(base, num_angles, full_circle=False):
    """Differentiably rotate a canonical three-channel kernel."""
    extent = (
        2 * math.pi if full_circle else math.pi
    )

    angles = (
        torch.arange(
            num_angles,
            device=base.device,
            dtype=base.dtype,
        )
        * extent / num_angles
    )

    rotations = torch.zeros(
        (num_angles, 2, 3),
        device=base.device,
        dtype=base.dtype,
    )

    rotations[:, 0, 0] = torch.cos(angles)
    rotations[:, 0, 1] = -torch.sin(angles)
    rotations[:, 1, 0] = torch.sin(angles)
    rotations[:, 1, 1] = torch.cos(angles)

    grid = F.affine_grid(
        rotations,
        (num_angles, 3, *base.shape[-2:]),
        align_corners=True,
    )

    return F.grid_sample(
        base.unsqueeze(0).expand(
            num_angles, -1, -1, -1
        ),
        grid,
        align_corners=True,
        padding_mode="zeros",
    )


def _normalized_bank(bank):
    bank = bank - bank.mean(
        dim=(-2, -1), keepdim=True
    )

    norm = (
        bank.clamp_min(0)
        .sum(dim=(1, 2, 3), keepdim=True)
        .clamp_min(1e-8)
    )

    return bank / norm


# ============================================================
# Energy model
# ============================================================

class BridgeEnergy(nn.Module):
    families = ("separator", "boundary")

    def __init__(self, cfg: Config):
        super().__init__()
        cfg.validate()
        self.cfg = cfg

        for family in self.families:
            strength = getattr(
                cfg, f"{family}_init_weight"
            )

            raw = math.log(
                math.expm1(max(strength, 1e-8))
            )

            setattr(
                self,
                f"raw_{family}_strength",
                nn.Parameter(
                    torch.tensor(
                        raw, dtype=torch.float32
                    )
                ),
            )

            base = _base_template_torch(
                cfg,
                family,
                torch.float32,
                torch.device("cpu"),
            )

            self.register_buffer(
                f"{family}_base_kernel",
                base.detach(),
            )

            if cfg.mode == "A":
                n = cfg.barrier_orientations * (
                    2 if family == "boundary" else 1
                )

                bank = _normalized_bank(
                    _rotate_bank(
                        base,
                        n,
                        family == "boundary",
                    )
                )

                self.register_buffer(
                    f"{family}_fixed_bank",
                    bank.detach(),
                )

            elif cfg.mode == "C":
                setattr(
                    self,
                    f"{family}_residual",
                    nn.Parameter(
                        torch.zeros_like(base)
                    ),
                )

        if cfg.mode == "B":
            self.raw_bw = nn.Parameter(
                torch.tensor(
                    math.log(
                        math.expm1(cfg.bridge_width)
                    ),
                    dtype=torch.float32,
                )
            )

            self.raw_sw = nn.Parameter(
                torch.tensor(
                    math.log(
                        math.expm1(cfg.side_width)
                    ),
                    dtype=torch.float32,
                )
            )

            self.raw_sigma = nn.Parameter(
                torch.tensor(
                    math.log(
                        math.expm1(
                            cfg.barrier_longitudinal_sigma
                        )
                    ),
                    dtype=torch.float32,
                )
            )

            self.raw_amp = nn.Parameter(
                torch.tensor(
                    0.0, dtype=torch.float32
                )
            )

    def strength(self, family="separator"):
        return F.softplus(
            getattr(
                self, f"raw_{family}_strength"
            )
        )

    def kernels(self, family="separator"):
        if family not in self.families:
            raise ValueError(
                "unknown family: " + str(family)
            )

        if self.cfg.mode == "A":
            return getattr(
                self, f"{family}_fixed_bank"
            )

        if self.cfg.mode == "B":
            base = getattr(
                self, f"{family}_base_kernel"
            )

            base = _base_template_torch(
                self.cfg,
                family,
                base.dtype,
                base.device,
                bw=F.softplus(self.raw_bw) + 1e-4,
                sw=F.softplus(self.raw_sw) + 1e-4,
                sigma=F.softplus(self.raw_sigma) + 1e-4,
                amp=torch.exp(
                    self.raw_amp.clamp(-2, 2)
                ),
            )

        else:
            base = (
                getattr(
                    self, f"{family}_base_kernel"
                )
                + getattr(
                    self, f"{family}_residual"
                )
            )

        n = self.cfg.barrier_orientations * (
            2 if family == "boundary" else 1
        )

        return _normalized_bank(
            _rotate_bank(
                base,
                n,
                family == "boundary",
            )
        )

    def responses(self, p, family="separator"):
        k = self.kernels(family).to(
            dtype=p.dtype
        )

        r = k.shape[-1] // 2

        return F.conv2d(
            F.pad(
                p,
                (r, r, r, r),
                mode="circular",
            ),
            k,
        )

    def family_energy(self, p, family):
        response = self.responses(
            p, family
        ).amax(dim=1)

        return (
            -0.5
            * self.strength(family)
            * F.relu(response)
            .square()
            .flatten(1)
            .sum(1)
        )

    def barrier_energy(self, p):
        return (
            self.family_energy(p, "separator")
            + self.family_energy(p, "boundary")
        )

    def terms(self, p):
        cfg = self.cfg

        dx = (
            torch.roll(p, -1, dims=-1) - p
        )

        dy = (
            torch.roll(p, -1, dims=-2) - p
        )

        diffusion = (
            cfg.lam
            * cfg.epsilon
            * (dx.square() + dy.square())
            .flatten(1)
            .sum(1)
        )

        well = (
            cfg.lam
            / cfg.epsilon
            * well_potential(
                p, cfg.well_cubic_weight
            )
            .flatten(1)
            .sum(1)
        )

        return {
            "diffusion": diffusion,
            "well": well,
            "separator": self.family_energy(
                p, "separator"
            ),
            "boundary": self.family_energy(
                p, "boundary"
            ),
        }

    def forward(self, p):
        return sum(
            self.terms(p).values()
        )

    def kernel_penalty(self):
        if self.cfg.mode == "C":
            return sum(
                getattr(
                    self, f"{family}_residual"
                ).square().mean()
                for family in self.families
            ) / len(self.families)

        if self.cfg.mode == "B":
            return (
                (
                    F.softplus(self.raw_bw)
                    - self.cfg.bridge_width
                ).square()
                +
                (
                    F.softplus(self.raw_sw)
                    - self.cfg.side_width
                ).square()
                +
                (
                    F.softplus(self.raw_sigma)
                    - self.cfg.barrier_longitudinal_sigma
                ).square()
                +
                self.raw_amp.square()
            ) / 4

        return self.raw_separator_strength.new_zeros(
            ()
        )

    def geometry(self):
        with torch.no_grad():
            d = {
                f"{family}_strength": float(
                    self.strength(family)
                )
                for family in self.families
            }

            d.update(
                bridge_width=float(
                    self.cfg.bridge_width
                ),
                side_width=float(
                    self.cfg.side_width
                ),
                separator_orientations=(
                    self.cfg.barrier_orientations
                ),
                boundary_orientations=(
                    2 * self.cfg.barrier_orientations
                ),
            )

            if self.cfg.mode == "B":
                d.update(
                    bridge_width=float(
                        F.softplus(self.raw_bw)
                    ),
                    side_width=float(
                        F.softplus(self.raw_sw)
                    ),
                    longitudinal_sigma=float(
                        F.softplus(self.raw_sigma)
                    ),
                    nonnuclear_amplitude=float(
                        torch.exp(
                            self.raw_amp.clamp(-2, 2)
                        )
                    ),
                )

            if self.cfg.mode == "C":
                for family in self.families:
                    residual = getattr(
                        self, f"{family}_residual"
                    )

                    d[f"{family}_residual_rms"] = float(
                        residual.square().mean().sqrt()
                    )

            return d


# ============================================================
# Differentiable energy evolution
# ============================================================

def step_energy(model, p, *, differentiable=True):
    """Explicit filters + implicit diffusion + approximate implicit well."""
    cfg = model.cfg

    with torch.enable_grad():
        x = (
            p if differentiable
            else p.detach()
        )

        if not x.requires_grad:
            x = x.requires_grad_(True)

        bridge = model.barrier_energy(x).sum()

        grad, = torch.autograd.grad(
            bridge,
            x,
            create_graph=differentiable,
        )

        explicit = x - cfg.dt * grad

        diffuse = implicit_diffusion(
            explicit,
            cfg.dt,
            cfg.lam,
            cfg.epsilon,
        )

        diffuse = simplex_project(diffuse)

        q = diffuse

        s = (
            cfg.dt
            * cfg.lam
            / cfg.epsilon
        )

        for _ in range(cfg.well_iterations):
            target = simplex_project(
                diffuse
                - s * well_gradient(
                    q, cfg.well_cubic_weight
                )
            )

            q = (
                (1 - cfg.well_relaxation) * q
                + cfg.well_relaxation * target
            )

    return (
        q if differentiable
        else q.detach()
    )


def rollout(
    model,
    p,
    n_steps,
    *,
    differentiable=True,
):
    states = []

    for _ in range(n_steps):
        p = step_energy(
            model,
            p,
            differentiable=differentiable,
        )
        states.append(p)

    return states


# ============================================================
# Synthetic nuclei
# ============================================================

def _disk_morph(mask, radius, expand):
    if radius <= 0:
        return mask.clone()

    offsets = [
        (i, j)
        for i in range(-radius, radius + 1)
        for j in range(-radius, radius + 1)
        if i * i + j * j <= radius * radius
    ]

    stack = torch.stack(
        [
            torch.roll(
                mask, (i, j), (-2, -1)
            )
            for i, j in offsets
        ]
    )

    return (
        stack.any(dim=0)
        if expand
        else stack.all(dim=0)
    )


def synthetic_batch(cfg, seed):
    """Return (corrupted probabilities, clean onehot, integer labels).

    Nuclei are nonoverlapping random ellipses with nonnuclear halos.
    Corruptions can dilate, erode, or connect nuclei.
    """
    rng = np.random.default_rng(seed)

    yy, xx = np.mgrid[
        :cfg.size, :cfg.size
    ]

    labels = []
    corrupted = []

    for b in range(cfg.batch_size):
        nucleus = np.zeros(
            (cfg.size, cfg.size),
            bool,
        )

        halo = np.zeros_like(nucleus)
        centers = []

        margin = (
            cfg.radius_max
            + cfg.halo_width
            + cfg.dilation_radius
            + 2
        )

        wanted = int(
            rng.integers(
                cfg.nuclei_min,
                cfg.nuclei_max + 1,
            )
        )

        for _ in range(wanted * 65):
            if len(centers) >= wanted:
                break

            rx = float(
                rng.uniform(
                    cfg.radius_min,
                    cfg.radius_max,
                )
            )

            ry = float(
                rng.uniform(
                    cfg.radius_min,
                    cfg.radius_max,
                )
            )

            cx = float(
                rng.uniform(
                    margin,
                    cfg.size - margin,
                )
            )

            cy = float(
                rng.uniform(
                    margin,
                    cfg.size - margin,
                )
            )

            angle = float(
                rng.uniform(0, math.pi)
            )

            dx = xx - cx
            dy = yy - cy

            cs = math.cos(angle)
            sn = math.sin(angle)

            xp = cs * dx + sn * dy
            yp = -sn * dx + cs * dy

            inner = (
                (xp / rx) ** 2
                + (yp / ry) ** 2
                <= 1
            )

            outer = (
                (xp / (rx + cfg.halo_width)) ** 2
                + (yp / (ry + cfg.halo_width)) ** 2
                <= 1
            )

            # Reject overlapping halos.
            if np.any(outer & halo):
                continue

            nucleus |= inner
            halo |= outer
            centers.append((cx, cy))

        lab = np.zeros_like(
            nucleus,
            dtype=np.int64,
        )

        lab[halo] = 2
        lab[nucleus] = 1

        dirty = lab.copy()

        kind = cfg.corruption

        if kind == "mixed":
            kind = str(
                rng.choice(
                    ["dilation", "erosion", "bridge"]
                )
            )

        if kind in ("dilation", "erosion"):
            m = torch.from_numpy(
                nucleus.copy()
            )

            r = (
                cfg.dilation_radius
                if kind == "dilation"
                else cfg.erosion_radius
            )

            modified = _disk_morph(
                m,
                r,
                kind == "dilation",
            ).numpy()

            if kind == "dilation":
                dirty[modified] = 1

            else:
                removed = (
                    nucleus & ~modified
                )

                dirty[removed] = 2

        elif (
            kind == "bridge"
            and len(centers) >= 2
        ):
            pair = min(
                (
                    (i, j)
                    for i in range(len(centers))
                    for j in range(
                        i + 1, len(centers)
                    )
                ),
                key=lambda ij: sum(
                    (
                        centers[ij[0]][d]
                        - centers[ij[1]][d]
                    ) ** 2
                    for d in (0, 1)
                ),
            )

            (x1, y1), (x2, y2) = [
                centers[z] for z in pair
            ]

            den = (
                (x2 - x1) ** 2
                + (y2 - y1) ** 2
                + 1e-9
            )

            t = np.clip(
                (
                    (xx - x1) * (x2 - x1)
                    + (yy - y1) * (y2 - y1)
                ) / den,
                0,
                1,
            )

            dline = np.hypot(
                xx - (
                    x1 + t * (x2 - x1)
                ),
                yy - (
                    y1 + t * (y2 - y1)
                ),
            )

            dirty[
                dline
                <= cfg.corruption_bridge_width / 2
            ] = 1

        labels.append(lab)
        corrupted.append(dirty)

    y = torch.tensor(
        np.stack(labels),
        device=cfg.device,
        dtype=torch.long,
    )

    d = torch.tensor(
        np.stack(corrupted),
        device=cfg.device,
        dtype=torch.long,
    )

    clean = (
        F.one_hot(y, 3)
        .permute(0, 3, 1, 2)
        .to(torch.float32)
    )

    corrupted_hot = (
        F.one_hot(d, 3)
        .permute(0, 3, 1, 2)
        .to(torch.float32)
    )

    alpha = cfg.input_confidence

    p0 = (
        alpha * corrupted_hot
        + 0.5 * (1 - alpha)
        * (1 - corrupted_hot)
    )

    return p0, clean, y


# ============================================================
# Loss and metrics
# ============================================================

def probability_loss(p, y, cfg):
    w = p.new_tensor(
        [
            1.0,
            cfg.nuclear_loss_weight,
            1.0,
        ]
    ).view(1, 3, 1, 1)

    return (
        (p - y).square() * w
    ).mean()


def nuclear_dice(p, y):
    a = p.argmax(1) == 1
    b = y == 1

    num = (
        2 * (a & b)
        .sum(dim=(-2, -1))
        .float()
        + 1e-8
    )

    den = (
        a.sum(dim=(-2, -1))
        + b.sum(dim=(-2, -1))
    ).float() + 1e-8

    return (
        num / den
    ).mean().item()


# ============================================================
# Training
# ============================================================

def train_step(model, opt, cfg, seed):
    model.train()

    p0, clean, _ = synthetic_batch(
        cfg, seed
    )

    states = rollout(
        model,
        p0,
        cfg.outer_steps,
        differentiable=True,
    )

    sw = sum(cfg.rollout_weights)

    correction = sum(
        float(w) * probability_loss(
            p, clean, cfg
        )
        for w, p in zip(
            cfg.rollout_weights,
            states,
        )
    ) / sw

    if cfg.stationarity_weight:
        stay_states = rollout(
            model,
            clean,
            cfg.stationarity_steps,
            differentiable=True,
        )

        stationarity = sum(
            probability_loss(
                p, clean, cfg
            )
            for p in stay_states
        ) / len(stay_states)

    else:
        stationarity = correction.new_zeros(
            ()
        )

    kreg = model.kernel_penalty()

    loss = (
        correction
        + cfg.stationarity_weight
        * stationarity
        + cfg.kernel_reg_weight * kreg
    )

    opt.zero_grad(set_to_none=True)

    loss.backward()

    grad_norm = float(
        torch.nn.utils.clip_grad_norm_(
            model.parameters(),
            cfg.grad_clip,
        )
    )

    opt.step()

    return {
        "loss": float(loss.detach()),
        "correction": float(
            correction.detach()
        ),
        "stationarity": float(
            stationarity.detach()
        ),
        "kernel_reg": float(
            kreg.detach()
        ),
        "grad_norm": grad_norm,
    }


# ============================================================
# Evaluation
# ============================================================

def evaluate(
    model,
    cfg,
    seed,
    *,
    plot_path: Optional[Path] = None,
):
    model.eval()

    p0, y, target = synthetic_batch(
        cfg, seed
    )

    states = rollout(
        model,
        p0,
        cfg.eval_rollout_steps,
        differentiable=False,
    )

    clean_states = rollout(
        model,
        y,
        min(cfg.eval_rollout_steps, 10),
        differentiable=False,
    )

    with torch.no_grad():
        out = {
            "input_dice": nuclear_dice(
                p0, target
            ),
            "output_dice": nuclear_dice(
                states[cfg.outer_steps - 1],
                target,
            ),
            "long_dice": nuclear_dice(
                states[-1], target
            ),
            "clean_long_dice": nuclear_dice(
                clean_states[-1], target
            ),
            "clean_one_step_rms": float(
                (clean_states[0] - y)
                .square()
                .mean()
                .sqrt()
            ),
            "nuclear_fraction_long": float(
                (
                    states[-1].argmax(1)
                    == 1
                ).float().mean()
            ),
            "energy_input_per_pixel": float(
                model(p0).mean()
                / cfg.size ** 2
            ),
            "energy_corrected_per_pixel": float(
                model(
                    states[
                        cfg.outer_steps - 1
                    ]
                ).mean()
                / cfg.size ** 2
            ),
            "geometry": model.geometry(),
        }

    if plot_path is not None:
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(
            1, 5, figsize=(14, 3)
        )

        image_states = [
            y,
            p0,
            states[0],
            states[cfg.outer_steps - 1],
            states[-1],
        ]

        titles = [
            "Clean target",
            "Corrupted input",
            "After 1",
            "After 3",
            "Long rollout",
        ]

        for a, p, title in zip(
            ax, image_states, titles
        ):
            a.imshow(
                p[0]
                .argmax(0)
                .detach()
                .cpu()
                .numpy(),
                vmin=0,
                vmax=2,
                interpolation="nearest",
            )

            a.set_title(title)
            a.axis("off")

        fig.tight_layout()
        fig.savefig(
            plot_path, dpi=150
        )
        plt.close(fig)

    return out


# ============================================================
# Training loop and checkpointing
# ============================================================

def train(cfg):
    cfg.validate()
    torch.manual_seed(cfg.seed)

    out = Path(cfg.output_dir)
    out.mkdir(
        parents=True, exist_ok=True
    )

    model = BridgeEnergy(cfg).to(
        cfg.device
    )

    opt = torch.optim.Adam(
        model.parameters(),
        lr=cfg.lr,
    )

    history = []

    start = evaluate(
        model,
        cfg,
        cfg.seed + 10000,
        plot_path=out / "before.png",
    )

    print(
        "INITIAL",
        json.dumps(start),
    )

    for t in range(
        1, cfg.steps + 1
    ):
        m = train_step(
            model,
            opt,
            cfg,
            cfg.seed + t,
        )

        if (
            t == 1
            or t % cfg.eval_every == 0
            or t == cfg.steps
        ):
            ev = evaluate(
                model,
                cfg,
                cfg.seed + 10000,
                plot_path=out / "after.png",
            )

            item = {
                "step": t,
                **m,
                **ev,
            }

            history.append(item)

            print(
                "STEP",
                t,
                json.dumps(item),
            )

    torch.save(
        {
            "config": asdict(cfg),
            "state_dict": model.state_dict(),
        },
        out / "model.pt",
    )

    (
        out / "history.json"
    ).write_text(
        json.dumps(
            history, indent=2
        )
    )

    (
        out / "config.json"
    ).write_text(
        json.dumps(
            asdict(cfg), indent=2
        )
    )

    return model, history


def load_checkpoint(
    path,
    device="cpu",
):
    ck = torch.load(
        path,
        map_location=device,
        weights_only=False,
    )

    cfg = Config(
        **ck["config"]
    )
    cfg.device = str(device)

    model = BridgeEnergy(
        cfg
    ).to(device)

    model.load_state_dict(
        ck["state_dict"]
    )

    model.eval()

    return model, cfg


# ============================================================
# Command-line interface
# ============================================================

def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--mode",
        choices=["A", "B", "C"],
        default="A",
    )

    parser.add_argument(
        "--steps",
        type=int,
        default=200,
    )

    parser.add_argument(
        "--size",
        type=int,
        default=48,
    )

    parser.add_argument(
        "--batch-size",
        type=int,
        default=2,
    )

    parser.add_argument(
        "--device",
        default="cpu",
    )

    parser.add_argument(
        "--out",
        default="mm_bridge_run",
    )

    parser.add_argument(
        "--corruption",
        choices=[
            "dilation",
            "erosion",
            "bridge",
            "mixed",
        ],
        default="mixed",
    )

    parser.add_argument(
        "--stationarity-weight",
        type=float,
        default=0.0,
    )

    parser.add_argument(
        "--dt",
        type=float,
        default=0.05,
    )

    parser.add_argument(
        "--bridge-width",
        type=float,
        default=3.0,
    )

    parser.add_argument(
        "--side-width",
        type=float,
        default=2.0,
    )

    parser.add_argument(
        "--separator-weight",
        type=float,
        default=0.1,
    )

    parser.add_argument(
        "--boundary-weight",
        type=float,
        default=0.1,
    )

    args = parser.parse_args()

    cfg = Config(
        mode=args.mode,
        steps=args.steps,
        size=args.size,
        batch_size=args.batch_size,
        device=args.device,
        output_dir=args.out,
        corruption=args.corruption,
        stationarity_weight=(
            args.stationarity_weight
        ),
        dt=args.dt,
        bridge_width=args.bridge_width,
        side_width=args.side_width,
        separator_init_weight=(
            args.separator_weight
        ),
        boundary_init_weight=(
            args.boundary_weight
        ),
    )

    train(cfg)


if __name__ == "__main__":
    main()
