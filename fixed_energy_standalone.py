
"""Standalone three-class segmentation energy experiment.

Classes: 0=background, 1=nuclear, 2=nonnuclear.

Energy:
    E = lambda*epsilon * sum |grad p|^2
      + lambda/epsilon * sum W(p)
      - barrier_weight/2 * sum ReLU(max_orientation(K*p))^2

W(p) = p0^2*p1^2 + p0^2*p2^2 + p1^2*p2^2
       + well_cubic_weight*p0*p1*p2

The barrier filters span the nuclear and nonnuclear channels
simultaneously. Their default cross-section is:

    NN BBB NN

Multiple orientations are evaluated, with the maximum response
used at each pixel. No learned parameters or response threshold.

Requires: numpy, scipy, torch, matplotlib.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Union

import numpy as np
import torch
from torch import Tensor, nn
from torch.nn import functional as F
from scipy import fft
from scipy.ndimage import gaussian_filter, rotate


# ============================================================
# Configuration
# ============================================================

@dataclass
class FixedEnergyConfig:

    # Modica-Mortola
    epsilon: float = 0.2
    lam: float = 1.0
    well_cubic_weight: float = 1.0

    # Hand-designed barrier filter
    barrier_weight: float = 0.1
    barrier_width: int = 3
    nuclear_band_width: int = 2
    barrier_length: int = 13
    barrier_longitudinal_sigma: float = 3.5
    barrier_orientations: int = 8

    # Minimization
    steps: int = 200
    step_size: float = 0.05
    backtracking: bool = True
    backtracking_factor: float = 0.5
    max_backtracks: int = 20
    min_step_size: float = 1e-10
    save_every: int = 20
    verbose_every: int = 20

    # Synthetic input corruption
    seed: int = 42
    base_logit_magnitude: float = 1.5
    perturb_strength: float = 1.0
    perturb_probability: float = 1.0
    gt_blur_sigma: float = 0.0

    spectral_bdwidth: float = 0.9
    spectral_wavelengths: tuple = (
        250., 150., 100., 50., 25., 12.5, 1.
    )
    spectral_amplitudes: tuple = (
        15., 12., 8., 2., 1., 0.5, 0.3
    )
    spectral_amplitude_jitter: tuple = (0.6, 1.5)

    ignore_index: int = -100

    # Visualization
    max_plot_snapshots: int = 8
    plot_class_probabilities: bool = False
    plot_dpi: int = 140

    def validate(self):
        if self.epsilon <= 0:
            raise ValueError("epsilon must be positive")

        if self.lam < 0 or self.well_cubic_weight < 0:
            raise ValueError("Invalid Modica-Mortola weights")

        if self.barrier_weight < 0:
            raise ValueError("barrier_weight must be nonnegative")

        if self.barrier_width < 1 or self.barrier_width % 2 != 1:
            raise ValueError("barrier_width must be positive and odd")

        if self.nuclear_band_width < 1:
            raise ValueError("nuclear_band_width must be positive")

        if self.barrier_length < 1 or self.barrier_length % 2 != 1:
            raise ValueError("barrier_length must be positive and odd")

        if self.barrier_longitudinal_sigma <= 0:
            raise ValueError("barrier_longitudinal_sigma must be positive")

        if self.barrier_orientations < 1:
            raise ValueError("barrier_orientations must be positive")

        if self.steps < 0 or self.step_size <= 0:
            raise ValueError("Invalid minimization settings")

        if self.save_every < 1 or self.verbose_every < 1:
            raise ValueError("Invalid logging intervals")

        if not 0 < self.backtracking_factor < 1:
            raise ValueError("Invalid backtracking factor")

        if self.max_backtracks < 0 or self.min_step_size <= 0:
            raise ValueError("Invalid backtracking settings")

        if self.base_logit_magnitude <= 0:
            raise ValueError("base_logit_magnitude must be positive")

        if self.perturb_strength < 0:
            raise ValueError("perturb_strength must be nonnegative")

        if not 0 <= self.perturb_probability <= 1:
            raise ValueError("Invalid perturb_probability")

        if self.gt_blur_sigma < 0 or self.spectral_bdwidth <= 0:
            raise ValueError("Invalid perturbation settings")

        if len(self.spectral_wavelengths) != len(
            self.spectral_amplitudes
        ):
            raise ValueError("Spectral bands must have matching lengths")

        if not self.spectral_wavelengths:
            raise ValueError("At least one spectral band required")

        if any(x <= 0 for x in self.spectral_wavelengths):
            raise ValueError("Spectral wavelengths must be positive")

        if any(x < 0 for x in self.spectral_amplitudes):
            raise ValueError("Spectral amplitudes must be nonnegative")

        lo, hi = self.spectral_amplitude_jitter
        if lo < 0 or hi < lo:
            raise ValueError("Invalid spectral amplitude jitter")

        if self.max_plot_snapshots < 1:
            raise ValueError("max_plot_snapshots must be positive")


# ============================================================
# Three-phase simplex coordinates
# ============================================================

def project_triangle(q: Tensor) -> Tensor:
    """
    Euclidean projection onto the reduced-coordinate triangle.

    q: [B,2,H,W]
    q[:,0] = nuclear
    q[:,1] = nonnuclear

    Background probability is 1 - nuclear - nonnuclear.
    """

    if q.ndim != 4 or q.shape[1] != 2:
        raise ValueError("Expected [B,2,H,W]")

    u = q[:, 0:1]
    w = q[:, 1:2]

    # Candidate points on each triangle edge.
    on_u0 = torch.cat([
        torch.zeros_like(u),
        w.clamp(0, 1)
    ], dim=1)

    on_w0 = torch.cat([
        u.clamp(0, 1),
        torch.zeros_like(w)
    ], dim=1)

    t = ((u - w + 1) / 2).clamp(0, 1)

    on_background0 = torch.cat([
        t, 1 - t
    ], dim=1)

    inside = (
        (u >= 0)
        & (w >= 0)
        & (u + w <= 1)
    )

    candidates = torch.stack([
        on_u0,
        on_w0,
        on_background0
    ], dim=1)

    distances = (
        candidates - q.unsqueeze(1)
    ).square().sum(dim=2, keepdim=True)

    index = distances.argmin(dim=1, keepdim=True)

    closest = candidates.gather(
        1,
        index.expand(-1, -1, 2, -1, -1)
    ).squeeze(1)

    return torch.where(
        inside.expand_as(q),
        q,
        closest
    )


def reduced_to_prob(q: Tensor) -> Tensor:
    """[nuclear, nonnuclear] -> [background, nuclear, nonnuclear]."""

    u = q[:, 0:1]
    w = q[:, 1:2]

    return torch.cat([
        1 - u - w,
        u,
        w
    ], dim=1)


def prob_to_reduced(p: Tensor) -> Tensor:
    if p.ndim != 4 or p.shape[1] != 3:
        raise ValueError("Expected probabilities [B,3,H,W]")

    return project_triangle(p[:, 1:3])


def periodic_laplacian(x: Tensor) -> Tensor:
    return (
        torch.roll(x, 1, -2)
        + torch.roll(x, -1, -2)
        + torch.roll(x, 1, -1)
        + torch.roll(x, -1, -1)
        - 4*x
    )


# ============================================================
# Multichannel, multi-orientation barrier filter bank
# ============================================================

def make_barrier_filter_bank(
    cfg: FixedEnergyConfig
) -> Tensor:
    """
    Returns [orientations, 2, S, S].

    Channel 0: nuclear
    Channel 1: nonnuclear

    Default cross-section:
        NN BBB NN

    The two channel templates are signed complements.
    Each channel has zero spatial mean.

    Rotated kernels are renormalized so positive weights
    sum to one across both channels.
    """

    cfg.validate()

    width = (
        cfg.barrier_width
        + 2 * cfg.nuclear_band_width
    )

    size = max(width, cfg.barrier_length)

    if size % 2 == 0:
        size += 1

    center = size // 2
    first = center - width // 2
    last = first + width

    # Nuclear probability is preferred in outer bands.
    nuclear = np.ones(width, dtype=np.float64)

    start = cfg.nuclear_band_width
    end = start + cfg.barrier_width

    nuclear[start:end] = 0.0

    # Nonnuclear probability is preferred in the middle.
    nonnuclear = 1.0 - nuclear

    # Signed, zero-mean cross-sections.
    nuclear -= nuclear.mean()
    nonnuclear -= nonnuclear.mean()

    # Gaussian weighting along the barrier length.
    y = np.arange(size, dtype=np.float64) - center

    along = np.exp(
        -0.5 * (
            y / cfg.barrier_longitudinal_sigma
        ) ** 2
    )

    along[np.abs(y) > cfg.barrier_length // 2] = 0.0

    base = np.zeros(
        (2, size, size),
        dtype=np.float64
    )

    base[0, :, first:last] = (
        along[:, None] * nuclear[None, :]
    )

    base[1, :, first:last] = (
        along[:, None] * nonnuclear[None, :]
    )

    kernels = []

    for i in range(cfg.barrier_orientations):

        angle = (
            180.0 * i / cfg.barrier_orientations
        )

        if i == 0:
            kernel = base.copy()

        else:
            kernel = np.stack([
                rotate(
                    base[c],
                    angle,
                    reshape=False,
                    order=1,
                    mode="constant",
                    cval=0.0,
                    prefilter=False,
                )
                for c in range(2)
            ])

        # Remove numerical DC bias from rotation.
        kernel -= kernel.mean(
            axis=(1, 2),
            keepdims=True
        )

        positive_sum = np.maximum(
            kernel, 0.0
        ).sum()

        if positive_sum <= 0:
            raise ValueError("Degenerate filter")

        kernel /= positive_sum

        kernels.append(kernel)

    return torch.tensor(
        np.stack(kernels),
        dtype=torch.float32
    )


# ============================================================
# Fixed energy
# ============================================================

class ModicaMortolaBarrierEnergy(nn.Module):
    """
    Fixed, nonlearned three-phase energy.

    All energy terms return one scalar per batch element.
    """

    def __init__(self, cfg: FixedEnergyConfig):
        super().__init__()

        cfg.validate()
        self.cfg = cfg

        self.register_buffer(
            "barrier_kernels",
            make_barrier_filter_bank(cfg)
        )

    def barrier_responses(self, p: Tensor) -> Tensor:
        """
        Returns [B,orientations,H,W].

        Uses nuclear and nonnuclear channels only.
        Periodic boundary handling.
        """

        kernel = self.barrier_kernels.to(dtype=p.dtype)
        radius = kernel.shape[-1] // 2

        pair = p[:, 1:3]

        # For ordinary microscopy images.
        if radius < min(pair.shape[-2:]):

            padded = F.pad(
                pair,
                (radius, radius, radius, radius),
                mode="circular"
            )

            return F.conv2d(padded, kernel)

        # Fallback for unusually small synthetic images.
        out = p.new_zeros((
            p.shape[0],
            kernel.shape[0],
            *p.shape[-2:]
        ))

        for row in range(kernel.shape[-2]):
            for col in range(kernel.shape[-1]):

                weights = kernel[:, :, row, col]

                shifted = torch.roll(
                    pair,
                    shifts=(radius-row, radius-col),
                    dims=(-2, -1)
                )

                out = out + torch.einsum(
                    "bchw,oc->bohw",
                    shifted,
                    weights
                )

        return out

    def energy_terms_reduced(
        self,
        q: Tensor
    ) -> dict[str, Tensor]:

        p = reduced_to_prob(q)

        z = p[:, 0]
        u = p[:, 1]
        w = p[:, 2]

        # Triple-well potential.
        W = (
            u.square()*w.square()
            + u.square()*z.square()
            + w.square()*z.square()
            + self.cfg.well_cubic_weight*u*w*z
        )

        well = (
            self.cfg.lam / self.cfg.epsilon
        ) * W.flatten(1).sum(1)

        # Three-phase Dirichlet energy:
        # lambda*epsilon * sum_k |grad p_k|^2

        dx = torch.roll(p, -1, -1) - p
        dy = torch.roll(p, -1, -2) - p

        smooth = (
            self.cfg.lam * self.cfg.epsilon
        ) * (
            dx.square().flatten(1).sum(1)
            + dy.square().flatten(1).sum(1)
        )

        # Oriented barrier reward.
        if self.cfg.barrier_weight > 0:

            responses = self.barrier_responses(p)

            # Best orientation at each pixel.
            response = responses.amax(dim=1)

            # Positive responses only, no threshold.
            activated = F.relu(response)

            barrier = (
                -0.5 * self.cfg.barrier_weight
                * activated.square().flatten(1).sum(1)
            )

        else:
            barrier = torch.zeros_like(well)

        return {
            "dirichlet": smooth,
            "triple_well": well,
            "barrier": barrier,
        }

    def energy_terms(self, p: Tensor):
        return self.energy_terms_reduced(
            prob_to_reduced(p)
        )

    def energy_reduced(self, q: Tensor):
        return sum(
            self.energy_terms_reduced(q).values()
        )

    def forward(self, p: Tensor):
        return sum(
            self.energy_terms(p).values()
        )


# ============================================================
# Spectral perturbation generator
# ============================================================

def spectral_noise(
    shape,
    bands,
    width,
    rng
):
    h, w = shape

    white = rng.standard_normal(shape)

    ky = fft.fftfreq(h)[:, None]
    kx = fft.rfftfreq(w)[None, :]

    k = np.hypot(kx, ky)

    filt = np.zeros_like(k)

    for amplitude, wavelength in bands:

        filt += amplitude * np.exp(
            -0.5 * (
                (k - 1.0/wavelength)
                / (width/wavelength)
            ) ** 2
        )

    noise = fft.irfftn(
        fft.rfftn(white) * filt,
        s=shape
    ).real

    noise -= noise.mean()

    noise /= max(
        float(np.max(np.abs(noise))),
        1e-8
    )

    return noise.astype(np.float32)


def synthetic_logits(
    target: Tensor,
    cfg: FixedEnergyConfig,
    *,
    seed=None
) -> Tensor:

    cfg.validate()

    if target.ndim == 2:
        target = target.unsqueeze(0)

    rng = np.random.default_rng(
        cfg.seed if seed is None else seed
    )

    B, H, W = target.shape

    target_np = target.detach().cpu().numpy().astype(
        np.int64
    )

    safe = target_np.copy()
    safe[safe == cfg.ignore_index] = 0

    if not np.isin(safe, [0, 1, 2]).all():
        raise ValueError("Invalid target classes")

    onehot = np.eye(
        3, dtype=np.float32
    )[safe].transpose(0, 3, 1, 2)

    if cfg.gt_blur_sigma > 0:
        onehot = gaussian_filter(
            onehot,
            sigma=(
                0, 0,
                cfg.gt_blur_sigma,
                cfg.gt_blur_sigma
            ),
            mode="reflect"
        )

    span = 2.0 * cfg.base_logit_magnitude

    logits = (
        -cfg.base_logit_magnitude
        + span * onehot
    )

    waves = np.asarray(
        cfg.spectral_wavelengths,
        dtype=np.float64
    )

    amps = np.asarray(
        cfg.spectral_amplitudes,
        dtype=np.float64
    )

    if cfg.perturb_strength > 0:

        for b in range(B):

            if rng.random() >= cfg.perturb_probability:
                continue

            for c in range(3):

                jitter = rng.uniform(
                    *cfg.spectral_amplitude_jitter,
                    size=amps.shape
                )

                bands = np.column_stack([
                    amps*jitter,
                    waves
                ])

                noise = spectral_noise(
                    (H, W),
                    bands,
                    cfg.spectral_bdwidth,
                    rng
                )

                logits[b, c] += (
                    cfg.perturb_strength
                    * span
                    * noise
                )

    return torch.from_numpy(
        np.asarray(logits, dtype=np.float32)
    ).to(target.device)


def initial_probabilities(
    target,
    cfg,
    *,
    seed=None
):
    return synthetic_logits(
        target, cfg, seed=seed
    ).softmax(dim=1)


def target_from_stain_labels(
    labels: Tensor,
    ignore_index=-100
) -> Tensor:
    """
    Supports:
        [B,Z,3,H,W]
        [B,3,H,W]

    Stain channels:
        0 nuclear
        1 nonnuclear
        2 supervision boost (ignored here)
    """

    if labels.ndim == 5:

        B, Z, C, H, W = labels.shape

        labels = labels.reshape(
            B*Z, C, H, W
        )

    if labels.ndim != 4 or labels.shape[1] < 2:
        raise ValueError("Invalid stain label shape")

    nuc = labels[:, 0] > 0.5
    non = labels[:, 1] > 0.5

    target = torch.zeros_like(
        nuc, dtype=torch.long
    )

    target[nuc & ~non] = 1
    target[non & ~nuc] = 2
    target[nuc & non] = ignore_index

    return target


# ============================================================
# Dice diagnostic
# ============================================================

def hard_dice(
    p: Tensor,
    target: Tensor,
    ignore_index=-100
):
    valid = target != ignore_index
    pred = p.argmax(dim=1)

    result = {}

    for k, name in enumerate([
        "background",
        "nuclear",
        "nonnuclear"
    ]):

        x = (pred == k) & valid
        y = (target == k) & valid

        denominator = x.sum() + y.sum()

        result[f"dice_{name}"] = (
            (2.0 * (x & y).sum() / denominator).item()
            if denominator.item() > 0
            else float("nan")
        )

    return result


# ============================================================
# Result container
# ============================================================

@dataclass
class FixedEnergyResult:

    initial: Tensor
    final: Tensor
    snapshots: dict
    history: list
    target: Optional[Tensor] = None
    stopped_early: bool = False


# ============================================================
# Experiment / minimizer
# ============================================================

class FixedEnergyExperiment:

    def __init__(
        self,
        cfg: Optional[FixedEnergyConfig] = None,
        device: Union[str, torch.device] = "cpu"
    ):

        self.cfg = (
            FixedEnergyConfig()
            if cfg is None else cfg
        )

        self.cfg.validate()

        self.device = torch.device(device)

        self.energy = ModicaMortolaBarrierEnergy(
            self.cfg
        ).to(self.device)

    @torch.no_grad()
    def _diagnostics(
        self,
        q,
        iteration,
        step,
        target
    ):

        p = reduced_to_prob(q)

        H, W = p.shape[-2:]
        area = H * W

        terms = self.energy.energy_terms_reduced(q)

        record = {
            "iteration": iteration,
            "step_size": float(step),
        }

        for name, value in terms.items():
            record[name] = (
                value / area
            ).mean().item()

        record["total"] = sum(
            record[name] for name in terms
        )

        record["mean_confidence"] = (
            p.amax(dim=1).mean().item()
        )

        for k, name in enumerate([
            "background",
            "nuclear",
            "nonnuclear"
        ]):

            record[f"fraction_{name}"] = (
                (p.argmax(dim=1) == k)
                .float().mean().item()
            )

        if target is not None:

            record.update(
                hard_dice(
                    p,
                    target,
                    self.cfg.ignore_index
                )
            )

        return record

    def run(
        self,
        *,
        target=None,
        p0=None,
        logits0=None,
        seed=None,
        verbose=True,
    ):
        """
        Supply:
          target -> automatically generate corrupted logits
          p0 -> start from existing probabilities
          logits0 -> start from existing logits

        target can also accompany p0/logits0 for diagnostics.
        """

        if p0 is not None and logits0 is not None:
            raise ValueError(
                "Provide only one of p0 or logits0"
            )

        if target is None and p0 is None and logits0 is None:
            raise ValueError(
                "Need target, p0 or logits0"
            )

        if target is not None:

            target = torch.as_tensor(
                target,
                device=self.device
            ).long()

            if target.ndim == 2:
                target = target[None]

        if p0 is None and logits0 is None:

            p = initial_probabilities(
                target, self.cfg, seed=seed
            )

        elif logits0 is not None:

            p = torch.as_tensor(
                logits0,
                device=self.device,
                dtype=torch.float32
            )

            if p.ndim == 3:
                p = p[None]

            p = p.softmax(dim=1)

        else:

            p = torch.as_tensor(
                p0,
                device=self.device,
                dtype=torch.float32
            )

        if p.ndim == 3:
            p = p[None]

        if p.ndim != 4 or p.shape[1] != 3:
            raise ValueError(
                "Expected probabilities [B,3,H,W]"
            )

        if target is not None:
            if target.shape != (
                p.shape[0],
                *p.shape[-2:]
            ):
                raise ValueError(
                    "Target and input shapes do not match"
                )

        if not torch.isfinite(p).all():
            raise ValueError("Nonfinite probabilities")

        q = prob_to_reduced(p.detach()).detach()

        p_initial = (
            reduced_to_prob(q)
            .detach().cpu().clone()
        )

        history = [
            self._diagnostics(
                q, 0, self.cfg.step_size, target
            )
        ]

        snapshots = {0: p_initial}

        trial_step = self.cfg.step_size
        stopped_early = False

        for iteration in range(
            1, self.cfg.steps + 1
        ):

            # Gradient with respect to state only.
            with torch.enable_grad():

                x = q.detach().requires_grad_(True)

                energy = self.energy.energy_reduced(
                    x
                ).sum()

                grad, = torch.autograd.grad(
                    energy, x
                )

            # Projected gradient descent with backtracking.
            with torch.no_grad():

                accepted = False
                step = trial_step

                attempts = (
                    self.cfg.max_backtracks + 1
                    if self.cfg.backtracking else 1
                )

                for _ in range(attempts):

                    candidate = project_triangle(
                        q - step*grad
                    )

                    next_energy = (
                        self.energy.energy_reduced(
                            candidate
                        ).sum()
                    )

                    if torch.isfinite(next_energy) and (
                        not self.cfg.backtracking
                        or next_energy <= energy.detach() + 1e-9
                    ):

                        accepted = True
                        break

                    step *= self.cfg.backtracking_factor

                    if step < self.cfg.min_step_size:
                        break

                if not accepted:

                    stopped_early = True

                    if verbose:
                        print(
                            "No energy-decreasing step at "
                            f"iteration {iteration}"
                        )

                    break

                q = candidate.detach()
                trial_step = step

            should_save = (
                iteration % self.cfg.save_every == 0
                or iteration == self.cfg.steps
            )

            if should_save:

                history.append(
                    self._diagnostics(
                        q,
                        iteration,
                        trial_step,
                        target
                    )
                )

                snapshots[iteration] = (
                    reduced_to_prob(q)
                    .detach().cpu().clone()
                )

            if verbose and (
                iteration % self.cfg.verbose_every == 0
                or iteration == self.cfg.steps
            ):

                r = self._diagnostics(
                    q,
                    iteration,
                    trial_step,
                    target
                )

                message = (
                    f"it={iteration:4d}  "
                    f"E/pix={r['total']:+.6f}  "
                    f"D={r['dirichlet']:.6f}  "
                    f"W={r['triple_well']:.6f}  "
                    f"Barrier={r['barrier']:+.6f}  "
                    f"maxP={r['mean_confidence']:.3f}  "
                    f"step={trial_step:.3g}"
                )

                if target is not None:

                    message += (
                        f"  nuclearDice="
                        f"{r['dice_nuclear']:.3f}"
                        f"  nonDice="
                        f"{r['dice_nonnuclear']:.3f}"
                    )

                print(message)

        final = (
            reduced_to_prob(q)
            .detach().cpu().clone()
        )

        return FixedEnergyResult(
            initial=p_initial,
            final=final,
            snapshots=snapshots,
            history=history,
            target=(
                None if target is None
                else target.cpu()
            ),
            stopped_early=stopped_early,
        )

    # ========================================================
    # Visualization
    # ========================================================

    def plot(
        self,
        result: FixedEnergyResult,
        *,
        sample=0,
        save_path=None,
        show=True,
    ):

        import matplotlib.pyplot as plt
        from matplotlib.colors import (
            ListedColormap, BoundaryNorm
        )

        keys = sorted(result.snapshots)

        if len(keys) > self.cfg.max_plot_snapshots:

            selected = np.linspace(
                0,
                len(keys)-1,
                self.cfg.max_plot_snapshots
            ).round().astype(int)

            keys = [
                keys[int(i)]
                for i in np.unique(selected)
            ]

        nrows = (
            5 if self.cfg.plot_class_probabilities
            else 3
        )

        if result.target is not None:
            nrows += 1

        ncols = len(keys)

        fig, axes = plt.subplots(
            nrows,
            ncols,
            figsize=(3*ncols, 2.8*nrows),
            squeeze=False
        )

        cmap = ListedColormap([
            "#151515",
            "#4285F4",
            "#F7B64C"
        ])

        norm = BoundaryNorm(
            [-0.5, 0.5, 1.5, 2.5],
            cmap.N
        )

        row = 0

        if result.target is not None:

            gt = result.target[
                sample
            ].numpy().copy()

            gt[
                gt == self.cfg.ignore_index
            ] = -1

            for col in range(ncols):

                axes[row, col].imshow(
                    gt,
                    cmap=cmap,
                    norm=norm,
                    interpolation="nearest"
                )

                if col == 0:
                    axes[row, col].set_title(
                        "Ground truth"
                    )

            row += 1

        for col, iteration in enumerate(keys):

            p = result.snapshots[
                iteration
            ][sample].numpy()

            axes[row, col].imshow(
                p.argmax(axis=0),
                cmap=cmap,
                norm=norm,
                interpolation="nearest"
            )

            axes[row, col].set_title(
                f"Iteration {iteration}"
            )

            axes[row+1, col].imshow(
                p[1],
                vmin=0,
                vmax=1,
                cmap="magma",
                interpolation="nearest"
            )

            axes[row+1, col].set_title(
                "Nuclear probability"
            )

            axes[row+2, col].imshow(
                p.max(axis=0),
                vmin=0,
                vmax=1,
                cmap="viridis",
                interpolation="nearest"
            )

            axes[row+2, col].set_title(
                "Maximum probability"
            )

            if self.cfg.plot_class_probabilities:

                axes[row+3, col].imshow(
                    p[0],
                    vmin=0,
                    vmax=1,
                    cmap="magma",
                    interpolation="nearest"
                )

                axes[row+3, col].set_title(
                    "Background probability"
                )

                axes[row+4, col].imshow(
                    p[2],
                    vmin=0,
                    vmax=1,
                    cmap="magma",
                    interpolation="nearest"
                )

                axes[row+4, col].set_title(
                    "Nonnuclear probability"
                )

        for ax in axes.flat:
            ax.axis("off")

        fig.tight_layout()

        if save_path is not None:

            path = Path(save_path)
            path.parent.mkdir(
                parents=True, exist_ok=True
            )

            fig.savefig(
                path,
                dpi=self.cfg.plot_dpi,
                bbox_inches="tight"
            )

        if show:
            plt.show()

        return fig

    def plot_history(
        self,
        result: FixedEnergyResult,
        *,
        save_path=None,
        show=True,
    ):

        import matplotlib.pyplot as plt

        h = result.history

        x = [
            r["iteration"] for r in h
        ]

        has_target = result.target is not None

        fig, axes = plt.subplots(
            1,
            2 if has_target else 1,
            figsize=(
                11 if has_target else 6,
                3.8
            ),
            squeeze=False
        )

        ax = axes[0, 0]

        for name in [
            "total",
            "dirichlet",
            "triple_well",
            "barrier"
        ]:

            ax.plot(
                x,
                [r[name] for r in h],
                label=name
            )

        ax.set(
            xlabel="Iteration",
            ylabel="Energy per pixel",
            title="Fixed energy components"
        )

        ax.legend()

        if has_target:

            ax = axes[0, 1]

            for name in [
                "dice_nuclear",
                "dice_nonnuclear"
            ]:

                ax.plot(
                    x,
                    [r[name] for r in h],
                    label=name
                )

            ax.set(
                xlabel="Iteration",
                ylabel="Hard Dice",
                ylim=(0, 1),
                title="Segmentation vs ground truth"
            )

            ax.legend()

        fig.tight_layout()

        if save_path is not None:

            path = Path(save_path)
            path.parent.mkdir(
                parents=True, exist_ok=True
            )

            fig.savefig(
                path,
                dpi=self.cfg.plot_dpi,
                bbox_inches="tight"
            )

        if show:
            plt.show()

        return fig


# ============================================================
# Optional synthetic demo
# ============================================================

def demo_target(H=80, W=88):
    """Create a simple three-class test segmentation."""

    yy, xx = torch.meshgrid(
        torch.arange(H),
        torch.arange(W),
        indexing="ij"
    )

    y = torch.zeros(
        H, W, dtype=torch.long
    )

    for cx, cy, radius in [
        (23, 22, 9),
        (61, 20, 7),
        (30, 58, 10),
        (64, 57, 8),
    ]:

        rr = (
            (xx-cx).square()
            + (yy-cy).square()
        )

        y[rr < (2*radius)**2] = 2
        y[rr < radius**2] = 1

    return y.unsqueeze(0)


if __name__ == "__main__":

    cfg = FixedEnergyConfig(
        epsilon=0.5,
        lam=1.0,
        barrier_weight=0.1,
        perturb_strength=1.0,
        steps=100,
        save_every=20
    )

    experiment = FixedEnergyExperiment(
        cfg,
        device=(
            "cuda"
            if torch.cuda.is_available()
            else "cpu"
        )
    )

    result = experiment.run(
        target=demo_target(),
        verbose=True
    )

    experiment.plot(result)
    experiment.plot_history(result)
