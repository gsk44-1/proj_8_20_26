
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



# ============================================================
# Backward-compatible extension: optional second two-channel
# boundary family, optional learnable strengths and kernel
# residuals. The original implementation above is retained
# verbatim, including the projected reduced-coordinate metric.
# ============================================================

from dataclasses import replace
from scipy.ndimage import binary_dilation, binary_erosion
import argparse


@dataclass
class Config(FixedEnergyConfig):
    """Extends the original config; defaults reproduce the fixed baseline.

    `barrier_*` names refer to the original separator family and preserve
    the original meanings. Set boundary_weight=0 to exclude the new family.
    """

    boundary_weight: float = 0.0
    boundary_middle_width: int = 3
    boundary_outer_width: int = 2
    boundary_orientations: int = 16
    learn_strengths: bool = False
    learn_residuals: bool = False

    # Larger, independently generated synthetic examples (optional).
    synthetic_size: int = 96
    synthetic_batch_size: int = 1
    nuclei_min: int = 2
    nuclei_max: int = 4
    radius_min: float = 8.0
    radius_max: float = 12.0
    halo_width: float = 3.0
    corruption: str = "mixed"  # none, dilation, erosion, bridge, mixed
    corruption_radius: int = 2
    input_confidence: float = 0.90

    def validate(self):
        super().validate()
        if self.boundary_weight < 0:
            raise ValueError("boundary_weight must be nonnegative")
        if self.boundary_middle_width < 1 or self.boundary_outer_width < 1:
            raise ValueError("Boundary widths must be positive")
        if self.boundary_orientations < 1:
            raise ValueError("boundary_orientations must be >= 1")
        if self.synthetic_size < 16 or self.synthetic_batch_size < 1:
            raise ValueError("Invalid synthetic image size or batch")
        if self.nuclei_min < 1 or self.nuclei_max < self.nuclei_min:
            raise ValueError("Invalid nuclei count")
        if self.radius_min <= 0 or self.radius_max < self.radius_min:
            raise ValueError("Invalid nucleus radii")
        if self.halo_width <= 0 or not 0 < self.input_confidence <= 1:
            raise ValueError("Invalid halo width or confidence")
        if self.corruption not in ("none", "dilation", "erosion", "bridge", "mixed"):
            raise ValueError("Invalid corruption")


def _normalize_rotated_kernel(bank: np.ndarray) -> np.ndarray:
    """Exact same per-orientation centering/normalization as original."""
    bank = bank.copy()
    bank -= bank.mean(axis=(-2, -1), keepdims=True)
    pos = np.maximum(bank, 0.0).sum(axis=(-3, -2, -1), keepdims=True)
    if np.any(pos <= 0):
        raise ValueError("Degenerate filter bank")
    return (bank / pos).astype(np.float32)


def _rotate_canonical(canonical: np.ndarray, count: int, full_circle: bool):
    """SciPy order=1 rotations, retaining exact original settings."""
    rotated = []
    extent = 360.0 if full_circle else 180.0
    for i in range(count):
        degrees = extent * i / count
        if i == 0:
            k = canonical.copy()
        else:
            k = np.stack([
                rotate(channel, degrees, reshape=False, order=1,
                       mode="constant", cval=0.0, prefilter=False)
                for channel in canonical
            ])
        rotated.append(k)
    return _normalize_rotated_kernel(np.stack(rotated))


def _canonical_separator(cfg: Config) -> np.ndarray:
    """Unnormalized original NN|XXX|NN two-channel hard template."""
    band_width = cfg.barrier_width + 2 * cfg.nuclear_band_width
    size = max(band_width, cfg.barrier_length)
    if size % 2 == 0:
        size += 1
    center = size // 2
    left = center - band_width // 2
    nuclear = np.ones(band_width, dtype=np.float64)
    nuclear[cfg.nuclear_band_width:cfg.nuclear_band_width + cfg.barrier_width] = 0
    nonnuclear = 1 - nuclear
    nuclear -= nuclear.mean()
    nonnuclear -= nonnuclear.mean()
    y = np.arange(size, dtype=np.float64) - center
    along = np.exp(-0.5 * (y / cfg.barrier_longitudinal_sigma)**2)
    along[np.abs(y) > cfg.barrier_length // 2] = 0
    base = np.zeros((2, size, size), dtype=np.float64)
    base[0, :, left:left + band_width] = along[:, None] * nuclear
    base[1, :, left:left + band_width] = along[:, None] * nonnuclear
    return base


def make_boundary_filter_bank(cfg: Config) -> Tensor:
    """Directional BG|NON|NUC detector using only NUC and NON input.

    The background preference is represented by contrasts to the eliminated
    background channel. No background convolution input is used.
    """
    mid, side = cfg.boundary_middle_width, cfg.boundary_outer_width
    width = mid + 2 * side
    size = max(width, cfg.barrier_length)
    if size % 2 == 0:
        size += 1
    center = size // 2
    first = center - width // 2
    y = np.arange(size, dtype=np.float64) - center
    along = np.exp(-0.5 * (y / cfg.barrier_longitudinal_sigma)**2)
    along[np.abs(y) > cfg.barrier_length // 2] = 0

    # Phase order in cross-section: background, nonnuclear, nuclear.
    desired = np.zeros((3, width), dtype=np.float64)
    desired[0, :side] = 1
    desired[2, side:side + mid] = 1
    desired[1, side + mid:] = 1
    desired -= desired.mean(axis=0, keepdims=True)
    desired -= desired.mean(axis=1, keepdims=True)

    three = np.zeros((3, size, size), dtype=np.float64)
    three[:, :, first:first + width] = along[None, :, None] * desired[:, None, :]
    # p_background=1-p_nuclear-p_nonnuclear; background template has zero DC
    # after spatial centering, so response equals reduced two-channel contrast.
    two = np.stack((three[1] - three[0], three[2] - three[0]))
    return torch.from_numpy(_rotate_canonical(two, cfg.boundary_orientations, True))


def _scipy_rotation_matrices(size: int, n_angles: int, full_circle: bool) -> Tensor:
    """Linear interpolation matrices exactly implementing scipy.ndimage.rotate.

    Precomputed once, so torch matmul gives gradients w.r.t. canonical
    residual pixels while matching SciPy's forward interpolation.
    """
    count = size * size
    basis = np.eye(count, dtype=np.float64).reshape(count, size, size)
    out = []
    for i in range(n_angles):
        angle = (360.0 if full_circle else 180.0) * i / n_angles
        if i == 0:
            out.append(np.eye(count, dtype=np.float32))
        else:
            rotated = np.stack([
                rotate(b, angle, reshape=False, order=1,
                       mode="constant", cval=0.0, prefilter=False).reshape(-1)
                for b in basis
            ], axis=1)
            out.append(rotated.astype(np.float32))
    return torch.from_numpy(np.stack(out))


class Energy(ModicaMortolaBarrierEnergy):
    """Original MM+separator, optionally extended by a boundary family.

    When boundary_weight=0 and learning flags are False, calls the original
    energy implementation unchanged, including exact old separator kernels.
    """
    def __init__(self, cfg: Config):
        super().__init__(cfg)
        self.cfg = cfg
        self.register_buffer("boundary_kernels", make_boundary_filter_bank(cfg))
        if cfg.learn_strengths:
            self.separator_weight_parameter = nn.Parameter(torch.tensor(float(cfg.barrier_weight)))
            self.boundary_weight_parameter = nn.Parameter(torch.tensor(float(cfg.boundary_weight)))
        if cfg.learn_residuals:
            self.separator_residual = nn.Parameter(torch.zeros_like(_canonical_separator_tensor(cfg)))
            boundary_size = self.boundary_kernels.shape[-1]
            self.boundary_residual = nn.Parameter(torch.zeros(2, boundary_size, boundary_size))
            self.register_buffer("separator_rotmat", _scipy_rotation_matrices(
                self.barrier_kernels.shape[-1], cfg.barrier_orientations, False))
            self.register_buffer("boundary_rotmat", _scipy_rotation_matrices(
                boundary_size, cfg.boundary_orientations, True))

    def strength(self, family="separator") -> Tensor:
        if family not in ("separator", "boundary"):
            raise ValueError("family must be separator or boundary")
        param = "separator_weight_parameter" if family == "separator" else "boundary_weight_parameter"
        if hasattr(self, param):
            return getattr(self, param).clamp_min(0)
        value = self.cfg.barrier_weight if family == "separator" else self.cfg.boundary_weight
        return self.barrier_kernels.new_tensor(float(value))

    def kernels(self, family="separator") -> Tensor:
        if family not in ("separator", "boundary"):
            raise ValueError("family must be separator or boundary")
        base = self.barrier_kernels if family == "separator" else self.boundary_kernels
        residual_name = "separator_residual" if family == "separator" else "boundary_residual"
        if not hasattr(self, residual_name):
            return base
        residual = getattr(self, residual_name)
        mat = self.separator_rotmat if family == "separator" else self.boundary_rotmat
        # mat [orientations, pixels_out, pixels_in], residual [2, pixels_in]
        rotated = torch.einsum("oij,cj->oci", mat.to(residual.dtype), residual.flatten(1))
        rotated = rotated.reshape_as(base)
        full = base + rotated
        full = full - full.mean(dim=(-2, -1), keepdim=True)
        return full / full.clamp_min(0).sum(dim=(1, 2, 3), keepdim=True).clamp_min(1e-12)

    @staticmethod
    def _convolve_periodic(p: Tensor, kernels: Tensor) -> Tensor:
        radius = kernels.shape[-1] // 2
        if radius < min(p.shape[-2:]):
            return F.conv2d(F.pad(p, (radius,) * 4, mode="circular"), kernels.to(p.dtype))
        # Same fallback as the original for tiny images.
        out = p.new_zeros((p.shape[0], kernels.shape[0], *p.shape[-2:]))
        for row in range(kernels.shape[-2]):
            for col in range(kernels.shape[-1]):
                shifted = torch.roll(p, (radius-row, radius-col), dims=(-2, -1))
                out = out + torch.einsum("bchw,oc->bohw", shifted,
                                         kernels[:, :, row, col].to(p.dtype))
        return out

    def barrier_responses(self, p: Tensor) -> Tensor:
        if not self.cfg.learn_residuals:
            return super().barrier_responses(p)
        return self._convolve_periodic(p[:, 1:3], self.kernels("separator"))

    def responses(self, p: Tensor, family="separator") -> Tensor:
        if family == "separator":
            return self.barrier_responses(p)
        if family == "boundary":
            return self._convolve_periodic(p[:, 1:3], self.kernels("boundary"))
        raise ValueError("Unknown family")

    def energy_terms_reduced(self, q: Tensor) -> dict[str, Tensor]:
        terms = super().energy_terms_reduced(q)
        if self.cfg.learn_strengths:
            p = reduced_to_prob(q)
            response = F.relu(self.barrier_responses(p).amax(dim=1))
            terms["barrier"] = -0.5 * self.strength("separator") * response.square().flatten(1).sum(1)
        weight = self.strength("boundary")
        if self.cfg.boundary_weight > 0 or self.cfg.learn_strengths:
            p = reduced_to_prob(q)
            response = F.relu(self.responses(p, "boundary").amax(dim=1))
            terms["boundary"] = -0.5 * weight * response.square().flatten(1).sum(1)
        else:
            terms["boundary"] = torch.zeros_like(terms["barrier"])
        return terms

    def geometry(self) -> dict:
        return {
            "separator_strength": float(self.strength("separator").detach()),
            "boundary_strength": float(self.strength("boundary").detach()),
            "separator_angles": self.cfg.barrier_orientations,
            "boundary_angles": self.cfg.boundary_orientations,
            "separator_kernel_shape": tuple(self.barrier_kernels.shape),
            "boundary_kernel_shape": tuple(self.boundary_kernels.shape),
            "learn_strengths": self.cfg.learn_strengths,
            "learn_residuals": self.cfg.learn_residuals,
        }


def _canonical_separator_tensor(cfg):
    return torch.from_numpy(_canonical_separator(cfg).astype(np.float32))


class Experiment(FixedEnergyExperiment):
    """Original reduced-coordinate minimizer, now applied to extended Energy.

    For fixed kernels it retains the original backtracking algorithm exactly.
    For future differentiable training use `projected_step` below instead.
    """
    def __init__(self, cfg=None, device="cpu"):
        cfg = Config() if cfg is None else cfg
        super().__init__(cfg, device=device)
        self.energy = Energy(cfg).to(self.device)


def projected_step(model: Energy, p: Tensor, step_size: float = 0.05,
                   differentiable: bool = True) -> Tensor:
    """Differentiable, fixed-step *original reduced-metric* update.

    No line search: suitable for future unrolled learning. Use Experiment.run
    for the faithful original fixed-energy backtracking evaluation.
    """
    with torch.enable_grad():
        q = prob_to_reduced(p)
        if not q.requires_grad:
            q = q.detach().requires_grad_(True)
        energy = model.energy_reduced(q).sum()
        grad, = torch.autograd.grad(energy, q, create_graph=differentiable)
        next_q = project_triangle(q - step_size * grad)
        result = reduced_to_prob(next_q)
    return result if differentiable else result.detach()


def rollout(model: Energy, p0: Tensor, steps: int = 40,
            step_size: float = 0.05, differentiable: bool = False):
    """Return [p0, p1, ..., p_steps] with the original metric and fixed step."""
    p = p0
    states = [p.detach().clone() if not differentiable else p]
    for _ in range(steps):
        p = projected_step(model, p, step_size, differentiable=differentiable)
        states.append(p)
    return states


def _disk(radius):
    r = max(int(radius), 0)
    y, x = np.ogrid[-r:r+1, -r:r+1]
    return x*x + y*y <= r*r


def synthetic_batch(cfg: Config, seed=None, device="cpu"):
    """Larger ellipse nuclei and halos; returns (p0, clean, target).

    Corruptions act on nuclei while preserving nonnuclear halo labels.
    To reproduce the original spectral-noise experiment use
    `initial_probabilities(target, cfg)` instead of this function.
    """
    cfg.validate()
    rng = np.random.default_rng(cfg.seed if seed is None else seed)
    B, S = cfg.synthetic_batch_size, cfg.synthetic_size
    labels = np.zeros((B, S, S), dtype=np.int64)
    corrupted = np.zeros_like(labels)
    yy, xx = np.mgrid[:S, :S]
    for b in range(B):
        nuc = np.zeros((S, S), dtype=bool)
        halo = np.zeros((S, S), dtype=bool)
        centers = []
        for _ in range(int(rng.integers(cfg.nuclei_min, cfg.nuclei_max+1))):
            for attempt in range(250):
                rx = rng.uniform(cfg.radius_min, cfg.radius_max)
                ry = rx * rng.uniform(0.85, 1.15)
                pad = cfg.halo_width + 2
                cx = rng.uniform(rx+pad, S-rx-pad)
                cy = rng.uniform(ry+pad, S-ry-pad)
                angle = rng.uniform(0, np.pi)
                dx, dy = xx-cx, yy-cy
                u = np.cos(angle)*dx + np.sin(angle)*dy
                v = -np.sin(angle)*dx + np.cos(angle)*dy
                core = (u/rx)**2 + (v/ry)**2 <= 1
                outer = (u/(rx+cfg.halo_width))**2 + (v/(ry+cfg.halo_width))**2 <= 1
                if not np.any(outer & halo):
                    nuc |= core
                    halo |= outer
                    centers.append((cx,cy))
                    break
        labels[b, halo] = 2
        labels[b, nuc] = 1
        mode = cfg.corruption
        if mode == "mixed":
            mode = rng.choice(["dilation", "erosion", "bridge"])
        changed = nuc.copy()
        if mode == "dilation":
            changed = binary_dilation(nuc, structure=_disk(cfg.corruption_radius))
        elif mode == "erosion":
            changed = binary_erosion(nuc, structure=_disk(cfg.corruption_radius))
        elif mode == "bridge" and len(centers) >= 2:
            ia, ib = rng.choice(len(centers), size=2, replace=False)
            ax, ay = centers[ia]
            bx, by = centers[ib]
            for t in np.linspace(0, 1, 100):
                cx, cy = ax*(1-t)+bx*t, ay*(1-t)+by*t
                changed |= (xx-cx)**2+(yy-cy)**2 <= cfg.corruption_radius**2
        corrupted[b] = labels[b]
        corrupted[b, changed] = 1
        corrupted[b, nuc & ~changed] = 2
    target = torch.from_numpy(labels).long().to(device)
    wrong = torch.from_numpy(corrupted).long().to(device)
    clean = F.one_hot(target, num_classes=3).permute(0, 3, 1, 2).float()
    confidence = cfg.input_confidence
    p0 = (F.one_hot(wrong, num_classes=3).permute(0, 3, 1, 2).float()
          * (confidence - (1-confidence)/2) + (1-confidence)/2)
    return p0, clean, target


# ============================================================
# Optional learning: truncated unrolling of the exact original
# reduced-coordinate projected gradient update.
# The fixed-energy Experiment above remains unchanged.
# ============================================================

import json
import copy
import random
from dataclasses import asdict


@dataclass
class TrainConfig:
    """Parameters for the opt-in synthetic training experiment.

    Defaults are deliberately modest; for long-range optimization, increase
    updates and warmup lengths after verifying the fixed baseline.
    """
    updates: int = 150
    learning_rate: float = 0.002
    batch_size: int = 2
    training_samples: int = 48
    validation_samples: int = 12
    warmup_steps: tuple[int, ...] = (0, 10, 30, 60, 120)
    differentiable_steps: int = 8
    step_size: float = 0.02
    nuclear_loss_weight: float = 3.0
    nonnuclear_loss_weight: float = 2.0
    background_loss_weight: float = 1.0
    late_background_weight: float = 0.5
    stationarity_weight: float = 0.05
    residual_penalty_weight: float = 0.01
    strength_drift_weight: float = 0.001
    gradient_clip_norm: float = 5.0
    eval_every: int = 25
    validation_steps: int = 80
    long_validation_steps: int = 400
    long_validation_examples: int = 1
    seed: int = 123
    output_dir: str = "mm_fixed_training"
    device: str = "cpu"
    learn_strengths: bool = True
    learn_residuals: bool = False

    def validate(self):
        if self.updates < 1 or self.training_samples < 1 or self.validation_samples < 1:
            raise ValueError("Need positive update and data counts")
        if self.batch_size < 1 or self.differentiable_steps < 1:
            raise ValueError("Need positive batch size and unroll length")
        if self.step_size <= 0 or self.learning_rate <= 0:
            raise ValueError("Step sizes and learning rate must be positive")
        if not self.warmup_steps or min(self.warmup_steps) < 0:
            raise ValueError("warmup_steps must contain nonnegative integers")
        if self.eval_every < 1 or self.validation_steps < 1 or self.long_validation_steps < 0:
            raise ValueError("Invalid validation interval or horizon")
        if self.long_validation_examples < 0:
            raise ValueError("long_validation_examples must be nonnegative")
        if not (self.learn_strengths or self.learn_residuals):
            raise ValueError("Enable at least one trainable parameter family")
        if min(self.nuclear_loss_weight, self.nonnuclear_loss_weight,
               self.background_loss_weight, self.late_background_weight,
               self.stationarity_weight, self.residual_penalty_weight,
               self.strength_drift_weight) < 0:
            raise ValueError("Loss coefficients must be nonnegative")


def _weighted_probability_mse(p: Tensor, clean: Tensor, cfg: TrainConfig) -> Tensor:
    weights = p.new_tensor([cfg.background_loss_weight,
                            cfg.nuclear_loss_weight,
                            cfg.nonnuclear_loss_weight]).view(1, 3, 1, 1)
    return ((p - clean).square() * weights).mean()


def _background_false_positive_loss(p: Tensor, target: Tensor) -> Tensor:
    mask = (target == 0).to(p.dtype)
    non_bg = p[:, 1].square() + p[:, 2].square()
    return (mask * non_bg).sum() / mask.sum().clamp_min(1)


def _stationarity_loss(model: Energy, clean: Tensor, step_size: float) -> Tensor:
    """Projected-gradient residual; respects simplex vertex constraints."""
    moved = projected_step(model, clean, step_size, differentiable=True)
    return (moved - clean).square().mean()


def _residual_penalty(model: Energy) -> Tensor:
    penalty = next(model.parameters()).new_zeros(())
    for name in ('separator_residual', 'boundary_residual'):
        residual = getattr(model, name, None)
        if residual is not None:
            penalty = penalty + residual.square().mean()
    return penalty


def _sample_synthetic_dataset(cfg: Config, n: int, seed_base: int,
                              device='cpu'):
    """Fixed synthetic samples; held-out seeds differ from training seeds."""
    single_cfg = replace(cfg, synthetic_batch_size=1)
    inputs, cleans, labels = [], [], []
    for i in range(n):
        p, c, y = synthetic_batch(single_cfg, seed=seed_base+i, device='cpu')
        inputs.append(p)
        cleans.append(c)
        labels.append(y)
    return (torch.cat(inputs).to(device), torch.cat(cleans).to(device),
            torch.cat(labels).to(device))


def _rollout_end(model: Energy, p: Tensor, steps: int, step_size: float):
    """Fixed-step evaluation without tracking a long autograd graph."""
    for _ in range(steps):
        p = projected_step(model, p, step_size, differentiable=False)
    return p


def _class_dice(p: Tensor, target: Tensor, k: int) -> float:
    pred = (p.argmax(dim=1) == k)
    true = (target == k)
    num = (2 * (pred & true).sum()).item()
    den = (pred.sum() + true.sum()).item()
    return float(num / den) if den else float('nan')


def evaluate_training(model: Energy, dataset, tc: TrainConfig,
                      horizons=None, max_samples=None):
    """Held-out hard Dice, MSE, false-positive fraction for requested horizons.

    The long horizon is deliberately checked separately from short-horizon
    training; lower short-term MSE does not imply stable minimizers.
    """
    p0, clean, target = dataset
    if max_samples is not None:
        p0, clean, target = p0[:max_samples], clean[:max_samples], target[:max_samples]
    horizons = (0, tc.validation_steps) if horizons is None else tuple(horizons)
    current = p0.detach()
    last_step = 0
    results = {}
    for n in sorted(set(horizons)):
        current = _rollout_end(model, current, n-last_step, tc.step_size)
        last_step = n
        with torch.no_grad():
            hard = current.argmax(dim=1)
            bg = target == 0
            fpr = ((hard != 0) & bg).sum().float() / bg.sum().clamp_min(1)
            results[str(n)] = {
                'mse': float(_weighted_probability_mse(current, clean, tc)),
                'dice_nuclear': _class_dice(current, target, 1),
                'dice_nonnuclear': _class_dice(current, target, 2),
                'background_false_positive_fraction': float(fpr),
                'mean_confidence': float(current.max(dim=1).values.mean()),
            }
    return results


def _save_training_preview(model: Energy, val_data, tc: TrainConfig,
                           output_dir: Path, suffix: str):
    import matplotlib.pyplot as plt
    p0, clean, target = (v[:1] for v in val_data)
    horizons = sorted(set((0, tc.validation_steps, tc.long_validation_steps)))
    current = p0
    last = 0
    states = {}
    for t in horizons:
        current = _rollout_end(model, current, t-last, tc.step_size)
        last = t
        states[t] = current.detach().cpu()
    ncols = len(horizons) + 1
    fig, axes = plt.subplots(3, ncols, figsize=(2.7*ncols, 8), squeeze=False)
    cols = [('Target', clean.detach().cpu())] + [(f'Step {t}', states[t]) for t in horizons]
    for j, (label, probs) in enumerate(cols):
        maps = [probs[0].argmax(0), probs[0, 1], probs[0, 2]]
        for i in range(3):
            axes[i, j].imshow(maps[i].numpy(),
                              cmap='viridis' if i == 0 else 'magma',
                              vmin=0, vmax=2 if i == 0 else 1,
                              interpolation='nearest')
            axes[i, j].axis('off')
        axes[0, j].set_title(label)
    for ax, title in zip(axes[:, 0], ['Hard segmentation', 'Nuclear', 'Nonnuclear']):
        ax.set_ylabel(title)
    fig.tight_layout()
    fig.savefig(output_dir / f'{suffix}_preview.png', dpi=125)
    plt.close(fig)


def train(cfg: Config | None = None, training_cfg: TrainConfig | None = None):
    """Learn strengths and/or residual kernels with randomized truncated unrolling.

    Returns (trained_model, history). Saves best model, metrics, and figures.

    Validation is held-out by deterministic seed range. The *fixed* original
    `Experiment.run` with backtracking remains available for final evaluation.
    """
    tc = TrainConfig() if training_cfg is None else training_cfg
    tc.validate()
    cfg = Config() if cfg is None else cfg
    cfg = replace(cfg, learn_strengths=tc.learn_strengths,
                  learn_residuals=tc.learn_residuals)
    cfg.validate()
    torch.manual_seed(tc.seed)
    np.random.seed(tc.seed)
    random.seed(tc.seed)
    out = Path(tc.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    device = torch.device(tc.device)
    model = Energy(cfg).to(device)
    params = [x for x in model.parameters() if x.requires_grad]
    if not params:
        raise ValueError('No trainable parameters')
    optimizer = torch.optim.Adam(params, lr=tc.learning_rate)
    initial_strengths = {
        k: model.strength(k).detach().clone() for k in ('separator', 'boundary')
    }
    train_data = _sample_synthetic_dataset(cfg, tc.training_samples,
                                          tc.seed + 1000, device)
    val_data = _sample_synthetic_dataset(cfg, tc.validation_samples,
                                        tc.seed + 100000, device)
    history = []
    best_state = copy.deepcopy(model.state_dict())

    # Keep the fixed baseline if training fails to improve held-out MSE.
    baseline = evaluate_training(model, val_data, tc)
    best_score = baseline[str(tc.validation_steps)]['mse']
    torch.save({'model_state': best_state,
                'config': asdict(cfg), 'training_config': asdict(tc),
                'update': 0, 'val_mse': best_score}, out / 'best_model.pt')
    print('Initial validation:', baseline)
    _save_training_preview(model, val_data, tc, out, 'initial')
    generator = torch.Generator(device='cpu')
    generator.manual_seed(tc.seed + 54321)

    for update in range(1, tc.updates + 1):
        inds = torch.randint(0, tc.training_samples, (tc.batch_size,),
                             generator=generator).to(device)
        p0, clean, target = (v[inds] for v in train_data)
        warmup = tc.warmup_steps[int(torch.randint(
            len(tc.warmup_steps), (1,), generator=generator).item())]

        # Discard derivatives through the warm-up trajectory. This is TBPTT;
        # it does *not* compute the exact gradient of the full-length rollout.
        p = _rollout_end(model, p0, warmup, tc.step_size).detach()

        # Short differentiable rollout of the ORIGINAL reduced-coordinate PGD.
        for _ in range(tc.differentiable_steps):
            p = projected_step(model, p, tc.step_size, differentiable=True)

        mse = _weighted_probability_mse(p, clean, tc)
        bg_loss = _background_false_positive_loss(p, target)
        stationarity = (_stationarity_loss(model, clean, tc.step_size)
                        if tc.stationarity_weight else mse.new_zeros(()))
        reg = (_residual_penalty(model) if tc.learn_residuals
               else mse.new_zeros(()))
        drift = sum((model.strength(k) - initial_strengths[k]).square()
                    for k in ('separator', 'boundary'))
        loss = (mse + tc.late_background_weight * bg_loss
                + tc.stationarity_weight * stationarity
                + tc.residual_penalty_weight * reg
                + tc.strength_drift_weight * drift)

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(params, tc.gradient_clip_norm)
        if not torch.isfinite(loss) or not torch.isfinite(grad_norm):
            raise FloatingPointError(f'Nonfinite training loss or gradients at update {update}')
        optimizer.step()
        # The original energy requires nonnegative strengths. Keep the
        # underlying parameters feasible, not merely their forward clamped values.
        with torch.no_grad():
            for name in ('separator_weight_parameter', 'boundary_weight_parameter'):
                param = getattr(model, name, None)
                if param is not None:
                    param.clamp_(min=0.)

        record = {'update': update, 'warmup': warmup,
                  'loss': float(loss.detach()), 'mse': float(mse.detach()),
                  'background_loss': float(bg_loss.detach()),
                  'stationarity': float(stationarity.detach()),
                  'residual_penalty': float(reg.detach()),
                  'strength_drift': float(drift.detach()),
                  'grad_norm': float(grad_norm.detach()),
                  'separator_strength': float(model.strength('separator').detach()),
                  'boundary_strength': float(model.strength('boundary').detach())}

        if update == 1 or update % tc.eval_every == 0 or update == tc.updates:
            validation = evaluate_training(model, val_data, tc)
            # Main selection criterion: held-out probability loss at the
            # requested evaluation horizon, not the training-step loss.
            score = validation[str(tc.validation_steps)]['mse']
            record['validation'] = validation
            if score < best_score:
                best_score = score
                best_state = copy.deepcopy(model.state_dict())
                torch.save({'model_state': best_state,
                            'config': asdict(cfg), 'training_config': asdict(tc),
                            'update': update, 'val_mse': score},
                           out / 'best_model.pt')
            print(f"Update {update:4d}/{tc.updates}  "
                  f"loss={record['loss']:.6f} "
                  f"valMSE={score:.6f} "
                  f"S={record['separator_strength']:.4g} "
                  f"B={record['boundary_strength']:.4g}")
        history.append(record)

    model.load_state_dict(best_state)
    final_val = evaluate_training(model, val_data, tc)
    long_val = None
    if tc.long_validation_steps > 0 and tc.long_validation_examples > 0:
        long_val = evaluate_training(
            model, val_data, tc,
            horizons=(0, tc.validation_steps, tc.long_validation_steps),
            max_samples=tc.long_validation_examples)
    _save_training_preview(model, val_data, tc, out, 'trained')
    with (out / 'history.json').open('w') as f:
        json.dump({'baseline': baseline, 'history': history,
                   'best_validation': final_val, 'long_validation': long_val,
                   'energy_config': asdict(cfg), 'training_config': asdict(tc)},
                  f, indent=2)
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    axes[0].plot([r['update'] for r in history],
                 [r['loss'] for r in history], label='Training loss')
    vals = [(r['update'], r['validation'][str(tc.validation_steps)]['mse'])
            for r in history if 'validation' in r]
    if vals:
        axes[0].plot([v[0] for v in vals], [v[1] for v in vals],
                     label='Validation MSE')
    axes[0].legend(); axes[0].set_xlabel('Optimizer update'); axes[0].set_title('Loss')
    axes[1].plot([r['update'] for r in history],
                 [r['separator_strength'] for r in history], label='Separator')
    axes[1].plot([r['update'] for r in history],
                 [r['boundary_strength'] for r in history], label='Boundary')
    axes[1].legend(); axes[1].set_xlabel('Optimizer update'); axes[1].set_title('Learned strengths')
    fig.tight_layout()
    fig.savefig(out / 'training_history.png', dpi=140)
    plt.close(fig)
    print('Best validation:', final_val)
    print('Long-horizon validation:', long_val)
    print('Saved outputs to:', out.resolve())
    return model, history


def load_trained(path: str, device: str = 'cpu'):
    """Load a saved training checkpoint and its original energy configuration."""
    checkpoint = torch.load(path, map_location=device, weights_only=False)
    cfg = Config(**checkpoint['config'])
    model = Energy(cfg).to(device)
    model.load_state_dict(checkpoint['model_state'])
    model.eval()
    return model, cfg


def main():
    parser = argparse.ArgumentParser(description="Original fixed MM solver with optional learnable filter extension")
    parser.add_argument("--train", action="store_true", help="Run the optional synthetic-data trainer")
    parser.add_argument("--train-updates", type=int, default=150)
    parser.add_argument("--train-out", default="mm_fixed_training")
    parser.add_argument("--learning-rate", type=float, default=0.002)
    parser.add_argument("--learn-residuals", action="store_true")
    parser.add_argument("--out", default="mm_fixed_compatible_demo.png")
    parser.add_argument("--steps", type=int, default=100)
    parser.add_argument("--epsilon", type=float, default=0.5)
    parser.add_argument("--lam", type=float, default=1.0)
    parser.add_argument("--boundary-weight", type=float, default=0.0)
    parser.add_argument("--barrier-weight", type=float, default=0.1)
    parser.add_argument("--synthetic", action="store_true", help="Use new large-cell synthetic data rather than original demo")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()
    cfg = Config(epsilon=args.epsilon, lam=args.lam, boundary_weight=args.boundary_weight,
                 barrier_weight=args.barrier_weight, steps=args.steps, save_every=max(1,args.steps//8))
    if args.train:
        tc = TrainConfig(updates=args.train_updates, learning_rate=args.learning_rate,
                         output_dir=args.train_out, device=args.device,
                         learn_strengths=True, learn_residuals=args.learn_residuals)
        train(cfg, tc)
        return
    experiment = Experiment(cfg, device=args.device)
    if args.synthetic:
        p0, clean, target = synthetic_batch(cfg, seed=cfg.seed, device=args.device)
        result = experiment.run(target=target, p0=p0, verbose=True)
    else:
        result = experiment.run(target=demo_target(), verbose=True)
    experiment.plot(result, save_path=args.out)
    experiment.plot_history(result, save_path=str(Path(args.out).with_name(Path(args.out).stem + "_history.png")))
    print("Geometry:", experiment.energy.geometry())


if __name__ == "__main__":
    main()
