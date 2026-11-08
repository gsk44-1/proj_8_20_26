"""Self-contained three-class fixed-energy segmentation experiment.

No learned network or training module is needed.

Classes in probability tensors [B,3,H,W]: background, nuclear, nonnuclear.
Targets [B,H,W] use 0/1/2 (and optionally ignore_index).

Energy:
  E(p) = lam*epsilon*sum_{x,k,d} |D_d p_k(x)|^2
       + lam/epsilon*sum_x W(p(x))
       - log_weight/2 * sum_x [sigma^2 Delta(G_sigma*p_nuclear)(x)]^2

  W(p) = p0^2*p1^2 + p0^2*p2^2 + p1^2*p2^2 + cubic_weight*p0*p1*p2.

This uses periodic forward differences for the Modica--Mortola Dirichlet energy.
The reduced-coordinate diffusion gradient matches the coupled Laplacian in
triple_well_pairwise(5).py: 2*lam*epsilon*(2*Delta(u)+Delta(w),
Delta(u)+2*Delta(w)) for the negative gradient flow, with u=nuclear,
w=nonnuclear, and background=1-u-w.

The LoG uses Gaussian convolution and the 5-point discrete Laplacian with
CIRCULAR boundaries. Everything is differentiated from ONE scalar energy, so
no implementation-dependent mismatch between numerical forces and energy.

Projected gradient descent acts in (u,w) reduced coordinates with Euclidean
projection onto the triangle. Backtracking optionally enforces monotone energy
on each step. This is the same energy as the original splitting prototype,
not the exact same numerical splitting algorithm.

Requires numpy, scipy, torch; matplotlib for plotting.

Run a standalone example:
    python fixed_energy_standalone.py --save-dir ./fixed_energy_demo

For your own target tensor:
    from fixed_energy_standalone import FixedEnergyConfig, FixedEnergyExperiment
    cfg = FixedEnergyConfig(perturb_strength=1.0, epsilon=0.5, log_weight=0.01)
    experiment = FixedEnergyExperiment(cfg, device='cuda')
    result = experiment.run(target=target, verbose=True)
    experiment.plot(result)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Union
import argparse
import math

import numpy as np
import torch
from torch import Tensor, nn
from torch.nn import functional as F
from scipy import fft
from scipy.ndimage import gaussian_filter


@dataclass
class FixedEnergyConfig:
    # Three-phase Modica--Mortola energy (matches earlier reduced-coordinate model)
    epsilon: float = 0.2
    lam: float = 1.0
    well_cubic_weight: float = 1.0

    # Optional nuclear LoG contribution. log_sigma means Gaussian SPATIAL scale.
    log_sigma: float = 5.0
    log_weight: float = 0.01
    log_kernel_radius_sigma: float = 4.0

    # Projected gradient descent (not learned; independent of training models)
    steps: int = 200
    step_size: float = 0.05
    backtracking: bool = True
    backtracking_factor: float = 0.5
    max_backtracks: int = 20
    min_step_size: float = 1e-10
    save_every: int = 20
    verbose_every: int = 20

    # Corrupted ground truth -> synthetic initial logits -> softmax.
    seed: int = 42
    base_logit_magnitude: float = 1.5
    perturb_strength: float = 1.0
    perturb_probability: float = 1.0
    gt_blur_sigma: float = 0.0
    spectral_bdwidth: float = 0.9
    spectral_wavelengths: tuple[float, ...] = (250., 150., 100., 50., 25., 12.5, 1.)
    spectral_amplitudes: tuple[float, ...] = (15., 12., 8., 2., 1., 0.5, 0.3)
    spectral_amplitude_jitter: tuple[float, float] = (0.6, 1.5)
    ignore_index: int = -100

    # Visualization
    max_plot_snapshots: int = 8
    plot_class_probabilities: bool = False  # If True, show all 3 probability channels
    plot_dpi: int = 140

    def validate(self):
        if self.epsilon <= 0 or self.lam < 0 or self.well_cubic_weight < 0:
            raise ValueError('epsilon>0, lam>=0, well_cubic_weight>=0 required')
        if self.log_sigma <= 0 or self.log_weight < 0 or self.log_kernel_radius_sigma <= 0:
            raise ValueError('log_sigma>0, log_weight>=0, log_kernel_radius_sigma>0 required')
        if self.steps < 0 or self.step_size <= 0 or self.save_every < 1 or self.verbose_every < 1:
            raise ValueError('steps>=0, step_size>0, save_every>=1, verbose_every>=1')
        if not 0 < self.backtracking_factor < 1 or self.max_backtracks < 0 or self.min_step_size <= 0:
            raise ValueError('invalid backtracking settings')
        if self.base_logit_magnitude <= 0 or self.perturb_strength < 0:
            raise ValueError('base_logit_magnitude>0, perturb_strength>=0 required')
        if not 0 <= self.perturb_probability <= 1 or self.gt_blur_sigma < 0 or self.spectral_bdwidth <= 0:
            raise ValueError('invalid perturbation settings')
        if len(self.spectral_wavelengths) == 0 or len(self.spectral_wavelengths) != len(self.spectral_amplitudes):
            raise ValueError('nonempty, matching spectral wavelengths/amplitudes required')
        if any(x <= 0 for x in self.spectral_wavelengths) or any(x < 0 for x in self.spectral_amplitudes):
            raise ValueError('invalid spectral wavelengths/amplitudes')
        lo, hi = self.spectral_amplitude_jitter
        if lo < 0 or hi < lo:
            raise ValueError('invalid amplitude jitter')
        if self.max_plot_snapshots < 1:
            raise ValueError('max_plot_snapshots must be >=1')


def project_triangle(q: Tensor) -> Tensor:
    """Project [B,2,H,W] onto {(u,w): u>=0, w>=0, u+w<=1}.

    Euclidean metric in the INDEPENDENT (u,w) coordinates, matching source.
    """
    if q.ndim != 4 or q.shape[1] != 2:
        raise ValueError('Expected q [B,2,H,W]')
    u, w = q[:, 0:1], q[:, 1:2]
    on_u0 = torch.cat((torch.zeros_like(u), w.clamp(0, 1)), dim=1)
    on_w0 = torch.cat((u.clamp(0, 1), torch.zeros_like(w)), dim=1)
    t = ((u - w + 1) / 2).clamp(0, 1)
    on_background0 = torch.cat((t, 1 - t), dim=1)
    inside = (u >= 0) & (w >= 0) & (u + w <= 1)
    candidates = torch.stack((on_u0, on_w0, on_background0), dim=1)
    distances = (candidates - q.unsqueeze(1)).square().sum(dim=2, keepdim=True)
    ix = distances.argmin(dim=1, keepdim=True)
    closest = candidates.gather(1, ix.expand(-1, -1, 2, -1, -1)).squeeze(1)
    return torch.where(inside.expand_as(q), q, closest)


def reduced_to_prob(q: Tensor) -> Tensor:
    u, w = q[:, 0:1], q[:, 1:2]
    return torch.cat((1 - u - w, u, w), dim=1)


def prob_to_reduced(p: Tensor) -> Tensor:
    if p.ndim != 4 or p.shape[1] != 3:
        raise ValueError('Expected p [B,3,H,W]')
    return project_triangle(p[:, 1:3])


def periodic_laplacian(x: Tensor) -> Tensor:
    return (torch.roll(x, 1, -2) + torch.roll(x, -1, -2)
            + torch.roll(x, 1, -1) + torch.roll(x, -1, -1) - 4*x)


class ModicaMortolaLoGEnergy(nn.Module):
    """Per-image fixed energy E: [B,3,H,W] -> [B]. No trainable weights."""

    def __init__(self, cfg: FixedEnergyConfig):
        super().__init__()
        cfg.validate()
        self.cfg = cfg
        radius = max(1, math.ceil(cfg.log_kernel_radius_sigma * cfg.log_sigma))
        t = torch.arange(-radius, radius + 1, dtype=torch.float32)
        kernel = torch.exp(-0.5 * (t / cfg.log_sigma)**2)
        kernel /= kernel.sum()
        self.register_buffer('gauss_x', kernel.reshape(1, 1, 1, -1))
        self.register_buffer('gauss_y', kernel.reshape(1, 1, -1, 1))

    def log_response(self, nuclear: Tensor) -> Tensor:
        """K_sigma u = sigma^2 Delta (G_sigma * u), circular boundary."""
        if nuclear.ndim != 4 or nuclear.shape[1] != 1:
            raise ValueError('Expected nuclear [B,1,H,W]')
        # torch.roll Gaussian convolution handles even very small images with
        # radius larger than their dimension (unlike circular F.pad).
        out = torch.zeros_like(nuclear)
        rx = self.gauss_x.shape[-1] // 2
        gx = self.gauss_x.flatten()
        for k, weight in enumerate(gx):
            out = out + weight * torch.roll(nuclear, k-rx, dims=-1)
        smoothed = torch.zeros_like(out)
        ry = self.gauss_y.shape[-2] // 2
        gy = self.gauss_y.flatten()
        for k, weight in enumerate(gy):
            smoothed = smoothed + weight * torch.roll(out, k-ry, dims=-2)
        return (self.cfg.log_sigma**2) * periodic_laplacian(smoothed)

    def energy_terms_reduced(self, q: Tensor) -> dict[str, Tensor]:
        p = reduced_to_prob(q)
        u, w, z = p[:, 1], p[:, 2], p[:, 0]
        W = (u.square()*w.square() + u.square()*z.square()
             + w.square()*z.square() + self.cfg.well_cubic_weight*u*w*z)
        well = (self.cfg.lam / self.cfg.epsilon) * W.flatten(1).sum(1)
        dx = torch.roll(p, -1, -1) - p
        dy = torch.roll(p, -1, -2) - p
        smooth = (self.cfg.lam * self.cfg.epsilon) * (
            dx.square().flatten(1).sum(1) + dy.square().flatten(1).sum(1)
        )
        if self.cfg.log_weight:
            response = self.log_response(p[:, 1:2])
            log_term = -0.5*self.cfg.log_weight*response.square().flatten(1).sum(1)
        else:
            log_term = well.new_zeros(well.shape)
        return {'dirichlet': smooth, 'triple_well': well, 'nuclear_log': log_term}

    def energy_terms(self, p: Tensor) -> dict[str, Tensor]:
        return self.energy_terms_reduced(prob_to_reduced(p))

    def energy_reduced(self, q: Tensor) -> Tensor:
        return sum(self.energy_terms_reduced(q).values())

    def forward(self, p: Tensor) -> Tensor:
        return sum(self.energy_terms(p).values())


def spectral_noise(shape: tuple[int,int], bands: np.ndarray, width: float, rng: np.random.Generator) -> np.ndarray:
    """Exact radial-band FFT noise construction from the original trainer."""
    h, w = shape
    white = rng.standard_normal(shape)
    ky = fft.fftfreq(h)[:, None]
    kx = fft.rfftfreq(w)[None, :]
    k = np.hypot(kx, ky)
    filt = np.zeros_like(k)
    for amplitude, wavelength in bands:
        filt += amplitude * np.exp(-0.5 * ((k - 1/wavelength) / (width/wavelength))**2)
    noise = fft.irfftn(fft.rfftn(white) * filt, s=shape).real
    noise -= noise.mean()
    noise /= max(float(np.max(np.abs(noise))), 1e-8)
    return noise.astype(np.float32)


def synthetic_logits(target: Tensor, cfg: FixedEnergyConfig, *, seed: Optional[int] = None) -> Tensor:
    """Generate corrupted logits from integer ground truth [B,H,W].

    Identical recipe to train_probability_energy.py, without importing it.
    Uses independent spectral fields in each image/class. The seed is that of
    the NumPy RNG directly (so it need not reproduce torch->NumPy seeding).
    """
    cfg.validate()
    if target.ndim == 2:
        target = target.unsqueeze(0)
    if target.ndim != 3:
        raise ValueError('Expected target [B,H,W] or [H,W]')
    rng = np.random.default_rng(cfg.seed if seed is None else seed)
    b,h,w = target.shape
    target_np = target.detach().cpu().numpy().astype(np.int64)
    safe = target_np.copy()
    safe[safe == cfg.ignore_index] = 0
    if not np.isin(safe, [0,1,2]).all():
        raise ValueError('Target must contain classes 0/1/2 and optional ignore_index')
    onehot = np.eye(3, dtype=np.float32)[safe].transpose(0,3,1,2)
    if cfg.gt_blur_sigma > 0:
        onehot = gaussian_filter(onehot, sigma=(0,0,cfg.gt_blur_sigma,cfg.gt_blur_sigma), mode='reflect')
    span = 2.0*cfg.base_logit_magnitude
    logits = -cfg.base_logit_magnitude + span * onehot
    waves = np.asarray(cfg.spectral_wavelengths, dtype=np.float64)
    amps = np.asarray(cfg.spectral_amplitudes, dtype=np.float64)
    if cfg.perturb_strength and cfg.perturb_probability:
        for bi in range(b):
            if rng.random() >= cfg.perturb_probability:
                continue
            for cls in range(3):
                jittered = amps * rng.uniform(*cfg.spectral_amplitude_jitter, size=amps.shape)
                bands = np.column_stack((jittered, waves))
                logits[bi,cls] += cfg.perturb_strength*span*spectral_noise((h,w), bands,cfg.spectral_bdwidth,rng)
    return torch.from_numpy(np.asarray(logits, dtype=np.float32)).to(target.device)


def initial_probabilities(target: Tensor, cfg: FixedEnergyConfig, *, seed: Optional[int] = None) -> Tensor:
    return synthetic_logits(target, cfg, seed=seed).softmax(dim=1)


def target_from_stain_labels(labels: Tensor, ignore_index: int = -100) -> Tensor:
    """Convert old [B,3,H,W] or [B,Z,3,H,W] stain labels into integer classes.

    channels: nuclear mask, nonnuclear mask, supervision-boost mask.
    Nuclear/nonnuclear overlap receives ignore_index.
    """
    if labels.ndim == 5:
        b,z,c,h,w = labels.shape
        labels = labels.reshape(b*z,c,h,w)
    if labels.ndim != 4 or labels.shape[1] < 2:
        raise ValueError('Expected labels [B,3,H,W] or [B,Z,3,H,W]')
    nuc = labels[:,0] > 0.5
    non = labels[:,1] > 0.5
    t = torch.zeros_like(nuc, dtype=torch.long)
    t[nuc & ~non] = 1
    t[non & ~nuc] = 2
    t[nuc & non] = ignore_index
    return t


def _hard_dice_per_sample(p: Tensor, target: Tensor, ignore_index: int) -> dict[str,float]:
    if target.ndim == 2:
        target = target.unsqueeze(0)
    valid = target != ignore_index
    pred = p.argmax(dim=1)
    result = {}
    for k, name in [(0,'background'),(1,'nuclear'),(2,'nonnuclear')]:
        x = (pred==k)&valid
        y = (target==k)&valid
        denom = x.sum()+y.sum()
        result['dice_'+name] = (2*(x&y).sum()/denom).item() if denom>0 else float('nan')
    return result


@dataclass
class FixedEnergyResult:
    initial: Tensor                        # CPU [B,3,H,W]
    final: Tensor                          # CPU [B,3,H,W]
    snapshots: dict[int,Tensor]             # CPU, iteration -> [B,3,H,W]
    history: list[dict[str,float]]
    target: Optional[Tensor] = None         # CPU [B,H,W]
    stopped_early: bool = False


class FixedEnergyExperiment:
    """One config encapsulates corruption, total fixed energy, solver and plots."""

    def __init__(self, cfg: Optional[FixedEnergyConfig] = None,
                 device: Union[str,torch.device] = 'cpu'):
        self.cfg = FixedEnergyConfig() if cfg is None else cfg
        self.cfg.validate()
        self.device = torch.device(device)
        self.energy = ModicaMortolaLoGEnergy(self.cfg).to(self.device)

    @torch.no_grad()
    def _diagnostics(self, q: Tensor, iteration: int, step: float,
                     target: Optional[Tensor]) -> dict[str,float]:
        p = reduced_to_prob(q)
        area = p.shape[-2]*p.shape[-1]
        e = self.energy.energy_terms_reduced(q)
        d = {'iteration': iteration, 'step_size': float(step)}
        d.update({k: (v/area).mean().item() for k,v in e.items()})
        d['total'] = sum(d[k] for k in e)
        d['mean_confidence'] = p.amax(dim=1).mean().item()
        for k,name in enumerate(('background','nuclear','nonnuclear')):
            d['fraction_'+name] = (p.argmax(dim=1)==k).float().mean().item()
        if target is not None:
            d.update(_hard_dice_per_sample(p,target,self.cfg.ignore_index))
        return d

    def run(self, *, target: Optional[Tensor] = None,
            p0: Optional[Tensor] = None,
            logits0: Optional[Tensor] = None,
            seed: Optional[int] = None,
            verbose: bool = True) -> FixedEnergyResult:
        """Minimize energy from one of three input types.

        target: integer GT [B,H,W] => automatically generate spectral corruption.
        p0: probabilities [B,3,H,W] => use as initial state (no new noise).
        logits0: logits [B,3,H,W] => softmax then minimize.
        You may provide target alongside p0/logits0 for Dice/plots.
        """
        if p0 is not None and logits0 is not None:
            raise ValueError('Supply at most one of p0 and logits0')
        if target is None and p0 is None and logits0 is None:
            raise ValueError('Provide target or p0 or logits0')
        if target is not None:
            target = torch.as_tensor(target, device=self.device).long()
            if target.ndim == 2:
                target = target[None]
            if target.ndim != 3:
                raise ValueError('target must be [B,H,W] or [H,W]')
        if p0 is None and logits0 is None:
            p = initial_probabilities(target, self.cfg, seed=seed)
        elif logits0 is not None:
            p = torch.as_tensor(logits0, device=self.device, dtype=torch.float32).softmax(dim=1)
        else:
            p = torch.as_tensor(p0, device=self.device, dtype=torch.float32)
        if p.ndim == 3:
            p = p[None]
        if p.ndim != 4 or p.shape[1] != 3:
            raise ValueError('p0/logits0 must be [B,3,H,W] or [3,H,W]')
        if target is not None and target.shape != (p.shape[0],*p.shape[-2:]):
            raise ValueError('target spatial/batch shape does not match p0')
        if not torch.isfinite(p).all():
            raise ValueError('non-finite initial state')
        q = prob_to_reduced(p.detach()).detach()
        p_initial = reduced_to_prob(q).detach().cpu().clone()
        history = [self._diagnostics(q,0,self.cfg.step_size,target)]
        snapshots = {0:p_initial}
        trial_step = self.cfg.step_size
        stopped_early = False
        for it in range(1,self.cfg.steps+1):
            with torch.enable_grad():
                x = q.detach().requires_grad_(True)
                energy = self.energy.energy_reduced(x).sum()
                grad, = torch.autograd.grad(energy, x)
            with torch.no_grad():
                accepted = False
                step = trial_step
                for _ in range(self.cfg.max_backtracks+1 if self.cfg.backtracking else 1):
                    candidate = project_triangle(q - step*grad)
                    next_energy = self.energy.energy_reduced(candidate).sum()
                    if torch.isfinite(next_energy) and (
                        not self.cfg.backtracking or next_energy <= energy.detach() + 1e-9
                    ):
                        accepted = True
                        break
                    step *= self.cfg.backtracking_factor
                    if step < self.cfg.min_step_size:
                        break
                if not accepted:
                    stopped_early = True
                    if verbose:
                        print(f'No energy-decreasing projected step at iteration {it}.')
                    break
                q = candidate.detach()
                trial_step = step
            if it%self.cfg.save_every==0 or it==self.cfg.steps:
                r = self._diagnostics(q,it,trial_step,target)
                history.append(r)
                snapshots[it] = reduced_to_prob(q).detach().cpu().clone()
            if verbose and (it%self.cfg.verbose_every==0 or it==self.cfg.steps):
                r = self._diagnostics(q,it,trial_step,target)
                message = (f'it={it:4d}  E/pix={r["total"]:+.6f}  '
                           f'D={r["dirichlet"]:.6f}  W={r["triple_well"]:.6f}  '
                           f'LoG={r["nuclear_log"]:+.6f}  '
                           f'maxP={r["mean_confidence"]:.3f}  step={trial_step:.3g}')
                if target is not None:
                    message += f'  nuclearDice={r["dice_nuclear"]:.3f}  nonDice={r["dice_nonnuclear"]:.3f}'
                print(message)
        final = reduced_to_prob(q).detach().cpu().clone()
        return FixedEnergyResult(initial=p_initial,final=final,snapshots=snapshots,
                                 history=history,target=None if target is None else target.cpu(),
                                 stopped_early=stopped_early)

    def plot(self, result: FixedEnergyResult, *, sample: int = 0,
             save_path: Optional[Union[str,Path]] = None,
             show: bool = True):
        """Columns are snapshots; rows include class mask and probabilities."""
        import matplotlib.pyplot as plt
        from matplotlib.colors import ListedColormap, BoundaryNorm
        keys = sorted(result.snapshots)
        if len(keys) > self.cfg.max_plot_snapshots:
            selected = np.linspace(0, len(keys)-1, self.cfg.max_plot_snapshots).round().astype(int)
            keys = [keys[int(i)] for i in np.unique(selected)]
        if not 0 <= sample < result.initial.shape[0]:
            raise IndexError('sample out of range')
        nrows = 5 if self.cfg.plot_class_probabilities else 3
        if result.target is not None:
            nrows += 1
        ncols = len(keys)
        fig, axes = plt.subplots(nrows, ncols, figsize=(3*ncols, 2.8*nrows), squeeze=False)
        cmap = ListedColormap(['#151515', '#4285F4', '#F7B64C'])
        norm = BoundaryNorm([-0.5,0.5,1.5,2.5], cmap.N)
        row = 0
        if result.target is not None:
            gt = result.target[sample].numpy().copy()
            gt[gt == self.cfg.ignore_index] = -1
            for col in range(ncols):
                axes[row,col].imshow(gt, cmap=cmap, norm=norm, interpolation='nearest')
                axes[row,col].set_title('Ground truth' if col==0 else '')
            row += 1
        for j, step in enumerate(keys):
            p = result.snapshots[step][sample].numpy()
            axes[row,j].imshow(p.argmax(axis=0), cmap=cmap, norm=norm, interpolation='nearest')
            axes[row,j].set_title(f'Iteration {step}')
            axes[row+1,j].imshow(p[1], vmin=0, vmax=1, cmap='magma', interpolation='nearest')
            axes[row+1,j].set_title('Nuclear probability')
            axes[row+2,j].imshow(p.max(axis=0), vmin=0, vmax=1, cmap='viridis', interpolation='nearest')
            axes[row+2,j].set_title('Maximum probability')
            if self.cfg.plot_class_probabilities:
                axes[row+3,j].imshow(p[0], vmin=0, vmax=1, cmap='magma', interpolation='nearest')
                axes[row+3,j].set_title('Background probability')
                axes[row+4,j].imshow(p[2], vmin=0, vmax=1, cmap='magma', interpolation='nearest')
                axes[row+4,j].set_title('Nonnuclear probability')
        for ax in axes.flat:
            ax.axis('off')
        fig.tight_layout()
        if save_path is not None:
            out = Path(save_path)
            out.parent.mkdir(parents=True, exist_ok=True)
            fig.savefig(out, dpi=self.cfg.plot_dpi, bbox_inches='tight')
        if show:
            plt.show()
        return fig

    def plot_history(self, result: FixedEnergyResult, *, save_path=None, show=True):
        """Energy contributions and (if available) Dice vs iteration."""
        import matplotlib.pyplot as plt
        h = result.history
        x = [r['iteration'] for r in h]
        has_target = result.target is not None
        fig, axes = plt.subplots(1, 2 if has_target else 1,
                                 figsize=(11 if has_target else 6, 3.8), squeeze=False)
        ax = axes[0,0]
        for name in ('total','dirichlet','triple_well','nuclear_log'):
            ax.plot(x,[r[name] for r in h],label=name)
        ax.set(xlabel='Iteration',ylabel='Energy per pixel',title='Fixed energy components')
        ax.legend()
        if has_target:
            ax = axes[0,1]
            for name in ('dice_nuclear','dice_nonnuclear'):
                ax.plot(x,[r[name] for r in h],label=name)
            ax.set(xlabel='Iteration',ylabel='Hard Dice',ylim=(0,1),title='Segmentation vs ground truth')
            ax.legend()
        fig.tight_layout()
        if save_path is not None:
            out = Path(save_path)
            out.parent.mkdir(parents=True, exist_ok=True)
            fig.savefig(out, dpi=self.cfg.plot_dpi, bbox_inches='tight')
        if show:
            plt.show()
        return fig


def _demo_target(h=80, w=88) -> Tensor:
    """Small synthetic three-class segmentation for the no-data demo."""
    yy,xx = torch.meshgrid(torch.arange(h),torch.arange(w),indexing='ij')
    y = torch.zeros(h,w,dtype=torch.long)
    for cx,cy,rad in [(23,22,9),(61,20,7),(30,58,10),(64,57,8)]:
        rr=(xx-cx).square()+(yy-cy).square()
        y[rr < (2*rad)**2] = 2
        y[rr < rad**2] = 1
    return y.unsqueeze(0)


def main():
    ap=argparse.ArgumentParser(description='Self-contained fixed-energy segmentation sanity check')
    ap.add_argument('--save-dir',default='./fixed_energy_demo')
    ap.add_argument('--steps',type=int,default=100)
    ap.add_argument('--epsilon',type=float,default=0.5)
    ap.add_argument('--log-weight',type=float,default=0.01)
    ap.add_argument('--perturb-strength',type=float,default=1.0)
    ap.add_argument('--device',default='cpu')
    args=ap.parse_args()
    cfg=FixedEnergyConfig(steps=args.steps,epsilon=args.epsilon,
                          log_weight=args.log_weight,perturb_strength=args.perturb_strength)
    exp=FixedEnergyExperiment(cfg,args.device)
    result=exp.run(target=_demo_target().to(exp.device),verbose=True)
    directory=Path(args.save_dir)
    directory.mkdir(parents=True, exist_ok=True)
    exp.plot(result,save_path=directory/'snapshots.png',show=False)
    exp.plot_history(result,save_path=directory/'history.png',show=False)
    print(f'Saved {directory/"snapshots.png"} and {directory/"history.png"}')


if __name__ == '__main__':
    main()
