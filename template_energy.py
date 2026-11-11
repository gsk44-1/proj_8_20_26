"""Fixed painted 21x21 template dictionary with trainable capped strengths.

Black = nuclear preference, white = nonnuclear preference,
indigo = don't care (zero weights before rotation).

Existing MM implementation and projected metric are preserved. The original
separator/boundary reward terms are disabled, with no learned kernel pixels.
"""
from __future__ import annotations

from dataclasses import replace
from pathlib import Path
import math

import numpy as np
import torch
from PIL import Image
from torch import nn
from torch.nn import functional as F
from scipy.ndimage import rotate

from mm_fixed_compatible import Config, Energy, prob_to_reduced, reduced_to_prob, project_triangle
from anchored_refinement import anchor_gradient

COLORS = {
    'nuclear': (0, 0, 0),
    'nonnuclear': (255, 255, 255),
    'unspecified': (63, 72, 204),
}


def inverse_sigmoid(p: float) -> float:
    if not 0 < p < 1:
        raise ValueError('initial strength must be strictly between zero and cap')
    return math.log(p / (1 - p))


def load_template(path: Path) -> np.ndarray:
    """[2,H,W]: nuclear-vs-nonnuclear contrasts; unspecified EXACTLY zero.

    Black: (+1,-1); white: (-1,+1); indigo: (0,0).
    This representation does not distinguish background from a half/half mixture.
    It does distinguish nuclear from nonnuclear, the intended structures.
    """
    rgb = np.asarray(Image.open(path).convert('RGB'), dtype=np.uint8)
    if rgb.ndim != 3 or rgb.shape[:2] != (21, 21):
        raise ValueError(f'{path.name}: expected exactly 21x21 pixels, got {rgb.shape}')
    masks = {name: np.all(rgb == np.array(color, dtype=np.uint8), axis=-1)
             for name, color in COLORS.items()}
    invalid = ~(masks['nuclear'] | masks['nonnuclear'] | masks['unspecified'])
    if invalid.any():
        bad = np.unique(rgb[invalid], axis=0)
        raise ValueError(f'{path.name}: unknown RGB color(s) {bad.tolist()[:8]}; save as PNG without antialiasing')
    if not masks['nuclear'].any() and not masks['nonnuclear'].any():
        raise ValueError(f'{path.name}: no specified nuclear/nonnuclear pixels')
    nuclear = masks['nuclear'].astype(np.float32) - masks['nonnuclear'].astype(np.float32)
    nonnuclear = -nuclear
    return np.stack([nuclear, nonnuclear])


def rotated_bank(template: np.ndarray, n_orientations: int = 16) -> np.ndarray:
    """Bank [O,2,21,21] using the old SciPy order=1 rotation settings.

    Zero-valued indigo support stays precisely zero at angle 0. At other
    angles bilinear interpolation causes edge mixing; no mean-subtraction
    is performed, so there is no global unspecified-region bias.
    Positive-sum normalization is per rotated filter, like the original.
    """
    if n_orientations < 1:
        raise ValueError('n_orientations must be positive')
    bank = []
    for i in range(n_orientations):
        angle = 360.0 * i / n_orientations
        if i == 0:
            kernel = template.copy()
        else:
            kernel = np.stack([
                rotate(template[c], angle, reshape=False, order=1,
                       mode='constant', cval=0.0, prefilter=False)
                for c in range(2)
            ]).astype(np.float32)
        norm = np.maximum(kernel, 0).sum()
        if norm <= 0:
            raise ValueError('Degenerate rotation')
        bank.append(kernel / norm)
    return np.stack(bank).astype(np.float32)


class TemplateEnergy(Energy):
    """Complete scalar geometric energy: original MM + painted filter rewards.

    Only `template_strength_logits` are learnable. Filenames are sorted for
    deterministic indexing. Kernels live in state_dict buffers; checkpoints
    also store template names and configuration in the trainer.
    """
    def __init__(self, cfg: Config, template_dir: str | Path | None = None,
                 orientations: int = 16, strength_cap: float = 0.3,
                 initial_strength: float = 0.05):
        # Use original MM unchanged while DISABLING old separator/boundary terms.
        base_cfg = replace(cfg, barrier_weight=0., boundary_weight=0.,
                           learn_strengths=False, learn_residuals=False)
        super().__init__(base_cfg)
        if not 0 < initial_strength < strength_cap:
            raise ValueError('Need 0 < initial_strength < strength_cap')
        root = Path(template_dir) if template_dir is not None else Path(__file__).resolve().parent / 'templates'
        paths = sorted(root.glob('*.png'))
        if not paths:
            raise FileNotFoundError(f'No *.png templates in {root}')
        self.template_names = tuple(p.name for p in paths)
        self.template_dir = str(root)
        self.orientations = int(orientations)
        self.strength_cap = float(strength_cap)
        canonical = np.stack([load_template(p) for p in paths])
        bank = np.stack([rotated_bank(k, self.orientations) for k in canonical])
        self.register_buffer('template_canonical', torch.from_numpy(canonical))
        self.register_buffer('template_bank', torch.from_numpy(bank))
        self.template_strength_logits = nn.Parameter(torch.full(
            (len(paths),), inverse_sigmoid(initial_strength / strength_cap), dtype=torch.float32))

    def strengths(self):
        return self.strength_cap * torch.sigmoid(self.template_strength_logits)

    def template_responses(self, p):
        bank = self.template_bank.flatten(0, 1)
        r = self._convolve_periodic(p[:, 1:3], bank)
        b, _, h, w = r.shape
        return r.reshape(b, len(self.template_names), self.orientations, h, w).amax(2).relu()

    def energy_terms_reduced(self, q):
        # Parent returns exact old MM terms and *zero* separator/boundary terms.
        terms = super().energy_terms_reduced(q)
        p = reduced_to_prob(q)
        responses = self.template_responses(p)
        terms['templates'] = -0.5 * (
            responses.square().flatten(2).sum(-1) * self.strengths()[None, :]
        ).sum(dim=1)
        return terms

    def diagnostics(self):
        return {path: float(s) for path, s in zip(self.template_names, self.strengths().detach().cpu())}


def anchored_step(model, p, p_anchor, mu=0.4, step_size=0.02, differentiable=True):
    with torch.enable_grad():
        q = prob_to_reduced(p)
        if not q.requires_grad:
            q = q.detach().requires_grad_(True)
        q0 = p_anchor[:, 1:3].detach()
        (grad_e,) = torch.autograd.grad(model.energy_reduced(q).sum(), q,
                                       create_graph=differentiable)
        q_next = project_triangle(q - step_size * (grad_e + anchor_gradient(q, q0, mu)))
        result = reduced_to_prob(q_next)
    return result if differentiable else result.detach()


def anchored_rollout(model, p, p_anchor, steps, mu, step_size, differentiable=False):
    for _ in range(steps):
        p = anchored_step(model, p, p_anchor, mu, step_size, differentiable)
    return p
