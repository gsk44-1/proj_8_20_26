"""Experiment B: shared, bounded trainable geometric energy and anchored PGD."""
import math
import torch
from torch import nn
from torch.nn import functional as F
from mm_fixed_compatible import Config, Energy, prob_to_reduced, reduced_to_prob, project_triangle, _scipy_rotation_matrices
from anchored_refinement import anchor_gradient


def inverse_sigmoid(x):
    x = min(max(float(x), 1e-5), 1-1e-5)
    return math.log(x/(1-x))

class StructuredEnergy(Energy):
    """Keep EXACT original MM and fixed kernels. Learn capped strengths and
    additional two-channel canonical kernels tied over rotated copies.
    Each learned feature uses 16 rotations over 360 degrees by default;
    orientations are aggregated by max, then ReLU and squared reward.
    """
    def __init__(self, cfg: Config, n_extra=3, filter_size=13,
                 strength_cap=1.5, extra_strength_cap=0.3, kernel_raw_cap=2.0,
                 initial_extra_strength=0.05, seed=17, extra_orientations=16):
        if n_extra not in (2, 3, 4):
            raise ValueError('n_extra must be 2, 3 or 4')
        if int(extra_orientations) != extra_orientations or extra_orientations < 1:
            raise ValueError('extra_orientations must be a positive integer')
        if not 0 < initial_extra_strength < extra_strength_cap:
            raise ValueError('Initial extra strength must lie strictly inside cap')
        if filter_size % 2 != 1 or filter_size < 3:
            raise ValueError('filter_size must be odd and >=3')
        if not 0 < cfg.barrier_weight < strength_cap or not 0 < cfg.boundary_weight < strength_cap:
            raise ValueError('Initial strengths must lie strictly inside cap')
        # Do not create original unbounded learnable parameters.
        cfg.learn_strengths = False
        cfg.learn_residuals = False
        super().__init__(cfg)
        self.strength_cap = float(strength_cap)
        self.extra_strength_cap = float(extra_strength_cap)
        self.kernel_raw_cap = float(kernel_raw_cap)
        self.n_extra = n_extra
        self.extra_orientations = int(extra_orientations)
        self.filter_size = int(filter_size)
        # Identical bilinear SciPy rotation convention to the fixed filters.
        # Buffer follows .to(device) and is included in state_dict.
        self.register_buffer("extra_rotmat", _scipy_rotation_matrices(
            filter_size, self.extra_orientations, True))
        self.fixed_strength_logits = nn.Parameter(torch.tensor([
            inverse_sigmoid(cfg.barrier_weight / strength_cap),
            inverse_sigmoid(cfg.boundary_weight / strength_cap)], dtype=torch.float32))
        self.extra_strength_logits = nn.Parameter(torch.full((n_extra,),
            inverse_sigmoid(initial_extra_strength / extra_strength_cap)))
        rng = torch.Generator().manual_seed(seed)
        self.extra_raw = nn.Parameter(torch.randn(n_extra, 2, filter_size, filter_size, generator=rng)*0.08)

    def strengths(self):
        return self.strength_cap * torch.sigmoid(self.fixed_strength_logits)

    def extra_strengths(self):
        return self.extra_strength_cap * torch.sigmoid(self.extra_strength_logits)

    def normalized_extra_kernels(self):
        # Every raw parameter is bounded via tanh, followed by zero-DC and
        # positive coefficient sum normalization, matching the original bank.
        k = self.kernel_raw_cap * torch.tanh(self.extra_raw)
        k = k - k.mean((-2, -1), keepdim=True)
        normalizer = k.clamp_min(0).sum((1, 2, 3), keepdim=True).clamp_min(1e-8)
        return k / normalizer

    def rotated_extra_kernels(self):
        """Return [n_extra, orientations, 2, S, S] trainably rotated kernels.

        Each orientation is spatially zero-mean in each channel, and its
        positive-coefficient sum over both channels equals one. Rotation is
        linear in the canonical coefficients; all operations are differentiable.
        """
        canonical = self.normalized_extra_kernels()
        n, channels, size, _ = canonical.shape
        rotated = torch.einsum(
            'oij,ncj->noci',
            self.extra_rotmat.to(dtype=canonical.dtype),
            canonical.flatten(2),
        ).reshape(n, self.extra_orientations, channels, size, size)
        rotated = rotated - rotated.mean(dim=(-2, -1), keepdim=True)
        denom = rotated.clamp_min(0).sum(dim=(-3, -2, -1), keepdim=True).clamp_min(1e-8)
        return rotated / denom

    def extra_responses(self, p):
        """Return [B,n_extra,H,W] after orientation max and ReLU."""
        bank = self.rotated_extra_kernels().reshape(
            self.n_extra * self.extra_orientations, 2, self.filter_size, self.filter_size
        )
        response = self._convolve_periodic(p[:, 1:3], bank)
        B, _, H, W = response.shape
        response = response.reshape(B, self.n_extra, self.extra_orientations, H, W)
        return response.amax(dim=2).relu()

    def energy_terms_reduced(self, q):
        # The original separator term is recomputed with bounded strength;
        # MM terms are unchanged and boundary uses the original bank.
        terms = super().energy_terms_reduced(q)
        p = reduced_to_prob(q)
        weights = self.strengths()
        sep = F.relu(self.barrier_responses(p).amax(dim=1))
        boundary = F.relu(self.responses(p, 'boundary').amax(dim=1))
        terms['barrier'] = -0.5 * weights[0] * sep.square().flatten(1).sum(1)
        terms['boundary'] = -0.5 * weights[1] * boundary.square().flatten(1).sum(1)
        extra = self.extra_responses(p)
        terms['extra'] = -0.5 * (extra.square().flatten(2).sum(2) *
                                 self.extra_strengths()[None]).sum(1)
        return terms

    def diagnostics(self):
        with torch.no_grad():
            return {'separator_strength': float(self.strengths()[0]),
                    'boundary_strength': float(self.strengths()[1]),
                    **{f'extra_strength_{i}': float(v) for i, v in enumerate(self.extra_strengths())}}


def anchored_step(model, p, p_anchor, mu=0.4, step_size=0.02, differentiable=True):
    """Full-three-class L2 fidelity with original reduced-coordinate metric.
    Works for detached warmups and differentiable truncated unrolling.
    """
    with torch.enable_grad():
        q = prob_to_reduced(p)
        if not q.requires_grad:
            q = q.detach().requires_grad_(True)
        q0 = p_anchor[:, 1:3].detach()
        grad_e, = torch.autograd.grad(model.energy_reduced(q).sum(), q,
                                     create_graph=differentiable)
        next_q = project_triangle(q - step_size * (grad_e + anchor_gradient(q, q0, mu)))
        out = reduced_to_prob(next_q)
    return out if differentiable else out.detach()


def anchored_rollout(model, p, p_anchor, steps, mu, step_size, differentiable=False):
    for _ in range(steps):
        p = anchored_step(model, p, p_anchor, mu, step_size, differentiable)
    return p
