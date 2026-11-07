"""Image-free, three-phase hierarchical scalar energy for segmentation correction.

State: p with shape [B, 3, H, W], p >= 0 and sum_classes(p) = 1.
Class order: background (0), nuclear (1), nonnuclear (2).

The same learned energy is used at every iteration:

    E(p) = sum_s weight_s * integral h_s(p_s) dx
           - log_weight/2 * || sigma^2 Laplacian(G_sigma * p_nuclear) ||^2.

The learned density heads use Softplus outputs, so the learned contribution is
nonnegative. The negative LoG term is bounded below on the finite simplex.

Default iteration: p_next = project_simplex(p - step * grad_p E(p)).

A separate three-phase well and its approximate constrained proximal step are
included, but are NOT applied in the default step / training path. Setting
use_triple_well=True enables this separate splitting step.

No microscopy image, initial logits, or other conditioning is supplied to E:
only the current three-channel probability state is used.

Training an energy-gradient update requires create_graph=True, since the
parameter derivatives involve mixed second derivatives of E. Use
unroll_truncated(..., backprop_steps=3) for truncated BPTT with shared weights.
"""

from __future__ import annotations

import math
import torch
from torch import Tensor, nn
from torch.nn import functional as F


def project_simplex(v: Tensor) -> Tensor:
    """Euclidean projection onto the 3-class probability simplex per pixel.

    v: [B, 3, H, W]. Differentiable almost everywhere (piecewise linear).
    """
    if v.ndim != 4 or v.shape[1] != 3:
        raise ValueError("Expected tensor [B, 3, H, W]")
    sorted_v = torch.sort(v, dim=1, descending=True).values
    cs = sorted_v.cumsum(dim=1) - 1.0
    j = torch.arange(1, 4, device=v.device, dtype=v.dtype).view(1, 3, 1, 1)
    valid = sorted_v > cs / j
    rho = valid.sum(dim=1, keepdim=True).clamp(min=1)
    theta = cs.gather(1, rho - 1) / rho.to(dtype=v.dtype)
    return (v - theta).clamp_min(0.0)


def three_phase_well(p: Tensor, cubic_weight: float = 1.0) -> Tensor:
    """Per-pixel symmetric three-phase potential, [B, 1, H, W].

    W(p) = sum_{i<j} p_i^2 p_j^2 + cubic_weight * p_0 p_1 p_2.
    On the simplex and for cubic_weight >= 0, W >= 0 with minima at vertices.
    """
    a, b, c = p[:, 0:1], p[:, 1:2], p[:, 2:3]
    return (a*a*b*b + a*a*c*c + b*b*c*c
            + cubic_weight*a*b*c)


def three_phase_well_gradient(p: Tensor, cubic_weight: float = 1.0) -> Tensor:
    """Ambient gradient of three_phase_well in all three channels."""
    a, b, c = p[:, 0:1], p[:, 1:2], p[:, 2:3]
    return torch.cat((
        2*a*(b*b + c*c) + cubic_weight*b*c,
        2*b*(a*a + c*c) + cubic_weight*a*c,
        2*c*(a*a + b*b) + cubic_weight*a*b,
    ), dim=1)


def three_phase_well_prox(
    v: Tensor, strength: float, *, iterations: int = 8,
    relaxation: float = 0.25, cubic_weight: float = 1.0,
) -> Tensor:
    """Approximate constrained proximal step, independently at each pixel.

    Tries to solve argmin_{p in simplex} 0.5*||p-v||^2 + strength*W(p)
    by damped projected fixed-point iteration:
        p <- (1-relaxation)*p + relaxation*Pi(v-strength*grad W(p)).

    Nonconvex: this is an approximation, NOT a guaranteed global prox solver.
    Both input and output have shape [B, 3, H, W].
    """
    if strength < 0 or iterations < 0 or not 0 < relaxation <= 1:
        raise ValueError("strength >= 0, iterations >= 0, 0 < relaxation <= 1")
    q = project_simplex(v)
    for _ in range(iterations):
        target = project_simplex(v - strength * three_phase_well_gradient(q, cubic_weight))
        q = (1.0 - relaxation)*q + relaxation*target
    return q


def _conv_block(cin: int, cout: int) -> nn.Sequential:
    return nn.Sequential(
        nn.Conv2d(cin, cout, 3, padding=1), nn.SiLU(),
        nn.Conv2d(cout, cout, 3, padding=1), nn.SiLU(),
    )


class _DensityHead(nn.Module):
    """Nonnegative scalar energy density at one spatial resolution."""
    def __init__(self, cin: int, width: int):
        super().__init__()
        self.features = _conv_block(cin, width)
        self.out = nn.Conv2d(width, 1, 1)

    def forward(self, x: Tensor) -> Tensor:
        return F.softplus(self.out(self.features(x)))


class ProbabilityHierarchicalEnergy(nn.Module):
    """Autonomous learned prior with a fixed nuclear-only LoG reward."""

    def __init__(
        self,
        *,
        state_width: int = 16,
        head_width: int = 24,
        scale_weights: tuple[float, float, float] = (1.0, 0.5, 0.25),
        log_sigma: float = 5.0,
        log_weight: float = 0.01,
        initial_step: float = 0.1,
        well_weight: float = 0.1,
        well_cubic_weight: float = 1.0,
        well_iterations: int = 8,
        well_relaxation: float = 0.25,
    ):
        super().__init__()
        if state_width < 1 or head_width < 1 or len(scale_weights) != 3:
            raise ValueError("Invalid widths or scale_weights")
        if (any(w < 0 for w in scale_weights) or log_sigma <= 0
                or log_weight < 0 or initial_step <= 0 or well_weight < 0
                or well_cubic_weight < 0):
            raise ValueError("Weights must be nonnegative; log_sigma and step positive")
        self.scale_weights = tuple(float(w) for w in scale_weights)
        self.log_weight = float(log_weight)
        self.well_weight = float(well_weight)
        self.well_cubic_weight = float(well_cubic_weight)
        self.well_iterations = int(well_iterations)
        self.well_relaxation = float(well_relaxation)
        self.log_step = nn.Parameter(torch.tensor(math.log(initial_step)))

        c = state_width
        self.K1 = nn.Conv2d(3, c, 3, padding=1)
        self.K2 = nn.Conv2d(c, 2*c, 3, padding=1)
        self.head_fine = _DensityHead(3, head_width)
        self.head_half = _DensityHead(c, head_width)
        self.head_quarter = _DensityHead(2*c, head_width)

        # Fixed separable Gaussian (one-dimensional x/y) and five-point Laplacian.
        # Replicate padding preserves constant fields and avoids artificial
        # background responses at image boundaries. Autograd handles its adjoint.
        radius = max(1, math.ceil(4 * log_sigma))
        x = torch.arange(-radius, radius + 1, dtype=torch.float32)
        g = torch.exp(-0.5*(x / log_sigma)**2)
        g = g / g.sum()
        self.register_buffer("gaussian_x", g.view(1, 1, 1, -1))
        self.register_buffer("gaussian_y", g.view(1, 1, -1, 1))
        lap = torch.tensor([[0., 1., 0.], [1., -4., 1.], [0., 1., 0.]])
        self.register_buffer("laplacian", lap.view(1, 1, 3, 3))
        self.log_sigma = float(log_sigma)

    @staticmethod
    def from_logits(logits: Tensor) -> Tensor:
        """Convert noisy three-class logits to a probability state."""
        if logits.ndim != 4 or logits.shape[1] != 3:
            raise ValueError("Expected logits [B, 3, H, W]")
        return logits.softmax(dim=1)

    def log_response(self, p: Tensor) -> Tensor:
        """Approximate scale-normalized LoG of the NUCLEAR probability only.

        Uses sigma^2 * discrete_laplacian(G_sigma * p[:, 1:2]).
        The filter is linear; all kernels are fixed (nonlearned).
        """
        x = p[:, 1:2]
        r = self.gaussian_x.shape[-1] // 2
        x = F.conv2d(F.pad(x, (r, r, 0, 0), mode="replicate"), self.gaussian_x)
        x = F.conv2d(F.pad(x, (0, 0, r, r), mode="replicate"), self.gaussian_y)
        x = F.conv2d(F.pad(x, (1, 1, 1, 1), mode="replicate"), self.laplacian)
        return (self.log_sigma ** 2) * x

    def energy_terms(self, p: Tensor) -> dict[str, Tensor]:
        """Return per-image energy contributions, each shape [B].

        Scale energies are weighted by full-resolution area / scale area, so
        all three energy densities contribute on comparable extensive scales.
        """
        if p.ndim != 4 or p.shape[1] != 3:
            raise ValueError("Expected state [B, 3, H, W]")
        h, w = p.shape[-2:]
        u1 = F.silu(self.K1(F.avg_pool2d(p, 2, ceil_mode=True)))
        u2 = F.silu(self.K2(F.avg_pool2d(u1, 2, ceil_mode=True)))
        densities = (
            self.head_fine(p), self.head_half(u1), self.head_quarter(u2)
        )
        terms = {}
        for name, weight, density in zip(
            ("fine", "half", "quarter"), self.scale_weights, densities
        ):
            terms[name] = (weight * h*w * density.flatten(1).mean(dim=1))
        if self.log_weight:
            response = self.log_response(p)
            terms["nuclear_log"] = (-0.5 * self.log_weight
                                     * response.square().flatten(1).sum(dim=1))
        else:
            terms["nuclear_log"] = p.new_zeros(p.shape[0])
        return terms

    def energy(self, p: Tensor) -> Tensor:
        """Total scalar energy per batch element [B], without triple well."""
        terms = self.energy_terms(p)
        return sum(terms.values())

    def energy_gradient(self, p: Tensor, *, create_graph: bool = False) -> Tensor:
        """Exact Euclidean state gradient; autograd supplies transposed convs.

        Set create_graph=True when training THROUGH a gradient update.
        """
        with torch.enable_grad():
            if not p.requires_grad:
                p = p.detach().requires_grad_(True)
            return torch.autograd.grad(
                self.energy(p).sum(), p, create_graph=create_graph
            )[0]

    def step(self, p: Tensor, *, differentiable: bool = False,
             use_triple_well: bool = False) -> Tensor:
        """One projected gradient step with optional separate well substep.

        The default path excludes the triple-well entirely, including in
        training. If enabled, use a nonconvex proximal splitting approximation.
        """
        with torch.enable_grad():
            if not p.requires_grad:
                p = p.detach().requires_grad_(True)
            grad = self.energy_gradient(p, create_graph=differentiable)
            trial = p - self.log_step.exp() * grad
            if use_triple_well:
                result = three_phase_well_prox(
                    trial,
                    strength=self.log_step.exp() * self.well_weight,
                    iterations=self.well_iterations,
                    relaxation=self.well_relaxation,
                    cubic_weight=self.well_cubic_weight,
                )
            else:
                result = project_simplex(trial)
        return result if differentiable else result.detach()

    def well_step(self, p: Tensor, *, strength: float | None = None) -> Tensor:
        """Standalone three-phase well update; not used by default."""
        return three_phase_well_prox(
            p, strength=self.well_weight if strength is None else strength,
            iterations=self.well_iterations,
            relaxation=self.well_relaxation,
            cubic_weight=self.well_cubic_weight,
        )

    def unroll_truncated(
        self, p0: Tensor, *, steps: int, backprop_steps: int = 3,
        use_triple_well: bool = False,
    ) -> Tensor:
        """Run shared-weight steps, tracking ONLY the final backprop_steps.

        Run the early steps with the state detached after each update. The
        final steps retain higher-order autograd graphs for training.
        """
        if steps < 0 or backprop_steps < 0:
            raise ValueError("steps and backprop_steps must be nonnegative")
        tracked = min(steps, backprop_steps)
        p = p0.detach()
        for _ in range(steps - tracked):
            p = self.step(p, differentiable=False,
                          use_triple_well=use_triple_well)
        for _ in range(tracked):
            p = self.step(p, differentiable=True,
                          use_triple_well=use_triple_well)
        return p

    def forward(self, p0: Tensor, *, steps: int = 12,
                differentiable: bool = False,
                use_triple_well: bool = False) -> Tensor:
        """Full rollout with shared energy; default inference is detached.

        For training use differentiable=True (full unroll) or, preferably,
        unroll_truncated(..., backprop_steps=3).
        """
        p = p0 if differentiable else p0.detach()
        for _ in range(steps):
            p = self.step(p, differentiable=differentiable,
                          use_triple_well=use_triple_well)
        return p


if __name__ == "__main__":
    torch.manual_seed(2)
    model = ProbabilityHierarchicalEnergy(
        state_width=4, head_width=6, log_sigma=2.0, log_weight=0.01
    )
    logits = torch.randn(2, 3, 17, 19)
    p0 = model.from_logits(logits)
    p = model.unroll_truncated(p0, steps=5, backprop_steps=2)
    assert p.shape == p0.shape
    assert p.min().item() >= -1e-6
    assert torch.allclose(p.sum(dim=1), torch.ones_like(p[:, 0]), atol=1e-5)
    loss = (p[:, 1] - 0.4).square().mean()
    loss.backward()
    assert model.K1.weight.grad is not None
    assert model.K2.weight.grad is not None
    assert model.head_fine.out.weight.grad is not None
    assert model.log_step.grad is not None
    with torch.no_grad():
        output = model(p0, steps=2, use_triple_well=True)
    assert torch.allclose(output.sum(1), torch.ones_like(output[:, 0]), atol=1e-5)
    print("Smoke test passed; truncated second-order gradients and simplex valid.")
