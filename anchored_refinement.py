"""Anchored refinement using the UNMODIFIED fixed three-phase MM energy.

No trainable parameters, no model-dependent term. Optimization is in the
original two-coordinate Euclidean metric. Full 3-class L2 anchor.
"""
from dataclasses import dataclass
import torch
from mm_fixed_compatible import Energy, Config, project_triangle, reduced_to_prob


@dataclass(frozen=True)
class RefinementSettings:
    anchor_mu: float = 0.1
    step_size: float = 0.02
    steps: int = 400
    checkpoints: tuple = (0, 25, 50, 100, 200, 300, 400)
    backtracking: bool = False
    backtracking_factor: float = 0.5
    max_backtracks: int = 12
    min_step_size: float = 1e-8

    def validate(self):
        if self.anchor_mu < 0 or self.step_size <= 0 or self.steps < 0:
            raise ValueError("Invalid anchor/step settings")
        if any(t < 0 or t > self.steps for t in self.checkpoints):
            raise ValueError("checkpoints must lie in [0, steps]")
        if not 0 < self.backtracking_factor < 1:
            raise ValueError("Invalid backtracking_factor")


def anchor_energy(q, q0, mu):
    """Full 3-channel squared-L2 fidelity in reduced coordinates, per sample."""
    du, dw = (q - q0).unbind(1)
    return 0.5 * mu * (du.square() + dw.square() + (du + dw).square()).flatten(1).sum(1)


def anchor_gradient(q, q0, mu):
    """Gradient of fidelity: mu * [[2,1],[1,2]] @ (q-q0)."""
    delta = q - q0
    return mu * torch.stack((2 * delta[:, 0] + delta[:, 1],
                             delta[:, 0] + 2 * delta[:, 1]), dim=1)


def total_energy(model, q, q0, mu):
    return model.energy_reduced(q) + anchor_energy(q, q0, mu)


def refine(model: Energy, original_prob: torch.Tensor, settings: RefinementSettings):
    """Return {iteration: probabilities} and per-step objective trace.

    original_prob: [B,3,H,W], softmax output. Original is held fixed.
    Backtracking, if enabled, tests decrease of the TOTAL anchored objective
    jointly over the batch; every step projects onto the reduced triangle.
    Each result is detached; no training graph is built.
    """
    settings.validate()
    if original_prob.ndim != 4 or original_prob.shape[1] != 3:
        raise ValueError("Expected [B,3,H,W] softmax probabilities")
    p0 = original_prob.detach().float()
    q0 = p0[:, 1:3].clone()
    q = project_triangle(q0).detach()
    wanted = set(settings.checkpoints) | {0, settings.steps}
    snapshots = {0: reduced_to_prob(q).detach().cpu().clone()}
    trace = []
    model.eval()
    for t in range(1, settings.steps + 1):
        with torch.enable_grad():
            current = q.detach().requires_grad_(True)
            energy = model.energy_reduced(current).sum()
            grad_e, = torch.autograd.grad(energy, current, create_graph=False)
            grad = grad_e.detach() + anchor_gradient(current.detach(), q0, settings.anchor_mu)
        with torch.no_grad():
            old_value = None
            if settings.backtracking:
                old_value = total_energy(model, q, q0, settings.anchor_mu).sum()
            step = settings.step_size
            accepted = False
            for attempt in range(settings.max_backtracks + 1 if settings.backtracking else 1):
                proposal = project_triangle(q - step * grad)
                if not settings.backtracking:
                    accepted = True
                    break
                new_value = total_energy(model, proposal, q0, settings.anchor_mu).sum()
                if new_value <= old_value + 1e-7:
                    accepted = True
                    break
                step *= settings.backtracking_factor
                if step < settings.min_step_size:
                    break
            if accepted:
                q = proposal.detach()
            # If line search failed, leave q unchanged for this step.
            if t in wanted:
                snapshots[t] = reduced_to_prob(q).detach().cpu().clone()
                trace.append({"iteration": t, "total_energy_per_pixel": float(
                    total_energy(model, q, q0, settings.anchor_mu).mean().item() / (q.shape[-2] * q.shape[-1])),
                    "step_size": float(step if accepted else 0)})
    return snapshots, trace
