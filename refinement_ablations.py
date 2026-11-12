"""Controlled MM/template energy ablations, preserving the original discretization.

Defaults use the original periodic Dirichlet energy, reduced-coordinate gradient,
anchoring, and painted templates. No U-Net checkpoint is required.
"""
from __future__ import annotations
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Optional
import torch
from mm_fixed_compatible import Config, Energy, reduced_to_prob
from template_energy import TemplateEnergy

@dataclass(frozen=True)
class Ablation:
    name: str
    mm: bool
    coefficients: tuple[float, float, float] = (1., 1., 1.)  # (uz, uw, wz)
    templates: bool = False

ABLATIONS = (
    Ablation('No refinement (anchor only)', False, templates=False),
    Ablation('Symmetric MM', True),
    Ablation('Asymmetric MM', True, (3., 2., 1.)),
    Ablation('Templates only', False, templates=True),
    Ablation('Symmetric MM + templates', True, templates=True),
    Ablation('Asymmetric MM + templates', True, (3., 2., 1.), True),
)

class _AsymmetricWellMixin:
    """Replace ONLY the original symmetric quartic potential term.

    The untouched parent also computes cubic W, Dirichlet, and optional template
    rewards. We subtract its old entire well and add our new well, with the
    same original cubic coefficient and lam/epsilon scaling.
    """
    def energy_terms_reduced(self, q):
        terms = super().energy_terms_reduced(q)
        p = reduced_to_prob(q)
        z, u, w = p[:, 0], p[:, 1], p[:, 2]
        a_uz, a_uw, a_wz = self.pair_coefficients
        W = (a_uz * u.square() * z.square()
             + a_uw * u.square() * w.square()
             + a_wz * w.square() * z.square()
             + self.cfg.well_cubic_weight * u * w * z)
        terms['triple_well'] = (self.cfg.lam / self.cfg.epsilon) * W.flatten(1).sum(1)
        return terms

class PairwiseMM(_AsymmetricWellMixin, Energy):
    def __init__(self, cfg, coefficients):
        self.pair_coefficients = tuple(float(v) for v in coefficients)
        super().__init__(cfg)

class PairwiseTemplateMM(_AsymmetricWellMixin, TemplateEnergy):
    def __init__(self, cfg, coefficients, **kwargs):
        self.pair_coefficients = tuple(float(v) for v in coefficients)
        super().__init__(cfg, **kwargs)

def make_energy(spec: Ablation, *, lam=0.2, epsilon=0.5,
                well_cubic_weight=1., template_dir=None, orientations=16,
                fixed_strength=0.05, strength_cap=0.3,
                checkpoint: Optional[str] = None, device='cpu'):
    """Build energy with optionally fixed strengths or saved trained strengths.

    Loading a checkpoint checks template name order and uses strict state_dict
    matching. Checkpoints represent the original template feature parameterization.
    """
    if any(a < 0 for a in spec.coefficients):
        raise ValueError('Pairwise coefficients must be nonnegative')
    cfg = Config(lam=lam if spec.mm else 0., epsilon=epsilon,
                 well_cubic_weight=well_cubic_weight,
                 barrier_weight=0., boundary_weight=0.,
                 learn_strengths=False, learn_residuals=False)
    if spec.templates:
        model = PairwiseTemplateMM(cfg, spec.coefficients,
                                  template_dir=template_dir, orientations=orientations,
                                  strength_cap=strength_cap, initial_strength=fixed_strength)
        if checkpoint:
            obj = torch.load(checkpoint, map_location='cpu', weights_only=False)
            names = obj.get('template_names') if isinstance(obj, dict) else None
            if names is not None and tuple(names) != model.template_names:
                raise ValueError(f'Checkpoint template order {names} differs from {model.template_names}')
            state = obj.get('model_state_dict', obj) if isinstance(obj, dict) else obj
            model.load_state_dict(state, strict=True)
        model.template_strength_logits.requires_grad_(False)
    else:
        if checkpoint:
            raise ValueError('Checkpoint applies only to template-containing models')
        model = PairwiseMM(cfg, spec.coefficients)
    return model.to(device).eval()


def energy_breakdown(model, prob, anchor, mu):
    """Energy components per pixel for interpretability, including anchor."""
    from anchored_refinement import anchor_energy
    from mm_fixed_compatible import prob_to_reduced
    q = prob_to_reduced(prob.detach())
    terms = model.energy_terms_reduced(q)
    terms['anchor'] = anchor_energy(q, prob_to_reduced(anchor), mu)
    n_pixels = prob.shape[-2] * prob.shape[-1]
    return {k: float(v.detach().mean().cpu()) / n_pixels for k, v in terms.items()}
