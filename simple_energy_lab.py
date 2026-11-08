"""Minimal 3-class learned scalar energy and reproducible diagnostics.

No image channels, fixed energies, or external project modules are required.

CLI examples:
  python simple_energy_lab.py diagnose --out energy_before
  python simple_energy_lab.py train-demo --out energy_demo --train-steps 80
  python simple_energy_lab.py diagnose --checkpoint energy_demo/demo_checkpoint.pt --out energy_after

The model operates on [B,3,H,W] simplex probabilities. Its forward
returns an *extensive* scalar energy per sample [B]; energy_per_pixel
is provided for resolution-independent reporting.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import json
import math
from pathlib import Path
from typing import Optional

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F


@dataclass
class LabConfig:
    width: int = 12
    image_size: int = 64
    batch_size: int = 2
    seed: int = 7
    init_head_std: float = 0.015
    test_seeds: int = 3
    curvature_directions: int = 6
    curvature_delta: float = 0.025
    descent_steps: int = 30
    descent_step_size: float = 1.0
    max_backtracks: int = 12
    spectrum_bins: int = 6
    device: str = 'cpu'


class SimpleSegmentationEnergy(nn.Module):
    """Scalar, translation-equivariant local energy, with circular padding.

    The extensive sum is intentional: state gradients are not divided
    by H*W. For comparisons, use energy_per_pixel().
    """
    def __init__(self, width: int = 12, init_head_std: float = 0.015):
        super().__init__()
        self.width = width
        self.init_head_std = init_head_std
        self.conv1 = nn.Conv2d(3, width, 5, padding=2, padding_mode='circular')
        self.conv2 = nn.Conv2d(width, width, 3, padding=1, padding_mode='circular')
        self.head = nn.Conv2d(width, 1, 1)
        self.act = nn.SiLU()
        nn.init.normal_(self.head.weight, mean=0.0, std=init_head_std)
        nn.init.zeros_(self.head.bias)

    def density(self, p: torch.Tensor) -> torch.Tensor:
        if p.ndim != 4 or p.shape[1] != 3:
            raise ValueError('Expected [B,3,H,W]')
        return self.head(self.act(self.conv2(self.act(self.conv1(p)))))

    def forward(self, p: torch.Tensor) -> torch.Tensor:
        return self.density(p).sum(dim=(1, 2, 3))

    def energy_per_pixel(self, p: torch.Tensor) -> torch.Tensor:
        return self.forward(p) / (p.shape[-2] * p.shape[-1])


def simplex_project(x: torch.Tensor) -> torch.Tensor:
    """Full Euclidean simplex projection independently at each pixel."""
    if x.ndim != 4 or x.shape[1] != 3:
        raise ValueError('Expected [B,3,H,W]')
    sorted_x, _ = torch.sort(x, dim=1, descending=True)
    cssv = sorted_x.cumsum(dim=1) - 1
    indices = torch.arange(1, 4, device=x.device, dtype=x.dtype).view(1, 3, 1, 1)
    supports = sorted_x > cssv / indices
    rho = supports.sum(dim=1, keepdim=True).clamp_min(1)
    theta = cssv.gather(1, rho - 1) / rho.to(x.dtype)
    return (x - theta).clamp_min(0)


def tangent(x: torch.Tensor) -> torch.Tensor:
    return x - x.mean(dim=1, keepdim=True)


def state_gradient(model: nn.Module, p: torch.Tensor, create_graph: bool = False):
    with torch.enable_grad():
        x = p.detach().requires_grad_(True)
        g, = torch.autograd.grad(model(x).sum(), x, create_graph=create_graph)
    return g


def finite_difference_direction(model, p, v, delta):
    """Second directional derivative of extensive energy divided by area.

    p +/- delta*v must remain in the simplex. Inputs are interior and
    v has unit global RMS; delta is clipped for feasibility.
    """
    directions = tangent(v)
    scale = directions.square().mean(dim=(1, 2, 3), keepdim=True).sqrt().clamp_min(1e-12)
    directions = directions / scale
    allowed_plus = torch.where(directions < 0, p / (-directions).clamp_min(1e-30), torch.inf)
    allowed_minus = torch.where(directions > 0, p / directions.clamp_min(1e-30), torch.inf)
    bound = torch.minimum(allowed_plus.amin(), allowed_minus.amin()).item()
    h = min(float(delta), 0.45 * bound)
    if h <= 1e-7:
        return {'finite_difference': float('nan'), 'h': h}
    with torch.no_grad():
        fplus = model(p + h * directions).mean()
        fzero = model(p).mean()
        fminus = model(p - h * directions).mean()
        val = ((fplus - 2*fzero + fminus) / (h*h*p.shape[-2]*p.shape[-1])).item()
    return {'finite_difference': val, 'h': h}


def hessian_direction(model, p, v):
    """Exact Hessian directional quadratic form / pixel, no matrix formed."""
    v = tangent(v)
    v = v / v.square().mean(dim=(1, 2, 3), keepdim=True).sqrt().clamp_min(1e-12)
    with torch.enable_grad():
        x = p.detach().requires_grad_(True)
        g, = torch.autograd.grad(model(x).sum(), x, create_graph=True)
        Hv, = torch.autograd.grad((g*v).sum(), x)
    value = (v*Hv).sum(dim=(1, 2, 3)).mean() / (p.shape[-2]*p.shape[-1])
    return value.item()


def sample_masks(size=64, batch=2, seed=7, device='cpu'):
    """Procedurally generated 3-class segmentations: 0 bg, 1 nucleus, 2 surround."""
    rng = np.random.default_rng(seed)
    yy, xx = np.mgrid[:size, :size]
    labels = np.zeros((batch, size, size), dtype=np.int64)
    for b in range(batch):
        count = int(rng.integers(4, 9))
        for _ in range(count):
            cx, cy = rng.uniform(7, size-7, size=2)
            a, c = rng.uniform(3.5, 7.5), rng.uniform(3.5, 7.5)
            d = ((xx-cx)/a)**2 + ((yy-cy)/c)**2
            labels[b][d <= 2.2] = 2
            labels[b][d <= 1.0] = 1
    y = torch.from_numpy(labels).to(device)
    onehot = F.one_hot(y, 3).permute(0, 3, 1, 2).float()
    return y, onehot


def sample_states(cfg: LabConfig, seed: Optional[int] = None):
    """Return clean, mild, strong, and near-uniform valid probability maps."""
    seed = cfg.seed if seed is None else seed
    _, y = sample_masks(cfg.image_size, cfg.batch_size, seed, cfg.device)
    gen = torch.Generator(device='cpu').manual_seed(seed + 12013)
    noise = torch.randn(y.shape, generator=gen).to(y.device)
    smooth = F.avg_pool2d(F.pad(noise, (4,4,4,4), mode='circular'), 9, stride=1)
    smooth = smooth / smooth.square().mean(dim=(1,2,3), keepdim=True).sqrt().clamp_min(1e-8)
    # Softmax input corruption, as in the segmentation training experiment.
    base = (2*y-1)*1.5
    mild = F.softmax(base + 0.65*smooth, dim=1)
    strong = F.softmax(base + 1.8*smooth, dim=1)
    # An interior point permits feasible two-sided Hessian diagnostics.
    interior = 0.75*mild + 0.25/3
    uniform = torch.full_like(y, 1/3)
    return {'clean': y, 'mild': mild, 'strong': strong,
            'interior': interior, 'uniform': uniform}


def _stats(x):
    return {'rms': float(x.square().mean().sqrt().item()),
            'max_abs': float(x.abs().max().item()),
            'mean': float(x.mean().item())}


def gradient_spectrum(g, bins=6):
    """Radial power fractions for tangent gradients; zero spatial mode reported separately."""
    g = tangent(g.detach()).float().cpu()
    _, _, h, w = g.shape
    fourier = torch.fft.rfft2(g, norm='ortho')
    power = fourier.abs().square().mean(dim=(0, 1)).numpy()
    ky = np.fft.fftfreq(h)[:,None]
    kx = np.fft.rfftfreq(w)[None,:]
    kr = np.sqrt(ky*ky + kx*kx)
    # rfft half-plane: approximate fractions (no Hermitian frequency doubling).
    edges = np.linspace(0, float(kr.max()) + 1e-12, bins+1)
    total = max(float(power.sum()), 1e-20)
    fractions = [float(power[(kr >= a) & (kr < b)].sum()/total)
                 for a, b in zip(edges[:-1], edges[1:])]
    return {'radial_power_fractions': fractions, 'radial_edges_cycles_per_pixel': edges.tolist(),
            'dc_fraction': float(power[0,0]/total)}


def descent(model, p0, steps=30, step_size=1.0, max_backtracks=12):
    """Backtracking projected descent, checking *extensive* energy decreases.

    The selected scalar step is common to all samples. This method is for
    energy diagnostics, not differentiation through the descent trajectory.
    """
    p = p0.detach().clone()
    trajectory = []
    with torch.no_grad():
        e0 = model(p).detach()
    trajectory.append({'step':0, 'energy_per_pixel': float((e0/(p.shape[-2]*p.shape[-1])).mean()),
                       'mean_confidence':float(p.amax(dim=1).mean())})
    for i in range(1, steps+1):
        g = state_gradient(model, p).detach()
        old = model(p).detach()
        rate = float(step_size)
        accepted = False
        for _ in range(max_backtracks+1):
            with torch.no_grad():
                candidate = simplex_project(p-rate*g)
                after = model(candidate)
                good = bool(torch.isfinite(after).all() and (after <= old+1e-7).all())
            if good:
                accepted = True
                break
            rate *= 0.5
        if not accepted:
            break
        movement = float((candidate-p).square().mean().sqrt())
        p = candidate.detach()
        trajectory.append({'step':i,
                           'energy_per_pixel':float((after/(p.shape[-2]*p.shape[-1])).mean()),
                           'mean_confidence':float(p.amax(dim=1).mean()),
                           'step_size':rate, 'state_rms_change':movement})
    return p, trajectory


def analyze_model(model, cfg: LabConfig, *, seed: Optional[int] = None):
    """No parameter updates. Suitable before and after any training method."""
    seed = cfg.seed if seed is None else seed
    model.eval()
    states = sample_states(cfg, seed)
    report = {'seed':seed, 'parameters':sum(p.numel() for p in model.parameters()),
              'energy_per_pixel':{}, 'gradients':{}, 'curvature':{}, 'logit_limits':{},
              'descent':{}, 'unconstrained_rays':{}}
    for name, p in states.items():
        with torch.no_grad():
            e = model.energy_per_pixel(p)
        report['energy_per_pixel'][name] = {'mean':float(e.mean()), 'std':float(e.std(unbiased=False))}
        g = tangent(state_gradient(model, p)).detach()
        by_channel = [float(g[:,ch].square().mean().sqrt()) for ch in range(3)]
        means = [float(g[:,ch].mean()) for ch in range(3)]
        report['gradients'][name] = {**_stats(g), 'channel_rms':by_channel,
                                      'channel_means':means,
                                      'spectrum':gradient_spectrum(g, cfg.spectrum_bins)}

    p = states['interior']
    gen = torch.Generator(device='cpu').manual_seed(seed+2333)
    directions = {'white': torch.randn(p.shape, generator=gen).to(p.device)}
    directions['smooth'] = F.avg_pool2d(F.pad(directions['white'], (4,4,4,4), mode='circular'), 9, stride=1)
    directions['constant_channel'] = torch.randn((p.shape[0],3,1,1),generator=gen).to(p.device).expand_as(p)
    # Asymmetric perturbations are not necessarily principal directions.
    for i in range(cfg.curvature_directions):
        directions[f'random_{i}'] = torch.randn(p.shape, generator=gen).to(p.device)
    for name, direction in directions.items():
        fd = finite_difference_direction(model, p, direction, cfg.curvature_delta)
        exact = hessian_direction(model, p, direction)
        report['curvature'][name] = {**fd, 'hvp':exact}

    # Compare energy/gradient behavior as logit differences grow. This
    # deliberately does NOT extrapolate p outside the probability simplex.
    labels = states['clean'].argmax(dim=1)
    direction = 2*states['clean']-1
    for magnitude in [0.0, 1.0, 2.0, 4.0, 8.0, 16.0, 32.0]:
        with torch.enable_grad():
            z = (magnitude*direction).detach().requires_grad_(True)
            q = F.softmax(z, dim=1)
            en = model(q).sum()
            gz, = torch.autograd.grad(en,z)
            gp = tangent(state_gradient(model,q))
        report['logit_limits'][str(magnitude)] = {
            'energy_per_pixel':float(en.detach()/(q.shape[0]*q.shape[-2]*q.shape[-1])),
            'logit_gradient_rms':float(gz.square().mean().sqrt()),
            'prob_gradient_tangent_rms':float(gp.square().mean().sqrt()),
            'mean_max_probability':float(q.detach().amax(dim=1).mean())}

    # Algebraic extension outside the simplex: this is NOT a valid segmentation
    # trajectory, but reveals whether this CNN has coercive asymptotics.
    ray = tangent(states['interior'] - 1/3)
    for multiplier in [0., 1., 2., 4., 8., 16., 32.]:
        x = (1/3 + multiplier*ray).detach()
        with torch.no_grad():
            e = model.energy_per_pixel(x).mean()
        grad = state_gradient(model, x).detach()
        report['unconstrained_rays'][str(multiplier)] = {
            'energy_per_pixel':float(e),
            'gradient_rms':float(tangent(grad).square().mean().sqrt()),
            'min_input':float(x.min()), 'max_input':float(x.max())}

    for name in ['clean','strong']:
        _, hist = descent(model, states[name], cfg.descent_steps,
                          cfg.descent_step_size, cfg.max_backtracks)
        report['descent'][name] = hist
    return report


def plot_report(report, outdir):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(2,2,figsize=(11,7))
    names = list(report['gradients'])
    ax[0,0].bar(names, [report['gradients'][k]['rms'] for k in names])
    ax[0,0].set_title('Initial tangent gradient RMS');ax[0,0].tick_params(axis='x',rotation=30)
    directions = list(report['curvature'])
    ax[0,1].bar(np.arange(len(directions))-.2,[report['curvature'][k]['hvp'] for k in directions],width=.4,label='Hessian-vector')
    ax[0,1].bar(np.arange(len(directions))+.2,[report['curvature'][k]['finite_difference'] for k in directions],width=.4,label='Finite difference')
    ax[0,1].set_xticks(range(len(directions)),directions,rotation=45,ha='right')
    ax[0,1].set_title('Directional curvature per pixel');ax[0,1].legend(fontsize=8)
    for key, h in report['descent'].items():
        ax[1,0].plot([q['step'] for q in h],[q['energy_per_pixel'] for q in h],label=key)
    ax[1,0].set(xlabel='Projected descent step',ylabel='Energy per pixel',title='Energy descent')
    ax[1,0].legend()
    x=[float(k) for k in report['logit_limits']]
    ax[1,1].plot(x,[report['logit_limits'][str(k)]['logit_gradient_rms'] for k in x],label='Logit gradient RMS')
    ax[1,1].plot(x,[report['logit_limits'][str(k)]['prob_gradient_tangent_rms'] for k in x],label='Probability tangent gradient RMS')
    ax[1,1].set(xlabel='Logit magnitude',ylabel='Gradient RMS',title='Softmax saturation')
    ax[1,1].legend(fontsize=8)
    fig.tight_layout()
    path=outdir/'diagnostics.png'
    fig.savefig(path,dpi=150)
    plt.close(fig)
    return path


def plot_state_evolution(model, cfg, outdir, *, seed=None):
    """Saved visual baseline: input state, final state, gradient, and change."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    inputs=sample_states(cfg,seed)
    start=inputs['strong']
    end,history=descent(model,start,cfg.descent_steps,
                        cfg.descent_step_size,cfg.max_backtracks)
    gt=inputs['clean'].argmax(dim=1)
    g=tangent(state_gradient(model,start)).detach()
    fig,ax=plt.subplots(2,4,figsize=(12,6))
    pairs=[('GT classes',gt[0].cpu().numpy(),0,2),
           ('Noisy classes',start[0].argmax(0).cpu().numpy(),0,2),
           ('Final classes',end[0].argmax(0).cpu().numpy(),0,2),
           ('Changed hard labels',(start[0].argmax(0)!=end[0].argmax(0)).float().cpu().numpy(),0,1),
           ('Initial nuclear P',start[0,1].cpu().numpy(),0,1),
           ('Final nuclear P',end[0,1].cpu().numpy(),0,1),
           ('Initial nuclear tangent gradient',g[0,1].cpu().numpy(),None,None),
           ('Nuclear probability change',(end-start)[0,1].cpu().numpy(),None,None)]
    for axis,(title,data,lo,hi) in zip(ax.flat,pairs):
        axis.imshow(data,vmin=lo,vmax=hi,cmap='viridis')
        axis.set_title(title,fontsize=10)
        axis.axis('off')
    fig.tight_layout()
    outdir=Path(outdir)
    outdir.mkdir(parents=True,exist_ok=True)
    path=outdir/'state_evolution.png'
    fig.savefig(path,dpi=140)
    plt.close(fig)
    return path


def save_checkpoint(model, path, *, extra=None):
    path=Path(path)
    path.parent.mkdir(parents=True,exist_ok=True)
    torch.save({'model':model.state_dict(), 'width':model.width,
                'init_head_std':model.init_head_std,'extra':extra or {}},path)


def load_checkpoint(path, device='cpu'):
    try:
        blob=torch.load(path,map_location=device,weights_only=True)
    except TypeError:
        blob=torch.load(path,map_location=device)
    model=SimpleSegmentationEnergy(width=int(blob['width']),
                                    init_head_std=float(blob.get('init_head_std',0.015))).to(device)
    model.load_state_dict(blob['model'])
    return model


def train_demo(model, cfg, steps=80, lr=1e-3):
    """Optional *toy* ranking exercise, not a proposed segmentation trainer.

    It exists to verify that diagnostics can be run before/after training.
    Rank clean synthetic masks below interpolated noisy masks.
    """
    model.train()
    opt=torch.optim.Adam(model.parameters(),lr=lr)
    losses=[]
    for i in range(steps):
        states=sample_states(cfg, seed=cfg.seed+100+i)
        y=states['clean'].detach()
        moderate=states['mild'].detach()
        strong=states['strong'].detach()
        e0=model.energy_per_pixel(y)
        e1=model.energy_per_pixel(moderate)
        e2=model.energy_per_pixel(strong)
        # Ranking clean below mild and strong. Margins are intentionally small
        # because these are average (per-pixel) energies.
        loss=(F.relu(0.002+e0-e1)+F.relu(0.002+e0-e2)).mean()
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        losses.append(float(loss.detach()))
    return losses


def run_seed_survey(cfg, outdir):
    all_reports=[]
    for seed in range(cfg.seed,cfg.seed+cfg.test_seeds):
        torch.manual_seed(seed)
        model=SimpleSegmentationEnergy(cfg.width,cfg.init_head_std).to(cfg.device)
        report=analyze_model(model,cfg,seed=cfg.seed+997)
        all_reports.append({'model_seed':seed,
                            'gradient_rms':report['gradients']['interior']['rms'],
                            'curvature_smooth':report['curvature']['smooth']['hvp'],
                            'energy_gap_strong_clean':report['energy_per_pixel']['strong']['mean']-report['energy_per_pixel']['clean']['mean']})
    Path(outdir).mkdir(parents=True,exist_ok=True)
    (Path(outdir)/'seed_survey.json').write_text(json.dumps(all_reports,indent=2))
    return all_reports


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('command',choices=['diagnose','train-demo','seed-survey'],nargs='?',default='diagnose')
    ap.add_argument('--checkpoint',type=str,default=None)
    ap.add_argument('--out',type=str,default='energy_lab_results')
    ap.add_argument('--seed',type=int,default=7)
    ap.add_argument('--width',type=int,default=12)
    ap.add_argument('--size',type=int,default=64)
    ap.add_argument('--batch-size',type=int,default=2)
    ap.add_argument('--device',default='cpu')
    ap.add_argument('--descent-steps',type=int,default=30)
    ap.add_argument('--curvature-directions',type=int,default=6)
    ap.add_argument('--test-seeds',type=int,default=3)
    ap.add_argument('--train-steps',type=int,default=80)
    args=ap.parse_args()
    cfg=LabConfig(seed=args.seed,width=args.width,image_size=args.size,
                  batch_size=args.batch_size,device=args.device,
                  descent_steps=args.descent_steps,curvature_directions=args.curvature_directions,
                  test_seeds=args.test_seeds)
    torch.manual_seed(cfg.seed)
    torch.set_num_threads(max(1,min(torch.get_num_threads(),4)))
    out=Path(args.out)
    out.mkdir(parents=True,exist_ok=True)
    if args.command=='seed-survey':
        survey=run_seed_survey(cfg,out)
        print('Initialization survey:',survey)
        return
    model=(load_checkpoint(args.checkpoint,cfg.device) if args.checkpoint else
           SimpleSegmentationEnergy(cfg.width,cfg.init_head_std).to(cfg.device))
    if args.command=='train-demo':
        before=analyze_model(model,cfg)
        (out/'before.json').write_text(json.dumps(before,indent=2))
        plot_report(before,out/'before')
        losses=train_demo(model,cfg,args.train_steps)
        save_checkpoint(model,out/'demo_checkpoint.pt',extra={'train_steps':args.train_steps})
        (out/'training_losses.json').write_text(json.dumps(losses))
    report=analyze_model(model,cfg)
    (out/'diagnostics.json').write_text(json.dumps(report,indent=2))
    path=plot_report(report,out)
    state_path=plot_state_evolution(model,cfg,out)
    print('Parameters:',report['parameters'])
    print('Energy/pixel:',{k:round(v['mean'],6) for k,v in report['energy_per_pixel'].items()})
    print('Tangent gradient RMS:',{k:round(v['rms'],6) for k,v in report['gradients'].items()})
    print('Curvature:',{k:round(v['hvp'],6) for k,v in report['curvature'].items()})
    print('Descent endpoints:',{k:(len(v)-1, round(v[0]['energy_per_pixel'],6),round(v[-1]['energy_per_pixel'],6)) for k,v in report['descent'].items()})
    print('Saved report:',out/'diagnostics.json')
    print('Saved plot:',path)
    print('Saved state evolution:',state_path)


if __name__=='__main__':
    main()
