"""Differentiable Modica--Mortola + learned oriented bridge energy.

Run: python train_mm_learned_bridge.py --mode A --steps 100 --device cpu
     python train_mm_learned_bridge.py --mode B --steps 100 --device cuda
     python train_mm_learned_bridge.py --mode C --steps 100 --device cuda

Dependencies: torch, numpy, scipy, matplotlib (for optional plots).
No other project files required. Checkpoints include config and model state_dict.

E = lam*eps*sum|forward differences p|^2 + lam/eps*sum W_gamma(p)
    - barrier_strength/2 * sum ReLU(max_orient K_orient * [p_n,p_nn])^2.

One *outer* step: explicit barrier force, EXACT implicit periodic diffusion
(FFT), then n fixed-point iterations toward the simplex-constrained implicit
local multiwell proximal step. The nonlinear proximal step is APPROXIMATE;
the complete splitting is not exact backward Euler for the combined energy.

A: train barrier strength only, frozen original-style bank.
B: train continuous geometric widths, length scale, channel amplitude, strength.
C: train one canonical residual kernel, rotate into a bank; strength trainable.
All kernels have per-channel zero DC and positive-weight L1 normalization.
"""
from __future__ import annotations
import argparse
import json
import math
from dataclasses import asdict, dataclass, fields
from pathlib import Path
from typing import Optional

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F


@dataclass
class Config:
    mode: str = 'A'
    size: int = 48
    batch_size: int = 2
    nuclei_min: int = 2
    nuclei_max: int = 4
    radius_min: float = 3.5
    radius_max: float = 6.0
    halo_width: float = 2.0
    corruption: str = 'mixed'  # dilation, erosion, bridge, mixed
    dilation_radius: int = 2
    erosion_radius: int = 1
    bridge_width: int = 2
    input_confidence: float = .90
    epsilon: float = .5
    lam: float = 1.0
    well_cubic_weight: float = 1.0
    barrier_init_weight: float = .1
    barrier_width: float = 3.0
    nuclear_band_width: float = 2.0
    barrier_length: int = 13
    barrier_longitudinal_sigma: float = 3.5
    barrier_orientations: int = 8
    # Outer time step and inner well fixed-point solver
    outer_steps: int = 3
    dt: float = .05
    well_iterations: int = 3
    well_relaxation: float = .25
    # User may set first two to zero, final to 1
    rollout_weights: tuple[float, ...] = (0.0, 0.0, 1.0)
    stationarity_weight: float = 0.0
    stationarity_steps: int = 1
    kernel_reg_weight: float = 0.01
    nuclear_loss_weight: float = 3.0
    steps: int = 200
    lr: float = .003
    grad_clip: float = 5.0
    eval_every: int = 20
    eval_rollout_steps: int = 15
    seed: int = 13
    device: str = 'cpu'
    output_dir: str = 'mm_bridge_run'

    def validate(self):
        if self.mode not in ('A','B','C'): raise ValueError('mode A/B/C required')
        if self.size < self.barrier_length or self.size < 24: raise ValueError('image too small')
        if self.outer_steps < 1 or len(self.rollout_weights) != self.outer_steps:
            raise ValueError('rollout_weights must have one entry per outer step')
        if not any(w > 0 for w in self.rollout_weights) or any(w < 0 for w in self.rollout_weights):
            raise ValueError('rollout weights must be nonnegative with positive sum')
        if self.dt <= 0 or self.epsilon <= 0 or self.lam < 0: raise ValueError('bad MM settings')
        if self.well_iterations < 1 or not (0 < self.well_relaxation <= 1):
            raise ValueError('bad implicit well settings')
        if not 1/3 < self.input_confidence < 1: raise ValueError('bad input confidence')
        if self.corruption not in ('dilation','erosion','bridge','mixed'):
            raise ValueError('bad corruption')
        if self.barrier_orientations < 1 or self.barrier_length % 2 != 1:
            raise ValueError('orientations >= 1 and odd length required')


def simplex_project(v):
    """Euclidean projection onto the 3-class simplex at every pixel."""
    v_sort, _ = v.sort(dim=1, descending=True)
    cssv = v_sort.cumsum(dim=1) - 1
    seq = torch.arange(1, v.shape[1]+1, device=v.device, dtype=v.dtype).view(1,-1,1,1)
    active = v_sort > cssv / seq
    rho = active.long().sum(dim=1, keepdim=True).clamp_min(1)
    theta = cssv.gather(1, rho-1) / rho
    return (v-theta).clamp_min(0)


def well_potential(p, gamma):
    a,b,c = p[:,0], p[:,1], p[:,2]
    return a*a*b*b + a*a*c*c + b*b*c*c + gamma*a*b*c


def well_gradient(p, gamma):
    a,b,c = p[:,0:1],p[:,1:2],p[:,2:3]
    return torch.cat((2*a*(b*b+c*c)+gamma*b*c,
                      2*b*(a*a+c*c)+gamma*a*c,
                      2*c*(a*a+b*b)+gamma*a*b), dim=1)


def implicit_diffusion(p, dt, lam, eps):
    """Solve (I-2*dt*lam*eps*discrete_Delta) q=p, periodic boundary."""
    h,w=p.shape[-2:]
    fy=torch.fft.fftfreq(h,device=p.device,dtype=p.dtype)
    fx=torch.fft.rfftfreq(w,device=p.device,dtype=p.dtype)
    # -Delta eigenvalues for four-neighbor periodic discrete Laplacian.
    negative_delta=4*(torch.sin(math.pi*fy)[:,None]**2 +
                       torch.sin(math.pi*fx)[None,:]**2)
    denom=1 + 2*dt*lam*eps*negative_delta
    q=torch.fft.irfft2(torch.fft.rfft2(p,dim=(-2,-1))/denom,
                       s=(h,w),dim=(-2,-1))
    return q


def _base_template_torch(cfg, dtype, device, bw=None, nw=None, sigma=None, amp=None):
    """Soft, differentiable NN|BBB|NN cross-section, centered at x=0."""
    n=cfg.barrier_length
    x=torch.arange(n,device=device,dtype=dtype)-(n//2)
    y=x
    bw=torch.as_tensor(cfg.barrier_width,device=device,dtype=dtype) if bw is None else bw
    nw=torch.as_tensor(cfg.nuclear_band_width,device=device,dtype=dtype) if nw is None else nw
    sigma=torch.as_tensor(cfg.barrier_longitudinal_sigma,device=device,dtype=dtype) if sigma is None else sigma
    amp=torch.as_tensor(1.,device=device,dtype=dtype) if amp is None else amp
    sharp=.40
    outer=torch.sigmoid((bw/2+nw-x.abs())/sharp)
    inner=torch.sigmoid((bw/2-x.abs())/sharp)
    bands=(outer-inner).clamp_min(0)
    middle=inner
    # Center the cross-sectional feature over the whole kernel, separately per channel.
    envelope=torch.exp(-.5*(y/sigma)**2)
    base=torch.stack((bands[None,:]*envelope[:,None],
                      amp*middle[None,:]*envelope[:,None]),dim=0)
    base=base-base.mean(dim=(-2,-1),keepdim=True)
    return base


def _rotate_bank(base, num_angles):
    """Rotate one 2-channel template differentiably, preserving shared geometry."""
    angles=torch.arange(num_angles,device=base.device,dtype=base.dtype)*math.pi/num_angles
    rotations=torch.zeros((num_angles,2,3),device=base.device,dtype=base.dtype)
    rotations[:,0,0]=torch.cos(angles)
    rotations[:,0,1]=-torch.sin(angles)
    rotations[:,1,0]=torch.sin(angles)
    rotations[:,1,1]=torch.cos(angles)
    grid=F.affine_grid(rotations,(num_angles,2,*base.shape[-2:]),align_corners=True)
    return F.grid_sample(base.unsqueeze(0).expand(num_angles,-1,-1,-1),grid,
                         align_corners=True,padding_mode='zeros')


def _normalized_bank(bank):
    bank=bank-bank.mean(dim=(-2,-1),keepdim=True)
    norm=bank.clamp_min(0).sum(dim=(1,2,3),keepdim=True).clamp_min(1e-8)
    return bank/norm


class BridgeEnergy(nn.Module):
    def __init__(self,cfg: Config):
        super().__init__()
        cfg.validate(); self.cfg=cfg
        # Softplus inverse: init effective positive strength at requested value.
        v=math.log(math.expm1(max(cfg.barrier_init_weight,1e-8)))
        self.raw_strength=nn.Parameter(torch.tensor(v,dtype=torch.float32))
        base=_base_template_torch(cfg,torch.float32,torch.device('cpu'))
        self.register_buffer('base_kernel',base.detach())
        if cfg.mode=='A':
            self.register_buffer('fixed_bank',_normalized_bank(_rotate_bank(base,cfg.barrier_orientations)).detach())
        elif cfg.mode=='B':
            self.raw_bw=nn.Parameter(torch.tensor(math.log(math.expm1(cfg.barrier_width)),dtype=torch.float32))
            self.raw_nw=nn.Parameter(torch.tensor(math.log(math.expm1(cfg.nuclear_band_width)),dtype=torch.float32))
            self.raw_sigma=nn.Parameter(torch.tensor(math.log(math.expm1(cfg.barrier_longitudinal_sigma)),dtype=torch.float32))
            self.raw_amp=nn.Parameter(torch.tensor(0.,dtype=torch.float32))
        else:
            self.residual=nn.Parameter(torch.zeros_like(base))

    def strength(self): return F.softplus(self.raw_strength)

    def kernels(self):
        if self.cfg.mode=='A': return self.fixed_bank
        if self.cfg.mode=='B':
            # Positive, bounded below; slight over-width is possible, inspect learned values.
            bw=F.softplus(self.raw_bw)+1e-4
            nw=F.softplus(self.raw_nw)+1e-4
            sig=F.softplus(self.raw_sigma)+1e-4
            amp=torch.exp(self.raw_amp.clamp(-2,2))
            base=_base_template_torch(self.cfg,self.base_kernel.dtype,self.base_kernel.device,
                                      bw=bw,nw=nw,sigma=sig,amp=amp)
        else:
            base=self.base_kernel+self.residual
        return _normalized_bank(_rotate_bank(base,self.cfg.barrier_orientations))

    def barrier_energy(self,p):
        k=self.kernels().to(dtype=p.dtype)
        r=k.shape[-1]//2
        pair=F.pad(p[:,1:3],(r,r,r,r),mode='circular')
        response=F.conv2d(pair,k).amax(dim=1)
        return -.5*self.strength()*F.relu(response).square().flatten(1).sum(1)

    def terms(self,p):
        cfg=self.cfg
        dx=torch.roll(p,-1,dims=-1)-p
        dy=torch.roll(p,-1,dims=-2)-p
        d=cfg.lam*cfg.epsilon*(dx.square()+dy.square()).flatten(1).sum(1)
        w=cfg.lam/cfg.epsilon*well_potential(p,cfg.well_cubic_weight).flatten(1).sum(1)
        return {'diffusion':d,'well':w,'barrier':self.barrier_energy(p)}

    def forward(self,p): return sum(self.terms(p).values())

    def kernel_penalty(self):
        if self.cfg.mode=='C': return self.residual.square().mean()
        if self.cfg.mode=='B':
            return ((F.softplus(self.raw_bw)-self.cfg.barrier_width)**2+
                    (F.softplus(self.raw_nw)-self.cfg.nuclear_band_width)**2+
                    (F.softplus(self.raw_sigma)-self.cfg.barrier_longitudinal_sigma)**2+
                    self.raw_amp.square())/4
        return self.raw_strength.new_zeros(())

    def geometry(self):
        with torch.no_grad():
            d={'strength':float(self.strength())}
            if self.cfg.mode=='B':
                d.update(barrier_width=float(F.softplus(self.raw_bw)),
                         nuclear_band_width=float(F.softplus(self.raw_nw)),
                         longitudinal_sigma=float(F.softplus(self.raw_sigma)),
                         channel_ratio=float(torch.exp(self.raw_amp.clamp(-2,2))))
            if self.cfg.mode=='C':d['residual_rms']=float(self.residual.square().mean().sqrt())
            return d


def step_energy(model,p,*,differentiable=True):
    """Differentiable operator splitting: explicit bridge, implicit diffusion, implicit-ish well.

    During training, gradients propagate through both the barrier force and
    all well solver iterates; during evaluation, return detached state.
    """
    cfg=model.cfg
    with torch.enable_grad():
        x=p if differentiable else p.detach()
        if not x.requires_grad: x=x.requires_grad_(True)
        bridge=model.barrier_energy(x).sum()
        grad,=torch.autograd.grad(bridge,x,create_graph=differentiable)
        explicit=x-cfg.dt*grad
        # Diffusion is exact backward Euler for the discrete Dirichlet term.
        diffuse=implicit_diffusion(explicit,cfg.dt,cfg.lam,cfg.epsilon)
        # Diffusion preserves channel sums only when they were already one;
        # explicit barrier gradients may move off the affine simplex plane.
        diffuse=simplex_project(diffuse)
        q=diffuse
        s=cfg.dt*cfg.lam/cfg.epsilon
        for _ in range(cfg.well_iterations):
            target=simplex_project(diffuse-s*well_gradient(q,cfg.well_cubic_weight))
            q=(1-cfg.well_relaxation)*q+cfg.well_relaxation*target
    return q if differentiable else q.detach()


def rollout(model,p,n_steps,*,differentiable=True):
    states=[]
    for _ in range(n_steps):
        p=step_energy(model,p,differentiable=differentiable)
        states.append(p)
    return states


def _disk_morph(mask,radius,expand):
    if radius<=0:return mask.clone()
    offsets=[(i,j) for i in range(-radius,radius+1)
             for j in range(-radius,radius+1) if i*i+j*j<=radius*radius]
    stack=torch.stack([torch.roll(mask,(i,j),(-2,-1)) for i,j in offsets])
    return stack.any(dim=0) if expand else stack.all(dim=0)


def synthetic_batch(cfg,seed):
    """Return (corrupted soft probabilities, clean onehot, target integer labels).

    Nuclei are non-overlapping random ellipses with nonnuclear halos. Corruption
    may dilate, erode, or join a pair using a thin bridge (when two are present).
    Inputs are intentionally softened to provide a separator signal.
    """
    rng=np.random.default_rng(seed)
    yy,xx=np.mgrid[:cfg.size,:cfg.size]
    labels=[]; corrupted=[]
    for b in range(cfg.batch_size):
        nucleus=np.zeros((cfg.size,cfg.size),bool)
        halo=np.zeros_like(nucleus)
        centers=[]
        margin=cfg.radius_max+cfg.halo_width+cfg.dilation_radius+2
        wanted=int(rng.integers(cfg.nuclei_min,cfg.nuclei_max+1))
        for _ in range(wanted*65):
            if len(centers)>=wanted:break
            rx=float(rng.uniform(cfg.radius_min,cfg.radius_max))
            ry=float(rng.uniform(cfg.radius_min,cfg.radius_max))
            cx=float(rng.uniform(margin,cfg.size-margin))
            cy=float(rng.uniform(margin,cfg.size-margin))
            angle=float(rng.uniform(0,math.pi))
            dx=xx-cx;dy=yy-cy;cs=math.cos(angle);sn=math.sin(angle)
            xp=cs*dx+sn*dy;yp=-sn*dx+cs*dy
            inner=(xp/rx)**2+(yp/ry)**2<=1
            outer=(xp/(rx+cfg.halo_width))**2+(yp/(ry+cfg.halo_width))**2<=1
            # Reject overlaps but allow close nuclei for bridge patterns.
            if np.any(outer&halo):continue
            nucleus|=inner;halo|=outer;centers.append((cx,cy))
        lab=np.zeros_like(nucleus,dtype=np.int64);lab[halo]=2;lab[nucleus]=1
        dirty=lab.copy()
        kind=cfg.corruption
        if kind=='mixed':kind=str(rng.choice(['dilation','erosion','bridge']))
        if kind in ('dilation','erosion'):
            m=torch.from_numpy(nucleus.copy())
            r=cfg.dilation_radius if kind=='dilation' else cfg.erosion_radius
            modified=_disk_morph(m,r,kind=='dilation').numpy()
            if kind=='dilation': dirty[modified]=1
            else:
                removed=nucleus & ~modified
                dirty[removed]=2
        elif kind=='bridge' and len(centers)>=2:
            pair=min(((i,j) for i in range(len(centers)) for j in range(i+1,len(centers))),
                     key=lambda ij:sum((centers[ij[0]][d]-centers[ij[1]][d])**2 for d in (0,1)))
            (x1,y1),(x2,y2)=[centers[z] for z in pair]
            gap=np.sqrt((xx-x1)**2+(yy-y1)**2) # no effect, use line distances below
            den=(x2-x1)**2+(y2-y1)**2+1e-9
            t=np.clip(((xx-x1)*(x2-x1)+(yy-y1)*(y2-y1))/den,0,1)
            dline=np.hypot(xx-(x1+t*(x2-x1)),yy-(y1+t*(y2-y1)))
            dirty[dline<=cfg.bridge_width/2]=1
        labels.append(lab);corrupted.append(dirty)
    y=torch.tensor(np.stack(labels),device=cfg.device,dtype=torch.long)
    d=torch.tensor(np.stack(corrupted),device=cfg.device,dtype=torch.long)
    clean=F.one_hot(y,3).permute(0,3,1,2).to(torch.float32)
    corrupted_hot=F.one_hot(d,3).permute(0,3,1,2).to(torch.float32)
    alpha=cfg.input_confidence
    p0=alpha*corrupted_hot+(.5*(1-alpha))*(1-corrupted_hot)
    return p0,clean,y


def probability_loss(p,y,cfg):
    w=p.new_tensor([1.,cfg.nuclear_loss_weight,1.]).view(1,3,1,1)
    return ((p-y).square()*w).mean()


def nuclear_dice(p,y):
    a=p.argmax(1)==1;b=y==1
    num=(2*(a&b).sum(dim=(-2,-1)).float()+1e-8)
    den=(a.sum(dim=(-2,-1))+b.sum(dim=(-2,-1))).float()+1e-8
    return (num/den).mean().item()


def train_step(model,opt,cfg,seed):
    model.train()
    p0,clean,_=synthetic_batch(cfg,seed)
    states=rollout(model,p0,cfg.outer_steps,differentiable=True)
    sw=sum(cfg.rollout_weights)
    correction=sum(float(w)*probability_loss(p,clean,cfg) for w,p in zip(cfg.rollout_weights,states))/sw
    if cfg.stationarity_weight:
        stay_states=rollout(model,clean,cfg.stationarity_steps,differentiable=True)
        stationarity=sum(probability_loss(p,clean,cfg) for p in stay_states)/len(stay_states)
    else:stationarity=correction.new_zeros(())
    kreg=model.kernel_penalty()
    loss=correction+cfg.stationarity_weight*stationarity+cfg.kernel_reg_weight*kreg
    opt.zero_grad(set_to_none=True)
    loss.backward()
    grad_norm=float(torch.nn.utils.clip_grad_norm_(model.parameters(),cfg.grad_clip))
    opt.step()
    return {'loss':float(loss.detach()),'correction':float(correction.detach()),
            'stationarity':float(stationarity.detach()),'kernel_reg':float(kreg.detach()),
            'grad_norm':grad_norm}


def evaluate(model,cfg,seed,*,plot_path:Optional[Path]=None):
    model.eval()
    p0,y,target=synthetic_batch(cfg,seed)
    states=rollout(model,p0,cfg.eval_rollout_steps,differentiable=False)
    clean_states=rollout(model,y,min(cfg.eval_rollout_steps,10),differentiable=False)
    with torch.no_grad():
        out={'input_dice':nuclear_dice(p0,target),
             'output_dice':nuclear_dice(states[cfg.outer_steps-1],target),
             'long_dice':nuclear_dice(states[-1],target),
             'clean_long_dice':nuclear_dice(clean_states[-1],target),
             'clean_one_step_rms':float((clean_states[0]-y).square().mean().sqrt()),
             'nuclear_fraction_long':float((states[-1].argmax(1)==1).float().mean()),
             'energy_input_per_pixel':float(model(p0).mean()/cfg.size**2),
             'energy_corrected_per_pixel':float(model(states[cfg.outer_steps-1]).mean()/cfg.size**2),
             'geometry':model.geometry()}
    if plot_path is not None:
        import matplotlib.pyplot as plt
        fig,ax=plt.subplots(1,5,figsize=(14,3))
        for a,p,title in zip(ax,[y,p0,states[0],states[cfg.outer_steps-1],states[-1]],
                             ['Clean target','Corrupted input','After 1','After 3','Long rollout']):
            a.imshow(p[0].argmax(0).detach().cpu().numpy(),vmin=0,vmax=2,interpolation='nearest')
            a.set_title(title);a.axis('off')
        fig.tight_layout();fig.savefig(plot_path,dpi=150);plt.close(fig)
    return out


def train(cfg):
    cfg.validate();torch.manual_seed(cfg.seed)
    out=Path(cfg.output_dir);out.mkdir(parents=True,exist_ok=True)
    model=BridgeEnergy(cfg).to(cfg.device)
    opt=torch.optim.Adam(model.parameters(),lr=cfg.lr)
    history=[]
    start=evaluate(model,cfg,cfg.seed+10000,plot_path=out/'before.png')
    print('INITIAL',json.dumps(start))
    for t in range(1,cfg.steps+1):
        m=train_step(model,opt,cfg,cfg.seed+t)
        if t==1 or t%cfg.eval_every==0 or t==cfg.steps:
            ev=evaluate(model,cfg,cfg.seed+10000,plot_path=out/'after.png')
            item={'step':t,**m,**ev};history.append(item)
            print('STEP',t,json.dumps(item))
    torch.save({'config':asdict(cfg),'state_dict':model.state_dict()},out/'model.pt')
    (out/'history.json').write_text(json.dumps(history,indent=2))
    (out/'config.json').write_text(json.dumps(asdict(cfg),indent=2))
    return model,history


def load_checkpoint(path,device='cpu'):
    ck=torch.load(path,map_location=device,weights_only=False)
    cfg=Config(**ck['config']);cfg.device=str(device)
    model=BridgeEnergy(cfg).to(device)
    model.load_state_dict(ck['state_dict']);model.eval()
    return model,cfg


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--mode',choices=['A','B','C'],default='A')
    parser.add_argument('--steps',type=int,default=200)
    parser.add_argument('--size',type=int,default=48)
    parser.add_argument('--batch-size',type=int,default=2)
    parser.add_argument('--device',default='cpu')
    parser.add_argument('--out',default='mm_bridge_run')
    parser.add_argument('--corruption',choices=['dilation','erosion','bridge','mixed'],default='mixed')
    parser.add_argument('--stationarity-weight',type=float,default=0.)
    parser.add_argument('--dt',type=float,default=.05)
    args=parser.parse_args()
    cfg=Config(mode=args.mode,steps=args.steps,size=args.size,batch_size=args.batch_size,
               device=args.device,output_dir=args.out,corruption=args.corruption,
               stationarity_weight=args.stationarity_weight,dt=args.dt)
    train(cfg)


if __name__=='__main__': main()
