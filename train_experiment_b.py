"""Reusable experiment B trainer: frozen U-Net, anchored shared structured energy.

Caveats: truncated backprop ignores derivatives through random detached warmup.
Caching synthetic volumes avoids repeated expensive generation within training.
"""
from __future__ import annotations
import copy, csv, json, random
from dataclasses import dataclass, asdict, field, replace
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import DataLoader
from mm_fixed_compatible import Config, synthetic_logits, target_from_stain_labels
from synthetic_fluor_stains_three_classes_weighted import SyntheticFluorStains
from unet_inference import load_unet, predict_volume
from learned_energy import StructuredEnergy, anchored_rollout
from experiment_b_loss import supervised_loss
from train_mm_real import corrupt_warmup_state, RealTrainConfig
from instance_metrics import score_batch

@dataclass
class TrainBConfig:
    checkpoint: str = ''
    output_dir: str = '/content/drive/MyDrive/Colab Notebooks/proj_26/experiment_b'
    device: str = 'cuda'
    seed: int = 2400
    train_seed: int = 200000
    val_seed: int = 300000
    train_volumes: int = 8  # cache 8*16 slices; increase after pilot
    val_volumes: int = 3
    N: int = 192
    z_slices: int = 16
    z_ratio: float = 2.
    n_seeds: int = 45
    class_probs: tuple = (2/5, 1/5, 2/5)
    train_domain: str = 'a'
    val_domain: str = 'a'
    n_extra: int = 3
    extra_filter_size: int = 13
    strength_cap: float = 1.5
    extra_strength_cap: float = 0.3
    initial_extra_strength: float = 0.002
    kernel_raw_cap: float = 2.
    anchor_mu: float = 0.4
    step_size: float = 0.02
    updates: int = 250
    learning_rate: float = 1e-3
    momentum: float = 0.
    weight_decay: float = 1e-5
    batch_size: int = 2
    warmups: tuple = (0, 20, 80, 160, 300)
    differentiable_steps: int = 4
    source_mode: str = 'mixed'  # 'corrupted', 'unet', or 'mixed'
    prob_unet_in_mixed: float = .5
    gt_blur_range: tuple = (0., 1.5)
    gt_spectral_range: tuple = (0.4, 1.2)
    # Restart corruptions are on the warmup state; zero allows a no-restart subset.
    restart_probability: float = .5
    restart_blur_range: tuple = (0., .8)
    restart_spectral_range: tuple = (0., .5)
    strength_drift_penalty: float = 1e-3
    extra_strength_penalty: float = 1e-3
    kernel_smoothness_penalty: float = 1e-4
    grad_clip: float = 5.
    eval_every: int = 25
    val_horizons: tuple = (0, 50, 150, 300, 400)
    validation_max_slices: int = 8
    inference_batch_size: int = 4
    energy: dict = field(default_factory=lambda: dict(
        lam=.2, epsilon=.5, well_cubic_weight=1.,
        barrier_weight=.5, barrier_width=5, nuclear_band_width=2,
        barrier_length=13, barrier_longitudinal_sigma=3.5, barrier_orientations=8,
        boundary_weight=.5, boundary_middle_width=5, boundary_outer_width=2,
        boundary_orientations=16, learn_strengths=False, learn_residuals=False))


def _write_csv(path, rows):
    if not rows: return
    path.parent.mkdir(parents=True, exist_ok=True)
    keys = list(dict.fromkeys(k for row in rows for k in row))
    with path.open('w', newline='') as f:
        writer = csv.DictWriter(f, keys); writer.writeheader(); writer.writerows(rows)


def prepare_cache(unet, tc, seed, n_volumes, domain):
    """Store CPU [Z,3,H,W] probabilities, targets and original 3-channel labels."""
    ds = SyntheticFluorStains(samples_per_epoch=n_volumes, seed=seed,
        N=tc.N, z_slices=tc.z_slices, z_ratio=tc.z_ratio,
        n_seeds=tc.n_seeds, class_probs=tc.class_probs, domain=domain)
    records=[]
    for vi in range(n_volumes):
        image, label = ds[vi]
        image = torch.as_tensor(image).float()
        label = torch.as_tensor(label).float()
        pred = predict_volume(unet, image, tc.device, tc.inference_batch_size)
        target = target_from_stain_labels(label)
        for zi in range(len(target)):
            records.append((pred[zi], target[zi].cpu(), label[zi, 2].cpu(), vi, zi))
        print(f'Cached {vi+1}/{n_volumes} volumes ({domain})', flush=True)
    return records


def make_initial(target, energy_cfg, tc, rng):
    b = float(rng.uniform(*tc.gt_blur_range))
    noise = float(rng.uniform(*tc.gt_spectral_range))
    source_cfg = replace(energy_cfg, gt_blur_sigma=b, perturb_strength=noise)
    return synthetic_logits(target.cpu(), source_cfg, seed=int(rng.integers(1, 2**31-1))).softmax(1)


def evaluate(model, cache, tc, max_slices=None):
    model.eval()
    rows=[]
    selected = cache if max_slices is None else [cache[int(i)] for i in np.linspace(0, len(cache)-1, min(max_slices, len(cache)), dtype=int)]
    for pred, target, _, vi, zi in selected:
        original=pred[None].to(tc.device)
        cur=original
        last=0
        for horizon in sorted(set((0,)+tuple(tc.val_horizons))):
            if horizon>last:
                cur=anchored_rollout(model, cur, original, horizon-last,
                    tc.anchor_mu, tc.step_size, differentiable=False)
            last=horizon
            stats=score_batch(cur.detach().cpu().numpy(), target[None].numpy())[0]
            rows.append({'volume':vi,'slice':zi,'iteration':horizon,**stats})
    summary=[]
    for horizon in sorted(set((0,)+tuple(tc.val_horizons))):
        subset=[r for r in rows if r['iteration']==horizon]
        tp=sum(r['tp'] for r in subset); fp=sum(r['fp'] for r in subset); fn=sum(r['fn'] for r in subset)
        n_pred=sum(r['pred_instances'] for r in subset); n_true=sum(r['gt_instances'] for r in subset)
        summary.append({'iteration':horizon,
          'micro_instance_f1':2*tp/max(1,2*tp+fp+fn),
          'merge_rate_pooled':sum(r['merge_count'] for r in subset)/max(1,n_pred),
          'split_rate_pooled':sum(r['split_count'] for r in subset)/max(1,n_true),
          **{f'mean_dice_{k}':float(np.nanmean([r[f'dice_{k}'] for r in subset])) for k in range(3)}})
    return rows,summary


def train(tc: TrainBConfig):
    if tc.source_mode not in ('mixed','corrupted','unet'): raise ValueError('Bad source_mode')
    if not tc.checkpoint: raise ValueError('Set checkpoint to selected pretrained U-Net file')
    if not 0 <= tc.prob_unet_in_mixed <= 1: raise ValueError('Invalid mixture probability')
    torch.manual_seed(tc.seed); np.random.seed(tc.seed); random.seed(tc.seed)
    rng=np.random.default_rng(tc.seed)
    device=torch.device(tc.device)
    out=Path(tc.output_dir); out.mkdir(parents=True,exist_ok=True)
    (out/'config.json').write_text(json.dumps(asdict(tc),indent=2))
    unet,ckpt=load_unet(tc.checkpoint,device)
    print('Frozen U-Net:',ckpt)
    cfg=Config(**tc.energy)
    model=StructuredEnergy(cfg,tc.n_extra,tc.extra_filter_size,tc.strength_cap,
      tc.extra_strength_cap,tc.kernel_raw_cap,tc.initial_extra_strength,tc.seed).to(device)
    params=list(model.parameters())
    optimizer=torch.optim.SGD(params,lr=tc.learning_rate,momentum=tc.momentum,weight_decay=tc.weight_decay)
    print('Trainable parameter count:',sum(p.numel() for p in params))
    print('Initial strengths:',model.diagnostics())
    train_cache=prepare_cache(unet,tc,tc.train_seed,tc.train_volumes,tc.train_domain)
    val_cache=prepare_cache(unet,tc,tc.val_seed,tc.val_volumes,tc.val_domain)
    del unet
    initial_fixed=model.strengths().detach().clone()
    history=[]; val_history=[]; best=-float('inf')
    rc=RealTrainConfig(device=str(device), restart_corruption_probability=1.,
       restart_blur_range=tc.restart_blur_range,
       restart_spectral_range=tc.restart_spectral_range)
    for update in range(1,tc.updates+1):
        model.train()
        indices=rng.integers(0,len(train_cache),size=tc.batch_size)
        batch=[train_cache[int(j)] for j in indices]
        raw=torch.stack([x[0] for x in batch]).to(device)
        target=torch.stack([x[1] for x in batch]).to(device)
        mask=torch.stack([x[2] for x in batch]).to(device)
        source='unet' if (tc.source_mode=='unet' or
            (tc.source_mode=='mixed' and rng.random()<tc.prob_unet_in_mixed)) else 'corrupted'
        original=raw if source=='unet' else make_initial(target,cfg,tc,rng).to(device)
        # Anchor is ALWAYS initial prediction, never re-anchored at restart.
        warmup=int(rng.choice(tc.warmups))
        current=anchored_rollout(model,original,original,warmup,
                                 tc.anchor_mu,tc.step_size,False)
        # Second corruption is optional and applied to evolved state, not ground truth.
        if rng.random()<tc.restart_probability:
            current=corrupt_warmup_state(current,cfg,rc,rng)
        current=current.detach()
        for _ in range(tc.differentiable_steps):
            current=anchored_rollout(model,current,original,1,
                                     tc.anchor_mu,tc.step_size,True)
        loss,parts=supervised_loss(current,target,mask)
        strengths=model.strengths()
        drift=(strengths-initial_fixed).square().sum()*tc.strength_drift_penalty
        extra_pen=model.extra_strengths().square().sum()*tc.extra_strength_penalty
        kernels=model.normalized_extra_kernels()
        smooth=(kernels[...,1:,:]-kernels[...,:-1,:]).square().mean()+\
               (kernels[...,:,1:]-kernels[...,:,:-1]).square().mean()
        total=loss+drift+extra_pen+tc.kernel_smoothness_penalty*smooth
        optimizer.zero_grad(set_to_none=True)
        total.backward()
        grad_norm=torch.nn.utils.clip_grad_norm_(params,tc.grad_clip)
        if not torch.isfinite(total).item() or not torch.isfinite(grad_norm).item():
            raise FloatingPointError(f'Nonfinite loss/gradient at update {update}')
        optimizer.step()
        row={'update':update,'source':source,'warmup':warmup,'loss':float(total.detach()),
             'ce':float(parts['ce']),'soft_dice_loss':float(parts['dice_loss']),
             'grad_norm':float(grad_norm),**model.diagnostics()}
        history.append(row)
        if update==1 or update%tc.eval_every==0 or update==tc.updates:
            raw_rows,summary=evaluate(model,val_cache,tc,tc.validation_max_slices)
            for r in summary: val_history.append({'update':update,**r})
            # Select model by long-horizon instance F1, not training Dice.
            select=max(summary,key=lambda r:r['iteration'])
            print(f"step {update}/{tc.updates}: loss={float(total):.4f}, grad={float(grad_norm):.3g}, "
                  f"val F1@{select['iteration']}={select['micro_instance_f1']:.4f}",flush=True)
            state={'model_state_dict':model.state_dict(),'model_config':asdict(cfg),
              'training_config':asdict(tc),'update':update,'optimizer_state_dict':optimizer.state_dict()}
            torch.save(state,out/'latest.pt')
            if select['micro_instance_f1']>best:
                best=select['micro_instance_f1']; torch.save(state,out/'best.pt')
                _write_csv(out/'best_validation_slices.csv',raw_rows)
            _write_csv(out/'train_history.csv',history)
            _write_csv(out/'validation_history.csv',val_history)
        elif update%10==0:
            print(f"step {update}/{tc.updates}: loss={float(total):.4f}, grad={float(grad_norm):.3g}",flush=True)
    return model,history,val_history
