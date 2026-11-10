# Paste this entire cell after Experiment B's training cells.
# Assumes cfg and CHECKPOINT are defined by the original notebook.
from pathlib import Path
import numpy as np
import torch
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, BoundaryNorm
from synthetic_fluor_stains_three_classes_weighted import SyntheticFluorStains
from unet_inference import load_unet, predict_volume
from mm_fixed_compatible import Config, Energy, target_from_stain_labels
from learned_energy import StructuredEnergy, anchored_rollout
from instance_metrics import instance_stats, dice_by_class

# -------- Choose what to inspect --------
VIS_DOMAIN = 'b'             # 'a', 'b', or 'c'
VIS_SEED = 400000           # use held-out seed for domain comparisons
VIS_VOLUME = 0             # synthetic dataset index, zero-based
VIS_SLICE = 8              # 0 ... cfg.z_slices-1
VIS_CHECKPOINT = 'best.pt'  # 'best.pt' or 'latest.pt'
VIS_ITERATIONS = (0, 100, 200, 300, 400)
VIS_SAVE_PNG = False
VIS_DEVICE = cfg.device

# -------- Load learned energy and reconstruct INITIAL fixed energy --------
state_path = Path(cfg.output_dir) / VIS_CHECKPOINT
if not state_path.is_file():
    raise FileNotFoundError(f'Missing {state_path}; train Experiment B or adjust VIS_CHECKPOINT')
state = torch.load(state_path, map_location='cpu', weights_only=False)
saved_tc = state['training_config']
saved_energy_cfg = state['model_config']
# Use parameters from the checkpoint, not possibly modified notebook cfg values.
energy_cfg = Config(**saved_energy_cfg)
learned_energy = StructuredEnergy(
    energy_cfg, n_extra=saved_tc['n_extra'],
    filter_size=saved_tc['extra_filter_size'],
    strength_cap=saved_tc['strength_cap'],
    extra_strength_cap=saved_tc['extra_strength_cap'],
    kernel_raw_cap=saved_tc['kernel_raw_cap'],
    initial_extra_strength=saved_tc['initial_extra_strength'],
    seed=saved_tc['seed'],
    extra_orientations=saved_tc.get('extra_orientations', 16),
).to(VIS_DEVICE)
learned_energy.load_state_dict(state['model_state_dict'], strict=True)
learned_energy.eval().requires_grad_(False)

# Original fixed energy: no extra filters, with strengths 0.5 and 0.5.
# These values are taken from the saved energy config, which was initialized at 0.5.
fixed_energy = Energy(Config(**saved_energy_cfg)).to(VIS_DEVICE)
fixed_energy.eval().requires_grad_(False)
print('Learned strengths:', learned_energy.diagnostics())
print('Original strengths:', saved_energy_cfg['barrier_weight'], saved_energy_cfg['boundary_weight'])
print('Loaded update:', state.get('update'))

# -------- Load a SINGLE synthetic volume and predict its selected slice --------
dataset = SyntheticFluorStains(
    samples_per_epoch=max(VIS_VOLUME + 1, 1), seed=VIS_SEED,
    N=saved_tc['N'], z_slices=saved_tc['z_slices'],
    z_ratio=saved_tc['z_ratio'], n_seeds=saved_tc['n_seeds'],
    class_probs=tuple(saved_tc['class_probs']), domain=VIS_DOMAIN,
)
images, labels = dataset[VIS_VOLUME]
assert 0 <= VIS_SLICE < images.shape[0]
unet, unet_path = load_unet(saved_tc['checkpoint'], VIS_DEVICE)
raw_volume = predict_volume(unet, images, VIS_DEVICE,
                            batch_size=saved_tc['inference_batch_size'])
del unet
raw = raw_volume[VIS_SLICE:VIS_SLICE+1].to(VIS_DEVICE).detach()
target = target_from_stain_labels(torch.as_tensor(labels))[VIS_SLICE].cpu().numpy()

# -------- Track both trajectories from EXACTLY the same original softmax --------
iterations = sorted(set(VIS_ITERATIONS) | {0})
mu = float(saved_tc['anchor_mu'])
tau = float(saved_tc['step_size'])

def snapshots(energy):
    result = {0: raw.detach().cpu().squeeze(0).clone()}
    current = raw
    last = 0
    for t in iterations[1:]:
        current = anchored_rollout(energy, current, raw,
                                   t - last, mu, tau, differentiable=False)
        result[t] = current.detach().cpu().squeeze(0).clone()
        last = t
    return result

fixed_history = snapshots(fixed_energy)
learned_history = snapshots(learned_energy)
raw_cpu = raw.detach().cpu().squeeze(0)

# -------- Compare classes and hard argmax labels at selected iteration --------
SHOW_STEP = max(iterations)    # change to any value from VIS_ITERATIONS
class_names = ('Background', 'Nuclear', 'Nonnuclear')
views = [
    ('Ground truth', None),
    ('Raw U-Net', raw_cpu.numpy()),
    (f'Initial energy ({SHOW_STEP})', fixed_history[SHOW_STEP].numpy()),
    (f'Learned energy ({SHOW_STEP})', learned_history[SHOW_STEP].numpy()),
]
semantic_cmap = ListedColormap(['#202934', '#72ace6', '#eeae60'])
semantic_norm = BoundaryNorm([-0.5, 0.5, 1.5, 2.5], 3)
fig, axes = plt.subplots(4, 4, figsize=(17, 15), constrained_layout=True)
for col, (name, probs) in enumerate(views):
    hard = target if probs is None else probs.argmax(axis=0)
    for row in range(4):
        ax = axes[row, col]
        if row == 0:
            ax.imshow(np.ma.masked_where(hard == -100, hard), cmap=semantic_cmap,
                      norm=semantic_norm, interpolation='nearest')
        else:
            k = row - 1
            channel = (target == k).astype(float) if probs is None else probs[k]
            ax.imshow(channel, cmap='viridis', vmin=0, vmax=1, interpolation='nearest')
        if row == 0:
            if probs is None:
                subtitle = 'Reference instances'
            else:
                st = instance_stats(hard, target)
                subtitle = f"F1={st['instance_f1']:.3f}  merges={st['merge_count']}  splits={st['split_count']}"
            ax.set_title(name + '\n' + subtitle, fontsize=11)
        if col == 0:
            ax.set_ylabel('Argmax labels' if row == 0 else class_names[row - 1], fontsize=12)
        ax.set_xticks([]); ax.set_yticks([])
fig.suptitle(f'Domain {VIS_DOMAIN.upper()}, seed {VIS_SEED}, volume {VIS_VOLUME}, Z={VIS_SLICE}', fontsize=15)
plt.show()

# -------- Optional: visualize how instance performance evolves --------
fig2, axes2 = plt.subplots(1, 2, figsize=(11, 4), constrained_layout=True)
for name, history in [('Initial fixed energy', fixed_history), ('Learned energy', learned_history)]:
    scores = [instance_stats(history[t].argmax(0).numpy(), target) for t in iterations]
    axes2[0].plot(iterations, [s['instance_f1'] for s in scores], marker='o', label=name)
    axes2[1].plot(iterations, [s['merge_count'] for s in scores], marker='o', label=name + ' merges')
    axes2[1].plot(iterations, [s['split_count'] for s in scores], marker='s', linestyle='--', label=name + ' splits')
axes2[0].set(title='Instance F1 on selected slice', xlabel='Iteration', ylabel='F1')
axes2[1].set(title='Instance merge / split counts', xlabel='Iteration', ylabel='Count')
for ax in axes2: ax.legend(fontsize=8); ax.grid(alpha=.2)
plt.show()

if VIS_SAVE_PNG:
    dest = Path(cfg.output_dir) / 'visualizations'
    dest.mkdir(parents=True, exist_ok=True)
    tag = f'{VIS_DOMAIN}_seed{VIS_SEED}_v{VIS_VOLUME}_z{VIS_SLICE}_{VIS_CHECKPOINT.split(".")[0]}'
    fig.savefig(dest / f'{tag}_channels.png', dpi=170, bbox_inches='tight')
    fig2.savefig(dest / f'{tag}_trajectory.png', dpi=170, bbox_inches='tight')
    print('Saved figures in', dest)
