"""Frozen baseline U-Net loading and 2D inference for synthetic volumes."""
from pathlib import Path
import torch
from UNet_baseline import UNet
from mm_fixed_compatible import target_from_stain_labels


def locate_checkpoint(path):
    p = Path(path)
    if p.is_file():
        return p
    if not p.is_dir():
        raise FileNotFoundError(f'Checkpoint path not found: {p}')
    files = [x for ext in ('*.pt', '*.pth', '*.ckpt', '*.bin') for x in p.rglob(ext)]
    if not files:
        raise FileNotFoundError(f'No .pt/.pth/.ckpt/.bin files inside {p}')
    if len(files) == 1:
        return files[0]
    # Avoid silently selecting a different trained model or epoch.
    names = '\n'.join(' - ' + str(f) for f in sorted(files))
    raise ValueError(f'Multiple checkpoints found; provide explicit filename:\n{names}')


def load_unet(path, device='cpu', model_kwargs=None):
    """Strict state_dict loading; fails loudly on architecture mismatch.

    Supports common checkpoint keys. Supply model_kwargs if the original
    training used nondefault UNet channels/features/groups.
    """
    checkpoint_path = locate_checkpoint(path)
    ckpt = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
    state = ckpt
    if isinstance(ckpt, dict):
        for key in ('model_state_dict', 'state_dict', 'model_state', 'model', 'net', 'unet_state_dict'):
            if key in ckpt and isinstance(ckpt[key], dict):
                state = ckpt[key]
                break
    if not isinstance(state, dict) or not state or not all(torch.is_tensor(x) for x in state.values()):
        raise ValueError(f'Unrecognized checkpoint structure in {checkpoint_path}. Keys: {list(ckpt) if isinstance(ckpt, dict) else type(ckpt)}')
    state = {k.removeprefix('module.').removeprefix('_orig_mod.'): v for k, v in state.items()}
    model = UNet(**(model_kwargs or {}))
    model.load_state_dict(state, strict=True)
    model = model.to(device).eval()
    model.requires_grad_(False)
    return model, checkpoint_path


@torch.inference_mode()
def predict_volume(model, images, device, batch_size=4):
    """Input [Z,2,H,W] or [1,Z,2,H,W]; returns CPU [Z,3,H,W]."""
    images = torch.as_tensor(images)
    if images.ndim == 5 and images.shape[0] == 1:
        images = images[0]
    if images.ndim != 4 or images.shape[1] != 2:
        raise ValueError(f'Expected [Z,2,H,W], got {tuple(images.shape)}')
    outputs = []
    for piece in images.split(batch_size):
        logits = model(piece.to(device=device, dtype=torch.float32, non_blocking=True))
        outputs.append(logits.softmax(dim=1).cpu())
    return torch.cat(outputs, dim=0)


def volume_labels_to_semantic(labels):
    labels = torch.as_tensor(labels)
    if labels.ndim == 5 and labels.shape[0] == 1:
        labels = labels[0]
    return target_from_stain_labels(labels).cpu()
