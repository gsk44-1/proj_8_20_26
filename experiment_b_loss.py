"""Faithful probability-domain implementation of user's U-Net training loss."""
import torch
from torch.nn import functional as F


def supervised_loss(prob, target, supervision_mask, ignore_index=-100,
                    ce_class_weights=(1., 2., 1.), dice_class_weights=(.2, .4, .4),
                    spatial_boost=2., dice_multiplier=1.5, eps=1e-7):
    valid = target != ignore_index
    safe = target.masked_fill(~valid, 0).long()
    cw = prob.new_tensor(ce_class_weights)
    dw = prob.new_tensor(dice_class_weights)
    pixel_weight = (1 + spatial_boost * supervision_mask.to(prob.dtype)) * valid
    nll = -prob.clamp_min(eps).log().gather(1, safe.unsqueeze(1)).squeeze(1)
    numerator = (nll * pixel_weight * cw[safe]).sum()
    denominator = (pixel_weight * cw[safe]).sum().clamp_min(1)
    ce = numerator / denominator
    onehot = F.one_hot(safe, num_classes=3).permute(0, 3, 1, 2).to(prob.dtype)
    mask = valid[:, None]
    p = prob * mask
    y = onehot * mask
    dice = (2*(p*y).sum((0, 2, 3)) + 1e-6) / ((p+y).sum((0, 2, 3)) + 1e-6)
    dice_loss = 1 - (dice * dw).sum()
    return ce + dice_multiplier*dice_loss, {'ce':ce.detach(), 'dice_loss':dice_loss.detach()}
