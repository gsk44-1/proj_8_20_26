
"""Analyze the effective receptive field of a trained segmentation model.

IMPORTANT:
GroupNorm couples every spatial location through its mean/variance.

freeze_groupnorm_stats=True preserves the exact GroupNorm forward
activation but stops gradients through spatial mean/variance so
the result measures local convolution/pooling/upsampling paths,
not global normalization coupling.

The ERF is input- and checkpoint-dependent. Run this on trained
weights and representative validation images; optionally average
over several examples.
"""

from contextlib import contextmanager
import torch
from torch import nn


@contextmanager
def _frozen_groupnorm_stats(model):
    handles = []

    def local_gradient_norm(module, inputs, output):
        x = inputs[0]
        n, c, h, w = x.shape
        g = module.num_groups
        z = x.reshape(n, g, -1)

        # Preserve forward values, but stop gradients
        # through global moments.
        mu = z.mean(dim=-1, keepdim=True).detach()
        var = z.var(
            dim=-1, unbiased=False, keepdim=True
        ).detach()

        y = (
            (z - mu) * torch.rsqrt(var + module.eps)
        ).reshape_as(x)

        if module.affine:
            y = y * module.weight[None, :, None, None]
            y = y + module.bias[None, :, None, None]

        return y

    try:
        for module in model.modules():
            if isinstance(module, nn.GroupNorm):
                handles.append(
                    module.register_forward_hook(
                        local_gradient_norm
                    )
                )
        yield
    finally:
        for handle in handles:
            handle.remove()


def effective_receptive_field(
    model,
    image,
    class_index=0,
    location=None,
    freeze_groupnorm_stats=True
):
    """Return gradient heatmap and square widths enclosing
    50%, 90%, and 95% of gradient magnitude.

    Args:
        model: Trained segmentation model.
        image: Float tensor [1, C, H, W], preprocessed
               exactly as for training.
        class_index: Which output logit to analyze.
        location: (row, col); defaults to image center.
        freeze_groupnorm_stats: Whether to stop gradients
               through GroupNorm's spatial statistics.

    Returns:
        influence: Normalized gradient heatmap [H, W].
        widths: Dictionary of enclosing square widths,
                in input pixels.
    """
    if image.ndim != 4 or image.shape[0] != 1:
        raise ValueError(
            "image must have shape [1, C, H, W]"
        )

    was_training = model.training
    model.eval()

    try:
        x = image.detach().clone().requires_grad_(True)

        context = (
            _frozen_groupnorm_stats(model)
            if freeze_groupnorm_stats
            else _nullcontext()
        )

        with context:
            logits = model(x)
            h, w = logits.shape[-2:]

            row, col = (
                location
                if location is not None
                else (h // 2, w // 2)
            )

            grad = torch.autograd.grad(
                logits[0, class_index, row, col],
                x
            )[0]

        influence = (
            grad.detach()
            .abs()
            .sum(dim=1)[0]
            .float()
            .cpu()
        )

        mass = influence.sum()

        if mass <= 0:
            raise RuntimeError(
                "Zero input gradient; choose another "
                "image/class/pixel"
            )

        influence /= mass

        # Input-space coordinates assume stride-1
        # output alignment.
        hi, wi = influence.shape
        rr = torch.arange(hi)[:, None]
        cc = torch.arange(wi)[None, :]

        radius = torch.maximum(
            (rr - row).abs(),
            (cc - col).abs()
        )

        widths = {}

        mass_by_radius = torch.bincount(
            radius.flatten(),
            weights=influence.flatten()
        )
        cumulative = mass_by_radius.cumsum(0)

        for fraction in (0.5, 0.9, 0.95):
            r = int(
                torch.searchsorted(
                    cumulative.contiguous(),
                    torch.tensor(fraction)
                ).item()
            )
            widths[fraction] = 2 * r + 1

        return influence, widths

    finally:
        model.train(was_training)


@contextmanager
def _nullcontext():
    yield


if __name__ == "__main__":
    from klein_dilated_cnn import KleinDilatedCNN

    torch.manual_seed(0)

    model = KleinDilatedCNN(
        dilations=(1, 2, 4)
    )
    img = torch.randn(1, 2, 48, 48)

    heatmap, widths = effective_receptive_field(
        model, img
    )

    print(
        "ERF fractions (UNTRAINED model):",
        widths
    )
    print(
        "Gradient heatmap:",
        heatmap.shape,
        "mass:",
        heatmap.sum().item()
    )
