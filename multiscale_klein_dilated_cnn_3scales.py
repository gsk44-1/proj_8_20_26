"""Three-scale full-resolution Klein-filter segmentation CNN.

Adapted from the supplied multiscale_klein_dilated_cnn(1).py.
Three branches share ONE fixed 3x3 Klein filter bank, applied per input channel:
    input ---------------------------> Klein bank --+
    input -> Gaussian(sigma=1.0) ----> Klein bank --+--> concatenate
    input -> Gaussian(sigma=2.0) ----> Klein bank --+    -> 1x1 projection
                                                        -> residual dilated CNN
                                                        -> pixelwise logits

Requires klein_dilated_cnn.py in the same directory (FixedKleinBank,
DilatedResidualBlock). No topology-restricted connections are included.
"""

import math
import torch
import torch.nn.functional as F
from torch import nn

from klein_dilated_cnn import FixedKleinBank, DilatedResidualBlock


class FixedGaussianBlur(nn.Module):
    """Fixed, normalized depthwise Gaussian kernel with reflection padding."""

    def __init__(self, channels: int, sigma: float, radius=None):
        super().__init__()
        if channels < 1 or sigma <= 0:
            raise ValueError("channels must be positive and sigma > 0")
        if radius is None:
            radius = max(1, math.ceil(3 * sigma))
        if radius < 1:
            raise ValueError("radius must be >= 1")
        self.channels = int(channels)
        self.sigma = float(sigma)
        self.radius = int(radius)

        positions = torch.arange(-radius, radius + 1, dtype=torch.float64)
        g = torch.exp(-0.5 * (positions / sigma) ** 2)
        g /= g.sum()
        kernel = torch.outer(g, g).float()
        self.register_buffer(
            "weight", kernel[None, None].repeat(channels, 1, 1, 1)
        )

    def forward(self, x):
        if x.ndim != 4 or x.shape[1] != self.channels:
            raise ValueError(f"Expected input shape [B, {self.channels}, H, W]")
        if min(x.shape[-2:]) <= self.radius:
            raise ValueError("H and W must exceed Gaussian blur radius")
        r = self.radius
        return F.conv2d(
            F.pad(x, (r, r, r, r), mode="reflect"),
            self.weight,
            groups=self.channels,
        )


class MultiScaleKleinDilatedCNN(nn.Module):
    """Raw + two Gaussian scales, shared frozen Klein bank, dilated backbone.

    Args:
        blur_sigmas: exactly two positive Gaussian scales, in input pixels.
        blur_radii: optional pair of radii, each defaults to ceil(3*sigma).

    Produces raw logits of shape [B, out_channels, H, W].
    """

    def __init__(
        self,
        in_channels=2,
        out_channels=3,
        features=32,
        num_groups=8,
        n_orientations=4,
        n_phases=4,
        dilations=(1, 2, 4, 8, 16, 32, 16, 8),
        normalize_filters=True,
        blur_sigmas=(1.0, 2.0),
        blur_radii=(None, None),
    ):
        super().__init__()
        if features % num_groups != 0:
            raise ValueError("features must be divisible by num_groups")
        if len(blur_sigmas) != 2 or len(blur_radii) != 2:
            raise ValueError("Expected two blur sigmas and two blur radii")

        self.blur_sigmas = tuple(float(s) for s in blur_sigmas)
        self.dilations = tuple(int(d) for d in dilations)
        if not self.dilations or any(d < 1 for d in self.dilations):
            raise ValueError("dilations must be nonempty positive integers")

        self.blurs = nn.ModuleList([
            FixedGaussianBlur(in_channels, sigma, radius)
            for sigma, radius in zip(self.blur_sigmas, blur_radii)
        ])
        self.klein = FixedKleinBank(
            in_channels, n_orientations, n_phases, normalize_filters
        )

        features_per_scale = in_channels * self.klein.filters_per_channel
        self.project = nn.Sequential(
            nn.Conv2d(3 * features_per_scale, features, kernel_size=1, bias=False),
            nn.GroupNorm(num_groups, features),
            nn.ReLU(inplace=True),
        )
        self.blocks = nn.Sequential(*[
            DilatedResidualBlock(features, d, num_groups)
            for d in self.dilations
        ])
        self.final = nn.Conv2d(features, out_channels, kernel_size=1)

    @property
    def structural_rf(self):
        """Longest-path RF width away from boundaries (ignoring GN coupling)."""
        return 1 + 2 * max(blur.radius for blur in self.blurs) + 2 * (
            1 + sum(self.dilations) + len(self.dilations)
        )

    def fixed_features(self, x):
        """Return [raw, blur_sigma_1, blur_sigma_2] Klein features."""
        return torch.cat(
            [self.klein(x)] + [self.klein(blur(x)) for blur in self.blurs],
            dim=1,
        )

    def forward(self, x):
        x = self.fixed_features(x)
        x = self.project(x)
        x = self.blocks(x)
        return self.final(x)


if __name__ == "__main__":
    net = MultiScaleKleinDilatedCNN()
    sample = torch.randn(1, 2, 64, 64)
    logits = net(sample)
    assert logits.shape == (1, 3, 64, 64)
    logits.square().mean().backward()
    assert net.klein.conv.weight.grad is None
    assert all(b.weight.grad is None for b in net.blurs)
    assert net.final.weight.grad is not None
    print("Output shape:", tuple(logits.shape))
    print("Fixed feature maps:", net.fixed_features(sample).shape[1])
    print("Structural RF:", net.structural_rf)
    print("Trainable parameters:", sum(p.numel() for p in net.parameters() if p.requires_grad))
