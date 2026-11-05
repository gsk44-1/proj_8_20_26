"""Full-resolution multiscale Klein-filter segmentation network.

Two branches (shared fixed 3x3 Klein bank):
    input -> Klein filters -------------------------------> concat
    input -> fixed Gaussian blur (sigma=1) -> Klein filters -> concat
    concat -> learned 1x1 projection -> residual dilated CNN -> logits

The Gaussian blur and Klein filters act independently on every input channel.
This is a proposed segmentation architecture, not the original TCNN's KOL.

Example:
    model = MultiScaleKleinDilatedCNN(in_channels=2, out_channels=3,
                                     features=32, num_groups=8)
    logits = model(torch.randn(2, 2, 128, 128))  # (2,3,128,128)
"""
import math

import torch
import torch.nn.functional as F
from torch import nn

# Use the original bank and dilated blocks without modifying them.
from klein_dilated_cnn import FixedKleinBank, DilatedResidualBlock


class FixedGaussianBlur(nn.Module):
    """Depthwise normalized Gaussian smoothing with reflection padding.

    sigma is specified in image pixels. A buffer stores the kernel so it
    follows .to(device), is saved in state_dict, and is never trained.
    """
    def __init__(self, channels: int, sigma: float = 1.0, radius=None):
        super().__init__()
        if channels < 1 or sigma <= 0:
            raise ValueError('channels must be positive and sigma > 0')
        if radius is None:
            radius = max(1, math.ceil(3 * sigma))
        if radius < 1:
            raise ValueError('radius must be positive')
        self.channels = channels
        self.sigma = float(sigma)
        self.radius = int(radius)
        positions = torch.arange(-radius, radius + 1, dtype=torch.float64)
        g = torch.exp(-0.5 * (positions / sigma) ** 2)
        g = g / g.sum()
        kernel = torch.outer(g, g).float()
        # Shape: [C, 1, kernel_h, kernel_w] for depthwise filtering.
        self.register_buffer('weight', kernel[None, None].repeat(channels, 1, 1, 1))

    def forward(self, x):
        if x.ndim != 4 or x.shape[1] != self.channels:
            raise ValueError(f'Expected input [B, {self.channels}, H, W]')
        if min(x.shape[-2:]) <= self.radius:
            raise ValueError('H and W must exceed the Gaussian blur radius for reflect padding')
        r = self.radius
        return F.conv2d(F.pad(x, (r, r, r, r), mode='reflect'),
                        self.weight, groups=self.channels)


class MultiScaleKleinDilatedCNN(nn.Module):
    """Fixed raw+Gaussian-smoothed Klein features and a learned dilated backbone.

    Uses *one shared* frozen Klein bank on the raw and blurred images, not
    two independently learned banks. Output is raw per-pixel class logits.

    With the default 7x7 Gaussian, the longest structural receptive-field
    side is 6 pixels wider than the original raw-only architecture.
    """
    def __init__(self, in_channels=2, out_channels=3, features=32,
                 num_groups=8, n_orientations=4, n_phases=4,
                 dilations=(1, 2, 4, 8, 16, 32, 16, 8),
                 normalize_filters=True, blur_sigma=1.0, blur_radius=None):
        super().__init__()
        if features % num_groups != 0:
            raise ValueError('features must be divisible by num_groups')
        self.dilations = tuple(int(d) for d in dilations)
        if not self.dilations or any(d < 1 for d in self.dilations):
            raise ValueError('dilations must be nonempty positive integers')

        self.blur = FixedGaussianBlur(in_channels, blur_sigma, blur_radius)
        self.klein = FixedKleinBank(in_channels, n_orientations, n_phases,
                                    normalize_filters)
        # Each of the two scales yields in_channels * filters_per_channel maps.
        features_per_scale = in_channels * self.klein.filters_per_channel
        self.project = nn.Sequential(
            nn.Conv2d(2 * features_per_scale, features, kernel_size=1, bias=False),
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
        # The larger (blurred) input branch determines maximum path support.
        return 1 + 2 * self.blur.radius + 2 * (
            1 + sum(self.dilations) + len(self.dilations)
        )

    def fixed_features(self, x):
        """Return [raw Klein responses, blurred Klein responses] along channel axis."""
        raw = self.klein(x)
        smoothed = self.klein(self.blur(x))
        return torch.cat((raw, smoothed), dim=1)

    def forward(self, x):
        x = self.fixed_features(x)
        x = self.project(x)
        x = self.blocks(x)
        return self.final(x)


if __name__ == '__main__':
    net = MultiScaleKleinDilatedCNN()
    sample = torch.randn(1, 2, 64, 64)
    prediction = net(sample)
    assert prediction.shape == (1, 3, 64, 64)
    prediction.square().mean().backward()
    assert net.klein.conv.weight.grad is None
    assert net.blur.weight.grad is None
    assert net.final.weight.grad is not None
    print('Output shape:', tuple(prediction.shape))
    print('Fixed feature maps:', net.fixed_features(sample).shape[1])
    print('Structural receptive field:', net.structural_rf)
    print('Trainable parameters:', sum(p.numel() for p in net.parameters() if p.requires_grad))
