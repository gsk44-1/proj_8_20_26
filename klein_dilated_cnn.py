"""Full-resolution segmentation CNN with fixed 3x3 Klein-bottle filters.

Love et al. (2023), 'Topological Convolutional Layers for Deep Learning',
Eq. (4) and Definition 9. This is a new segmentation architecture, NOT the
published TCNN architecture. It does not use topology-restricted connections.

Usage:
    model = KleinDilatedCNN(in_channels=2, out_channels=3, num_groups=8)
    logits = model(images)  # [B, 3, H, W]; feed directly to CrossEntropyLoss
"""
import math

import torch
from torch import nn


def klein_filter_bank(n_orientations=4, n_phases=4, normalize=True):
    """Return [n_orientations*n_phases, 3, 3] fixed Klein filters.

    Exact cell *averages* of the polynomial F_K on the square [-1,1]^2,
    rather than just sampling at cell centers; multiplying all cell averages
    by the cell area would give the paper's cell-integral convention.

    F_K(theta,phi)(x,y) = sin(phi)*t + cos(phi)*(2*t*t - 1),
    t = cos(theta)*x + sin(theta)*y.

    Evenly sample theta in [0,pi), phi in [0,2pi). Their quotient symmetries
    mean that the latter parameterization covers the Klein bottle once.
    """
    if n_orientations < 1 or n_phases < 1:
        raise ValueError('n_orientations and n_phases must be positive')
    centers = torch.tensor([-2.0/3.0, 0.0, 2.0/3.0], dtype=torch.float64)
    yy, xx = torch.meshgrid(centers, centers, indexing='ij')
    kernels = []
    for i in range(n_orientations):
        theta = math.pi * i / n_orientations
        t = math.cos(theta) * xx + math.sin(theta) * yy
        # For uniformly distributed (x,y) in each cell of side 2/3,
        # E[(cos(theta)x+sin(theta)y)^2] = t_center^2 + 1/27.
        q_mean = 2 * (t.square() + 1.0/27.0) - 1.0
        for j in range(n_phases):
            phi = 2 * math.pi * j / n_phases
            filt = math.sin(phi) * t + math.cos(phi) * q_mean
            if normalize:
                filt = filt / filt.square().sum().sqrt().clamp_min(1e-12)
            kernels.append(filt)
    return torch.stack(kernels).float()


class FixedKleinBank(nn.Module):
    """A separate, identical fixed filter bank on each input channel."""
    def __init__(self, in_channels=2, n_orientations=4, n_phases=4,
                 normalize=True):
        super().__init__()
        bank = klein_filter_bank(n_orientations, n_phases, normalize)
        self.filters_per_channel = bank.shape[0]
        self.in_channels = in_channels
        self.conv = nn.Conv2d(
            in_channels, in_channels * self.filters_per_channel,
            kernel_size=3, padding=1, groups=in_channels, bias=False
        )
        with torch.no_grad():
            self.conv.weight.copy_(bank[:, None].repeat(in_channels, 1, 1, 1))
        self.conv.weight.requires_grad_(False)

    def forward(self, x):
        return self.conv(x)


class DilatedResidualBlock(nn.Module):
    """Two spatial convolutions per block; first is dilated, second local.

    A local 3x3 second convolution helps prevent disconnected dilation grids.
    """
    def __init__(self, channels, dilation, num_groups=8):
        super().__init__()
        self.body = nn.Sequential(
            nn.Conv2d(channels, channels, 3, padding=dilation,
                      dilation=dilation, bias=False),
            nn.GroupNorm(num_groups, channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels, channels, 3, padding=1, bias=False),
            nn.GroupNorm(num_groups, channels),
        )
        self.activation = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.activation(x + self.body(x))


class KleinDilatedCNN(nn.Module):
    """Full-resolution, 3-class pixelwise segmenter (raw logits).

    Defaults to 8 residual blocks, width 32, with dilation 1,2,4,8,16,32,16,8.
    The second convolution inside each block has dilation 1.
    Receptive-field width on the deepest path, away from boundaries:
        1 + 2*(1 + sum(dilations) + len(dilations))
    where the first 1 is for the frozen 3x3 filters.
    """
    def __init__(self, in_channels=2, out_channels=3, features=32,
                 num_groups=8, n_orientations=4, n_phases=4,
                 dilations=(1, 2, 4, 8, 16, 32, 16, 2),
                 normalize_filters=True):
        super().__init__()
        if features % num_groups:
            raise ValueError('features must be divisible by num_groups')
        self.dilations = tuple(int(d) for d in dilations)
        if not self.dilations or any(d < 1 for d in self.dilations):
            raise ValueError('dilations must be nonempty positive integers')
        self.klein = FixedKleinBank(in_channels, n_orientations, n_phases,
                                    normalize_filters)
        bank_channels = in_channels * self.klein.filters_per_channel
        self.project = nn.Sequential(
            nn.Conv2d(bank_channels, features, kernel_size=1, bias=False),
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
        return 1 + 2*(1 + sum(self.dilations) + len(self.dilations))

    def forward(self, x):
        x = self.klein(x)
        x = self.project(x)
        x = self.blocks(x)
        return self.final(x)


if __name__ == '__main__':
    model = KleinDilatedCNN()
    x = torch.randn(1, 2, 64, 64)
    y = model(x)
    assert y.shape == (1, 3, 64, 64)
    loss = y.square().mean()
    loss.backward()
    assert model.klein.conv.weight.grad is None
    assert model.final.weight.grad is not None
    print('Output:', tuple(y.shape))
    print('Structural RF:', model.structural_rf)
    print('Trainable parameters:', sum(p.numel() for p in model.parameters() if p.requires_grad))
