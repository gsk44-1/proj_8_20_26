import torch
import torch.nn as nn
import torch.nn.functional as F


class DoubleConv(nn.Module):
    """
    Conv -> GroupNorm -> ReLU -> Conv -> GroupNorm -> ReLU
    """
    def __init__(self, in_channels, out_channels, num_groups=8):
        super().__init__()

        self.block = nn.Sequential(
            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=3,
                padding=1,
                bias=False
            ),
            nn.GroupNorm(
                num_groups=num_groups,
                num_channels=out_channels
            ),
            nn.ReLU(inplace=True),

            nn.Conv2d(
                out_channels,
                out_channels,
                kernel_size=3,
                padding=1,
                bias=False
            ),
            nn.GroupNorm(
                num_groups=num_groups,
                num_channels=out_channels
            ),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.block(x)


class UNet(nn.Module):
    def __init__(
        self,
        in_channels=2,
        out_channels=2,
        features=(16, 32, 64, 128),
        num_groups=8
    ):
        super().__init__()

        f1, f2, f3, f4 = features

        # Encoder
        self.enc1 = DoubleConv(in_channels, f1, num_groups)
        self.pool1 = nn.MaxPool2d(2)

        self.enc2 = DoubleConv(f1, f2, num_groups)
        self.pool2 = nn.MaxPool2d(2)

        self.enc3 = DoubleConv(f2, f3, num_groups)
        self.pool3 = nn.MaxPool2d(2)

        self.enc4 = DoubleConv(f3, f4, num_groups)
        self.pool4 = nn.MaxPool2d(2)

        # Bottleneck
        self.bottleneck = DoubleConv(
            f4,
            f4 * 2,
            num_groups
        )

        # Decoder
        self.up4 = nn.ConvTranspose2d(
            f4 * 2, f4, kernel_size=2, stride=2
        )
        self.dec4 = DoubleConv(
            f4 * 2, f4, num_groups
        )

        self.up3 = nn.ConvTranspose2d(
            f4, f3, kernel_size=2, stride=2
        )
        self.dec3 = DoubleConv(
            f3 * 2, f3, num_groups
        )

        self.up2 = nn.ConvTranspose2d(
            f3, f2, kernel_size=2, stride=2
        )
        self.dec2 = DoubleConv(
            f2 * 2, f2, num_groups
        )

        self.up1 = nn.ConvTranspose2d(
            f2, f1, kernel_size=2, stride=2
        )
        self.dec1 = DoubleConv(
            f1 * 2, f1, num_groups
        )

        # No normalization here:
        # map 16 feature channels directly to the desired output channels.
        self.final = nn.Conv2d(
            f1,
            out_channels,
            kernel_size=1
        )

    def forward(self, x):

        # Encoder
        x1 = self.enc1(x)
        x2 = self.enc2(self.pool1(x1))
        x3 = self.enc3(self.pool2(x2))
        x4 = self.enc4(self.pool3(x3))

        # Bottleneck
        x5 = self.bottleneck(self.pool4(x4))

        # Decoder
        x = self.up4(x5)
        x = torch.cat([x, x4], dim=1)
        x = self.dec4(x)

        x = self.up3(x)
        x = torch.cat([x, x3], dim=1)
        x = self.dec3(x)

        x = self.up2(x)
        x = torch.cat([x, x2], dim=1)
        x = self.dec2(x)

        x = self.up1(x)
        x = torch.cat([x, x1], dim=1)
        x = self.dec1(x)

        return self.final(x)