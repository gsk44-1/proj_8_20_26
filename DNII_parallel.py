import torch
import torch.nn as nn
import torchvision.transforms.functional as TF


GROUPS = 8


# ---------------------------------------------------------
# Double-well fixed point iteration
# ---------------------------------------------------------

def cubic_iter(x, s=1.0, num_iter=3):
    """
    Approximate implicit double-well step.

    s = alpha = 2 * dt * lambda / epsilon
    """
    xx = x

    for _ in range(num_iter):
        out = (
            x
            - 2.0 * s * xx**3
            + 3.0 * s * xx**2
        ) / (1.0 + s)

        xx = out

    return xx


# ---------------------------------------------------------
# Fixed Laplacian
# ---------------------------------------------------------

def laplace_kern(in_chan=1, out_chan=1):
    weight = torch.zeros(
        out_chan,
        in_chan,
        3,
        3,
        requires_grad=False
    )

    weight[:, :, 1, 0] = 1
    weight[:, :, 1, 2] = 1
    weight[:, :, 0, 1] = 1
    weight[:, :, 2, 1] = 1
    weight[:, :, 1, 1] = -4

    return weight


# ---------------------------------------------------------
# Small UNet
# ---------------------------------------------------------

class DoubleConv(nn.Module):

    def __init__(self, in_chan, out_chan):
        super().__init__()

        self.conv = nn.Sequential(
            nn.Conv2d(
                in_chan,
                out_chan,
                kernel_size=3,
                stride=1,
                padding=1,
                bias=False
            ),

            nn.GroupNorm(GROUPS, out_chan),
            nn.ReLU(inplace=True),

            nn.Conv2d(
                out_chan,
                out_chan,
                kernel_size=3,
                stride=1,
                padding=1,
                bias=False
            ),

            nn.GroupNorm(GROUPS, out_chan),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.conv(x)


class UNET(nn.Module):

    def __init__(
        self,
        in_chan=2,
        out_chan=1,
        features=(32, 64, 128)
    ):
        super().__init__()

        self.ups = nn.ModuleList()
        self.downs = nn.ModuleList()

        self.pool = nn.MaxPool2d(
            kernel_size=2,
            stride=2
        )

        # encoder
        for feature in features:

            self.downs.append(
                DoubleConv(in_chan, feature)
            )

            in_chan = feature

        # bottleneck
        self.bottleneck = DoubleConv(
            features[-1],
            features[-1] * 2
        )

        # decoder
        self.ups.append(
            nn.ConvTranspose2d(
                features[-1] * 2,
                features[-1],
                kernel_size=2,
                stride=2
            )
        )

        self.ups.append(
            DoubleConv(
                features[-1] * 2,
                features[-1]
            )
        )

        for j in reversed(range(len(features) - 1)):

            self.ups.append(
                nn.ConvTranspose2d(
                    features[j + 1],
                    features[j],
                    kernel_size=2,
                    stride=2
                )
            )

            self.ups.append(
                DoubleConv(
                    features[j] * 2,
                    features[j]
                )
            )

        self.final_conv = nn.Conv2d(
            features[0],
            out_chan,
            kernel_size=1
        )

    def forward(self, x):

        skip_connections = []

        for down in self.downs:

            x = down(x)

            skip_connections.append(x)

            x = self.pool(x)

        x = self.bottleneck(x)

        skip_connections = skip_connections[::-1]

        for idx in range(0, len(self.ups), 2):

            x = self.ups[idx](x)

            skip_connection = skip_connections[idx // 2]

            if x.shape[2:] != skip_connection.shape[2:]:

                x = TF.resize(
                    x,
                    size=skip_connection.shape[2:]
                )

            x = torch.cat(
                [skip_connection, x],
                dim=1
            )

            x = self.ups[idx + 1](x)

        return self.final_conv(x)


# ---------------------------------------------------------
# Model-II double-well block
# ---------------------------------------------------------

class ConvBlockII(nn.Module):

    def __init__(
        self,
        features=(32, 64, 128),
        dt=0.1,
        ep=0.2,
        lam=1.0
    ):
        super().__init__()

        self.dt = dt
        self.ep = ep
        self.lam = lam

        # G_n(u, f)
        #
        # u = one channel
        # f = one channel
        # concatenated input = two channels
        self.G = UNET(
            in_chan=2,
            out_chan=1,
            features=features
        )

        # fixed Laplacian
        self.convDiff = nn.Conv2d(
            1,
            1,
            kernel_size=3,
            stride=1,
            padding=1,
            padding_mode="circular",
            bias=False
        )

        self.convDiff.weight = nn.Parameter(
            laplace_kern(1, 1),
            requires_grad=False
        )

        self.sig = nn.Sigmoid()

    def forward(self, u, f):

        # G_n(u^n, f)
        G_input = torch.cat(
            [u, f],
            dim=1
        )

        Guf = self.G(G_input)

        # Model II explicit substep:
        #
        # u^(n+1/2)
        # =
        # u^n
        # + dt * lambda * epsilon * Laplacian(u^n)
        # + dt * G_n(u^n, f)

        u_half = (
            u
            + self.dt
              * self.lam
              * self.ep
              * self.convDiff(u)
            + self.dt * Guf
        )

        # bounded activation before fixed-point solve
        u_half = self.sig(u_half)

        # implicit double-well step
        s = (
            2.0
            * self.dt
            * self.lam
            / self.ep
        )

        u_new = cubic_iter(
            u_half,
            s=s
        )

        return u_new


# ---------------------------------------------------------
# Two completely independent Model-II segmentations
# ---------------------------------------------------------

class DNIIParallel(nn.Module):

    def __init__(
        self,
        features=(32, 64, 128),
        num_blocks=1,
        dt=0.1,
        ep=0.2,
        lam=1.0
    ):
        super().__init__()

        self.dt = dt
        self.ep = ep
        self.lam = lam

        self.num_blocks = num_blocks

        self.sig = nn.Sigmoid()

        # -------------------------------------------------
        # Initial condition H(f)
        # Nuclear
        # -------------------------------------------------

        self.layer1_n = nn.Conv2d(
            1,
            1,
            kernel_size=3,
            stride=1,
            padding=1,
            padding_mode="circular",
            bias=True
        )

        # -------------------------------------------------
        # Initial condition H(f)
        # Nonnuclear
        # -------------------------------------------------

        self.layer1_nn = nn.Conv2d(
            1,
            1,
            kernel_size=3,
            stride=1,
            padding=1,
            padding_mode="circular",
            bias=True
        )

        # -------------------------------------------------
        # Completely separate Model-II blocks
        # -------------------------------------------------

        self.blocks_n = nn.ModuleList()

        self.blocks_nn = nn.ModuleList()

        for _ in range(num_blocks):

            self.blocks_n.append(
                ConvBlockII(
                    features=features,
                    dt=dt,
                    ep=ep,
                    lam=lam
                )
            )

            self.blocks_nn.append(
                ConvBlockII(
                    features=features,
                    dt=dt,
                    ep=ep,
                    lam=lam
                )
            )

        # -------------------------------------------------
        # Independent final segmentation heads
        # -------------------------------------------------

        self.final_n = nn.Conv2d(
            1,
            1,
            kernel_size=3,
            stride=1,
            padding=1,
            padding_mode="circular"
        )

        self.final_nn = nn.Conv2d(
            1,
            1,
            kernel_size=3,
            stride=1,
            padding=1,
            padding_mode="circular"
        )

    def forward(self, x):

        # input:
        # x.shape = (B, 2, H, W)

        f_n = x[:, 0:1, :, :]
        f_nn = x[:, 1:2, :, :]

        # ---------------------------------------------
        # Initial states u^0 = Q(Sig(H(f)))
        # ---------------------------------------------

        s = (
            2.0
            * self.dt
            * self.lam
            / self.ep
        )

        # nuclear
        u_n = self.layer1_n(f_n)

        u_n = self.sig(u_n)

        u_n = cubic_iter(
            u_n,
            s=s
        )

        # nonnuclear
        u_nn = self.layer1_nn(f_nn)

        u_nn = self.sig(u_nn)

        u_nn = cubic_iter(
            u_nn,
            s=s
        )

        # ---------------------------------------------
        # Independent Model-II evolution
        # ---------------------------------------------

        for idx in range(self.num_blocks):

            u_n = self.blocks_n[idx](
                u_n,
                f_n
            )

            u_nn = self.blocks_nn[idx](
                u_nn,
                f_nn
            )

        # ---------------------------------------------
        # Final logits
        # ---------------------------------------------

        logit_n = self.final_n(u_n)

        logit_nn = self.final_nn(u_nn)

        out = torch.cat(
            [logit_n, logit_nn],
            dim=1
        )

        return out