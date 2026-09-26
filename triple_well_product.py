"""Two-channel, three-phase Model-II prototype.

The phase fields are q=(u,w), z=1-u-w. x is (B,2,H,W).
The diffusion is the exact reduced-coordinate gradient flow of
    lambda * epsilon * (|grad u|²+|grad w|²+|grad(u+w)|²).
The learned G terms retain the source network's fidelity/control interpretation.
The well step approximates the *simplex-constrained* backward Euler substep.
"""
import torch
from torch import nn
from torch.nn import functional as F

GROUPS = 8


def project_triangle(q):
    """Euclidean projection of (u,w) onto u>=0,w>=0,u+w<=1.

    q has shape (B,2,H,W); the projection uses reduced-coordinate distance.
    """
    u, w = q[:, 0:1], q[:, 1:2]
    on_u_zero = torch.cat((torch.zeros_like(u), w.clamp(0, 1)), dim=1)
    on_w_zero = torch.cat((u.clamp(0, 1), torch.zeros_like(w)), dim=1)
    t = ((u-w+1)/2).clamp(0, 1)
    on_z_zero = torch.cat((t, 1-t), dim=1)
    inside = (u >= 0) & (w >= 0) & (u+w <= 1)
    candidates = torch.stack((on_u_zero, on_w_zero, on_z_zero), dim=1)
    distance = ((candidates - q.unsqueeze(1))**2).sum(dim=2, keepdim=True)
    index = distance.argmin(dim=1, keepdim=True)
    closest = candidates.gather(1, index.expand(-1, -1, 2, -1, -1)).squeeze(1)
    return torch.where(inside.expand_as(q), q, closest)


def potential(q):
    u, w = q[:, 0:1], q[:, 1:2]
    z = 1-u-w
    a = (u-1).square()+w.square()+z.square()
    b = u.square()+(w-1).square()+z.square()
    c = u.square()+w.square()+(z-1).square()
    return a*b*c


def well_gradient(q):
    """Derivative after substituting z=1-u-w (not an ambient gradient)."""
    u, w = q[:, 0:1], q[:, 1:2]
    z = 1-u-w
    a = (u-1).square()+w.square()+z.square()
    b = u.square()+(w-1).square()+z.square()
    c = u.square()+w.square()+(z-1).square()
    # dW/du = (dW/du_1 - dW/du_3)|_{u_3=1-u-w}.
    du = 2*((u-z-1)*b*c + (u-z)*a*c + (u-z+1)*a*b)
    dw = 2*((w-z)*b*c + (w-z-1)*a*c + (w-z+1)*a*b)
    return torch.cat((du,dw), dim=1)


def triple_well(q0, s, num_iter=5, relaxation=0.25):
    """Damped projected fixed point for q = Pi_triangle(q0-s*grad W(q)).

    This is a stationary point of min_{q in triangle} .5|q-q0|²+s W(q).
    For nonconvex W, a few iterations need not find the global minimizer.
    Inputs and outputs use (B,2,H,W), unlike the pasted code's leading stack axis.
    """
    if num_iter < 0 or not (0 < relaxation <= 1) or s < 0:
        raise ValueError('num_iter >= 0, 0 < relaxation <= 1, and s >= 0 required')
    q0 = project_triangle(q0)
    q = q0
    for _ in range(num_iter):
        target = project_triangle(q0 - s * well_gradient(q))
        q = (1-relaxation)*q + relaxation*target
    return q


def laplace_kern():
    weight = torch.zeros(1, 1, 3, 3)
    weight[0, 0, 1, 0] = weight[0, 0, 1, 2] = 1
    weight[0, 0, 0, 1] = weight[0, 0, 2, 1] = 1
    weight[0, 0, 1, 1] = -4
    return weight


class DoubleConv(nn.Module):
    def __init__(self, in_chan, out_chan):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_chan, out_chan, 3, padding=1, bias=False),
            nn.GroupNorm(GROUPS, out_chan), nn.ReLU(inplace=True),
            nn.Conv2d(out_chan, out_chan, 3, padding=1, bias=False),
            nn.GroupNorm(GROUPS, out_chan), nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.conv(x)


class UNET(nn.Module):
    def __init__(self, in_chan=2, out_chan=1, features=(32,64,128)):
        super().__init__()
        if not features or any(c % GROUPS for c in features):
            raise ValueError('All feature widths must be divisible by 8')
        self.downs = nn.ModuleList()
        for width in features:
            self.downs.append(DoubleConv(in_chan, width))
            in_chan = width
        self.pool = nn.MaxPool2d(2)
        self.bottleneck = DoubleConv(features[-1], 2*features[-1])
        self.ups = nn.ModuleList()
        prev = 2*features[-1]
        for width in reversed(features):
            self.ups.append(nn.ConvTranspose2d(prev, width, 2, stride=2))
            self.ups.append(DoubleConv(2*width, width))
            prev = width
        self.final_conv = nn.Conv2d(features[0], out_chan, 1)

    def forward(self, x):
        skips = []
        for down in self.downs:
            x = down(x)
            skips.append(x)
            x = self.pool(x)
        x = self.bottleneck(x)
        for i in range(0, len(self.ups), 2):
            x = self.ups[i](x)
            skip = skips[-1-i//2]
            if x.shape[-2:] != skip.shape[-2:]:
                x = F.interpolate(x, size=skip.shape[-2:], mode='bilinear', align_corners=False)
            x = self.ups[i+1](torch.cat((skip,x), dim=1))
        return self.final_conv(x)


class ConvBlockII(nn.Module):
    """Explicit coupled diffusion and independent learned fidelity forcing."""
    def __init__(self, features=(32,64,128), dt=0.1, ep=0.2, lam=1.0):
        super().__init__()
        self.dt, self.ep, self.lam = dt, ep, lam
        self.G = UNET(in_chan=2, out_chan=1, features=features)
        self.convDiff = nn.Conv2d(1,1,3,padding=1,padding_mode='circular',bias=False)
        self.convDiff.weight = nn.Parameter(laplace_kern(), requires_grad=False)

    def forward(self, u, f, other):
        guf = self.G(torch.cat((u,f), dim=1))
        u_half = u + self.dt * (2*self.lam*self.ep*(2*self.convDiff(u)+self.convDiff(other)) + guf)
        return u_half, guf


class DNIIParallel(nn.Module):
    """Compatible output: 2 logits; optional G tensors and diagnostics."""
    def __init__(self, features=(32,64,128), num_blocks=1, dt=0.1,
                 ep=0.2, lam=1.0, n_iter=5, relaxation=0.25):
        super().__init__()
        if dt <= 0 or ep <= 0 or lam < 0 or num_blocks < 0:
            raise ValueError('dt, ep > 0; lam and num_blocks >= 0 required')
        self.dt, self.ep, self.lam = dt, ep, lam
        self.n_iter, self.num_blocks, self.relaxation = n_iter, num_blocks, relaxation
        self.sig = nn.Sigmoid()
        self.layer1_n = nn.Conv2d(1,1,3,padding=1,padding_mode='circular')
        self.layer1_nn = nn.Conv2d(1,1,3,padding=1,padding_mode='circular')
        self.blocks_n = nn.ModuleList([ConvBlockII(features,dt,ep,lam) for _ in range(num_blocks)])
        self.blocks_nn = nn.ModuleList([ConvBlockII(features,dt,ep,lam) for _ in range(num_blocks)])
        # The pasted prototype declared final_n/final_nn but did not use them.
        # Kept for checkpoint compatibility; output stays 5*(phase-.5).
        self.final_n = nn.Conv2d(1,1,3,padding=1,padding_mode='circular')
        self.final_nn = nn.Conv2d(1,1,3,padding=1,padding_mode='circular')

    def forward(self, x, return_diag=False):
        if x.ndim != 4 or x.shape[1] != 2:
            raise ValueError('Expected image of shape (B,2,H,W)')
        f_n, f_nn = x[:,0:1], x[:,1:2]
        q = torch.cat((self.sig(self.layer1_n(f_n)), self.sig(self.layer1_nn(f_nn))), dim=1)
        s = self.dt*self.lam/self.ep
        q = triple_well(q, s, self.n_iter, self.relaxation)
        g_outs_n, g_outs_nn = [], []
        for block_n, block_nn in zip(self.blocks_n, self.blocks_nn):
            # Both explicit steps use the SAME old pair (Jacobi splitting).
            old_n, old_nn = q[:,0:1], q[:,1:2]
            half_n, gn = block_n(old_n, f_n, old_nn)
            half_nn, gnn = block_nn(old_nn, f_nn, old_n)
            g_outs_n.append(gn)
            g_outs_nn.append(gnn)
            q = triple_well(torch.cat((half_n,half_nn), dim=1), s,
                            self.n_iter, self.relaxation)
        logits = 5.0*(q-0.5)
        shape = (x.shape[0],0,1,*x.shape[-2:])
        gn_all = torch.stack(g_outs_n, dim=1) if g_outs_n else x.new_empty(shape)
        gnn_all = torch.stack(g_outs_nn, dim=1) if g_outs_nn else x.new_empty(shape)
        if return_diag:
            diagnostics = {'u_n': logits[:,0:1], 'u_nn': logits[:,1:2],
                           'g_n': gn_all, 'g_nn': gnn_all,
                           'phases': torch.cat((q,1-q.sum(dim=1,keepdim=True)),dim=1)}
            return logits, gn_all, gnn_all, diagnostics
        return logits, gn_all, gnn_all
