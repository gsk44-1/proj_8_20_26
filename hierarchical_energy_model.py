"""Hierarchical scalar-energy segmentation with learned, composed state operators.

u0 = z
u1 = SiLU(K1(u0) + B1(f1(x)))
u2 = SiLU(K2(u1) + B2(f2(x)))
E(x,z) = sum_s scale_weight[s] * area_factor[s] * sum(head_s(u_s, f_s(x)))
         + 0.5*state_l2*||z||^2.

The gradient wrt z is computed by autograd, giving an additive, gated,
transposed-convolution hierarchy with exact weight tying to the encoder.
Cached image features are reused over iterations. State operators are learned.
"""
import math
import torch
from torch import nn
from torch.nn import functional as F


def block(cin, cout):
    return nn.Sequential(nn.Conv2d(cin, cout, 3, padding=1), nn.SiLU(),
                         nn.Conv2d(cout, cout, 3, padding=1), nn.SiLU())


class HierarchicalSegmentationEnergy(nn.Module):
    def __init__(self, image_channels=2, classes=3, image_width=16,
                 head_width=24, state_width=16, initial_step=0.3,
                 state_l2=1e-3, threshold=(0.25, 0.25),
                 threshold_scale=3.0, scale_weights=(1., .5, .25)):
        super().__init__()
        if image_channels != 2 or classes != 3:
            raise ValueError('This prototype requires two stain channels and three classes')
        if len(scale_weights) != 3 or len(threshold) != 2 or initial_step <= 0:
            raise ValueError('Expected three scale weights, two thresholds, positive step')
        self.classes = classes
        self.threshold = tuple(map(float, threshold))
        self.threshold_scale = float(threshold_scale)
        self.scale_weights = tuple(map(float, scale_weights))
        self.state_l2 = float(state_l2)
        w, c = image_width, state_width
        # Image-only hierarchy, reusable across iterations.
        self.image_fine = block(2, w)
        self.image_half = block(w, 2*w)
        self.image_quarter = block(2*w, 3*w)
        # K2 K1 hierarchy: K1 maps ORIGINAL logits to learned half-scale
        # features; K2 consumes those learned features, not separately pooled z.
        self.K1 = nn.Conv2d(3, c, 3, stride=2, padding=1)
        self.K2 = nn.Conv2d(c, 2*c, 3, stride=2, padding=1)
        # Image-dependent biases to the state encoder; all gates depend on z.
        self.B1 = nn.Conv2d(2*w, c, 1)
        self.B2 = nn.Conv2d(3*w, 2*c, 1)
        # Scalar energy density heads (no independent trainable decoder).
        self.head_fine = nn.Sequential(block(w+3, head_width), nn.Conv2d(head_width, 1, 1))
        self.head_half = nn.Sequential(block(2*w+c, head_width), nn.Conv2d(head_width, 1, 1))
        self.head_quarter = nn.Sequential(block(3*w+2*c, head_width), nn.Conv2d(head_width, 1, 1))
        self.log_step = nn.Parameter(torch.tensor(math.log(initial_step)))

    def initial_state(self, image, *, threshold=None):
        t = torch.as_tensor(self.threshold if threshold is None else threshold,
                            device=image.device, dtype=image.dtype)
        if t.ndim == 1:
            t = t.view(1, 2, 1, 1)
        return torch.cat((torch.zeros_like(image[:, :1]),
                          self.threshold_scale * (image[:, :2]-t)), dim=1)

    def encode_image(self, image):
        f0 = self.image_fine(image)
        f1 = self.image_half(F.avg_pool2d(f0, 2, ceil_mode=True))
        f2 = self.image_quarter(F.avg_pool2d(f1, 2, ceil_mode=True))
        return f0, f1, f2

    def energy(self, image, state, *, features=None):
        if features is None:
            features = self.encode_image(image)
        f0, f1, f2 = features
        u0 = state
        u1 = F.silu(self.K1(u0) + self.B1(f1))
        u2 = F.silu(self.K2(u1) + self.B2(f2))
        densities = (
            self.head_fine(torch.cat((f0, u0), dim=1)),
            self.head_half(torch.cat((f1, u1), dim=1)),
            self.head_quarter(torch.cat((f2, u2), dim=1)),
        )
        area = state.shape[-2]*state.shape[-1]
        result = sum(w * d.sum() * (area / (d.shape[-2]*d.shape[-1]))
                     for w, d in zip(self.scale_weights, densities))
        return result + .5*self.state_l2*state.square().sum()

    def state_gradient(self, image, state, *, create_graph, features=None):
        with torch.enable_grad():
            if not state.requires_grad:
                state = state.detach().requires_grad_(True)
            e = self.energy(image, state, features=features)
            grad, = torch.autograd.grad(e, state, create_graph=create_graph)
        return grad

    def update(self, image, state, *, differentiable=False, features=None):
        grad = self.state_gradient(image, state, create_graph=differentiable, features=features)
        new = state - self.log_step.exp()*grad
        return new if differentiable else new.detach()

    def infer(self, image, *, steps=12, state=None):
        state = self.initial_state(image) if state is None else state
        state = state.detach()
        with torch.no_grad():
            features = self.encode_image(image)
        for _ in range(steps):
            state = self.update(image, state, differentiable=False, features=features)
        return state.detach()

    def forward(self, image, steps=12):
        return self.infer(image, steps=steps)


class RandomEnergyPerturber(nn.Module):
    """Frozen random scalar-energy CNN, same corruption scheme as earlier model."""
    def __init__(self, width=32):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(5, width, 3, padding=1), nn.SiLU(),
            nn.Conv2d(width, width, 3, dilation=3, padding=3), nn.SiLU(),
            nn.Conv2d(width, width, 3, dilation=7, padding=7), nn.SiLU(),
            nn.Conv2d(width, 1, 1))
        self.reset_random_weights()

    def reset_random_weights(self):
        for layer in self.net.modules():
            if hasattr(layer, 'reset_parameters'):
                layer.reset_parameters()
        for param in self.parameters():
            param.requires_grad_(False)

    def forward(self, image, state):
        with torch.enable_grad():
            q = state.detach().requires_grad_(True)
            e = self.net(torch.cat((image.detach(), q), 1)).sum()
            grad, = torch.autograd.grad(e, q)
        noise = -grad.detach()
        rms = noise.square().mean((1,2,3), keepdim=True).sqrt().clamp_min(1e-6)
        return noise/rms


if __name__ == '__main__':
    torch.manual_seed(2)
    model = HierarchicalSegmentationEnergy(image_width=4, state_width=4, head_width=6)
    x = torch.rand(2,2,15,17)
    z = model.initial_state(x)
    next_z = model.update(x,z,differentiable=True)
    next_z.square().mean().backward()
    assert model.K1.weight.grad is not None
    assert model.K2.weight.grad is not None
    assert model.image_half[0].weight.grad is not None
    with torch.no_grad():
        output = model.infer(x, steps=3)
    assert output.shape == (2,3,15,17)
    print('Hierarchical model smoke test passed')
