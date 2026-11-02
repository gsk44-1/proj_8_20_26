
"""Hierarchical scalar-energy segmentation with image-dependent linear drive.

State hierarchy:
    u0 = z
    u1 = SiLU(K1(u0) + B1(f1(x)))
    u2 = SiLU(K2(u1) + B2(f2(x)))

Energy:
    E(x,z) = hierarchical multiscale energy
             - linear_drive_scale * <b_theta(x), z>
             + 0.5 * state_l2 * ||z||^2

The image-only driving field b_theta(x) supplies a direct additive
correction to the gradient-descent update.

All state-dependent corrections still come from the gradient of
a single scalar energy.
"""

import math

import torch
from torch import nn
from torch.nn import functional as F


def block(cin, cout):
    return nn.Sequential(
        nn.Conv2d(cin, cout, 3, padding=1),
        nn.SiLU(),
        nn.Conv2d(cout, cout, 3, padding=1),
        nn.SiLU(),
    )


class HierarchicalSegmentationEnergy(nn.Module):

    def __init__(
        self,
        image_channels=2,
        classes=3,
        image_width=16,
        head_width=24,
        state_width=16,
        initial_step=0.3,
        state_l2=1e-3,
        threshold=(0.25, 0.25),
        threshold_scale=3.0,
        scale_weights=(1.0, 0.5, 0.25),
        linear_drive_scale=1.0,
    ):
        super().__init__()

        if image_channels != 2 or classes != 3:
            raise ValueError(
                "This prototype requires 2 image channels and 3 classes"
            )

        if (
            len(scale_weights) != 3
            or len(threshold) != 2
            or initial_step <= 0
        ):
            raise ValueError("Invalid scale weights, thresholds, or step")

        self.classes = classes
        self.threshold = tuple(map(float, threshold))
        self.threshold_scale = float(threshold_scale)
        self.scale_weights = tuple(map(float, scale_weights))
        self.state_l2 = float(state_l2)
        self.linear_drive_scale = float(linear_drive_scale)

        w = image_width
        c = state_width

        # -----------------------------------------------------
        # Image encoder
        # -----------------------------------------------------

        self.image_fine = block(2, w)
        self.image_half = block(w, 2 * w)
        self.image_quarter = block(2 * w, 3 * w)

        # -----------------------------------------------------
        # Hierarchical state encoder
        # -----------------------------------------------------

        self.K1 = nn.Conv2d(
            3, c, kernel_size=3, stride=1, padding=1
        )

        self.K2 = nn.Conv2d(
            c, 2 * c, kernel_size=3, stride=1, padding=1
        )

        # Image-dependent biases for the state hierarchy
        self.B1 = nn.Conv2d(2 * w, c, kernel_size=1)
        self.B2 = nn.Conv2d(3 * w, 2 * c, kernel_size=1)

        # -----------------------------------------------------
        # Scalar energy density heads
        # -----------------------------------------------------

        self.head_fine = nn.Sequential(
            block(w + 3, head_width),
            nn.Conv2d(head_width, 1, kernel_size=1),
        )

        self.head_half = nn.Sequential(
            block(2 * w + c, head_width),
            nn.Conv2d(head_width, 1, kernel_size=1),
        )

        self.head_quarter = nn.Sequential(
            block(3 * w + 2 * c, head_width),
            nn.Conv2d(head_width, 1, kernel_size=1),
        )

        # -----------------------------------------------------
        # NEW: image-dependent driving field
        # -----------------------------------------------------

        # Combines image features from all three resolutions,
        # upsamples them to full resolution, and predicts
        # a three-channel field b_theta(x).

        self.image_drive_head = nn.Sequential(
            nn.Conv2d(
                6 * w, head_width, kernel_size=3, padding=1
            ),
            nn.SiLU(),
            nn.Conv2d(
                head_width, classes, kernel_size=1
            ),
        )

        # Initialize the new driving field to zero.
        # The original hierarchical energy therefore gives
        # the initial state update, and the driving field
        # learns to contribute during training.

        nn.init.zeros_(self.image_drive_head[-1].weight)
        nn.init.zeros_(self.image_drive_head[-1].bias)

        # Learnable positive gradient-descent step size
        self.log_step = nn.Parameter(
            torch.tensor(math.log(initial_step))
        )

    # =========================================================
    # Initial segmentation logits
    # =========================================================

    def initial_state(self, image, *, threshold=None):
        t = torch.as_tensor(
            self.threshold if threshold is None else threshold,
            device=image.device,
            dtype=image.dtype,
        )

        if t.ndim == 1:
            t = t.view(1, 2, 1, 1)

        return torch.cat(
            (
                torch.zeros_like(image[:, :1]),
                self.threshold_scale * (image[:, :2] - t),
            ),
            dim=1,
        )

    # =========================================================
    # Image feature pyramid
    # =========================================================

    def encode_image(self, image):
        f0 = self.image_fine(image)

        f1 = self.image_half(
            F.avg_pool2d(f0, 2, ceil_mode=True)
        )

        f2 = self.image_quarter(
            F.avg_pool2d(f1, 2, ceil_mode=True)
        )

        return f0, f1, f2

    # =========================================================
    # NEW: image-dependent driving field
    # =========================================================

    def image_drive(self, features):
        """
        Predict b_theta(x), shape [B, 3, H, W].

        This depends only on the image features,
        not on the segmentation state.
        """

        f0, f1, f2 = features

        size = f0.shape[-2:]

        f1_up = F.interpolate(
            f1,
            size=size,
            mode="bilinear",
            align_corners=False,
        )

        f2_up = F.interpolate(
            f2,
            size=size,
            mode="bilinear",
            align_corners=False,
        )

        combined = torch.cat(
            (f0, f1_up, f2_up),
            dim=1,
        )

        return self.image_drive_head(combined)

    # =========================================================
    # Scalar energy
    # =========================================================

    def energy(self, image, state, *, features=None):

        if features is None:
            features = self.encode_image(image)

        f0, f1, f2 = features

        # Hierarchical state features
        u0 = state
        u1 = F.silu(self.K1(u0) + self.B1(f1))
        u2 = F.silu(self.K2(u1) + self.B2(f2))

        # Scalar energy densities at each scale
        densities = (
            self.head_fine(
                torch.cat((f0, u0), dim=1)
            ),
            self.head_half(
                torch.cat((f1, u1), dim=1)
            ),
            self.head_quarter(
                torch.cat((f2, u2), dim=1)
            ),
        )

        area = state.shape[-2] * state.shape[-1]

        result = sum(
            weight * density.sum()
            * (area / (density.shape[-2] * density.shape[-1]))
            for weight, density in zip(
                self.scale_weights, densities
            )
        )

        # -----------------------------------------------------
        # NEW: linear-in-state energy contribution
        # -----------------------------------------------------

        drive = self.image_drive(features)

        linear_energy = (
            -self.linear_drive_scale
            * (drive * state).sum()
        )

        l2_energy = (
            0.5 * self.state_l2 * state.square().sum()
        )

        return result + linear_energy + l2_energy

    # =========================================================
    # Gradient of scalar energy
    # =========================================================

    def state_gradient(
        self,
        image,
        state,
        *,
        create_graph,
        features=None,
    ):
        with torch.enable_grad():

            if not state.requires_grad:
                state = state.detach().requires_grad_(True)

            energy = self.energy(
                image, state, features=features
            )

            grad, = torch.autograd.grad(
                energy,
                state,
                create_graph=create_graph,
            )

        return grad

    # =========================================================
    # One energy-gradient update
    # =========================================================

    def update(
        self,
        image,
        state,
        *,
        differentiable=False,
        features=None,
    ):
        grad = self.state_gradient(
            image,
            state,
            create_graph=differentiable,
            features=features,
        )

        new_state = (
            state - self.log_step.exp() * grad
        )

        if differentiable:
            return new_state

        return new_state.detach()

    # =========================================================
    # Iterative inference
    # =========================================================

    def infer(self, image, *, steps=12, state=None):

        if state is None:
            state = self.initial_state(image)

        state = state.detach()

        # Cache image features once.
        with torch.no_grad():
            features = self.encode_image(image)

        for _ in range(steps):
            state = self.update(
                image,
                state,
                differentiable=False,
                features=features,
            )

        return state.detach()

    def forward(self, image, steps=12):
        return self.infer(image, steps=steps)


class RandomEnergyPerturber(nn.Module):
    """
    Frozen random scalar-energy CNN used to generate
    structured perturbations during rollout training.
    """

    def __init__(self, width=32):
        super().__init__()

        self.net = nn.Sequential(
            nn.Conv2d(5, width, 3, padding=1),
            nn.SiLU(),
            nn.Conv2d(
                width, width, 3,
                dilation=3, padding=3
            ),
            nn.SiLU(),
            nn.Conv2d(
                width, width, 3,
                dilation=7, padding=7
            ),
            nn.SiLU(),
            nn.Conv2d(width, 1, 1),
        )

        self.reset_random_weights()

    def reset_random_weights(self):
        for layer in self.net.modules():
            if hasattr(layer, "reset_parameters"):
                layer.reset_parameters()

        for param in self.parameters():
            param.requires_grad_(False)

    def forward(self, image, state):
        with torch.enable_grad():
            q = state.detach().requires_grad_(True)

            energy = self.net(
                torch.cat((image.detach(), q), dim=1)
            ).sum()

            grad, = torch.autograd.grad(
                energy, q
            )

        noise = -grad.detach()

        rms = (
            noise.square()
            .mean((1, 2, 3), keepdim=True)
            .sqrt()
            .clamp_min(1e-6)
        )

        return noise / rms


# =============================================================
# Smoke test
# =============================================================

if __name__ == "__main__":

    torch.manual_seed(2)

    model = HierarchicalSegmentationEnergy(
        image_width=4,
        state_width=4,
        head_width=6,
    )

    image = torch.rand(2, 2, 15, 17)
    state = model.initial_state(image)

    updated = model.update(
        image,
        state,
        differentiable=True,
    )

    updated.square().mean().backward()

    assert model.K1.weight.grad is not None
    assert model.K2.weight.grad is not None
    assert model.image_half[0].weight.grad is not None

    # Check that the added head receives gradients.
    assert model.image_drive_head[-1].weight.grad is not None

    with torch.no_grad():
        prediction = model.infer(image, steps=3)

    assert prediction.shape == (2, 3, 15, 17)

    print("Hierarchical energy model smoke test passed")
