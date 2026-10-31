"""Input-conditioned, shared-energy variational nuclei segmentation with CSV diagnostics.

Input: 2 fixed imaging channels. Output: 3 segmentation logits.
Mutable state: 3 logits + configurable auxiliary channels. The same scalar
energy is used for every optimization step. The energy CNN uses SiLU
activations and a 1-3-7 dilation schedule without GroupNorm. Inference uses
Euler-started Adams-Bashforth 2 (AB2) by default, with Euler as a control.

Use CSVTrainingLogger.log_step() after backward, before optimizer.step().
Use CSVTrainingLogger.log_trajectory() occasionally on a *small fixed*
validation batch; this reruns the optimization and is comparatively costly.
"""

import csv
from pathlib import Path
import time

import torch
from torch import nn
from torch.nn import functional as F


class ConditionalEnergy(nn.Module):
    def __init__(self, input_channels=2, state_channels=8,
                 hidden_channels=328, num_groups=8, state_l2=0.01):
        super().__init__()
        if hidden_channels % num_groups:
            raise ValueError('hidden_channels must be divisible by num_groups')
        self.state_l2 = state_l2
        # Increasing dilation expands context without adding parameters.
        # 3x3 kernels with dilations (1, 3, 7) have a 23x23
        # receptive field for each energy-density output pixel.
        # Set padding=dilation to keep the spatial dimensions fixed.
        self.net = nn.Sequential(
            nn.Conv2d(
                input_channels + state_channels,
                hidden_channels,
                kernel_size=3,
                padding=1,
                dilation=1,
                bias=False,
            ),
            nn.SiLU(),

            nn.Conv2d(
                hidden_channels,
                hidden_channels,
                kernel_size=3,
                padding=3,
                dilation=3,
                bias=False,
            ),
            nn.SiLU(),

            nn.Conv2d(
                hidden_channels,
                hidden_channels,
                kernel_size=3,
                padding=3,
                dilation=3,
                bias=False,
            ),
            nn.SiLU(),

            nn.Conv2d(
                hidden_channels,
                hidden_channels,
                kernel_size=3,
                padding=7,
                dilation=7,
                bias=False,
            ),
            nn.SiLU(),

            nn.Conv2d(hidden_channels, 1, kernel_size=1, bias=False),
        )

    def per_sample(self, image, state):
        density = self.net(torch.cat((image, state), dim=1))
        return density.flatten(1).sum(dim=1) + (
            0.5 * self.state_l2 * state.square().flatten(1).sum(dim=1)
        )

    def forward(self, image, state):
        # SUM, not mean, over batch. Each image then follows exactly the
        # same gradient trajectory regardless of other images in the batch.
        return self.per_sample(image, state).sum()


class VariationalNucleiSegmenter(nn.Module):
    def __init__(self, input_channels=2, out_channels=3, auxiliary_channels=5,
                 hidden_channels=328, num_groups=8, steps=10,
                 initial_step=0.1, state_l2=0.01, integrator="ab2"):
        super().__init__()
        if integrator not in ("ab2", "euler"):
            raise ValueError("integrator must be 'ab2' or 'euler'")
        self.integrator = integrator
        if initial_step <= 0:
            raise ValueError('initial_step must be positive')
        self.out_channels = out_channels
        self.state_channels = out_channels + auxiliary_channels
        self.steps = steps
        self.energy_model = ConditionalEnergy(
            input_channels, self.state_channels, hidden_channels,
            num_groups, state_l2
        )
        self.log_step = nn.Parameter(torch.tensor(initial_step).log())
        # Positive diagonal preconditioner, one multiplier per state channel.
        # Zero initialization gives all multipliers = 1 (original Euler update).
        self.log_channel_scale = nn.Parameter(torch.zeros(self.state_channels))

    def channel_scales(self):
        # Geometric mean of scales is 1: overall speed stays in log_step,
        # while these parameters learn only relative per-channel speeds.
        log_scales = self.log_channel_scale
        return (log_scales - log_scales.mean()).exp().view(1, -1, 1, 1)

    def forward(self, image, steps=None, return_state=False):
        """Euler startup, then AB2 (or Euler throughout if requested).

        Every step evaluates the CURRENT energy gradient exactly once. The
        PREVIOUS gradient is reused without recomputation. Training retains
        its graph so differentiation flows through both AB2 branches.
        """
        nsteps = self.steps if steps is None else steps
        state = image.new_zeros(
            image.shape[0], self.state_channels, *image.shape[-2:]
        )
        step = self.log_step.exp()
        channel_scales = self.channel_scales()
        previous_gradient = None

        for _ in range(nsteps):
            with torch.enable_grad():
                state = state.requires_grad_(True)
                energy = self.energy_model(image, state)
                gradient = torch.autograd.grad(
                    energy, state, create_graph=self.training
                )[0]

            if self.integrator == "ab2" and previous_gradient is not None:
                direction = 1.5 * gradient - 0.5 * previous_gradient
            else:
                direction = gradient  # Euler startup (or Euler baseline)

            state = state - step * channel_scales * direction
            if not self.training:
                state = state.detach()
                previous_gradient = gradient.detach()
            else:
                previous_gradient = gradient

        logits = state[:, :self.out_channels]
        return (logits, state) if return_state else logits

    def diagnostic_trajectory(self, image, labels=None, steps=40):
        """Separate diagnostic forward pass; no parameter gradients retained.

        Each entry describes the state BEFORE the corresponding update. The
        final state is included as an additional entry at index `steps`.
        Does not modify model parameters, mode, or the actual training state.
        """
        state = image.detach().new_zeros(
            image.shape[0], self.state_channels, *image.shape[-2:]
        )
        step_size = self.log_step.detach().exp()
        channel_scales = self.channel_scales().detach()
        records = []
        previous_gradient = None
        for t in range(steps + 1):
            with torch.enable_grad():
                state = state.detach().requires_grad_(True)
                # Detach parameters implicitly: autograd.grad requests only
                # state derivatives, and create_graph=False prevents second order.
                energy = self.energy_model(image.detach(), state)
                g = torch.autograd.grad(energy, state)[0]
            if self.integrator == "ab2" and previous_gradient is not None:
                direction = 1.5 * g - 0.5 * previous_gradient
            else:
                direction = g
            with torch.no_grad():
                q = state.detach()
                logits = q[:, :self.out_channels]
                row = {
                    'iteration': t,
                    'energy_per_pixel': (energy.detach() / (image.shape[0] * image.shape[-2] * image.shape[-1])).item(),
                    'state_grad_rms': g.square().mean().sqrt().item(),
                    'state_rms': q.square().mean().sqrt().item(),
                    'logits_rms': logits.square().mean().sqrt().item(),
                    'aux_rms': q[:, self.out_channels:].square().mean().sqrt().item()
                    if self.state_channels > self.out_channels else 0.0,
                    'update_rms': (step_size * channel_scales * direction).square().mean().sqrt().item() if t < steps else 0.0,
                    'step_size': step_size.item(),
                }
                for c in range(self.state_channels):
                    ch = q[:, c].float()
                    row[f'state_{c}_mean'] = ch.mean().item()
                    row[f'state_{c}_std'] = ch.std(unbiased=False).item()
                    row[f'channel_scale_{c}'] = channel_scales[0, c, 0, 0].item()
                if labels is not None:
                    row['cross_entropy'] = F.cross_entropy(logits, labels).item()
                    pred = logits.argmax(dim=1)
                    for c in range(self.out_channels):
                        pr = pred == c
                        gt = labels == c
                        intersect = (pr & gt).sum().item()
                        denom = pr.sum().item() + gt.sum().item()
                        row[f'dice_class_{c}'] = (2 * intersect / denom) if denom else 1.0
                records.append(row)
                if t < steps:
                    state = q - step_size * channel_scales * direction.detach()
                    previous_gradient = g.detach()
        for t in range(steps):
            records[t]['energy_change'] = (records[t+1]['energy_per_pixel'] - records[t]['energy_per_pixel'])
            records[t]['energy_increased'] = int(records[t]['energy_change'] > 1e-7)
        records[-1]['energy_change'] = 0.0
        records[-1]['energy_increased'] = 0
        return records


def num_parameters(module):
    return sum(p.numel() for p in module.parameters() if p.requires_grad)


class CSVTrainingLogger:
    """Append training scalars and occasional expensive trajectories to CSV.

    No modification to the optimizer/training loop except calls to log_step()
    after loss.backward() and optional log_trajectory() on validation data.
    """
    def __init__(self, directory='variational_logs', scalar_every=25,
                 detailed_every=200, flush_every=10):
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=True)
        self.scalar_every = scalar_every
        self.detailed_every = detailed_every
        self.flush_every = flush_every
        self._writers = {}
        self._files = {}
        self._counts = {}
        self._start = time.monotonic()

    def _write(self, filename, data):
        if filename not in self._writers:
            path = self.directory / filename
            exists = path.exists() and path.stat().st_size > 0
            if exists:
                with path.open('r', newline='') as existing:
                    old_header = next(csv.reader(existing))
                if old_header != list(data):
                    raise ValueError(f'CSV columns in {path} differ from current configuration')
            f = path.open('a', newline='', buffering=1)
            self._files[filename] = f
            self._writers[filename] = csv.DictWriter(f, fieldnames=list(data))
            self._counts[filename] = 0
            if not exists:
                self._writers[filename].writeheader()
        self._writers[filename].writerow(data)
        self._counts[filename] += 1
        if self._counts[filename] % self.flush_every == 0:
            self._files[filename].flush()

    @torch.no_grad()
    def log_step(self, model, step, loss, *, val_loss=None, extra=None):
        """Call AFTER backward and BEFORE optimizer.zero_grad()/step().

        GPU-to-CPU .item() operations synchronize the GPU. Keeping these
        infrequent is the main way to limit logging overhead.
        """
        if step % self.scalar_every:
            return
        row = {
            'step': step,
            'wall_seconds': round(time.monotonic() - self._start, 2),
            'train_loss': float(loss.detach()),
            'val_loss': float(val_loss.detach()) if isinstance(val_loss, torch.Tensor)
            else (float(val_loss) if val_loss is not None else ''),
            'step_size': model.log_step.detach().exp().item(),
        }
        if extra:
            row.update(extra)  # Keep keys consistent between calls.
        self._write('training.csv', row)
        if step % self.detailed_every:
            return
        summary = {'step': step}
        for name, p in model.named_parameters():
            summary[f'{name}.weight_rms'] = p.detach().float().square().mean().sqrt().item()
            summary[f'{name}.grad_rms'] = (
                p.grad.detach().float().square().mean().sqrt().item()
                if p.grad is not None else ''
            )
        self._write('parameter_stats.csv', summary)

    def log_trajectory(self, model, image, labels, step, *, iterations=40):
        """Run on a small FIXED validation batch, e.g., every 500 steps.

        Requires no optimizer step; keep train/eval mode unchanged.
        Images/labels should already be on the model device.
        """
        rows = model.diagnostic_trajectory(image, labels, steps=iterations)
        for row in rows:
            self._write('trajectories.csv', {'step': step, **row})
        return rows

    def close(self):
        for f in self._files.values():
            f.close()
        self._files.clear()
        self._writers.clear()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()


if __name__ == '__main__':
    torch.manual_seed(0)
    model = VariationalNucleiSegmenter(hidden_channels=16, steps=2)
    x = torch.randn(1, 2, 16, 16)
    y = torch.randint(0, 3, (1, 16, 16))
    pred = model(x)
    loss = F.cross_entropy(pred, y)
    loss.backward()
    assert pred.shape == (1, 3, 16, 16)
    assert model.energy_model.net[0].weight.grad is not None
    assert model.log_channel_scale.grad is not None
    assert torch.allclose(model.channel_scales(), torch.ones_like(model.channel_scales()), atol=1e-5)
    assert model.energy_model.net[4].dilation == (7, 7)
    model.eval()
    with torch.no_grad():
        assert model(x).shape == pred.shape
        trajectory = model.diagnostic_trajectory(x, y, steps=3)
    assert len(trajectory) == 4
    # When only one update is requested, AB2 and Euler must agree.
    euler = VariationalNucleiSegmenter(hidden_channels=16, steps=2, integrator="euler")
    euler.load_state_dict(model.state_dict())
    model.train()
    euler.train()
    assert torch.allclose(model(x, steps=1), euler(x, steps=1), atol=1e-6)
    print('Smoke test passed; default parameters:', num_parameters(VariationalNucleiSegmenter()))
