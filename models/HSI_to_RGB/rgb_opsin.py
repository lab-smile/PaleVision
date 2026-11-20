"""RGB opsin layer for MinnApple detection/counting (3-channel input)."""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class RGBOpsinLayer(nn.Module):
    """Simulate opsin spectral response on RGB channels with Gaussian kernels."""

    def __init__(self, peaks=(650.0, 550.0, 450.0), trainable=False, std=25.0):
        super().__init__()
        self.std = std
        # Map RGB band indices to pseudo-wavelengths 450-650 nm
        wl = torch.tensor([450.0, 550.0, 650.0])
        self.register_buffer('wavelengths', wl)
        self.lamda_max_1 = nn.Parameter(torch.tensor(float(peaks[0])), requires_grad=trainable)
        self.lamda_max_2 = nn.Parameter(torch.tensor(float(peaks[1])), requires_grad=trainable)
        self.lamda_max_3 = nn.Parameter(torch.tensor(float(peaks[2])), requires_grad=trainable)
        self.trainable = trainable

    def _kernel(self, lam):
        std = self.std
        x = self.wavelengths
        return (1 / (std * (2 * math.pi) ** 0.5) *
                torch.exp(-((x - lam) ** 2) / (2 * std ** 2)))

    def forward(self, x):
        # 3 opsins × 3 RGB bands -> [3, 3] spectral mixing matrix
        k1 = self._kernel(self.lamda_max_1)
        k2 = self._kernel(self.lamda_max_2)
        k3 = self._kernel(self.lamda_max_3)
        M = torch.stack([k1, k2, k3], dim=0)
        M = M / (M.sum(dim=1, keepdim=True) + 1e-8)
        weight = M.view(3, 3, 1, 1)
        return F.conv2d(x, weight)


class RGBFreeConvOpsin(nn.Module):
    """FreeConv-style learnable 3×3 spectral mixing."""

    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(3, 3, kernel_size=1, bias=False)

    def forward(self, x):
        return self.conv(x)
