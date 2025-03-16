import torch
import torch.nn as nn
import math
import numpy as np


def autopad(k, p=None):
    """Pad to 'same' shape outputs."""
    if p is None:
        p = k // 2 if isinstance(k, int) else [x // 2 for x in k]
    return p


class Conv(nn.Module):
    """Standard convolution with batch normalization and SiLU activation."""
    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, act=True):
        super().__init__()
        self.conv = nn.Conv2d(c1, c2, k, s, autopad(k, p), groups=g, bias=False)
        self.bn = nn.BatchNorm2d(c2)
        self.act = nn.SiLU() if act is True else (act if isinstance(act, nn.Module) else nn.Identity())

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))


class Bottleneck(nn.Module):
    """Standard bottleneck."""
    def __init__(self, c1, c2, shortcut=True, g=1, e=0.5):
        super().__init__()
        c_ = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = Conv(c_, c2, 3, 1, g=g)
        self.add = shortcut and c1 == c2

    def forward(self, x):
        return x + self.cv2(self.cv1(x)) if self.add else self.cv2(self.cv1(x))


class C2f(nn.Module):
    """CSP Bottleneck with 2 convolutions."""
    def __init__(self, c1, c2, n=1, shortcut=False, g=1, e=0.5):
        super().__init__()
        self.c = max(1, int(c2 * e))  # hidden channels, ensure at least 1 channel
        self.cv1 = Conv(c1, 2 * self.c, 1, 1)
        self.cv2 = Conv(max(1, (2 + n) * self.c), c2, 1)  # ensure at least 1 channel
        self.m = nn.ModuleList(Bottleneck(self.c, self.c, shortcut, g, e=1.0) for _ in range(n))

    def forward(self, x):
        y = list(self.cv1(x).chunk(2, 1))
        y.extend(m(y[-1]) for m in self.m)
        return self.cv2(torch.cat(y, 1))


class SCDown(nn.Module):
    """Spatial Connectivity Down module for efficient downsampling."""
    def __init__(self, c1, c2, k=3, s=2):
        super().__init__()
        self.conv = Conv(c1, c2, k, s)
        self.pool = nn.Sequential(
            nn.AvgPool2d(kernel_size=s, stride=s),
            Conv(c1, c2, 1, 1)
        )

    def forward(self, x):
        return self.conv(x) + self.pool(x)


class SPPF(nn.Module):
    """Spatial Pyramid Pooling - Fast (SPPF) layer."""
    def __init__(self, c1, c2, k=5):  # equivalent to SPP(k=(5, 9, 13))
        super().__init__()
        c_ = c1 // 2  # hidden channels
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = Conv(c_ * 4, c2, 1, 1)
        self.m = nn.MaxPool2d(kernel_size=k, stride=1, padding=k // 2)

    def forward(self, x):
        x = self.cv1(x)
        y1 = self.m(x)
        y2 = self.m(y1)
        return self.cv2(torch.cat((x, y1, y2, self.m(y2)), 1))


class C2fCIB(nn.Module):
    """CSP Bottleneck with 2 convolutions and Compact Involution Blocks."""
    def __init__(self, c1, c2, n=1, shortcut=False, g=1, use_cib=True):
        super().__init__()
        self.c = max(1, int(c2 * 0.5))  # hidden channels, ensure at least 1 channel
        self.cv1 = Conv(c1, 2 * self.c, 1, 1)
        self.cv2 = Conv(max(1, (2 + n) * self.c), c2, 1)  # ensure at least 1 channel
        
        if use_cib:
            # Create CIB blocks
            self.m = nn.ModuleList(CompactInvolutionBlock(self.c, self.c, shortcut) for _ in range(n))
        else:
            # Create regular bottlenecks
            self.m = nn.ModuleList(Bottleneck(self.c, self.c, shortcut, g) for _ in range(n))

    def forward(self, x):
        y = list(self.cv1(x).chunk(2, 1))
        y.extend(m(y[-1]) for m in self.m)
        return self.cv2(torch.cat(y, 1))


class CompactInvolutionBlock(nn.Module):
    """Compact Involution Block as described in YOLOv10."""
    def __init__(self, c1, c2, shortcut=True):
        super().__init__()
        c_ = c1 // 2  # hidden channels
        self.cv1 = Conv(c1, c_, 1, 1)
        self.inv = Involution(c_, c_, 3, 1)
        self.cv2 = Conv(c_, c2, 1, 1)
        self.shortcut = shortcut and c1 == c2

    def forward(self, x):
        return x + self.cv2(self.inv(self.cv1(x))) if self.shortcut else self.cv2(self.inv(self.cv1(x)))


class Involution(nn.Module):
    """Involution operation."""
    def __init__(self, c1, c2, k=3, s=1):
        super().__init__()
        self.channels = c2
        reduction = 4
        # Ensure group_channels is at least 1
        self.group_channels = max(1, 16)
        # Ensure groups is at least 1
        self.groups = max(1, self.channels // self.group_channels)
        self.conv1 = Conv(c1, c2, 1, 1)  # pointwise before involution
        self.conv2 = Conv(c2, c2, 1, 1)  # pointwise after involution
        
        # Kernel generation network
        reduced_channels = max(1, c1 // reduction)  # ensure at least 1 channel
        self.kernel_gen = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            Conv(c1, reduced_channels, 1),
            nn.SiLU(),
            Conv(reduced_channels, k * k * self.groups, 1)
        )
        self.kernel_size = k
        self.stride = s
        self.padding = k // 2

    def forward(self, x):
        b, c, h, w = x.shape
        x = self.conv1(x)
        
        # Generate kernels
        kernels = self.kernel_gen(x)
        kernels = kernels.view(b, self.groups, self.kernel_size, self.kernel_size, h, w)
        
        # Apply convolution with generated kernels (simplified implementation)
        out = x.view(b, self.groups, self.group_channels, h, w)
        out = self._involution(out, kernels)
        out = out.view(b, self.channels, h, w)
        
        return self.conv2(out)
    
    def _involution(self, x, kernels):
        b, g, c, h, w = x.shape
        # This is a simplified implementation, would normally use unfold+fold or a specialized CUDA kernel
        # Here we just apply the same kernel to all spatial positions
        # For a full implementation, kernels would vary per position
        k = self.kernel_size
        k_h = kernels.mean(dim=[-2, -1])  # Average kernels for simplicity
        
        # Apply uniform kernel to all positions
        out = torch.zeros_like(x)
        for i in range(g):
            out[:, i] = nn.functional.conv2d(
                x[:, i].unsqueeze(1),
                k_h[:, i].view(-1, 1, k, k),
                padding=self.padding,
                groups=b
            ).squeeze(1)
        
        return out


class Concat(nn.Module):
    """Concatenate a list of tensors along dimension."""
    def __init__(self, dimension=1):
        super().__init__()
        self.d = dimension

    def forward(self, x):
        return torch.cat(x, self.d) 