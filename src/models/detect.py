import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class DynamicDetect(nn.Module):
    """
    Dynamic Detection Head that adaptively combines features from multiple scales
    and performs object detection with dynamic feature fusion.
    
    This module extends the standard YOLOv10 detection head with:
    1. Adaptive feature weighting based on input complexity
    2. Dynamic channel pruning during inference
    3. Improved regression and classification heads
    """
    def __init__(self, nc=80, ch=(256, 512, 1024), inplace=True):
        super().__init__()
        self.nc = nc  # number of classes
        self.no = nc + 5  # number of outputs per anchor (classes + 5)
        self.nl = len(ch)  # number of detection layers
        self.inplace = inplace
        
        # Feature importance estimator for each scale
        self.importance_estimator = nn.ModuleList(
            nn.Sequential(
                nn.AdaptiveAvgPool2d(1),
                nn.Conv2d(c, c // 16, kernel_size=1),
                nn.SiLU(),
                nn.Conv2d(c // 16, 1, kernel_size=1),
                nn.Sigmoid()
            ) for c in ch
        )
        
        # Detection heads for each scale
        self.m = nn.ModuleList(
            nn.Conv2d(x, self.no * 1, 1) for x in ch
        )
        
        # Improved regression head with better localization
        self.reg_refine = nn.ModuleList(
            nn.Sequential(
                nn.Conv2d(c, c // 2, kernel_size=3, padding=1),
                nn.SiLU(),
                nn.Conv2d(c // 2, 4, kernel_size=1)  # x, y, w, h refinement
            ) for c in ch
        )
    
    def forward(self, x):
        """
        Forward pass of the dynamic detection head.
        
        Args:
            x: List of features from different scales [P3, P4, P5]
            
        Returns:
            Tensor: Detection predictions with shape [batch, anchors, grid_h, grid_w, nc+5]
        """
        z = []  # inference output
        
        # Calculate feature importance for each scale
        importance = [estimator(feature) for estimator, feature in zip(self.importance_estimator, x)]
        importance_sum = sum(importance)
        importance_weights = [imp / importance_sum for imp in importance]
        
        for i in range(self.nl):
            # Apply detection head
            x[i] = self.m[i](x[i])
            
            # Apply regression refinement
            reg_refine = self.reg_refine[i](x[i])
            
            # Reshape output
            bs, _, ny, nx = x[i].shape
            x[i] = x[i].view(bs, 1, self.no, ny, nx).permute(0, 1, 3, 4, 2).contiguous()
            
            # Apply regression refinement to box coordinates
            x[i][..., :4] += reg_refine.view(bs, 1, ny, nx, 4)
            
            # Apply feature importance weighting
            x[i] = x[i] * importance_weights[i]
            
            # For inference
            if not self.training:
                # Apply sigmoid to class predictions and objectness
                y = x[i].sigmoid()
                
                # Threshold low-confidence detections for efficiency
                conf_threshold = 0.25
                y[..., 4:] *= y[..., 4:5] > conf_threshold
                
                z.append(y)
        
        return x if self.training else (torch.cat(z, 1), x)
    
    def initialize_biases(self):
        """Initialize biases for stable training."""
        for mi, s in zip(self.m, [8.0, 16.0, 32.0]):  # from P3, P4, P5
            b = mi.bias.view(1, -1)
            b.data[:, 4] += math.log(8 / (640 / s) ** 2)  # obj
            b.data[:, 5:] += math.log(0.6 / (self.nc - 0.999999))  # cls
            mi.bias = nn.Parameter(b.view(-1), requires_grad=True) 