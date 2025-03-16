import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class DynamicSparseAttention(nn.Module):
    """
    Dynamic Sparse Attention module that selectively activates computational pathways
    based on input complexity. This module combines the efficiency of sparse attention
    with the adaptability of dynamic networks.
    """
    def __init__(self, dim, reduction=8, min_channels=32):
        super().__init__()
        self.dim = dim
        self.reduction = reduction
        self.min_channels = min_channels
        
        # Channel attention components
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        
        # Complexity estimator
        self.complexity_estimator = nn.Sequential(
            nn.Conv2d(dim, max(dim // reduction, min_channels), kernel_size=1, bias=False),
            nn.SiLU(),
            nn.Conv2d(max(dim // reduction, min_channels), 1, kernel_size=1, bias=False),
            nn.Sigmoid()
        )
        
        # Sparse attention components
        self.q_conv = nn.Conv2d(dim, dim, kernel_size=1, bias=False)
        self.k_conv = nn.Conv2d(dim, dim, kernel_size=1, bias=False)
        self.v_conv = nn.Conv2d(dim, dim, kernel_size=1, bias=False)
        
        # Output projection
        self.proj = nn.Conv2d(dim, dim, kernel_size=1, bias=False)
        
        # Initialization
        self._reset_parameters()
    
    def _reset_parameters(self):
        # Initialize the complexity estimator with small weights
        for m in self.complexity_estimator.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
        
        # Initialize attention components
        nn.init.xavier_uniform_(self.q_conv.weight)
        nn.init.xavier_uniform_(self.k_conv.weight)
        nn.init.xavier_uniform_(self.v_conv.weight)
        nn.init.xavier_uniform_(self.proj.weight)
    
    def forward(self, x):
        B, C, H, W = x.shape
        
        # Estimate input complexity
        avg_feat = self.avg_pool(x)
        max_feat = self.max_pool(x)
        complexity = self.complexity_estimator(avg_feat + max_feat)
        
        # Apply sparse attention based on complexity
        if self.training or complexity.mean() > 0.5:
            # Full attention path
            q = self.q_conv(x).flatten(2).transpose(1, 2)  # B, HW, C
            k = self.k_conv(x).flatten(2)  # B, C, HW
            v = self.v_conv(x).flatten(2).transpose(1, 2)  # B, HW, C
            
            # Compute attention scores with linear complexity approximation
            attn = torch.bmm(q, k) * (1.0 / math.sqrt(C))
            attn = F.softmax(attn, dim=-1)
            
            # Apply attention and reshape
            x_attn = torch.bmm(attn, v).transpose(1, 2).reshape(B, C, H, W)
            
            # Combine with original features based on complexity
            x_out = x + self.proj(x_attn) * complexity
        else:
            # Skip attention for low-complexity inputs (inference optimization)
            x_out = x
        
        return x_out


class LightweightTransformer(nn.Module):
    """
    Lightweight Transformer block optimized for object detection.
    Uses linear attention mechanism for efficiency.
    """
    def __init__(self, dim, num_heads=8, head_dim=128, dropout=0.1):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.scale = head_dim ** -0.5
        
        # Multi-head linear attention
        self.qkv = nn.Conv2d(dim, 3 * num_heads * head_dim, kernel_size=1, bias=False)
        self.proj = nn.Conv2d(num_heads * head_dim, dim, kernel_size=1, bias=False)
        
        # Feed-forward network
        self.ffn = nn.Sequential(
            nn.Conv2d(dim, dim * 2, kernel_size=1),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Conv2d(dim * 2, dim, kernel_size=1)
        )
        
        # Layer normalization (implemented as instance norm for 2D feature maps)
        self.norm1 = nn.GroupNorm(1, dim)
        self.norm2 = nn.GroupNorm(1, dim)
        
        # Initialization
        self._reset_parameters()
    
    def _reset_parameters(self):
        nn.init.xavier_uniform_(self.qkv.weight)
        nn.init.xavier_uniform_(self.proj.weight)
        for m in self.ffn.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
    
    def forward(self, x):
        B, C, H, W = x.shape
        
        # Layer norm
        shortcut = x
        x = self.norm1(x)
        
        # Multi-head linear attention
        qkv = self.qkv(x).reshape(B, 3, self.num_heads, self.head_dim, H * W)
        q, k, v = qkv.unbind(1)  # Each is [B, num_heads, head_dim, H*W]
        
        # Linear attention (more efficient than standard attention)
        k = F.softmax(k, dim=-1)
        context = torch.matmul(k, v.transpose(-2, -1))  # [B, num_heads, head_dim, head_dim]
        out = torch.matmul(context, q)  # [B, num_heads, head_dim, H*W]
        
        # Reshape and project
        out = out.reshape(B, self.num_heads * self.head_dim, H, W)
        out = self.proj(out)
        
        # First residual connection
        x = shortcut + out
        
        # FFN with residual
        x = x + self.ffn(self.norm2(x))
        
        return x 