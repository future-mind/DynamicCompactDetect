import torch
import torch.nn as nn
import torch.nn.functional as F
import platform
import math
from typing import List, Tuple, Dict, Optional, Union

class DWConv(nn.Module):
    """Depthwise separable convolution for lightweight feature extraction."""
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1, dilation=1):
        super().__init__()
        self.depthwise = nn.Conv2d(
            in_channels, in_channels, kernel_size, stride, padding, dilation, 
            groups=in_channels, bias=False
        )
        self.pointwise = nn.Conv2d(in_channels, out_channels, 1, 1, 0, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)
        self.act = nn.SiLU()
        
    def forward(self, x):
        x = self.depthwise(x)
        x = self.pointwise(x)
        x = self.bn(x)
        x = self.act(x)
        return x


class LightweightAttention(nn.Module):
    """Efficient attention mechanism that captures global context with minimal computation."""
    def __init__(self, channels, reduction_ratio=8):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc1 = nn.Conv2d(channels, channels // reduction_ratio, kernel_size=1)
        self.act = nn.SiLU()
        self.fc2 = nn.Conv2d(channels // reduction_ratio, channels, kernel_size=1)
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, x):
        b, c, _, _ = x.shape
        y = self.avg_pool(x)
        y = self.fc1(y)
        y = self.act(y)
        y = self.fc2(y)
        y = self.sigmoid(y)
        return x * y


class EfficientResidual(nn.Module):
    """Efficient residual block using depthwise separable convolutions."""
    def __init__(self, channels, expansion_factor=2):
        super().__init__()
        expanded_channels = channels * expansion_factor
        self.conv1 = nn.Conv2d(channels, expanded_channels, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(expanded_channels)
        self.act1 = nn.SiLU()
        
        self.conv2 = DWConv(expanded_channels, expanded_channels, kernel_size=3, padding=1)
        
        self.attention = LightweightAttention(expanded_channels)
        
        self.conv3 = nn.Conv2d(expanded_channels, channels, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(channels)
        
        # Skip connection
        self.skip = nn.Identity()
        self.act_out = nn.SiLU()
        
    def forward(self, x):
        residual = x
        
        # Expansion
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.act1(x)
        
        # Depthwise convolution
        x = self.conv2(x)
        
        # Attention
        x = self.attention(x)
        
        # Projection
        x = self.conv3(x)
        x = self.bn3(x)
        
        # Skip connection
        x += self.skip(residual)
        x = self.act_out(x)
        
        return x


class GatingModule(nn.Module):
    """Gating module for selective routing of features based on input complexity."""
    def __init__(self, channels, threshold=0.5):
        super().__init__()
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc1 = nn.Linear(channels, channels // 4)
        self.act = nn.SiLU()
        self.fc2 = nn.Linear(channels // 4, 1)
        self.sigmoid = nn.Sigmoid()
        self.threshold = threshold
        
    def forward(self, x):
        b, c, _, _ = x.shape
        y = self.pool(x).view(b, c)
        y = self.fc1(y)
        y = self.act(y)
        y = self.fc2(y)
        gate_value = self.sigmoid(y)
        
        # Return the gate value and a binary decision
        return gate_value, (gate_value > self.threshold)


class DynamicBlock(nn.Module):
    """A dynamic block that can be bypassed based on input complexity."""
    def __init__(self, channels, threshold=0.5):
        super().__init__()
        self.gate = GatingModule(channels, threshold)
        self.main_path = EfficientResidual(channels)
        self.skip_path = nn.Identity()
        
    def forward(self, x):
        gate_value, use_main_path = self.gate(x)
        
        if self.training:
            # During training, use soft gating with gate_value
            main_output = self.main_path(x)
            skip_output = self.skip_path(x)
            return gate_value * main_output + (1 - gate_value) * skip_output
        else:
            # During inference, use hard gating based on the threshold
            if use_main_path:
                return self.main_path(x)
            else:
                return self.skip_path(x)


class SelectiveRoutingModule(nn.Module):
    """Module that selectively routes computation through different paths based on input complexity."""
    def __init__(self, in_channels, out_channels, num_experts=4):
        super().__init__()
        self.num_experts = num_experts
        
        # Router network to decide which experts to use
        self.router = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(in_channels, 128),
            nn.SiLU(),
            nn.Linear(128, num_experts),
            nn.Softmax(dim=1)
        )
        
        # Expert networks (different computational paths)
        self.experts = nn.ModuleList([
            EfficientResidual(in_channels) for _ in range(num_experts)
        ])
        
        # Final projection if needed
        self.projection = nn.Conv2d(in_channels, out_channels, kernel_size=1) if in_channels != out_channels else nn.Identity()
        
    def forward(self, x):
        # Compute routing weights
        routing_weights = self.router(x)  # [batch_size, num_experts]
        
        if self.training:
            # During training, compute weighted sum of all experts
            out = torch.zeros_like(x)
            for i, expert in enumerate(self.experts):
                weight = routing_weights[:, i].view(-1, 1, 1, 1)
                out += weight * expert(x)
        else:
            # During inference, only use the top expert (most confident)
            expert_idx = torch.argmax(routing_weights, dim=1)
            out = torch.zeros_like(x)
            for b in range(x.shape[0]):  # Process each item in batch
                # Get the selected expert for this batch item
                idx = expert_idx[b].item()
                out[b] = self.experts[idx](x[b:b+1])
        
        return self.projection(out)


class EarlyExitBlock(nn.Module):
    """Early exit block that can terminate inference early if confidence is high enough."""
    def __init__(self, in_channels, num_classes, confidence_threshold=0.8):
        super().__init__()
        self.confidence_threshold = confidence_threshold
        
        # Lightweight feature processor
        self.processor = nn.Sequential(
            DWConv(in_channels, in_channels // 2),
            nn.SiLU(),
            DWConv(in_channels // 2, in_channels // 4),
            nn.SiLU(),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten()
        )
        
        # Classifier
        self.classifier = nn.Sequential(
            nn.Linear(in_channels // 4, in_channels // 8),
            nn.SiLU(),
            nn.Linear(in_channels // 8, num_classes),
        )
        
        # Confidence estimator
        self.confidence = nn.Sequential(
            nn.Linear(in_channels // 4, 64),
            nn.SiLU(),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )
        
    def forward(self, x, return_confidence=False):
        features = self.processor(x)
        logits = self.classifier(features)
        confidence = self.confidence(features)
        
        if return_confidence:
            return logits, confidence
        return logits
    
    def is_confident(self, x):
        """Check if the early exit is confident enough."""
        _, confidence = self.forward(x, return_confidence=True)
        return confidence > self.confidence_threshold


class DetectionHead(nn.Module):
    """Detection head that generates bounding boxes and class probabilities."""
    def __init__(self, in_channels, num_classes, num_anchors=3):
        super().__init__()
        
        # Each anchor box predicts 5 values (x, y, w, h, obj) + num_classes
        out_channels = num_anchors * (5 + num_classes)
        
        self.conv = nn.Sequential(
            DWConv(in_channels, in_channels),
            DWConv(in_channels, in_channels),
            nn.Conv2d(in_channels, out_channels, kernel_size=1)
        )
        
        self.num_classes = num_classes
        self.num_anchors = num_anchors
        
    def forward(self, x):
        return self.conv(x)


class FeaturePyramidNetwork(nn.Module):
    """Feature Pyramid Network for multi-scale feature extraction."""
    def __init__(self, in_channels_list, out_channels):
        super().__init__()
        
        # Lateral connections (1x1 convs to reduce channel dimensions)
        self.lateral_convs = nn.ModuleList([
            nn.Conv2d(in_channels, out_channels, kernel_size=1)
            for in_channels in in_channels_list
        ])
        
        # FPN connections (3x3 convs after upsampling)
        self.fpn_convs = nn.ModuleList([
            DWConv(out_channels, out_channels, kernel_size=3, padding=1)
            for _ in range(len(in_channels_list))
        ])
        
    def forward(self, features):
        # features is a list of feature maps from different stages
        
        # Process the deepest layer first
        results = [self.lateral_convs[-1](features[-1])]
        
        # Process from deep to shallow
        for i in range(len(features) - 2, -1, -1):
            # Lateral connection
            lateral = self.lateral_convs[i](features[i])
            
            # Top-down connection with upsampling
            top_down = F.interpolate(results[0], size=lateral.shape[2:], mode='nearest')
            
            # Merge
            results.insert(0, lateral + top_down)
        
        # Apply 3x3 convolutions
        for i in range(len(results)):
            results[i] = self.fpn_convs[i](results[i])
            
        return results


class HardwareOptimizer:
    """Class to handle hardware-specific optimizations."""
    @staticmethod
    def get_device():
        """Determine the best available device for training/inference."""
        if torch.cuda.is_available():
            return torch.device('cuda')
        elif platform.system() == 'Darwin' and hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            return torch.device('mps')  # For Mac with Apple Silicon
        else:
            return torch.device('cpu')
    
    @staticmethod
    def optimize_for_platform(model):
        """Apply platform-specific optimizations to the model."""
        device = HardwareOptimizer.get_device()
        
        if device.type == 'cuda':
            # CUDA-specific optimizations
            if torch.cuda.is_available():
                # Use mixed precision where appropriate
                model = model.to(device)
                
                # Check if TensorRT is available (no implementation here, just a placeholder)
                try:
                    import tensorrt
                    print("TensorRT available for further optimization")
                except ImportError:
                    pass
                    
        elif device.type == 'mps':
            # Metal-specific optimizations for Mac
            model = model.to(device)
            
        return model, device


class Backbone(nn.Module):
    """Backbone network with dynamic routing capabilities."""
    def __init__(self, in_channels=3, base_channels=32, dynamic_blocks=True):
        super().__init__()
        
        # Initial convolution
        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, base_channels, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(base_channels),
            nn.SiLU()
        )
        
        # Stage 1
        self.stage1 = self._make_stage(base_channels, base_channels * 2, num_blocks=2, 
                                       dynamic=dynamic_blocks, stride=2)
        
        # Stage 2
        self.stage2 = self._make_stage(base_channels * 2, base_channels * 4, num_blocks=3, 
                                       dynamic=dynamic_blocks, stride=2)
        
        # Stage 3
        self.stage3 = self._make_stage(base_channels * 4, base_channels * 8, num_blocks=4, 
                                       dynamic=dynamic_blocks, stride=2)
        
        # Stage 4
        self.stage4 = self._make_stage(base_channels * 8, base_channels * 16, num_blocks=2, 
                                        dynamic=dynamic_blocks, stride=2)
        
        # Early exit branches
        self.early_exit1 = EarlyExitBlock(base_channels * 4, 80)  # After stage 2
        self.early_exit2 = EarlyExitBlock(base_channels * 8, 80)  # After stage 3
        
        self.dynamic_blocks = dynamic_blocks
        
    def _make_stage(self, in_channels, out_channels, num_blocks, dynamic=True, stride=1):
        # Downsample at the beginning of each stage
        layers = [DWConv(in_channels, out_channels, stride=stride)]
        
        # Add blocks
        for _ in range(num_blocks):
            if dynamic:
                layers.append(DynamicBlock(out_channels))
            else:
                layers.append(EfficientResidual(out_channels))
                
        return nn.Sequential(*layers)
    
    def forward(self, x, early_exit=True):
        # Feature extraction
        x = self.stem(x)
        x = self.stage1(x)
        x = self.stage2(x)
        
        # First early exit point
        if early_exit and not self.training and self.early_exit1.is_confident(x):
            return self.early_exit1(x), None, None
            
        f1 = x  # Save intermediate feature map
        
        x = self.stage3(x)
        
        # Second early exit point
        if early_exit and not self.training and self.early_exit2.is_confident(x):
            return self.early_exit2(x), f1, None
            
        f2 = x  # Save intermediate feature map
        
        x = self.stage4(x)
        f3 = x  # Final feature map
        
        return None, [f1, f2, f3], None  # No early exit used, return all feature maps


class DynamicCompactDetect(nn.Module):
    """Main DynamicCompactDetect model integrating all components."""
    def __init__(self, num_classes=80, in_channels=3, base_channels=32):
        super().__init__()
        
        self.num_classes = num_classes
        
        # Backbone
        self.backbone = Backbone(in_channels, base_channels, dynamic_blocks=True)
        
        # Feature Pyramid Network
        in_channels_list = [base_channels * 4, base_channels * 8, base_channels * 16]
        fpn_channels = base_channels * 8
        self.fpn = FeaturePyramidNetwork(in_channels_list, fpn_channels)
        
        # Selective routing modules for each FPN level
        self.routing1 = SelectiveRoutingModule(fpn_channels, fpn_channels)
        self.routing2 = SelectiveRoutingModule(fpn_channels, fpn_channels)
        self.routing3 = SelectiveRoutingModule(fpn_channels, fpn_channels)
        
        # Detection heads for each level
        self.det_head1 = DetectionHead(fpn_channels, num_classes)
        self.det_head2 = DetectionHead(fpn_channels, num_classes)
        self.det_head3 = DetectionHead(fpn_channels, num_classes)
        
        # Hardware optimization
        self.hardware_optimizer = HardwareOptimizer()
        
        # Initialize weights
        self._initialize_weights()
        
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='leaky_relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
                    
    def forward(self, x, early_exit=True):
        # Get device-optimized model if needed
        device = self.hardware_optimizer.get_device()
        x = x.to(device)
        
        # Run backbone with potential early exits
        early_exit_result, features, _ = self.backbone(x, early_exit=early_exit)
        
        # If early exit was taken, return its result
        if early_exit_result is not None:
            return {'early_exit': early_exit_result}
            
        # Process features through FPN
        fpn_features = self.fpn(features)
        
        # Apply selective routing to each level
        routed_features = [
            self.routing1(fpn_features[0]),
            self.routing2(fpn_features[1]),
            self.routing3(fpn_features[2])
        ]
        
        # Apply detection heads
        outputs = [
            self.det_head1(routed_features[0]),
            self.det_head2(routed_features[1]),
            self.det_head3(routed_features[2])
        ]
        
        return {
            'outputs': outputs,
            'features': features,
            'fpn_features': fpn_features,
            'routed_features': routed_features
        }
    
    def optimize_for_platform(self):
        """Apply platform-specific optimizations to the model."""
        return self.hardware_optimizer.optimize_for_platform(self)


if __name__ == '__main__':
    # Create model
    model = DynamicCompactDetect(num_classes=80)
    
    # Optimize for current platform
    model, device = model.optimize_for_platform()
    
    # Test forward pass with a dummy input
    x = torch.randn(1, 3, 640, 640).to(device)
    outputs = model(x)
    
    # Print model structure and output shapes
    print(f"Model running on: {device}")
    for k, v in outputs.items():
        if k == 'outputs':
            for i, out in enumerate(v):
                print(f"Detection output {i} shape: {out.shape}")
        elif k == 'features' or k == 'fpn_features' or k == 'routed_features':
            if v is not None:
                for i, feat in enumerate(v):
                    print(f"{k} {i} shape: {feat.shape}")
        else:
            print(f"{k} shape: {v.shape}") 