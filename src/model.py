import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import List, Tuple, Dict, Optional, Union, Any


class ConvBlock(nn.Module):
    """
    Standard convolution block with BatchNorm and activation.
    Supports groups, bias, and different padding modes.
    """
    def __init__(
        self, 
        in_channels: int, 
        out_channels: int, 
        kernel_size: int = 3, 
        stride: int = 1, 
        padding: Optional[int] = None, 
        groups: int = 1, 
        dilation: int = 1, 
        bias: bool = False, 
        activation: str = 'silu'
    ):
        super().__init__()
        if padding is None:
            padding = kernel_size // 2
            
        self.conv = nn.Conv2d(
            in_channels, 
            out_channels, 
            kernel_size=kernel_size, 
            stride=stride, 
            padding=padding, 
            groups=groups, 
            dilation=dilation, 
            bias=bias
        )
        self.bn = nn.BatchNorm2d(out_channels)
        
        if activation == 'silu':
            self.act = nn.SiLU(inplace=True)
        elif activation == 'relu':
            self.act = nn.ReLU(inplace=True)
        elif activation == 'leaky_relu':
            self.act = nn.LeakyReLU(0.1, inplace=True)
        elif activation == 'none':
            self.act = nn.Identity()
        else:
            raise ValueError(f"Unsupported activation: {activation}")
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(self.bn(self.conv(x)))


class RepConv(nn.Module):
    """
    Reparameterizable Convolution block for inference optimization.
    During training, it uses a 3x3 conv and a 1x1 conv in parallel,
    but during inference, it fuses them into a single conv for speed.
    """
    def __init__(
        self, 
        in_channels: int, 
        out_channels: int, 
        kernel_size: int = 3, 
        stride: int = 1, 
        padding: int = 1, 
        groups: int = 1, 
        activation: str = 'silu'
    ):
        super().__init__()
        
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.groups = groups
        self.activation = activation
        
        # Standard 3x3 convolution
        self.conv1 = nn.Conv2d(
            in_channels, 
            out_channels, 
            kernel_size=kernel_size, 
            stride=stride, 
            padding=padding, 
            groups=groups, 
            bias=False
        )
        self.bn1 = nn.BatchNorm2d(out_channels)
        
        # Identity branch with 1x1 convolution if needed
        if in_channels == out_channels and stride == 1:
            self.conv2 = nn.Conv2d(
                in_channels, 
                out_channels, 
                kernel_size=1, 
                stride=1, 
                padding=0, 
                groups=groups, 
                bias=False
            )
            self.bn2 = nn.BatchNorm2d(out_channels)
        else:
            self.conv2 = None
            self.bn2 = None
            
        # Set activation
        if activation == 'silu':
            self.act = nn.SiLU(inplace=True)
        elif activation == 'relu':
            self.act = nn.ReLU(inplace=True)
        elif activation == 'leaky_relu':
            self.act = nn.LeakyReLU(0.1, inplace=True)
        else:
            raise ValueError(f"Unsupported activation: {activation}")
            
        # Flag to indicate if layers are fused
        self.is_fused = False
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.is_fused and hasattr(self, 'fused_conv'):
            # Use the fused convolution during inference
            return self.act(self.fused_conv(x))
        else:
            # During training, use parallel branches
            y1 = self.bn1(self.conv1(x))
            if self.conv2 is not None:
                y2 = self.bn2(self.conv2(x))
                y = y1 + y2
            else:
                y = y1
            return self.act(y)
    
    def fuse(self) -> None:
        """
        Fuse the parallel convolutions into a single conv for inference.
        """
        if self.is_fused:
            return
            
        # Create the fused convolution
        fused_conv = nn.Conv2d(
            self.in_channels, 
            self.out_channels, 
            kernel_size=self.kernel_size, 
            stride=self.stride, 
            padding=self.padding, 
            groups=self.groups, 
            bias=True
        )
        
        # Initialize with the weights from the first branch
        fused_conv.weight.data = self.conv1.weight.data.clone()
        
        # Get running statistics from batch norms
        scale_1 = self.bn1.weight.data / torch.sqrt(self.bn1.running_var + self.bn1.eps)
        bias_1 = self.bn1.bias.data - self.bn1.weight.data * self.bn1.running_mean / torch.sqrt(self.bn1.running_var + self.bn1.eps)
        
        # Prepare output bias
        fused_conv.bias.data = bias_1.clone()
        
        # Scale the weights of the 3x3 conv
        for i in range(self.out_channels):
            fused_conv.weight.data[i] = fused_conv.weight.data[i] * scale_1[i].reshape(-1, 1, 1)
        
        # Add the weights from the second branch if it exists
        if self.conv2 is not None:
            # Apply a 3x3 kernel with a 1x1 weight in the center
            central_idx = self.kernel_size // 2
            for i in range(self.out_channels):
                scale_2 = self.bn2.weight.data[i] / torch.sqrt(self.bn2.running_var[i] + self.bn2.eps)
                bias_2 = self.bn2.bias.data[i] - self.bn2.weight.data[i] * self.bn2.running_mean[i] / torch.sqrt(self.bn2.running_var[i] + self.bn2.eps)
                
                # Add the bias
                fused_conv.bias.data[i] += bias_2
                
                # Add the 1x1 weight to the center of the 3x3 kernel
                for j in range(self.in_channels):
                    fused_conv.weight.data[i, j % self.in_channels, central_idx, central_idx] += self.conv2.weight.data[i, j % self.in_channels, 0, 0] * scale_2
        
        # Store the fused conv
        self.fused_conv = fused_conv
        self.is_fused = True


class ELAN(nn.Module):
    """
    ELAN (Efficient Layer Aggregation Network) as used in YOLOv10.
    This design improves information flow by using multiple parallel paths with different 
    convolutional layers that are then aggregated.
    """
    def __init__(
        self, 
        in_channels: int, 
        out_channels: int, 
        depth: int = 1,
        expansion: float = 0.5,
        use_repconv: bool = True,
        activation: str = 'silu'
    ):
        super().__init__()
        
        hidden_channels = int(out_channels * expansion)
        self.depth = max(1, int(depth))
        
        # 1. Downsample if needed
        if in_channels != out_channels:
            self.downsample = ConvBlock(in_channels, out_channels, kernel_size=1, activation=activation)
        else:
            self.downsample = nn.Identity()
        
        # 2. Main branch
        self.conv1 = ConvBlock(out_channels, hidden_channels, kernel_size=1, activation=activation)
        
        # 3. Create parallel branches
        self.branches = nn.ModuleList()
        for i in range(self.depth):
            # Progressively use more channels in deeper layers
            branch_channels = hidden_channels * (i + 1)
            
            # Use RepConv in the last layer for efficiency
            if i == self.depth - 1 and use_repconv:
                self.branches.append(RepConv(branch_channels, hidden_channels, activation=activation))
            else:
                self.branches.append(ConvBlock(branch_channels, hidden_channels, kernel_size=3, activation=activation))
        
        # 4. Pointwise projection back to out_channels
        self.proj = ConvBlock(hidden_channels * (self.depth + 1), out_channels, kernel_size=1, activation=activation)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.downsample(x)
        
        # Initial 1x1 conv
        y0 = self.conv1(x)
        
        # Prepare feature list with the initial branch
        features = [y0]
        
        # Process each branch
        for i in range(self.depth):
            # Concatenate all previous features
            branch_input = torch.cat(features, dim=1)
            branch_output = self.branches[i](branch_input)
            features.append(branch_output)
        
        # Concatenate all features for final projection
        out = torch.cat(features, dim=1)
        out = self.proj(out)
        
        # Skip connection
        return out + x


class SPPFBlock(nn.Module):
    """
    Spatial Pyramid Pooling - Fast (SPPF) with different kernel sizes.
    Faster than SPP while maintaining the same receptive field.
    """
    def __init__(
        self, 
        in_channels: int, 
        out_channels: int, 
        kernel_size: int = 5, 
        activation: str = 'silu'
    ):
        super().__init__()
        
        hidden_channels = in_channels // 2
        
        self.cv1 = ConvBlock(in_channels, hidden_channels, kernel_size=1, activation=activation)
        self.cv2 = ConvBlock(hidden_channels * 4, out_channels, kernel_size=1, activation=activation)
        self.pool = nn.MaxPool2d(kernel_size=kernel_size, stride=1, padding=kernel_size // 2)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.cv1(x)
        y1 = self.pool(x)
        y2 = self.pool(y1)
        y3 = self.pool(y2)
        out = torch.cat([x, y1, y2, y3], dim=1)
        out = self.cv2(out)
        return out


class CSPLayer(nn.Module):
    """
    CSP (Cross Stage Partial) Layer with ELAN blocks.
    """
    def __init__(
        self, 
        in_channels: int, 
        out_channels: int, 
        n_blocks: int = 1, 
        expansion: float = 0.5,
        use_repconv: bool = True,
        activation: str = 'silu'
    ):
        super().__init__()
        
        hidden_channels = int(out_channels * expansion)
        
        # Input convolution
        self.conv1 = ConvBlock(in_channels, hidden_channels, kernel_size=1, activation=activation)
        self.conv2 = ConvBlock(in_channels, hidden_channels, kernel_size=1, activation=activation)
        
        # ELAN module
        self.elan = ELAN(
            hidden_channels, 
            hidden_channels, 
            depth=n_blocks,
            expansion=expansion,
            use_repconv=use_repconv,
            activation=activation
        )
        
        # Output convolution
        self.conv3 = ConvBlock(hidden_channels * 2, out_channels, kernel_size=1, activation=activation)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Split into two paths
        y1 = self.conv1(x)
        y2 = self.conv2(x)
        
        # Process one path through ELAN
        y2 = self.elan(y2)
        
        # Concatenate and process
        out = torch.cat([y1, y2], dim=1)
        out = self.conv3(out)
        
        return out


class DarknetCSP(nn.Module):
    """
    Improved CSPDarknet backbone for YOLOv10.
    This backbone features a faster and more efficient design compared to previous versions.
    """
    def __init__(
        self, 
        in_channels: int = 3, 
        base_channels: int = 64, 
        depth_multiple: float = 1.0, 
        width_multiple: float = 1.0,
        use_repconv: bool = True,
        activation: str = 'silu'
    ):
        super().__init__()
        
        # Apply width multiplier
        def make_divisible(x):
            return max(int(x * width_multiple), 1)
        
        # Apply depth multiplier
        def get_depth(x):
            return max(round(x * depth_multiple), 1)
        
        # Define channel sizes
        ch1 = make_divisible(base_channels)      # 64
        ch2 = make_divisible(base_channels * 2)  # 128
        ch3 = make_divisible(base_channels * 4)  # 256
        ch4 = make_divisible(base_channels * 8)  # 512
        ch5 = make_divisible(base_channels * 16) # 1024
        
        # Initial convolution
        self.stem = ConvBlock(in_channels, ch1, kernel_size=3, stride=2, activation=activation)
        
        # Stage 1
        self.dark1 = nn.Sequential(
            ConvBlock(ch1, ch2, kernel_size=3, stride=2, activation=activation),
            CSPLayer(ch2, ch2, n_blocks=get_depth(3), use_repconv=use_repconv, activation=activation)
        )
        
        # Stage 2
        self.dark2 = nn.Sequential(
            ConvBlock(ch2, ch3, kernel_size=3, stride=2, activation=activation),
            CSPLayer(ch3, ch3, n_blocks=get_depth(6), use_repconv=use_repconv, activation=activation)
        )
        
        # Stage 3
        self.dark3 = nn.Sequential(
            ConvBlock(ch3, ch4, kernel_size=3, stride=2, activation=activation),
            CSPLayer(ch4, ch4, n_blocks=get_depth(9), use_repconv=use_repconv, activation=activation)
        )
        
        # Stage 4
        self.dark4 = nn.Sequential(
            ConvBlock(ch4, ch5, kernel_size=3, stride=2, activation=activation),
            CSPLayer(ch5, ch5, n_blocks=get_depth(3), use_repconv=use_repconv, activation=activation),
            SPPFBlock(ch5, ch5, kernel_size=5, activation=activation)
        )
        
        # Save output channels for neck
        self.output_channels = [ch3, ch4, ch5]
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        x = self.stem(x)
        x = self.dark1(x)
        c3 = self.dark2(x)    # P3 features
        c4 = self.dark3(c3)   # P4 features
        c5 = self.dark4(c4)   # P5 features
        
        return c3, c4, c5
    
    def fuse(self) -> None:
        """
        Fuse RepConv modules for inference efficiency.
        """
        for m in self.modules():
            if isinstance(m, RepConv):
                m.fuse()


class PANNeck(nn.Module):
    """
    Path Aggregation Network (PAN) neck for YOLOv10.
    
    This neck takes features from the backbone at different scales and performs
    feature fusion through both top-down and bottom-up paths.
    """
    
    def __init__(
        self, 
        in_channels: List[int], 
        depth_multiple: float = 1.0,
        width_multiple: float = 1.0,
        use_repconv: bool = True,
        activation: str = 'silu'
    ):
        """
        Initialize the PANNeck module.
        
        Args:
            in_channels: List of channel sizes from the backbone (P3, P4, P5)
            depth_multiple: Depth multiplier for scaling model complexity
            width_multiple: Width multiplier for scaling model complexity
            use_repconv: Whether to use reparameterizable convolutions
            activation: Type of activation to use
        """
        super().__init__()
        
        def get_depth(x):
            """Get scaled depth based on depth_multiple."""
            return max(round(x * depth_multiple), 1) if x > 1 else x
        
        # Make channels divisible by 8
        def make_divisible(x):
            return int(math.ceil(x / 8) * 8)
        
        # Ensure in_channels are properly scaled
        c3, c4, c5 = in_channels
        
        # Store output channels
        self._out_channels = [c3, c4, c5]
        
        # Top-down path
        self.conv_p5 = ConvBlock(c5, c5//2, kernel_size=1, activation=activation)
        self.up1 = nn.Upsample(scale_factor=2, mode='nearest')
        
        # Ensure c4 + c5//2 is calculated correctly
        merged_c4_channels = c4 + c5//2
        
        self.csp1 = CSPLayer(
            merged_c4_channels, 
            c4, 
            n_blocks=get_depth(3), 
            expansion=0.5,
            use_repconv=use_repconv,
            activation=activation
        )
        
        self.conv_p4 = ConvBlock(c4, c4//2, kernel_size=1, activation=activation)
        self.up2 = nn.Upsample(scale_factor=2, mode='nearest')
        
        # Ensure c3 + c4//2 is calculated correctly
        merged_c3_channels = c3 + c4//2
        
        self.csp2 = CSPLayer(
            merged_c3_channels, 
            c3, 
            n_blocks=get_depth(3), 
            expansion=0.5,
            use_repconv=use_repconv,
            activation=activation
        )
        
        # Bottom-up path
        self.conv_p3 = ConvBlock(c3, c3, kernel_size=3, stride=2, activation=activation)
        
        # Calculate the correct input channels for pan_conv1
        merged_p3_p4_channels = c3 + c4
        
        self.pan_conv1 = CSPLayer(
            merged_p3_p4_channels, 
            c4, 
            n_blocks=get_depth(3), 
            expansion=0.5,
            use_repconv=use_repconv, 
            activation=activation
        )
        
        self.conv_p4_2 = ConvBlock(c4, c4, kernel_size=3, stride=2, activation=activation)
        
        # Calculate the correct input channels for pan_conv2
        merged_p4_p5_channels = c4 + c5
        
        self.pan_conv2 = CSPLayer(
            merged_p4_p5_channels, 
            c5, 
            n_blocks=get_depth(3), 
            expansion=0.5,
            use_repconv=use_repconv, 
            activation=activation
        )
        
    @property
    def out_channels(self) -> List[int]:
        """Return the output channels of the neck."""
        return self._out_channels
        
    def forward(self, inputs: Tuple[torch.Tensor, torch.Tensor, torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Forward pass of PAN neck.
        
        Args:
            inputs: Tuple of features (P3, P4, P5) from the backbone
            
        Returns:
            Tuple of fused features at different scales
        """
        p3, p4, p5 = inputs
        
        # Top-down path
        p5_ = self.conv_p5(p5)
        p5_up = self.up1(p5_)
        
        p4 = torch.cat([p5_up, p4], dim=1)
        p4 = self.csp1(p4)
        
        p4_ = self.conv_p4(p4)
        p4_up = self.up2(p4_)
        
        p3 = torch.cat([p4_up, p3], dim=1)
        p3 = self.csp2(p3)
        
        # Bottom-up path
        p3_down = self.conv_p3(p3)
        p4 = torch.cat([p3_down, p4], dim=1)
        p4 = self.pan_conv1(p4)
        
        p4_down = self.conv_p4_2(p4)
        p5 = torch.cat([p4_down, p5], dim=1)
        p5 = self.pan_conv2(p5)
        
        return (p3, p4, p5)
    
    def fuse(self) -> None:
        """Fuse Conv+BN layers for inference."""
        for module in self.modules():
            if isinstance(module, RepConv):
                module.fuse()


class YOLOHead(nn.Module):
    """
    Decoupled head for YOLOv10, separating classification and regression tasks.
    This design allows for better learning of each task while reducing inference costs.
    """
    def __init__(
        self, 
        in_channels: List[int], 
        num_classes: int, 
        num_anchors: int = 1,  # Anchor-free mode uses 1
        width_multiple: float = 1.0,
        activation: str = 'silu'
    ):
        super().__init__()
        
        self.in_channels = in_channels
        self.num_classes = num_classes
        self.num_anchors = num_anchors
        
        # Create detection heads for each scale level
        self.heads = nn.ModuleList()
        
        for channels in in_channels:
            # Create a decoupled head for each level
            head = nn.ModuleDict({
                # Classification branch
                'cls_conv': ConvBlock(channels, channels, kernel_size=3, activation=activation),
                'cls_pred': nn.Conv2d(channels, num_anchors * num_classes, kernel_size=1),
                
                # Regression branch (box + obj)
                'reg_conv': ConvBlock(channels, channels, kernel_size=3, activation=activation),
                'reg_pred': nn.Conv2d(channels, num_anchors * 4, kernel_size=1),
                'obj_pred': nn.Conv2d(channels, num_anchors * 1, kernel_size=1),
            })
            
            self.heads.append(head)
    
    def forward(self, features: List[torch.Tensor]) -> List[torch.Tensor]:
        """
        Forward pass of the YOLO head.
        
        Args:
            features: List of feature maps from the neck
            
        Returns:
            List of detection outputs [batch_size, grid_h*grid_w*n_anchors, n_classes+5]
        """
        outputs = []
        
        for i, (feature, head) in enumerate(zip(features, self.heads)):
            batch_size, _, grid_h, grid_w = feature.shape
            
            # Classification branch
            cls_feat = head['cls_conv'](feature)
            cls_pred = head['cls_pred'](cls_feat)
            
            # Regression branch
            reg_feat = head['reg_conv'](feature)
            box_pred = head['reg_pred'](reg_feat)
            obj_pred = head['obj_pred'](reg_feat)
            
            # Reshape for output
            cls_pred = cls_pred.permute(0, 2, 3, 1).reshape(batch_size, grid_h * grid_w * self.num_anchors, self.num_classes)
            box_pred = box_pred.permute(0, 2, 3, 1).reshape(batch_size, grid_h * grid_w * self.num_anchors, 4)
            obj_pred = obj_pred.permute(0, 2, 3, 1).reshape(batch_size, grid_h * grid_w * self.num_anchors, 1)
            
            # Create grid for anchor-free detection
            grid_x = torch.arange(grid_w, device=feature.device).repeat(grid_h, 1)
            grid_y = torch.arange(grid_h, device=feature.device).repeat(grid_w, 1).t()
            
            grid_xy = torch.stack([grid_x, grid_y], dim=-1).reshape(-1, 2)
            grid_xy = grid_xy.unsqueeze(0).repeat(batch_size, self.num_anchors, 1).reshape(batch_size, -1, 2)
            
            # Apply sigmoid to predictions and adjust box positions
            box_pred[..., :2] = (torch.sigmoid(box_pred[..., :2]) + grid_xy) / torch.tensor([grid_w, grid_h], device=feature.device)
            box_pred[..., 2:4] = torch.exp(box_pred[..., 2:4]) * (2 ** i)
            
            obj_pred = torch.sigmoid(obj_pred)
            cls_pred = torch.sigmoid(cls_pred)
            
            # Combine predictions
            out = torch.cat([box_pred, obj_pred, cls_pred], dim=-1)
            outputs.append(out)
        
        return outputs


class DynamicCompactDetect(nn.Module):
    """
    DynamicCompactDetect Model (based on YOLOv10 architecture)
    
    Args:
        num_classes: Number of classes for detection
        in_channels: Number of input channels
        base_channels: Base channel count for the model
        width_multiple: Factor to scale the channel count
        depth_multiple: Factor to scale the depth of the model
        backbone_type: Type of backbone ('csp' or other backbone types)
        head_type: Type of head ('decoupled' or other head types)
        use_repconv: Whether to use RepConv layer
        activation: Activation function to use ('silu', 'relu', 'leaky', etc.)
    """
    def __init__(
        self, 
        num_classes: int = 80, 
        in_channels: int = 3,
        base_channels: int = 64,
        width_multiple: float = 1.0,
        depth_multiple: float = 1.0,
        backbone_type: str = 'csp',
        head_type: str = 'decoupled',
        use_repconv: bool = True,
        activation: str = 'silu'
    ):
        super().__init__()
        self.num_classes = num_classes
        
        # Create backbone
        if backbone_type == 'csp':
            self.backbone = DarknetCSP(
                in_channels=in_channels, 
                base_channels=base_channels, 
                depth_multiple=depth_multiple, 
                width_multiple=width_multiple,
                use_repconv=use_repconv,
                activation=activation
            )
        else:
            raise ValueError(f"Unsupported backbone type: {backbone_type}")
        
        # Create neck
        self.neck = PANNeck(
            in_channels=self.backbone.output_channels, 
            depth_multiple=depth_multiple,
            width_multiple=width_multiple,
            use_repconv=use_repconv,
            activation=activation
        )
        
        # Create head
        if head_type == 'decoupled':
            self.head = YOLOHead(
                in_channels=self.neck.out_channels, 
                num_classes=num_classes,
                width_multiple=width_multiple,
                activation=activation
            )
        else:
            raise ValueError(f"Unsupported head type: {head_type}")
        
        # Initialize weights
        self.apply(_init_weights)
    
    def forward(self, x: torch.Tensor, augment: bool = False) -> Union[List[torch.Tensor], torch.Tensor]:
        """
        Forward pass
        
        Args:
            x: Input tensor of shape [batch_size, channels, height, width]
            augment: Whether to use test-time augmentation
            
        Returns:
            List of outputs from each detection head, or
            combined output for inference
        """
        if augment:
            return self._forward_augment(x)
        
        # Backbone
        features = self.backbone(x)
        
        # Neck
        features = self.neck(features)
        
        # Head (returns raw outputs for training or NMS for inference)
        outputs = self.head(features)
            
        return outputs
    
    def _forward_augment(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass with test-time augmentation
        
        Args:
            x: Input tensor of shape [batch_size, channels, height, width]
            
        Returns:
            Augmented predictions
        """
        # Define image sizes
        img_size = x.shape[-2:]
        s = [1, 0.83, 0.67]  # Scales
        
        # Run inference on multiple scales
        y = []
        for i, scale in enumerate(s):
            # Scale image down for multi-scale inference
            img = F.interpolate(x, size=[int(x * scale) for x in img_size], mode='bilinear', align_corners=False)
            
            # Forward pass
            y.append(self.forward(img))
            
        # Same image flipped
        x_flip = torch.flip(x, [3])  # Flip batch axis 3 (width)
        y.append(torch.flip(self.forward(x_flip), [3]))
        
        # Merge scales and flip
        return torch.cat(y, 1)
    
    def fuse(self) -> None:
        """
        Fuse Conv2d and BatchNorm2d layers for inference
        """
        for m in self.modules():
            if hasattr(m, 'fuse') and m.__class__.__name__ != 'DynamicCompactDetect':
                m.fuse()
        return self


def create_dynamiccompact_model(
    num_classes: int = 80,
    width_multiple: float = 0.25,
    depth_multiple: float = 0.33,
    backbone_type: str = 'csp',
    head_type: str = 'decoupled',
    activation: str = 'silu',
    use_repconv: bool = True
) -> nn.Module:
    """
    Create DynamicCompactDetect model
    
    Args:
        num_classes: Number of classes for detection
        width_multiple: Factor to scale the channel count, model width
        depth_multiple: Factor to scale the depth of the model
        backbone_type: Type of backbone ('csp' or other backbone types)
        head_type: Type of head ('decoupled' or other head types)
        activation: Activation function to use ('silu', 'relu', 'leaky', etc.)
        use_repconv: Whether to use RepConv layer
        
    Returns:
        Initialized model
    """
    # Validate inputs
    valid_backbone_types = ['csp']
    valid_head_types = ['decoupled']
    valid_activations = ['silu', 'relu', 'leaky']
    
    if backbone_type not in valid_backbone_types:
        raise ValueError(f"Invalid backbone_type: {backbone_type}. Must be one of {valid_backbone_types}")
    if head_type not in valid_head_types:
        raise ValueError(f"Invalid head_type: {head_type}. Must be one of {valid_head_types}")
    if activation not in valid_activations:
        raise ValueError(f"Invalid activation: {activation}. Must be one of {valid_activations}")
    
    # Create model based on size
    if width_multiple <= 0.25:
        base_channels = 64
    elif width_multiple <= 0.5:
        base_channels = 64
    elif width_multiple <= 0.75:
        base_channels = 64
    else:
        base_channels = 64
    
    # Create model
    model = DynamicCompactDetect(
        num_classes=num_classes,
        base_channels=base_channels,
        width_multiple=width_multiple,
        depth_multiple=depth_multiple,
        backbone_type=backbone_type,
        head_type=head_type,
        activation=activation,
        use_repconv=use_repconv
    )
    
    return model


def _init_weights(m: nn.Module) -> None:
    """
    Initialize model weights.
    
    Args:
        m: Module to initialize
    """
    if isinstance(m, nn.Conv2d):
        # Use kaiming normal for conv layers
        nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
        if m.bias is not None:
            nn.init.constant_(m.bias, 0)
    elif isinstance(m, nn.BatchNorm2d):
        # BatchNorm layers
        nn.init.constant_(m.weight, 1)
        nn.init.constant_(m.bias, 0)
    elif isinstance(m, nn.Linear):
        # Linear layers
        nn.init.normal_(m.weight, 0, 0.01)
        nn.init.constant_(m.bias, 0) 