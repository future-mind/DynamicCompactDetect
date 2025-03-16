import torch
import torch.nn as nn
import yaml
import math
from pathlib import Path
from copy import deepcopy

# Import common layers
from .common import *
from .attention import DynamicSparseAttention, LightweightTransformer
from .detect import DynamicDetect


class DynamicCompactDetect(nn.Module):
    """
    DynamicCompact-Detect: Efficient Object Detection for Resource-Constrained Environments
    
    This model combines the efficiency of YOLOv10 with the global reasoning capabilities
    of transformers from RT-DETR, optimized for resource-constrained environments.
    """
    def __init__(self, cfg='configs/dynamiccompact_s.yaml', ch=3, nc=None, anchors=None):
        super().__init__()
        self.yaml = cfg
        
        # Load model configuration
        if isinstance(cfg, dict):
            self.yaml_dict = cfg  # model dict
        else:
            # Load cfg from yaml file
            import yaml  # for torch hub
            self.yaml_dict = self._load_yaml(cfg)
        
        # Define model
        ch = self.yaml_dict.get('ch', ch)  # input channels
        if nc and nc != self.yaml_dict['nc']:
            print(f'Overriding model.yaml nc={self.yaml_dict["nc"]} with nc={nc}')
            self.yaml_dict['nc'] = nc  # override yaml value
        
        # Parse model architecture from yaml
        self.model, self.save = self._parse_model(deepcopy(self.yaml_dict), ch=[ch])
        self.names = [f'class{i}' for i in range(self.yaml_dict['nc'])]  # default names
        self.inplace = self.yaml_dict.get('inplace', True)
        
        # Initialize weights
        self._initialize_weights()
        
        # Print model summary
        self._print_model_info()
    
    def forward(self, x):
        """
        Forward pass of the model.
        
        Args:
            x: Input tensor with shape [batch_size, channels, height, width]
            
        Returns:
            Tensor: Detection predictions
        """
        y, dt = [], []  # outputs
        for m in self.model:
            if m.f != -1:  # if not from previous layer
                x = y[m.f] if isinstance(m.f, int) else [x if j == -1 else y[j] for j in m.f]  # from earlier layers
            
            x = m(x)  # run
            y.append(x if m.i in self.save else None)  # save output
        
        return x
    
    def _load_yaml(self, cfg):
        """Load and parse YAML file."""
        with open(cfg, encoding='ascii', errors='ignore') as f:
            return yaml.safe_load(f)  # model dict
    
    def _parse_model(self, d, ch):
        """Parse model architecture from YAML dictionary."""
        print(f"\n{'':>3}{'from':>18}{'n':>3}{'params':>10}  {'module':<40}{'arguments':<30}")
        anchors, nc, gd, gw = d['anchors'], d['nc'], d.get('depth_multiple', 1.0), d.get('width_multiple', 1.0)
        
        # Scale channels and layers based on model size
        if 'scales' in d:
            if isinstance(d['scales'], dict):
                # Extract model size from configuration file path
                model_size = self.yaml.split('_')[-1].split('.')[0] if isinstance(self.yaml, str) else 's'  # Default to 's' if no size
                for size in d['scales']:
                    if size == model_size:  # Use model_size instead of cfg
                        gd, gw, max_channels = d['scales'][size]
                        print(f'Using model scale {size}: depth_multiple={gd}, width_multiple={gw}, max_channels={max_channels}')
                        break
        
        na = (len(anchors[0]) // 2) if isinstance(anchors, list) else anchors  # number of anchors
        no = na * (nc + 5)  # number of outputs = anchors * (classes + 5)
        
        layers, save, c2 = [], [], ch[-1]  # layers, savelist, ch out
        for i, (f, n, m, args) in enumerate(d['backbone'] + d.get('neck', []) + d['head']):  # from, number, module, args
            m = eval(m) if isinstance(m, str) else m  # eval strings
            for j, a in enumerate(args):
                try:
                    args[j] = eval(a) if isinstance(a, str) else a  # eval strings
                except NameError:
                    pass
            
            n = max(round(n * gd), 1) if n > 1 else n  # depth gain
            if m in [nn.Conv2d, nn.ConvTranspose2d, Conv, DynamicSparseAttention, LightweightTransformer]:
                c1, c2 = ch[f], args[0]
                if c2 != no:  # if not output
                    c2 = min(max_channels if 'max_channels' in locals() else float('inf'), math.ceil(c2 * gw))
                
                args = [c1, c2, *args[1:]]
                if m in [DynamicSparseAttention, LightweightTransformer]:
                    args = [c2, *args[1:]]
            elif m is DynamicDetect:
                args = [nc, [ch[x] for x in f]]
            
            module = nn.Sequential(*(m(*args) for _ in range(n))) if n > 1 else m(*args)
            t = str(m)[8:-2].replace('__main__.', '')  # module type
            np = sum(x.numel() for x in module.parameters())  # number params
            module.i, module.f, module.type, module.np = i, f, t, np  # attach index, 'from' index, type, number params
            print(f'{i:>3}{str(f):>18}{n:>3}{np:10.0f}  {t:<40}{str(args):<30}')  # print
            save.extend(x % i for x in ([f] if isinstance(f, int) else f) if x != -1)  # append to savelist
            layers.append(module)
            if i == 0:
                ch = []
            ch.append(c2)
        return nn.Sequential(*layers), sorted(save)
    
    def _initialize_weights(self):
        """Initialize model weights."""
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
                nn.init.zeros_(m.bias)
        
        # Initialize detection head biases for stable training
        for m in self.model:
            if isinstance(m, DynamicDetect):
                m.initialize_biases()
    
    def _print_model_info(self):
        """Print model information."""
        n_p = sum(x.numel() for x in self.parameters())  # number parameters
        n_g = sum(x.numel() for x in self.parameters() if x.requires_grad)  # number gradients
        print(f"Model Summary: {len(list(self.modules()))} layers, {n_p:,} parameters, {n_g:,} gradients")


def load_model(weights, device=None):
    """
    Load DynamicCompact-Detect model from weights file.
    
    Args:
        weights: Path to weights file (.pt)
        device: torch.device or str (optional)
        
    Returns:
        model: Loaded model
    """
    device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
    ckpt = torch.load(weights, map_location=device)  # load checkpoint
    
    # Load model
    model = DynamicCompactDetect(cfg=ckpt.get('cfg') or ckpt['model'].yaml)
    
    # Load weights
    state_dict = ckpt['model'].float().state_dict()
    model.load_state_dict(state_dict, strict=False)
    
    return model.to(device) 