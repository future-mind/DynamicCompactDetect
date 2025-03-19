import math
import torch
import torch.nn as nn
import os
import copy
import torch.nn.functional as F
from models.dynamic_compact_detect import DynamicCompactDetect

def init_model(model):
    """Initialize model weights."""
    for m in model.modules():
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
    
    return model

def save_checkpoint(model, optimizer, scheduler, epoch, best_map=0.0, 
                   path='checkpoint.pth', scaler=None, ema=None):
    """Save model checkpoint."""
    state = {
        'epoch': epoch,
        'best_mAP': best_map,
        'model': model.state_dict(),
        'optimizer': optimizer.state_dict(),
        'scheduler': scheduler.state_dict() if scheduler else None
    }
    
    # Add scaler state (for mixed precision training)
    if scaler is not None:
        state['scaler'] = scaler.state_dict()
    
    # Add EMA model state
    if ema is not None:
        state['ema'] = ema.ema.state_dict()
    
    # Save the checkpoint
    torch.save(state, path)
    print(f"Checkpoint saved to {path}")

def load_checkpoint(path, model, optimizer=None, scheduler=None, scaler=None, ema=None):
    """Load model checkpoint."""
    if not os.path.exists(path):
        print(f"Checkpoint not found at {path}")
        return None
    
    print(f"Loading checkpoint from {path}")
    checkpoint = torch.load(path, map_location='cpu')
    
    # Load model state
    model.load_state_dict(checkpoint['model'])
    
    # Load optimizer state if provided
    if optimizer is not None and 'optimizer' in checkpoint:
        optimizer.load_state_dict(checkpoint['optimizer'])
    
    # Load scheduler state if provided
    if scheduler is not None and 'scheduler' in checkpoint and checkpoint['scheduler'] is not None:
        scheduler.load_state_dict(checkpoint['scheduler'])
    
    # Load scaler state if provided (for mixed precision training)
    if scaler is not None and 'scaler' in checkpoint:
        scaler.load_state_dict(checkpoint['scaler'])
    
    # Load EMA model state if provided
    if ema is not None and 'ema' in checkpoint:
        ema.ema.load_state_dict(checkpoint['ema'])
    
    print(f"Loaded checkpoint from epoch {checkpoint['epoch']}")
    return checkpoint

def count_parameters(model):
    """Count total trainable parameters in the model."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

class ModelEMA:
    """Model Exponential Moving Average."""
    def __init__(self, model, decay=0.9999, updates=0):
        # Create EMA model
        self.ema = copy.deepcopy(model)
        self.ema.eval()
        self.decay = decay
        self.updates = updates
        
        # Set requires_grad=False for all params
        for param in self.ema.parameters():
            param.requires_grad_(False)
    
    def update(self, model):
        """Update EMA parameters."""
        with torch.no_grad():
            self.updates += 1
            d = self.decay * (1 - math.exp(-self.updates / 2000))  # Decay increases with updates
            
            # Update EMA parameters
            for ema_param, model_param in zip(self.ema.parameters(), model.parameters()):
                ema_param.data.mul_(d).add_(model_param.data, alpha=1 - d)

def export_onnx(model, onnx_path, input_size=(640, 640), batch_size=1, opset_version=11):
    """Export the model to ONNX format."""
    # Create dummy input
    dummy_input = torch.randn(batch_size, 3, input_size[1], input_size[0])
    
    # Ensure model is in evaluation mode
    model.eval()
    
    # Set device
    device = next(model.parameters()).device
    dummy_input = dummy_input.to(device)
    
    try:
        # Export model to ONNX
        torch.onnx.export(
            model,
            dummy_input,
            onnx_path,
            export_params=True,
            opset_version=opset_version,
            do_constant_folding=True,
            input_names=['input'],
            output_names=['output'],
            dynamic_axes={
                'input': {0: 'batch_size'},
                'output': {0: 'batch_size'}
            }
        )
        
        print(f"ONNX model exported to {onnx_path}")
        
        # Verify ONNX model (if onnx package is available)
        try:
            import onnx
            onnx_model = onnx.load(onnx_path)
            onnx.checker.check_model(onnx_model)
            print("ONNX model verified successfully")
        except ImportError:
            print("ONNX package not available, skipping verification")
        except Exception as e:
            print(f"ONNX verification failed: {e}")
            
        return True
    
    except Exception as e:
        print(f"ONNX export failed: {e}")
        return False

def create_model_with_config(config, checkpoint_path=None):
    """Create a model instance from configuration."""
    # Create model
    model = DynamicCompactDetect(
        num_classes=config['model']['num_classes'],
        base_channels=config['model']['base_channels']
    )
    
    # Initialize weights
    model = init_model(model)
    
    # Load checkpoint if provided
    if checkpoint_path and os.path.exists(checkpoint_path):
        load_checkpoint(checkpoint_path, model)
    
    # Apply hardware optimizations if configured
    if config['hardware']['use_platform_optimizations']:
        model, device = model.optimize_for_platform()
    
    return model

def freeze_backbone(model, freeze=True):
    """Freeze or unfreeze the backbone for fine-tuning."""
    for name, param in model.named_parameters():
        if 'backbone' in name:
            param.requires_grad_(not freeze)
    
    return model

def freeze_selective_blocks(model, dynamic_blocks=False):
    """Freeze or unfreeze specific blocks for selective fine-tuning."""
    # Example: freeze all but the dynamic blocks or routing components
    for name, param in model.named_parameters():
        if dynamic_blocks:
            if ('dynamic' in name or 'routing' in name):
                param.requires_grad_(True)
            else:
                param.requires_grad_(False)
        else:
            param.requires_grad_(True)
    
    return model

def get_layer_by_name(model, name):
    """Get a specific layer from the model by name."""
    for n, m in model.named_modules():
        if n == name:
            return m
    return None

def layer_wise_learning_rate(model, lr=1e-3, backbone_lr_factor=0.1):
    """Create parameter groups with different learning rates."""
    # Create parameter groups for optimizer
    # Backbone gets lower learning rate, detection heads get higher
    
    backbone_params = []
    head_params = []
    routing_params = []
    
    for name, param in model.named_parameters():
        if param.requires_grad:
            if 'backbone' in name:
                backbone_params.append(param)
            elif 'routing' in name:
                routing_params.append(param)
            else:
                head_params.append(param)
    
    param_groups = [
        {'params': backbone_params, 'lr': lr * backbone_lr_factor},
        {'params': routing_params, 'lr': lr * 0.5},
        {'params': head_params, 'lr': lr}
    ]
    
    return param_groups 