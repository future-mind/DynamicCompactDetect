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

def save_checkpoint(model, optimizer, scheduler, epoch, metrics=None, best_map=0.0, history=None, path=None, scaler=None, ema=None):
    """
    Save model checkpoint to file.
    
    Args:
        model: Model to save
        optimizer: Optimizer state to save
        scheduler: Scheduler state to save
        epoch: Current epoch
        metrics: Dictionary of current metrics (optional)
        best_map: Best mAP value achieved (optional)
        history: Training history (optional)
        path: Path to save the checkpoint (required)
        scaler: AMP Scaler if using mixed precision (optional)
        ema: EMA model if using (optional)
    """
    # Handle old calling convention
    if isinstance(metrics, float) and path is None:
        # Old format: model, optimizer, scheduler, epoch, best_map, path, scaler, ema
        best_map = metrics  # metrics is actually best_map
        path = history      # history is actually path
        scaler = best_map   # best_map is actually scaler
        ema = metrics       # metrics is actually ema
        metrics = None
        history = None
    
    # Make sure we have a valid path
    if path is None:
        raise ValueError("Path must be provided to save checkpoint")
    
    # Create a simplified metrics dict if not provided
    if metrics is None:
        metrics = {'best_map': best_map}
    
    # Create minimal history if not provided
    if history is None:
        history = {}
    
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict() if scheduler else None,
        'metrics': metrics,
        'best_map': best_map,
        'history': history
    }
    
    # Add scaler if provided
    if scaler is not None:
        checkpoint['scaler'] = scaler.state_dict()
    
    # Add EMA if provided
    if ema is not None:
        checkpoint['ema'] = ema.ema.state_dict()
    
    torch.save(checkpoint, path)
    print(f"Checkpoint saved to {path}")

def load_checkpoint(checkpoint_path, model, optimizer=None, scheduler=None, map_location=None):
    """
    Load model checkpoint from file.
    
    Args:
        checkpoint_path: Path to the checkpoint file
        model: Model to load the checkpoint into
        optimizer: Optional optimizer to load state
        scheduler: Optional scheduler to load state
        map_location: Optional device mapping for loading the checkpoint
        
    Returns:
        Dictionary with checkpoint data
    """
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint file not found: {checkpoint_path}")
    
    print(f"Loading checkpoint from {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=map_location)
    
    # Check if it's just the model state dict or a complete checkpoint
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
        
        if optimizer is not None and 'optimizer_state_dict' in checkpoint:
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        if scheduler is not None and 'scheduler_state_dict' in checkpoint:
            scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        
        return checkpoint
    else:
        # Assume it's just the model state dict
        model.load_state_dict(checkpoint)
        return {'model_state_dict': checkpoint}

def count_parameters(model):
    """
    Count the number of trainable and non-trainable parameters in a model.
    
    Args:
        model: PyTorch model
        
    Returns:
        Tuple of (total_params, trainable_params)
    """
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    return total_params, trainable_params

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

def export_onnx(model, input_size=(640, 640), path="model.onnx", dynamic=True, simplify=True):
    """
    Export PyTorch model to ONNX format.
    
    Args:
        model: PyTorch model
        input_size: Input size as (width, height)
        path: Output path for ONNX model
        dynamic: Whether to use dynamic axes
        simplify: Whether to simplify the ONNX model
    """
    import torch.onnx
    
    model.eval()
    
    # Create dummy input
    width, height = input_size
    dummy_input = torch.randn(1, 3, height, width)
    
    # Input and output names
    input_names = ["input"]
    output_names = ["output"]
    
    # Dynamic axes for variable batch size and input dimensions
    dynamic_axes = {
        "input": {0: "batch_size", 2: "height", 3: "width"},
        "output": {0: "batch_size"}
    } if dynamic else None
    
    # Export the model
    torch.onnx.export(
        model,
        dummy_input,
        path,
        verbose=False,
        input_names=input_names,
        output_names=output_names,
        dynamic_axes=dynamic_axes,
        opset_version=12,
    )
    
    print(f"Model exported to {path}")
    
    # Simplify if requested
    if simplify:
        try:
            import onnx
            import onnxsim
            
            # Load and simplify the model
            onnx_model = onnx.load(path)
            simplified_model, check = onnxsim.simplify(onnx_model)
            
            if check:
                onnx.save(simplified_model, path)
                print(f"Simplified ONNX model saved to {path}")
            else:
                print("Failed to simplify ONNX model")
        except ImportError:
            print("onnx-simplifier not installed. Run 'pip install onnxsim' to simplify models.")
        except Exception as e:
            print(f"Error simplifying model: {e}")

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

def freeze_backbone(model, unfreeze_last_n_layers=0):
    """
    Freeze backbone layers of the model.
    
    Args:
        model: PyTorch model
        unfreeze_last_n_layers: Number of last layers to keep unfrozen
    """
    if not hasattr(model, 'backbone'):
        return
    
    # Get all backbone modules
    backbone_modules = list(model.backbone.modules())
    
    # Determine which modules to freeze
    freeze_modules = backbone_modules[:-unfreeze_last_n_layers] if unfreeze_last_n_layers > 0 else backbone_modules
    
    # Freeze parameters
    for module in freeze_modules:
        for param in module.parameters():
            param.requires_grad = False
    
    # Print freezing information
    total_params = sum(p.numel() for p in model.parameters())
    frozen_params = sum(p.numel() for p in model.parameters() if not p.requires_grad)
    print(f"Frozen {frozen_params:,} parameters out of {total_params:,} total parameters")
    print(f"({frozen_params / total_params:.2%} of parameters are frozen)")

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

def get_param_groups(model, weight_decay=0.0005):
    """
    Get parameter groups for optimizer with weight decay applied only to weights.
    This is a common practice in vision models to avoid applying weight decay to biases.
    
    Args:
        model: PyTorch model
        weight_decay: Weight decay factor
        
    Returns:
        List of parameter group dictionaries
    """
    # Separate weights from biases and normalization layers
    decay = []
    no_decay = []
    
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
            
        if len(param.shape) == 1 or name.endswith(".bias") or \
           'bn' in name or 'norm' in name:
            no_decay.append(param)
        else:
            decay.append(param)
    
    return [
        {'params': no_decay, 'weight_decay': 0.0},
        {'params': decay, 'weight_decay': weight_decay}
    ]

def get_activation_function(name):
    """
    Get activation function by name.
    
    Args:
        name: Name of activation function
        
    Returns:
        PyTorch activation function
    """
    activations = {
        'relu': nn.ReLU,
        'leaky_relu': nn.LeakyReLU,
        'mish': nn.Mish,
        'swish': lambda: nn.SiLU(),
        'silu': nn.SiLU,
        'gelu': nn.GELU,
        'sigmoid': nn.Sigmoid,
        'tanh': nn.Tanh,
        'identity': nn.Identity
    }
    
    if name.lower() not in activations:
        raise ValueError(f"Activation function '{name}' not supported. Available options: {list(activations.keys())}")
    
    return activations[name.lower()]()

def get_optimizer(name, params, lr, weight_decay=0):
    """
    Get optimizer by name.
    
    Args:
        name: Name of optimizer
        params: Model parameters
        lr: Learning rate
        weight_decay: Weight decay factor
        
    Returns:
        PyTorch optimizer
    """
    name = name.lower()
    
    if name == 'sgd':
        return torch.optim.SGD(params, lr=lr, momentum=0.9, weight_decay=weight_decay)
    elif name == 'adam':
        return torch.optim.Adam(params, lr=lr, weight_decay=weight_decay)
    elif name == 'adamw':
        return torch.optim.AdamW(params, lr=lr, weight_decay=weight_decay)
    else:
        raise ValueError(f"Optimizer '{name}' not supported")

def get_lr_scheduler(name, optimizer, epochs, warmup_epochs=0):
    """
    Get learning rate scheduler by name.
    
    Args:
        name: Name of scheduler
        optimizer: PyTorch optimizer
        epochs: Total number of epochs
        warmup_epochs: Number of warmup epochs
        
    Returns:
        PyTorch learning rate scheduler
    """
    name = name.lower()
    
    if name == 'step':
        # Step LR with step size of 30 epochs
        return torch.optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.1)
    elif name == 'multistep':
        # MultiStep LR with milestones at 50% and 75% of training
        milestones = [epochs // 2, int(epochs * 0.75)]
        return torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=milestones, gamma=0.1)
    elif name == 'cosine':
        # Cosine annealing scheduler
        return torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs - warmup_epochs)
    elif name == 'linear':
        # Linear LR scheduler (custom implementation)
        lambda_fn = lambda epoch: 1.0 - (epoch - warmup_epochs) / (epochs - warmup_epochs) if epoch >= warmup_epochs else 1.0
        return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda_fn)
    elif name == 'polynomial':
        # Polynomial LR scheduler (custom implementation)
        power = 0.9
        lambda_fn = lambda epoch: (1.0 - (epoch - warmup_epochs) / (epochs - warmup_epochs)) ** power if epoch >= warmup_epochs else 1.0
        return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda_fn)
    else:
        raise ValueError(f"Scheduler '{name}' not supported") 