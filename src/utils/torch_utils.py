"""
PyTorch utilities for YOLOv11 training and inference
"""

import torch
import torch.nn as nn
import math
import random
import numpy as np
import os
import logging
import time
from copy import deepcopy
import platform
import torch.nn.functional as F
import torchvision
import re
import warnings
from pathlib import Path


def select_device(device='', batch_size=None):
    """
    Select the appropriate device based on availability
    
    Args:
        device: Device to use (cuda device, i.e. 0 or 0,1,2,3 or cpu or mps)
        batch_size: Batch size to validate against
        
    Returns:
        Device for PyTorch operations
    """
    # Get logging info
    logger = logging.getLogger(__name__)
    
    # Ensure device is string
    device = str(device).strip().lower().replace('cuda:', '')
    
    # Determine device type
    cpu = device == 'cpu'
    mps = device == 'mps'  # Apple Metal Performance Shaders (MPS)
    cuda = not cpu and not mps and torch.cuda.is_available()
    
    if cuda:
        # Check CUDA device specs
        if device:  # i.e. cuda:0 or cuda:0,1,2,3
            devices = device.split(',') if ',' in device else [device]
            devices = [int(d) for d in devices if d.isnumeric()]  # i.e. ['0', '1', ... ] -> [0, 1, ...]
            device = ','.join(str(d) for d in devices)
            if len(devices) > 1 and batch_size:  # Batch size should be divisible by device count
                assert batch_size % len(devices) == 0, f'batch-size {batch_size} not multiple of GPU count {len(devices)}'
        torch_device = torch.device(f'cuda:{device}' if device else 'cuda:0')
        device_info = f"PyTorch {torch.__version__} CUDA:{torch.version.cuda} "
        for d in devices if device else range(torch.cuda.device_count()):
            p = torch.cuda.get_device_properties(d)
            device_info += f"({p.name}, {p.total_memory / 1024 ** 2:.0f}MiB) "
        logger.info(device_info)
    elif mps and getattr(torch, 'has_mps', False) and torch.backends.mps.is_available():
        # Check MPS availability
        torch_device = torch.device('mps')
        logger.info(f"PyTorch {torch.__version__} MPS (Apple Metal)")
    else:
        # CPU
        torch_device = torch.device('cpu')
        logger.info(f"PyTorch {torch.__version__} CPU ({platform.processor()})")
    
    return torch_device


class ModelEMA:
    """
    Model Exponential Moving Average
    Keep a moving average of everything in the model state_dict (parameters and buffers)
    """
    def __init__(self, model, decay=0.9999, updates=0):
        """
        Initialize EMA
        
        Args:
            model: Model to apply EMA to
            decay: EMA decay factor (higher means slower updates)
            updates: Current update counter
        """
        # Create EMA
        self.ema = model.module if hasattr(model, 'module') else model  # FP32 EMA
        self.updates = updates  # Number of EMA updates
        self.decay = lambda x: decay * (1 - math.exp(-x / 2000))  # Decay exponential ramp (to help early epochs)
        
        # Create a copy of the model state
        self.ema_state_dict = self.ema.state_dict()
        
        # Copy state to EMA
        for k, v in self.ema.state_dict().items():
            if isinstance(v, torch.Tensor):
                self.ema_state_dict[k] = v.clone().detach()
                
    def update(self, model):
        """
        Update EMA parameters
        
        Args:
            model: Model to update EMA from
        """
        # Update EMA parameters
        self.updates += 1
        d = self.decay(self.updates)
        
        # Get model state dict
        msd = model.module.state_dict() if hasattr(model, 'module') else model.state_dict()
        
        # Update EMA parameters
        for k, v in self.ema_state_dict.items():
            if isinstance(v, torch.Tensor):
                v.copy_(d * v + (1 - d) * msd[k].detach())


def initialize_weights(model):
    """
    Initialize model weights to random values
    
    Args:
        model: Model to initialize
    """
    for m in model.modules():
        if isinstance(m, nn.Conv2d):
            n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            m.weight.data.normal_(0, math.sqrt(2. / n))
            if m.bias is not None:
                m.bias.data.zero_()
        elif isinstance(m, nn.BatchNorm2d):
            m.weight.data.fill_(1)
            m.bias.data.zero_()
        elif isinstance(m, nn.Linear):
            m.weight.data.normal_(0, 0.01)
            m.bias.data.zero_()


def time_synchronized():
    """
    Get accurate time for PyTorch operations
    Returns synchronized time in seconds
    """
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    return time.time()


def model_info(model, verbose=False, img_size=640):
    """
    Print model information
    
    Args:
        model: Model
        verbose: Print layer by layer information
        img_size: Input image size
        
    Returns:
        Tuple of (num params, num gradients)
    """
    # Get number of parameters and gradients
    n_p = sum(x.numel() for x in model.parameters())  # number parameters
    n_g = sum(x.numel() for x in model.parameters() if x.requires_grad)  # number gradients
    
    # Print model information
    if verbose:
        print(f'Model Summary: {len(list(model.modules()))} layers, {n_p} parameters, {n_g} gradients')
        
    # Calculate model FLOPs
    try:
        from thop import profile
        stride = max(int(model.stride.max()), 32) if hasattr(model, 'stride') else 32
        img = torch.zeros((1, 3, stride, stride), device=next(model.parameters()).device)  # input
        flops = profile(model, inputs=(img,), verbose=False)[0] / 1E9 * 2  # stride GFLOPs
        img_size = img_size if isinstance(img_size, list) else [img_size, img_size]  # expand if int/float
        fs = ', %.1f GFLOPs' % (flops * img_size[0] / stride * img_size[1] / stride)  # 640x640 GFLOPs
        
        if verbose:
            print(f'Model FLOPs: {fs}')
    except Exception as e:
        if verbose:
            print(f'Failed to calculate FLOPs: {e}')
        fs = ''
        
    return n_p, n_g


def copy_weights(weights, model):
    """
    Copy weights from one model to another
    
    Args:
        weights: Source weights
        model: Target model
    """
    # Copy weights
    state_dict = weights.float().state_dict() if hasattr(weights, 'state_dict') else weights
    
    # Check compatibility
    expected_keys = list(model.state_dict().keys())
    received_keys = list(state_dict.keys())
    
    # Filter out unexpected keys
    missing = [k for k in expected_keys if k not in received_keys]
    
    # Print warning for missing keys
    if len(missing) > 0:
        print(f'WARNING: Missing keys: {missing}')
        
    # Load compatible weights
    model.load_state_dict(state_dict, strict=False)


def freeze_layers(model, freeze=True):
    """
    Freeze/unfreeze model layers for transfer learning
    
    Args:
        model: Model to freeze/unfreeze
        freeze: Whether to freeze layers
    """
    # Freeze specified layers
    if freeze:
        for param in model.parameters():
            param.requires_grad = False
    else:
        for param in model.parameters():
            param.requires_grad = True


def is_parallel(model):
    """
    Check if model is in parallel mode (DP or DDP)
    
    Args:
        model: PyTorch model
        
    Returns:
        Boolean indicating if model is parallel
    """
    return type(model) in (nn.parallel.DataParallel, nn.parallel.DistributedDataParallel)


def fuse_conv_and_bn(conv, bn):
    """
    Fuse Conv2d and BatchNorm2d layers for inference optimization
    
    Args:
        conv: Convolution layer
        bn: BatchNorm layer
        
    Returns:
        Fused convolution layer
    """
    # Prepare filters
    w = conv.weight
    mean = bn.running_mean
    var = bn.running_var
    beta = bn.bias if bn.bias is not None else torch.zeros_like(mean)
    gamma = bn.weight
    
    # Prepare spatial dimension
    b, c, h, w = conv.weight.shape
    
    # Prepare normalization constants
    eps = bn.eps
    std = (var + eps).sqrt()
    t = gamma / std
    
    # Prepare bias
    bias = beta - mean * gamma / std
    
    # Create fused layer
    fused_conv = nn.Conv2d(
        conv.in_channels,
        conv.out_channels,
        conv.kernel_size,
        conv.stride,
        conv.padding,
        conv.dilation,
        conv.groups,
        True
    )
    
    # Set weights and bias
    fused_conv.weight.data = conv.weight * t.reshape(c, 1, 1, 1)
    fused_conv.bias.data = bias if conv.bias is None else conv.bias + bias
    
    return fused_conv


def fuse_model(model):
    """
    Fuse Conv2d and BatchNorm2d layers for inference optimization
    
    Args:
        model: Model to fuse
        
    Returns:
        Fused model
    """
    # Get a copy of the model
    model_copy = model
    
    # Iterate through model modules
    for module_name in list(model_copy._modules):
        # Check if the module has submodules
        if len(list(model_copy._modules[module_name]._modules)) > 0:
            # Recursively fuse submodules
            model_copy._modules[module_name] = fuse_model(model_copy._modules[module_name])
        else:
            # Check if we have a Conv+BN sequence
            if isinstance(model_copy._modules[module_name], nn.Conv2d) and \
               module_name < len(list(model_copy._modules)) - 1 and \
               isinstance(model_copy._modules[list(model_copy._modules.keys())[list(model_copy._modules.keys()).index(module_name) + 1]], nn.BatchNorm2d):
                
                # Get Conv and BN modules
                conv = model_copy._modules[module_name]
                bn = model_copy._modules[list(model_copy._modules.keys())[list(model_copy._modules.keys()).index(module_name) + 1]]
                
                # Fuse modules
                model_copy._modules[module_name] = fuse_conv_and_bn(conv, bn)
                
                # Remove BN module
                model_copy._modules.pop(list(model_copy._modules.keys())[list(model_copy._modules.keys()).index(module_name) + 1])
            
    return model_copy 


def init_seeds(seed=0):
    """Initialize random seeds for reproducibility"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # for multi-GPU
    # torch.backends.cudnn.deterministic = True  # may impact performance
    # torch.backends.cudnn.benchmark = False  # may impact performance


def fuse_conv_and_bn(conv, bn):
    """
    Fuse Conv2d and BatchNorm2d layers for inference optimization
    
    Args:
        conv: Convolution layer
        bn: BatchNorm layer
        
    Returns:
        Fused convolution layer
    """
    # Prepare filters
    w = conv.weight
    mean = bn.running_mean
    var = bn.running_var
    beta = bn.bias if bn.bias is not None else torch.zeros_like(mean)
    gamma = bn.weight
    
    # Prepare spatial dimension
    b, c, h, w = conv.weight.shape
    
    # Prepare normalization constants
    eps = bn.eps
    std = (var + eps).sqrt()
    t = gamma / std
    
    # Prepare bias
    bias = beta - mean * gamma / std
    
    # Create fused layer
    fused_conv = nn.Conv2d(
        conv.in_channels,
        conv.out_channels,
        conv.kernel_size,
        conv.stride,
        conv.padding,
        conv.dilation,
        conv.groups,
        True
    )
    
    # Set weights and bias
    fused_conv.weight.data = conv.weight * t.reshape(c, 1, 1, 1)
    fused_conv.bias.data = bias if conv.bias is None else conv.bias + bias
    
    return fused_conv


def fuse_model(model):
    """
    Fuse Conv2d and BatchNorm2d layers for inference optimization
    
    Args:
        model: Model to fuse
        
    Returns:
        Fused model
    """
    # Get a copy of the model
    model_copy = model
    
    # Iterate through model modules
    for module_name in list(model_copy._modules):
        # Check if the module has submodules
        if len(list(model_copy._modules[module_name]._modules)) > 0:
            # Recursively fuse submodules
            model_copy._modules[module_name] = fuse_model(model_copy._modules[module_name])
        else:
            # Check if we have a Conv+BN sequence
            if isinstance(model_copy._modules[module_name], nn.Conv2d) and \
               module_name < len(list(model_copy._modules)) - 1 and \
               isinstance(model_copy._modules[list(model_copy._modules.keys())[list(model_copy._modules.keys()).index(module_name) + 1]], nn.BatchNorm2d):
                
                # Get Conv and BN modules
                conv = model_copy._modules[module_name]
                bn = model_copy._modules[list(model_copy._modules.keys())[list(model_copy._modules.keys()).index(module_name) + 1]]
                
                # Fuse modules
                model_copy._modules[module_name] = fuse_conv_and_bn(conv, bn)
                
                # Remove BN module
                model_copy._modules.pop(list(model_copy._modules.keys())[list(model_copy._modules.keys()).index(module_name) + 1])
            
    return model_copy 