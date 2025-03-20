import os
import sys
import yaml
import argparse
import torch
import time
from tqdm import tqdm
from datetime import datetime
import json
import matplotlib.pyplot as plt
import numpy as np
import timeit
import psutil
# Add mixed precision support
try:
    import torch.amp as amp
except ImportError:
    # Fallback if amp is not available
    amp = None

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.dynamic_compact_detect import DynamicCompactDetect, HardwareOptimizer
from utils.data_utils import COCODataset, create_data_loaders, get_augmentations
from utils.train_utils import train_one_epoch, validate, save_checkpoint
from utils.visualization import plot_training_metrics
from utils.model_utils import count_parameters
from data_loader import CocoDataset
from data_utils import DataLoader
from utils.bbox_utils import bbox_iou
from utils.loss_utils import YOLOLoss

def train_model(config_path, output_dir, resume_from=None, force_device=None):
    """Train the DynamicCompactDetect model on the full COCO dataset."""
    # Get start time
    start_time = timeit.default_timer()
    
    # System information
    print("=" * 50)
    print("SYSTEM INFORMATION")
    print(f"CPU: {psutil.cpu_count(logical=True)} logical cores")
    print(f"Memory: {psutil.virtual_memory().total / (1024 * 1024 * 1024):.2f} GB total")
    print(f"PyTorch version: {torch.__version__}")
    print(f"MPS available: {hasattr(torch.backends, 'mps') and torch.backends.mps.is_available()}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"CUDA device: {torch.cuda.get_device_name(0)}")
    print("=" * 50)
    
    # Load configuration
    with open(config_path, 'r') as f:
        cfg = yaml.safe_load(f)
    
    # Create output directories
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(os.path.join(output_dir, 'weights'), exist_ok=True)
    os.makedirs(os.path.join(output_dir, 'logs'), exist_ok=True)
    
    # Save config for reproducibility
    with open(os.path.join(output_dir, 'config.yaml'), 'w') as f:
        yaml.dump(cfg, f)
    
    # Initialize device (with override)
    if force_device:
        device = torch.device(force_device)
        print(f"Using forced device: {device}")
    elif torch.cuda.is_available() and len(cfg['hardware']['gpu_ids']) > 0:
        device = torch.device(f"cuda:{cfg['hardware']['gpu_ids'][0]}")
        print(f"Using CUDA device: {device}")
    elif torch.backends.mps.is_available():
        try:
            device = torch.device("mps")
            print(f"Using Apple Metal device: {device}")
            
            # Apply MPS-specific optimizations if available
            torch.backends.mps.enable_buffer_sharing = True
            if 'hardware' in cfg and cfg['hardware'].get('memory_efficient', False):
                print("Memory efficient mode enabled for MPS")
        except Exception as e:
            print(f"Warning: MPS initialization failed with error: {e}")
            print("Falling back to CPU")
            device = torch.device("cpu")
    else:
        device = torch.device("cpu")
        print("Using CPU device")
    
    # Setup mixed precision training if enabled and supported
    use_mixed_precision = cfg['training'].get('mixed_precision', False)
    scaler = None
    
    if use_mixed_precision:
        if amp is not None:
            if device.type == 'cuda' or (device.type == 'mps' and hasattr(amp, 'GradScaler')):
                scaler = amp.GradScaler()
                print("Using mixed precision training with gradient scaling")
            else:
                print("Mixed precision requested but not supported on this device, falling back to full precision")
        else:
            print("Mixed precision requested but torch.amp not available, falling back to full precision")

    # Data augmentation
    train_transforms, val_transforms = get_augmentations(cfg)
    
    # Create datasets
    train_dataset = COCODataset(
        img_dir=cfg['dataset']['train_images'],
        ann_file=cfg['dataset']['train_annotations'],
        input_size=cfg['model']['input_size'],
        transforms=train_transforms
    )
    
    val_dataset = COCODataset(
        img_dir=cfg['dataset']['val_images'],
        ann_file=cfg['dataset']['val_annotations'],
        input_size=cfg['model']['input_size'],
        transforms=val_transforms
    )
    
    # Create data loaders
    train_loader, val_loader = create_data_loaders(
        train_dataset, val_dataset, 
        batch_size=cfg['training']['batch_size'],
        num_workers=cfg['hardware']['num_workers'],
        pin_memory=cfg['hardware']['pinned_memory']
    )
    
    print(f"Training on {len(train_dataset)} images")
    print(f"Validating on {len(val_dataset)} images")
    
    # Initialize model
    model = DynamicCompactDetect(
        num_classes=cfg['model']['num_classes'],
        in_channels=cfg['input']['channels'],
        base_channels=cfg['model']['base_channels']
    )
    
    # Before loading the model to device, reset the backbone's dynamic_blocks attribute
    # and rebuild the stages with the correct dynamic_blocks setting
    use_dynamic_blocks = cfg['model'].get('use_dynamic_blocks', False)
    model.backbone.dynamic_blocks = use_dynamic_blocks
    
    # Count and report parameters
    total_params, trainable_params = count_parameters(model)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    
    # Platform-specific optimizations
    if cfg['hardware'].get('use_platform_optimizations', False):
        model, device = HardwareOptimizer.optimize_for_platform(model)
    else:
        model = model.to(device)
    
    # Training setup
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=cfg['training']['learning_rate'],
        weight_decay=cfg['training']['weight_decay']
    )
    
    # Learning rate scheduler
    if cfg['training']['lr_scheduler'] == 'cosine':
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, 
            T_max=cfg['training']['epochs'] - cfg['training']['warmup_epochs']
        )
    else:
        scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer, 
            step_size=30, 
            gamma=0.1
        )
    
    # Resume from checkpoint if specified
    start_epoch = 0
    best_map = 0
    training_history = {'train_loss': [], 'val_loss': [], 'map': [], 'lr': []}
    
    if resume_from:
        print(f"Resuming from checkpoint: {resume_from}")
        checkpoint = torch.load(resume_from, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        start_epoch = checkpoint['epoch'] + 1
        best_map = checkpoint.get('best_map', 0)
        training_history = checkpoint.get('history', training_history)
        print(f"Resumed from epoch {start_epoch}")
    
    # Main training loop
    print(f"Starting training for {cfg['training']['epochs']} epochs")
    train_start_time = time.time()
    
    for epoch in range(start_epoch, cfg['training']['epochs']):
        # Train for one epoch
        epoch_start_time = time.time()
        
        if scaler is not None:
            # Use train_one_epoch with mixed precision
            train_loss = train_one_epoch_mixed_precision(
                model, optimizer, train_loader, device, epoch, scaler,
                cfg['training']['clip_gradients'],
                log_interval=cfg['logging']['log_interval']
            )
        else:
            # Use standard train_one_epoch
            train_loss = train_one_epoch(
                model, optimizer, train_loader, device, epoch, 
                cfg['training']['clip_gradients'],
                log_interval=cfg['logging']['log_interval']
            )
        
        # Update learning rate
        scheduler.step()
        
        # Validate model
        if (epoch + 1) % cfg['validation']['frequency'] == 0:
            val_loss, map_score = validate(
                model, val_loader, device,
                iou_thres=cfg['validation']['iou_threshold'],
                conf_thres=cfg['validation']['conf_threshold']
            )
            
            # Record training history
            training_history['train_loss'].append(train_loss)
            training_history['val_loss'].append(val_loss)
            training_history['map'].append(map_score)
            training_history['lr'].append(optimizer.param_groups[0]['lr'])
            
            # Update best mAP
            is_best = map_score > best_map
            if is_best:
                best_map = map_score
            
            # Print epoch summary
            epoch_time = time.time() - epoch_start_time
            print(f"Epoch {epoch+1}/{cfg['training']['epochs']}: "
                  f"Train Loss: {train_loss:.4f} | "
                  f"Val Loss: {val_loss:.4f} | "
                  f"mAP: {map_score:.4f} | "
                  f"Time: {epoch_time:.2f}s")
            
            # Save checkpoint
            if (epoch + 1) % cfg['checkpointing']['save_every'] == 0 or is_best:
                checkpoint_path = os.path.join(
                    output_dir, 'weights', 
                    f"dynamiccompactdetect_epoch{epoch+1}.pt"
                )
                
                save_checkpoint(
                    model, optimizer, scheduler, epoch, 
                    {'train_loss': train_loss, 'val_loss': val_loss, 'map': map_score},
                    best_map, training_history, checkpoint_path
                )
                
                if is_best:
                    best_path = os.path.join(output_dir, 'weights', 'best_model.pt')
                    torch.save(model.state_dict(), best_path)
                    print(f"Saved best model with mAP: {best_map:.4f}")
        
        # Plot and save training progress
        if (epoch + 1) % 5 == 0:
            plot_training_metrics(training_history, os.path.join(output_dir, 'logs', 'training_metrics.png'))
    
    # Training complete
    total_time = time.time() - train_start_time
    hours, remainder = divmod(total_time, 3600)
    minutes, seconds = divmod(remainder, 60)
    
    print(f"Training complete in {int(hours)}h {int(minutes)}m {int(seconds)}s")
    print(f"Best mAP: {best_map:.4f}")
    
    # Save final model
    final_path = os.path.join(output_dir, 'weights', 'final_model.pt')
    torch.save(model.state_dict(), final_path)
    
    # Save training metrics
    metrics_path = os.path.join(output_dir, 'logs', 'training_metrics.json')
    with open(metrics_path, 'w') as f:
        json.dump(training_history, f)
    
    # Plot final metrics
    plot_training_metrics(training_history, os.path.join(output_dir, 'logs', 'final_training_metrics.png'))
    
    # Return the trained model and best mAP
    return model, best_map, training_history

def train_one_epoch_mixed_precision(model, optimizer, dataloader, device, epoch, scaler, clip_gradients=None, log_interval=10):
    """
    Train the model for one epoch using mixed precision.
    
    Args:
        model: The model to train
        optimizer: Optimizer for updating weights
        dataloader: DataLoader for training data
        device: Device to train on
        epoch: Current epoch number
        scaler: GradScaler for mixed precision training
        clip_gradients: Maximum norm for gradient clipping (None to disable)
        log_interval: Log training stats every N batches
        
    Returns:
        Average loss for the epoch
    """
    model.train()
    loss_meter = AverageMeter()
    box_loss_meter = AverageMeter()
    obj_loss_meter = AverageMeter()
    cls_loss_meter = AverageMeter()
    
    start_time = time.time()
    for i, (images, targets) in enumerate(dataloader):
        # Move data to device
        images = images.to(device)
        targets = [{k: v.to(device) if isinstance(v, torch.Tensor) else v 
                    for k, v in t.items()} for t in targets]
        
        # Forward pass with mixed precision
        with amp.autocast(device_type=device.type):
            loss_dict = model(images, targets)
            
            # Calculate total loss
            losses = sum(loss for loss in loss_dict.values() 
                        if isinstance(loss, (int, float, torch.Tensor)))
            
            # If no valid losses were found, handle this case
            if not isinstance(losses, torch.Tensor):
                # Create a dummy loss of zero if needed
                losses = torch.tensor(0.0, device=device)
                print("Warning: No valid losses found in loss_dict")
        
        # Update meters
        loss_meter.update(losses.item())
        if 'box_loss' in loss_dict:
            box_loss_meter.update(loss_dict['box_loss'].item())
        if 'obj_loss' in loss_dict:
            obj_loss_meter.update(loss_dict['obj_loss'].item())
        if 'cls_loss' in loss_dict:
            cls_loss_meter.update(loss_dict['cls_loss'].item())
        
        # Backward pass with gradient scaling
        optimizer.zero_grad()
        scaler.scale(losses).backward()
        
        # Gradient clipping with scaler
        if clip_gradients:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), clip_gradients)
        
        # Update weights with scaler
        scaler.step(optimizer)
        scaler.update()
        
        # Log progress
        if (i + 1) % log_interval == 0:
            lr = optimizer.param_groups[0]['lr']
            elapsed = time.time() - start_time
            print(f'Epoch [{epoch+1}][{i+1}/{len(dataloader)}] '
                  f'Loss: {loss_meter.avg:.4f} '
                  f'Box: {box_loss_meter.avg:.4f} '
                  f'Obj: {obj_loss_meter.avg:.4f} '
                  f'Cls: {cls_loss_meter.avg:.4f} '
                  f'LR: {lr:.6f} '
                  f'Time: {elapsed:.2f}s')
            start_time = time.time()
    
    return loss_meter.avg

class AverageMeter:
    """Computes and stores the average and current value."""
    
    def __init__(self):
        self.reset()
    
    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0
    
    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

def main():
    parser = argparse.ArgumentParser(description='Train DynamicCompactDetect on full COCO dataset')
    parser.add_argument('--config', type=str, default='config/full_coco_config.yaml',
                        help='Path to configuration file')
    parser.add_argument('--output-dir', type=str, default='results/full_training',
                        help='Directory to save outputs')
    parser.add_argument('--resume', type=str, default=None,
                        help='Path to checkpoint to resume from')
    parser.add_argument('--device', type=str, default=None,
                        help='Force device to use (cpu, cuda, or mps). If not specified, will auto-select.')
    args = parser.parse_args()
    
    train_model(args.config, args.output_dir, args.resume, args.device)

if __name__ == "__main__":
    main() 