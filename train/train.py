import os
import sys
import time
import yaml
import argparse
import numpy as np
from pathlib import Path
from tqdm import tqdm
from datetime import datetime

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.cuda import amp
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.dynamic_compact_detect import DynamicCompactDetect
from utils.data_utils import COCODataset, create_data_loaders, get_augmentations
from utils.model_utils import (
    init_model, load_checkpoint, save_checkpoint, 
    ModelEMA, count_parameters, export_onnx
)
from utils.benchmark_utils import AverageMeter, ProgressMeter, compute_mAP

class FocalLoss(nn.Module):
    """Focal Loss for classification."""
    def __init__(self, gamma=2.0, alpha=0.25, reduction='mean'):
        super().__init__()
        self.gamma = gamma
        self.alpha = alpha
        self.reduction = reduction
        
    def forward(self, input, target):
        ce_loss = F.binary_cross_entropy_with_logits(input, target, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * ce_loss
        
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss

class YOLOLoss(nn.Module):
    """Loss function for the YOLO detection model."""
    def __init__(self, cfg, device):
        super().__init__()
        self.box_loss_weight = cfg['loss']['box_loss_weight']
        self.cls_loss_weight = cfg['loss']['cls_loss_weight']
        self.obj_loss_weight = cfg['loss']['obj_loss_weight']
        self.early_exit_aux_loss_weight = cfg['loss']['early_exit_aux_loss_weight']
        self.label_smoothing = cfg['loss']['label_smoothing']
        
        self.num_classes = cfg['model']['num_classes']
        self.device = device
        self.cls_loss = FocalLoss(gamma=cfg['loss']['focal_loss_gamma'])
        
    def forward(self, predictions, targets, model):
        # This is a simplified loss function - in a real implementation,
        # this would need to handle anchor boxes, ground truth matching, etc.
        
        # If early exit was taken, compute auxiliary loss
        if 'early_exit' in predictions:
            return self._compute_early_exit_loss(predictions['early_exit'], targets)
            
        # Extract outputs from different detector levels
        detector_outputs = predictions['outputs']
        
        # Initialize loss components
        box_loss = torch.tensor(0.0, device=self.device)
        obj_loss = torch.tensor(0.0, device=self.device)
        cls_loss = torch.tensor(0.0, device=self.device)
        
        # Process each detector level
        for level, output in enumerate(detector_outputs):
            # Process targets for this level (simplified)
            # In practice, this requires careful target assignment to anchors
            level_box_loss, level_obj_loss, level_cls_loss = self._compute_level_loss(output, targets, level)
            
            # Accumulate losses
            box_loss += level_box_loss
            obj_loss += level_obj_loss
            cls_loss += level_cls_loss
        
        # Combine loss components with weights
        total_loss = (
            self.box_loss_weight * box_loss + 
            self.obj_loss_weight * obj_loss + 
            self.cls_loss_weight * cls_loss
        )
        
        return total_loss, {
            'box_loss': box_loss.item(),
            'obj_loss': obj_loss.item(),
            'cls_loss': cls_loss.item(),
            'total_loss': total_loss.item()
        }
    
    def _compute_level_loss(self, output, targets, level):
        """Compute loss for a single detection level."""
        # This is a placeholder function - a real implementation would:
        # 1. Extract predictions (bboxes, objectness, class scores) from output tensor
        # 2. Match predictions to ground truth targets
        # 3. Calculate IoU between predictions and targets
        # 4. Compute box loss (e.g., GIoU, CIoU)
        # 5. Compute objectness loss (BCE)
        # 6. Compute classification loss (BCE with focal loss)
        
        # For this template, we'll use dummy losses based on tensor shapes
        batch_size = output.shape[0]
        
        # Placeholder - simulate computing losses
        box_loss = torch.sum(output.abs()) * 0.01 / batch_size
        obj_loss = torch.sum(output.abs()) * 0.01 / batch_size
        cls_loss = torch.sum(output.abs()) * 0.01 / batch_size
        
        return box_loss, obj_loss, cls_loss
    
    def _compute_early_exit_loss(self, early_exit, targets):
        """Compute loss for early exit predictions."""
        # Again, this is simplified - actual implementation would depend
        # on how early exit targets are formatted and processed
        
        batch_size = early_exit.shape[0]
        loss = torch.sum(early_exit.abs()) * 0.01 / batch_size
        
        return loss * self.early_exit_aux_loss_weight, {
            'early_exit_loss': loss.item(),
            'total_loss': (loss * self.early_exit_aux_loss_weight).item()
        }

def train_one_epoch(
    model, dataloader, optimizer, scheduler, loss_fn, 
    epoch, device, cfg, scaler=None, ema=None, writer=None
):
    """Train the model for one epoch."""
    model.train()
    
    # Metrics
    batch_time = AverageMeter('Batch', ':6.3f')
    data_time = AverageMeter('Data', ':6.3f')
    losses = AverageMeter('Loss', ':.4f')
    box_losses = AverageMeter('BoxLoss', ':.4f')
    obj_losses = AverageMeter('ObjLoss', ':.4f')
    cls_losses = AverageMeter('ClsLoss', ':.4f')
    
    log_interval = cfg['logging']['log_interval']
    clip_grad = cfg['training']['clip_gradients']
    
    progress = ProgressMeter(
        len(dataloader),
        [batch_time, data_time, losses, box_losses, obj_losses, cls_losses],
        prefix=f"Epoch: [{epoch}]"
    )
    
    # Start time measurement
    end = time.time()
    
    # Adjust dynamic routing parameters
    dynamic_routing_schedule = max(0, min(1, 
        (epoch - cfg['dynamic_features']['dynamic_routing_schedule_start']) / 
        (cfg['training']['epochs'] * 0.5)
    ))
    
    # Training loop
    for batch_idx, (images, targets) in enumerate(dataloader):
        # Measure data loading time
        data_time.update(time.time() - end)
        
        # Move data to device
        images = images.to(device)
        targets = [t.to(device) for t in targets]
        
        # Forward pass with mixed precision if enabled
        if cfg['training']['mixed_precision'] and scaler is not None:
            with amp.autocast():
                predictions = model(images)
                loss, loss_items = loss_fn(predictions, targets, model)
                
            # Backward pass with gradient scaling
            scaler.scale(loss).backward()
            
            # Gradient clipping
            if clip_grad > 0:
                nn.utils.clip_grad_norm_(model.parameters(), clip_grad)
                
            # Optimizer step with scaler
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()
        else:
            # Standard forward and backward pass
            predictions = model(images)
            loss, loss_items = loss_fn(predictions, targets, model)
            
            loss.backward()
            
            # Gradient clipping
            if clip_grad > 0:
                nn.utils.clip_grad_norm_(model.parameters(), clip_grad)
                
            optimizer.step()
            optimizer.zero_grad()
        
        # Update EMA model if enabled
        if ema is not None:
            ema.update(model)
        
        # Update metrics
        losses.update(loss_items['total_loss'], images.size(0))
        if 'box_loss' in loss_items:
            box_losses.update(loss_items['box_loss'], images.size(0))
            obj_losses.update(loss_items['obj_loss'], images.size(0))
            cls_losses.update(loss_items['cls_loss'], images.size(0))
        
        # Measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()
        
        # Log progress
        if batch_idx % log_interval == 0:
            progress.display(batch_idx)
            
            # Log to TensorBoard
            if writer is not None:
                step = epoch * len(dataloader) + batch_idx
                writer.add_scalar('train/loss', losses.val, step)
                writer.add_scalar('train/box_loss', box_losses.val, step)
                writer.add_scalar('train/obj_loss', obj_losses.val, step)
                writer.add_scalar('train/cls_loss', cls_losses.val, step)
                writer.add_scalar('train/lr', optimizer.param_groups[0]['lr'], step)
                
                # Log images and predictions periodically
                if cfg['logging']['log_images'] and batch_idx % (log_interval * 10) == 0:
                    # Log sample images with predictions
                    log_images(writer, images, predictions, targets, step)
    
    # Update learning rate
    scheduler.step()
    
    return losses.avg

def validate(model, dataloader, loss_fn, device, cfg):
    """Validate the model on the validation set."""
    model.eval()
    
    # Metrics
    losses = AverageMeter('Loss', ':.4f')
    
    all_predictions = []
    all_targets = []
    
    # Disable gradients for validation
    with torch.no_grad():
        for images, targets in tqdm(dataloader, desc="Validating"):
            # Move data to device
            images = images.to(device)
            targets = [t.to(device) for t in targets]
            
            # Forward pass
            predictions = model(images, early_exit=False)  # Disable early exit for validation
            
            # Compute loss
            loss, loss_items = loss_fn(predictions, targets, model)
            
            # Update metrics
            losses.update(loss_items['total_loss'], images.size(0))
            
            # Store predictions and targets for mAP calculation
            all_predictions.append(predictions)
            all_targets.append(targets)
    
    # Compute mAP
    mAP, mAP50 = compute_mAP(all_predictions, all_targets, 
                               iou_threshold=cfg['validation']['iou_threshold'],
                               conf_threshold=cfg['validation']['conf_threshold'])
    
    print(f"Validation: Loss: {losses.avg:.4f}, mAP: {mAP:.4f}, mAP@0.5: {mAP50:.4f}")
    
    return losses.avg, mAP, mAP50

def log_images(writer, images, predictions, targets, step):
    """Log images with predictions to TensorBoard."""
    # This is a placeholder function
    # In a real implementation, this would:
    # 1. Convert predictions to bounding boxes
    # 2. Draw boxes on images
    # 3. Log the annotated images to TensorBoard
    
    # For simplicity, we'll just log the input images
    grid = torchvision.utils.make_grid(images[:4])
    writer.add_image('train/input_images', grid, step)

def main():
    parser = argparse.ArgumentParser(description='Train DynamicCompactDetect model')
    parser.add_argument('--config', type=str, default='train/config.yaml', 
                        help='Path to the configuration file')
    parser.add_argument('--resume', type=str, default='', 
                        help='Path to the checkpoint to resume from')
    parser.add_argument('--epochs', type=int, default=None, 
                        help='Number of epochs to train (overrides config)')
    parser.add_argument('--batch-size', type=int, default=None, 
                        help='Batch size (overrides config)')
    parser.add_argument('--no-mixed-precision', action='store_true', 
                        help='Disable mixed precision training')
    parser.add_argument('--no-ema', action='store_true', 
                        help='Disable Exponential Moving Average model')
    args = parser.parse_args()
    
    # Load configuration
    with open(args.config, 'r') as f:
        cfg = yaml.safe_load(f)
    
    # Override config with command line arguments
    if args.epochs is not None:
        cfg['training']['epochs'] = args.epochs
    if args.batch_size is not None:
        cfg['training']['batch_size'] = args.batch_size
    if args.no_mixed_precision:
        cfg['training']['mixed_precision'] = False
    if args.no_ema:
        cfg['training']['use_ema'] = False
    
    # Set random seed for reproducibility
    seed = cfg['advanced']['seed']
    torch.manual_seed(seed)
    np.random.seed(seed)
    
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        
    # Set deterministic mode if requested
    if cfg['advanced']['deterministic']:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    else:
        torch.backends.cudnn.benchmark = cfg['advanced']['cudnn_benchmark']
    
    # Create output directories
    os.makedirs(cfg['checkpoints']['save_dir'], exist_ok=True)
    os.makedirs(cfg['logging']['log_dir'], exist_ok=True)
    
    # Initialize device
    if torch.cuda.is_available() and len(cfg['hardware']['gpu_ids']) > 0:
        device = torch.device(f"cuda:{cfg['hardware']['gpu_ids'][0]}")
    else:
        device = torch.device("cpu")
    
    # Create data loaders
    train_transforms, val_transforms = get_augmentations(cfg)
    train_dataset = COCODataset(
        img_dir=cfg['dataset']['train_images'],
        ann_file=cfg['dataset']['train_annotations'],
        transforms=train_transforms,
        img_size=cfg['model']['input_size']
    )
    val_dataset = COCODataset(
        img_dir=cfg['dataset']['val_images'],
        ann_file=cfg['dataset']['val_annotations'],
        transforms=val_transforms,
        img_size=cfg['model']['input_size']
    )
    
    train_loader, val_loader = create_data_loaders(
        train_dataset, val_dataset, 
        batch_size=cfg['training']['batch_size'],
        num_workers=cfg['hardware']['num_workers'],
        pin_memory=cfg['hardware']['pinned_memory']
    )
    
    # Initialize model
    model = DynamicCompactDetect(
        num_classes=cfg['model']['num_classes'],
        base_channels=cfg['model']['base_channels']
    )
    
    # Platform-specific optimizations
    if cfg['hardware']['use_platform_optimizations']:
        model, device = model.optimize_for_platform()
    else:
        model = model.to(device)
    
    # Print model summary
    print(f"Model initialized with {count_parameters(model):,} parameters")
    
    # Multi-GPU if available
    if torch.cuda.is_available() and len(cfg['hardware']['gpu_ids']) > 1:
        model = nn.DataParallel(model, device_ids=cfg['hardware']['gpu_ids'])
    
    # Initialize optimizer
    optimizer_name = cfg['training']['optimizer'].lower()
    if optimizer_name == 'sgd':
        optimizer = optim.SGD(
            model.parameters(),
            lr=cfg['training']['initial_lr'],
            momentum=cfg['training']['momentum'],
            weight_decay=cfg['training']['weight_decay']
        )
    elif optimizer_name == 'adam':
        optimizer = optim.Adam(
            model.parameters(),
            lr=cfg['training']['initial_lr'],
            weight_decay=cfg['training']['weight_decay']
        )
    elif optimizer_name == 'adamw':
        optimizer = optim.AdamW(
            model.parameters(),
            lr=cfg['training']['initial_lr'],
            weight_decay=cfg['training']['weight_decay']
        )
    else:
        raise ValueError(f"Unsupported optimizer: {optimizer_name}")
    
    # Initialize learning rate scheduler
    scheduler_name = cfg['training']['lr_scheduler'].lower()
    if scheduler_name == 'step':
        scheduler = optim.lr_scheduler.StepLR(
            optimizer, step_size=30, gamma=0.1
        )
    elif scheduler_name == 'cosine':
        scheduler = optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=cfg['training']['epochs']
        )
    elif scheduler_name == 'linear':
        lf = lambda x: (1 - x / cfg['training']['epochs']) * (1.0 - 0.1) + 0.1  # linear
        scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lf)
    else:
        raise ValueError(f"Unsupported scheduler: {scheduler_name}")
    
    # Initialize mixed precision scaler
    scaler = amp.GradScaler() if cfg['training']['mixed_precision'] and torch.cuda.is_available() else None
    
    # Initialize EMA model
    ema = ModelEMA(model, decay=cfg['training']['ema_decay']) if cfg['training']['use_ema'] else None
    
    # Initialize loss function
    loss_fn = YOLOLoss(cfg, device)
    
    # Resume from checkpoint if specified
    start_epoch = 0
    best_mAP = 0.0
    if args.resume:
        checkpoint = load_checkpoint(args.resume, model, optimizer, scheduler, scaler, ema)
        if checkpoint:
            start_epoch = checkpoint['epoch'] + 1
            best_mAP = checkpoint.get('best_mAP', 0.0)
            print(f"Resumed from epoch {start_epoch}, best mAP: {best_mAP:.4f}")
    
    # Initialize TensorBoard writer
    log_dir = os.path.join(cfg['logging']['log_dir'], datetime.now().strftime("%Y%m%d-%H%M%S"))
    writer = SummaryWriter(log_dir) if cfg['logging']['tensorboard'] else None
    
    # Training loop
    num_epochs = cfg['training']['epochs']
    val_interval = cfg['validation']['val_interval']
    save_interval = cfg['checkpoints']['save_interval']
    early_stopping_patience = cfg['training']['early_stopping_patience']
    early_stopping_counter = 0
    
    try:
        for epoch in range(start_epoch, num_epochs):
            print(f"Epoch {epoch+1}/{num_epochs}")
            
            # Train one epoch
            train_loss = train_one_epoch(
                model, train_loader, optimizer, scheduler, loss_fn,
                epoch, device, cfg, scaler, ema, writer
            )
            
            # Log training metrics
            if writer is not None:
                writer.add_scalar('epoch/train_loss', train_loss, epoch)
                writer.add_scalar('epoch/learning_rate', optimizer.param_groups[0]['lr'], epoch)
            
            # Validate periodically
            if (epoch + 1) % val_interval == 0 or epoch == num_epochs - 1:
                # Use EMA model for validation if available
                val_model = ema.ema if ema is not None else model
                
                val_loss, mAP, mAP50 = validate(val_model, val_loader, loss_fn, device, cfg)
                
                # Log validation metrics
                if writer is not None:
                    writer.add_scalar('epoch/val_loss', val_loss, epoch)
                    writer.add_scalar('epoch/mAP', mAP, epoch)
                    writer.add_scalar('epoch/mAP50', mAP50, epoch)
                
                # Save best model
                if cfg['checkpoints']['save_best'] and mAP > best_mAP:
                    best_mAP = mAP
                    save_checkpoint(
                        model, optimizer, scheduler, epoch, best_mAP,
                        os.path.join(cfg['checkpoints']['save_dir'], 'best.pth'),
                        scaler, ema
                    )
                    early_stopping_counter = 0
                else:
                    early_stopping_counter += 1
                
                # Early stopping
                if early_stopping_patience > 0 and early_stopping_counter >= early_stopping_patience:
                    print(f"Early stopping at epoch {epoch+1}")
                    break
            
            # Save checkpoint periodically
            if (epoch + 1) % save_interval == 0:
                save_checkpoint(
                    model, optimizer, scheduler, epoch, best_mAP,
                    os.path.join(cfg['checkpoints']['save_dir'], f'checkpoint_epoch_{epoch+1}.pth'),
                    scaler, ema
                )
            
            # Adjust early exit threshold dynamically if enabled
            if (cfg['dynamic_features']['enable_dynamic_early_exit_threshold'] and 
                epoch >= cfg['dynamic_features']['adjust_early_exit_epoch']):
                # This would be implemented based on validation metrics
                # Placeholder for the dynamic adjustment logic
                pass
    
    except KeyboardInterrupt:
        print("Training interrupted by user")
    
    finally:
        # Save final model
        if cfg['checkpoints']['save_last']:
            save_checkpoint(
                model, optimizer, scheduler, epoch, best_mAP,
                os.path.join(cfg['checkpoints']['save_dir'], 'last.pth'),
                scaler, ema
            )
        
        # Export to ONNX if enabled
        if cfg['advanced']['onnx_export']:
            model_to_export = ema.ema if ema is not None else model
            export_onnx(
                model_to_export, 
                os.path.join(cfg['checkpoints']['save_dir'], 'model.onnx'),
                cfg['model']['input_size']
            )
        
        # Close TensorBoard writer
        if writer is not None:
            writer.close()
    
    print(f"Training completed. Best mAP: {best_mAP:.4f}")

if __name__ == "__main__":
    main() 