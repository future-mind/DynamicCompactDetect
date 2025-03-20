import os
import torch
import time
import numpy as np
from tqdm import tqdm

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

def train_one_epoch(model, optimizer, dataloader, device, epoch, clip_gradients=None, log_interval=10):
    """
    Train the model for one epoch.
    
    Args:
        model: The model to train
        optimizer: Optimizer for updating weights
        dataloader: DataLoader for training data
        device: Device to train on
        epoch: Current epoch number
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
        
        # Forward pass
        loss_dict = model(images, targets)
        
        # Calculate total loss
        losses = sum(loss for loss in loss_dict.values() if isinstance(loss, (int, float, torch.Tensor)))
        
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
        
        # Backward pass
        optimizer.zero_grad()
        losses.backward()
        
        # Gradient clipping
        if clip_gradients:
            torch.nn.utils.clip_grad_norm_(model.parameters(), clip_gradients)
        
        # Update weights
        optimizer.step()
        
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

def validate(model, dataloader, device, iou_thres=0.5, conf_thres=0.01, nms_thres=0.5):
    """
    Validate the model on the validation dataset.
    
    Args:
        model: The model to validate
        dataloader: DataLoader for validation data
        device: Device to validate on
        iou_thres: IoU threshold for validation
        conf_thres: Confidence threshold for validation
        nms_thres: NMS threshold for validation
        
    Returns:
        tuple: (validation_loss, map_score)
    """
    model.eval()
    loss_meter = AverageMeter()
    
    # Run validation
    all_detections = []
    all_targets = []
    
    with torch.no_grad():
        for images, targets in tqdm(dataloader, desc="Validating"):
            # Move data to device
            images = images.to(device)
            targets = [{k: v.to(device) if isinstance(v, torch.Tensor) else v 
                        for k, v in t.items()} for t in targets]
            
            # Forward pass
            outputs = model(images, targets)
            
            # Calculate loss
            if isinstance(outputs, dict) and 'loss' in outputs:
                loss = outputs['loss']
                loss_meter.update(loss.item())
            
            # Get detections
            if isinstance(outputs, list):
                detections = outputs
            elif isinstance(outputs, dict) and 'detections' in outputs:
                detections = outputs['detections']
            else:
                detections = model.predict(images)
            
            # Store detections and targets for mAP calculation
            all_detections.extend(detections)
            all_targets.extend(targets)
    
    # Calculate mAP
    from utils.metrics import calculate_map
    map50, map = calculate_map(all_detections, all_targets, iou_threshold=iou_thres)
    
    print(f'Validation Loss: {loss_meter.avg:.4f}, mAP@0.5: {map50:.4f}, mAP@0.5:0.95: {map:.4f}')
    
    return loss_meter.avg, map50

def save_checkpoint(model, optimizer, scheduler, epoch, metrics, best_map, history, path):
    """
    Save model checkpoint.
    
    Args:
        model: Model to save
        optimizer: Optimizer state to save
        scheduler: Scheduler state to save
        epoch: Current epoch
        metrics: Dictionary of current metrics
        best_map: Best mAP value achieved
        history: Training history
        path: Path to save the checkpoint
    """
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(),
        'metrics': metrics,
        'best_map': best_map,
        'history': history
    }
    
    torch.save(checkpoint, path)
    print(f"Checkpoint saved to {path}")

class EarlyStopping:
    """
    Early stopping handler.
    
    Args:
        patience: Number of epochs to wait after validation score stops improving
        min_delta: Minimum change in validation score to qualify as improvement
        mode: 'min' for loss, 'max' for metrics like accuracy
    """
    
    def __init__(self, patience=5, min_delta=0, mode='min'):
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.counter = 0
        self.best_score = None
        self.early_stop = False
    
    def __call__(self, score):
        if self.best_score is None:
            self.best_score = score
            return False
        
        if self.mode == 'min':
            if score < self.best_score - self.min_delta:
                self.best_score = score
                self.counter = 0
            else:
                self.counter += 1
        else:  # mode == 'max'
            if score > self.best_score + self.min_delta:
                self.best_score = score
                self.counter = 0
            else:
                self.counter += 1
        
        if self.counter >= self.patience:
            self.early_stop = True
        
        return self.early_stop

class WarmupScheduler:
    """
    Warmup learning rate scheduler.
    
    Args:
        base_scheduler: Base scheduler to wrap
        warmup_epochs: Number of warmup epochs
        warmup_lr_init: Initial learning rate for warmup
        warmup_momentum: Momentum during warmup
    """
    
    def __init__(self, base_scheduler, warmup_epochs=3, warmup_lr_init=0, warmup_momentum=None):
        self.base_scheduler = base_scheduler
        self.warmup_epochs = warmup_epochs
        self.warmup_lr_init = warmup_lr_init
        self.warmup_momentum = warmup_momentum
        
        # Get base learning rate and momentum from optimizer
        self.optimizer = base_scheduler.optimizer
        self.base_lr = [group['lr'] for group in self.optimizer.param_groups]
        self.base_momentum = None
        
        if warmup_momentum is not None:
            self.base_momentum = [group['momentum'] for group in self.optimizer.param_groups 
                                 if 'momentum' in group]
        
        self.current_epoch = 0
    
    def step(self, epoch=None):
        """Step with warmup."""
        if epoch is not None:
            self.current_epoch = epoch
        
        if self.current_epoch < self.warmup_epochs:
            # Apply warmup
            warmup_factor = self.current_epoch / self.warmup_epochs
            
            # Update learning rate
            for i, group in enumerate(self.optimizer.param_groups):
                group['lr'] = self.warmup_lr_init + (self.base_lr[i] - self.warmup_lr_init) * warmup_factor
            
            # Update momentum if needed
            if self.warmup_momentum is not None and len(self.base_momentum) > 0:
                for i, group in enumerate(self.optimizer.param_groups):
                    if 'momentum' in group:
                        group['momentum'] = self.warmup_momentum + (self.base_momentum[i] - self.warmup_momentum) * warmup_factor
        else:
            # Use base scheduler after warmup
            self.base_scheduler.step()
        
        self.current_epoch += 1
    
    def state_dict(self):
        """Return the state dict for checkpoint."""
        return {
            'base_scheduler': self.base_scheduler.state_dict(),
            'warmup_epochs': self.warmup_epochs,
            'warmup_lr_init': self.warmup_lr_init,
            'warmup_momentum': self.warmup_momentum,
            'base_lr': self.base_lr,
            'base_momentum': self.base_momentum,
            'current_epoch': self.current_epoch
        }
    
    def load_state_dict(self, state_dict):
        """Load state dict from checkpoint."""
        self.base_scheduler.load_state_dict(state_dict['base_scheduler'])
        self.warmup_epochs = state_dict['warmup_epochs']
        self.warmup_lr_init = state_dict['warmup_lr_init']
        self.warmup_momentum = state_dict['warmup_momentum']
        self.base_lr = state_dict['base_lr']
        self.base_momentum = state_dict['base_momentum']
        self.current_epoch = state_dict['current_epoch']

def train_with_dynamic_routing_schedule(model, optimizer, dataloader, device, epoch, total_epochs, 
                                        dynamic_routing_schedule_start=0.1, clip_gradients=None, log_interval=10):
    """
    Train one epoch with dynamic routing schedule.
    
    Args:
        model: The model to train
        optimizer: Optimizer for updating weights
        dataloader: DataLoader for training data
        device: Device to train on
        epoch: Current epoch number
        total_epochs: Total number of epochs for training
        dynamic_routing_schedule_start: Percentage of training when to start enforcing dynamic routing
        clip_gradients: Maximum norm for gradient clipping (None to disable)
        log_interval: Log training stats every N batches
        
    Returns:
        Average loss for the epoch
    """
    model.train()
    loss_meter = AverageMeter()
    
    # Calculate dynamic routing factor
    progress = (epoch + 1) / total_epochs
    dynamic_routing_factor = min(1.0, max(0.0, (progress - dynamic_routing_schedule_start) / (1 - dynamic_routing_schedule_start)))
    
    # Set dynamic routing factor in model
    if hasattr(model, 'set_dynamic_routing_factor'):
        model.set_dynamic_routing_factor(dynamic_routing_factor)
    
    start_time = time.time()
    for i, (images, targets) in enumerate(dataloader):
        # Move data to device
        images = images.to(device)
        targets = [{k: v.to(device) if isinstance(v, torch.Tensor) else v 
                    for k, v in t.items()} for t in targets]
        
        # Forward pass
        loss_dict = model(images, targets)
        
        # Calculate total loss
        losses = sum(loss for loss in loss_dict.values() if isinstance(loss, (int, float, torch.Tensor)))
        
        # If no valid losses were found, handle this case
        if not isinstance(losses, torch.Tensor):
            # Create a dummy loss of zero if needed
            losses = torch.tensor(0.0, device=device)
            print("Warning: No valid losses found in loss_dict")
        
        # Update meter
        loss_meter.update(losses.item())
        
        # Backward pass
        optimizer.zero_grad()
        losses.backward()
        
        # Gradient clipping
        if clip_gradients:
            torch.nn.utils.clip_grad_norm_(model.parameters(), clip_gradients)
        
        # Update weights
        optimizer.step()
        
        # Log progress
        if (i + 1) % log_interval == 0:
            lr = optimizer.param_groups[0]['lr']
            elapsed = time.time() - start_time
            print(f'Epoch [{epoch+1}][{i+1}/{len(dataloader)}] '
                  f'Loss: {loss_meter.avg:.4f} '
                  f'LR: {lr:.6f} '
                  f'DR Factor: {dynamic_routing_factor:.2f} '
                  f'Time: {elapsed:.2f}s')
            start_time = time.time()
    
    return loss_meter.avg 

def count_parameters(model):
    """Count the total and trainable parameters in a model.
    
    Args:
        model: The model to count parameters for
        
    Returns:
        tuple: (total_parameters, trainable_parameters)
    """
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total_params, trainable_params 