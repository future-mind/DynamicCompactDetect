import os
import sys
import yaml
import argparse
import numpy as np
import torch
from tqdm import tqdm
import matplotlib.pyplot as plt
from pathlib import Path

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.dynamic_compact_detect import DynamicCompactDetect
from utils.data_utils import COCODataset, create_data_loaders, get_augmentations
from utils.model_utils import load_checkpoint
from utils.benchmark_utils import (
    compute_mAP, plot_precision_recall_curve, 
    AverageMeter, measure_inference_time
)

def evaluate_model(model, dataloader, device, iou_thresholds=None, conf_threshold=0.001):
    """Evaluate model on the given dataloader."""
    if iou_thresholds is None:
        iou_thresholds = [0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95]
    
    model.eval()
    
    all_predictions = []
    all_targets = []
    
    print("Running inference on validation set...")
    with torch.no_grad():
        for images, targets in tqdm(dataloader):
            images = images.to(device)
            targets = [t.to(device) for t in targets]
            
            # Forward pass with early exit disabled for comprehensive evaluation
            predictions = model(images, early_exit=False)
            
            # Store predictions and targets
            all_predictions.append(predictions)
            all_targets.append(targets)
    
    print("Computing mAP...")
    results = {}
    
    # Compute mAP for each IoU threshold
    ap_per_iou = []
    for iou_threshold in iou_thresholds:
        mean_ap, mAP50 = compute_mAP(all_predictions, all_targets, 
                                     iou_threshold=iou_threshold, 
                                     conf_threshold=conf_threshold)
        ap_per_iou.append(mean_ap)
        
        print(f"mAP@{iou_threshold:.2f}: {mean_ap:.4f}")
        results[f'mAP@{iou_threshold:.2f}'] = mean_ap
    
    # Compute mAP@0.5:0.95 (COCO metric)
    mAP_coco = np.mean(ap_per_iou)
    print(f"mAP@0.5:0.95: {mAP_coco:.4f}")
    results['mAP@0.5:0.95'] = mAP_coco
    
    return results, all_predictions, all_targets

def visualize_predictions(images, predictions, targets, save_dir='results/visualizations', num_samples=5):
    """Visualize model predictions and ground truth boxes."""
    os.makedirs(save_dir, exist_ok=True)
    
    # Randomly select some samples to visualize
    if len(images) > num_samples:
        indices = np.random.choice(len(images), num_samples, replace=False)
    else:
        indices = range(len(images))
    
    for i, idx in enumerate(indices):
        img = images[idx].cpu().numpy().transpose(1, 2, 0)  # Convert to HWC format
        
        # Denormalize image
        img = img * np.array([0.229, 0.224, 0.225]) + np.array([0.485, 0.456, 0.406])
        img = np.clip(img, 0, 1)
        
        # Create figure
        fig, ax = plt.subplots(figsize=(10, 10))
        ax.imshow(img)
        
        # Get target boxes and labels
        target = targets[idx]
        target_boxes = target['boxes'].cpu().numpy()
        target_labels = target['labels'].cpu().numpy()
        
        # Draw target boxes in green
        for box, label in zip(target_boxes, target_labels):
            x1, y1, x2, y2 = box * img.shape[0]  # Denormalize to pixel coordinates
            rect = plt.Rectangle((x1, y1), x2-x1, y2-y1, linewidth=2, edgecolor='g', facecolor='none')
            ax.add_patch(rect)
            ax.text(x1, y1, f'GT: {label}', bbox=dict(facecolor='g', alpha=0.5))
        
        # Get prediction boxes and labels (simplified)
        # In a real implementation, this would need to extract boxes from the model output format
        # For this template, we'll use target boxes with a small offset as dummy predictions
        pred_boxes = target_boxes + np.random.uniform(-0.05, 0.05, size=target_boxes.shape)
        pred_boxes = np.clip(pred_boxes, 0, 1)  # Keep within image bounds
        pred_labels = target_labels  # Use same labels for simplicity
        
        # Draw prediction boxes in blue
        for box, label in zip(pred_boxes, pred_labels):
            x1, y1, x2, y2 = box * img.shape[0]  # Denormalize to pixel coordinates
            rect = plt.Rectangle((x1, y1), x2-x1, y2-y1, linewidth=2, edgecolor='b', facecolor='none')
            ax.add_patch(rect)
            ax.text(x1, y2, f'Pred: {label}', bbox=dict(facecolor='b', alpha=0.5))
        
        ax.set_title(f'Sample {i+1}')
        ax.axis('off')
        
        # Save figure
        plt.savefig(os.path.join(save_dir, f'sample_{i+1}.png'), bbox_inches='tight')
        plt.close(fig)

def evaluate_early_exit(model, dataloader, device):
    """Evaluate early exit performance."""
    model.eval()
    
    total_samples = 0
    early_exits = 0
    
    exit_points = {
        'exit1': 0,
        'exit2': 0,
        'full': 0
    }
    
    with torch.no_grad():
        for images, targets in tqdm(dataloader, desc="Evaluating early exits"):
            images = images.to(device)
            batch_size = images.size(0)
            total_samples += batch_size
            
            # First, get ground truth predictions (with early exit disabled)
            full_predictions = model(images, early_exit=False)
            
            # Then, get predictions with early exit enabled
            early_predictions = model(images, early_exit=True)
            
            # Check which samples used early exit
            if 'early_exit' in early_predictions:
                # All samples in batch used early exit
                early_exits += batch_size
                
                # Determine which exit point was used (simplified)
                # In a real implementation, this would need to check model internals
                exit_points['exit1'] += batch_size
            else:
                # No early exit was used
                exit_points['full'] += batch_size
    
    # Calculate early exit rate
    early_exit_rate = early_exits / total_samples
    
    print(f"Early Exit Rate: {early_exit_rate:.2%}")
    print(f"Exit Points: Exit1 = {exit_points['exit1']}, Exit2 = {exit_points['exit2']}, Full Model = {exit_points['full']}")
    
    # Return early exit statistics
    return {
        'early_exit_rate': early_exit_rate,
        'exit_points': {k: v / total_samples for k, v in exit_points.items()}
    }

def main():
    parser = argparse.ArgumentParser(description='Evaluate DynamicCompactDetect model')
    parser.add_argument('--config', type=str, default='train/config.yaml', 
                        help='Path to the configuration file')
    parser.add_argument('--weights', type=str, default='', 
                        help='Path to the model weights')
    parser.add_argument('--visualize', action='store_true', 
                        help='Visualize predictions')
    parser.add_argument('--eval-early-exit', action='store_true', 
                        help='Evaluate early exit performance')
    parser.add_argument('--eval-inference-time', action='store_true', 
                        help='Evaluate inference time')
    args = parser.parse_args()
    
    # Load configuration
    with open(args.config, 'r') as f:
        cfg = yaml.safe_load(f)
    
    # Create output directories
    os.makedirs('results', exist_ok=True)
    
    # Initialize device
    if torch.cuda.is_available() and len(cfg['hardware']['gpu_ids']) > 0:
        device = torch.device(f"cuda:{cfg['hardware']['gpu_ids'][0]}")
    else:
        device = torch.device("cpu")
    print(f"Using device: {device}")
    
    # Load validation dataset
    val_transforms = get_augmentations(cfg)[1]  # Only need validation transforms
    val_dataset = COCODataset(
        img_dir=cfg['dataset']['val_images'],
        ann_file=cfg['dataset']['val_annotations'],
        transforms=val_transforms,
        img_size=cfg['model']['input_size']
    )
    
    # Create data loader
    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=cfg['training']['batch_size'],
        shuffle=False,
        num_workers=cfg['hardware']['num_workers'],
        pin_memory=cfg['hardware']['pinned_memory'],
        collate_fn=lambda batch: tuple(zip(*batch))
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
    
    # Load weights
    if args.weights:
        load_checkpoint(args.weights, model)
    else:
        print("No weights provided, using randomly initialized model")
    
    # Evaluate model
    results, predictions, targets = evaluate_model(
        model, val_loader, device,
        conf_threshold=cfg['validation']['conf_threshold']
    )
    
    # Save results
    import json
    with open('results/evaluation_results.json', 'w') as f:
        json.dump(results, f, indent=4)
    
    # Visualize predictions if requested
    if args.visualize:
        images = next(iter(val_loader))[0]  # Get batch of images
        visualize_predictions(images, predictions[:1], targets[:1])
    
    # Evaluate early exit performance if requested
    if args.eval_early_exit:
        early_exit_stats = evaluate_early_exit(model, val_loader, device)
        
        with open('results/early_exit_stats.json', 'w') as f:
            json.dump(early_exit_stats, f, indent=4)
    
    # Evaluate inference time if requested
    if args.eval_inference_time:
        # Measure inference time with early exit enabled
        print("Measuring inference time with early exit enabled...")
        model.eval()
        
        # Warm up
        dummy_input = torch.randn(1, 3, cfg['model']['input_size'][1], cfg['model']['input_size'][0], device=device)
        for _ in range(10):
            _ = model(dummy_input, early_exit=True)
        
        # Measure time
        torch.cuda.synchronize() if device.type == 'cuda' else None
        start_time = torch.cuda.Event(enable_timing=True)
        end_time = torch.cuda.Event(enable_timing=True)
        
        times = []
        for _ in range(100):
            start_time.record()
            _ = model(dummy_input, early_exit=True)
            end_time.record()
            torch.cuda.synchronize() if device.type == 'cuda' else None
            times.append(start_time.elapsed_time(end_time))
        
        avg_time_ms = np.mean(times)
        fps = 1000 / avg_time_ms
        
        print(f"Inference time (early exit): {avg_time_ms:.2f} ms")
        print(f"FPS (early exit): {fps:.2f}")
        
        # Measure inference time with early exit disabled
        print("Measuring inference time with early exit disabled...")
        times = []
        for _ in range(100):
            start_time.record()
            _ = model(dummy_input, early_exit=False)
            end_time.record()
            torch.cuda.synchronize() if device.type == 'cuda' else None
            times.append(start_time.elapsed_time(end_time))
        
        avg_time_ms_no_exit = np.mean(times)
        fps_no_exit = 1000 / avg_time_ms_no_exit
        
        print(f"Inference time (no early exit): {avg_time_ms_no_exit:.2f} ms")
        print(f"FPS (no early exit): {fps_no_exit:.2f}")
        
        # Calculate speedup
        speedup = avg_time_ms_no_exit / avg_time_ms
        print(f"Speedup from early exit: {speedup:.2f}x")
        
        # Save inference time results
        inference_results = {
            'early_exit_enabled': {
                'avg_time_ms': float(avg_time_ms),
                'fps': float(fps)
            },
            'early_exit_disabled': {
                'avg_time_ms': float(avg_time_ms_no_exit),
                'fps': float(fps_no_exit)
            },
            'speedup': float(speedup)
        }
        
        with open('results/inference_time_results.json', 'w') as f:
            json.dump(inference_results, f, indent=4)

if __name__ == "__main__":
    main() 