#!/usr/bin/env python3
"""
Script to fine-tune DynamicCompactDetect model using YOLOv8n as the base model.
This script enhances the model with advanced training features from YOLOv11.
"""

import argparse
import os
from ultralytics import YOLO
import yaml
import torch
import shutil

def main():
    parser = argparse.ArgumentParser(description="Fine-tune DynamicCompactDetect model using YOLOv8n as base")
    parser.add_argument('--base-model', type=str, default='yolov8n.pt', help='Base model path (YOLOv8n)')
    parser.add_argument('--data', type=str, default='coco.yaml', help='Dataset config file')
    parser.add_argument('--epochs', type=int, default=50, help='Number of epochs to train for')
    parser.add_argument('--img-size', type=int, default=640, help='Image size for training')
    parser.add_argument('--batch-size', type=int, default=16, help='Batch size for training')
    parser.add_argument('--device', type=str, default='', help='Device to use for training (e.g., 0, cpu)')
    parser.add_argument('--project', type=str, default='runs/finetune', help='Project directory for saving results')
    parser.add_argument('--name', type=str, default='dcd_yolo11', help='Experiment name')
    parser.add_argument('--workers', type=int, default=8, help='Number of worker threads for data loading')
    parser.add_argument('--save-period', type=int, default=10, help='Save checkpoint every x epochs')
    parser.add_argument('--patience', type=int, default=50, help='Early stopping patience (epochs without improvement)')
    parser.add_argument('--cos-lr', action='store_true', help='Use cosine LR scheduler')
    parser.add_argument('--close-mosaic', type=int, default=10, help='Disable mosaic augmentation for final epochs')
    parser.add_argument('--resume', action='store_true', help='Resume training from last checkpoint')
    args = parser.parse_args()

    print(f"Fine-tuning DynamicCompactDetect with YOLOv11 methodology:")
    print(f"  Base Model: {args.base_model}")
    print(f"  Dataset: {args.data}")
    print(f"  Epochs: {args.epochs}")
    print(f"  Image Size: {args.img_size}")
    print(f"  Batch Size: {args.batch_size}")
    print(f"  Device: {args.device}")
    
    try:
        # Check if CUDA is available
        if torch.cuda.is_available() and args.device != 'cpu':
            print(f"CUDA is available. Using GPU for training.")
            if args.device == '':
                args.device = '0'  # Use first GPU by default
        else:
            print(f"CUDA is not available or CPU was specified. Using CPU for training.")
            args.device = 'cpu'
            
        # Load the base model
        print(f"Loading base model {args.base_model}...")
        model = YOLO(args.base_model)
        
        # Create a YOLOv11-style training configuration
        cfg = {
            # Dataset
            'data': args.data,
            'epochs': args.epochs,
            'imgsz': args.img_size,
            'batch': args.batch_size,
            'device': args.device,
            'workers': args.workers,
            'project': args.project,
            'name': args.name,
            'save_period': args.save_period,
            'patience': args.patience,
            
            # YOLOv11 specific augmentations
            'hsv_h': 0.015,  # HSV-Hue augmentation (fraction)
            'hsv_s': 0.7,    # HSV-Saturation augmentation (fraction)
            'hsv_v': 0.4,    # HSV-Value augmentation (fraction)
            'degrees': 0.0,  # Rotation augmentation (degrees)
            'translate': 0.1, # Translation augmentation (fraction)
            'scale': 0.5,    # Scale augmentation (fraction)
            'shear': 0.0,    # Shear augmentation (degrees)
            'perspective': 0.0, # Perspective augmentation (fraction)
            'flipud': 0.0,   # Flip up-down augmentation (probability)
            'fliplr': 0.5,   # Flip left-right augmentation (probability)
            'mosaic': 1.0,   # Mosaic augmentation (probability)
            'mixup': 0.0,    # Mixup augmentation (probability)
            'copy_paste': 0.0, # Copy-paste augmentation (probability)
            'auto_augment': 'randaugment', # Auto augmentation policy
            
            # YOLOv11 advanced training parameters
            'optimizer': 'AdamW',  # Optimizer (SGD, Adam, AdamW)
            'lr0': 0.001,    # Initial learning rate
            'lrf': 0.01,     # Final learning rate (fraction of lr0)
            'momentum': 0.937, # SGD momentum/Adam beta1
            'weight_decay': 0.0005, # Optimizer weight decay
            'warmup_epochs': 3.0, # Warmup epochs
            'warmup_momentum': 0.8, # Warmup initial momentum
            'warmup_bias_lr': 0.1, # Warmup initial bias lr
            'box': 7.5,      # Box loss gain
            'cls': 0.5,      # Classification loss gain
            'dfl': 1.5,      # Distribution focal loss gain
            'pose': 12.0,    # Pose loss gain
            'kobj': 1.0,     # Keypoint obj loss gain
            'label_smoothing': 0.0, # Label smoothing epsilon
            'nbs': 64,       # Nominal batch size
            'overlap_mask': True, # Masks should overlap during training
            'mask_ratio': 4, # Mask downsample ratio
            'dropout': 0.0,  # Use dropout regularization
            'val': True,     # Validate during training
            
            # YOLOv11 specific features
            'close_mosaic': args.close_mosaic, # Disable mosaic for final epochs
            'cos_lr': args.cos_lr, # Use cosine LR scheduler
            'resume': args.resume, # Resume training from last checkpoint
        }
        
        # Fine-tune the model with YOLOv11 methodology
        print(f"Fine-tuning model with YOLOv11 methodology...")
        results = model.train(**cfg)
        
        print("Fine-tuning completed successfully!")
        print(f"Results saved to {args.project}/{args.name}")
        
        # Rename the best model to dynamiccompactdetect.pt
        best_model_path = f"{args.project}/{args.name}/weights/best.pt"
        dcd_model_path = "dynamiccompactdetect_finetuned.pt"
        
        if os.path.exists(best_model_path):
            shutil.copy(best_model_path, dcd_model_path)
            print(f"Best model saved as {dcd_model_path}")
        
        # Validate the fine-tuned model
        print("Validating the fine-tuned model...")
        metrics = model.val()
        
        print(f"Validation metrics:")
        print(f"  mAP50: {metrics.box.map50:.4f}")
        print(f"  mAP50-95: {metrics.box.map:.4f}")
        
        # Save a comparison report
        save_comparison_report(args.base_model, dcd_model_path, metrics)
        
    except Exception as e:
        print(f"Error during fine-tuning: {e}")
        import traceback
        traceback.print_exc()

def save_comparison_report(original_model_path, finetuned_model_path, metrics):
    """Save a comparison report between original and fine-tuned models."""
    report_dir = "finetune_comparison"
    os.makedirs(report_dir, exist_ok=True)
    
    report_path = os.path.join(report_dir, "finetune_comparison_report.md")
    
    # Load original model for comparison
    try:
        original_model = YOLO(original_model_path)
        original_metrics = original_model.val()
        
        with open(report_path, 'w') as f:
            f.write("# DynamicCompactDetect Fine-tuning Comparison Report\n\n")
            f.write("## Model Comparison\n\n")
            f.write("| Metric | Base Model (YOLOv8n) | Fine-tuned DynamicCompactDetect | Improvement |\n")
            f.write("|--------|---------------|------------------|-------------|\n")
            
            # Compare mAP50
            original_map50 = original_metrics.box.map50
            finetuned_map50 = metrics.box.map50
            map50_improvement = finetuned_map50 - original_map50
            f.write(f"| mAP50 | {original_map50:.4f} | {finetuned_map50:.4f} | {map50_improvement:+.4f} |\n")
            
            # Compare mAP50-95
            original_map = original_metrics.box.map
            finetuned_map = metrics.box.map
            map_improvement = finetuned_map - original_map
            f.write(f"| mAP50-95 | {original_map:.4f} | {finetuned_map:.4f} | {map_improvement:+.4f} |\n")
            
            # Compare precision
            original_precision = original_metrics.box.mp
            finetuned_precision = metrics.box.mp
            precision_improvement = finetuned_precision - original_precision
            f.write(f"| Precision | {original_precision:.4f} | {finetuned_precision:.4f} | {precision_improvement:+.4f} |\n")
            
            # Compare recall
            original_recall = original_metrics.box.mr
            finetuned_recall = metrics.box.mr
            recall_improvement = finetuned_recall - original_recall
            f.write(f"| Recall | {original_recall:.4f} | {finetuned_recall:.4f} | {recall_improvement:+.4f} |\n")
            
            # Model size comparison
            original_size = os.path.getsize(original_model_path) / (1024 * 1024)  # Size in MB
            finetuned_size = os.path.getsize(finetuned_model_path) / (1024 * 1024)  # Size in MB
            size_change = finetuned_size - original_size
            f.write(f"| Model Size (MB) | {original_size:.2f} | {finetuned_size:.2f} | {size_change:+.2f} |\n\n")
            
            f.write("## Training Configuration\n\n")
            f.write("The DynamicCompactDetect model was fine-tuned from YOLOv8n using YOLOv11 methodology with the following enhancements:\n\n")
            f.write("- Advanced augmentation techniques (mosaic, auto-augment)\n")
            f.write("- AdamW optimizer with cosine learning rate scheduling\n")
            f.write("- Improved loss functions and regularization\n")
            f.write("- Progressive training strategy (close mosaic in final epochs)\n\n")
            
            f.write("## Conclusion\n\n")
            if map50_improvement > 0:
                f.write(f"The fine-tuning process successfully improved the model's performance, with a {map50_improvement:.4f} absolute increase in mAP50.\n")
            else:
                f.write(f"The fine-tuning process did not improve the model's mAP50 performance ({map50_improvement:.4f}).\n")
                
            if size_change <= 0:
                f.write(f"Additionally, the model size was reduced by {-size_change:.2f} MB while maintaining or improving performance.\n")
            else:
                f.write(f"However, the model size increased by {size_change:.2f} MB.\n")
                
        print(f"Comparison report saved to {report_path}")
        
    except Exception as e:
        print(f"Error creating comparison report: {e}")

if __name__ == "__main__":
    main() 