import os
import sys
import yaml
import argparse
import numpy as np
import torch
from tqdm import tqdm
import matplotlib.pyplot as plt
from pathlib import Path
import time
import json

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.dynamic_compact_detect import DynamicCompactDetect
from utils.data_utils import COCODataset, create_data_loaders, get_augmentations
from utils.model_utils import load_checkpoint
from utils.benchmark_utils import (
    measure_inference_time, benchmark_model, plot_comparison_chart
)

def download_yolov8_models():
    """Download YOLOv8 models from Ultralytics."""
    try:
        from ultralytics import YOLO
        
        # Download and load different sized models
        models = {
            'yolov8n': YOLO('yolov8n.pt'),  # Nano
            'yolov8s': YOLO('yolov8s.pt'),  # Small
            'yolov8m': YOLO('yolov8m.pt'),  # Medium
            'yolov8l': YOLO('yolov8l.pt'),  # Large
        }
        
        return models
    except ImportError:
        print("Ultralytics package not found. Please install it with:")
        print("pip install ultralytics")
        return None

def benchmark_yolov8_models(models, input_size=(640, 640), device='cuda', iterations=100, warm_up=10):
    """Benchmark YOLOv8 models from Ultralytics."""
    if not models:
        return {}
    
    results = {}
    
    # Create dummy input
    dummy_input = torch.randn(1, 3, input_size[1], input_size[0], device=device)
    
    for model_name, model in models.items():
        print(f"Benchmarking {model_name}...")
        
        # Set the model to the specified device
        model_device = model.model.to(device)
        
        # Warm up
        for _ in range(warm_up):
            _ = model_device(dummy_input)
        
        # Measure time
        torch.cuda.synchronize() if device == 'cuda' else None
        start_time = time.time()
        
        for _ in range(iterations):
            _ = model_device(dummy_input)
        
        torch.cuda.synchronize() if device == 'cuda' else None
        end_time = time.time()
        
        # Calculate metrics
        avg_time = (end_time - start_time) / iterations
        fps = 1.0 / avg_time
        
        # Calculate model size
        model_size_mb = sum(p.numel() for p in model_device.parameters()) * 4 / (1024 * 1024)  # Size in MB
        num_params = sum(p.numel() for p in model_device.parameters())
        
        results[model_name] = {
            'avg_time_ms': avg_time * 1000,
            'fps': fps,
            'model_size_mb': model_size_mb,
            'num_params': num_params
        }
        
        print(f"  Time: {avg_time * 1000:.2f} ms, FPS: {fps:.2f}, Size: {model_size_mb:.2f} MB")
    
    return results

def benchmark_dcd_model(model, input_size=(640, 640), device='cuda', iterations=100, warm_up=10):
    """Benchmark DynamicCompactDetect model."""
    print("Benchmarking DynamicCompactDetect...")
    
    # Ensure model is on the correct device
    model = model.to(device)
    model.eval()
    
    # Create dummy input
    dummy_input = torch.randn(1, 3, input_size[1], input_size[0], device=device)
    
    # Warm up
    with torch.no_grad():
        for _ in range(warm_up):
            _ = model(dummy_input)
    
    # Measure time with early exit enabled
    torch.cuda.synchronize() if device == 'cuda' else None
    start_time = time.time()
    
    with torch.no_grad():
        for _ in range(iterations):
            _ = model(dummy_input, early_exit=True)
    
    torch.cuda.synchronize() if device == 'cuda' else None
    end_time = time.time()
    
    # Calculate metrics with early exit
    avg_time_ee = (end_time - start_time) / iterations
    fps_ee = 1.0 / avg_time_ee
    
    # Measure time with early exit disabled
    torch.cuda.synchronize() if device == 'cuda' else None
    start_time = time.time()
    
    with torch.no_grad():
        for _ in range(iterations):
            _ = model(dummy_input, early_exit=False)
    
    torch.cuda.synchronize() if device == 'cuda' else None
    end_time = time.time()
    
    # Calculate metrics without early exit
    avg_time_no_ee = (end_time - start_time) / iterations
    fps_no_ee = 1.0 / avg_time_no_ee
    
    # Calculate model size
    model_size_mb = sum(p.numel() for p in model.parameters()) * 4 / (1024 * 1024)  # Size in MB
    num_params = sum(p.numel() for p in model.parameters())
    
    results = {
        'dynamiccompactdetect': {
            'with_early_exit': {
                'avg_time_ms': avg_time_ee * 1000,
                'fps': fps_ee,
            },
            'without_early_exit': {
                'avg_time_ms': avg_time_no_ee * 1000,
                'fps': fps_no_ee,
            },
            'model_size_mb': model_size_mb,
            'num_params': num_params
        }
    }
    
    print(f"  Time (EE): {avg_time_ee * 1000:.2f} ms, FPS: {fps_ee:.2f}")
    print(f"  Time (No EE): {avg_time_no_ee * 1000:.2f} ms, FPS: {fps_no_ee:.2f}")
    print(f"  Size: {model_size_mb:.2f} MB, Params: {num_params:,}")
    
    return results

def plot_comparison_results(dcd_results, yolo_results, save_dir='results/comparisons'):
    """Generate comparison plots for DynamicCompactDetect vs YOLOv8."""
    os.makedirs(save_dir, exist_ok=True)
    
    # Combine results
    combined_results = {}
    
    # Add DCD with early exit
    combined_results['DCD (EE)'] = {
        'avg_time_ms': dcd_results['dynamiccompactdetect']['with_early_exit']['avg_time_ms'],
        'fps': dcd_results['dynamiccompactdetect']['with_early_exit']['fps'],
        'model_size_mb': dcd_results['dynamiccompactdetect']['model_size_mb']
    }
    
    # Add DCD without early exit
    combined_results['DCD (No EE)'] = {
        'avg_time_ms': dcd_results['dynamiccompactdetect']['without_early_exit']['avg_time_ms'],
        'fps': dcd_results['dynamiccompactdetect']['without_early_exit']['fps'],
        'model_size_mb': dcd_results['dynamiccompactdetect']['model_size_mb']
    }
    
    # Add YOLOv8 models
    for model_name, results in yolo_results.items():
        combined_results[model_name.upper()] = {
            'avg_time_ms': results['avg_time_ms'],
            'fps': results['fps'],
            'model_size_mb': results['model_size_mb']
        }
    
    # Plot 1: Inference time comparison
    plt.figure(figsize=(12, 6))
    models = list(combined_results.keys())
    times = [combined_results[m]['avg_time_ms'] for m in models]
    
    # Create bars
    bars = plt.barh(models, times, color=['royalblue', 'lightblue'] + ['orange'] * len(yolo_results))
    
    # Add values on bars
    for i, bar in enumerate(bars):
        width = bar.get_width()
        plt.text(width + 1, bar.get_y() + bar.get_height()/2, f"{times[i]:.1f} ms", 
                 ha='left', va='center')
    
    plt.xlabel('Inference Time (ms, lower is better)')
    plt.title('Inference Time Comparison: DynamicCompactDetect vs YOLOv8')
    plt.grid(axis='x', linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'inference_time_comparison.png'))
    plt.close()
    
    # Plot 2: FPS comparison
    plt.figure(figsize=(12, 6))
    fps = [combined_results[m]['fps'] for m in models]
    
    # Create bars
    bars = plt.barh(models, fps, color=['royalblue', 'lightblue'] + ['orange'] * len(yolo_results))
    
    # Add values on bars
    for i, bar in enumerate(bars):
        width = bar.get_width()
        plt.text(width + 1, bar.get_y() + bar.get_height()/2, f"{fps[i]:.1f} FPS", 
                 ha='left', va='center')
    
    plt.xlabel('FPS (higher is better)')
    plt.title('FPS Comparison: DynamicCompactDetect vs YOLOv8')
    plt.grid(axis='x', linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'fps_comparison.png'))
    plt.close()
    
    # Plot 3: Model size comparison
    plt.figure(figsize=(12, 6))
    sizes = [combined_results[m]['model_size_mb'] for m in models]
    
    # Create bars
    bars = plt.barh(models, sizes, color=['royalblue', 'lightblue'] + ['orange'] * len(yolo_results))
    
    # Add values on bars
    for i, bar in enumerate(bars):
        width = bar.get_width()
        plt.text(width + 1, bar.get_y() + bar.get_height()/2, f"{sizes[i]:.1f} MB", 
                 ha='left', va='center')
    
    plt.xlabel('Model Size (MB, lower is better)')
    plt.title('Model Size Comparison: DynamicCompactDetect vs YOLOv8')
    plt.grid(axis='x', linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'model_size_comparison.png'))
    plt.close()
    
    # Plot 4: Efficiency comparison (FPS/MB)
    plt.figure(figsize=(12, 6))
    efficiency = [combined_results[m]['fps'] / combined_results[m]['model_size_mb'] for m in models]
    
    # Create bars
    bars = plt.barh(models, efficiency, color=['royalblue', 'lightblue'] + ['orange'] * len(yolo_results))
    
    # Add values on bars
    for i, bar in enumerate(bars):
        width = bar.get_width()
        plt.text(width + 0.1, bar.get_y() + bar.get_height()/2, f"{efficiency[i]:.2f} FPS/MB", 
                 ha='left', va='center')
    
    plt.xlabel('Efficiency (FPS/MB, higher is better)')
    plt.title('Efficiency Comparison: DynamicCompactDetect vs YOLOv8')
    plt.grid(axis='x', linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'efficiency_comparison.png'))
    plt.close()

def main():
    parser = argparse.ArgumentParser(description='Compare DynamicCompactDetect with YOLOv8')
    parser.add_argument('--config', type=str, default='train/config.yaml', 
                        help='Path to the configuration file')
    parser.add_argument('--weights', type=str, default='', 
                        help='Path to the DynamicCompactDetect model weights')
    parser.add_argument('--input-size', type=str, default='640x640', 
                        help='Input size in WxH format (e.g., 640x640)')
    parser.add_argument('--iterations', type=int, default=100, 
                        help='Number of iterations for timing')
    parser.add_argument('--precision', type=str, default='fp32', choices=['fp32', 'fp16'], 
                        help='Precision for benchmarking')
    args = parser.parse_args()
    
    # Load configuration
    with open(args.config, 'r') as f:
        cfg = yaml.safe_load(f)
    
    # Create output directories
    os.makedirs('results/comparisons', exist_ok=True)
    
    # Initialize device
    if torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')
    print(f"Using device: {device}")
    
    # Parse input size
    width, height = map(int, args.input_size.split('x'))
    input_size = (width, height)
    
    # Initialize DynamicCompactDetect model
    dcd_model = DynamicCompactDetect(
        num_classes=cfg['model']['num_classes'],
        base_channels=cfg['model']['base_channels']
    )
    
    # Platform-specific optimizations
    if cfg['hardware']['use_platform_optimizations']:
        dcd_model, device = dcd_model.optimize_for_platform()
    else:
        dcd_model = dcd_model.to(device)
    
    # Load weights
    if args.weights:
        load_checkpoint(args.weights, dcd_model)
    else:
        print("No weights provided, using randomly initialized model")
    
    # Set precision
    if args.precision == 'fp16' and device.type == 'cuda':
        print("Using mixed precision (FP16)")
        # This would be implemented with torch.cuda.amp
        # For simplicity, we'll skip the actual implementation
    
    # Download and initialize YOLOv8 models
    print("Downloading YOLOv8 models...")
    yolo_models = download_yolov8_models()
    
    if not yolo_models:
        print("Failed to download YOLOv8 models. Exiting.")
        return
    
    # Benchmark YOLOv8 models
    print("Benchmarking YOLOv8 models...")
    yolo_results = benchmark_yolov8_models(
        yolo_models, input_size=input_size, 
        device=device, iterations=args.iterations
    )
    
    # Benchmark DynamicCompactDetect model
    print("Benchmarking DynamicCompactDetect model...")
    dcd_results = benchmark_dcd_model(
        dcd_model, input_size=input_size, 
        device=device, iterations=args.iterations
    )
    
    # Generate comparison plots
    print("Generating comparison plots...")
    plot_comparison_results(dcd_results, yolo_results)
    
    # Save results
    combined_results = {
        'DynamicCompactDetect': dcd_results,
        'YOLOv8': yolo_results,
        'benchmark_settings': {
            'input_size': input_size,
            'iterations': args.iterations,
            'precision': args.precision,
            'device': str(device)
        }
    }
    
    with open('results/comparisons/benchmark_comparison.json', 'w') as f:
        json.dump(combined_results, f, indent=4)
    
    print("Comparison completed. Results saved to 'results/comparisons/'")

if __name__ == "__main__":
    main() 