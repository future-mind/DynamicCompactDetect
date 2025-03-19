import os
import sys
import yaml
import argparse
import numpy as np
import torch
from tqdm import tqdm
import matplotlib.pyplot as plt
from pathlib import Path
import json
import csv
import platform
import datetime

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.dynamic_compact_detect import DynamicCompactDetect
from utils.data_utils import COCODataset, create_data_loaders, get_augmentations
from utils.model_utils import load_checkpoint
from utils.benchmark_utils import (
    measure_inference_time, benchmark_model, plot_comparison_chart
)

def benchmark_on_multiple_inputs(model, input_sizes, device, iterations=100, warm_up=10):
    """Benchmark model on multiple input sizes."""
    results = {}
    
    for width, height in input_sizes:
        print(f"Benchmarking on input size: {width}x{height}")
        
        # Measure inference time with early exit enabled
        avg_time_ee, fps_ee = measure_inference_time(
            model, input_size=(width, height), 
            iterations=iterations, warm_up=warm_up, device=device
        )
        
        # Temporarily disable early exit
        model.eval()
        model_forward_orig = model.forward
        model.forward = lambda x, early_exit=False: model_forward_orig(x, early_exit=False)
        
        # Measure inference time with early exit disabled
        avg_time_no_ee, fps_no_ee = measure_inference_time(
            model, input_size=(width, height), 
            iterations=iterations, warm_up=warm_up, device=device
        )
        
        # Restore original forward
        model.forward = model_forward_orig
        
        # Calculate speedup
        speedup = avg_time_no_ee / avg_time_ee
        
        results[f"{width}x{height}"] = {
            'early_exit_enabled': {
                'avg_time_ms': avg_time_ee * 1000,
                'fps': fps_ee
            },
            'early_exit_disabled': {
                'avg_time_ms': avg_time_no_ee * 1000,
                'fps': fps_no_ee
            },
            'speedup': speedup
        }
    
    return results

def profile_memory_usage(model, input_sizes, device):
    """Profile memory usage for different input sizes."""
    results = {}
    
    for width, height in input_sizes:
        print(f"Profiling memory usage for input size: {width}x{height}")
        
        # Create dummy input
        dummy_input = torch.randn(1, 3, height, width, device=device)
        
        # Profile memory usage
        if device.type == 'cuda':
            # CUDA specific profiling
            torch.cuda.reset_peak_memory_stats(device)
            torch.cuda.empty_cache()
            
            # Warm-up
            for _ in range(5):
                _ = model(dummy_input)
            
            # Measure
            torch.cuda.reset_peak_memory_stats(device)
            _ = model(dummy_input)
            memory_usage = torch.cuda.max_memory_allocated(device) / (1024 * 1024)  # MB
        else:
            # CPU memory profiling is less precise
            import psutil
            
            process = psutil.Process(os.getpid())
            base_memory = process.memory_info().rss / (1024 * 1024)  # MB
            
            # Warm-up
            for _ in range(5):
                _ = model(dummy_input)
            
            # Measure
            _ = model(dummy_input)
            memory_usage = process.memory_info().rss / (1024 * 1024) - base_memory  # MB
        
        results[f"{width}x{height}"] = {
            'memory_usage_mb': float(memory_usage),
        }
    
    return results

def benchmark_model_variations(base_model, device, input_size=(640, 640), iterations=100):
    """Benchmark different variations of the model (e.g., with/without features)."""
    results = {}
    
    print("Benchmarking base model (all features enabled)...")
    avg_time_base, fps_base = measure_inference_time(
        base_model, input_size=input_size, 
        iterations=iterations, warm_up=10, device=device
    )
    
    results['base_model'] = {
        'avg_time_ms': avg_time_base * 1000,
        'fps': fps_base,
        'model_size_mb': sum(p.numel() for p in base_model.parameters()) * 4 / (1024 * 1024)
    }
    
    # Create model variations
    variations = {
        'no_dynamic_blocks': {
            'create_fn': lambda: DynamicCompactDetect(num_classes=80, base_channels=32),
            'setup_fn': lambda m: setattr(m.backbone, 'dynamic_blocks', False)
        },
        'no_early_exit': {
            'create_fn': lambda: DynamicCompactDetect(num_classes=80, base_channels=32),
            'setup_fn': lambda m: None
        }
    }
    
    for name, config in variations.items():
        print(f"Benchmarking {name}...")
        
        # Create and configure model variation
        model_var = config['create_fn']().to(device)
        if config['setup_fn']:
            config['setup_fn'](model_var)
        
        # Benchmark
        avg_time, fps = measure_inference_time(
            model_var, input_size=input_size, 
            iterations=iterations, warm_up=10, device=device
        )
        
        results[name] = {
            'avg_time_ms': avg_time * 1000,
            'fps': fps,
            'model_size_mb': sum(p.numel() for p in model_var.parameters()) * 4 / (1024 * 1024)
        }
    
    return results

def plot_benchmark_results(benchmark_results, save_dir='results/plots'):
    """Generate plots for benchmark results."""
    os.makedirs(save_dir, exist_ok=True)
    
    # Plot 1: Inference time comparison
    plt.figure(figsize=(10, 6))
    models = list(benchmark_results.keys())
    times = [benchmark_results[m]['avg_time_ms'] for m in models]
    
    plt.barh(models, times, color='skyblue')
    plt.xlabel('Inference Time (ms)')
    plt.title('Inference Time Comparison')
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'inference_time_comparison.png'))
    plt.close()
    
    # Plot 2: FPS comparison
    plt.figure(figsize=(10, 6))
    fps = [benchmark_results[m]['fps'] for m in models]
    
    plt.barh(models, fps, color='lightgreen')
    plt.xlabel('FPS')
    plt.title('Frames Per Second Comparison')
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'fps_comparison.png'))
    plt.close()
    
    # Plot 3: Model size comparison
    plt.figure(figsize=(10, 6))
    sizes = [benchmark_results[m]['model_size_mb'] for m in models]
    
    plt.barh(models, sizes, color='salmon')
    plt.xlabel('Model Size (MB)')
    plt.title('Model Size Comparison')
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'model_size_comparison.png'))
    plt.close()

def save_benchmark_results(results, filename='results/benchmark_results.json'):
    """Save benchmark results to a JSON file."""
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    
    # Add system info
    results['system_info'] = {
        'platform': platform.platform(),
        'processor': platform.processor(),
        'python_version': platform.python_version(),
        'torch_version': torch.__version__,
        'cuda_available': torch.cuda.is_available(),
        'cuda_version': torch.version.cuda if torch.cuda.is_available() else None,
        'device_count': torch.cuda.device_count() if torch.cuda.is_available() else 0,
        'device_name': torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        'timestamp': datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    }
    
    with open(filename, 'w') as f:
        json.dump(results, f, indent=4)
    
    print(f"Benchmark results saved to {filename}")

def export_results_to_csv(results, filename='results/benchmark_results.csv'):
    """Export benchmark results to a CSV file for easy comparison."""
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    
    # Extract metrics
    csv_data = []
    for model_name, metrics in results.items():
        if model_name == 'system_info':
            continue
            
        row = {'model': model_name}
        for metric_name, value in metrics.items():
            if isinstance(value, dict):
                for submetric_name, subvalue in value.items():
                    row[f"{metric_name}_{submetric_name}"] = subvalue
            else:
                row[metric_name] = value
        
        csv_data.append(row)
    
    # Write to CSV
    if csv_data:
        fieldnames = csv_data[0].keys()
        with open(filename, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(csv_data)
        
        print(f"CSV results exported to {filename}")

def main():
    parser = argparse.ArgumentParser(description='Benchmark DynamicCompactDetect model')
    parser.add_argument('--config', type=str, default='train/config.yaml', 
                        help='Path to the configuration file')
    parser.add_argument('--weights', type=str, default='', 
                        help='Path to the model weights')
    parser.add_argument('--input-sizes', nargs='+', default=['640x640'], 
                        help='Input sizes to benchmark in WxH format (e.g., 640x640)')
    parser.add_argument('--iterations', type=int, default=100, 
                        help='Number of iterations for timing')
    parser.add_argument('--variations', action='store_true', 
                        help='Benchmark different model variations')
    parser.add_argument('--profile-memory', action='store_true', 
                        help='Profile memory usage')
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
    
    # Parse input sizes
    input_sizes = []
    for size_str in args.input_sizes:
        width, height = map(int, size_str.split('x'))
        input_sizes.append((width, height))
    
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
    
    # Prepare benchmark
    model.eval()
    
    # Record all benchmark results
    all_results = {}
    
    # Benchmark on multiple input sizes
    print("Benchmarking on multiple input sizes...")
    input_size_results = benchmark_on_multiple_inputs(
        model, input_sizes, device, iterations=args.iterations
    )
    all_results['input_sizes'] = input_size_results
    
    # Benchmark model variations if requested
    if args.variations:
        print("Benchmarking model variations...")
        variation_results = benchmark_model_variations(
            model, device, input_size=input_sizes[0], iterations=args.iterations
        )
        all_results['variations'] = variation_results
    
    # Profile memory usage if requested
    if args.profile_memory:
        print("Profiling memory usage...")
        memory_results = profile_memory_usage(model, input_sizes, device)
        all_results['memory_usage'] = memory_results
    
    # Calculate model size and parameters
    model_size_mb = sum(p.numel() for p in model.parameters()) * 4 / (1024 * 1024)  # Size in MB
    num_params = sum(p.numel() for p in model.parameters())
    
    print(f"Model size: {model_size_mb:.2f} MB")
    print(f"Number of parameters: {num_params:,}")
    
    all_results['model_info'] = {
        'model_size_mb': float(model_size_mb),
        'num_parameters': int(num_params)
    }
    
    # Save and visualize results
    save_benchmark_results(all_results)
    export_results_to_csv(all_results)
    
    # Create plots
    if args.variations:
        plot_benchmark_results(variation_results)

if __name__ == "__main__":
    main() 