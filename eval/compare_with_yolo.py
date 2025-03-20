import os
import sys
import yaml
import argparse
import torch
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import json
import time
import platform
from tqdm import tqdm

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.dynamic_compact_detect import DynamicCompactDetect
from utils.data_utils import COCODataset, create_data_loaders, get_augmentations
from utils.model_utils import load_checkpoint
from utils.benchmark_utils import measure_inference_time, plot_comparison_chart
from utils.visualization import visualize_detections, draw_multiple_detections

try:
    import ultralytics
    from ultralytics import YOLO
    ULTRALYTICS_AVAILABLE = True
except ImportError:
    ULTRALYTICS_AVAILABLE = False
    print("Ultralytics not installed. Only DynamicCompactDetect will be evaluated.")

def load_models(dcd_weights, input_size=(640, 640), device='cpu'):
    """Load DynamicCompactDetect and YOLO models for comparison."""
    models = {}
    
    # Load DynamicCompactDetect
    print("Loading DynamicCompactDetect...")
    dcd_model = DynamicCompactDetect(num_classes=80, base_channels=32).to(device)
    if dcd_weights:
        load_checkpoint(dcd_weights, dcd_model)
    
    models['DynamicCompactDetect'] = dcd_model
    
    # Load YOLO models if available
    if ULTRALYTICS_AVAILABLE:
        print("Loading YOLOv8 models...")
        yolo_models = ['yolov8n', 'yolov8s', 'yolov8m', 'yolov8l']
        
        for model_name in yolo_models:
            try:
                model = YOLO(f"{model_name}.pt")
                models[model_name] = model
                print(f"Loaded {model_name}")
            except Exception as e:
                print(f"Error loading {model_name}: {e}")
    
    return models

def benchmark_models(models, input_sizes, device, iterations=100, warm_up=10):
    """Benchmark all models on different input sizes."""
    results = {}
    
    for model_name, model in models.items():
        print(f"Benchmarking {model_name}...")
        model_results = {}
        
        for width, height in input_sizes:
            print(f"  Input size: {width}x{height}")
            
            if model_name.startswith('yolov'):
                # YOLO model uses a different benchmarking approach
                dummy_input = torch.randn(1, 3, height, width).to(device)
                
                # Warm up
                for _ in range(warm_up):
                    if device.type == 'cuda':
                        torch.cuda.synchronize()
                    _ = model(dummy_input, verbose=False)
                    if device.type == 'cuda':
                        torch.cuda.synchronize()
                
                # Benchmark
                start_time = time.time()
                for _ in range(iterations):
                    if device.type == 'cuda':
                        torch.cuda.synchronize()
                    _ = model(dummy_input, verbose=False)
                    if device.type == 'cuda':
                        torch.cuda.synchronize()
                
                elapsed = time.time() - start_time
                avg_time = elapsed / iterations
                fps = 1 / avg_time
            else:
                # DynamicCompactDetect model
                # With early exit
                avg_time_ee, fps_ee = measure_inference_time(
                    model, input_size=(width, height), 
                    iterations=iterations, warm_up=warm_up, device=device,
                    early_exit=True
                )
                
                # Without early exit
                avg_time_no_ee, fps_no_ee = measure_inference_time(
                    model, input_size=(width, height), 
                    iterations=iterations, warm_up=warm_up, device=device,
                    early_exit=False
                )
                
                # Use early exit times as default for comparison
                avg_time = avg_time_ee
                fps = fps_ee
                
                # Store additional DCD-specific metrics
                model_results[f"{width}x{height}_early_exit"] = {
                    'avg_time_ms': avg_time_ee * 1000,
                    'fps': fps_ee
                }
                
                model_results[f"{width}x{height}_no_early_exit"] = {
                    'avg_time_ms': avg_time_no_ee * 1000,
                    'fps': fps_no_ee,
                    'speedup': avg_time_no_ee / avg_time_ee if avg_time_ee > 0 else 1.0
                }
            
            model_results[f"{width}x{height}"] = {
                'avg_time_ms': avg_time * 1000,
                'fps': fps
            }
        
        # Calculate model size
        if model_name.startswith('yolov'):
            # For YOLO models, get size from file
            model_file = f"{model_name}.pt"
            if os.path.exists(model_file):
                model_size_mb = os.path.getsize(model_file) / (1024 * 1024)
            else:
                model_size_mb = 0  # Unknown
                
            # Get parameters count from YOLO model info
            try:
                model_info = model.info()
                params_count = model_info.get('parameters', 0)
            except:
                params_count = 0  # Unknown
        else:
            # For DynamicCompactDetect, calculate directly
            params_count = sum(p.numel() for p in model.parameters())
            model_size_mb = params_count * 4 / (1024 * 1024)  # Approximate size in MB
        
        model_results['model_size_mb'] = model_size_mb
        model_results['parameters'] = params_count
        
        results[model_name] = model_results
    
    return results

def evaluate_models_coco(models, val_loader, device, conf_threshold=0.25, iou_threshold=0.45):
    """Evaluate all models on COCO validation dataset."""
    results = {}
    
    for model_name, model in models.items():
        print(f"Evaluating {model_name} on COCO validation set...")
        
        if model_name.startswith('yolov'):
            # YOLO evaluation
            try:
                metrics = model.val(data='coco.yaml', verbose=False)
                map50 = metrics.box.map50
                map = metrics.box.map
                
                results[model_name] = {
                    'mAP@0.5': map50,
                    'mAP@0.5:0.95': map
                }
            except Exception as e:
                print(f"Error evaluating {model_name}: {e}")
                results[model_name] = {
                    'mAP@0.5': 0,
                    'mAP@0.5:0.95': 0
                }
        else:
            # DynamicCompactDetect evaluation with and without early exit
            model.eval()
            
            # Evaluate with early exit
            with torch.no_grad():
                all_detections_ee = []
                all_targets = []
                
                for images, targets in tqdm(val_loader, desc=f"Evaluating {model_name} (with early exit)"):
                    images = images.to(device)
                    results = model(images, early_exit=True)
                    
                    # Process results and targets
                    all_detections_ee.extend(results)
                    all_targets.extend(targets)
            
            # Calculate mAP
            from utils.metrics import calculate_map
            map50_ee, map_ee = calculate_map(all_detections_ee, all_targets, iou_threshold=iou_threshold)
            
            # Evaluate without early exit
            with torch.no_grad():
                all_detections_no_ee = []
                
                for images, _ in tqdm(val_loader, desc=f"Evaluating {model_name} (without early exit)"):
                    images = images.to(device)
                    results = model(images, early_exit=False)
                    all_detections_no_ee.extend(results)
            
            # Calculate mAP
            map50_no_ee, map_no_ee = calculate_map(all_detections_no_ee, all_targets, iou_threshold=iou_threshold)
            
            results[model_name] = {
                'mAP@0.5_early_exit': map50_ee,
                'mAP@0.5:0.95_early_exit': map_ee,
                'mAP@0.5_no_early_exit': map50_no_ee,
                'mAP@0.5:0.95_no_early_exit': map_no_ee
            }
    
    return results

def compare_detections_on_samples(models, val_dataset, device, num_samples=10, conf_threshold=0.25):
    """Compare model detections on sample images from the validation set."""
    comparison_results = []
    
    # Randomly sample images
    indices = np.random.choice(len(val_dataset), min(num_samples, len(val_dataset)), replace=False)
    
    for idx in indices:
        image, target = val_dataset[idx]
        image_tensor = torch.unsqueeze(image, 0).to(device)
        
        sample_result = {
            'image': image.permute(1, 2, 0).cpu().numpy(),
            'target': target,
            'detections': {}
        }
        
        # Get detections from each model
        for model_name, model in models.items():
            print(f"Getting detections from {model_name} on sample {idx}...")
            
            if model_name.startswith('yolov'):
                # YOLO model
                try:
                    results = model(image_tensor, verbose=False)
                    boxes = results[0].boxes
                    
                    detections = {
                        'boxes': boxes.xyxy.cpu().numpy(),
                        'scores': boxes.conf.cpu().numpy(),
                        'labels': boxes.cls.cpu().numpy().astype(int)
                    }
                except Exception as e:
                    print(f"Error getting detections from {model_name}: {e}")
                    detections = {'boxes': [], 'scores': [], 'labels': []}
            else:
                # DynamicCompactDetect with early exit
                model.eval()
                with torch.no_grad():
                    detections_ee = model(image_tensor, early_exit=True)[0]
                    
                # DynamicCompactDetect without early exit
                with torch.no_grad():
                    detections_no_ee = model(image_tensor, early_exit=False)[0]
                
                sample_result['detections'][f"{model_name}_early_exit"] = detections_ee
                sample_result['detections'][f"{model_name}_no_early_exit"] = detections_no_ee
                
                # Use early exit detections as default for this model
                detections = detections_ee
            
            sample_result['detections'][model_name] = detections
        
        comparison_results.append(sample_result)
    
    return comparison_results

def visualize_comparisons(comparison_results, output_dir, class_names):
    """Create visualizations of model detection comparisons."""
    os.makedirs(output_dir, exist_ok=True)
    
    for i, result in enumerate(comparison_results):
        plt.figure(figsize=(20, 20))
        
        # Get a list of all model names
        model_names = list(result['detections'].keys())
        num_models = len(model_names)
        
        # Calculate grid dimensions
        grid_size = int(np.ceil(np.sqrt(num_models + 1)))  # +1 for ground truth
        
        # Plot ground truth
        plt.subplot(grid_size, grid_size, 1)
        plt.title("Ground Truth")
        plt.imshow(result['image'])
        
        if 'boxes' in result['target']:
            boxes = result['target']['boxes']
            labels = result['target']['labels']
            
            for box, label in zip(boxes, labels):
                x1, y1, x2, y2 = box
                rect = plt.Rectangle(
                    (x1, y1), x2 - x1, y2 - y1,
                    linewidth=2, edgecolor='green', facecolor='none'
                )
                plt.gca().add_patch(rect)
                plt.gca().text(
                    x1, y1, class_names[label],
                    bbox=dict(facecolor='green', alpha=0.5),
                    fontsize=8, color='white'
                )
        
        # Plot model detections
        for j, model_name in enumerate(model_names):
            plt.subplot(grid_size, grid_size, j + 2)
            plt.title(model_name)
            plt.imshow(result['image'])
            
            detections = result['detections'][model_name]
            
            if 'boxes' in detections and len(detections['boxes']) > 0:
                boxes = detections['boxes']
                scores = detections['scores']
                labels = detections['labels']
                
                for box, score, label in zip(boxes, scores, labels):
                    if score < 0.25:  # Confidence threshold
                        continue
                        
                    x1, y1, x2, y2 = box
                    rect = plt.Rectangle(
                        (x1, y1), x2 - x1, y2 - y1,
                        linewidth=2, edgecolor='blue', facecolor='none'
                    )
                    plt.gca().add_patch(rect)
                    
                    label_name = class_names[label] if label < len(class_names) else f"Class {label}"
                    plt.gca().text(
                        x1, y1, f"{label_name} {score:.2f}",
                        bbox=dict(facecolor='blue', alpha=0.5),
                        fontsize=8, color='white'
                    )
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f"comparison_{i}.png"))
        plt.close()
    
    print(f"Saved {len(comparison_results)} comparison visualizations to {output_dir}")

def plot_benchmark_results(benchmark_results, output_dir):
    """Create visualizations of benchmark results."""
    os.makedirs(output_dir, exist_ok=True)
    
    # Extract data for plotting
    model_names = list(benchmark_results.keys())
    
    # Model sizes (MB)
    model_sizes = [benchmark_results[model]['model_size_mb'] for model in model_names]
    
    # Extract inference times and FPS for 640x640 input size
    inference_times = []
    fps_values = []
    
    for model in model_names:
        if '640x640' in benchmark_results[model]:
            inference_times.append(benchmark_results[model]['640x640']['avg_time_ms'])
            fps_values.append(benchmark_results[model]['640x640']['fps'])
        else:
            # Use first available size if 640x640 not available
            for key in benchmark_results[model]:
                if 'x' in key and not key.endswith('early_exit') and not key.endswith('no_early_exit'):
                    inference_times.append(benchmark_results[model][key]['avg_time_ms'])
                    fps_values.append(benchmark_results[model][key]['fps'])
                    break
    
    # Parameter counts (millions)
    param_counts = [benchmark_results[model]['parameters'] / 1_000_000 for model in model_names]
    
    # Plot 1: Model Size vs Inference Time
    plt.figure(figsize=(10, 7))
    plt.scatter(model_sizes, inference_times, s=100)
    
    for i, model in enumerate(model_names):
        plt.annotate(model, (model_sizes[i], inference_times[i]), fontsize=10)
    
    plt.xlabel('Model Size (MB)')
    plt.ylabel('Inference Time (ms)')
    plt.title('Model Size vs Inference Time')
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.savefig(os.path.join(output_dir, 'size_vs_time.png'))
    plt.close()
    
    # Plot 2: Model Size vs FPS
    plt.figure(figsize=(10, 7))
    plt.scatter(model_sizes, fps_values, s=100)
    
    for i, model in enumerate(model_names):
        plt.annotate(model, (model_sizes[i], fps_values[i]), fontsize=10)
    
    plt.xlabel('Model Size (MB)')
    plt.ylabel('FPS')
    plt.title('Model Size vs FPS')
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.savefig(os.path.join(output_dir, 'size_vs_fps.png'))
    plt.close()
    
    # Plot 3: Parameters vs Inference Time
    plt.figure(figsize=(10, 7))
    plt.scatter(param_counts, inference_times, s=100)
    
    for i, model in enumerate(model_names):
        plt.annotate(model, (param_counts[i], inference_times[i]), fontsize=10)
    
    plt.xlabel('Parameters (millions)')
    plt.ylabel('Inference Time (ms)')
    plt.title('Parameters vs Inference Time')
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.savefig(os.path.join(output_dir, 'params_vs_time.png'))
    plt.close()
    
    # Plot 4: Model Efficiency (FPS/MB)
    efficiency = [fps / size if size > 0 else 0 for fps, size in zip(fps_values, model_sizes)]
    
    plt.figure(figsize=(10, 7))
    bars = plt.bar(model_names, efficiency)
    
    # Add value labels on bars
    for bar, value in zip(bars, efficiency):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1,
                f'{value:.2f}', ha='center', fontsize=10)
    
    plt.xlabel('Model')
    plt.ylabel('Efficiency (FPS/MB)')
    plt.title('Model Efficiency (FPS/MB)')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'efficiency_comparison.png'))
    plt.close()
    
    # Plot 5: Comprehensive comparison
    plot_comparison_chart(
        model_names, 
        model_sizes, 
        param_counts, 
        inference_times, 
        fps_values,
        os.path.join(output_dir, 'comprehensive_comparison.png')
    )
    
    print(f"Saved benchmark visualizations to {output_dir}")

def create_comparison_report(benchmark_results, evaluation_results, output_dir):
    """Create a comprehensive report combining benchmark and evaluation results."""
    os.makedirs(output_dir, exist_ok=True)
    
    # Combine results
    comprehensive_results = {}
    
    for model in benchmark_results:
        comprehensive_results[model] = {
            'benchmark': benchmark_results[model]
        }
        
        if model in evaluation_results:
            comprehensive_results[model]['evaluation'] = evaluation_results[model]
    
    # Save as JSON
    with open(os.path.join(output_dir, 'comparison_results.json'), 'w') as f:
        json.dump(comprehensive_results, f, indent=4)
    
    # Create markdown report
    with open(os.path.join(output_dir, 'comparison_report.md'), 'w') as f:
        f.write("# DynamicCompactDetect vs YOLO Models Comparison\n\n")
        
        # System information
        f.write("## System Information\n\n")
        f.write(f"- Platform: {platform.platform()}\n")
        f.write(f"- Processor: {platform.processor()}\n")
        f.write(f"- Python: {platform.python_version()}\n")
        f.write(f"- PyTorch: {torch.__version__}\n")
        f.write(f"- CUDA Available: {torch.cuda.is_available()}\n")
        if torch.cuda.is_available():
            f.write(f"- CUDA Version: {torch.version.cuda}\n")
            f.write(f"- GPU: {torch.cuda.get_device_name(0)}\n")
        f.write("\n")
        
        # Summary table
        f.write("## Summary\n\n")
        f.write("| Model | Size (MB) | Parameters (M) | Inference Time (ms) | FPS | mAP@0.5 |\n")
        f.write("|-------|-----------|----------------|---------------------|-----|--------|\n")
        
        for model in benchmark_results:
            size = benchmark_results[model]['model_size_mb']
            params = benchmark_results[model]['parameters'] / 1_000_000
            
            # Get inference time and FPS for 640x640
            inf_time = 0
            fps = 0
            
            for key in benchmark_results[model]:
                if key == '640x640':
                    inf_time = benchmark_results[model][key]['avg_time_ms']
                    fps = benchmark_results[model][key]['fps']
                    break
            
            # Get mAP
            map50 = 0
            if model in evaluation_results:
                if 'mAP@0.5' in evaluation_results[model]:
                    map50 = evaluation_results[model]['mAP@0.5']
                elif 'mAP@0.5_early_exit' in evaluation_results[model]:
                    map50 = evaluation_results[model]['mAP@0.5_early_exit']
            
            f.write(f"| {model} | {size:.2f} | {params:.2f} | {inf_time:.2f} | {fps:.2f} | {map50:.4f} |\n")
        
        f.write("\n")
        
        # Benchmark details
        f.write("## Benchmark Details\n\n")
        
        for model in benchmark_results:
            f.write(f"### {model}\n\n")
            f.write(f"- Model Size: {benchmark_results[model]['model_size_mb']:.2f} MB\n")
            f.write(f"- Parameters: {benchmark_results[model]['parameters']:,}\n")
            
            # Print results for different input sizes
            f.write("\n**Performance on different input sizes:**\n\n")
            f.write("| Input Size | Inference Time (ms) | FPS |\n")
            f.write("|------------|---------------------|-----|\n")
            
            for key in benchmark_results[model]:
                if 'x' in key and not key.endswith('early_exit') and not key.endswith('no_early_exit'):
                    inf_time = benchmark_results[model][key]['avg_time_ms']
                    fps = benchmark_results[model][key]['fps']
                    f.write(f"| {key} | {inf_time:.2f} | {fps:.2f} |\n")
            
            f.write("\n")
            
            # For DynamicCompactDetect, show early exit vs no early exit
            if model == 'DynamicCompactDetect':
                f.write("\n**Early Exit Performance:**\n\n")
                f.write("| Input Size | Mode | Inference Time (ms) | FPS | Speedup |\n")
                f.write("|------------|------|---------------------|-----|--------|\n")
                
                for key in benchmark_results[model]:
                    if key.endswith('_early_exit'):
                        size = key.replace('_early_exit', '')
                        
                        ee_time = benchmark_results[model][key]['avg_time_ms']
                        ee_fps = benchmark_results[model][key]['fps']
                        
                        no_ee_key = f"{size}_no_early_exit"
                        if no_ee_key in benchmark_results[model]:
                            no_ee_time = benchmark_results[model][no_ee_key]['avg_time_ms']
                            no_ee_fps = benchmark_results[model][no_ee_key]['fps']
                            speedup = benchmark_results[model][no_ee_key]['speedup']
                            
                            f.write(f"| {size} | With Early Exit | {ee_time:.2f} | {ee_fps:.2f} | - |\n")
                            f.write(f"| {size} | Without Early Exit | {no_ee_time:.2f} | {no_ee_fps:.2f} | {speedup:.2f}x |\n")
                
                f.write("\n")
        
        # Evaluation details
        if evaluation_results:
            f.write("## Evaluation Results\n\n")
            
            for model in evaluation_results:
                f.write(f"### {model}\n\n")
                
                if 'mAP@0.5' in evaluation_results[model]:
                    # Standard model
                    map50 = evaluation_results[model]['mAP@0.5']
                    map = evaluation_results[model]['mAP@0.5:0.95']
                    
                    f.write(f"- mAP@0.5: {map50:.4f}\n")
                    f.write(f"- mAP@0.5:0.95: {map:.4f}\n")
                else:
                    # DynamicCompactDetect with early exit modes
                    map50_ee = evaluation_results[model]['mAP@0.5_early_exit']
                    map_ee = evaluation_results[model]['mAP@0.5:0.95_early_exit']
                    map50_no_ee = evaluation_results[model]['mAP@0.5_no_early_exit']
                    map_no_ee = evaluation_results[model]['mAP@0.5:0.95_no_early_exit']
                    
                    f.write(f"- With Early Exit:\n")
                    f.write(f"  - mAP@0.5: {map50_ee:.4f}\n")
                    f.write(f"  - mAP@0.5:0.95: {map_ee:.4f}\n\n")
                    f.write(f"- Without Early Exit:\n")
                    f.write(f"  - mAP@0.5: {map50_no_ee:.4f}\n")
                    f.write(f"  - mAP@0.5:0.95: {map_no_ee:.4f}\n")
                
                f.write("\n")
        
        # Visualizations
        f.write("## Visualizations\n\n")
        f.write("### Benchmark Results\n\n")
        f.write("![Size vs Time](size_vs_time.png)\n\n")
        f.write("![Size vs FPS](size_vs_fps.png)\n\n")
        f.write("![Parameters vs Time](params_vs_time.png)\n\n")
        f.write("![Efficiency Comparison](efficiency_comparison.png)\n\n")
        f.write("![Comprehensive Comparison](comprehensive_comparison.png)\n\n")
        
        f.write("### Detection Comparisons\n\n")
        f.write("Several sample images from the validation set were processed by all models to compare detection results.\n")
        f.write("Detection comparison images are available in the `detection_comparisons/` directory.\n\n")
    
    print(f"Saved comparison report to {output_dir}/comparison_report.md")

def main():
    parser = argparse.ArgumentParser(description='Compare DynamicCompactDetect with YOLO models')
    parser.add_argument('--config', type=str, default='train/config.yaml', 
                        help='Path to the configuration file')
    parser.add_argument('--dcd-weights', type=str, default='',
                        help='Path to DynamicCompactDetect weights')
    parser.add_argument('--input-sizes', nargs='+', default=['320x320', '640x640', '1280x1280'],
                        help='Input sizes to benchmark in WxH format')
    parser.add_argument('--output-dir', type=str, default='results/comparisons',
                        help='Directory to save comparison results')
    parser.add_argument('--iterations', type=int, default=100,
                        help='Number of iterations for benchmarking')
    parser.add_argument('--num-samples', type=int, default=10,
                        help='Number of samples for detection comparison')
    parser.add_argument('--eval', action='store_true',
                        help='Run evaluation on COCO validation set')
    parser.add_argument('--benchmark-only', action='store_true',
                        help='Run only benchmarking, no evaluation or visual comparison')
    args = parser.parse_args()
    
    # Load configuration
    with open(args.config, 'r') as f:
        cfg = yaml.safe_load(f)
    
    # Create output directories
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Initialize device
    if torch.cuda.is_available() and len(cfg['hardware']['gpu_ids']) > 0:
        device = torch.device(f"cuda:{cfg['hardware']['gpu_ids'][0]}")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    print(f"Using device: {device}")
    
    # Parse input sizes
    input_sizes = []
    for size_str in args.input_sizes:
        width, height = map(int, size_str.split('x'))
        input_sizes.append((width, height))
    
    # Load models
    models = load_models(args.dcd_weights, input_size=input_sizes[0], device=device)
    
    # Benchmarking
    print("Running benchmarks...")
    benchmark_results = benchmark_models(
        models, input_sizes, device, 
        iterations=args.iterations, warm_up=10
    )
    
    # Save benchmark results
    benchmark_path = os.path.join(args.output_dir, 'benchmark_results.json')
    with open(benchmark_path, 'w') as f:
        json.dump(benchmark_results, f, indent=4)
    
    # Create benchmark visualizations
    plot_benchmark_results(benchmark_results, args.output_dir)
    
    if args.benchmark_only:
        print("Benchmark complete. Skipping evaluation and visual comparison.")
        create_comparison_report(benchmark_results, {}, args.output_dir)
        return
    
    # Evaluation and visual comparison if requested
    evaluation_results = {}
    
    if args.eval:
        # Create validation dataset for evaluation
        val_transforms, _ = get_augmentations(cfg)
        
        val_dataset = COCODataset(
            img_dir=cfg['dataset']['val_images'],
            ann_file=cfg['dataset']['val_annotations'],
            input_size=cfg['model']['input_size'],
            transforms=val_transforms
        )
        
        val_loader, _ = create_data_loaders(
            None, val_dataset, 
            batch_size=cfg['training']['batch_size'],
            num_workers=cfg['hardware']['num_workers'],
            pin_memory=cfg['hardware']['pinned_memory']
        )
        
        # Run evaluation
        print("Running evaluation...")
        evaluation_results = evaluate_models_coco(
            models, val_loader, device,
            conf_threshold=cfg['validation']['conf_threshold'],
            iou_threshold=cfg['validation']['iou_threshold']
        )
        
        # Save evaluation results
        eval_path = os.path.join(args.output_dir, 'evaluation_results.json')
        with open(eval_path, 'w') as f:
            json.dump(evaluation_results, f, indent=4)
        
        # Compare detections on samples
        print(f"Comparing detections on {args.num_samples} sample images...")
        comparison_results = compare_detections_on_samples(
            models, val_dataset, device, 
            num_samples=args.num_samples,
            conf_threshold=cfg['validation']['conf_threshold']
        )
        
        # Load class names from COCO
        class_names = []
        try:
            from pycocotools.coco import COCO
            coco = COCO(cfg['dataset']['val_annotations'])
            categories = coco.loadCats(coco.getCatIds())
            class_names = [cat['name'] for cat in categories]
        except:
            # Default class names if COCO API not available
            class_names = [f"Class {i}" for i in range(80)]
        
        # Create visualizations
        visualize_comparisons(
            comparison_results, 
            os.path.join(args.output_dir, 'detection_comparisons'),
            class_names
        )
    
    # Create comprehensive report
    create_comparison_report(benchmark_results, evaluation_results, args.output_dir)
    print(f"Comparison complete. Results saved to {args.output_dir}")

if __name__ == "__main__":
    main() 