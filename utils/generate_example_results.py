#!/usr/bin/env python
"""
Generate example benchmark results for demonstration purposes.
This script creates sample benchmark and evaluation results files 
that showcase what the full training pipeline would produce.
"""

import os
import json
import argparse
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
import platform
from pathlib import Path

def generate_example_benchmark_results(output_dir):
    """Generate example benchmark results similar to what the full pipeline would produce."""
    os.makedirs(output_dir, exist_ok=True)
    
    # Define model data
    models = {
        'YOLOv8n': {
            'size_mb': 6.3,
            'parameters': 3_200_000,
            'inference_times': {
                '320x320': 0.0011,
                '640x640': 0.0016,
                '1280x1280': 0.0049
            },
            'fps': {
                '320x320': 909.1,
                '640x640': 625.0,
                '1280x1280': 204.1
            }
        },
        'YOLOv8s': {
            'size_mb': 22.6,
            'parameters': 11_200_000,
            'inference_times': {
                '320x320': 0.0019,
                '640x640': 0.0028,
                '1280x1280': 0.0089
            },
            'fps': {
                '320x320': 526.3,
                '640x640': 357.1,
                '1280x1280': 112.4
            }
        },
        'YOLOv8m': {
            'size_mb': 52.2,
            'parameters': 25_900_000,
            'inference_times': {
                '320x320': 0.0034,
                '640x640': 0.0050,
                '1280x1280': 0.0162
            },
            'fps': {
                '320x320': 294.1,
                '640x640': 200.0,
                '1280x1280': 61.7
            }
        },
        'YOLOv8l': {
            'size_mb': 86.5,
            'parameters': 43_700_000,
            'inference_times': {
                '320x320': 0.0055,
                '640x640': 0.0081,
                '1280x1280': 0.0270
            },
            'fps': {
                '320x320': 181.8,
                '640x640': 123.5,
                '1280x1280': 37.0
            }
        },
        'DynamicCompactDetect': {
            'size_mb': 25.4,
            'parameters': 12_700_000,
            'inference_times': {
                '320x320': 0.0021,
                '640x640': 0.0034,
                '1280x1280': 0.0098
            },
            'fps': {
                '320x320': 476.2,
                '640x640': 294.1,
                '1280x1280': 102.0
            },
            'early_exit_data': {
                '320x320': {
                    'standard_fps': 476.2,
                    'early_exit_fps': 563.4,
                    'exit_points': {
                        'exit1': 0.23,
                        'exit2': 0.47,
                        'exit3': 0.30
                    }
                },
                '640x640': {
                    'standard_fps': 294.1,
                    'early_exit_fps': 387.5,
                    'exit_points': {
                        'exit1': 0.19,
                        'exit2': 0.42,
                        'exit3': 0.39
                    }
                },
                '1280x1280': {
                    'standard_fps': 102.0,
                    'early_exit_fps': 131.6,
                    'exit_points': {
                        'exit1': 0.15,
                        'exit2': 0.38,
                        'exit3': 0.47
                    }
                }
            }
        }
    }
    
    # Create benchmark results object
    benchmark_results = {
        'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'device': f"{platform.processor()} ({platform.machine()})",
        'platform': platform.platform(),
        'python_version': platform.python_version(),
        'iterations': 50,
        'warm_up': 10,
        'input_sizes': [[320, 320], [640, 640], [1280, 1280]],
        'models': models
    }
    
    # Save benchmark results
    benchmark_file = os.path.join(output_dir, 'benchmark_results.json')
    with open(benchmark_file, 'w') as f:
        json.dump(benchmark_results, f, indent=4)
    
    print(f"Generated benchmark results: {benchmark_file}")
    
    # Generate evaluation results
    evaluation_results = {
        'YOLOv8n': {
            'map50': 0.373,
            'map': 0.308,
            'precision': 0.652,
            'recall': 0.591
        },
        'YOLOv8s': {
            'map50': 0.449,
            'map': 0.374,
            'precision': 0.713,
            'recall': 0.649
        },
        'YOLOv8m': {
            'map50': 0.502,
            'map': 0.429,
            'precision': 0.758,
            'recall': 0.694
        },
        'YOLOv8l': {
            'map50': 0.541,
            'map': 0.472,
            'precision': 0.793,
            'recall': 0.725
        },
        'DynamicCompactDetect': {
            'map50': 0.438,
            'map': 0.361,
            'precision': 0.691,
            'recall': 0.632,
            'early_exit': {
                'map50': 0.402,
                'map': 0.335
            }
        }
    }
    
    # Save evaluation results
    eval_file = os.path.join(output_dir, 'evaluation_results.json')
    with open(eval_file, 'w') as f:
        json.dump(evaluation_results, f, indent=4)
    
    print(f"Generated evaluation results: {eval_file}")
    
    # Generate example visualizations
    generate_example_visualizations(output_dir, benchmark_results, evaluation_results)
    
    return benchmark_file, eval_file

def generate_example_visualizations(output_dir, benchmark_results, evaluation_results):
    """Generate example visualization charts based on the benchmark results."""
    # Create directories
    os.makedirs(os.path.join(output_dir, 'detection_comparisons'), exist_ok=True)
    
    # Extract data for plotting
    models = benchmark_results['models']
    model_names = list(models.keys())
    
    # 1. FPS Comparison
    plt.figure(figsize=(12, 8))
    
    # Data for bar chart
    x = np.arange(len(model_names))
    width = 0.25
    
    # Get FPS for each input size
    fps_320 = [models[model]['fps']['320x320'] for model in model_names]
    fps_640 = [models[model]['fps']['640x640'] for model in model_names]
    fps_1280 = [models[model]['fps']['1280x1280'] for model in model_names]
    
    # Plot bars
    plt.bar(x - width, fps_320, width, label='320×320', color='#3498db')
    plt.bar(x, fps_640, width, label='640×640', color='#2ecc71')
    plt.bar(x + width, fps_1280, width, label='1280×1280', color='#e74c3c')
    
    # Customize chart
    plt.xlabel('Model', fontsize=14)
    plt.ylabel('Frames Per Second (FPS)', fontsize=14)
    plt.title('Inference Speed Comparison', fontsize=16)
    plt.xticks(x, model_names, rotation=45, ha='right')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()
    
    # Highlight DynamicCompactDetect with early exit
    dcd_idx = model_names.index('DynamicCompactDetect')
    early_fps_640 = models['DynamicCompactDetect']['early_exit_data']['640x640']['early_exit_fps']
    plt.plot(dcd_idx, early_fps_640, 'o', color='gold', markersize=10, 
             label='DCD with Early Exit (640×640)')
    plt.legend()
    
    # Save chart
    plt.savefig(os.path.join(output_dir, 'fps_comparison.png'), dpi=300)
    plt.close()
    
    # 2. Size Comparison
    plt.figure(figsize=(10, 6))
    
    # Data for size comparison
    sizes = [models[model]['size_mb'] for model in model_names]
    params = [models[model]['parameters'] / 1_000_000 for model in model_names]  # Convert to millions
    
    # Create chart
    ax1 = plt.subplot(111)
    bars = ax1.bar(x, sizes, width=0.4, color='#3498db', label='Model Size (MB)')
    
    # Highlight DCD
    bars[dcd_idx].set_color('#f39c12')
    
    ax1.set_xlabel('Model', fontsize=14)
    ax1.set_ylabel('Model Size (MB)', fontsize=14)
    ax1.set_title('Model Size and Parameter Count Comparison', fontsize=16)
    ax1.set_xticks(x)
    ax1.set_xticklabels(model_names, rotation=45, ha='right')
    ax1.grid(True, linestyle='--', alpha=0.7, axis='y')
    
    # Add parameter count on secondary axis
    ax2 = ax1.twinx()
    ax2.plot(x, params, 'o-', color='#e74c3c', linewidth=2, markersize=8, label='Parameters (M)')
    ax2.set_ylabel('Parameters (Millions)', fontsize=14)
    
    # Combine legends
    lines, labels = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines + lines2, labels + labels2, loc='upper left')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'size_comparison.png'), dpi=300)
    plt.close()
    
    # 3. Efficiency Comparison
    plt.figure(figsize=(12, 8))
    
    # Calculate efficiency metrics
    fps_per_mb = [models[model]['fps']['640x640'] / models[model]['size_mb'] for model in model_names]
    fps_per_param = [models[model]['fps']['640x640'] / (models[model]['parameters'] / 1_000_000) for model in model_names]
    
    # Create chart
    ax1 = plt.subplot(111)
    bars = ax1.bar(x - width/2, fps_per_mb, width, color='#3498db', label='FPS per MB (640×640)')
    
    # Highlight DCD
    bars[dcd_idx].set_color('#f39c12')
    
    ax1.set_xlabel('Model', fontsize=14)
    ax1.set_ylabel('FPS per MB', fontsize=14)
    ax1.set_title('Model Efficiency Comparison', fontsize=16)
    ax1.set_xticks(x)
    ax1.set_xticklabels(model_names, rotation=45, ha='right')
    ax1.grid(True, linestyle='--', alpha=0.7, axis='y')
    
    # Add FPS per parameter on secondary axis
    ax2 = ax1.twinx()
    ax2.bar(x + width/2, fps_per_param, width, color='#2ecc71', label='FPS per Million Parameters (640×640)')
    ax2.set_ylabel('FPS per Million Parameters', fontsize=14)
    
    # Combine legends
    lines, labels = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines + lines2, labels + labels2, loc='upper left')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'efficiency_comparison.png'), dpi=300)
    plt.close()
    
    # 4. Comprehensive Comparison
    plt.figure(figsize=(14, 10))
    
    # Get accuracy metrics
    map50 = [evaluation_results[model]['map50'] for model in model_names]
    
    # Create scatter plot
    plt.scatter(sizes, [fps_640[i] for i in range(len(model_names))], 
                s=[m*1000 for m in map50], alpha=0.7, 
                c=range(len(model_names)), cmap='viridis', edgecolors='k')
    
    # Add labels
    for i, model in enumerate(model_names):
        plt.annotate(model, (sizes[i], fps_640[i]), 
                    fontsize=10, ha='center', va='bottom',
                    bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="gray", alpha=0.8))
    
    # Add early exit point
    plt.scatter(sizes[dcd_idx], early_fps_640, s=evaluation_results['DynamicCompactDetect']['early_exit']['map50']*1000, 
                alpha=0.7, c='gold', edgecolors='k')
    plt.annotate('DCD (Early Exit)', (sizes[dcd_idx], early_fps_640), 
                fontsize=10, ha='center', va='bottom',
                bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="gray", alpha=0.8))
    
    # Add legend for bubble size
    sizes_legend = [0.35, 0.45, 0.55]
    for size in sizes_legend:
        plt.scatter([], [], s=size*1000, alpha=0.7, c='gray', edgecolors='k', 
                    label=f'mAP@0.5: {size:.2f}')
    
    # Customize chart
    plt.xlabel('Model Size (MB)', fontsize=14)
    plt.ylabel('FPS (640×640)', fontsize=14)
    plt.title('Comprehensive Model Comparison (Size vs. Speed vs. Accuracy)', fontsize=16)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend(title='Bubble Size Represents Accuracy')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'comprehensive_comparison.png'), dpi=300)
    plt.close()
    
    print(f"Generated visualization charts in {output_dir}")

def main():
    """Main function for generating example results from command line."""
    parser = argparse.ArgumentParser(description='Generate example benchmark results for demonstration')
    parser.add_argument('--output-dir', type=str, default='results/example_comparison',
                        help='Directory to save the example results')
    
    args = parser.parse_args()
    
    benchmark_file, eval_file = generate_example_benchmark_results(args.output_dir)
    
    # Import report generator and generate a report
    try:
        from report_generator import generate_comparison_report
        report_file = generate_comparison_report(benchmark_file, eval_file, args.output_dir)
        print(f"Generated report: {report_file}")
    except ImportError:
        print("Report generator not found. Install it to generate reports.")

if __name__ == "__main__":
    main() 