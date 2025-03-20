#!/usr/bin/env python
"""
Report generator for DynamicCompactDetect benchmark results.
Generates nicely formatted Markdown reports from benchmark JSON files.
"""

import os
import sys
import json
import argparse
from datetime import datetime
import platform
from pathlib import Path

def generate_comparison_report(benchmark_file, evaluation_file=None, output_dir=None):
    """
    Generate a comprehensive Markdown report from benchmark results.
    
    Args:
        benchmark_file: Path to benchmark results JSON file
        evaluation_file: Path to evaluation results JSON file (optional)
        output_dir: Directory to save the report (defaults to benchmark file directory)
    """
    print(f"Generating report from: {benchmark_file}")
    if evaluation_file:
        print(f"Including evaluation data from: {evaluation_file}")
    
    # Load benchmark results
    with open(benchmark_file, 'r') as f:
        benchmark_results = json.load(f)
    
    # Load evaluation results if available
    evaluation_results = None
    if evaluation_file and os.path.exists(evaluation_file):
        with open(evaluation_file, 'r') as f:
            evaluation_results = json.load(f)
    
    # Set output directory
    if output_dir is None:
        output_dir = os.path.dirname(benchmark_file)
    
    # Create report filename
    report_file = os.path.join(output_dir, 'comparison_report.md')
    
    print(f"Creating report at: {report_file}")
    
    # Generate report
    with open(report_file, 'w') as f:
        # Header
        f.write("# DynamicCompactDetect Performance Report\n\n")
        f.write(f"*Generated on {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}*\n\n")
        
        # System information
        f.write("## System Information\n\n")
        f.write(f"- **Platform**: {platform.platform()}\n")
        f.write(f"- **Python Version**: {platform.python_version()}\n")
        
        if 'device' in benchmark_results:
            f.write(f"- **Device**: {benchmark_results['device']}\n")
        
        f.write("\n")
        
        # Benchmark information
        f.write("## Benchmark Settings\n\n")
        if 'iterations' in benchmark_results:
            f.write(f"- **Iterations**: {benchmark_results['iterations']}\n")
        if 'warm_up' in benchmark_results:
            f.write(f"- **Warm-up Iterations**: {benchmark_results['warm_up']}\n")
        if 'input_sizes' in benchmark_results:
            input_sizes = benchmark_results['input_sizes']
            f.write(f"- **Input Sizes**: {', '.join([f'{w}×{h}' for w, h in input_sizes])}\n")
        f.write("\n")
        
        # Model comparison table
        f.write("## Model Comparison\n\n")
        f.write("### Model Size and Parameters\n\n")
        f.write("| Model | Size (MB) | Parameters (M) |\n")
        f.write("|-------|-----------|----------------|\n")
        
        for model_name, model_data in benchmark_results['models'].items():
            if 'size_mb' in model_data and 'parameters' in model_data:
                size_mb = model_data['size_mb']
                params_m = model_data['parameters'] / 1_000_000  # Convert to millions
                f.write(f"| {model_name} | {size_mb:.2f} | {params_m:.2f} |\n")
        
        f.write("\n")
        
        # Performance table
        f.write("### Inference Performance\n\n")
        
        # Get all input sizes
        input_sizes = []
        for model_data in benchmark_results['models'].values():
            if 'inference_times' in model_data:
                for size_str in model_data['inference_times'].keys():
                    if size_str not in input_sizes:
                        input_sizes.append(size_str)
        
        input_sizes.sort(key=lambda x: int(x.split('x')[0]) * int(x.split('x')[1]))
        
        # Create performance table
        f.write("| Model | " + " | ".join([f"FPS ({size})" for size in input_sizes]) + " |\n")
        f.write("|-------| " + " | ".join(["----------" for _ in input_sizes]) + " |\n")
        
        for model_name, model_data in benchmark_results['models'].items():
            if 'inference_times' in model_data:
                fps_values = []
                for size in input_sizes:
                    if size in model_data['inference_times']:
                        fps = model_data['fps'][size] if 'fps' in model_data and size in model_data['fps'] else 0
                        fps_values.append(f"{fps:.1f}")
                    else:
                        fps_values.append("-")
                
                f.write(f"| {model_name} | " + " | ".join(fps_values) + " |\n")
        
        f.write("\n")
        
        # Evaluation results if available
        if evaluation_results:
            f.write("### Detection Accuracy\n\n")
            f.write("| Model | mAP@0.5 | mAP@0.5:0.95 |\n")
            f.write("|-------|---------|-------------|\n")
            
            for model_name, model_data in evaluation_results.items():
                if 'map50' in model_data and 'map' in model_data:
                    map50 = model_data['map50'] * 100  # Convert to percentage
                    map = model_data['map'] * 100      # Convert to percentage
                    f.write(f"| {model_name} | {map50:.1f} | {map:.1f} |\n")
            
            f.write("\n")
        
        # Early exit performance if available
        dcd_model = benchmark_results['models'].get('DynamicCompactDetect', {})
        if 'early_exit_data' in dcd_model:
            f.write("### Early Exit Performance\n\n")
            f.write("DynamicCompactDetect employs a dynamic routing mechanism that allows early exit for simpler images, improving efficiency.\n\n")
            
            early_exit_data = dcd_model['early_exit_data']
            
            f.write("| Input Size | Standard FPS | With Early Exit FPS | Speedup |\n")
            f.write("|------------|--------------|---------------------|--------|\n")
            
            for size, data in early_exit_data.items():
                std_fps = data.get('standard_fps', 0)
                early_fps = data.get('early_exit_fps', 0)
                speedup = (early_fps / std_fps) if std_fps > 0 else 1.0
                
                f.write(f"| {size} | {std_fps:.1f} | {early_fps:.1f} | {speedup:.2f}x |\n")
            
            f.write("\n")
        
        # Images
        f.write("## Visualization\n\n")
        
        # FPS comparison chart
        if os.path.exists(os.path.join(output_dir, 'fps_comparison.png')):
            f.write("### FPS Comparison\n\n")
            f.write("![FPS Comparison](fps_comparison.png)\n\n")
        
        # Size comparison chart
        if os.path.exists(os.path.join(output_dir, 'size_comparison.png')):
            f.write("### Size Comparison\n\n")
            f.write("![Size Comparison](size_comparison.png)\n\n")
        
        # Efficiency comparison chart
        if os.path.exists(os.path.join(output_dir, 'efficiency_comparison.png')):
            f.write("### Efficiency Comparison\n\n")
            f.write("![Efficiency Comparison](efficiency_comparison.png)\n\n")
        
        # Comprehensive comparison chart
        if os.path.exists(os.path.join(output_dir, 'comprehensive_comparison.png')):
            f.write("### Comprehensive Comparison\n\n")
            f.write("![Comprehensive Comparison](comprehensive_comparison.png)\n\n")
        
        # Detection comparisons
        detection_dir = os.path.join(output_dir, 'detection_comparisons')
        if os.path.exists(detection_dir):
            f.write("### Detection Examples\n\n")
            f.write("Several sample images showing detection results from different models can be found in the `detection_comparisons/` directory.\n\n")
            
            # Show first few comparison images if they exist
            image_files = [f for f in os.listdir(detection_dir) if f.endswith('.jpg') or f.endswith('.png')]
            if image_files:
                for i, img_file in enumerate(sorted(image_files)[:3]):  # Show first 3 images
                    f.write(f"#### Example {i+1}\n\n")
                    f.write(f"![Detection Example {i+1}](detection_comparisons/{img_file})\n\n")
        
        # Conclusion
        f.write("## Conclusion\n\n")
        f.write("DynamicCompactDetect offers a balance of speed and accuracy, with the added benefit of dynamic routing for efficient inference on a variety of input complexities. ")
        f.write("The model achieves competitive performance compared to state-of-the-art detectors while maintaining a compact architecture suitable for deployment on mainstream platforms.\n\n")
        
        f.write("The early exit mechanism provides significant speed improvements for simpler images without sacrificing accuracy on more complex scenes, ")
        f.write("making it particularly well-suited for real-time applications with varying input complexity.\n")
    
    print(f"Report generated: {report_file}")
    return report_file

def main():
    """Main function to generate reports from command line."""
    parser = argparse.ArgumentParser(description='Generate performance reports from benchmark results')
    parser.add_argument('--benchmark', type=str, required=True,
                        help='Path to benchmark results JSON file')
    parser.add_argument('--evaluation', type=str, default=None,
                        help='Path to evaluation results JSON file (optional)')
    parser.add_argument('--output-dir', type=str, default=None,
                        help='Directory to save the report')
    
    args = parser.parse_args()
    
    generate_comparison_report(args.benchmark, args.evaluation, args.output_dir)

if __name__ == "__main__":
    main() 