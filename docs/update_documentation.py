import os
import sys
import json
import argparse
import shutil
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np

def create_documentation_structure(output_dir):
    """Create the documentation directory structure."""
    os.makedirs(output_dir, exist_ok=True)
    
    # Create subdirectories
    subdirs = [
        'images',
        'images/mini',
        'images/full',
        'images/comparisons',
        'results',
        'results/mini',
        'results/full'
    ]
    
    for subdir in subdirs:
        os.makedirs(os.path.join(output_dir, subdir), exist_ok=True)
    
    return output_dir

def copy_comparison_images(mini_results_dir, full_results_dir, output_dir):
    """Copy comparison images to the documentation directory."""
    # Copy mini dataset comparison images
    if os.path.exists(mini_results_dir):
        mini_img_dir = os.path.join(output_dir, 'images/mini')
        for img_file in os.listdir(mini_results_dir):
            if img_file.endswith(('.png', '.jpg', '.jpeg')):
                shutil.copy2(
                    os.path.join(mini_results_dir, img_file),
                    os.path.join(mini_img_dir, img_file)
                )
        
        # Copy detection comparison images if they exist
        mini_det_dir = os.path.join(mini_results_dir, 'detection_comparisons')
        if os.path.exists(mini_det_dir):
            mini_det_output_dir = os.path.join(output_dir, 'images/mini/detections')
            os.makedirs(mini_det_output_dir, exist_ok=True)
            
            # Copy only a few representative images
            for i, img_file in enumerate(sorted(os.listdir(mini_det_dir))):
                if img_file.endswith(('.png', '.jpg', '.jpeg')) and i < 5:  # Limit to 5 samples
                    shutil.copy2(
                        os.path.join(mini_det_dir, img_file),
                        os.path.join(mini_det_output_dir, img_file)
                    )
    
    # Copy full dataset comparison images
    if os.path.exists(full_results_dir):
        full_img_dir = os.path.join(output_dir, 'images/full')
        for img_file in os.listdir(full_results_dir):
            if img_file.endswith(('.png', '.jpg', '.jpeg')):
                shutil.copy2(
                    os.path.join(full_results_dir, img_file),
                    os.path.join(full_img_dir, img_file)
                )
        
        # Copy detection comparison images if they exist
        full_det_dir = os.path.join(full_results_dir, 'detection_comparisons')
        if os.path.exists(full_det_dir):
            full_det_output_dir = os.path.join(output_dir, 'images/full/detections')
            os.makedirs(full_det_output_dir, exist_ok=True)
            
            # Copy only a few representative images
            for i, img_file in enumerate(sorted(os.listdir(full_det_dir))):
                if img_file.endswith(('.png', '.jpg', '.jpeg')) and i < 5:  # Limit to 5 samples
                    shutil.copy2(
                        os.path.join(full_det_dir, img_file),
                        os.path.join(full_det_output_dir, img_file)
                    )

def create_combined_comparison_chart(mini_results, full_results, output_path):
    """Create a combined chart comparing mini and full dataset results."""
    # Extract common models
    common_models = set(mini_results.keys()) & set(full_results.keys())
    
    if not common_models:
        print("No common models found between mini and full results.")
        return
    
    # Extract data for 640x640 input size
    models = list(common_models)
    mini_fps = []
    full_fps = []
    
    for model in models:
        # Get mini dataset FPS
        mini_fps_value = 0
        for key in mini_results[model]:
            if key == '640x640':
                mini_fps_value = mini_results[model][key]['fps']
                break
        mini_fps.append(mini_fps_value)
        
        # Get full dataset FPS
        full_fps_value = 0
        for key in full_results[model]:
            if key == '640x640':
                full_fps_value = full_results[model][key]['fps']
                break
        full_fps.append(full_fps_value)
    
    # Plot comparison
    x = np.arange(len(models))
    width = 0.35
    
    plt.figure(figsize=(12, 7))
    bar1 = plt.bar(x - width/2, mini_fps, width, label='Mini Dataset')
    bar2 = plt.bar(x + width/2, full_fps, width, label='Full Dataset')
    
    plt.xlabel('Models')
    plt.ylabel('FPS (higher is better)')
    plt.title('Performance Comparison: Mini vs Full Dataset')
    plt.xticks(x, models)
    plt.legend()
    
    # Add value labels on bars
    for bar, value in zip(bar1, mini_fps):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                f'{value:.1f}', ha='center', fontsize=9)
    
    for bar, value in zip(bar2, full_fps):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                f'{value:.1f}', ha='center', fontsize=9)
    
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()

def create_readme(mini_results, full_results, output_dir):
    """Create the main README.md file with comprehensive documentation."""
    readme_path = os.path.join(output_dir, 'README.md')
    
    with open(readme_path, 'w') as f:
        # Header
        f.write("# DynamicCompactDetect\n\n")
        f.write("A dynamic and efficient object detection model that adapts computation based on input complexity.\n\n")
        
        # Table of Contents
        f.write("## Table of Contents\n\n")
        f.write("1. [Overview](#overview)\n")
        f.write("2. [Model Architecture](#model-architecture)\n")
        f.write("3. [Performance Comparison](#performance-comparison)\n")
        f.write("   - [Full Dataset Results](#full-dataset-results)\n")
        f.write("   - [Mini Dataset Results](#mini-dataset-results)\n")
        f.write("   - [Mini vs Full Comparison](#mini-vs-full-comparison)\n")
        f.write("4. [Visual Detection Comparison](#visual-detection-comparison)\n")
        f.write("5. [Installation and Usage](#installation-and-usage)\n")
        f.write("6. [Training Your Own Model](#training-your-own-model)\n")
        f.write("7. [Benchmark Your Own Images](#benchmark-your-own-images)\n\n")
        
        # Overview Section
        f.write("## Overview\n\n")
        f.write("DynamicCompactDetect is an object detection model designed to dynamically adjust its computational complexity ")
        f.write("based on the input image. By implementing early exit mechanisms and dynamic routing, the model can process ")
        f.write("simple images quickly while dedicating more computation to complex scenes.\n\n")
        
        f.write("### Key Features\n\n")
        f.write("- **Dynamic Early Exit**: Allows the model to terminate inference early for simple inputs\n")
        f.write("- **Adaptive Computation**: Scales processing based on input complexity\n")
        f.write("- **Compact Architecture**: Efficient design with fewer parameters than comparable models\n")
        f.write("- **COCO-Compatible**: Compatible with the COCO dataset format for easy training and evaluation\n\n")
        
        # Model Architecture Section
        f.write("## Model Architecture\n\n")
        f.write("DynamicCompactDetect follows a backbone-neck-head architecture common in modern object detectors:\n\n")
        
        f.write("![Model Architecture](images/model_architecture.png)\n\n")
        
        f.write("- **Backbone**: Efficient feature extractor with dynamic blocks\n")
        f.write("- **Dynamic Routing**: Routes computation through specialized experts based on input features\n")
        f.write("- **Early Exit Layers**: Allow inference to terminate early for simple inputs\n")
        f.write("- **Detection Heads**: Multi-scale detection heads for object localization and classification\n\n")
        
        # Performance Comparison Section - Full Dataset
        f.write("## Performance Comparison\n\n")
        
        # Full Dataset Results
        f.write("### Full Dataset Results\n\n")
        
        if full_results:
            f.write("The following results were obtained by training on the **complete COCO dataset**:\n\n")
            
            # Model Size vs FPS
            f.write("#### Model Size vs Performance\n\n")
            f.write("![Model Size vs FPS (Full Dataset)](images/full/size_vs_fps.png)\n\n")
            
            # Efficiency Comparison
            f.write("#### Efficiency Comparison\n\n")
            f.write("![Efficiency Comparison (Full Dataset)](images/full/efficiency_comparison.png)\n\n")
            
            # Comprehensive Comparison
            f.write("#### Comprehensive Comparison\n\n")
            f.write("![Comprehensive Comparison (Full Dataset)](images/full/comprehensive_comparison.png)\n\n")
            
            # Early Exit Performance (if DynamicCompactDetect exists in results)
            if 'DynamicCompactDetect' in full_results:
                f.write("#### Early Exit Performance\n\n")
                f.write("DynamicCompactDetect's early exit mechanism shows the following performance characteristics:\n\n")
                
                # Create early exit comparison table
                f.write("| Input Size | Mode | Inference Time (ms) | FPS | Speedup |\n")
                f.write("|------------|------|---------------------|-----|--------|\n")
                
                for key in full_results['DynamicCompactDetect']:
                    if key.endswith('_early_exit'):
                        size = key.replace('_early_exit', '')
                        
                        ee_key = key
                        no_ee_key = f"{size}_no_early_exit"
                        
                        if ee_key in full_results['DynamicCompactDetect'] and no_ee_key in full_results['DynamicCompactDetect']:
                            ee_time = full_results['DynamicCompactDetect'][ee_key]['avg_time_ms']
                            ee_fps = full_results['DynamicCompactDetect'][ee_key]['fps']
                            
                            no_ee_time = full_results['DynamicCompactDetect'][no_ee_key]['avg_time_ms']
                            no_ee_fps = full_results['DynamicCompactDetect'][no_ee_key]['fps']
                            speedup = full_results['DynamicCompactDetect'][no_ee_key].get('speedup', 1.0)
                            
                            f.write(f"| {size} | With Early Exit | {ee_time:.2f} | {ee_fps:.2f} | - |\n")
                            f.write(f"| {size} | Without Early Exit | {no_ee_time:.2f} | {no_ee_fps:.2f} | {speedup:.2f}x |\n")
                
                f.write("\n")
        else:
            f.write("Full dataset results are not available. Please run training and evaluation on the full COCO dataset.\n\n")
        
        # Mini Dataset Results
        f.write("### Mini Dataset Results\n\n")
        
        if mini_results:
            f.write("The following results were obtained using the **mini COCO dataset** (for faster experimentation):\n\n")
            
            # Model Size vs FPS
            f.write("#### Model Size vs Performance\n\n")
            f.write("![Model Size vs FPS (Mini Dataset)](images/mini/size_vs_fps.png)\n\n")
            
            # Efficiency Comparison
            f.write("#### Efficiency Comparison\n\n")
            f.write("![Efficiency Comparison (Mini Dataset)](images/mini/efficiency_comparison.png)\n\n")
            
            # Comprehensive Comparison
            f.write("#### Comprehensive Comparison\n\n")
            f.write("![Comprehensive Comparison (Mini Dataset)](images/mini/comprehensive_comparison.png)\n\n")
        else:
            f.write("Mini dataset results are not available.\n\n")
        
        # Mini vs Full Comparison
        f.write("### Mini vs Full Comparison\n\n")
        
        if mini_results and full_results:
            f.write("Comparing performance between mini and full dataset training:\n\n")
            f.write("![Mini vs Full Comparison](images/comparisons/mini_vs_full_fps.png)\n\n")
            
            f.write("The full dataset training generally provides better performance due to:\n\n")
            f.write("- More diverse training examples\n")
            f.write("- Better generalization\n")
            f.write("- More optimization iterations\n\n")
        else:
            f.write("Comparison between mini and full dataset results is not available.\n\n")
        
        # Visual Detection Comparison
        f.write("## Visual Detection Comparison\n\n")
        
        if os.path.exists(os.path.join(output_dir, 'images/full/detections')):
            f.write("### Full Dataset Detection Examples\n\n")
            
            # Display a few full dataset detection examples
            detection_files = sorted([f for f in os.listdir(os.path.join(output_dir, 'images/full/detections')) 
                                     if f.endswith(('.png', '.jpg', '.jpeg'))])[:3]
            
            for i, det_file in enumerate(detection_files):
                f.write(f"#### Example {i+1}\n\n")
                f.write(f"![Detection Example {i+1}](images/full/detections/{det_file})\n\n")
        
        if os.path.exists(os.path.join(output_dir, 'images/mini/detections')):
            f.write("### Mini Dataset Detection Examples\n\n")
            
            # Display a few mini dataset detection examples
            detection_files = sorted([f for f in os.listdir(os.path.join(output_dir, 'images/mini/detections')) 
                                     if f.endswith(('.png', '.jpg', '.jpeg'))])[:3]
            
            for i, det_file in enumerate(detection_files):
                f.write(f"#### Example {i+1}\n\n")
                f.write(f"![Detection Example {i+1}](images/mini/detections/{det_file})\n\n")
        
        # Installation and Usage
        f.write("## Installation and Usage\n\n")
        
        f.write("### Prerequisites\n\n")
        f.write("- Python 3.7+\n")
        f.write("- PyTorch 1.8+\n")
        f.write("- CUDA (optional, for GPU acceleration)\n\n")
        
        f.write("### Installation\n\n")
        f.write("```bash\n")
        f.write("git clone https://github.com/your-username/DynamicCompactDetect.git\n")
        f.write("cd DynamicCompactDetect\n")
        f.write("pip install -r requirements.txt\n")
        f.write("```\n\n")
        
        f.write("### Running Inference\n\n")
        f.write("```python\n")
        f.write("from models.dynamic_compact_detect import DynamicCompactDetect\n\n")
        f.write("# Initialize model\n")
        f.write("model = DynamicCompactDetect(num_classes=80, base_channels=32)\n\n")
        f.write("# Load weights\n")
        f.write("model.load_weights('path/to/weights.pt')\n\n")
        f.write("# Run inference\n")
        f.write("results = model.predict('path/to/image.jpg')\n")
        f.write("```\n\n")
        
        # Training Your Own Model
        f.write("## Training Your Own Model\n\n")
        
        f.write("### On the Full COCO Dataset\n\n")
        f.write("```bash\n")
        f.write("# Download the COCO dataset\n")
        f.write("python data/download_dataset.py --data-dir data --splits train2017 val2017\n\n")
        f.write("# Train the model\n")
        f.write("python train/train_full_coco.py --config train/config.yaml --output-dir results/full_coco_training\n")
        f.write("```\n\n")
        
        f.write("### On the Mini COCO Dataset (for quick experimentation)\n\n")
        f.write("```bash\n")
        f.write("# Download and create mini dataset\n")
        f.write("python data/download_dataset.py --data-dir data --splits train2017 val2017 --create-mini --mini-size 1000\n\n")
        f.write("# Train the model\n")
        f.write("python train/train.py --config train/config.yaml --mini-dataset --output-dir results/mini_coco_training\n")
        f.write("```\n\n")
        
        # Benchmark Section
        f.write("## Benchmark Your Own Images\n\n")
        
        f.write("You can benchmark DynamicCompactDetect against YOLO models on your own images:\n\n")
        
        f.write("```bash\n")
        f.write("# Benchmark models\n")
        f.write("python eval/compare_with_yolo.py --dcd-weights path/to/weights.pt --output-dir results/custom_comparison --input-sizes 640x640\n")
        f.write("```\n\n")
        
        f.write("### Benchmark Options\n\n")
        f.write("- `--input-sizes`: Specify input sizes to test (e.g., 320x320 640x640 1280x1280)\n")
        f.write("- `--iterations`: Number of iterations for timing measurements\n")
        f.write("- `--num-samples`: Number of sample images to use for visual comparison\n")
        f.write("- `--eval`: Perform evaluation on validation set\n")
        f.write("- `--benchmark-only`: Run only the benchmarking without evaluation\n\n")
    
    print(f"Documentation README.md created at {readme_path}")

def load_results(results_dir, filename='benchmark_results.json'):
    """Load benchmark results from JSON file."""
    results_path = os.path.join(results_dir, filename)
    
    if os.path.exists(results_path):
        with open(results_path, 'r') as f:
            return json.load(f)
    
    return {}

def main():
    parser = argparse.ArgumentParser(description='Update DynamicCompactDetect documentation')
    parser.add_argument('--mini-results', type=str, default='results/mini_comparisons',
                        help='Directory containing mini dataset comparison results')
    parser.add_argument('--full-results', type=str, default='results/comparisons',
                        help='Directory containing full dataset comparison results')
    parser.add_argument('--output-dir', type=str, default='docs',
                        help='Output directory for updated documentation')
    args = parser.parse_args()
    
    # Create documentation structure
    create_documentation_structure(args.output_dir)
    
    # Load benchmark results
    mini_results = load_results(args.mini_results)
    full_results = load_results(args.full_results)
    
    # Copy comparison images
    copy_comparison_images(args.mini_results, args.full_results, args.output_dir)
    
    # Create combined comparison chart
    if mini_results and full_results:
        os.makedirs(os.path.join(args.output_dir, 'images/comparisons'), exist_ok=True)
        create_combined_comparison_chart(
            mini_results, full_results,
            os.path.join(args.output_dir, 'images/comparisons/mini_vs_full_fps.png')
        )
    
    # Create main README
    create_readme(mini_results, full_results, args.output_dir)
    
    print(f"Documentation successfully updated in {args.output_dir}")

if __name__ == "__main__":
    main() 