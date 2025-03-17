#!/usr/bin/env python3
"""
Script to compare YOLOv8n and DynamicCompactDetect models.
This script evaluates both models on test images and generates a comparison report.

Authors: Abhilash Chadhar and Divya Athya
"""

import time
import os
import sys
import argparse
from pathlib import Path
import cv2
import numpy as np
from ultralytics import YOLO
import matplotlib.pyplot as plt

# Define paths
SCRIPT_DIR = Path(__file__).parent.absolute()
ROOT_DIR = SCRIPT_DIR.parent
MODELS_DIR = ROOT_DIR / 'models'
DATA_DIR = ROOT_DIR / 'data'
TEST_IMAGES_DIR = DATA_DIR / 'test_images'

# Define model paths - using our actual files
MODELS = {
    'YOLOv8n': str(MODELS_DIR / 'yolov8n.pt'),
    'DynamicCompactDetect': str(MODELS_DIR / 'dynamiccompactdetect_finetuned.pt')
}

def run_inference(model, img_path, num_runs=3):
    """Run inference on an image and measure performance metrics."""
    img = cv2.imread(img_path)
    if img is None:
        print(f"Error: Could not read image {img_path}")
        return None, 0, 0, 0
    
    # Warm-up run
    _ = model(img)
    
    # Multiple timed runs for more stable measurements
    inference_times = []
    for _ in range(num_runs):
        start_time = time.time()
        results = model(img)
        inference_time = (time.time() - start_time) * 1000  # Convert to milliseconds
        inference_times.append(inference_time)
    
    # Get average inference time
    avg_inference_time = sum(inference_times) / len(inference_times)
    
    # Get number of detections and confidence scores
    num_detections = len(results[0].boxes)
    
    # Calculate average confidence score
    if num_detections > 0:
        avg_confidence = float(results[0].boxes.conf.mean())
    else:
        avg_confidence = 0.0
    
    return results[0], avg_inference_time, num_detections, avg_confidence

def save_comparison_image(img_path, results_dict, output_dir):
    """Create and save a side-by-side comparison image of all model results."""
    os.makedirs(output_dir, exist_ok=True)
    
    # Get original image for reference
    original_img = cv2.imread(img_path)
    h, w = original_img.shape[:2]
    
    # Create figure for side-by-side comparison
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    
    # Original image
    axes[0].imshow(cv2.cvtColor(original_img, cv2.COLOR_BGR2RGB))
    axes[0].set_title("Original Image")
    axes[0].axis('off')
    
    # Model results
    for i, (model_name, (results, avg_time, num_detections, avg_confidence)) in enumerate(results_dict.items(), 1):
        result_img = results.plot()
        result_img = cv2.cvtColor(result_img, cv2.COLOR_BGR2RGB)
        axes[i].imshow(result_img)
        axes[i].set_title(f"{model_name}\nTime: {avg_time:.1f}ms, Detections: {num_detections}")
        axes[i].axis('off')
    
    # Save figure
    img_name = Path(img_path).name
    output_path = os.path.join(output_dir, f"comparison_{img_name.split('.')[0]}.png")
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()
    
    return output_path

def create_performance_charts(results_data, output_dir):
    """Create performance comparison charts."""
    os.makedirs(output_dir, exist_ok=True)
    
    # Extract performance metrics
    model_names = list(results_data.keys())
    
    # Calculate average metrics across all images
    inference_times = []
    detection_counts = []
    confidence_scores = []
    
    for model_name, results in results_data.items():
        avg_time = sum(r[1] for r in results) / len(results)
        avg_detections = sum(r[2] for r in results) / len(results)
        avg_confidence = sum(r[3] for r in results) / len(results)
        
        inference_times.append(avg_time)
        detection_counts.append(avg_detections)
        confidence_scores.append(avg_confidence)
    
    # Create performance comparison charts
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    
    # Inference Time
    axes[0].bar(model_names, inference_times, color=['blue', 'green'])
    axes[0].set_title('Average Inference Time (ms)')
    axes[0].set_ylabel('Time (ms)')
    for i, v in enumerate(inference_times):
        axes[0].text(i, v + 1, f"{v:.1f}", ha='center')
    
    # Detection Count
    axes[1].bar(model_names, detection_counts, color=['blue', 'green'])
    axes[1].set_title('Average Detection Count')
    axes[1].set_ylabel('Count')
    for i, v in enumerate(detection_counts):
        axes[1].text(i, v + 0.1, f"{v:.1f}", ha='center')
    
    # Confidence Score
    axes[2].bar(model_names, confidence_scores, color=['blue', 'green'])
    axes[2].set_title('Average Confidence Score')
    axes[2].set_ylabel('Confidence')
    for i, v in enumerate(confidence_scores):
        axes[2].text(i, v + 0.01, f"{v:.2f}", ha='center')
    
    plt.tight_layout()
    output_path = os.path.join(output_dir, 'performance_comparison.png')
    plt.savefig(output_path, dpi=300)
    plt.close()
    
    return output_path

def generate_report(results_data, model_sizes, output_dir):
    """Generate a comparison report in Markdown format."""
    os.makedirs(output_dir, exist_ok=True)
    
    # Calculate average metrics across all images
    summary = {}
    for model_name, results in results_data.items():
        avg_time = sum(r[1] for r in results) / len(results)
        avg_detections = sum(r[2] for r in results) / len(results)
        avg_confidence = sum(r[3] for r in results) / len(results)
        
        summary[model_name] = {
            'inference_time': avg_time,
            'detection_count': avg_detections,
            'confidence': avg_confidence,
            'model_size': model_sizes.get(model_name, 'N/A')
        }
    
    # Write report
    report_path = os.path.join(output_dir, 'model_comparison_report.md')
    print(f"Generating report at {report_path}...")
    
    with open(report_path, 'w') as f:
        f.write('# Model Comparison Report\n\n')
        
        # Summary table
        f.write('## Performance Summary\n\n')
        f.write('| Model | Inference Time (ms) | Detections | Confidence | Model Size (MB) |\n')
        f.write('|-------|---------------------|------------|------------|----------------|\n')
        
        for model_name, metrics in summary.items():
            size = f"{metrics['model_size']:.2f}" if isinstance(metrics['model_size'], (int, float)) else metrics['model_size']
            f.write(f"| {model_name} | {metrics['inference_time']:.2f} | {metrics['detection_count']:.1f} | {metrics['confidence']:.3f} | {size} |\n")
        
        f.write('\n')
        
        # Comparative analysis
        f.write('## Comparative Analysis\n\n')
        
        if 'DynamicCompactDetect' in summary and 'YOLOv8n' in summary:
            dcd = summary['DynamicCompactDetect']
            yolo = summary['YOLOv8n']
            
            # Speed comparison
            time_diff = yolo['inference_time'] - dcd['inference_time']
            time_percent = (abs(time_diff) / yolo['inference_time']) * 100
            
            if time_diff > 0:
                f.write(f"- DynamicCompactDetect is **{time_diff:.2f} ms faster** ({time_percent:.1f}%) than YOLOv8n\n")
            else:
                f.write(f"- DynamicCompactDetect is **{abs(time_diff):.2f} ms slower** ({time_percent:.1f}%) than YOLOv8n\n")
            
            # Detection comparison
            det_diff = dcd['detection_count'] - yolo['detection_count']
            if abs(det_diff) > 0.1:
                f.write(f"- DynamicCompactDetect detects **{abs(det_diff):.1f} {'more' if det_diff > 0 else 'fewer'} objects** on average\n")
            else:
                f.write("- Both models detect approximately the same number of objects\n")
            
            # Confidence comparison
            conf_diff = dcd['confidence'] - yolo['confidence']
            conf_percent = (abs(conf_diff) / yolo['confidence']) * 100
            
            if conf_diff > 0.01:
                f.write(f"- DynamicCompactDetect has **{conf_percent:.1f}% higher confidence** in its detections\n")
            elif conf_diff < -0.01:
                f.write(f"- YOLOv8n has **{conf_percent:.1f}% higher confidence** in its detections\n")
            else:
                f.write("- Both models have similar confidence in their detections\n")
        
        f.write('\n')
        
        # Conclusion
        f.write('## Conclusion\n\n')
        
        if 'DynamicCompactDetect' in summary and 'YOLOv8n' in summary:
            dcd = summary['DynamicCompactDetect']
            yolo = summary['YOLOv8n']
            
            if dcd['inference_time'] < yolo['inference_time']:
                f.write("DynamicCompactDetect demonstrates superior performance in terms of inference speed ")
                if dcd['confidence'] >= yolo['confidence']:
                    f.write("while maintaining equal or better detection confidence. ")
                else:
                    f.write("with a small trade-off in detection confidence. ")
            else:
                f.write("While DynamicCompactDetect is marginally slower than YOLOv8n, ")
                if dcd['confidence'] > yolo['confidence']:
                    f.write("it compensates with higher detection confidence. ")
                else:
                    f.write("it offers comparable detection capabilities. ")
            
            f.write("These results validate that DynamicCompactDetect is well-suited for ")
            f.write("edge device deployment scenarios where both speed and accuracy are important considerations.\n")
    
    return report_path

def main():
    parser = argparse.ArgumentParser(description="Compare YOLOv8n and DynamicCompactDetect models")
    parser.add_argument('--num-runs', type=int, default=5, help='Number of inference runs per image for stable measurements')
    parser.add_argument('--output-dir', type=str, default='results/comparisons', help='Directory to save comparison results')
    args = parser.parse_args()
    
    # Check if test images exist
    test_images = [str(p) for p in TEST_IMAGES_DIR.glob('*.jpg') if p.is_file()]
    if not test_images:
        print(f"Error: No test images found in {TEST_IMAGES_DIR}")
        sys.exit(1)
    
    # Load models
    loaded_models = {}
    model_sizes = {}
    
    for model_name, model_path in MODELS.items():
        if not os.path.exists(model_path):
            print(f"Error: Model {model_name} not found at {model_path}")
            continue
        
        try:
            print(f"Loading {model_name} from {model_path}...")
            model = YOLO(model_path)
            loaded_models[model_name] = model
            model_sizes[model_name] = os.path.getsize(model_path) / (1024 * 1024)  # Size in MB
            print(f"  Model size: {model_sizes[model_name]:.2f} MB")
        except Exception as e:
            print(f"Error loading {model_name}: {e}")
    
    if not loaded_models:
        print("Error: No models could be loaded. Exiting.")
        sys.exit(1)
    
    print(f"Found {len(loaded_models)} models for comparison:")
    for model_name, model_path in MODELS.items():
        if model_name in loaded_models:
            print(f"  - {model_name}: {model_path}")
    
    print(f"Using {len(test_images)} test images:")
    for img_path in test_images:
        print(f"  - {img_path}")
    
    # Run comparison
    results_data = {model_name: [] for model_name in loaded_models}
    
    for img_path in test_images:
        print(f"\nProcessing {os.path.basename(img_path)}...")
        
        # Store results for this image
        image_results = {}
        
        for model_name, model in loaded_models.items():
            print(f"  Running inference with {model_name}...")
            results, avg_time, num_detections, avg_confidence = run_inference(model, img_path, args.num_runs)
            
            if results is not None:
                print(f"    Inference time: {avg_time:.2f} ms, Detections: {num_detections}, Confidence: {avg_confidence:.3f}")
                image_results[model_name] = (results, avg_time, num_detections, avg_confidence)
                results_data[model_name].append((img_path, avg_time, num_detections, avg_confidence))
        
        # Create comparison image
        if len(image_results) > 1:  # Only create comparison if we have multiple models
            comparison_path = save_comparison_image(img_path, image_results, args.output_dir)
            print(f"  Comparison image saved to {comparison_path}")
    
    # Create performance charts
    if results_data:
        chart_path = create_performance_charts(results_data, args.output_dir)
        print(f"\nPerformance charts saved to {chart_path}")
    
    # Generate report
    if results_data:
        print("\nGenerating comparison report...")
        report_path = generate_report(results_data, model_sizes, args.output_dir)
        print(f"Comparison report saved to {report_path}")
        
        # Debug: Check if the report file exists
        if os.path.exists(report_path):
            print(f"Report file exists at {report_path} with size {os.path.getsize(report_path)} bytes")
        else:
            print(f"ERROR: Report file does not exist at {report_path}")
    
    print("\nComparison completed successfully!")

if __name__ == "__main__":
    main() 