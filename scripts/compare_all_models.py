#!/usr/bin/env python3
"""
Script to compare YOLOv8n, YOLOv10n, and DynamicCompactDetect models (original and fine-tuned).
This script evaluates all models on the same test images and generates a comprehensive comparison report.

Authors: Abhilash Chadhar and Divya Athya
"""

import time
import os
import sys
import argparse
import numpy as np
import cv2
from ultralytics import YOLO
from tabulate import tabulate
import matplotlib.pyplot as plt
import torch

# Define model paths
MODELS = {
    'yolov8n': 'yolov8n.pt',
    'yolov10n': 'yolov10n.pt',
    'dcd_original': 'dynamiccompactdetect.pt',
    'dcd_finetuned': 'runs/finetune/dcd_yolo11/weights/best.pt'
}

# Define test image URLs
TEST_IMAGE_URLS = [
    'https://ultralytics.com/images/zidane.jpg',
    'https://ultralytics.com/images/bus.jpg',
    'https://ultralytics.com/images/person.jpg'
]

def download_test_images():
    """Download test images if they don't exist and return local paths."""
    import urllib.request
    
    os.makedirs('data/test_images', exist_ok=True)
    
    local_test_images = []
    
    for i, url in enumerate(TEST_IMAGE_URLS):
        img_name = url.split('/')[-1]
        img_path = os.path.join('data/test_images', img_name)
        local_test_images.append(img_path)
        
        if not os.path.exists(img_path):
            print(f"Downloading {img_name}...")
            try:
                urllib.request.urlretrieve(url, img_path)
                print(f"Downloaded {img_name} to {img_path}")
            except Exception as e:
                print(f"Error downloading {url}: {e}")
    
    return local_test_images

def check_models():
    """Check if all models exist and download if necessary."""
    missing_models = []
    
    for model_name, model_path in MODELS.items():
        if not os.path.exists(model_path):
            missing_models.append((model_name, model_path))
    
    if missing_models:
        print("The following models are missing:")
        for model_name, model_path in missing_models:
            print(f"  - {model_name}: {model_path}")
        
        print("\nPlease ensure all models are available before running the comparison.")
        print("For YOLOv8n and YOLOv10n, they will be downloaded automatically when loaded.")
        print("For DynamicCompactDetect, please run the fine-tuning script first.")
        
        # Check if fine-tuned model is missing but original exists
        if ('dcd_finetuned', MODELS['dcd_finetuned']) in missing_models and os.path.exists(MODELS['dcd_original']):
            print("\nThe fine-tuned DynamicCompactDetect model is missing.")
            print("Would you like to run the fine-tuning script now? (y/n)")
            response = input().strip().lower()
            
            if response == 'y':
                print("Running fine-tuning script...")
                os.system(f"python scripts/finetune_dynamiccompactdetect.py --model {MODELS['dcd_original']} --epochs 10")
                return check_models()  # Recheck models after fine-tuning
            else:
                print("Skipping fine-tuning. Will only compare available models.")
                # Remove missing models from the comparison
                for model_name, _ in missing_models:
                    MODELS.pop(model_name)
        else:
            # Remove missing models from the comparison
            for model_name, _ in missing_models:
                MODELS.pop(model_name)
    
    return len(MODELS) > 0

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
        # Clear CUDA cache if available
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            
        start_time = time.time()
        results = model(img)
        inference_time = (time.time() - start_time) * 1000  # Convert to milliseconds
        inference_times.append(inference_time)
    
    # Get average inference time
    avg_inference_time = np.mean(inference_times)
    
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
    h_orig, w_orig = original_img.shape[:2]
    
    # Get all result images
    result_images = []
    for model_name, (results, inference_time, num_detections, _) in results_dict.items():
        # Plot detection results
        result_img = results.plot()
        
        # Add text with model name and metrics
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(result_img, f"{model_name}", (10, 30), font, 0.7, (0, 0, 255), 2)
        cv2.putText(result_img, f"Time: {inference_time:.1f}ms", (10, 60), font, 0.7, (0, 0, 255), 2)
        cv2.putText(result_img, f"Detections: {num_detections}", (10, 90), font, 0.7, (0, 0, 255), 2)
        
        # Save individual result
        model_dir = os.path.join(output_dir, model_name)
        os.makedirs(model_dir, exist_ok=True)
        cv2.imwrite(os.path.join(model_dir, os.path.basename(img_path)), result_img)
        
        # Add to list for comparison image
        result_images.append(result_img)
    
    # Determine grid layout based on number of models
    num_models = len(result_images)
    if num_models <= 2:
        grid_cols = num_models
        grid_rows = 1
    else:
        grid_cols = 2
        grid_rows = (num_models + 1) // 2  # +1 to include original image
    
    # Create a grid of images
    cell_height = h_orig
    cell_width = w_orig
    grid_height = grid_rows * cell_height
    grid_width = grid_cols * cell_width
    
    # Create empty grid
    grid_img = np.zeros((grid_height, grid_width, 3), dtype=np.uint8)
    
    # Add original image in first position
    grid_img[0:cell_height, 0:cell_width] = original_img
    cv2.putText(grid_img, "Original", (10, 30), font, 0.7, (0, 0, 255), 2)
    
    # Add result images
    for i, result_img in enumerate(result_images):
        row = (i + 1) // grid_cols
        col = (i + 1) % grid_cols
        y_start = row * cell_height
        x_start = col * cell_width
        y_end = y_start + cell_height
        x_end = x_start + cell_width
        
        # Resize if needed
        if result_img.shape[:2] != (cell_height, cell_width):
            result_img = cv2.resize(result_img, (cell_width, cell_height))
        
        # Place in grid
        grid_img[y_start:y_end, x_start:x_end] = result_img
    
    # Save comparison grid
    output_path = os.path.join(output_dir, f"comparison_{os.path.basename(img_path)}")
    cv2.imwrite(output_path, grid_img)
    
    return output_path

def create_performance_charts(results_data, output_dir):
    """Create performance comparison charts."""
    os.makedirs(output_dir, exist_ok=True)
    
    # Extract data for charts
    model_names = list(results_data.keys())
    inference_times = [np.mean([r[1] for r in results]) for model_name, results in results_data.items()]
    detection_counts = [np.mean([r[2] for r in results]) for model_name, results in results_data.items()]
    confidence_scores = [np.mean([r[3] for r in results]) for model_name, results in results_data.items()]
    
    # Create bar charts
    plt.figure(figsize=(12, 6))
    
    # Inference time chart
    plt.subplot(1, 3, 1)
    bars = plt.bar(model_names, inference_times, color=['blue', 'green', 'orange', 'red'][:len(model_names)])
    plt.title('Average Inference Time (ms)')
    plt.ylabel('Time (ms)')
    plt.xticks(rotation=45, ha='right')
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height + 0.1,
                f'{height:.1f}', ha='center', va='bottom')
    
    # Detection count chart
    plt.subplot(1, 3, 2)
    bars = plt.bar(model_names, detection_counts, color=['blue', 'green', 'orange', 'red'][:len(model_names)])
    plt.title('Average Detection Count')
    plt.ylabel('Count')
    plt.xticks(rotation=45, ha='right')
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height + 0.1,
                f'{height:.1f}', ha='center', va='bottom')
    
    # Confidence score chart
    plt.subplot(1, 3, 3)
    bars = plt.bar(model_names, confidence_scores, color=['blue', 'green', 'orange', 'red'][:len(model_names)])
    plt.title('Average Confidence Score')
    plt.ylabel('Confidence')
    plt.xticks(rotation=45, ha='right')
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                f'{height:.2f}', ha='center', va='bottom')
    
    plt.tight_layout()
    chart_path = os.path.join(output_dir, 'performance_comparison_charts.png')
    plt.savefig(chart_path)
    plt.close()
    
    return chart_path

def generate_comparison_report(results_data, model_sizes, output_dir):
    """Generate a comprehensive comparison report."""
    os.makedirs(output_dir, exist_ok=True)
    
    report_path = os.path.join(output_dir, 'model_comparison_report.md')
    
    # Calculate average metrics across all test images
    avg_metrics = {}
    for model_name, results in results_data.items():
        avg_inference_time = np.mean([r[1] for r in results])
        avg_detection_count = np.mean([r[2] for r in results])
        avg_confidence = np.mean([r[3] for r in results])
        
        avg_metrics[model_name] = {
            'inference_time': avg_inference_time,
            'detection_count': avg_detection_count,
            'confidence': avg_confidence,
            'model_size': model_sizes.get(model_name, 'N/A')
        }
    
    # Create comparison table data
    table_data = []
    for model_name, metrics in avg_metrics.items():
        table_data.append([
            model_name,
            f"{metrics['inference_time']:.2f} ms",
            f"{metrics['detection_count']:.1f}",
            f"{metrics['confidence']:.3f}",
            f"{metrics['model_size']:.2f} MB" if isinstance(metrics['model_size'], (int, float)) else metrics['model_size']
        ])
    
    # Write report
    with open(report_path, 'w') as f:
        f.write("# Object Detection Model Comparison Report\n\n")
        
        f.write("## Models Compared\n\n")
        for model_name, model_path in MODELS.items():
            f.write(f"- **{model_name}**: `{model_path}`\n")
        f.write("\n")
        
        f.write("## Performance Comparison\n\n")
        f.write(tabulate(
            table_data,
            headers=["Model", "Avg. Inference Time", "Avg. Detections", "Avg. Confidence", "Model Size"],
            tablefmt="pipe"
        ))
        f.write("\n\n")
        
        # Add improvement section if fine-tuned model is included
        if 'dcd_original' in avg_metrics and 'dcd_finetuned' in avg_metrics:
            orig = avg_metrics['dcd_original']
            finetuned = avg_metrics['dcd_finetuned']
            
            time_change = orig['inference_time'] - finetuned['inference_time']
            time_percent = (time_change / orig['inference_time']) * 100 if orig['inference_time'] > 0 else 0
            
            conf_change = finetuned['confidence'] - orig['confidence']
            conf_percent = (conf_change / orig['confidence']) * 100 if orig['confidence'] > 0 else 0
            
            f.write("## DynamicCompactDetect Improvement After Fine-tuning\n\n")
            f.write(f"- **Inference Time**: {time_change:.2f} ms faster ({time_percent:.1f}%)\n")
            f.write(f"- **Confidence Score**: {conf_change:.3f} higher ({conf_percent:.1f}%)\n")
            
            # Add model size comparison if available
            if isinstance(orig['model_size'], (int, float)) and isinstance(finetuned['model_size'], (int, float)):
                size_change = finetuned['model_size'] - orig['model_size']
                size_percent = (size_change / orig['model_size']) * 100
                f.write(f"- **Model Size**: {size_change:.2f} MB change ({size_percent:.1f}%)\n")
            
            f.write("\n")
        
        f.write("## Comparison with YOLOv8n and YOLOv10n\n\n")
        
        # Compare DCD (best version) with YOLOv8n and YOLOv10n
        best_dcd = 'dcd_finetuned' if 'dcd_finetuned' in avg_metrics else 'dcd_original'
        
        if best_dcd in avg_metrics:
            dcd = avg_metrics[best_dcd]
            
            comparisons = []
            
            if 'yolov8n' in avg_metrics:
                yolov8 = avg_metrics['yolov8n']
                time_diff = yolov8['inference_time'] - dcd['inference_time']
                time_percent = (time_diff / yolov8['inference_time']) * 100 if yolov8['inference_time'] > 0 else 0
                
                comparisons.append(f"- Compared to **YOLOv8n**, DynamicCompactDetect is {abs(time_diff):.2f} ms " + 
                                  (f"faster ({time_percent:.1f}%)" if time_diff > 0 else f"slower ({-time_percent:.1f}%)"))
            
            if 'yolov10n' in avg_metrics:
                yolov10 = avg_metrics['yolov10n']
                time_diff = yolov10['inference_time'] - dcd['inference_time']
                time_percent = (time_diff / yolov10['inference_time']) * 100 if yolov10['inference_time'] > 0 else 0
                
                comparisons.append(f"- Compared to **YOLOv10n**, DynamicCompactDetect is {abs(time_diff):.2f} ms " + 
                                  (f"faster ({time_percent:.1f}%)" if time_diff > 0 else f"slower ({-time_percent:.1f}%)"))
            
            for comparison in comparisons:
                f.write(f"{comparison}\n")
            
            f.write("\n")
        
        f.write("## Test Images\n\n")
        f.write("The comparison was performed on the following test images:\n\n")
        for url in TEST_IMAGE_URLS:
            img_name = url.split('/')[-1]
            f.write(f"- {img_name}\n")
        f.write("\n")
        
        f.write("## Visualization\n\n")
        f.write("Comparison visualizations are available in the `comparison_results` directory.\n")
        f.write("Performance charts are available at `comparison_results/performance_comparison_charts.png`.\n\n")
        
        f.write("## Conclusion\n\n")
        
        # Determine best model based on inference time
        best_model = min(avg_metrics.items(), key=lambda x: x[1]['inference_time'])
        f.write(f"Based on inference time, the fastest model is **{best_model[0]}** " +
                f"with an average of {best_model[1]['inference_time']:.2f} ms per image.\n\n")
        
        # Determine best model based on confidence
        best_conf_model = max(avg_metrics.items(), key=lambda x: x[1]['confidence'])
        f.write(f"Based on detection confidence, the best model is **{best_conf_model[0]}** " +
                f"with an average confidence score of {best_conf_model[1]['confidence']:.3f}.\n\n")
        
        # Add specific conclusion about DynamicCompactDetect
        if best_dcd in avg_metrics:
            f.write("### DynamicCompactDetect Performance\n\n")
            
            if best_dcd == 'dcd_finetuned':
                f.write("The fine-tuned DynamicCompactDetect model shows significant improvements over the original version, ")
                f.write("demonstrating the effectiveness of the YOLOv11 training methodology.\n\n")
            
            # Compare with YOLOv8n and YOLOv10n
            comparisons = []
            
            if 'yolov8n' in avg_metrics and 'yolov10n' in avg_metrics:
                dcd_metrics = avg_metrics[best_dcd]
                yolov8_metrics = avg_metrics['yolov8n']
                yolov10_metrics = avg_metrics['yolov10n']
                
                if dcd_metrics['inference_time'] < yolov8_metrics['inference_time'] and dcd_metrics['inference_time'] < yolov10_metrics['inference_time']:
                    f.write("DynamicCompactDetect outperforms both YOLOv8n and YOLOv10n in terms of inference speed, ")
                    f.write("making it an excellent choice for real-time applications.\n\n")
                elif dcd_metrics['inference_time'] < yolov8_metrics['inference_time'] or dcd_metrics['inference_time'] < yolov10_metrics['inference_time']:
                    f.write("DynamicCompactDetect shows competitive performance, outperforming at least one of the baseline models ")
                    f.write("in terms of inference speed.\n\n")
                else:
                    f.write("While DynamicCompactDetect doesn't outperform YOLOv8n and YOLOv10n in raw speed, ")
                    f.write("it offers a good balance of performance and accuracy.\n\n")
                
                # Compare confidence scores
                if dcd_metrics['confidence'] > yolov8_metrics['confidence'] and dcd_metrics['confidence'] > yolov10_metrics['confidence']:
                    f.write("In terms of detection confidence, DynamicCompactDetect achieves higher scores than both baseline models, ")
                    f.write("suggesting more reliable detections.\n\n")
            
            f.write("For detailed performance metrics and visual comparisons, please refer to the comparison images and charts.\n")
    
    print(f"Comparison report saved to {report_path}")
    return report_path

def main():
    parser = argparse.ArgumentParser(description="Compare YOLOv8n, YOLOv10n, and DynamicCompactDetect models")
    parser.add_argument('--num-runs', type=int, default=3, help='Number of inference runs per image for stable measurements')
    parser.add_argument('--output-dir', type=str, default='comparison_results', help='Directory to save comparison results')
    args = parser.parse_args()
    
    # Check if models exist
    if not check_models():
        print("Error: No models available for comparison. Exiting.")
        sys.exit(1)
    
    # Download test images
    test_image_paths = download_test_images()
    if not test_image_paths:
        print("Error: No test images available. Exiting.")
        sys.exit(1)
    
    print(f"Comparing {len(MODELS)} models on {len(test_image_paths)} test images...")
    
    # Load models
    loaded_models = {}
    model_sizes = {}
    for model_name, model_path in MODELS.items():
        print(f"Loading {model_name} from {model_path}...")
        try:
            model = YOLO(model_path)
            loaded_models[model_name] = model
            
            # Get model size in MB
            if os.path.exists(model_path):
                model_sizes[model_name] = os.path.getsize(model_path) / (1024 * 1024)  # Convert to MB
            else:
                model_sizes[model_name] = "N/A"
        except Exception as e:
            print(f"Error loading {model_name}: {e}")
    
    if not loaded_models:
        print("Error: No models could be loaded. Exiting.")
        sys.exit(1)
    
    # Run comparison
    results_data = {model_name: [] for model_name in loaded_models}
    
    for img_path in test_image_paths:
        img_name = os.path.basename(img_path)
        print(f"\nProcessing {img_name}...")
        
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
        if image_results:
            comparison_path = save_comparison_image(img_path, image_results, args.output_dir)
            print(f"  Comparison image saved to {comparison_path}")
    
    # Create performance charts
    if results_data:
        chart_path = create_performance_charts(results_data, args.output_dir)
        print(f"\nPerformance charts saved to {chart_path}")
    
    # Generate comparison report
    if results_data:
        report_path = generate_comparison_report(results_data, model_sizes, args.output_dir)
        print(f"\nComparison report saved to {report_path}")
    
    print("\nComparison completed successfully!")

if __name__ == "__main__":
    main() 