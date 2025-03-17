#!/usr/bin/env python3
"""
Script to compare YOLOv8n and DynamicCompactDetect models.
This script evaluates models on test images and generates a comparison report.
"""

import time
import os
import sys
import argparse
import numpy as np
import cv2
import torch
from ultralytics import YOLO
import matplotlib.pyplot as plt

# Define model paths
MODEL_PATHS = {
    'yolov8n': 'models/yolov8n.pt',
    'dcd_finetuned': 'models/dynamiccompactdetect_finetuned.pt'
}

# Define test images to download
TEST_IMAGE_URLS = [
    'https://ultralytics.com/images/zidane.jpg',
    'https://ultralytics.com/images/bus.jpg'
]

def download_test_images():
    """Download test images if they don't exist."""
    import urllib.request
    
    os.makedirs('data/test_images', exist_ok=True)
    
    local_test_images = []
    
    for url in TEST_IMAGE_URLS:
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
    """Check if models exist and return available models."""
    available_models = {}
    
    for model_name, model_path in MODEL_PATHS.items():
        if os.path.exists(model_path):
            available_models[model_name] = model_path
        else:
            print(f"Warning: Model {model_name} not found at {model_path}")
    
    return available_models

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
        grid_cols = num_models + 1  # +1 for original image
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

def create_performance_chart(results_data, output_dir):
    """Create performance comparison chart."""
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
    bars = plt.bar(model_names, inference_times, color=['blue', 'green'][:len(model_names)])
    plt.title('Average Inference Time (ms)')
    plt.ylabel('Time (ms)')
    plt.xticks(rotation=45, ha='right')
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height + 0.1,
                f'{height:.1f}', ha='center', va='bottom')
    
    # Detection count chart
    plt.subplot(1, 3, 2)
    bars = plt.bar(model_names, detection_counts, color=['blue', 'green'][:len(model_names)])
    plt.title('Average Detection Count')
    plt.ylabel('Count')
    plt.xticks(rotation=45, ha='right')
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height + 0.1,
                f'{height:.1f}', ha='center', va='bottom')
    
    # Confidence score chart
    plt.subplot(1, 3, 3)
    bars = plt.bar(model_names, confidence_scores, color=['blue', 'green'][:len(model_names)])
    plt.title('Average Confidence Score')
    plt.ylabel('Confidence')
    plt.xticks(rotation=45, ha='right')
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                f'{height:.2f}', ha='center', va='bottom')
    
    plt.tight_layout()
    chart_path = os.path.join(output_dir, 'performance_comparison_chart.png')
    plt.savefig(chart_path)
    plt.close()
    
    return chart_path

def main():
    parser = argparse.ArgumentParser(description="Compare YOLOv8n and DynamicCompactDetect models")
    parser.add_argument('--num-runs', type=int, default=3, help='Number of inference runs per image')
    parser.add_argument('--output-dir', type=str, default='results/comparisons', help='Output directory')
    args = parser.parse_args()
    
    # Check available models
    available_models = check_models()
    if not available_models:
        print("Error: No models available for comparison. Exiting.")
        sys.exit(1)
    
    print(f"Found {len(available_models)} models for comparison:")
    for model_name, model_path in available_models.items():
        print(f"  - {model_name}: {model_path}")
    
    # Download test images or use existing ones
    test_images = download_test_images()
    if not test_images:
        print("Warning: No test images available. Checking for local test images...")
        
        # Check for any images in the data/test_images directory
        if os.path.exists('data/test_images'):
            test_images = [os.path.join('data/test_images', f) for f in os.listdir('data/test_images') 
                         if f.endswith(('.jpg', '.jpeg', '.png'))]
        
        # Check for any images in the data directory
        if not test_images and os.path.exists('data'):
            test_images = [os.path.join('data', f) for f in os.listdir('data') 
                         if f.endswith(('.jpg', '.jpeg', '.png'))]
        
        if not test_images:
            print("Error: No test images found. Exiting.")
            sys.exit(1)
    
    print(f"Using {len(test_images)} test images:")
    for img_path in test_images:
        print(f"  - {img_path}")
    
    # Load models
    loaded_models = {}
    model_sizes = {}
    
    for model_name, model_path in available_models.items():
        print(f"Loading {model_name} from {model_path}...")
        try:
            model = YOLO(model_path)
            loaded_models[model_name] = model
            
            # Get model size in MB
            model_sizes[model_name] = os.path.getsize(model_path) / (1024 * 1024)  # Convert to MB
            print(f"  Model size: {model_sizes[model_name]:.2f} MB")
        except Exception as e:
            print(f"Error loading {model_name}: {e}")
    
    if not loaded_models:
        print("Error: No models could be loaded. Exiting.")
        sys.exit(1)
    
    # Run comparison
    results_data = {model_name: [] for model_name in loaded_models}
    
    for img_path in test_images:
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
        chart_path = create_performance_chart(results_data, args.output_dir)
        print(f"\nPerformance chart saved to {chart_path}")
    
    print("\nComparison completed successfully!")

if __name__ == "__main__":
    main() 