#!/usr/bin/env python3
"""
Test script for DynamicCompactDetect inference.
This script loads the model and runs inference on test images.

Authors: Abhilash Chadhar and Divya Athya
"""

import os
import sys
import time
import argparse
from pathlib import Path

try:
    import cv2
    import torch
    from ultralytics import YOLO
except ImportError:
    print("Error: Missing required packages. Install with: pip install ultralytics opencv-python torch")
    sys.exit(1)

# Define paths - use relative paths instead of hardcoded absolute paths
SCRIPT_DIR = Path(__file__).parent.absolute()
ROOT_DIR = SCRIPT_DIR.parent
MODELS_DIR = ROOT_DIR / 'models'
DATA_DIR = ROOT_DIR / 'data'
TEST_IMAGES_DIR = DATA_DIR / 'test_images'

# Model paths
YOLOV8N_PATH = MODELS_DIR / 'yolov8n.pt'
DCD_PATH = MODELS_DIR / 'dynamiccompactdetect_finetuned.pt'

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Test DynamicCompactDetect inference')
    parser.add_argument('--model', type=str, default='dcd', choices=['yolov8n', 'dcd'],
                        help='Model to use for inference (default: dcd)')
    parser.add_argument('--image', type=str, help='Path to test image',
                        default=str(TEST_IMAGES_DIR / 'bus.jpg'))
    parser.add_argument('--num-runs', type=int, default=5,
                        help='Number of inference runs for timing (default: 5)')
    parser.add_argument('--output-dir', type=str, default='results',
                        help='Directory to save results (default: results)')
    parser.add_argument('--save-output', action='store_true',
                        help='Save output images with detections')
    return parser.parse_args()

def create_output_dir(output_dir):
    """Create output directory if it doesn't exist."""
    os.makedirs(output_dir, exist_ok=True)
    print(f"Results will be saved to: {output_dir}")
    return output_dir

def load_model(model_type):
    """Load the specified model."""
    model_path = YOLOV8N_PATH if model_type == 'yolov8n' else DCD_PATH
    
    if not model_path.exists():
        print(f"Error: Model file not found at {model_path}")
        print("Please download the model or specify the correct path.")
        sys.exit(1)
    
    print(f"Loading model from {model_path}...")
    start_time = time.time()
    model = YOLO(str(model_path))
    load_time = (time.time() - start_time) * 1000  # ms
    print(f"Model loaded in {load_time:.2f} ms")
    
    return model, load_time

def run_inference(model, image_path, num_runs=5):
    """Run inference on the specified image multiple times and report average time."""
    if not os.path.exists(image_path):
        print(f"Error: Image file not found at {image_path}")
        sys.exit(1)

    print(f"Running inference on {image_path}...")
    
    # Warmup
    _ = model(image_path)
    
    inference_times = []
    for i in range(num_runs):
        start_time = time.time()
        results = model(image_path)
        inference_time = (time.time() - start_time) * 1000  # ms
        inference_times.append(inference_time)
        print(f"Run {i+1}/{num_runs}: {inference_time:.2f} ms")
    
    avg_time = sum(inference_times) / len(inference_times)
    print(f"Average inference time: {avg_time:.2f} ms")
    
    return results, avg_time

def save_results(results, output_dir, model_type, image_path):
    """Save detection results and visualization."""
    base_filename = os.path.basename(image_path)
    output_path = os.path.join(output_dir, f"{model_type}_{base_filename}")
    
    # Save visualization
    result_img = results[0].plot()
    cv2.imwrite(output_path, result_img)
    print(f"Detection visualization saved to {output_path}")
    
    # Save detection data
    detection_data = []
    for i, r in enumerate(results):
        boxes = r.boxes
        for box in boxes:
            x1, y1, x2, y2 = box.xyxy[0].tolist()
            conf = box.conf[0].item()
            cls = int(box.cls[0].item())
            cls_name = results[0].names[cls]
            detection_data.append({
                'class': cls_name,
                'confidence': conf,
                'box': [x1, y1, x2, y2]
            })
    
    # Print summary
    print(f"Detected {len(detection_data)} objects")
    for i, det in enumerate(detection_data):
        print(f"  {i+1}. {det['class']} ({det['confidence']:.2f})")

def main():
    """Main function."""
    args = parse_args()
    output_dir = create_output_dir(args.output_dir)
    
    model, load_time = load_model(args.model)
    results, inference_time = run_inference(model, args.image, args.num_runs)
    
    if args.save_output:
        save_results(results, output_dir, args.model, args.image)
    
    print("\nPerformance Summary:")
    print(f"Model: {args.model}")
    print(f"Load time: {load_time:.2f} ms")
    print(f"Inference time: {inference_time:.2f} ms")
    print(f"Total time: {load_time + inference_time:.2f} ms")

if __name__ == "__main__":
    main() 