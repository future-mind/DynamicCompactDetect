#!/usr/bin/env python3
"""
Script to generate benchmark data from model comparison results.
"""

import os
import sys
import json
import glob
import numpy as np
from pathlib import Path
import cv2
import time
import torch
from ultralytics import YOLO

# Define paths
SCRIPT_DIR = Path(__file__).parent.absolute()
ROOT_DIR = SCRIPT_DIR.parent
MODELS_DIR = ROOT_DIR / 'models'
RESULTS_DIR = ROOT_DIR / 'results'
DATA_DIR = ROOT_DIR / 'data'
TEST_IMAGES_DIR = DATA_DIR / 'test_images'
BENCHMARKS_DIR = RESULTS_DIR / 'benchmarks'

# Ensure directories exist
os.makedirs(BENCHMARKS_DIR, exist_ok=True)

def load_model(model_path):
    """Load a YOLO model."""
    try:
        model = YOLO(model_path)
        return model
    except Exception as e:
        print(f"Error loading model from {model_path}: {e}")
        return None

def run_inference(model, image_path):
    """Run inference on an image and return results."""
    try:
        # Run inference
        start_time = time.time()
        results = model(image_path)
        end_time = time.time()
        
        # Extract results
        inference_time = (end_time - start_time) * 1000  # Convert to ms
        
        # Calculate metrics
        num_detections = len(results[0].boxes)
        
        # Get confidence scores
        confidence_scores = []
        for box in results[0].boxes:
            confidence_scores.append(box.conf.item())
        
        confidence = np.mean(confidence_scores) if confidence_scores else 0
        
        return {
            'inference_time': inference_time,
            'num_detections': num_detections,
            'confidence': confidence
        }
    except Exception as e:
        print(f"Error during inference: {e}")
        return None

def benchmark_model(model_path, model_name):
    """Benchmark a model on test images."""
    print(f"Benchmarking {model_name}...")
    
    # Load model
    model = load_model(model_path)
    if model is None:
        print(f"Failed to load model {model_name}")
        return None
    
    # Get test images
    test_images = glob.glob(str(TEST_IMAGES_DIR / "*.jpg")) + glob.glob(str(TEST_IMAGES_DIR / "*.png"))
    if not test_images:
        print("No test images found.")
        return None
    
    # Run inference on each image
    inference_times = []
    detection_counts = []
    confidence_scores = []
    
    for image_path in test_images:
        print(f"  Processing {Path(image_path).name}...")
        results = run_inference(model, image_path)
        if results:
            inference_times.append(results['inference_time'])
            detection_counts.append(results['num_detections'])
            if results['num_detections'] > 0:
                confidence_scores.append(results['confidence'])
    
    # Calculate average metrics
    avg_inference_time = np.mean(inference_times) if inference_times else 0
    avg_detection_count = np.mean(detection_counts) if detection_counts else 0
    avg_confidence = np.mean(confidence_scores) if confidence_scores else 0
    
    # Create benchmark data
    benchmark_data = {
        'model_name': model_name,
        'model_path': str(model_path),
        'inference_time': avg_inference_time,
        'detection_count': avg_detection_count,
        'confidence': avg_confidence,
        'num_images': len(test_images),
        'images_with_detections': sum(1 for count in detection_counts if count > 0)
    }
    
    return benchmark_data

def save_benchmark_data(benchmark_data, model_name):
    """Save benchmark data to a JSON file."""
    output_path = BENCHMARKS_DIR / f"{model_name}.json"
    with open(output_path, 'w') as f:
        json.dump(benchmark_data, f, indent=2)
    print(f"Benchmark data saved to {output_path}")

def main():
    """Main function to benchmark models."""
    # Define models to benchmark
    models = {
        'YOLOv8n': MODELS_DIR / 'yolov8n.pt',
        'DynamicCompactDetect': MODELS_DIR / 'dynamiccompactdetect_finetuned.pt'
    }
    
    # Benchmark each model
    for model_name, model_path in models.items():
        if not model_path.exists():
            print(f"Model file not found: {model_path}")
            continue
        
        benchmark_data = benchmark_model(model_path, model_name)
        if benchmark_data:
            save_benchmark_data(benchmark_data, model_name)
    
    print("Benchmarking complete.")

if __name__ == "__main__":
    main() 