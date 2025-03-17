#!/usr/bin/env python3

import os
import time
import torch
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from ultralytics import YOLO

# Path configurations
TEST_IMAGES_DIR = "performance_test/images"
RESULTS_DIR = "performance_test/results"
DCD_MODEL_PATH = "dynamiccompactdetect.pt"
YOLOV8N_MODEL_PATH = "yolov8n.pt"
 
# Ensure results directory exists
os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(os.path.join(RESULTS_DIR, "dcd"), exist_ok=True)
os.makedirs(os.path.join(RESULTS_DIR, "yolov8n"), exist_ok=True)
os.makedirs(os.path.join(RESULTS_DIR, "comparison"), exist_ok=True)

def load_models():
    """Load DynamicCompactDetect and YOLOv8n models."""
    print("Loading models...")
    
    # Load DynamicCompactDetect model
    dcd_model = YOLO(DCD_MODEL_PATH)
    
    # Load YOLOv8n model
    yolov8n_model = YOLO(YOLOV8N_MODEL_PATH)
    
    return dcd_model, yolov8n_model

def measure_inference_time(model, image_path, num_runs=10, warmup=3):
    """Measure inference time for a model."""
    # Warmup runs
    for _ in range(warmup):
        _ = model.predict(image_path, verbose=False)
    
    # Timed runs
    times = []
    for _ in range(num_runs):
        start_time = time.time()
        _ = model.predict(image_path, verbose=False)
        inference_time = (time.time() - start_time) * 1000  # Convert to ms
        times.append(inference_time)
    
    avg_time = np.mean(times)
    std_time = np.std(times)
    
    return avg_time, std_time

def create_comparison_image(image_path, dcd_results, yolov8n_results, output_path):
    """Create a side-by-side comparison image."""
    # Get original image for background
    original_img = Image.open(image_path)
    
    # Create figure with 3 subplots
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    
    # Original image
    axes[0].imshow(np.array(original_img))
    axes[0].set_title("Original Image")
    axes[0].axis("off")
    
    # DCD results
    dcd_img = dcd_results[0].plot()
    axes[1].imshow(dcd_img)
    axes[1].set_title(f"DynamicCompactDetect\n{len(dcd_results[0].boxes)} detections")
    axes[1].axis("off")
    
    # YOLOv8n results
    yolov8n_img = yolov8n_results[0].plot()
    axes[2].imshow(yolov8n_img)
    axes[2].set_title(f"YOLOv8n\n{len(yolov8n_results[0].boxes)} detections")
    axes[2].axis("off")
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()

def compare_detections(dcd_results, yolov8n_results):
    """Compare detection counts and classes between models."""
    dcd_boxes = dcd_results[0].boxes
    yolov8n_boxes = yolov8n_results[0].boxes
    
    # Get class counts
    if len(dcd_boxes) > 0:
        dcd_classes = dcd_boxes.cls.cpu().numpy()
        dcd_class_names = dcd_results[0].names
        dcd_class_counts = {}
        for cls in dcd_classes:
            cls_name = dcd_class_names[int(cls)]
            dcd_class_counts[cls_name] = dcd_class_counts.get(cls_name, 0) + 1
    else:
        dcd_class_counts = {}
    
    if len(yolov8n_boxes) > 0:
        yolov8n_classes = yolov8n_boxes.cls.cpu().numpy()
        yolov8n_class_names = yolov8n_results[0].names
        yolov8n_class_counts = {}
        for cls in yolov8n_classes:
            cls_name = yolov8n_class_names[int(cls)]
            yolov8n_class_counts[cls_name] = yolov8n_class_counts.get(cls_name, 0) + 1
    else:
        yolov8n_class_counts = {}
    
    return dcd_class_counts, yolov8n_class_counts

def main():
    # Download YOLOv8n if not exists
    if not os.path.exists(YOLOV8N_MODEL_PATH):
        print("Downloading YOLOv8n model...")
        os.system(f"wget -q https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8n.pt -O {YOLOV8N_MODEL_PATH}")
    
    # Load models
    dcd_model, yolov8n_model = load_models()
    
    # Process each image
    print("\nProcessing images:")
    for img_file in os.listdir(TEST_IMAGES_DIR):
        if img_file.endswith(('.jpg', '.jpeg', '.png')):
            image_path = os.path.join(TEST_IMAGES_DIR, img_file)
            print(f"\nImage: {img_file}")
            
            # Measure inference time
            dcd_time, dcd_std = measure_inference_time(dcd_model, image_path)
            yolov8n_time, yolov8n_std = measure_inference_time(yolov8n_model, image_path)
            
            speedup = ((yolov8n_time - dcd_time) / yolov8n_time) * 100
            
            print(f"Inference Time:")
            print(f"  DynamicCompactDetect: {dcd_time:.2f} ± {dcd_std:.2f} ms")
            print(f"  YOLOv8n: {yolov8n_time:.2f} ± {yolov8n_std:.2f} ms")
            print(f"  Speedup: {speedup:.2f}%")
            
            # Run inference for visualization
            dcd_results = dcd_model.predict(image_path, save=True, project=RESULTS_DIR, name="dcd")
            yolov8n_results = yolov8n_model.predict(image_path, save=True, project=RESULTS_DIR, name="yolov8n")
            
            # Create comparison image
            output_path = os.path.join(RESULTS_DIR, "comparison", f"comparison_{img_file}")
            create_comparison_image(image_path, dcd_results, yolov8n_results, output_path)
            
            # Compare detections
            dcd_classes, yolov8n_classes = compare_detections(dcd_results, yolov8n_results)
            
            print(f"Detection Comparison:")
            print(f"  DynamicCompactDetect: {len(dcd_results[0].boxes)} detections")
            for cls, count in dcd_classes.items():
                print(f"    {cls}: {count}")
            
            print(f"  YOLOv8n: {len(yolov8n_results[0].boxes)} detections")
            for cls, count in yolov8n_classes.items():
                print(f"    {cls}: {count}")
    
    print("\nPerformance testing completed. Results saved to:", RESULTS_DIR)

if __name__ == "__main__":
    main() 