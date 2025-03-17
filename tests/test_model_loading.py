#!/usr/bin/env python3
"""
Test script for loading DynamicCompactDetect model and running basic inference.
"""

import os
import sys
import time
from pathlib import Path

import cv2
import torch
from ultralytics import YOLO

# Add the parent directory to the path to ensure imports work correctly
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

def test_model_loading():
    """Test loading both original and finetuned DynamicCompactDetect models."""
    print("Testing model loading...")
    
    # Check for model files
    model_paths = {
        "original": "dynamiccompactdetect.pt", 
        "finetuned": "dynamiccompactdetect_finetuned.pt"
    }
    
    for model_name, model_path in model_paths.items():
        if os.path.exists(model_path):
            print(f"Loading {model_name} model: {model_path}")
            try:
                model = YOLO(model_path)
                print(f"✅ Successfully loaded {model_name} model")
                
                # Print model info
                if hasattr(model, "model"):
                    print(f"Model type: {type(model.model)}")
                    if hasattr(model.model, "names"):
                        print(f"Classes: {model.model.names}")
                
            except Exception as e:
                print(f"❌ Error loading {model_name} model: {e}")
        else:
            print(f"⚠️ Model file not found: {model_path}")
    
    print("Model loading test complete.")

def test_inference():
    """Test running inference with the model on a sample image."""
    print("\nTesting model inference...")
    
    # Find any model
    available_models = ["dynamiccompactdetect_finetuned.pt", "dynamiccompactdetect.pt", "yolov8n.pt"]
    model_path = None
    
    for model in available_models:
        if os.path.exists(model):
            model_path = model
            break
    
    if not model_path:
        print("❌ No model file found for inference test")
        return
    
    print(f"Using model: {model_path}")
    
    # Find a test image
    test_image_paths = [
        "test_images/bus.jpg",
        "test_images/zidane.jpg",
        "data/images/bus.jpg",
        "data/images/zidane.jpg",
    ]
    
    test_image = None
    for img_path in test_image_paths:
        if os.path.exists(img_path):
            test_image = img_path
            break
    
    if not test_image:
        print("❌ No test image found")
        return
    
    print(f"Using test image: {test_image}")
    
    # Load model and run inference
    try:
        model = YOLO(model_path)
        
        # Run inference
        start_time = time.time()
        results = model(test_image)
        inference_time = time.time() - start_time
        
        print(f"✅ Inference successful")
        print(f"Inference time: {inference_time:.4f} seconds")
        print(f"Detections: {len(results[0].boxes)}")
        
        # Print detection details
        for i, box in enumerate(results[0].boxes):
            cls = int(box.cls[0])
            conf = float(box.conf[0])
            cls_name = model.model.names[cls]
            print(f"  {i+1}. {cls_name}: {conf:.2f}")
        
    except Exception as e:
        print(f"❌ Error during inference: {e}")
    
    print("Inference test complete.")

if __name__ == "__main__":
    print("=" * 50)
    print("DynamicCompactDetect Model Test")
    print("=" * 50)
    test_model_loading()
    test_inference()
    print("\nAll tests completed.") 