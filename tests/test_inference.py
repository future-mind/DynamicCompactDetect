#!/usr/bin/env python3
"""
Simple test script to verify model loading and inference.
"""

import os
import sys
import argparse
from pathlib import Path

try:
    from ultralytics import YOLO
except ImportError:
    print("Error: ultralytics package not found. Please install it with: pip install ultralytics")
    sys.exit(1)

def main():
    """Test loading models and running inference."""
    parser = argparse.ArgumentParser(description="Test model loading and inference")
    parser.add_argument('--model', type=str, default='models/dynamiccompactdetect_finetuned.pt',
                      help='Path to the model file')
    parser.add_argument('--image', type=str, default='data/test_images/zidane.jpg',
                      help='Path to a test image')
    args = parser.parse_args()

    # Check if model file exists
    if not os.path.exists(args.model):
        print(f"Error: Model file not found at {args.model}")
        print("Available models:")
        models_dir = Path('models')
        if models_dir.exists():
            for model_file in models_dir.glob('*.pt'):
                print(f"  - {model_file}")
        else:
            print("  No models directory found.")
        return 1

    # Check if image file exists
    if not os.path.exists(args.image):
        print(f"Error: Test image not found at {args.image}")
        print("Available test images:")
        test_images_dir = Path('data/test_images')
        if test_images_dir.exists():
            for img_file in test_images_dir.glob('*.*'):
                print(f"  - {img_file}")
        else:
            print("  No test images directory found.")
        return 1

    # Load the model
    print(f"Loading model: {args.model}")
    try:
        model = YOLO(args.model)
        print("✅ Model loaded successfully")
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        return 1

    # Run inference
    print(f"Running inference on: {args.image}")
    try:
        results = model(args.image)
        print("✅ Inference completed successfully")
        
        # Display results
        num_detections = len(results[0].boxes)
        print(f"Detected {num_detections} objects")
        
        for i, box in enumerate(results[0].boxes):
            x1, y1, x2, y2 = box.xyxy[0].tolist()
            conf = box.conf[0].item()
            cls = int(box.cls[0].item())
            print(f"  Object {i+1}: Class {cls}, Confidence {conf:.2f}, Coords: ({x1:.1f}, {y1:.1f}, {x2:.1f}, {y2:.1f})")
        
        return 0
    except Exception as e:
        print(f"❌ Error during inference: {e}")
        return 1

if __name__ == "__main__":
    sys.exit(main()) 