#!/usr/bin/env python3
"""
Script to test inference with a trained DynamicCompactDetect model
"""

import argparse
from ultralytics import YOLO
from pathlib import Path

def main():
    parser = argparse.ArgumentParser(description="Test DynamicCompactDetect model inference")
    parser.add_argument('--model', type=str, default='runs/train/dynamiccompactdetect/weights/best.pt', help='Path to model weights')
    parser.add_argument('--source', type=str, default=None, help='Path to image or directory of images')
    parser.add_argument('--conf', type=float, default=0.25, help='Confidence threshold for detections')
    parser.add_argument('--device', type=str, default='cpu', help='Device to use for inference (e.g., 0, cpu)')
    parser.add_argument('--save', action='store_true', help='Save detection results')
    parser.add_argument('--show', action='store_true', help='Show detection results')
    args = parser.parse_args()
    
    # If no source is provided, use the COCO8 validation images
    if args.source is None:
        # Find COCO8 dataset path
        datasets_path = Path.home() / 'Library' / 'CloudStorage' / 'OneDrive-AXELERAAI' / 'Documents' / 'cursor' / 'Object_detection' / 'datasets'
        if datasets_path.exists():
            args.source = datasets_path / 'coco8' / 'images' / 'val'
            if not args.source.exists():
                print(f"Error: COCO8 validation images not found at {args.source}")
                return
        else:
            print("Error: Please provide a source for inference")
            return
    
    print(f"Testing DynamicCompactDetect model inference with the following configuration:")
    print(f"  Model: {args.model}")
    print(f"  Source: {args.source}")
    print(f"  Confidence Threshold: {args.conf}")
    print(f"  Device: {args.device}")
    print(f"  Save Results: {args.save}")
    print(f"  Show Results: {args.show}")
    
    try:
        # Load the model
        print(f"Loading model {args.model}...")
        model = YOLO(args.model)
        
        # Perform inference
        print(f"Running inference on {args.source}...")
        results = model.predict(
            source=args.source,
            conf=args.conf,
            device=args.device,
            save=args.save,
            show=args.show
        )
        
        # Print results summary
        print("\nInference Results:")
        for i, result in enumerate(results):
            boxes = result.boxes
            print(f"Image {i+1}: Detected {len(boxes)} objects")
            
            # Print classes and counts
            if len(boxes) > 0:
                classes = result.names
                class_ids = boxes.cls.cpu().numpy().astype(int)
                class_counts = {}
                for cls_id in class_ids:
                    class_name = classes[cls_id]
                    class_counts[class_name] = class_counts.get(class_name, 0) + 1
                
                print("  Detections by class:")
                for cls_name, count in class_counts.items():
                    print(f"    {cls_name}: {count}")
        
        print("\nInference completed successfully!")
    
    except Exception as e:
        print(f"Error during inference: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main() 