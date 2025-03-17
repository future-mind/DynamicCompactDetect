#!/usr/bin/env python3
"""
Script to train DynamicCompactDetect model using Ultralytics API.
"""

import argparse
from ultralytics import YOLO

def main():
    parser = argparse.ArgumentParser(description="Train DynamicCompactDetect model on custom dataset")
    parser.add_argument('--model', type=str, default='dynamiccompactdetect.pt', help='Model to use (dynamiccompactdetect.pt)')
    parser.add_argument('--data', type=str, default='coco8.yaml', help='Dataset config file')
    parser.add_argument('--epochs', type=int, default=5, help='Number of epochs to train for')
    parser.add_argument('--img-size', type=int, default=640, help='Image size for training')
    parser.add_argument('--batch-size', type=int, default=16, help='Batch size for training')
    parser.add_argument('--device', type=str, default='', help='Device to use for training (e.g., 0, cpu)')
    parser.add_argument('--project', type=str, default='runs/train', help='Project directory for saving results')
    parser.add_argument('--name', type=str, default='dynamiccompactdetect', help='Experiment name')
    parser.add_argument('--save', action='store_true', help='Save the trained model')
    args = parser.parse_args()

    print(f"Training DynamicCompactDetect with the following configuration:")
    print(f"  Model: {args.model}")
    print(f"  Dataset: {args.data}")
    print(f"  Epochs: {args.epochs}")
    print(f"  Image Size: {args.img_size}")
    print(f"  Batch Size: {args.batch_size}")
    print(f"  Device: {args.device}")
    print(f"  Project: {args.project}")
    print(f"  Name: {args.name}")
    print(f"  Save: {args.save}")
    
    try:
        # Load the model (it will be downloaded automatically if not available)
        print(f"Loading model {args.model}...")
        model = YOLO(args.model)
        
        # Train the model
        print(f"Training model on {args.data}...")
        results = model.train(
            data=args.data,
            epochs=args.epochs,
            imgsz=args.img_size,
            batch=args.batch_size,
            device=args.device,
            project=args.project,
            name=args.name,
            save=args.save
        )
        
        print("Training completed successfully!")
        print(f"Results saved to {args.project}/{args.name}")
        
        # Optional: Validate the model after training
        if args.save:
            print("Validating the trained model...")
            model.val()
    
    except Exception as e:
        print(f"Error during training: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main() 