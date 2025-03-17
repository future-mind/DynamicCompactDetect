#!/usr/bin/env python3
"""
Script to compare the performance of the fine-tuned DynamicCompactDetect model with the original pretrained model
"""

import argparse
import os
import time
import numpy as np
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont
from ultralytics import YOLO

def get_model_predictions(model, source, conf=0.25):
    """Get predictions from a model on a set of images"""
    results = model.predict(source=source, conf=conf, verbose=False)
    return results

def measure_inference_time(model, source, iterations=10, warmup=3):
    """Measure inference time of a model on a set of images"""
    # Load images
    if isinstance(source, (str, Path)) and os.path.isdir(source):
        image_paths = [os.path.join(source, f) for f in os.listdir(source) 
                       if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp'))]
    else:
        if isinstance(source, (str, Path)) and os.path.isfile(source):
            image_paths = [source]
        else:
            image_paths = source if isinstance(source, list) else [source]
    
    # Warm up the model
    for _ in range(warmup):
        model.predict(source=image_paths[0], verbose=False)
    
    # Measure inference time
    times = []
    for _ in range(iterations):
        for img_path in image_paths:
            start_time = time.time()
            model.predict(source=img_path, verbose=False)
            end_time = time.time()
            times.append(end_time - start_time)
    
    # Calculate statistics
    mean_time = np.mean(times)
    std_time = np.std(times)
    
    return mean_time, std_time

def create_comparison_image(img1_path, img2_path, output_path):
    """Create a side-by-side comparison image of original and fine-tuned model predictions"""
    # Open images
    img1 = Image.open(img1_path)
    img2 = Image.open(img2_path)
    
    # Resize images to same height if necessary
    height = max(img1.height, img2.height)
    if img1.height != height:
        width = int(img1.width * (height / img1.height))
        img1 = img1.resize((width, height))
    if img2.height != height:
        width = int(img2.width * (height / img2.height))
        img2 = img2.resize((width, height))
    
    # Create new image with combined width
    new_img = Image.new('RGB', (img1.width + img2.width, height))
    
    # Paste images side by side
    new_img.paste(img1, (0, 0))
    new_img.paste(img2, (img1.width, 0))
    
    # Add text labels
    draw = ImageDraw.Draw(new_img)
    try:
        font = ImageFont.truetype("Arial.ttf", 24)
    except IOError:
        font = ImageFont.load_default()
    
    draw.text((10, 10), "Original DynamicCompactDetect", fill=(255, 255, 255), font=font)
    draw.text((img1.width + 10, 10), "Fine-tuned DynamicCompactDetect", fill=(255, 255, 255), font=font)
    
    # Save the result
    new_img.save(output_path)
    return output_path

def main():
    parser = argparse.ArgumentParser(description="Compare performance of DynamicCompactDetect models")
    parser.add_argument('--original-model', type=str, default='dynamiccompactdetect.pt', help='Path to original model weights')
    parser.add_argument('--fine-tuned-model', type=str, default='runs/train/dynamiccompactdetect/weights/best.pt', help='Path to fine-tuned model weights')
    parser.add_argument('--source', type=str, default=None, help='Path to images for inference comparison')
    parser.add_argument('--conf', type=float, default=0.25, help='Confidence threshold for detections')
    parser.add_argument('--output-dir', type=str, default='runs/compare', help='Directory to save comparison results')
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
            print("Error: Please provide a source for comparison")
            return
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True, parents=True)
    
    print(f"Comparing DynamicCompactDetect models with the following configuration:")
    print(f"  Original Model: {args.original_model}")
    print(f"  Fine-tuned Model: {args.fine_tuned_model}")
    print(f"  Source: {args.source}")
    print(f"  Confidence Threshold: {args.conf}")
    print(f"  Output Directory: {args.output_dir}")
    
    try:
        # Load models
        print("Loading models...")
        original_model = YOLO(args.original_model)
        fine_tuned_model = YOLO(args.fine_tuned_model)
        
        # Measure inference time
        print("Measuring inference time...")
        original_time, original_std = measure_inference_time(original_model, args.source)
        fine_tuned_time, fine_tuned_std = measure_inference_time(fine_tuned_model, args.source)
        
        print("\nInference Time Comparison:")
        print(f"  Original Model: {original_time*1000:.2f} ± {original_std*1000:.2f} ms")
        print(f"  Fine-tuned Model: {fine_tuned_time*1000:.2f} ± {fine_tuned_std*1000:.2f} ms")
        print(f"  Speedup: {(original_time/fine_tuned_time - 1)*100:.2f}%")
        
        # Get predictions from both models
        print("\nRunning inference for visual comparison...")
        
        # Run inference with original model
        original_results_dir = output_dir / "original"
        original_results = original_model.predict(source=args.source, conf=args.conf, save=True, project=str(output_dir), name="original")
        
        # Run inference with fine-tuned model
        fine_tuned_results_dir = output_dir / "fine_tuned"
        fine_tuned_results = fine_tuned_model.predict(source=args.source, conf=args.conf, save=True, project=str(output_dir), name="fine_tuned")
        
        # Create comparison images
        print("\nCreating comparison images...")
        comparison_dir = output_dir / "comparison"
        comparison_dir.mkdir(exist_ok=True)
        
        # Get image paths
        original_image_paths = sorted(list(original_results_dir.glob("*.jpg")))
        fine_tuned_image_paths = sorted(list(fine_tuned_results_dir.glob("*.jpg")))
        
        for i, (orig_path, fine_tuned_path) in enumerate(zip(original_image_paths, fine_tuned_image_paths)):
            output_path = comparison_dir / f"comparison_{i+1}.jpg"
            create_comparison_image(orig_path, fine_tuned_path, output_path)
            print(f"  Created comparison image: {output_path}")
        
        # Compare detection counts and classes
        print("\nDetection Comparison:")
        for orig_result, fine_tuned_result in zip(original_results, fine_tuned_results):
            image_name = Path(orig_result.path).name
            print(f"Image: {image_name}")
            
            # Original model detections
            orig_boxes = orig_result.boxes
            orig_class_counts = {}
            if len(orig_boxes) > 0:
                orig_classes = orig_result.names
                orig_class_ids = orig_boxes.cls.cpu().numpy().astype(int)
                for cls_id in orig_class_ids:
                    cls_name = orig_classes[cls_id]
                    orig_class_counts[cls_name] = orig_class_counts.get(cls_name, 0) + 1
            
            # Fine-tuned model detections
            fine_tuned_boxes = fine_tuned_result.boxes
            fine_tuned_class_counts = {}
            if len(fine_tuned_boxes) > 0:
                fine_tuned_classes = fine_tuned_result.names
                fine_tuned_class_ids = fine_tuned_boxes.cls.cpu().numpy().astype(int)
                for cls_id in fine_tuned_class_ids:
                    cls_name = fine_tuned_classes[cls_id]
                    fine_tuned_class_counts[cls_name] = fine_tuned_class_counts.get(cls_name, 0) + 1
            
            # Print detection counts
            print(f"  Original Model: {len(orig_boxes)} detections")
            for cls_name, count in orig_class_counts.items():
                print(f"    {cls_name}: {count}")
            
            print(f"  Fine-tuned Model: {len(fine_tuned_boxes)} detections")
            for cls_name, count in fine_tuned_class_counts.items():
                print(f"    {cls_name}: {count}")
            
            print("")
        
        print("Comparison completed successfully!")
        print(f"Results saved to {output_dir}")
    
    except Exception as e:
        print(f"Error during comparison: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main() 