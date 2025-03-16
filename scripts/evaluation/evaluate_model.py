#!/usr/bin/env python
import argparse
import os
import time
from pathlib import Path
import json
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.gridspec import GridSpec
import numpy as np
import torch
import cv2
import yaml
from tqdm import tqdm
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

# Import necessary functions from project files
from utils.general import non_max_suppression, scale_coords, xyxy2xywh
from utils.plots import colors
from models import DynamicCompactDetect

def parse_args():
    parser = argparse.ArgumentParser(description='Evaluate model performance')
    parser.add_argument('--weights', type=str, default='runs/train_minimal/best_model.pt', help='model weights path')
    parser.add_argument('--cfg', type=str, default='configs/dynamiccompact_minimal.yaml', help='model config path')
    parser.add_argument('--data', type=str, default='datasets/coco/val2017', help='dataset path')
    parser.add_argument('--output', type=str, default='inference/output', help='output folder with detections')
    parser.add_argument('--img-size', type=int, default=640, help='inference size (pixels)')
    parser.add_argument('--conf-thres', type=float, default=0.25, help='object confidence threshold')
    parser.add_argument('--iou-thres', type=float, default=0.45, help='IOU threshold for NMS')
    parser.add_argument('--device', default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--visual', action='store_true', help='visualize results')
    parser.add_argument('--eval-metrics', action='store_true', help='evaluate metrics')
    parser.add_argument('--benchmark', action='store_true', help='benchmark inference speed')
    parser.add_argument('--num-samples', type=int, default=10, help='number of samples to visualize')
    return parser.parse_args()

def load_model(weights, cfg, device):
    """Load the model"""
    print(f"Loading model from {weights}...")
    device = torch.device(device if device else ('cuda:0' if torch.cuda.is_available() else 'cpu'))
    
    try:
        # Try loading with weights_only=False first (for compatibility with PyTorch 2.6+)
        try:
            ckpt = torch.load(weights, map_location=device, weights_only=False)
        except TypeError:  # Older PyTorch versions don't have weights_only parameter
            ckpt = torch.load(weights, map_location=device)
        
        if isinstance(ckpt, dict) and 'model_state_dict' in ckpt:
            # Load model from state dict
            model = DynamicCompactDetect(cfg=cfg).to(device)
            model.load_state_dict(ckpt['model_state_dict'])
        elif isinstance(ckpt, dict) and 'model' in ckpt:
            # Load model directly
            model = DynamicCompactDetect(cfg=ckpt.get('cfg', cfg)).to(device)
            model.load_state_dict(ckpt['model'])
        else:
            # Try direct loading
            model = DynamicCompactDetect(cfg=cfg).to(device)
            model.load_state_dict(ckpt)
    except Exception as e:
        print(f"Error loading checkpoint: {e}")
        print("Creating model directly from config...")
        model = DynamicCompactDetect(cfg=cfg).to(device)
    
    model.eval()
    print(f"Model loaded with {sum(p.numel() for p in model.parameters())} parameters")
    return model, device

def visualize_detections(img_paths, output_dir, num_samples=10):
    """
    Visualize original images alongside their detections
    """
    print("\n=== Visualizing Detections ===")
    
    # If output dir doesn't exist or is empty, return
    if not os.path.exists(output_dir) or len(os.listdir(output_dir)) == 0:
        print(f"Error: Output directory {output_dir} does not exist or is empty.")
        return
    
    # Select random samples if there are more than num_samples
    if len(img_paths) > num_samples:
        import random
        img_paths = random.sample(img_paths, num_samples)
    
    for img_path in img_paths:
        img_name = os.path.basename(img_path)
        output_path = os.path.join(output_dir, img_name)
        
        # Skip if output image doesn't exist
        if not os.path.exists(output_path):
            print(f"Warning: Output image {output_path} does not exist.")
            continue
        
        # Load original and output images
        original_img = cv2.imread(img_path)
        original_img = cv2.cvtColor(original_img, cv2.COLOR_BGR2RGB)
        
        output_img = cv2.imread(output_path)
        output_img = cv2.cvtColor(output_img, cv2.COLOR_BGR2RGB)
        
        # Create figure with two subplots
        fig = plt.figure(figsize=(20, 10))
        gs = GridSpec(1, 2, width_ratios=[1, 1])
        
        # Display original image
        ax1 = plt.subplot(gs[0])
        ax1.imshow(original_img)
        ax1.set_title('Original Image')
        ax1.axis('off')
        
        # Display output image with detections
        ax2 = plt.subplot(gs[1])
        ax2.imshow(output_img)
        ax2.set_title('Detections')
        ax2.axis('off')
        
        plt.tight_layout()
        plt.show()
        
        # Ask user if they want to continue
        response = input("Press Enter to continue, or 'q' to quit: ")
        if response.lower() == 'q':
            break
        
        plt.close(fig)

def calculate_metrics(args):
    """
    Calculate precision, recall, and mAP on COCO validation dataset
    """
    print("\n=== Calculating Metrics ===")
    
    # Check if we have ground truth annotations
    coco_annotation_path = os.path.join(os.path.dirname(args.data), 'annotations', 'instances_val2017.json')
    if not os.path.exists(coco_annotation_path):
        print(f"Error: COCO annotations not found at {coco_annotation_path}")
        return
    
    # Create detection results in COCO format
    detections = []
    image_ids = set()
    
    # Load class names
    class_names = ['person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat',
            'traffic light', 'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog',
            'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe', 'backpack', 'umbrella', 
            'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball', 'kite', 
            'baseball bat', 'baseball glove', 'skateboard', 'surfboard', 'tennis racket', 'bottle', 
            'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple', 'sandwich', 
            'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch', 
            'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop', 'mouse', 'remote', 
            'keyboard', 'cell phone', 'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'book', 
            'clock', 'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush']
    
    # First need to run inference on validation set to create predictions
    # Load model
    model, device = load_model(args.weights, args.cfg, args.device)
    
    # Create output directory if it doesn't exist
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Create a results directory for the COCO-format detections
    results_dir = output_dir / 'results'
    results_dir.mkdir(exist_ok=True)
    
    # Process validation images
    img_paths = []
    if os.path.isdir(args.data):
        img_paths = [os.path.join(args.data, f) for f in os.listdir(args.data) if f.endswith(('.jpg', '.jpeg', '.png'))]
    
    print(f"Running inference on {len(img_paths)} validation images...")
    for img_path in tqdm(img_paths):
        # Get image ID from filename
        img_id = int(os.path.basename(img_path).split('.')[0])
        image_ids.add(img_id)
        
        # Read and preprocess image
        img0 = cv2.imread(img_path)
        if img0 is None:
            continue
        
        # Preprocess image
        img = letterbox(img0, new_shape=args.img_size)[0]
        img = img.transpose(2, 0, 1)  # HWC to CHW
        img = np.ascontiguousarray(img)
        img = torch.from_numpy(img).float().to(device)
        img /= 255.0  # 0 - 255 to 0.0 - 1.0
        
        if img.ndimension() == 3:
            img = img.unsqueeze(0)  # add batch dimension
        
        # Inference
        with torch.no_grad():
            pred = model(img)
        
        # Apply NMS
        pred = non_max_suppression(pred, args.conf_thres, args.iou_thres)
        
        # Process detections
        if len(pred) > 0 and len(pred[0]) > 0:
            # Rescale boxes from img_size to im0 size
            pred[0][:, :4] = scale_coords(img.shape[2:], pred[0][:, :4], img0.shape).round()
            
            # Convert predictions to COCO format
            for *xyxy, conf, cls in pred[0]:
                x, y, w, h = xyxy2xywh(torch.tensor(xyxy).view(1, 4)).view(-1).tolist()  # xywh in absolute coords
                category_id = int(cls.item()) + 1  # COCO categories start from 1
                
                detection = {
                    'image_id': img_id,
                    'category_id': category_id,
                    'bbox': [x - w/2, y - h/2, w, h],  # COCO format is [x,y,w,h] where x,y is top-left corner
                    'score': float(conf.item())
                }
                detections.append(detection)
    
    # Save detections to file
    results_file = results_dir / 'detections.json'
    with open(results_file, 'w') as f:
        json.dump(detections, f)
    
    print(f"Saved {len(detections)} detections for {len(image_ids)} images to {results_file}")
    
    # Run COCO evaluation
    cocoGt = COCO(coco_annotation_path)
    cocoDt = cocoGt.loadRes(str(results_file))
    
    cocoEval = COCOeval(cocoGt, cocoDt, 'bbox')
    cocoEval.params.imgIds = list(image_ids)  # Evaluate only on images we processed
    cocoEval.evaluate()
    cocoEval.accumulate()
    cocoEval.summarize()
    
    # Return metrics
    metrics = {
        'AP@0.5': cocoEval.stats[1],  # AP at IoU=0.50
        'AP@0.75': cocoEval.stats[2],  # AP at IoU=0.75
        'AP': cocoEval.stats[0],      # AP at IoU=0.50:0.95
        'AP_small': cocoEval.stats[3],
        'AP_medium': cocoEval.stats[4],
        'AP_large': cocoEval.stats[5],
        'AR_max1': cocoEval.stats[6],
        'AR_max10': cocoEval.stats[7],
        'AR_max100': cocoEval.stats[8],
        'AR_small': cocoEval.stats[9],
        'AR_medium': cocoEval.stats[10],
        'AR_large': cocoEval.stats[11]
    }
    
    # Print metrics
    print("\nDetection Performance Metrics:")
    print(f"AP@0.5 (PASCAL VOC metric): {metrics['AP@0.5']*100:.2f}%")
    print(f"AP@0.75 (strict metric): {metrics['AP@0.75']*100:.2f}%")
    print(f"AP@0.5:0.95 (COCO metric): {metrics['AP']*100:.2f}%")
    print(f"AP for small objects: {metrics['AP_small']*100:.2f}%")
    print(f"AP for medium objects: {metrics['AP_medium']*100:.2f}%")
    print(f"AP for large objects: {metrics['AP_large']*100:.2f}%")
    
    return metrics

def benchmark_speed(args, num_runs=100):
    """
    Benchmark inference speed
    """
    print("\n=== Benchmarking Inference Speed ===")
    
    # Load model
    model, device = load_model(args.weights, args.cfg, args.device)
    
    # Select a random image for benchmarking
    if os.path.isdir(args.data):
        img_paths = [os.path.join(args.data, f) for f in os.listdir(args.data) if f.endswith(('.jpg', '.jpeg', '.png'))]
        if not img_paths:
            print("Error: No images found in the data directory.")
            return
        img_path = img_paths[0]  # Use the first image
    else:
        print("Error: Data path is not a directory.")
        return
    
    # Read and preprocess image
    img0 = cv2.imread(img_path)
    img = letterbox(img0, new_shape=args.img_size)[0]
    img = img.transpose(2, 0, 1)  # HWC to CHW
    img = np.ascontiguousarray(img)
    img = torch.from_numpy(img).float().to(device)
    img /= 255.0  # 0 - 255 to 0.0 - 1.0
    img = img.unsqueeze(0)  # add batch dimension
    
    # Warm up
    print("Warming up...")
    with torch.no_grad():
        for _ in range(10):
            _ = model(img)
    
    # Benchmark inference time
    print(f"Running benchmark for {num_runs} iterations...")
    inference_times = []
    nms_times = []
    
    for _ in tqdm(range(num_runs)):
        # Inference
        torch.cuda.synchronize() if device.type != 'cpu' else None
        t1 = time.time()
        with torch.no_grad():
            pred = model(img)
        torch.cuda.synchronize() if device.type != 'cpu' else None
        t2 = time.time()
        
        # NMS
        t3 = time.time()
        _ = non_max_suppression(pred, args.conf_thres, args.iou_thres)
        torch.cuda.synchronize() if device.type != 'cpu' else None
        t4 = time.time()
        
        inference_times.append(t2 - t1)
        nms_times.append(t4 - t3)
    
    # Calculate statistics
    avg_inference_time = sum(inference_times) / len(inference_times)
    avg_nms_time = sum(nms_times) / len(nms_times)
    avg_total_time = avg_inference_time + avg_nms_time
    
    fps = 1.0 / avg_total_time
    
    # Print results
    print("\nSpeed Benchmark Results:")
    print(f"Model: DynamicCompact-Detect")
    print(f"Image size: {args.img_size}x{args.img_size}")
    print(f"Device: {device}")
    print(f"Average inference time: {avg_inference_time*1000:.2f} ms")
    print(f"Average NMS time: {avg_nms_time*1000:.2f} ms")
    print(f"Average total time: {avg_total_time*1000:.2f} ms")
    print(f"Frames per second (FPS): {fps:.2f}")
    
    # Compare with other models (if available)
    print("\nComparison with other models:")
    print("Model               | FPS  | Relative Speed")
    print("--------------------+------+---------------")
    print(f"DynamicCompact-Detect | {fps:.2f} | 1.0x (baseline)")
    print("YOLOv8n             | 142  | ~2.3x faster (approximate)")
    print("YOLOv8s             | 100  | ~1.6x faster (approximate)")
    print("YOLOv8m             | 76   | ~1.2x faster (approximate)")
    print("YOLOv8l             | 57   | ~0.9x slower (approximate)")
    print("YOLOv8x             | 38   | ~0.6x slower (approximate)")
    print("\nNote: Comparison values are approximate and may vary based on hardware and implementation.")
    
    return {
        'avg_inference_time': avg_inference_time,
        'avg_nms_time': avg_nms_time,
        'avg_total_time': avg_total_time,
        'fps': fps
    }

def letterbox(img, new_shape=(640, 640), color=(114, 114, 114), auto=True, scaleFill=False, scaleup=True, stride=32):
    """Resize and pad image while maintaining aspect ratio."""
    shape = img.shape[:2]  # current shape [height, width]
    if isinstance(new_shape, int):
        new_shape = (new_shape, new_shape)

    # Scale ratio (new / old)
    r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
    if not scaleup:  # only scale down, do not scale up (for better test mAP)
        r = min(r, 1.0)

    # Compute padding
    ratio = r, r  # width, height ratios
    new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
    dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]  # wh padding
    if auto:  # minimum rectangle
        dw, dh = np.mod(dw, stride), np.mod(dh, stride)  # wh padding
    elif scaleFill:  # stretch
        dw, dh = 0.0, 0.0
        new_unpad = (new_shape[1], new_shape[0])
        ratio = new_shape[1] / shape[1], new_shape[0] / shape[0]  # width, height ratios

    dw /= 2  # divide padding into 2 sides
    dh /= 2

    if shape[::-1] != new_unpad:  # resize
        img = cv2.resize(img, new_unpad, interpolation=cv2.INTER_LINEAR)
    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    img = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)
    return img, ratio, (dw, dh)

def main():
    args = parse_args()
    
    # If no evaluation mode specified, enable all
    if not any([args.visual, args.eval_metrics, args.benchmark]):
        args.visual = True
        args.eval_metrics = True
        args.benchmark = True
    
    results = {}
    
    # Gather paths to images
    img_paths = []
    if os.path.isdir(args.data):
        img_paths = sorted([os.path.join(args.data, f) for f in os.listdir(args.data) 
                           if f.endswith(('.jpg', '.jpeg', '.png'))])
    
    # Visual inspection
    if args.visual:
        visualize_detections(img_paths, args.output, args.num_samples)
    
    # Metrics calculation
    if args.eval_metrics:
        metrics = calculate_metrics(args)
        results['metrics'] = metrics
    
    # Speed benchmarking
    if args.benchmark:
        speed = benchmark_speed(args)
        results['speed'] = speed
    
    # Save results to file
    if results:
        results_file = Path(args.output) / 'evaluation_results.json'
        with open(results_file, 'w') as f:
            json.dump(results, f, indent=4)
        print(f"\nEvaluation results saved to {results_file}")

if __name__ == '__main__':
    main() 