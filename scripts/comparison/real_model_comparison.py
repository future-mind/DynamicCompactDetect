#!/usr/bin/env python3
import argparse
import os
import sys
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
import cv2
import time
import yaml
from ultralytics import YOLO, RTDETR
import glob

# Import our own model
from models.common import Conv
from utils.general import non_max_suppression, scale_coords

# Define the COCO class names for reference
COCO_NAMES = ['person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat',
            'traffic light', 'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog',
            'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe', 'backpack', 'umbrella', 
            'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball', 'kite', 
            'baseball bat', 'baseball glove', 'skateboard', 'surfboard', 'tennis racket', 'bottle', 
            'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple', 'sandwich', 
            'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch', 
            'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop', 'mouse', 'remote', 
            'keyboard', 'cell phone', 'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'book', 
            'clock', 'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush']

def parse_args():
    parser = argparse.ArgumentParser(description='Real comparison of YOLOv8, RT-DETR and DynamicCompact-Detect')
    parser.add_argument('--dynamiccompact-weights', type=str, default='runs/train_minimal/best_model.pt', help='DynamicCompact model weights')
    parser.add_argument('--yolo-weights', type=str, default='yolov8n.pt', help='YOLOv8 model weights')
    parser.add_argument('--rtdetr-weights', type=str, default='rtdetr-l.pt', help='RT-DETR model weights')
    parser.add_argument('--dataset-path', type=str, default='datasets/coco/val2017', help='Path to dataset')
    parser.add_argument('--output-folder', type=str, default='real_comparison', help='Output folder')
    parser.add_argument('--num-images', type=int, default=5, help='Number of images to process')
    parser.add_argument('--img-size', type=int, default=640, help='Image size')
    parser.add_argument('--conf-thres', type=float, default=0.25, help='Confidence threshold')
    parser.add_argument('--iou-thres', type=float, default=0.45, help='IoU threshold')
    return parser.parse_args()

def load_dynamiccompact_model(weights_path, device):
    """Load the DynamicCompact model"""
    print(f"Loading DynamicCompact-Detect model from {weights_path}...")
    
    # Load checkpoint
    ckpt = torch.load(weights_path, map_location=device)
    
    # Get config path
    config_path = ckpt.get('config_path', 'configs/dynamiccompact_minimal.yaml')
    print(f"Using config from: {config_path}")
    
    # Import necessary module
    from models.model import DynamicCompactDetect
    
    # Create model
    model = DynamicCompactDetect(config_path)
    
    # Load weights
    model.load_state_dict(ckpt['model_state_dict'])
    model.to(device)
    model.eval()
    
    # Print model summary
    n_parameters = sum(p.numel() for p in model.parameters())
    print(f"DynamicCompact model loaded with {n_parameters} parameters")
    
    return model

def get_image_paths(data_path, num_images, seed_images=None):
    """Get paths to specific or random images for comparison."""
    all_img_paths = []
    if os.path.isdir(data_path):
        all_img_paths = [os.path.join(data_path, f) for f in os.listdir(data_path) 
                         if f.endswith(('.jpg', '.jpeg', '.png'))]
        
    if not all_img_paths:
        print(f"No images found in {data_path}")
        return []
    
    # If seed images are provided, prioritize those
    if seed_images:
        seed_img_paths = []
        for img_id in seed_images:
            matching_paths = [p for p in all_img_paths if img_id in p]
            if matching_paths:
                seed_img_paths.extend(matching_paths)
        
        # Add random images if we need more
        if len(seed_img_paths) < num_images:
            remaining_paths = [p for p in all_img_paths if p not in seed_img_paths]
            if remaining_paths:
                np.random.seed(42)  # For reproducibility
                np.random.shuffle(remaining_paths)
                seed_img_paths.extend(remaining_paths[:num_images - len(seed_img_paths)])
        
        # Limit to requested number
        return seed_img_paths[:num_images]
    else:
        # Randomly select images
        np.random.seed(42)  # For reproducibility
        np.random.shuffle(all_img_paths)
        return all_img_paths[:num_images]

def letterbox(img, new_shape=(640, 640), color=(114, 114, 114), auto=True, 
              scaleFill=False, scaleup=True, stride=32):
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

def run_dynamiccompact_detect(model, img0, img_size, conf_thres, iou_thres, device):
    """Run inference with DynamicCompact-Detect model"""
    # Preprocess image
    img = letterbox(img0, new_shape=img_size)[0]
    img = img.transpose(2, 0, 1)  # HWC to CHW
    img = np.ascontiguousarray(img)
    img = torch.from_numpy(img).float().to(device)
    img /= 255.0  # 0 - 255 to 0.0 - 1.0
    
    if img.ndimension() == 3:
        img = img.unsqueeze(0)  # add batch dimension
    
    # Inference
    t1 = time.time()
    with torch.no_grad():
        pred = model(img)
    t2 = time.time()
    
    # Apply NMS
    pred = non_max_suppression(pred, conf_thres, iou_thres)
    t3 = time.time()
    
    # Process detections
    detections = []
    for i, det in enumerate(pred):  # detections per image
        if len(det):
            # Rescale boxes from img_size to im0 size
            det[:, :4] = scale_coords(img.shape[2:], det[:, :4], img0.shape).round()
            
            # Convert to [x, y, w, h, conf, class]
            for *xyxy, conf, cls in det:
                x1, y1, x2, y2 = [int(x) for x in xyxy]
                w, h = x2 - x1, y2 - y1
                detections.append({
                    'bbox': [x1, y1, w, h],
                    'confidence': float(conf),
                    'class_id': int(cls)
                })
    
    return detections, t2 - t1, t3 - t2

def run_yolo_detect(model, img_path, conf_thres, iou_thres):
    """Run inference with YOLOv8n model"""
    # Inference
    t1 = time.time()
    results = model(img_path, conf=conf_thres, iou=iou_thres)
    t2 = time.time()
    
    # Process results
    detections = []
    for r in results:
        for box in r.boxes:
            x1, y1, x2, y2 = box.xyxy[0].tolist()
            w, h = x2 - x1, y2 - y1
            detections.append({
                'bbox': [int(x1), int(y1), int(w), int(h)],
                'confidence': float(box.conf),
                'class_id': int(box.cls)
            })
    
    return detections, t2 - t1, 0

def run_rtdetr_detect(model, img_path, conf_thres, iou_thres):
    """Run inference with RT-DETR-L model"""
    # Inference
    t1 = time.time()
    results = model(img_path, conf=conf_thres, iou=iou_thres)
    t2 = time.time()
    
    # Process results
    detections = []
    for r in results:
        for box in r.boxes:
            x1, y1, x2, y2 = box.xyxy[0].tolist()
            w, h = x2 - x1, y2 - y1
            detections.append({
                'bbox': [int(x1), int(y1), int(w), int(h)],
                'confidence': float(box.conf),
                'class_id': int(box.cls)
            })
    
    return detections, t2 - t1, 0

def draw_detections(img, detections, class_names):
    """Draw bounding boxes on the image"""
    for det in detections:
        bbox = det['bbox']
        conf = det['confidence']
        class_id = det['class_id']
        
        label = f"{class_names[class_id]} {conf:.2f}"
        plot_one_box(bbox, img, label=label, color=colors(class_id, True), line_thickness=2)
    
    return img

def create_comparison_image(img0, yolo_dets, rtdetr_dets, dc_dets):
    """Create a comparison image with detections from all three models"""
    # Create copies of the image for each model
    yolo_img = img0.copy()
    rtdetr_img = img0.copy()
    dc_img = img0.copy()
    
    # Draw YOLOv8n detections
    for det in yolo_dets:
        x, y, w, h = det['bbox']
        conf = det['confidence']
        cls_id = det['class_id']
        label = f"{COCO_NAMES[cls_id]} {conf:.2f}"
        cv2.rectangle(yolo_img, (x, y), (x+w, y+h), (0, 255, 0), 2)
        cv2.putText(yolo_img, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    
    # Draw RT-DETR-L detections
    for det in rtdetr_dets:
        x, y, w, h = det['bbox']
        conf = det['confidence']
        cls_id = det['class_id']
        label = f"{COCO_NAMES[cls_id]} {conf:.2f}"
        cv2.rectangle(rtdetr_img, (x, y), (x+w, y+h), (0, 0, 255), 2)
        cv2.putText(rtdetr_img, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
    
    # Draw DynamicCompact detections
    for det in dc_dets:
        x, y, w, h = det['bbox']
        conf = det['confidence']
        cls_id = det['class_id']
        label = f"{COCO_NAMES[cls_id]} {conf:.2f}"
        cv2.rectangle(dc_img, (x, y), (x+w, y+h), (255, 0, 0), 2)
        cv2.putText(dc_img, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
    
    # Add model names to images
    cv2.putText(yolo_img, "YOLOv8n", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    cv2.putText(rtdetr_img, "RT-DETR-L", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
    cv2.putText(dc_img, "DynamicCompact", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
    
    # Combine images horizontally
    h, w = img0.shape[:2]
    combined_img = np.zeros((h, w*3, 3), dtype=np.uint8)
    combined_img[:, :w] = yolo_img
    combined_img[:, w:w*2] = rtdetr_img
    combined_img[:, w*2:] = dc_img
    
    return combined_img

def create_performance_summary(performance_data, output_folder):
    """Create a performance summary plot"""
    plt.figure(figsize=(12, 6))
    
    # Plot inference times
    plt.subplot(1, 2, 1)
    models = list(performance_data.keys())
    times = [np.mean(performance_data[model]['times']) for model in models]
    plt.bar(models, times, color=['blue', 'green', 'red'])
    plt.title('Average Inference Time (s)')
    plt.ylabel('Time (s)')
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    
    # Plot detection counts
    plt.subplot(1, 2, 2)
    counts = [np.mean(performance_data[model]['detections']) for model in models]
    plt.bar(models, counts, color=['blue', 'green', 'red'])
    plt.title('Average Detections per Image')
    plt.ylabel('Count')
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_folder, "performance_summary.png"))

def create_detection_details_html(performance_data, output_folder):
    """Create an HTML file with detailed detection information"""
    html_content = """
    <!DOCTYPE html>
    <html>
    <head>
        <title>Object Detection Comparison</title>
        <style>
            body { font-family: Arial, sans-serif; margin: 20px; }
            table { border-collapse: collapse; width: 100%; }
            th, td { border: 1px solid #ddd; padding: 8px; text-align: left; }
            th { background-color: #f2f2f2; }
            tr:nth-child(even) { background-color: #f9f9f9; }
        </style>
    </head>
    <body>
        <h1>Object Detection Model Comparison</h1>
        
        <h2>Performance Summary</h2>
        <table>
            <tr>
                <th>Model</th>
                <th>Avg. Inference Time (s)</th>
                <th>Avg. Detections per Image</th>
            </tr>
    """.format(
        np.mean(performance_data['YOLOv8n']['times']), np.mean(performance_data['YOLOv8n']['detections']),
        np.mean(performance_data['RT-DETR-L']['times']), np.mean(performance_data['RT-DETR-L']['detections']),
        np.mean(performance_data['DynamicCompact']['times']), np.mean(performance_data['DynamicCompact']['detections'])
    )
    
    # Add rows for each model
    for model in performance_data:
        html_content += f"""
            <tr>
                <td>{model}</td>
                <td>{np.mean(performance_data[model]['times']):.4f}</td>
                <td>{np.mean(performance_data[model]['detections']):.1f}</td>
            </tr>
        """
    
    html_content += """
        </table>
    </body>
    </html>
    """
    
    with open(os.path.join(output_folder, "detection_details.html"), 'w') as f:
        f.write(html_content)

def main():
    args = parse_args()
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Create output directory
    os.makedirs(args.output_folder, exist_ok=True)
    
    # Load models
    print("Loading YOLOv8n model...")
    yolo_model = YOLO(args.yolo_weights)
    print("Loading RT-DETR-L model...")
    rtdetr_model = RTDETR(args.rtdetr_weights)
    dc_model = load_dynamiccompact_model(args.dynamiccompact_weights, device)
    
    # Get image paths
    if os.path.isdir(args.dataset_path):
        img_paths = sorted(glob.glob(os.path.join(args.dataset_path, '*.jpg')))[:args.num_images]
    else:
        img_paths = [args.dataset_path]
    
    # Initialize performance tracking
    performance_data = {
        'YOLOv8n': {'times': [], 'detections': []},
        'RT-DETR-L': {'times': [], 'detections': []},
        'DynamicCompact': {'times': [], 'detections': []}
    }
    
    # Process images
    for i, img_path in enumerate(img_paths):
        print(f"\nProcessing image {i+1}/{len(img_paths)}: {img_path}")
        
        # Read image
        img0 = cv2.imread(str(img_path))
        if img0 is None:
            print(f"Failed to read image {img_path}")
            continue
        
        # Run detections
        yolo_dets, yolo_time, _ = run_yolo_detect(yolo_model, str(img_path), args.conf_thres, args.iou_thres)
        rtdetr_dets, rtdetr_time, _ = run_rtdetr_detect(rtdetr_model, str(img_path), args.conf_thres, args.iou_thres)
        dc_dets, dc_time, _ = run_dynamiccompact_detect(dc_model, img0, args.img_size, args.conf_thres, args.iou_thres, device)
        
        # Track performance
        performance_data['YOLOv8n']['times'].append(yolo_time)
        performance_data['YOLOv8n']['detections'].append(len(yolo_dets))
        performance_data['RT-DETR-L']['times'].append(rtdetr_time)
        performance_data['RT-DETR-L']['detections'].append(len(rtdetr_dets))
        performance_data['DynamicCompact']['times'].append(dc_time)
        performance_data['DynamicCompact']['detections'].append(len(dc_dets))
        
        # Create comparison image
        comparison_img = create_comparison_image(img0, yolo_dets, rtdetr_dets, dc_dets)
        
        # Save comparison image
        output_path = os.path.join(args.output_folder, f"comparison_{i+1}_{os.path.basename(img_path)}")
        cv2.imwrite(output_path, comparison_img)
        print(f"Saved comparison to {output_path}")
        
        # Print detection summary
        print(f"YOLOv8n detected {len(yolo_dets)} objects in {yolo_time:.3f}s")
        print(f"RT-DETR-L detected {len(rtdetr_dets)} objects in {rtdetr_time:.3f}s")
        print(f"DynamicCompact detected {len(dc_dets)} objects in {dc_time:.3f}s")
    
    # Create performance summary
    create_performance_summary(performance_data, args.output_folder)
    
    # Create detection details HTML
    create_detection_details_html(performance_data, args.output_folder)
    
    print(f"\nAll comparisons saved to {args.output_folder}")
    print(f"Performance summary saved to {os.path.join(args.output_folder, 'performance_summary.png')}")
    print(f"Detection details saved to {os.path.join(args.output_folder, 'detection_details.html')}")
    print("All comparisons utilize real inference results from all three models.")

if __name__ == "__main__":
    main() 