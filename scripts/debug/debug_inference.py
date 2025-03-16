#!/usr/bin/env python
import argparse
import os
import time
import cv2
import numpy as np
import torch
import yaml

from utils.datasets import letterbox
from utils.general import non_max_suppression, scale_coords
from utils.plots import Colors, plot_one_box

# Initialize colors
colors = Colors()

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', type=str, default='runs/train_minimal/best_model.pt', help='model weights')
    parser.add_argument('--img-path', type=str, default='datasets/coco/val2017/000000119445.jpg', help='image path')
    parser.add_argument('--img-size', type=int, default=640, help='inference size (pixels)')
    parser.add_argument('--conf-thres', type=float, default=0.01, help='object confidence threshold')
    parser.add_argument('--iou-thres', type=float, default=0.45, help='IOU threshold for NMS')
    parser.add_argument('--output', type=str, default='debug_output', help='output folder')
    return parser.parse_args()

def main():
    args = parse_args()
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Create output folder
    if not os.path.exists(args.output):
        os.makedirs(args.output)
    
    # Load model
    print(f"Loading model from {args.weights}...")
    ckpt = torch.load(args.weights, map_location=device)
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
    print(f"Model loaded with {n_parameters} parameters")
    
    # Load class names
    class_names = ['person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat', 'traffic light',
                  'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow',
                  'elephant', 'bear', 'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee',
                  'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard',
                  'tennis racket', 'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple',
                  'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch',
                  'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone',
                  'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'book', 'clock', 'vase', 'scissors', 'teddy bear',
                  'hair drier', 'toothbrush']
    
    # Load image
    img_path = args.img_path
    img0 = cv2.imread(img_path)  # BGR
    if img0 is None:
        print(f"Failed to read image {img_path}")
        return
    
    # Preprocess image
    img = letterbox(img0, new_shape=args.img_size)[0]
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
    
    # Debug prints
    print(f"Model output shape: {pred.shape}")
    print(f"Model output min: {pred.min().item()}, max: {pred.max().item()}")
    print(f"Sample values: {pred[0, :5, 0, 0].cpu().numpy()}")
    
    # Check objectness scores
    if pred.dim() == 4:
        batch_size, channels, height, width = pred.shape
        pred_reshaped = pred.permute(0, 2, 3, 1).contiguous()
        pred_reshaped = pred_reshaped.view(batch_size, -1, channels)
        obj_scores = pred_reshaped[0, :, 4]
        print(f"Objectness scores - min: {obj_scores.min().item()}, max: {obj_scores.max().item()}")
        print(f"Boxes with objectness > {args.conf_thres}: {(obj_scores > args.conf_thres).sum().item()}")
        
        # Top 5 boxes
        top_indices = obj_scores.argsort(descending=True)[:5]
        for i, idx in enumerate(top_indices):
            box = pred_reshaped[0, idx, :4].cpu().numpy()
            obj_score = obj_scores[idx].item()
            class_scores = pred_reshaped[0, idx, 5:].cpu().numpy()
            max_class = class_scores.argmax()
            max_class_score = class_scores[max_class]
            print(f"Box {i+1}: obj={obj_score:.4f}, class={max_class} (conf={max_class_score:.4f}), box={box}")
    
    # Apply NMS
    t_nms_start = time.time()
    pred = non_max_suppression(pred, args.conf_thres, args.iou_thres)
    t_nms_end = time.time()
    
    # Debug prints after NMS
    print(f"After NMS: {len(pred)} batch items, first item has {len(pred[0]) if len(pred) > 0 else 0} detections")
    if len(pred) > 0 and len(pred[0]) > 0:
        print(f"First detection: {pred[0][0]}")
    
    # Process detections
    detections = []
    for i, det in enumerate(pred):  # detections per image
        im0 = img0.copy()
        
        if len(det):
            # Rescale boxes from img_size to im0 size
            det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0.shape).round()
            
            # Print results
            print(f"Detected {len(det)} objects:")
            
            # Draw bounding boxes and labels
            for *xyxy, conf, cls in det:
                x1, y1, x2, y2 = [int(x) for x in xyxy]
                w, h = x2 - x1, y2 - y1
                label = f"{class_names[int(cls)]} {conf:.2f}"
                plot_one_box(xyxy, im0, label=label, color=colors(int(cls), True), line_thickness=2)
                print(f"  {label} at {xyxy}")
                
                detections.append({
                    'bbox': [x1, y1, w, h],
                    'confidence': float(conf),
                    'class_id': int(cls)
                })
        
        # Save results
        output_path = os.path.join(args.output, os.path.basename(img_path))
        cv2.imwrite(output_path, im0)
        print(f"Saved {output_path}")
    
    # Print summary
    detection_classes = [class_names[d['class_id']] for d in detections]
    detection_summary = [f"{detection_classes.count(c)} {c}s" for c in set(detection_classes) if detection_classes.count(c) > 0]
    print(f"Summary: {', '.join(detection_summary)}")
    print(f"Done. Inference time: {t2-t1:.4f}s, NMS time: {t_nms_end-t_nms_start:.4f}s")
    
    # Return detections for comparison
    return detections

if __name__ == "__main__":
    main() 