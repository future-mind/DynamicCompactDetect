#!/usr/bin/env python
import argparse
import os
import time
from pathlib import Path
import yaml
import numpy as np
import torch
import cv2
import torch.nn as nn

from utils.datasets import letterbox
from utils.general import non_max_suppression, scale_coords
from utils.plots import plot_one_box, colors

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', type=str, default='runs/train_minimal/best_model.pt', help='model weights')
    parser.add_argument('--img-path', type=str, default='datasets/coco/val2017/000000119445.jpg', help='image path')
    parser.add_argument('--img-size', type=int, default=640, help='image size')
    parser.add_argument('--conf-thres', type=float, default=0.01, help='confidence threshold')
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
    
    # Load configuration
    with open(config_path, 'r') as f:
        model_cfg = yaml.safe_load(f)
    
    # Import necessary module
    from models.common import Conv
    
    # Create model
    class Model(nn.Module):
        def __init__(self):
            super().__init__()
            layers = []
            in_channels = 3  # RGB input
            
            # Build the backbone
            for item in model_cfg['backbone']:
                # Format: [from, repeats, module, args]
                from_layer, repeats, module_name, args = item
                
                if module_name == 'Conv':
                    # args should be [out_channels, kernel_size, stride]
                    out_channels, kernel_size, stride = args
                    layers.append(Conv(in_channels, out_channels, kernel_size, stride))
                    in_channels = out_channels
            
            # Build the head
            for item in model_cfg['head']:
                # Format: [from, repeats, module, args]
                from_layer, repeats, module_name, args = item
                
                if module_name == 'Conv':
                    # args should be [out_channels, kernel_size, stride]
                    out_channels, kernel_size, stride = args
                    layers.append(Conv(in_channels, out_channels, kernel_size, stride))
                    in_channels = out_channels
            
            self.model = nn.Sequential(*layers)
        
        def forward(self, x):
            return self.model(x)
    
    model = Model()
    
    # Load weights
    model.load_state_dict(ckpt['model_state_dict'])
    model.to(device)
    model.eval()
    
    # Print model summary
    n_parameters = sum(p.numel() for p in model.parameters())
    print(f"DynamicCompact model loaded with {n_parameters} parameters")
    
    return model

def main():
    # Parse arguments
    args = parse_args()
    
    # Set device
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    
    # Load model
    model = load_dynamiccompact_model(args.weights, device)
    
    # Read image
    img_path = args.img_path
    img0 = cv2.imread(img_path)
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
    print(f"\nModel output shape: {pred.shape}")
    print(f"Model output min: {pred.min().item()}, max: {pred.max().item()}")
    print(f"Model output sample values (first 5 channels at 0,0): {pred[0, :5, 0, 0].cpu().numpy()}")
    
    # Analyze output structure
    print("\nAnalyzing output structure...")
    
    # Assuming YOLO format output with [x, y, w, h, obj_conf, class_conf1, class_conf2, ...]
    # Reshape to [batch_size, num_boxes, num_classes + 5]
    num_classes = pred.shape[1] - 5
    print(f"Detected {num_classes} classes in output")
    
    # Check if output has expected YOLO format
    if pred.dim() == 4:  # [batch, channels, height, width]
        print(f"Output is in grid format: {pred.shape}")
        
        # Reshape to detection format
        batch_size, channels, height, width = pred.shape
        pred_reshaped = pred.permute(0, 2, 3, 1).contiguous()  # [batch, height, width, channels]
        pred_reshaped = pred_reshaped.view(batch_size, -1, channels)  # [batch, height*width, channels]
        
        print(f"Reshaped to detection format: {pred_reshaped.shape}")
        
        # Check objectness scores
        obj_scores = pred_reshaped[0, :, 4]
        print(f"Objectness scores - min: {obj_scores.min().item()}, max: {obj_scores.max().item()}")
        print(f"Number of boxes with objectness > {args.conf_thres}: {(obj_scores > args.conf_thres).sum().item()}")
        
        # Get top 5 boxes with highest objectness
        top_obj_indices = obj_scores.argsort(descending=True)[:5]
        print("\nTop 5 boxes with highest objectness:")
        for i, idx in enumerate(top_obj_indices):
            box = pred_reshaped[0, idx, :4].cpu().numpy()
            obj_score = obj_scores[idx].item()
            class_scores = pred_reshaped[0, idx, 5:].cpu().numpy()
            max_class = class_scores.argmax()
            max_class_score = class_scores[max_class]
            print(f"Box {i+1}: obj={obj_score:.4f}, class={max_class} (conf={max_class_score:.4f}), box={box}")
    
    # Apply NMS
    print("\nApplying NMS...")
    try:
        nms_pred = non_max_suppression(pred, args.conf_thres, args.iou_thres)
        print(f"After NMS: {len(nms_pred)} batch items, first item has {len(nms_pred[0]) if len(nms_pred) > 0 else 0} detections")
        
        if len(nms_pred) > 0 and len(nms_pred[0]) > 0:
            print(f"First detection: {nms_pred[0][0]}")
            
            # Draw detections on image
            for *xyxy, conf, cls in nms_pred[0]:
                # Rescale boxes from img_size to im0 size
                xyxy_scaled = scale_coords(img.shape[2:], torch.tensor(xyxy).unsqueeze(0), img0.shape).round()[0]
                
                # Draw box
                label = f"Class {int(cls)} {conf:.2f}"
                plot_one_box(xyxy_scaled, img0, label=label, color=colors(int(cls), True), line_thickness=2)
            
            # Save image with detections
            output_path = "debug_output.jpg"
            cv2.imwrite(output_path, img0)
            print(f"Saved image with detections to {output_path}")
        else:
            print("No detections after NMS")
    except Exception as e:
        print(f"Error during NMS: {e}")
    
    print(f"\nInference time: {t2-t1:.4f}s")

if __name__ == "__main__":
    main() 