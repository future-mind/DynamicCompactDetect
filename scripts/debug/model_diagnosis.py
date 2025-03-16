#!/usr/bin/env python3

import argparse
import cv2
import numpy as np
import os
import torch
import matplotlib.pyplot as plt
from pathlib import Path

# Add utilities from model codebase
import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from utils.general import non_max_suppression, scale_coords

# Import the model class
from models import DynamicCompactDetect

def parse_args():
    parser = argparse.ArgumentParser(description='Analyze model outputs for debugging')
    parser.add_argument('--weights', type=str, default='runs/train_minimal/best_model.pt', help='model weights path')
    parser.add_argument('--cfg', type=str, default='configs/dynamiccompact_minimal.yaml', help='model config path')
    parser.add_argument('--img-path', type=str, default='inference/images/000000000139.jpg', help='input image path')
    parser.add_argument('--img-size', type=int, default=640, help='inference size')
    parser.add_argument('--device', default='', help='device to use (cuda or cpu)')
    return parser.parse_args()

def letterbox(img, new_shape=(640, 640), color=(114, 114, 114), auto=True, scaleFill=False, scaleup=True, stride=32):
    # Resize and pad image while meeting stride-multiple constraints
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

def load_model(weights_path, cfg_path, device_str):
    """Load the DynamicCompact model using the same approach as visual_model_comparison.py."""
    print(f"Loading DynamicCompact-Detect model from {weights_path}...")
    device = torch.device(device_str if device_str else ('cuda:0' if torch.cuda.is_available() else 'cpu'))
    
    try:
        try:
            ckpt = torch.load(weights_path, map_location=device, weights_only=False)
        except TypeError:
            ckpt = torch.load(weights_path, map_location=device)
        
        if isinstance(ckpt, dict) and 'model_state_dict' in ckpt:
            model = DynamicCompactDetect(cfg=cfg_path).to(device)
            model.load_state_dict(ckpt['model_state_dict'])
        elif isinstance(ckpt, dict) and 'model' in ckpt:
            model = DynamicCompactDetect(cfg=ckpt.get('cfg', cfg_path)).to(device)
            model.load_state_dict(ckpt['model'])
        else:
            model = DynamicCompactDetect(cfg=cfg_path).to(device)
            model.load_state_dict(ckpt)
    except Exception as e:
        print(f"Error loading checkpoint: {e}")
        print("Creating model directly from config...")
        model = DynamicCompactDetect(cfg=cfg_path).to(device)
    
    model.eval()
    print(f"Model loaded with {sum(p.numel() for p in model.parameters())} parameters")
    return model, device

def analyze_model_output(model, img_path, img_size, device):
    """Analyze the model's output format and detection capabilities."""
    # Load and preprocess image
    print(f"Analyzing image: {img_path}")
    img0 = cv2.imread(img_path)
    if img0 is None:
        print(f"Error: Unable to read image {img_path}")
        return
    
    # Preprocess image
    img = letterbox(img0, new_shape=img_size)[0]
    img = img.transpose(2, 0, 1)  # HWC to CHW
    img = np.ascontiguousarray(img)
    img = torch.from_numpy(img).float().to(device)
    img /= 255.0  # 0 - 255 to 0.0 - 1.0
    
    if img.ndimension() == 3:
        img = img.unsqueeze(0)  # add batch dimension
    
    # Run inference
    with torch.no_grad():
        pred = model(img)
    
    # Analyze raw output
    print(f"Model output shape: {pred.shape}")
    print(f"Output tensor type: {pred.dtype}")
    
    # The model appears to output feature maps rather than detection predictions
    # We need to convert this to the expected format for non_max_suppression
    
    # Check if the output is a feature map (common in YOLO models)
    if len(pred.shape) == 4:  # [batch, channels, height, width]
        print("Model outputs feature maps, need to convert to detection format")
        
        # Try to reshape to expected format for NMS
        try:
            # For YOLO-style outputs, we need to reshape and process
            # This is a simplified approach - actual processing depends on anchor configuration
            batch_size, channels, height, width = pred.shape
            
            # Assuming standard YOLO format with 3 anchors and 85 outputs per anchor (80 classes + 5 box params)
            num_anchors = 3
            outputs_per_anchor = channels // num_anchors
            
            print(f"Estimated anchors: {num_anchors}, outputs per anchor: {outputs_per_anchor}")
            
            # Reshape to [batch, anchors*height*width, outputs_per_anchor]
            pred_reshaped = pred.permute(0, 2, 3, 1).contiguous()
            pred_reshaped = pred_reshaped.view(batch_size, height*width*num_anchors, outputs_per_anchor)
            
            print(f"Reshaped for NMS: {pred_reshaped.shape}")
            
            # Apply NMS with different thresholds
            print("\nApplying NMS with different thresholds:")
            for conf_thresh in [0.25, 0.5, 0.7, 0.9]:
                for iou_thresh in [0.45]:
                    try:
                        nms_output = non_max_suppression(pred_reshaped, conf_thresh, iou_thresh, max_det=300)
                        num_detections = len(nms_output[0])
                        print(f"  Conf threshold={conf_thresh}, IoU threshold={iou_thresh}: {num_detections} detections")
                        
                        # Show top 5 detections
                        if num_detections > 0:
                            print(f"  Top 5 detections (conf threshold={conf_thresh}):")
                            for i in range(min(5, num_detections)):
                                bbox = nms_output[0][i]
                                print(f"    Detection {i+1}: "
                                     f"box=[{bbox[0]:.1f}, {bbox[1]:.1f}, {bbox[2]:.1f}, {bbox[3]:.1f}], "
                                     f"conf={bbox[4]:.4f}, class={int(bbox[5])}")
                    except Exception as e:
                        print(f"  Error applying NMS: {e}")
        except Exception as e:
            print(f"Error reshaping output: {e}")
            
        # Try to visualize the feature maps
        try:
            # Create a directory for feature maps
            os.makedirs("feature_maps", exist_ok=True)
            
            # Save a few feature maps as images
            print("\nSaving sample feature maps...")
            for i in range(min(5, channels)):
                feature_map = pred[0, i].cpu().numpy()
                plt.figure(figsize=(8, 6))
                plt.imshow(feature_map, cmap='viridis')
                plt.colorbar()
                plt.title(f"Feature Map {i}")
                plt.savefig(f"feature_maps/feature_map_{i}.png")
                plt.close()
            print(f"Feature maps saved to feature_maps/ directory")
        except Exception as e:
            print(f"Error visualizing feature maps: {e}")
    else:
        print(f"Unexpected model output shape: {pred.shape}")
        
    # Try to run the model through the original inference pipeline
    print("\nAttempting to run through original inference pipeline...")
    try:
        # Import the original NMS and scaling functions 
        from utils.general import non_max_suppression, scale_coords
        
        # Run inference with the model
        with torch.no_grad():
            pred = model(img)
        
        # Apply NMS using the original function
        pred_processed = non_max_suppression(pred, 0.25, 0.45, max_det=30)
        
        # Process detections
        detections = []
        
        if len(pred_processed[0]) > 0:
            # Rescale boxes from img_size to im0 size using the original function
            pred_processed[0][:, :4] = scale_coords(img.shape[2:], pred_processed[0][:, :4], img0.shape).round()
            
            # Convert to list of detections
            for *xyxy, conf, cls in pred_processed[0]:
                x1, y1, x2, y2 = [int(x.cpu().item()) for x in xyxy]
                cls_id = int(cls.cpu().item())
                conf_val = float(conf.cpu().item())
                
                detections.append({
                    "bbox": [x1, y1, x2, y2],
                    "confidence": conf_val,
                    "class_id": cls_id
                })
        
        print(f"Original pipeline detected {len(detections)} objects")
        
        # Draw detections on image
        output_img = img0.copy()
        for det in detections:
            x1, y1, x2, y2 = det["bbox"]
            conf = det["confidence"]
            cls_id = det["class_id"]
            
            # Draw rectangle
            cv2.rectangle(output_img, (x1, y1), (x2, y2), (0, 255, 0), 2)
            
            # Draw label
            label = f"Class {cls_id}: {conf:.2f}"
            cv2.putText(output_img, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        
        # Save output image
        cv2.imwrite("detection_output.jpg", output_img)
        print("Detection visualization saved to detection_output.jpg")
        
    except Exception as e:
        print(f"Error running through original pipeline: {e}")
        
    print("\nAnalysis complete.")

def main():
    args = parse_args()
    
    # Load model
    model, device = load_model(args.weights, args.cfg, args.device)
    if model is None:
        return
    
    # Analyze model output
    analyze_model_output(model, args.img_path, args.img_size, device)

if __name__ == "__main__":
    main() 