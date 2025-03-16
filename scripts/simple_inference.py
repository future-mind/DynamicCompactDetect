#!/usr/bin/env python
import argparse
import os
import time
import random
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
    parser.add_argument('--img-size', type=int, default=640, help='inference size (pixels)')
    parser.add_argument('--conf-thres', type=float, default=0.25, help='object confidence threshold')
    parser.add_argument('--iou-thres', type=float, default=0.45, help='IOU threshold for NMS')
    parser.add_argument('--source', type=str, default='datasets/coco/val2017', help='source')
    parser.add_argument('--output', type=str, default='output', help='output folder')
    parser.add_argument('--verbose', action='store_true', help='print detailed debug information')
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
    
    # Load configuration
    with open(config_path, 'r') as f:
        model_cfg = yaml.safe_load(f)
    
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
    
    # Set Dataloader
    if os.path.isdir(args.source):
        img_paths = [os.path.join(args.source, f) for f in os.listdir(args.source) if f.endswith(('.png', '.jpg', '.jpeg'))]
        # Sort to ensure consistent order
        img_paths.sort()
        # Take just a few images for debugging
        img_paths = img_paths[:1]
    elif os.path.isfile(args.source):
        img_paths = [args.source]
    else:
        raise Exception(f"Invalid source: {args.source}")
    
    # Always enable verbose for the first image
    verbose_first_image = True
    
    # Process images
    for img_path in img_paths:
        if verbose_first_image:
            print(f"\n\n===== Processing {img_path} =====")
        # Load image
        img0 = cv2.imread(img_path)  # BGR
        if img0 is None:
            print(f"Error: Could not read image {img_path}")
            continue
            
        if verbose_first_image:
            print(f"Image shape: {img0.shape}")
        
        # Preprocess image
        img = letterbox(img0, new_shape=args.img_size)[0]
        img = img.transpose(2, 0, 1)  # HWC to CHW
        img = np.ascontiguousarray(img)
        img = torch.from_numpy(img).float().to(device)
        img /= 255.0  # 0 - 255 to 0.0 - 1.0
        
        if img.ndimension() == 3:
            img = img.unsqueeze(0)  # add batch dimension
        
        if verbose_first_image:
            print(f"Preprocessed image shape: {img.shape}")
        
        # Inference
        t1 = time.time()
        with torch.no_grad():
            pred = model(img)
        t2 = time.time()
        
        # Debug prints
        if verbose_first_image:
            print(f"\nRAW MODEL OUTPUT:")
            print(f"Model output shape: {pred.shape}")
            print(f"Model output min: {pred.min().item():.6f}, max: {pred.max().item():.6f}")
            print(f"Model output mean: {pred.mean().item():.6f}, std: {pred.std().item():.6f}")
        
        # Check objectness scores
        if pred.dim() == 4:
            batch_size, channels, height, width = pred.shape
            if verbose_first_image:
                print(f"\nRESHAPING OUTPUT:")
                print(f"Channels: {channels}, expected format: [x, y, w, h, obj, class1, class2, ...]")
            pred_reshaped = pred.permute(0, 2, 3, 1).contiguous()
            pred_reshaped = pred_reshaped.view(batch_size, -1, channels)
            if verbose_first_image:
                print(f"Reshaped to: {pred_reshaped.shape}")
            
            # Analyze objectness scores
            obj_scores = pred_reshaped[0, :, 4]
            if verbose_first_image:
                print(f"\nOBJECTNESS SCORES ANALYSIS:")
                print(f"Objectness scores - min: {obj_scores.min().item():.6f}, max: {obj_scores.max().item():.6f}")
                print(f"Objectness scores - mean: {obj_scores.mean().item():.6f}, std: {obj_scores.std().item():.6f}")
                print(f"Boxes with objectness > {args.conf_thres}: {(obj_scores > args.conf_thres).sum().item()}")
            
            # Histogram of objectness scores
            if verbose_first_image:
                bins = [0.0, 0.001, 0.01, 0.1, 0.25, 0.5, 0.9]
                counts = [(obj_scores > bins[i]) & (obj_scores <= bins[i+1]) for i in range(len(bins)-1)]
                counts.append(obj_scores > bins[-1])
                print("\nObjectness score distribution:")
                for i in range(len(bins)):
                    if i < len(bins) - 1:
                        print(f"  Scores {bins[i]:.3f} to {bins[i+1]:.3f}: {counts[i].sum().item()}")
                    else:
                        print(f"  Scores > {bins[i]:.3f}: {counts[i].sum().item()}")
            
            # Analyze class probabilities
            class_scores = pred_reshaped[0, :, 5:]
            if verbose_first_image:
                print(f"\nCLASS PROBABILITIES ANALYSIS:")
                print(f"Class scores shape: {class_scores.shape}")
                class_probs_sum = class_scores.sum(dim=1)
                print(f"Class probability sums - min: {class_probs_sum.min().item():.6f}, max: {class_probs_sum.max().item():.6f}")
                print(f"Class probability sums - mean: {class_probs_sum.mean().item():.6f}, std: {class_probs_sum.std().item():.6f}")
            
            # Top 5 boxes
            if verbose_first_image:
                print("\nTOP 5 BOXES WITH HIGHEST OBJECTNESS:")
                top_indices = obj_scores.argsort(descending=True)[:5]
                for i, idx in enumerate(top_indices):
                    box = pred_reshaped[0, idx, :4].cpu().numpy()
                    obj_score = obj_scores[idx].item()
                    class_scores_np = pred_reshaped[0, idx, 5:].cpu().numpy()
                    max_class = class_scores_np.argmax()
                    max_class_score = class_scores_np[max_class]
                    print(f"Box {i+1}: obj={obj_score:.6f}, class={max_class} ({class_names[max_class]}, conf={max_class_score:.6f}), box={box}")
                    # Print top 3 classes for this box
                    top_classes = np.argsort(class_scores_np)[-3:][::-1]
                    print(f"  Top classes: " + ", ".join([f"{class_names[c]} ({class_scores_np[c]:.6f})" for c in top_classes]))
        
        # Apply NMS
        if verbose_first_image:
            print("\nAPPLYING NMS:")
            print(f"Confidence threshold: {args.conf_thres}, IoU threshold: {args.iou_thres}")
        t_nms_start = time.time()
        pred = non_max_suppression(pred, args.conf_thres, args.iou_thres)
        t_nms_end = time.time()
        
        # Debug prints after NMS
        if verbose_first_image:
            print(f"After NMS: {len(pred)} batch items, first item has {len(pred[0]) if len(pred) > 0 else 0} detections")
            if len(pred) > 0 and len(pred[0]) > 0:
                print("First 3 detections:")
                for i in range(min(3, len(pred[0]))):
                    d = pred[0][i].cpu().numpy()
                    print(f"  {i+1}: box=[{d[0]:.1f}, {d[1]:.1f}, {d[2]:.1f}, {d[3]:.1f}], conf={d[4]:.6f}, class={int(d[5])} ({class_names[int(d[5])]})")
            else:
                print("No detections after NMS")
                
                # Check raw predictions more deeply
                print("\nINVESTIGATING RAW PREDICTIONS IN MORE DETAIL:")
                # Check max values in each channel
                for i in range(min(10, channels)):
                    channel_max = pred_reshaped[0, :, i].max().item()
                    channel_min = pred_reshaped[0, :, i].min().item()
                    channel_mean = pred_reshaped[0, :, i].mean().item()
                    print(f"Channel {i} - min: {channel_min:.6f}, max: {channel_max:.6f}, mean: {channel_mean:.6f}")
                
                # Print sum of class probabilities for top anchors
                for i, idx in enumerate(top_indices[:3]):
                    class_sum = pred_reshaped[0, idx, 5:].sum().item()
                    print(f"Anchor {i+1} (top obj score) - class prob sum: {class_sum:.6f}")
                    
                # Try with a much lower confidence threshold
                very_low_conf = 0.0001
                print(f"\nTrying NMS with very low confidence threshold ({very_low_conf}):")
                pred_low_conf = non_max_suppression(pred_reshaped, very_low_conf, args.iou_thres)
                print(f"Results: {len(pred_low_conf)} batch items, first item has {len(pred_low_conf[0]) if len(pred_low_conf) > 0 else 0} detections")
                if len(pred_low_conf) > 0 and len(pred_low_conf[0]) > 0:
                    print("First 3 detections with low threshold:")
                    for i in range(min(3, len(pred_low_conf[0]))):
                        d = pred_low_conf[0][i].cpu().numpy()
                        print(f"  {i+1}: box=[{d[0]:.1f}, {d[1]:.1f}, {d[2]:.1f}, {d[3]:.1f}], conf={d[4]:.6f}, class={int(d[5])} ({class_names[int(d[5])]})")
        
        # Process detections
        detections_str = []
        for i, det in enumerate(pred):  # detections per image
            im0 = img0.copy()
            
            if len(det):
                # Rescale boxes from img_size to im0 size
                det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0.shape).round()
                
                # Print results
                if verbose_first_image:
                    print(f"\nDETECTION RESULTS:")
                    print(f"Detected {len(det)} objects:")
                
                # Draw bounding boxes and labels
                for *xyxy, conf, cls in det:
                    label = f"{class_names[int(cls)]} {conf:.2f}"
                    plot_one_box(xyxy, im0, label=label, color=colors(int(cls), True), line_thickness=2)
                    if verbose_first_image:
                        print(f"  {label} at {xyxy}")
                    detections_str.append(f"{class_names[int(cls)]}")
            
            # Save results
            output_path = os.path.join(args.output, os.path.basename(img_path))
            cv2.imwrite(output_path, im0)
            if verbose_first_image:
                print(f"Saved {output_path}")
        
        # Print summary
        if verbose_first_image:
            print(f"\nSUMMARY:")
        print(f"{img_path}: {', '.join([f'{detections_str.count(c)} {c}s' for c in set(detections_str) if detections_str.count(c) > 0])}, Done. ({t2-t1:.3f}s inference, {t_nms_end-t_nms_start:.3f}s NMS)")
        
        # Only process one image for debugging
        if verbose_first_image:
            break
        
        # Turn off verbose for subsequent images
        verbose_first_image = False
    
    print("Done!")

if __name__ == "__main__":
    main() 