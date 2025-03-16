#!/usr/bin/env python
import torch
import yaml
import numpy as np
import cv2
import os

def main():
    # Set device
    device = 'cpu'
    print(f"Using device: {device}")
    
    # Load model
    print("Loading model...")
    weights_path = 'runs/train_minimal/best_model.pt'
    ckpt = torch.load(weights_path, map_location=device)
    config_path = ckpt.get('config_path', 'configs/dynamiccompact_minimal.yaml')
    print(f"Using config from: {config_path}")
    
    # Import necessary module
    from models.model import DynamicCompactDetect
    
    # Create model
    model = DynamicCompactDetect(config_path)
    model.load_state_dict(ckpt['model_state_dict'])
    model.to(device)
    model.eval()
    
    # Print model summary
    n_parameters = sum(p.numel() for p in model.parameters())
    print(f"Model loaded with {n_parameters} parameters")
    
    # Load and preprocess image
    from utils.datasets import letterbox
    img_path = 'datasets/coco/val2017/000000119445.jpg'
    img0 = cv2.imread(img_path)
    img = letterbox(img0, new_shape=640)[0]
    img = img.transpose(2, 0, 1)  # HWC to CHW
    img = np.ascontiguousarray(img)
    img = torch.from_numpy(img).float().to(device)
    img /= 255.0  # 0 - 255 to 0.0 - 1.0
    img = img.unsqueeze(0)  # add batch dimension
    
    # Inference
    print("Running inference...")
    with torch.no_grad():
        pred = model(img)
    
    # Debug prints
    print(f"Output shape: {pred.shape}")
    print(f"Output min: {pred.min().item()}, max: {pred.max().item()}")
    print(f"Sample values: {pred[0, :5, 0, 0].cpu().numpy()}")
    
    # Check objectness scores
    if pred.dim() == 4:
        batch_size, channels, height, width = pred.shape
        print(f"Channels: {channels}, expected format: [x, y, w, h, obj, class1, class2, ...]")
        pred_reshaped = pred.permute(0, 2, 3, 1).contiguous()
        pred_reshaped = pred_reshaped.view(batch_size, -1, channels)
        obj_scores = pred_reshaped[0, :, 4]
        print(f"Objectness scores - min: {obj_scores.min().item()}, max: {obj_scores.max().item()}")
        print(f"Boxes with objectness > 0.01: {(obj_scores > 0.01).sum().item()}")
        
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
    print("Applying NMS...")
    from utils.general import non_max_suppression
    nms_pred = non_max_suppression(pred, 0.01, 0.45)
    print(f"After NMS: {len(nms_pred)} batch items, first item has {len(nms_pred[0]) if len(nms_pred) > 0 else 0} detections")
    
    if len(nms_pred) > 0 and len(nms_pred[0]) > 0:
        print(f"First detection: {nms_pred[0][0]}")
    else:
        print("No detections after NMS")
        
        # Check raw predictions more deeply
        print("\nInvestigating raw predictions in more detail:")
        # Check max values in each channel
        for i in range(channels):
            channel_max = pred[0, i].max().item()
            channel_min = pred[0, i].min().item()
            print(f"Channel {i} - min: {channel_min:.6f}, max: {channel_max:.6f}")
        
        # Print non-zero anchors
        anchor_count = pred_reshaped.size(1)
        print(f"\nTotal anchors: {anchor_count}")
        
        # Print sum of class probabilities for top anchors
        for i, idx in enumerate(top_indices[:3]):
            class_sum = pred_reshaped[0, idx, 5:].sum().item()
            print(f"Anchor {i+1} (top obj score) - class prob sum: {class_sum:.6f}")

if __name__ == "__main__":
    main() 