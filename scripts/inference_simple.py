#!/usr/bin/env python
import argparse
import os
import time
import random
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn as nn
import yaml

# Import necessary utilities
from utils.datasets import letterbox
from utils.general import non_max_suppression, scale_coords


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', type=str, default='runs/train_minimal/best_model.pt', help='model.pt path')
    parser.add_argument('--img-size', type=int, default=640, help='inference size (pixels)')
    parser.add_argument('--conf-thres', type=float, default=0.25, help='object confidence threshold')
    parser.add_argument('--iou-thres', type=float, default=0.45, help='IOU threshold for NMS')
    parser.add_argument('--source', type=str, default='datasets/coco/val2017', help='source')
    parser.add_argument('--output', type=str, default='inference/output', help='output folder')
    return parser.parse_args()


def colors(i, bgr=False):
    """Generate colors for drawing bounding boxes."""
    hex = ('FF3838', 'FF9D97', 'FF701F', 'FFB21D', 'CFD231', '48F90A', '92CC17', '3DDB86', '1A9334', '00D4BB',
           '2C99A8', '00C2FF', '344593', '6473FF', '0018EC', '8438FF', '520085', 'CB38FF', 'FF95C8', 'FF37C7')
    palette = []
    for j in range(len(hex)):
        h = '#' + hex[j]
        rgb = tuple(int(h[1 + i:1 + i + 2], 16) for i in (0, 2, 4))
        palette.append(rgb)
    
    num = len(palette)
    color = palette[int(i) % num]
    
    return (color[2], color[1], color[0]) if bgr else color  # BGR if bgr else RGB


def plot_one_box(x, img, color=None, label=None, line_thickness=None):
    """Plots one bounding box on the image."""
    tl = line_thickness or round(0.002 * (img.shape[0] + img.shape[1]) / 2) + 1  # line thickness
    color = color or [random.randint(0, 255) for _ in range(3)]
    c1, c2 = (int(x[0]), int(x[1])), (int(x[2]), int(x[3]))
    cv2.rectangle(img, c1, c2, color, thickness=tl, lineType=cv2.LINE_AA)
    
    if label:
        tf = max(tl - 1, 1)  # font thickness
        t_size = cv2.getTextSize(label, 0, fontScale=tl / 3, thickness=tf)[0]
        c2 = c1[0] + t_size[0], c1[1] - t_size[1] - 3
        cv2.rectangle(img, c1, c2, color, -1, cv2.LINE_AA)  # filled
        cv2.putText(img, label, (c1[0], c1[1] - 2), 0, tl / 3, [225, 255, 255], thickness=tf, lineType=cv2.LINE_AA)
    
    return img


def main():
    # Parse arguments
    args = parse_args()
    
    # Set device
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    
    # Create output directory
    os.makedirs(args.output, exist_ok=True)
    
    # Load model
    print(f"Loading model from {args.weights}")
    ckpt = torch.load(args.weights, map_location=device)
    
    # Create model from config
    config_path = ckpt.get('config_path', 'configs/dynamiccompact_minimal.yaml')
    print(f"Using config from: {config_path}")
    
    # Load model architecture
    from models.common import Conv
    
    # Load configuration
    with open(config_path, 'r') as f:
        model_cfg = yaml.safe_load(f)
    
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
    
    # Class names
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
    
    # Get input source
    source = str(args.source)
    if os.path.isdir(source):
        files = sorted([os.path.join(source, f) for f in os.listdir(source) if f.endswith(('.png', '.jpg', '.jpeg'))])
    elif os.path.isfile(source):
        files = [source]
    else:
        raise ValueError(f"Source {source} is not a valid file or directory")
    
    # Process each image
    for img_path in files:
        # Read image
        img0 = cv2.imread(img_path)
        if img0 is None:
            print(f"Failed to read image {img_path}")
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
        t1 = time.time()
        with torch.no_grad():
            pred = model(img)
        t2 = time.time()
        
        # Apply NMS
        pred = non_max_suppression(pred, args.conf_thres, args.iou_thres)
        t3 = time.time()
        
        # Process detections
        s = ""
        if len(pred) > 0 and len(pred[0]) > 0:
            # Rescale boxes from img_size to im0 size
            pred[0][:, :4] = scale_coords(img.shape[2:], pred[0][:, :4], img0.shape).round()
            
            # Print results
            for c in pred[0][:, -1].unique():
                n = (pred[0][:, -1] == c).sum()  # detections per class
                c_int = int(c.item())
                class_name = class_names[c_int] if c_int < len(class_names) else f"class_{c_int}"
                s += f"{n} {class_name}{'s' * (n > 1)}, "  # add to string
            
            # Draw boxes on image
            for *xyxy, conf, cls in pred[0]:
                label = f'{class_names[int(cls)]} {conf:.2f}'
                plot_one_box(xyxy, img0, label=label, color=colors(int(cls), True), line_thickness=3)
            
            # Save output image
            out_path = str(Path(args.output) / Path(img_path).name)
            cv2.imwrite(out_path, img0)
        
        # Print timing
        print(f'{img_path}: {s}Done. ({(t2 - t1):.3f}s inference, {(t3 - t2):.3f}s NMS)')
    
    print('Done!')


if __name__ == '__main__':
    main() 