import torch
import yaml
import numpy as np
import cv2
import os
from models.model import DynamicCompactDetect
from utils.datasets import letterbox
from utils.general import non_max_suppression

def debug_model_output():
    # Set device and load model
    device = 'cpu'
    print(f'Using device: {device}')
    weights_path = 'runs/train_minimal/best_model.pt'
    ckpt = torch.load(weights_path, map_location=device)
    config_path = ckpt.get('config_path', 'configs/dynamiccompact_minimal.yaml')
    print(f'Using config from: {config_path}')
    model = DynamicCompactDetect(config_path)
    model.load_state_dict(ckpt['model_state_dict'])
    model.to(device)
    model.eval()
    
    # Process a single image
    img_path = 'datasets/coco/val2017/000000119445.jpg'
    print(f'Processing image: {img_path}')
    if not os.path.exists(img_path):
        print(f'Warning: Image not found at {img_path}')
        for root, dirs, files in os.walk('datasets'):
            if files and files[0].endswith(('.jpg', '.jpeg', '.png')):
                img_path = os.path.join(root, files[0])
                print(f'Using alternative image: {img_path}')
                break
    
    img0 = cv2.imread(img_path)
    if img0 is None:
        print(f'Error: Could not read image {img_path}')
        return
    
    img = letterbox(img0, new_shape=640)[0]
    img = img.transpose(2, 0, 1)
    img = np.ascontiguousarray(img)
    img = torch.from_numpy(img).float().to(device)
    img /= 255.0
    img = img.unsqueeze(0)
    
    # Run inference
    print("Running inference...")
    with torch.no_grad():
        pred = model(img)
    
    # Print output stats
    print(f'Output shape: {pred.shape}')
    print(f'Output min: {pred.min().item():.6f}, max: {pred.max().item():.6f}')
    print(f'Output mean: {pred.mean().item():.6f}, std: {pred.std().item():.6f}')
    
    # Check if outputs are normalized properly
    batch_size, channels, height, width = pred.shape
    pred_reshaped = pred.permute(0, 2, 3, 1).contiguous()
    pred_reshaped = pred_reshaped.view(batch_size, -1, channels)
    obj_scores = pred_reshaped[0, :, 4]
    print(f'Objectness scores - min: {obj_scores.min().item():.6f}, max: {obj_scores.max().item():.6f}')
    print(f'Objectness scores - mean: {obj_scores.mean().item():.6f}, std: {obj_scores.std().item():.6f}')
    
    # Print a histogram-like summary of objectness scores
    bins = [0.0, 0.001, 0.01, 0.1, 0.25, 0.5, 0.9]
    counts = [(obj_scores > bins[i]) & (obj_scores <= bins[i+1]) for i in range(len(bins)-1)]
    counts.append(obj_scores > bins[-1])
    for i in range(len(bins)):
        if i < len(bins) - 1:
            print(f'Scores {bins[i]:.3f} to {bins[i+1]:.3f}: {counts[i].sum().item()}')
        else:
            print(f'Scores > {bins[i]:.3f}: {counts[i].sum().item()}')
    
    # Check if class probabilities are properly normalized
    class_scores = pred_reshaped[0, :, 5:]
    print(f'Class scores shape: {class_scores.shape}')
    class_probs_sum = class_scores.sum(dim=1)
    print(f'Class probability sums - min: {class_probs_sum.min().item():.6f}, max: {class_probs_sum.max().item():.6f}')
    print(f'Class probability sums - mean: {class_probs_sum.mean().item():.6f}, std: {class_probs_sum.std().item():.6f}')
    
    # Top 5 detections with highest objectness
    top_indices = obj_scores.argsort(descending=True)[:5]
    for i, idx in enumerate(top_indices):
        box = pred_reshaped[0, idx, :4].cpu().numpy()
        obj_score = obj_scores[idx].item()
        class_probs = pred_reshaped[0, idx, 5:].cpu().numpy()
        max_class = class_probs.argmax()
        max_class_prob = class_probs[max_class]
        print(f'Top {i+1}: obj={obj_score:.6f}, class={max_class} (prob={max_class_prob:.6f}), box={box}')
    
    # Try to run NMS with a very low threshold to see if any detections pass
    print("\nRunning NMS with very low thresholds:")
    nms_out = non_max_suppression(
        pred_reshaped, 
        conf_thres=0.001,
        iou_thres=0.7
    )
    
    print(f"NMS results: {len(nms_out)} batches")
    for i, det in enumerate(nms_out):
        print(f"  Batch {i}: {len(det)} detections")
        if len(det):
            print("  First 3 detections:")
            for j in range(min(3, len(det))):
                d = det[j].cpu().numpy()
                print(f"    {j+1}: box=[{d[0]:.1f}, {d[1]:.1f}, {d[2]:.1f}, {d[3]:.1f}], conf={d[4]:.6f}, class={int(d[5])}")
    
    # If we have detections, draw them on the image and save
    if len(nms_out) > 0 and len(nms_out[0]) > 0:
        print("\nSaving detection visualization...")
        det = nms_out[0]
        os.makedirs('debug_output', exist_ok=True)
        
        # Draw detections
        for *xyxy, conf, cls in reversed(det):
            x1, y1, x2, y2 = int(xyxy[0]), int(xyxy[1]), int(xyxy[2]), int(xyxy[3])
            cv2.rectangle(img0, (x1, y1), (x2, y2), (0, 255, 0), 2)
            label = f'{int(cls)}: {conf:.2f}'
            cv2.putText(img0, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        
        cv2.imwrite('debug_output/detected.jpg', img0)
        print("Saved to debug_output/detected.jpg")
    else:
        print("No detections to visualize")

if __name__ == "__main__":
    debug_model_output() 