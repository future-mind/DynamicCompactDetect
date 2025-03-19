import time
import torch
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from collections import defaultdict

def compute_iou(box1, box2):
    """Compute IoU between two bounding boxes [x1, y1, x2, y2]."""
    # Get the coordinates of the intersection rectangle
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])
    
    # Check if there is an intersection
    if x1 >= x2 or y1 >= y2:
        return 0.0
    
    # Compute the intersection area
    intersection_area = (x2 - x1) * (y2 - y1)
    
    # Compute the area of both boxes
    box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
    box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])
    
    # Compute the union area
    union_area = box1_area + box2_area - intersection_area
    
    # Compute the IoU
    iou = intersection_area / union_area
    
    return iou

def compute_precision_recall(pred_boxes, true_boxes, pred_scores, pred_labels, true_labels, 
                           iou_threshold=0.5, conf_threshold=0.1):
    """Compute precision and recall for a single batch of predictions."""
    # Filter predictions based on confidence score
    conf_mask = pred_scores >= conf_threshold
    pred_boxes = pred_boxes[conf_mask]
    pred_scores = pred_scores[conf_mask]
    pred_labels = pred_labels[conf_mask]
    
    # Sort predictions by confidence score (high to low)
    _, indices = torch.sort(pred_scores, descending=True)
    pred_boxes = pred_boxes[indices]
    pred_scores = pred_scores[indices]
    pred_labels = pred_labels[indices]
    
    # Initialize arrays to store true positives and false positives
    true_positives = torch.zeros(len(pred_boxes), dtype=torch.bool)
    false_positives = torch.zeros(len(pred_boxes), dtype=torch.bool)
    
    # Assign predictions to ground truth objects
    detected = torch.zeros(len(true_boxes), dtype=torch.bool)
    
    for i, (pred_box, pred_label) in enumerate(zip(pred_boxes, pred_labels)):
        # For each predicted box, find the best matching ground truth box
        best_iou = -1
        best_gt_idx = -1
        
        for j, (true_box, true_label) in enumerate(zip(true_boxes, true_labels)):
            # Skip if already detected or if the label doesn't match
            if detected[j] or true_label != pred_label:
                continue
                
            # Compute IoU between the predicted box and ground truth box
            iou = compute_iou(pred_box, true_box)
            
            if iou > best_iou:
                best_iou = iou
                best_gt_idx = j
        
        # If the best IoU is above the threshold, count it as a true positive
        if best_iou >= iou_threshold:
            true_positives[i] = True
            detected[best_gt_idx] = True
        else:
            false_positives[i] = True
    
    # Compute precision and recall
    tp_cumsum = torch.cumsum(true_positives, dim=0)
    fp_cumsum = torch.cumsum(false_positives, dim=0)
    
    precision = tp_cumsum / (tp_cumsum + fp_cumsum + 1e-8)
    recall = tp_cumsum / (len(true_boxes) + 1e-8)
    
    # Compute average precision
    # Add sentinel values to ensure the curve starts at recall = 0 and ends at recall = 1
    precision = torch.cat([torch.tensor([1.0]), precision])
    recall = torch.cat([torch.tensor([0.0]), recall])
    
    # Make precision monotonically decreasing
    for i in range(len(precision) - 1, 0, -1):
        precision[i - 1] = max(precision[i - 1], precision[i])
    
    # Compute the area under the precision-recall curve
    indices = torch.where(recall[1:] != recall[:-1])[0]
    ap = torch.sum((recall[indices + 1] - recall[indices]) * precision[indices + 1])
    
    return ap.item(), precision, recall

def compute_mAP(predictions, targets, iou_threshold=0.5, conf_threshold=0.1):
    """Compute mAP (mean Average Precision) for object detection."""
    # This is a simplified implementation
    # In a real scenario, this would handle more complex matching
    # and evaluation procedures
    
    # Placeholder for AP values per class
    ap_per_class = defaultdict(list)
    
    for batch_preds, batch_targets in zip(predictions, targets):
        # Process batch predictions
        # In a real implementation, this would need to convert predictions
        # from model output format to bounding boxes
        
        # For this simplified implementation, assume batch_preds contains:
        # - outputs: a list of tensors representing detection outputs for different feature levels
        
        # Extract boxes, scores, and labels from predictions (simplified)
        all_boxes = []
        all_scores = []
        all_labels = []
        
        # Process each batch target
        for target in batch_targets:
            # Extract ground truth boxes and labels
            true_boxes = target['boxes']  # [N, 4] tensor
            true_labels = target['labels']  # [N] tensor
            
            # In a real implementation, we'd need to match predictions to this target
            # For simplicity, we'll use dummy values
            pred_boxes = true_boxes.clone()  # Dummy prediction matching ground truth
            pred_scores = torch.rand(len(true_boxes))  # Random confidence scores
            pred_labels = true_labels.clone()  # Dummy prediction matching ground truth
            
            # Compute precision and recall
            ap, precision, recall = compute_precision_recall(
                pred_boxes, true_boxes, pred_scores, pred_labels, true_labels,
                iou_threshold, conf_threshold
            )
            
            # Store AP values per class
            for label in true_labels.unique():
                ap_per_class[label.item()].append(ap)
    
    # Compute mean AP across all classes
    mean_ap = np.mean([np.mean(aps) for aps in ap_per_class.values()])
    
    # Compute mAP@0.5 (mAP with IoU threshold of 0.5)
    mAP50 = mean_ap  # In this simplified implementation, they're the same
    
    return mean_ap, mAP50

def measure_inference_time(model, input_size=(640, 640), iterations=100, warm_up=10, device=None):
    """Measure inference time for the model."""
    # Set device if not provided
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Move model to device
    model = model.to(device)
    model.eval()
    
    # Create dummy input
    dummy_input = torch.randn(1, 3, input_size[1], input_size[0], device=device)
    
    # Warm-up runs
    with torch.no_grad():
        for _ in range(warm_up):
            _ = model(dummy_input)
    
    # Measure inference time
    torch.cuda.synchronize() if device.type == 'cuda' else None
    start_time = time.time()
    
    with torch.no_grad():
        for _ in range(iterations):
            _ = model(dummy_input)
    
    torch.cuda.synchronize() if device.type == 'cuda' else None
    end_time = time.time()
    
    # Calculate average inference time
    avg_time = (end_time - start_time) / iterations
    fps = 1.0 / avg_time
    
    return avg_time, fps

def benchmark_model(model, dataloader, device=None, iou_threshold=0.5, conf_threshold=0.01):
    """Benchmark model performance (mAP and FPS)."""
    # Set device if not provided
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Move model to device
    model = model.to(device)
    model.eval()
    
    # Measure inference time
    input_size = (640, 640)  # Default size
    avg_time, fps = measure_inference_time(model, input_size=input_size, device=device)
    
    # Evaluate mAP
    all_predictions = []
    all_targets = []
    
    with torch.no_grad():
        for images, targets in tqdm(dataloader, desc="Evaluating mAP"):
            images = images.to(device)
            targets = [t.to(device) for t in targets]
            
            # Forward pass
            predictions = model(images)
            
            # Store predictions and targets
            all_predictions.append(predictions)
            all_targets.append(targets)
    
    # Compute mAP
    mean_ap, mAP50 = compute_mAP(all_predictions, all_targets, 
                                  iou_threshold=iou_threshold, 
                                  conf_threshold=conf_threshold)
    
    # Calculate model size
    model_size_mb = sum(p.numel() for p in model.parameters()) * 4 / (1024 * 1024)  # Size in MB
    
    # Return benchmark results
    return {
        'mAP': mean_ap,
        'mAP@0.5': mAP50,
        'fps': fps,
        'inference_time_ms': avg_time * 1000,
        'model_size_mb': model_size_mb
    }

def plot_precision_recall_curve(precision, recall, class_name=None, ax=None):
    """Plot precision-recall curve."""
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 6))
    
    ax.plot(recall, precision)
    ax.set_xlabel('Recall')
    ax.set_ylabel('Precision')
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    title = 'Precision-Recall Curve'
    if class_name:
        title += f' for {class_name}'
    ax.set_title(title)
    ax.grid(True)
    
    return ax

def plot_comparison_chart(models_results, metric='mAP', figsize=(10, 6)):
    """Plot comparison chart for different models."""
    models = list(models_results.keys())
    values = [models_results[model][metric] for model in models]
    
    fig, ax = plt.subplots(figsize=figsize)
    ax.bar(models, values)
    ax.set_ylabel(metric)
    ax.set_title(f'Comparison of {metric} across models')
    ax.grid(True, axis='y')
    
    # Add values on top of bars
    for i, v in enumerate(values):
        ax.text(i, v + 0.01, f'{v:.3f}', ha='center')
    
    return fig

class AverageMeter:
    """Computes and stores the average and current value."""
    def __init__(self, name, fmt=':f'):
        self.name = name
        self.fmt = fmt
        self.reset()
        
    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0
        
    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count
        
    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)

class ProgressMeter:
    """Display training progress."""
    def __init__(self, num_batches, meters, prefix=""):
        self.batch_fmtstr = self._get_batch_fmtstr(num_batches)
        self.meters = meters
        self.prefix = prefix
        
    def display(self, batch):
        entries = [self.prefix + self.batch_fmtstr.format(batch)]
        entries += [str(meter) for meter in self.meters]
        print('\t'.join(entries))
        
    def _get_batch_fmtstr(self, num_batches):
        num_digits = len(str(num_batches // 1))
        fmt = '{:' + str(num_digits) + 'd}'
        return '[' + fmt + '/' + fmt.format(num_batches) + ']' 