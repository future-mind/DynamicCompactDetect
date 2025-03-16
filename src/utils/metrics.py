import torch
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
from collections import defaultdict


def box_iou(box1, box2):
    """
    Calculate IoU between box1 and box2.
    
    Args:
        box1, box2: Boxes in [x1, y1, x2, y2] format
        
    Returns:
        Tensor: IoU values
    """
    # Calculate intersection area
    x1 = torch.max(box1[:, None, 0], box2[:, 0])
    y1 = torch.max(box1[:, None, 1], box2[:, 1])
    x2 = torch.min(box1[:, None, 2], box2[:, 2])
    y2 = torch.min(box1[:, None, 3], box2[:, 3])
    
    intersection = torch.clamp(x2 - x1, min=0) * torch.clamp(y2 - y1, min=0)
    
    # Calculate union area
    box1_area = (box1[:, 2] - box1[:, 0]) * (box1[:, 3] - box1[:, 1])
    box2_area = (box2[:, 2] - box2[:, 0]) * (box2[:, 3] - box2[:, 1])
    union = box1_area[:, None] + box2_area - intersection
    
    # Calculate IoU
    iou = intersection / (union + 1e-7)
    
    return iou


def compute_ap(recall, precision):
    """
    Compute the average precision using the 11-point interpolation method.
    
    Args:
        recall: Recall values
        precision: Precision values
        
    Returns:
        float: Average precision
    """
    # 11-point interpolation
    ap = 0.0
    for t in np.arange(0.0, 1.1, 0.1):
        if np.sum(recall >= t) == 0:
            p = 0
        else:
            p = np.max(precision[recall >= t])
        ap += p / 11.0
    
    return ap


def compute_map(results, iou_thresholds=None, class_names=None):
    """
    Compute mAP for object detection results.
    
    Args:
        results: List of dictionaries containing detection results
        iou_thresholds: List of IoU thresholds for evaluation
        class_names: List of class names
        
    Returns:
        list: [mAP@0.5, mAP@0.5:0.95]
    """
    if iou_thresholds is None:
        iou_thresholds = torch.linspace(0.5, 0.95, 10)
    
    # Initialize statistics
    stats = []
    
    # Process each IoU threshold
    for iou_threshold in iou_thresholds:
        # Initialize per-class statistics
        class_stats = defaultdict(lambda: {'tp': 0, 'fp': 0, 'gt': 0})
        
        # Process each result
        for result in results:
            pred_boxes = result['boxes']
            pred_scores = result['scores']
            pred_labels = result['labels']
            gt_boxes = result['targets'][:, :4]
            gt_labels = result['targets'][:, 4].long()
            
            # Skip if no predictions or ground truth
            if len(pred_boxes) == 0 or len(gt_boxes) == 0:
                continue
            
            # Calculate IoU between predictions and ground truth
            ious = box_iou(pred_boxes, gt_boxes)
            
            # Process each prediction
            for i, (box, score, label) in enumerate(zip(pred_boxes, pred_scores, pred_labels)):
                # Get IoUs for this prediction with all ground truth boxes
                box_ious = ious[i]
                
                # Get ground truth boxes with the same class
                same_class_mask = gt_labels == label
                
                if not same_class_mask.any():
                    # No ground truth of this class, false positive
                    class_stats[label.item()]['fp'] += 1
                    continue
                
                # Get IoUs with ground truth boxes of the same class
                class_ious = box_ious[same_class_mask]
                
                if class_ious.max() >= iou_threshold:
                    # True positive
                    class_stats[label.item()]['tp'] += 1
                else:
                    # False positive
                    class_stats[label.item()]['fp'] += 1
            
            # Count ground truth boxes for each class
            for label in gt_labels:
                class_stats[label.item()]['gt'] += 1
        
        # Calculate precision and recall for each class
        precisions = []
        recalls = []
        
        for class_id, stat in class_stats.items():
            if stat['gt'] == 0:
                continue
            
            precision = stat['tp'] / (stat['tp'] + stat['fp'] + 1e-7)
            recall = stat['tp'] / (stat['gt'] + 1e-7)
            
            precisions.append(precision)
            recalls.append(recall)
        
        # Calculate mAP
        if len(precisions) > 0:
            map_score = sum(precisions) / len(precisions)
        else:
            map_score = 0.0
        
        stats.append(map_score)
    
    # Return mAP@0.5 and mAP@0.5:0.95
    map50 = stats[0]
    map = sum(stats) / len(stats)
    
    return [map50, map]


def plot_pr_curve(results, save_dir=None, class_names=None):
    """
    Plot precision-recall curves for object detection results.
    
    Args:
        results: List of dictionaries containing detection results
        save_dir: Directory to save the plots
        class_names: List of class names
    """
    if save_dir is not None:
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)
    
    # Initialize per-class statistics
    class_stats = defaultdict(lambda: {'scores': [], 'tp': [], 'fp': [], 'gt': 0})
    
    # Process each result
    for result in results:
        pred_boxes = result['boxes']
        pred_scores = result['scores']
        pred_labels = result['labels']
        gt_boxes = result['targets'][:, :4]
        gt_labels = result['targets'][:, 4].long()
        
        # Skip if no predictions or ground truth
        if len(pred_boxes) == 0 or len(gt_boxes) == 0:
            continue
        
        # Calculate IoU between predictions and ground truth
        ious = box_iou(pred_boxes, gt_boxes)
        
        # Process each prediction
        for i, (box, score, label) in enumerate(zip(pred_boxes, pred_scores, pred_labels)):
            # Get IoUs for this prediction with all ground truth boxes
            box_ious = ious[i]
            
            # Get ground truth boxes with the same class
            same_class_mask = gt_labels == label
            
            # Store prediction score
            class_stats[label.item()]['scores'].append(score.item())
            
            if not same_class_mask.any():
                # No ground truth of this class, false positive
                class_stats[label.item()]['tp'].append(0)
                class_stats[label.item()]['fp'].append(1)
                continue
            
            # Get IoUs with ground truth boxes of the same class
            class_ious = box_ious[same_class_mask]
            
            if class_ious.max() >= 0.5:
                # True positive
                class_stats[label.item()]['tp'].append(1)
                class_stats[label.item()]['fp'].append(0)
            else:
                # False positive
                class_stats[label.item()]['tp'].append(0)
                class_stats[label.item()]['fp'].append(1)
        
        # Count ground truth boxes for each class
        for label in gt_labels:
            class_stats[label.item()]['gt'] += 1
    
    # Plot precision-recall curves for each class
    plt.figure(figsize=(10, 8))
    
    for class_id, stat in class_stats.items():
        if stat['gt'] == 0 or len(stat['scores']) == 0:
            continue
        
        # Sort by confidence score
        indices = np.argsort(-np.array(stat['scores']))
        tp = np.array(stat['tp'])[indices]
        fp = np.array(stat['fp'])[indices]
        
        # Calculate cumulative TP and FP
        tp_cumsum = np.cumsum(tp)
        fp_cumsum = np.cumsum(fp)
        
        # Calculate precision and recall
        precision = tp_cumsum / (tp_cumsum + fp_cumsum)
        recall = tp_cumsum / stat['gt']
        
        # Add (1, 0) point for plotting
        precision = np.concatenate(([1], precision))
        recall = np.concatenate(([0], recall))
        
        # Calculate AP
        ap = compute_ap(recall, precision)
        
        # Plot precision-recall curve
        class_name = f"Class {class_id}" if class_names is None else class_names[class_id]
        plt.plot(recall, precision, label=f"{class_name} (AP={ap:.3f})")
    
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curves')
    plt.legend(loc='lower left')
    plt.grid(True)
    
    if save_dir is not None:
        plt.savefig(save_dir / 'pr_curve.png')
    
    plt.close()


def plot_confusion_matrix(results, num_classes, save_dir=None, class_names=None):
    """
    Plot confusion matrix for object detection results.
    
    Args:
        results: List of dictionaries containing detection results
        num_classes: Number of classes
        save_dir: Directory to save the plots
        class_names: List of class names
    """
    if save_dir is not None:
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)
    
    # Initialize confusion matrix
    confusion_matrix = np.zeros((num_classes, num_classes))
    
    # Process each result
    for result in results:
        pred_boxes = result['boxes']
        pred_scores = result['scores']
        pred_labels = result['labels']
        gt_boxes = result['targets'][:, :4]
        gt_labels = result['targets'][:, 4].long()
        
        # Skip if no predictions or ground truth
        if len(pred_boxes) == 0 or len(gt_boxes) == 0:
            continue
        
        # Calculate IoU between predictions and ground truth
        ious = box_iou(pred_boxes, gt_boxes)
        
        # Process each ground truth box
        for gt_idx, gt_label in enumerate(gt_labels):
            # Get IoUs for this ground truth with all predictions
            gt_ious = ious[:, gt_idx]
            
            if gt_ious.max() >= 0.5:
                # Find the prediction with highest IoU
                pred_idx = gt_ious.argmax()
                pred_label = pred_labels[pred_idx]
                
                # Update confusion matrix
                confusion_matrix[gt_label, pred_label] += 1
            else:
                # No prediction matched this ground truth
                confusion_matrix[gt_label, -1] += 1
    
    # Plot confusion matrix
    plt.figure(figsize=(12, 10))
    plt.imshow(confusion_matrix, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title('Confusion Matrix')
    plt.colorbar()
    
    if class_names is not None:
        tick_marks = np.arange(num_classes)
        plt.xticks(tick_marks, class_names, rotation=45)
        plt.yticks(tick_marks, class_names)
    
    plt.xlabel('Predicted')
    plt.ylabel('Ground Truth')
    plt.tight_layout()
    
    if save_dir is not None:
        plt.savefig(save_dir / 'confusion_matrix.png')
    
    plt.close() 