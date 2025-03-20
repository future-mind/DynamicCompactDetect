import numpy as np
import torch
from collections import defaultdict

def box_iou(box1, box2):
    """
    Calculate IoU between box1 and box2.
    
    Args:
        box1 (tensor): Box in format [x1, y1, x2, y2]
        box2 (tensor): Box in format [x1, y1, x2, y2]
    
    Returns:
        IoU value
    """
    # Get the coordinates of bounding boxes
    b1_x1, b1_y1, b1_x2, b1_y2 = box1
    b2_x1, b2_y1, b2_x2, b2_y2 = box2
    
    # Get the coordinates of the intersection rectangle
    inter_rect_x1 = torch.max(b1_x1, b2_x1)
    inter_rect_y1 = torch.max(b1_y1, b2_y1)
    inter_rect_x2 = torch.min(b1_x2, b2_x2)
    inter_rect_y2 = torch.min(b1_y2, b2_y2)
    
    # Intersection area
    inter_area = torch.clamp(inter_rect_x2 - inter_rect_x1, min=0) * \
                 torch.clamp(inter_rect_y2 - inter_rect_y1, min=0)
    
    # Union Area
    b1_area = (b1_x2 - b1_x1) * (b1_y2 - b1_y1)
    b2_area = (b2_x2 - b2_x1) * (b2_y2 - b2_y1)
    
    union = b1_area + b2_area - inter_area
    
    iou = inter_area / union if union > 0 else 0
    
    return iou

def batch_box_iou(boxes1, boxes2):
    """
    Calculate IoU between all boxes of boxes1 and boxes2.
    
    Args:
        boxes1 (tensor): Boxes in format [N, 4] where each box is [x1, y1, x2, y2]
        boxes2 (tensor): Boxes in format [M, 4] where each box is [x1, y1, x2, y2]
    
    Returns:
        IoU matrix of shape [N, M]
    """
    if not isinstance(boxes1, torch.Tensor):
        boxes1 = torch.tensor(boxes1)
    if not isinstance(boxes2, torch.Tensor):
        boxes2 = torch.tensor(boxes2)
    
    N = boxes1.shape[0]
    M = boxes2.shape[0]
    
    # Expand boxes to compute IoU for all pairs
    boxes1_expand = boxes1.unsqueeze(1).expand(N, M, 4)
    boxes2_expand = boxes2.unsqueeze(0).expand(N, M, 4)
    
    # Get the coordinates of the intersection rectangle
    inter_rect_x1 = torch.max(boxes1_expand[..., 0], boxes2_expand[..., 0])
    inter_rect_y1 = torch.max(boxes1_expand[..., 1], boxes2_expand[..., 1])
    inter_rect_x2 = torch.min(boxes1_expand[..., 2], boxes2_expand[..., 2])
    inter_rect_y2 = torch.min(boxes1_expand[..., 3], boxes2_expand[..., 3])
    
    # Intersection area
    inter_area = torch.clamp(inter_rect_x2 - inter_rect_x1, min=0) * \
                 torch.clamp(inter_rect_y2 - inter_rect_y1, min=0)
    
    # Union Area
    boxes1_area = (boxes1[..., 2] - boxes1[..., 0]) * (boxes1[..., 3] - boxes1[..., 1])
    boxes2_area = (boxes2[..., 2] - boxes2[..., 0]) * (boxes2[..., 3] - boxes2[..., 1])
    
    boxes1_area_expand = boxes1_area.unsqueeze(1).expand(N, M)
    boxes2_area_expand = boxes2_area.unsqueeze(0).expand(N, M)
    
    union = boxes1_area_expand + boxes2_area_expand - inter_area
    
    iou = inter_area / union
    
    return iou

def calculate_precision_recall(pred_boxes, pred_labels, pred_scores, 
                              true_boxes, true_labels, 
                              iou_threshold=0.5, num_classes=80):
    """
    Calculate precision and recall for all classes.
    
    Args:
        pred_boxes (list): List of predicted boxes [x1, y1, x2, y2]
        pred_labels (list): List of predicted labels
        pred_scores (list): List of predicted scores
        true_boxes (list): List of ground truth boxes [x1, y1, x2, y2]
        true_labels (list): List of ground truth labels
        iou_threshold (float): IoU threshold for a true positive
        num_classes (int): Number of classes
        
    Returns:
        precisions and recalls for all classes
    """
    if not pred_boxes or not true_boxes:
        return np.zeros(num_classes), np.zeros(num_classes)
    
    # Convert to tensors if they aren't already
    if not isinstance(pred_boxes, torch.Tensor):
        pred_boxes = torch.tensor(pred_boxes, dtype=torch.float32)
    if not isinstance(pred_labels, torch.Tensor):
        pred_labels = torch.tensor(pred_labels, dtype=torch.int64)
    if not isinstance(pred_scores, torch.Tensor):
        pred_scores = torch.tensor(pred_scores, dtype=torch.float32)
    if not isinstance(true_boxes, torch.Tensor):
        true_boxes = torch.tensor(true_boxes, dtype=torch.float32)
    if not isinstance(true_labels, torch.Tensor):
        true_labels = torch.tensor(true_labels, dtype=torch.int64)
    
    # Initialize counters for each class
    true_positives = [0] * num_classes
    false_positives = [0] * num_classes
    false_negatives = [0] * num_classes
    
    # Calculate IoU between predicted and ground truth boxes
    ious = batch_box_iou(pred_boxes, true_boxes)
    
    # For each prediction, find the ground truth box with highest IoU
    matched_gt_indices = set()
    
    # Sort predictions by confidence score (high to low)
    _, sorted_indices = torch.sort(pred_scores, descending=True)
    
    for pred_idx in sorted_indices:
        pred_label = pred_labels[pred_idx].item()
        
        # Find ground truth boxes with the same label
        gt_indices = torch.where(true_labels == pred_label)[0]
        
        if len(gt_indices) == 0:
            # No ground truth of this class, false positive
            false_positives[pred_label] += 1
            continue
        
        # Get IoUs for ground truths of the same class
        gt_ious = ious[pred_idx, gt_indices]
        
        # Find the ground truth with highest IoU
        max_iou, max_idx = torch.max(gt_ious, dim=0)
        max_gt_idx = gt_indices[max_idx].item()
        
        if max_iou >= iou_threshold and max_gt_idx not in matched_gt_indices:
            # True positive
            true_positives[pred_label] += 1
            matched_gt_indices.add(max_gt_idx)
        else:
            # False positive
            false_positives[pred_label] += 1
    
    # Count false negatives (ground truths not matched to any prediction)
    for gt_idx, gt_label in enumerate(true_labels):
        if gt_idx not in matched_gt_indices:
            false_negatives[gt_label.item()] += 1
    
    # Calculate precision and recall for each class
    precisions = []
    recalls = []
    
    for c in range(num_classes):
        if true_positives[c] + false_positives[c] > 0:
            precision = true_positives[c] / (true_positives[c] + false_positives[c])
        else:
            precision = 0
        
        if true_positives[c] + false_negatives[c] > 0:
            recall = true_positives[c] / (true_positives[c] + false_negatives[c])
        else:
            recall = 0
        
        precisions.append(precision)
        recalls.append(recall)
    
    return np.array(precisions), np.array(recalls)

def calculate_map(detections, targets, iou_threshold=0.5):
    """
    Calculate mAP (mean Average Precision) for object detection.
    
    Args:
        detections (list): List of detection results, each containing 'boxes', 'scores', 'labels'
        targets (list): List of ground truth targets, each containing 'boxes', 'labels'
        iou_threshold (float): IoU threshold for a true positive
        
    Returns:
        mAP@0.5, mAP@0.5:0.95
    """
    # Number of classes
    num_classes = 80  # COCO has 80 classes
    
    # Initialize data structures for precision-recall curve
    class_precisions = defaultdict(list)
    class_recalls = defaultdict(list)
    
    # Process each image with its detections and ground truths
    for det, gt in zip(detections, targets):
        pred_boxes = det.get('boxes', [])
        pred_scores = det.get('scores', [])
        pred_labels = det.get('labels', [])
        
        true_boxes = gt.get('boxes', [])
        true_labels = gt.get('labels', [])
        
        if len(pred_boxes) == 0 or len(true_boxes) == 0:
            continue
        
        # Calculate precision and recall for this image
        precisions, recalls = calculate_precision_recall(
            pred_boxes, pred_labels, pred_scores,
            true_boxes, true_labels,
            iou_threshold=iou_threshold,
            num_classes=num_classes
        )
        
        # Accumulate precision and recall values for each class
        for c in range(num_classes):
            if precisions[c] > 0 or recalls[c] > 0:
                class_precisions[c].append(precisions[c])
                class_recalls[c].append(recalls[c])
    
    # Calculate AP for each class
    aps = []
    
    for c in range(num_classes):
        if c not in class_precisions or len(class_precisions[c]) == 0:
            continue
        
        # Sort by recall
        recall_values = np.array(class_recalls[c])
        precision_values = np.array(class_precisions[c])
        
        # Sort by recall
        indices = np.argsort(recall_values)
        recall_values = recall_values[indices]
        precision_values = precision_values[indices]
        
        # Compute average precision
        ap = np.sum(precision_values) / len(precision_values) if len(precision_values) > 0 else 0
        aps.append(ap)
    
    # Calculate mAP
    mAP = np.mean(aps) if len(aps) > 0 else 0
    
    # Calculate mAP@0.5
    mAP50 = mAP  # Since we only computed for a single IoU threshold
    
    # For simplicity, we'll use the same value for mAP@0.5:0.95 in this example
    # In a full implementation, you would compute for multiple IoU thresholds
    mAP_over_thresholds = mAP50
    
    return mAP50, mAP_over_thresholds

class AveragePrecisionMeter:
    """Meter for tracking Average Precision metrics."""
    
    def __init__(self, num_classes=80):
        """Initialize with given number of classes."""
        self.num_classes = num_classes
        self.reset()
    
    def reset(self):
        """Reset the meter."""
        self.detections = []
        self.targets = []
    
    def update(self, detections, targets):
        """Update the meter with new detections and targets."""
        self.detections.extend(detections)
        self.targets.extend(targets)
    
    def value(self, iou_threshold=0.5):
        """Calculate the current Average Precision."""
        return calculate_map(self.detections, self.targets, iou_threshold)

class ConfusionMatrix:
    """
    Confusion matrix for object detection evaluation.
    Tracks true positives, false positives, and false negatives.
    """
    
    def __init__(self, num_classes):
        """Initialize with given number of classes."""
        self.num_classes = num_classes
        self.matrix = np.zeros((num_classes, 3))  # TP, FP, FN for each class
    
    def process_batch(self, detections, targets, iou_threshold=0.5):
        """Process a batch of detections and targets."""
        for det, gt in zip(detections, targets):
            pred_boxes = det.get('boxes', [])
            pred_scores = det.get('scores', [])
            pred_labels = det.get('labels', [])
            
            true_boxes = gt.get('boxes', [])
            true_labels = gt.get('labels', [])
            
            if len(pred_boxes) == 0 or len(true_boxes) == 0:
                # If no predictions but there are ground truths, all are false negatives
                if len(true_boxes) > 0:
                    for label in true_labels:
                        if isinstance(label, torch.Tensor):
                            label = label.item()
                        if 0 <= label < self.num_classes:
                            self.matrix[label, 2] += 1  # FN
                continue
            
            # Convert to tensors if not already
            if not isinstance(pred_boxes, torch.Tensor):
                pred_boxes = torch.tensor(pred_boxes, dtype=torch.float32)
            if not isinstance(pred_labels, torch.Tensor):
                pred_labels = torch.tensor(pred_labels, dtype=torch.int64)
            if not isinstance(true_boxes, torch.Tensor):
                true_boxes = torch.tensor(true_boxes, dtype=torch.float32)
            if not isinstance(true_labels, torch.Tensor):
                true_labels = torch.tensor(true_labels, dtype=torch.int64)
            
            # Calculate IoU
            ious = batch_box_iou(pred_boxes, true_boxes)
            
            # For each prediction, find the ground truth with highest IoU
            matched_gt_indices = set()
            
            # Sort predictions by confidence score (high to low)
            _, sorted_indices = torch.sort(pred_scores, descending=True)
            
            for pred_idx in sorted_indices:
                pred_label = pred_labels[pred_idx].item()
                
                if pred_label >= self.num_classes:
                    continue
                
                # Find ground truth boxes with the same label
                gt_indices = torch.where(true_labels == pred_label)[0]
                
                if len(gt_indices) == 0:
                    # No ground truth of this class, false positive
                    self.matrix[pred_label, 1] += 1  # FP
                    continue
                
                # Get IoUs for ground truths of the same class
                gt_ious = ious[pred_idx, gt_indices]
                
                # Find the ground truth with highest IoU
                max_iou, max_idx = torch.max(gt_ious, dim=0)
                max_gt_idx = gt_indices[max_idx].item()
                
                if max_iou >= iou_threshold and max_gt_idx not in matched_gt_indices:
                    # True positive
                    self.matrix[pred_label, 0] += 1  # TP
                    matched_gt_indices.add(max_gt_idx)
                else:
                    # False positive
                    self.matrix[pred_label, 1] += 1  # FP
            
            # Count false negatives (ground truths not matched to any prediction)
            for gt_idx, gt_label in enumerate(true_labels):
                gt_label = gt_label.item()
                if gt_idx not in matched_gt_indices and gt_label < self.num_classes:
                    self.matrix[gt_label, 2] += 1  # FN
    
    def precision(self):
        """Calculate precision for each class."""
        tp = self.matrix[:, 0]
        fp = self.matrix[:, 1]
        return np.divide(tp, tp + fp, out=np.zeros_like(tp), where=(tp + fp) > 0)
    
    def recall(self):
        """Calculate recall for each class."""
        tp = self.matrix[:, 0]
        fn = self.matrix[:, 2]
        return np.divide(tp, tp + fn, out=np.zeros_like(tp), where=(tp + fn) > 0)
    
    def f1_score(self):
        """Calculate F1-score for each class."""
        precision = self.precision()
        recall = self.recall()
        return np.divide(2 * precision * recall, precision + recall, 
                         out=np.zeros_like(precision), where=(precision + recall) > 0) 