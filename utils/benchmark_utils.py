import time
import torch
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from collections import defaultdict
from matplotlib.gridspec import GridSpec

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

def measure_inference_time(model, input_size=(640, 640), iterations=100, warm_up=10, device='cpu', early_exit=True):
    """
    Measure the inference time of a model.
    
    Args:
        model: The PyTorch model to benchmark
        input_size: Tuple of (width, height) for the input
        iterations: Number of iterations to measure
        warm_up: Number of warm-up iterations (not measured)
        device: Device to run inference on
        early_exit: Whether to enable early exit for models that support it
    
    Returns:
        Average time per inference in seconds, FPS
    """
    width, height = input_size
    
    # Create dummy input
    dummy_input = torch.randn(1, 3, height, width, device=device)
    
    # Warm up
    model.eval()
    with torch.no_grad():
        for _ in range(warm_up):
            if hasattr(model, 'forward') and 'early_exit' in model.forward.__code__.co_varnames:
                _ = model(dummy_input, early_exit=early_exit)
            else:
                _ = model(dummy_input)
    
    # Measure time
    start_time = time.time()
    with torch.no_grad():
        for _ in range(iterations):
            if device.type == 'cuda':
                torch.cuda.synchronize()
            
            if hasattr(model, 'forward') and 'early_exit' in model.forward.__code__.co_varnames:
                _ = model(dummy_input, early_exit=early_exit)
            else:
                _ = model(dummy_input)
            
            if device.type == 'cuda':
                torch.cuda.synchronize()
    
    end_time = time.time()
    
    # Calculate average time
    avg_time = (end_time - start_time) / iterations
    fps = 1.0 / avg_time
    
    return avg_time, fps

def benchmark_model(model, input_sizes, device, iterations=100, early_exit=True):
    """
    Benchmark a model on various input sizes.
    
    Args:
        model: The PyTorch model to benchmark
        input_sizes: List of (width, height) tuples
        device: Device to run inference on
        iterations: Number of iterations to measure
        early_exit: Whether to enable early exit for models that support it
    
    Returns:
        Dictionary of benchmark results
    """
    results = {}
    
    for input_size in input_sizes:
        width, height = input_size
        
        # Measure inference time
        avg_time, fps = measure_inference_time(
            model, input_size=(width, height), 
            iterations=iterations, warm_up=10, 
            device=device, early_exit=early_exit
        )
        
        results[f"{width}x{height}"] = {
            'avg_time_ms': avg_time * 1000,  # Convert to ms
            'fps': fps
        }
    
    # Add model size info
    num_params = sum(p.numel() for p in model.parameters())
    model_size_mb = num_params * 4 / (1024 * 1024)  # Approximate size in MB
    
    results['model_info'] = {
        'parameters': num_params,
        'model_size_mb': model_size_mb
    }
    
    return results

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

def plot_comparison_chart(model_names, model_sizes, param_counts, inference_times, fps_values, output_path):
    """
    Create a comprehensive comparison chart of model metrics.
    
    Args:
        model_names: List of model names
        model_sizes: List of model sizes in MB
        param_counts: List of parameter counts in millions
        inference_times: List of inference times in ms
        fps_values: List of FPS values
        output_path: Path to save the plot
    """
    plt.figure(figsize=(15, 10))
    gs = GridSpec(2, 2)
    
    # Convert param_counts to millions for display
    param_counts_m = [p / 1_000_000 for p in param_counts]
    
    # Find the balance between size and speed
    efficiency = [fps / size if size > 0 else 0 for fps, size in zip(fps_values, model_sizes)]
    
    # Plot 1: Model Size vs Inference Time
    ax1 = plt.subplot(gs[0, 0])
    scatter1 = ax1.scatter(model_sizes, inference_times, s=param_counts_m * 10, alpha=0.7)
    
    for i, model in enumerate(model_names):
        ax1.annotate(model, (model_sizes[i], inference_times[i]), fontsize=9)
    
    ax1.set_xlabel('Model Size (MB)')
    ax1.set_ylabel('Inference Time (ms)')
    ax1.set_title('Model Size vs Inference Time')
    ax1.grid(True, linestyle='--', alpha=0.7)
    
    # Plot 2: Model Size vs FPS
    ax2 = plt.subplot(gs[0, 1])
    scatter2 = ax2.scatter(model_sizes, fps_values, s=param_counts_m * 10, alpha=0.7)
    
    for i, model in enumerate(model_names):
        ax2.annotate(model, (model_sizes[i], fps_values[i]), fontsize=9)
    
    ax2.set_xlabel('Model Size (MB)')
    ax2.set_ylabel('FPS')
    ax2.set_title('Model Size vs FPS')
    ax2.grid(True, linestyle='--', alpha=0.7)
    
    # Plot 3: Efficiency comparison
    ax3 = plt.subplot(gs[1, 0])
    bars = ax3.bar(model_names, efficiency)
    
    # Add value labels on bars
    for bar, value in zip(bars, efficiency):
        ax3.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1,
                f'{value:.2f}', ha='center', fontsize=9)
    
    ax3.set_xlabel('Model')
    ax3.set_ylabel('Efficiency (FPS/MB)')
    ax3.set_title('Model Efficiency (FPS/MB)')
    ax3.tick_params(axis='x', rotation=45)
    
    # Plot 4: Parameter count vs inference time
    ax4 = plt.subplot(gs[1, 1])
    scatter4 = ax4.scatter(param_counts_m, inference_times, s=100, alpha=0.7)
    
    for i, model in enumerate(model_names):
        ax4.annotate(model, (param_counts_m[i], inference_times[i]), fontsize=9)
    
    ax4.set_xlabel('Parameters (M)')
    ax4.set_ylabel('Inference Time (ms)')
    ax4.set_title('Parameters vs Inference Time')
    ax4.grid(True, linestyle='--', alpha=0.7)
    
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()

def compare_model_implementations(implementation_results, output_path):
    """
    Compare different model implementations or configurations.
    
    Args:
        implementation_results: Dictionary mapping implementation names to results
        output_path: Path to save the comparison plot
    """
    implementations = list(implementation_results.keys())
    
    # Extract metrics for comparison
    fps_values = []
    model_sizes = []
    param_counts = []
    
    for impl in implementations:
        results = implementation_results[impl]
        
        # Try to find 640x640 results, or use first available
        fps = 0
        for key in results:
            if key == '640x640':
                fps = results[key]['fps']
                break
            elif 'x' in key and not key.endswith('_early_exit') and not key.endswith('_no_early_exit'):
                fps = results[key]['fps']
        
        fps_values.append(fps)
        
        if 'model_info' in results:
            model_sizes.append(results['model_info']['model_size_mb'])
            param_counts.append(results['model_info']['parameters'])
        else:
            model_sizes.append(0)
            param_counts.append(0)
    
    # Plot comparisons
    plt.figure(figsize=(12, 6))
    
    # FPS comparison
    plt.subplot(1, 2, 1)
    bars1 = plt.bar(implementations, fps_values)
    
    for bar, value in zip(bars1, fps_values):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                f'{value:.1f}', ha='center', fontsize=9)
    
    plt.xlabel('Implementation')
    plt.ylabel('FPS')
    plt.title('Performance Comparison')
    plt.xticks(rotation=45)
    
    # Efficiency comparison
    plt.subplot(1, 2, 2)
    efficiency = [fps / size if size > 0 else 0 for fps, size in zip(fps_values, model_sizes)]
    bars2 = plt.bar(implementations, efficiency)
    
    for bar, value in zip(bars2, efficiency):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.05,
                f'{value:.2f}', ha='center', fontsize=9)
    
    plt.xlabel('Implementation')
    plt.ylabel('Efficiency (FPS/MB)')
    plt.title('Efficiency Comparison')
    plt.xticks(rotation=45)
    
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()

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