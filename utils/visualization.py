import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import torch
from PIL import Image, ImageDraw, ImageFont

def visualize_detections(image, boxes, scores, labels, class_names=None, threshold=0.25):
    """
    Visualize detections on an image.
    
    Args:
        image: Image as numpy array (H, W, C)
        boxes: Bounding boxes as numpy array (N, 4) in format [x1, y1, x2, y2]
        scores: Detection scores as numpy array (N,)
        labels: Class labels as numpy array (N,)
        class_names: List of class names
        threshold: Score threshold for displaying detections
    
    Returns:
        Annotated image as numpy array
    """
    if isinstance(image, torch.Tensor):
        image = image.permute(1, 2, 0).cpu().numpy()
    
    # Make a copy to avoid modifying the original
    img_viz = image.copy()
    
    # Create figure and axis
    fig, ax = plt.subplots(1, figsize=(12, 9))
    ax.imshow(img_viz)
    
    # Draw each detection
    for box, score, label in zip(boxes, scores, labels):
        if score < threshold:
            continue
        
        # Convert label to int if it's a tensor
        if isinstance(label, torch.Tensor):
            label = label.item()
        
        # Get class name
        if class_names and label < len(class_names):
            class_name = class_names[label]
        else:
            class_name = f"Class {label}"
        
        # Create rectangle
        x1, y1, x2, y2 = box
        width, height = x2 - x1, y2 - y1
        
        # Add rectangle patch
        rect = Rectangle((x1, y1), width, height, 
                         fill=False, edgecolor='red', linewidth=2)
        ax.add_patch(rect)
        
        # Add label
        plt.text(x1, y1, f"{class_name}: {score:.2f}", 
                 bbox=dict(facecolor='red', alpha=0.5),
                 fontsize=10, color='white')
    
    # Remove axis
    plt.axis('off')
    
    # Convert figure to numpy array
    fig.canvas.draw()
    img_viz = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
    img_viz = img_viz.reshape(fig.canvas.get_width_height()[::-1] + (3,))
    
    plt.close(fig)
    
    return img_viz

def draw_multiple_detections(image, detection_results, model_names, class_names=None, threshold=0.25):
    """
    Draw detections from multiple models on the same image.
    
    Args:
        image: Image as numpy array (H, W, C)
        detection_results: Dictionary mapping model names to detection results
        model_names: List of model names to include
        class_names: List of class names
        threshold: Score threshold for displaying detections
    
    Returns:
        Grid of annotated images as numpy array
    """
    if isinstance(image, torch.Tensor):
        image = image.permute(1, 2, 0).cpu().numpy()
    
    # Calculate grid dimensions
    num_models = len(model_names)
    grid_size = int(np.ceil(np.sqrt(num_models + 1)))  # +1 for original image
    
    fig, axes = plt.subplots(grid_size, grid_size, figsize=(15, 15))
    axes = axes.flatten()
    
    # Original image
    axes[0].imshow(image)
    axes[0].set_title("Original Image")
    axes[0].axis('off')
    
    # Draw detections for each model
    for i, model_name in enumerate(model_names):
        if model_name not in detection_results:
            continue
        
        # Get detection results for this model
        result = detection_results[model_name]
        
        if not result or 'boxes' not in result:
            continue
        
        boxes = result['boxes']
        scores = result['scores']
        labels = result['labels']
        
        # Plot on axis
        axes[i + 1].imshow(image)
        axes[i + 1].set_title(model_name)
        axes[i + 1].axis('off')
        
        # Draw each detection
        for box, score, label in zip(boxes, scores, labels):
            if score < threshold:
                continue
            
            # Convert label to int if it's a tensor
            if isinstance(label, torch.Tensor):
                label = label.item()
            
            # Get class name
            if class_names and label < len(class_names):
                class_name = class_names[label]
            else:
                class_name = f"Class {label}"
            
            # Create rectangle
            x1, y1, x2, y2 = box
            width, height = x2 - x1, y2 - y1
            
            # Add rectangle patch
            rect = Rectangle((x1, y1), width, height, 
                             fill=False, edgecolor='red', linewidth=2)
            axes[i + 1].add_patch(rect)
            
            # Add label
            axes[i + 1].text(x1, y1, f"{class_name}: {score:.2f}", 
                           bbox=dict(facecolor='red', alpha=0.5),
                           fontsize=8, color='white')
    
    # Hide empty subplots
    for i in range(num_models + 1, grid_size * grid_size):
        axes[i].axis('off')
    
    plt.tight_layout()
    
    # Convert figure to numpy array
    fig.canvas.draw()
    img_grid = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
    img_grid = img_grid.reshape(fig.canvas.get_width_height()[::-1] + (3,))
    
    plt.close(fig)
    
    return img_grid

def plot_training_metrics(metrics, output_path):
    """
    Plot training metrics over time.
    
    Args:
        metrics: Dictionary of metrics with keys 'train_loss', 'val_loss', 'map', 'lr'
        output_path: Path to save the plot
    """
    # Create figure with multiple subplots
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    # Plot training and validation loss
    if 'train_loss' in metrics and 'val_loss' in metrics:
        ax = axes[0, 0]
        epochs = range(1, len(metrics['train_loss']) + 1)
        
        ax.plot(epochs, metrics['train_loss'], 'b-', label='Training Loss')
        ax.plot(epochs, metrics['val_loss'], 'r-', label='Validation Loss')
        ax.set_title('Loss')
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Loss')
        ax.legend()
        ax.grid(True, linestyle='--', alpha=0.6)
    
    # Plot mAP
    if 'map' in metrics:
        ax = axes[0, 1]
        epochs = range(1, len(metrics['map']) + 1)
        
        ax.plot(epochs, metrics['map'], 'g-', label='mAP')
        ax.set_title('Mean Average Precision (mAP)')
        ax.set_xlabel('Epoch')
        ax.set_ylabel('mAP')
        ax.grid(True, linestyle='--', alpha=0.6)
    
    # Plot learning rate
    if 'lr' in metrics:
        ax = axes[1, 0]
        epochs = range(1, len(metrics['lr']) + 1)
        
        ax.plot(epochs, metrics['lr'], 'm-', label='Learning Rate')
        ax.set_title('Learning Rate')
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Learning Rate')
        ax.grid(True, linestyle='--', alpha=0.6)
        
        # Use log scale if learning rate varies significantly
        if max(metrics['lr']) / min(metrics['lr']) > 10:
            ax.set_yscale('log')
    
    # Plot loss ratio (train/val)
    if 'train_loss' in metrics and 'val_loss' in metrics:
        ax = axes[1, 1]
        epochs = range(1, len(metrics['train_loss']) + 1)
        
        loss_ratio = [t / v if v > 0 else 1.0 for t, v in zip(metrics['train_loss'], metrics['val_loss'])]
        ax.plot(epochs, loss_ratio, 'c-', label='Train/Val Loss Ratio')
        ax.axhline(y=1.0, color='r', linestyle='--', alpha=0.6)
        ax.set_title('Train/Val Loss Ratio')
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Ratio')
        ax.grid(True, linestyle='--', alpha=0.6)
    
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()

def create_comparison_grid(images, labels, output_path):
    """
    Create a grid of comparison images with labels.
    
    Args:
        images: List of images as numpy arrays
        labels: List of labels for the images
        output_path: Path to save the grid
    """
    # Determine grid dimensions
    n = len(images)
    grid_size = int(np.ceil(np.sqrt(n)))
    
    fig, axes = plt.subplots(grid_size, grid_size, figsize=(15, 15))
    axes = axes.flatten()
    
    for i, (img, label) in enumerate(zip(images, labels)):
        if i < len(axes):
            axes[i].imshow(img)
            axes[i].set_title(label)
            axes[i].axis('off')
    
    # Hide empty subplots
    for i in range(n, grid_size * grid_size):
        axes[i].axis('off')
    
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close() 