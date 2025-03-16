import cv2
import numpy as np
import torch
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from pathlib import Path
import random
import math


class Colors:
    """
    Class for creating distinctive colors for different object classes.
    """
    def __init__(self):
        # Tableau palette
        self.palette = [
            (255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0), (255, 0, 255), (0, 255, 255),
            (128, 0, 0), (0, 128, 0), (0, 0, 128), (128, 128, 0), (128, 0, 128), (0, 128, 128),
            (64, 0, 0), (0, 64, 0), (0, 0, 64), (64, 64, 0), (64, 0, 64), (0, 64, 64),
            (192, 0, 0), (0, 192, 0), (0, 0, 192), (192, 192, 0), (192, 0, 192), (0, 192, 192)
        ]
        self.n = len(self.palette)

    def __call__(self, i, bgr=False):
        """
        Get color for class index i.
        
        Args:
            i: Class index
            bgr: Whether to return color in BGR format (for OpenCV)
            
        Returns:
            Color tuple
        """
        c = self.palette[int(i) % self.n]
        return (c[2], c[1], c[0]) if bgr else c


def plot_one_box(x, img, color=None, label=None, line_thickness=None):
    """
    Plots one bounding box on the image.
    
    Args:
        x: Bounding box coordinates [x1, y1, x2, y2]
        img: Image to plot on
        color: Color of the bounding box
        label: Label text to display
        line_thickness: Line thickness
        
    Returns:
        None (modifies img in-place)
    """
    # Convert tensor to numpy array if necessary
    if isinstance(x, torch.Tensor):
        x = x.cpu().numpy()
    
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


def plot_images(images, targets=None, paths=None, fname='images.jpg', names=None, max_size=640, max_subplots=16):
    """
    Plot a batch of images with targets (bounding boxes).
    
    Args:
        images: Batch of images
        targets: Targets with format [batch_idx, class_idx, x, y, w, h]
        paths: Image paths
        fname: Output filename
        names: Class names
        max_size: Max image size
        max_subplots: Max number of subplots
        
    Returns:
        None (saves plot to file)
    """
    # Convert tensor to numpy array if necessary
    if isinstance(images, torch.Tensor):
        images = images.cpu().numpy()
    
    # Create figure
    bs, _, h, w = images.shape
    bs = min(bs, max_subplots)  # limit plot images
    ns = np.ceil(bs ** 0.5)  # number of subplots (square)
    
    # Check if we should resize
    scale_factor = max_size / max(h, w)
    if scale_factor < 1:
        h = int(h * scale_factor)
        w = int(w * scale_factor)
    
    mosaic = np.full((int(ns * h), int(ns * w), 3), 255, dtype=np.uint8)  # init
    
    # Create colors
    colors = Colors()
    
    # Plot images and targets
    for i, img in enumerate(images):
        if i == max_subplots:  # if last batch has fewer images than we expect
            break
        
        # Convert to RGB with correct shape
        img = img.transpose(1, 2, 0)  # (3, h, w) -> (h, w, 3)
        img = (img * 255).astype(np.uint8)
        
        # Get subplot position
        block_x = int(w * (i % ns))
        block_y = int(h * (i // ns))
        
        # Add image to mosaic
        mosaic[block_y:block_y + h, block_x:block_x + w, :] = img
        
        # Draw bounding boxes
        if targets is not None:
            image_targets = targets[targets[:, 0] == i]
            boxes = xywh2xyxy(image_targets[:, 2:6]).T
            classes = image_targets[:, 1].astype('int')
            
            # Scale boxes to image size
            boxes[0:4] = boxes[0:4] * np.array([w, h, w, h])
            
            # Add boxes to mosaic
            for j, box in enumerate(boxes.T):
                cls = int(classes[j])
                color = colors(cls)
                cls = names[cls] if names else str(cls)
                plot_one_box(box, mosaic[block_y:block_y + h, block_x:block_x + w, :],
                           label=cls, color=color)
    
    # Save mosaic
    cv2.imwrite(str(fname), cv2.cvtColor(mosaic, cv2.COLOR_RGB2BGR))


def plot_results(file='', dir='', segment=False):
    """
    Plot training results from file. Supports plotting multiple metrics.
    
    Args:
        file: Results file (can be a list for multiple files)
        dir: Directory containing results files
        segment: Whether to segment plots
        
    Returns:
        None (creates plots)
    """
    fig, ax = plt.subplots(2, 5, figsize=(12, 6), tight_layout=True)
    ax = ax.ravel()
    files = list(Path(dir).glob('results*.txt')) if dir else [Path(f) for f in file if Path(f).exists()]
    
    for fi, f in enumerate(files):
        try:
            results = np.loadtxt(f, usecols=[2, 3, 4, 5, 6, 7, 8, 9]).T
            n = results.shape[1]  # number of rows
            x = range(n)
            
            # Plot metrics
            titles = ['mAP@0.5', 'mAP@0.5:0.95', 'Precision', 'Recall', 'Box Loss', 'Obj Loss', 'Cls Loss', 'Total Loss']
            for i, title in enumerate(titles):
                if i < results.shape[0]:
                    y = results[i, x]
                    ax[i].plot(x, y, marker='.', linewidth=2, markersize=8, label=f.stem)
                    ax[i].set_title(title)
            
            # Set axis labels
            for a in ax[:8]:
                a.set_xlabel('Epoch')
                a.legend()
        except Exception as e:
            print(f'Warning: Plotting error for {f}: {e}')
    
    # Save figure
    plt.savefig(dir / 'results.png', dpi=200)
    plt.close()


def plot_lr_scheduler(optimizer, scheduler, epochs=300):
    """Plot learning rate scheduler."""
    
    y = []
    for epoch in range(epochs):
        scheduler.step()
        y.append(optimizer.param_groups[0]['lr'])
    plt.plot(y, label='LR')
    plt.xlabel('epoch')
    plt.ylabel('LR')
    plt.tight_layout()
    plt.savefig('LR.png', dpi=200)


def plot_training_results(results_file='results.csv'):
    """Plot training results from a CSV file."""
    
    # Load CSV file
    try:
        data = np.loadtxt(results_file, delimiter=',', skiprows=1)
    except:
        return
        
    # Create figure and axes
    fig, ax = plt.subplots(2, 5, figsize=(12, 6), tight_layout=True)
    ax = ax.ravel()
    s = ['Box', 'Objectness', 'Classification', 'Precision', 'Recall',
         'val Box', 'val Objectness', 'val Classification', 'mAP@0.5', 'mAP@0.5:0.95']
         
    for i in range(10):
        if i < data.shape[1]:
            ax[i].plot(data[:, i])
        ax[i].set_title(s[i])
    
    fig.savefig('results.png', dpi=200)
    plt.close()


# Import needed for plotting
from utils.general import xywh2xyxy 