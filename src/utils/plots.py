"""
Plotting utilities for YOLOv11
"""

import os
import math
import numpy as np
import torch
import cv2
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from pathlib import Path

# Set matplotlib backend
plt.switch_backend('agg')


def color_list():
    """
    Returns list of colors for visualization
    """
    # Return first 80 colors in 'hsv' colormap as hex
    return plt.cm.hsv(np.linspace(0, 1, 80)).tolist()


def plot_one_box(x, img, color=None, label=None, line_thickness=3):
    """
    Plots one bounding box on image
    
    Args:
        x: box coordinates [x1, y1, x2, y2]
        img: image to plot on
        color: box color
        label: label to plot
        line_thickness: line thickness
    """
    tl = line_thickness or round(0.002 * (img.shape[0] + img.shape[1]) / 2) + 1  # line/font thickness
    color = color or [random.randint(0, 255) for _ in range(3)]
    c1, c2 = (int(x[0]), int(x[1])), (int(x[2]), int(x[3]))
    cv2.rectangle(img, c1, c2, color, thickness=tl, lineType=cv2.LINE_AA)
    if label:
        tf = max(tl - 1, 1)  # font thickness
        t_size = cv2.getTextSize(label, 0, fontScale=tl / 3, thickness=tf)[0]
        c2 = c1[0] + t_size[0], c1[1] - t_size[1] - 3
        cv2.rectangle(img, c1, c2, color, -1, cv2.LINE_AA)  # filled
        cv2.putText(img, label, (c1[0], c1[1] - 2), 0, tl / 3, [225, 255, 255], thickness=tf, lineType=cv2.LINE_AA)


def plot_images(images, targets, paths=None, fname='images.jpg', names=None, max_size=640, max_subplots=16):
    """
    Plot image grid with labels
    
    Args:
        images: batch of images
        targets: batch of targets
        paths: list of image paths
        fname: output filename
        names: class names
        max_size: maximum image size
        max_subplots: maximum number of subplots
    """
    # Convert from torch to numpy
    if isinstance(images, torch.Tensor):
        images = images.cpu().float().numpy()
    if isinstance(targets, torch.Tensor):
        targets = targets.cpu().numpy()
        
    # un-normalise
    if np.max(images[0]) <= 1:
        images *= 255

    tl = 3  # line thickness
    tf = max(tl - 1, 1)  # font thickness
    bs, _, h, w = images.shape  # batch size, _, height, width
    bs = min(bs, max_subplots)  # limit plot images
    ns = np.ceil(bs ** 0.5)  # number of subplots (square)

    # Check if we should resize
    scale_factor = max_size / max(h, w)
    if scale_factor < 1:
        h = math.ceil(scale_factor * h)
        w = math.ceil(scale_factor * w)

    colors = color_list()  # list of colors
    mosaic = np.full((int(ns * h), int(ns * w), 3), 255, dtype=np.uint8)  # init

    for i, img in enumerate(images):
        if i == max_subplots:  # if last batch has fewer images than we expect
            break

        block_x = int(w * (i // ns))
        block_y = int(h * (i % ns))

        img = img.transpose(1, 2, 0)
        if scale_factor < 1:
            img = cv2.resize(img, (w, h))

        mosaic[block_y:block_y + h, block_x:block_x + w, :] = img
        if len(targets) > 0:
            image_targets = targets[targets[:, 0] == i]
            boxes = xywh2xyxy(image_targets[:, 2:6]).T
            classes = image_targets[:, 1].astype('int')
            labels = image_targets.shape[1] == 6  # labels if no conf column
            conf = None if labels else image_targets[:, 6]  # check for confidence presence (label vs pred)

            if boxes.shape[1]:
                if boxes.max() <= 1.01:  # if normalized with tolerance 0.01
                    boxes[[0, 2]] *= w  # scale to pixels
                    boxes[[1, 3]] *= h
                elif scale_factor < 1:  # absolute coords need scale if image scales
                    boxes *= scale_factor
            boxes[[0, 2]] += block_x
            boxes[[1, 3]] += block_y
            for j, box in enumerate(boxes.T):
                cls = int(classes[j])
                color = colors[cls % len(colors)]
                cls = names[cls] if names else cls
                if labels or conf[j] > 0.25:  # 0.25 conf thresh
                    label = '%s' % cls if labels else '%s %.1f' % (cls, conf[j])
                    plot_one_box(box, mosaic, label=label, color=color, line_thickness=tl)

        # Draw image filename labels
        if paths:
            label = Path(paths[i]).name[:40]  # trim to 40 char
            t_size = cv2.getTextSize(label, 0, fontScale=tl / 3, thickness=tf)[0]
            cv2.putText(mosaic, label, (block_x + 5, block_y + t_size[1] + 5), 0, tl / 3, [220, 220, 220], thickness=tf,
                      lineType=cv2.LINE_AA)

        # Image border
        cv2.rectangle(mosaic, (block_x, block_y), (block_x + w, block_y + h), (255, 255, 255), thickness=3)

    if fname:
        r = min(1280. / max(h, w) / ns, 1.0)  # ratio to limit image size
        mosaic = cv2.resize(mosaic, (int(ns * w * r), int(ns * h * r)), interpolation=cv2.INTER_AREA)
        # cv2.imwrite(fname, cv2.cvtColor(mosaic, cv2.COLOR_BGR2RGB))  # cv2 save
        mosaic = cv2.cvtColor(mosaic, cv2.COLOR_BGR2RGB)  # convert to RGB
        plt.imshow(mosaic)
        plt.savefig(fname, dpi=300, bbox_inches='tight')
        plt.close()

    return mosaic


def plot_pr_curve(px, py, ap, save_dir='pr_curve.png', names=()):
    """
    Plots precision-recall curve
    
    Args:
        px: x-values for plotting
        py: y-values for plotting
        ap: average precision values
        save_dir: directory to save plot
        names: class names
    """
    fig, ax = plt.subplots(1, 1, figsize=(9, 6), tight_layout=True)
    py = np.stack(py, axis=1)

    if 0 < len(names) < 21:  # display per-class legend if < 21 classes
        for i, y in enumerate(py.T):
            ax.plot(px, y, linewidth=1, label=f'{names[i]} {ap[i, 0]:.3f}')  # plot(recall, precision)
    else:
        ax.plot(px, py, linewidth=1, color='grey')  # plot(recall, precision)

    ax.plot(px, py.mean(1), linewidth=3, color='blue', label='all classes %.3f mAP@0.5' % ap[:, 0].mean())
    ax.set_xlabel('Recall')
    ax.set_ylabel('Precision')
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    plt.legend(bbox_to_anchor=(1.04, 1), loc="upper left")
    fig.savefig(save_dir, dpi=250)
    plt.close()


def plot_mc_curve(px, py, save_dir='mc_curve.png', names=(), xlabel='Confidence', ylabel='Metric'):
    """
    Plots metric-confidence curve
    
    Args:
        px: x-values for plotting
        py: y-values for plotting
        save_dir: directory to save plot
        names: class names
        xlabel: x-axis label
        ylabel: y-axis label
    """
    fig, ax = plt.subplots(1, 1, figsize=(9, 6), tight_layout=True)

    if 0 < len(names) < 21:  # display per-class legend if < 21 classes
        for i, y in enumerate(py):
            ax.plot(px, y, linewidth=1, label=f'{names[i]}')  # plot(confidence, metric)
    else:
        ax.plot(px, py.mean(0), linewidth=2, color='grey')  # plot(confidence, metric)

    y = py.mean(0)
    ax.plot(px, y, linewidth=3, color='blue', label=f'all classes {y.max():.2f} at {px[y.argmax()]:.3f}')
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    plt.legend(bbox_to_anchor=(1.04, 1), loc="upper left")
    fig.savefig(save_dir, dpi=250)
    plt.close()


def plot_training_results(results, save_dir='results.png', best_fitness=None):
    """
    Plot training results
    
    Args:
        results: dictionary of training results
        save_dir: directory to save plot
        best_fitness: best fitness value
    """
    fig, ax = plt.subplots(2, 5, figsize=(15, 8), tight_layout=True)
    ax = ax.ravel()
    s = ['Box', 'Objectness', 'Classification', 'Precision', 'Recall',
         'val Box', 'val Objectness', 'val Classification', 'mAP@0.5', 'mAP@0.5:0.95']
    
    for i in range(10):
        if i < len(results):
            ax[i].plot(results[i], '.-', label=s[i])
            
    j = 0
    for key in ['box_loss', 'obj_loss', 'cls_loss']:  # plot loss
        if key in results:
            ax[j].set_title(s[j])
            ax[j].plot(results[key], '.-')
            j += 1
    
    j = 3
    for key in ['precision', 'recall']:  # plot precision recall
        if key in results:
            ax[j].set_title(s[j])
            ax[j].plot(results[key], '.-')
            j += 1
    
    j = 5
    for key in ['val_box_loss', 'val_obj_loss', 'val_cls_loss']:  # plot val loss
        if key in results:
            ax[j].set_title(s[j])
            ax[j].plot(results[key], '.-')
            j += 1
    
    j = 8
    for key in ['mAP_0.5', 'mAP_0.5:0.95']:  # plot mAP
        if key in results:
            ax[j].set_title(s[j])
            ax[j].plot(results[key], '.-')
            j += 1
        
    # Add title with training info if available
    if 'epoch' in results:
        fig.suptitle(f"Training Results (Epoch {len(results['epoch'])})", fontsize=16)
    
    # Mark best fitness
    if best_fitness:
        for i, y in enumerate(results['mAP_0.5']):
            if y == best_fitness:
                ax[8].plot(i, y, 'o', color='r', label=f'Best mAP: {y:.4f}')
                ax[8].legend()
                break
    
    # Save figure    
    plt.savefig(save_dir, dpi=300)
    plt.close()
    return


def plot_confusion_matrix(confusion_matrix, names, save_dir='confusion_matrix.png'):
    """
    Plot confusion matrix
    
    Args:
        confusion_matrix: confusion matrix
        names: class names
        save_dir: directory to save plot
    """
    try:
        import seaborn as sns
        
        # Normalize matrix
        array = confusion_matrix / (confusion_matrix.sum(1).reshape(-1, 1) + 1E-9)  # normalize
        
        # Create figure
        fig = plt.figure(figsize=(12, 10), tight_layout=True)
        
        # Use seaborn for better visualization
        sns.set(font_scale=1.0 if len(names) < 50 else 0.8)  # reduce font size if many classes
        sns.heatmap(array, annot=len(names) < 30, annot_kws={"size": 8}, cmap='Blues', fmt='.2f', square=True,
                    xticklabels=names, yticklabels=names)
        
        # Add labels
        fig.axes[0].set_xlabel('Predicted')
        fig.axes[0].set_ylabel('True')
        
        # Save figure
        plt.savefig(save_dir, dpi=250)
        plt.close()
        return
    
    except Exception as e:
        print(f'WARNING: confusion matrix plot failure: {e}')


def plot_labels(labels, names=(), save_dir='labels.jpg'):
    """
    Plot dataset labels
    
    Args:
        labels: labels
        names: class names
        save_dir: directory to save plot
    """
    c, b = labels[:, 0], labels[:, 1:].transpose()  # classes, boxes
    nc = int(c.max() + 1)  # number of classes
    
    # Create figure
    fig, ax = plt.subplots(2, 2, figsize=(8, 8), tight_layout=True)
    
    # Plot class histogram
    ax[0, 0].hist(c, bins=np.linspace(0, nc, nc + 1) - 0.5, rwidth=0.8)
    ax[0, 0].set_xlabel('Classes')
    ax[0, 0].grid(True)
    
    # Plot box aspect ratio
    ax[0, 1].scatter(b[0], b[3], c=c, cmap='tab20b', alpha=0.5)
    ax[0, 1].set_xlabel('Width')
    ax[0, 1].set_ylabel('Height')
    ax[0, 1].grid(True)
    
    # Plot box dimensions
    ax[1, 0].scatter(b[2], b[3], c=c, cmap='viridis', alpha=0.5)
    ax[1, 0].set_xlabel('Width')
    ax[1, 0].set_ylabel('Height')
    ax[1, 0].set_title('Dimensions')
    ax[1, 0].grid(True)
    
    # Plot box position
    ax[1, 1].scatter(b[0], b[1], c=c, cmap='jet', alpha=0.5)
    ax[1, 1].set_xlabel('X')
    ax[1, 1].set_ylabel('Y')
    ax[1, 1].set_title('Position')
    ax[1, 1].grid(True)
    
    # Save figure
    plt.savefig(save_dir, dpi=200)
    plt.close()
    
    # Add class names if available
    if len(names):
        for i, name in enumerate(names):
            print(f'Class {i}: {name}')


def plot_evolution(yaml_file='hyp.yaml', save_dir='evolution.png'):
    """
    Plot hyperparameter evolution results
    
    Args:
        yaml_file: hyperparameter file
        save_dir: directory to save plot
    """
    import yaml
    from scipy.stats import gaussian_kde
    
    with open(yaml_file) as f:
        hyp = yaml.safe_load(f)
    x = np.loadtxt('evolve.txt', ndmin=2)
    f = fitness(x)
    # weights = (f - f.min()) ** 2  # for weighted results
    fig, ax = plt.subplots(2, 5, figsize=(12, 6), tight_layout=True)
    ax = ax.ravel()
    for i, k in enumerate(hyp.keys()):
        if i < 10:
            y = x[:, i+1]
            mu = y.mean()
            sigma = y.std()
            ax[i].hist(y, 25, density=True, histtype='step', color='k')
            
            # Fit normal distribution and overlay
            try:
                density = gaussian_kde(y)
                xs = np.linspace(min(y), max(y), 1000)
                if i == 0:  # plot Gaussian in first subplot
                    ax[0].plot(xs, density(xs), color='b')
                    ax[0].plot(xs, (1 / (sigma * np.sqrt(2 * np.pi))) * np.exp(-((xs - mu) ** 2) / (2 * sigma ** 2)), 'r--')
            except:
                pass
            
            ax[i].set_title(k)
    
    # Add labels and save
    plt.savefig(save_dir, dpi=200)
    plt.close()
    print(f'Saved {save_dir}')


def feature_visualization(x, module_type, stage, n=32, save_dir='features'):
    """
    Visualize network features
    
    Args:
        x: features to visualize
        module_type: module type
        stage: module stage
        n: number of features to visualize
        save_dir: directory to save visualizations
    """
    # Create directory
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    
    # Convert to numpy and normalize
    x = x.cpu().numpy()
    if x.shape[1] > n:  # too many channels
        x = x[:, :n]  # select only first n channels
    
    for i in range(min(n, x.shape[1])):
        fig, ax = plt.subplots(1, 1, figsize=(6, 6), tight_layout=True)
        ax.imshow(x[0, i])  # feature i
        ax.axis('off')
        ax.set_title(f"{module_type} {stage} - Channel {i}")
        plt.savefig(str(save_dir / f"{module_type}_{stage}_ch{i}.png"), dpi=200)
        plt.close()


def fitness(x):
    """
    Hyperparameter fitness function
    
    Args:
        x: hyperparameters
        
    Returns:
        fitness value
    """
    # Model fitness as a weighted combination of metrics
    w = [0.0, 0.0, 0.1, 0.9]  # weights for [P, R, mAP@0.5, mAP@0.5:0.95]
    return (x[:, :4] * w).sum(1)


def xywh2xyxy(x):
    """
    Convert nx4 boxes from [x, y, w, h] to [x1, y1, x2, y2]
    
    Args:
        x: boxes in [x, y, w, h] format
        
    Returns:
        boxes in [x1, y1, x2, y2] format
    """
    if isinstance(x, torch.Tensor):
        y = x.clone()
        y[:, 0] = x[:, 0] - x[:, 2] / 2  # top left x
        y[:, 1] = x[:, 1] - x[:, 3] / 2  # top left y
        y[:, 2] = x[:, 0] + x[:, 2] / 2  # bottom right x
        y[:, 3] = x[:, 1] + x[:, 3] / 2  # bottom right y
        return y
    else:  # numpy
        y = np.copy(x)
        y[:, 0] = x[:, 0] - x[:, 2] / 2  # top left x
        y[:, 1] = x[:, 1] - x[:, 3] / 2  # top left y
        y[:, 2] = x[:, 0] + x[:, 2] / 2  # bottom right x
        y[:, 3] = x[:, 1] + x[:, 3] / 2  # bottom right y
        return y 