"""
Advanced data augmentation techniques for object detection training
Based on Ultralytics YOLOv11 augmentation pipeline
"""

import math
import random
import numpy as np
import torch
import torch.nn.functional as F
import torchvision.transforms as T
import torchvision.transforms.functional as TF
import cv2
from PIL import Image
from typing import List, Tuple, Dict, Optional, Union


def letterbox(img, new_shape=(640, 640), color=(114, 114, 114), 
              auto=True, scale_fill=False, scaleup=True, stride=32):
    """
    Resize and pad image while meeting stride-multiple constraints
    Args:
        img: Image to resize and pad
        new_shape: Target shape (h, w)
        color: Color for padding
        auto: Auto compute minimum ratio
        scale_fill: Stretch to new_shape (no padding)
        scaleup: Allow scale up (oversample)
        stride: Stride for network compatibility
    Returns:
        Resized and padded image
        Ratio (w_new / w, h_new / h)
        Padding (dw, dh)
    """
    # Convert PIL to numpy array if needed
    if isinstance(img, Image.Image):
        img = np.array(img)
        
    shape = img.shape[:2]  # current shape [height, width]
    
    if isinstance(new_shape, int):
        new_shape = (new_shape, new_shape)

    # Scale ratio (new / old)
    r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
    if not scaleup:  # only scale down, do not scale up (for better val mAP)
        r = min(r, 1.0)

    # Compute padding
    ratio = r, r  # width, height ratios
    new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
    dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]  # wh padding
    
    if auto:  # minimum rectangle
        dw, dh = np.mod(dw, stride), np.mod(dh, stride)  # wh padding
    
    elif scale_fill:  # stretch
        dw, dh = 0.0, 0.0
        new_unpad = (new_shape[1], new_shape[0])
        ratio = new_shape[1] / shape[1], new_shape[0] / shape[0]  # width, height ratios

    dw /= 2  # divide padding into 2 sides
    dh /= 2

    if shape[::-1] != new_unpad:  # resize
        img = cv2.resize(img, new_unpad, interpolation=cv2.INTER_LINEAR)
        
    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    
    img = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)
    
    return img, ratio, (dw, dh)


def random_perspective(img, targets=(), degrees=10, translate=0.1, scale=0.1, shear=10, 
                       perspective=0.0, border=(0, 0)):
    """
    Random perspective transform for image and bounding boxes
    Args:
        img: Image to transform
        targets: Bounding boxes in format [cls, xyxy]
        degrees: Rotation degrees
        translate: Translation fraction
        scale: Scale fraction
        shear: Shear degrees
        perspective: Perspective distortion factor
        border: Border to be excluded from cropping
    Returns:
        Transformed image
        Transformed targets
    """
    height = img.shape[0] + border[0] * 2  # shape(h,w,c)
    width = img.shape[1] + border[1] * 2

    # Center
    C = np.eye(3)
    C[0, 2] = -img.shape[1] / 2  # x translation (pixels)
    C[1, 2] = -img.shape[0] / 2  # y translation (pixels)

    # Perspective
    P = np.eye(3)
    P[2, 0] = random.uniform(-perspective, perspective)  # x perspective (about y)
    P[2, 1] = random.uniform(-perspective, perspective)  # y perspective (about x)

    # Rotation and Scale
    R = np.eye(3)
    a = random.uniform(-degrees, degrees)
    s = random.uniform(1 - scale, 1 + scale)
    R[:2] = cv2.getRotationMatrix2D(angle=a, center=(0, 0), scale=s)

    # Shear
    S = np.eye(3)
    S[0, 1] = math.tan(random.uniform(-shear, shear) * math.pi / 180)  # x shear (deg)
    S[1, 0] = math.tan(random.uniform(-shear, shear) * math.pi / 180)  # y shear (deg)

    # Translation
    T = np.eye(3)
    T[0, 2] = random.uniform(0.5 - translate, 0.5 + translate) * width  # x translation (pixels)
    T[1, 2] = random.uniform(0.5 - translate, 0.5 + translate) * height  # y translation (pixels)

    # Combined rotation matrix
    M = T @ S @ R @ P @ C  # order of operations (right to left) is IMPORTANT
    if (border[0] != 0) or (border[1] != 0) or (M != np.eye(3)).any():  # image changed
        if perspective:
            img = cv2.warpPerspective(img, M, dsize=(width, height), borderValue=(114, 114, 114))
        else:  # affine
            img = cv2.warpAffine(img, M[:2], dsize=(width, height), borderValue=(114, 114, 114))

    # Transform labels
    n = len(targets)
    if n:
        xy = np.ones((n * 4, 3))
        xy[:, :2] = targets[:, [1, 2, 3, 4, 1, 4, 3, 2]].reshape(n * 4, 2)  # x1y1, x2y2, x1y2, x2y1
        xy = xy @ M.T  # transform
        if perspective:
            xy = (xy[:, :2] / xy[:, 2:3]).reshape(n, 8)  # rescale
        else:  # affine
            xy = xy[:, :2].reshape(n, 8)

        # Create new boxes
        x = xy[:, [0, 2, 4, 6]]
        y = xy[:, [1, 3, 5, 7]]
        new = np.concatenate((x.min(1), y.min(1), x.max(1), y.max(1))).reshape(4, n).T

        # Clip
        new[:, [0, 2]] = new[:, [0, 2]].clip(0, width)
        new[:, [1, 3]] = new[:, [1, 3]].clip(0, height)

        # Filter candidates
        i = box_candidates(box1=targets[:, 1:5].T * s, box2=new.T, area_thr=0.01)
        targets = targets[i]
        targets[:, 1:5] = new[i]

    return img, targets


def box_candidates(box1, box2, wh_thr=2, ar_thr=20, area_thr=0.1, eps=1e-16):
    """
    Filter bounding boxes by "candidateness"
    Args:
        box1: Prior boxes with format [x1, y1, x2, y2]
        box2: New boxes with format [x1, y1, x2, y2]
        wh_thr: Width and height threshold
        ar_thr: Aspect ratio threshold
        area_thr: Area threshold
        eps: Small epsilon value
    Returns:
        Boolean mask of valid candidates
    """
    w1, h1 = box1[2] - box1[0], box1[3] - box1[1]
    w2, h2 = box2[2] - box2[0], box2[3] - box2[1]
    ar = np.maximum(w2 / (h2 + eps), h2 / (w2 + eps))  # aspect ratio
    return (w2 > wh_thr) & (h2 > wh_thr) & (w2 * h2 / (w1 * h1 + eps) > area_thr) & (ar < ar_thr)


def augment_hsv(img, h_gain=0.015, s_gain=0.7, v_gain=0.4):
    """
    Augment image by HSV color changes
    Args:
        img: Image to augment
        h_gain: Hue gain
        s_gain: Saturation gain
        v_gain: Value gain
    Returns:
        Augmented image
    """
    r = np.random.uniform(-1, 1, 3) * [h_gain, s_gain, v_gain] + 1  # random gains
    hue, sat, val = cv2.split(cv2.cvtColor(img, cv2.COLOR_BGR2HSV))
    dtype = img.dtype  # uint8

    x = np.arange(0, 256, dtype=np.int16)
    lut_hue = ((x * r[0]) % 180).astype(dtype)
    lut_sat = np.clip(x * r[1], 0, 255).astype(dtype)
    lut_val = np.clip(x * r[2], 0, 255).astype(dtype)

    img_hsv = cv2.merge((cv2.LUT(hue, lut_hue), cv2.LUT(sat, lut_sat), cv2.LUT(val, lut_val))).astype(dtype)
    cv2.cvtColor(img_hsv, cv2.COLOR_HSV2BGR, dst=img)  # no return needed


def cutout(img, labels=None, p=0.5, scale=0.5, ratio=1.0, fill_value=0):
    """
    Cutout augmentation
    Args:
        img: Image to augment
        labels: Labels in xyxy format
        p: Probability of applying cutout
        scale: Scale factor for cutout size
        ratio: Width/height ratio of cutout
    Returns:
        Augmented image
        Augmented labels
    """
    if random.random() >= p:
        return img, labels
        
    h, w = img.shape[:2]
    
    # Find random cutout size and position
    s = min(h, w) * scale
    r = ratio  # aspect ratio
    cutout_w = int(s * r)
    cutout_h = int(s / r)
    
    cx = random.randint(0, w)
    cy = random.randint(0, h)
    
    # Calculate cutout area
    xmin = max(0, cx - cutout_w // 2)
    ymin = max(0, cy - cutout_h // 2)
    xmax = min(w, cx + cutout_w // 2)
    ymax = min(h, cy + cutout_h // 2)
    
    # Apply cutout
    img[ymin:ymax, xmin:xmax] = fill_value
    
    if labels is not None:
        # Check if cutout overlaps with any labels
        for i, label in enumerate(labels):
            box = label[1:5]  # xyxy
            iou = calculate_iou([xmin, ymin, xmax, ymax], box)
            if iou > 0.8:  # If cutout removes most of the object, remove the label
                labels[i, 0] = -1  # Mark for removal
        
        # Filter out removed labels
        labels = labels[labels[:, 0] >= 0]
            
    return img, labels


def calculate_iou(box1, box2):
    """
    Calculate IoU between two boxes
    Args:
        box1: First box in xyxy format
        box2: Second box in xyxy format
    Returns:
        IoU value
    """
    # Calculate intersection coordinates
    xmin = max(box1[0], box2[0])
    ymin = max(box1[1], box2[1])
    xmax = min(box1[2], box2[2])
    ymax = min(box1[3], box2[3])
    
    # Calculate areas
    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
    
    # Calculate intersection and union areas
    inter_area = max(0, xmax - xmin) * max(0, ymax - ymin)
    union_area = area1 + area2 - inter_area
    
    # Calculate IoU
    return inter_area / union_area if union_area > 0 else 0


def mixup(img1, labels1, img2, labels2, alpha=0.5):
    """
    MixUp augmentation
    Args:
        img1: First image
        labels1: First image labels
        img2: Second image
        labels2: Second image labels
        alpha: Mixup factor
    Returns:
        Mixed image
        Combined labels
    """
    # Use a fixed lambda value for simplicity
    lam = np.random.beta(alpha, alpha)
    
    # Combine images
    img = lam * img1 + (1 - lam) * img2
    img = img.astype(np.uint8)
    
    # Combine labels with a weight column
    if len(labels1) > 0 and len(labels2) > 0:
        labels1 = np.concatenate([labels1, np.ones((len(labels1), 1)) * lam], axis=1)
        labels2 = np.concatenate([labels2, np.ones((len(labels2), 1)) * (1 - lam)], axis=1)
        labels = np.concatenate([labels1, labels2], axis=0)
    elif len(labels1) > 0:
        labels = np.concatenate([labels1, np.ones((len(labels1), 1)) * lam], axis=1)
    elif len(labels2) > 0:
        labels = np.concatenate([labels2, np.ones((len(labels2), 1)) * (1 - lam)], axis=1)
    else:
        labels = np.empty((0, 6))  # Empty labels with weight column
        
    return img, labels


def mosaic_augmentation(img_size, imgs, labels, p=1.0):
    """
    Create a mosaic image with 4 input images
    Args:
        img_size: Final image size
        imgs: List of 4 input images
        labels: List of 4 input labels
        p: Probability of applying mosaic augmentation
    Returns:
        Mosaic image
        Combined labels
    """
    if random.random() >= p or len(imgs) < 4:
        return imgs[0], labels[0]
        
    h, w = img_size, img_size
    
    # Center coordinates
    cx, cy = w // 2, h // 2
    
    # Create mosaic output image
    mosaic_img = np.full((h, w, 3), 114, dtype=np.uint8)
    
    # Output labels
    mosaic_labels = []
    
    # Process each of the 4 images
    for i, (img, img_labels) in enumerate(zip(imgs[:4], labels[:4])):
        # Calculate placement
        if i == 0:  # top left
            x1a, y1a, x2a, y2a = 0, 0, cx, cy
            x1b, y1b, x2b, y2b = w - cx, h - cy, w, h
        elif i == 1:  # top right
            x1a, y1a, x2a, y2a = cx, 0, w, cy
            x1b, y1b, x2b, y2b = 0, h - cy, cx, h
        elif i == 2:  # bottom left
            x1a, y1a, x2a, y2a = 0, cy, cx, h
            x1b, y1b, x2b, y2b = w - cx, 0, w, cy
        else:  # bottom right
            x1a, y1a, x2a, y2a = cx, cy, w, h
            x1b, y1b, x2b, y2b = 0, 0, cx, cy
            
        # Get input image dimensions
        img_h, img_w = img.shape[:2]
        
        # Place this image in the mosaic
        mosaic_img[y1a:y2a, x1a:x2a] = img[y1b:y2b, x1b:x2b]
        
        # Adjust labels based on placement
        if len(img_labels) > 0:
            # Convert from [class, x, y, w, h] to [class, x1, y1, x2, y2]
            if img_labels.shape[1] == 5:  # [class, x, y, w, h]
                labels_xyxy = img_labels.copy()
                labels_xyxy[:, 1] = img_labels[:, 1] - img_labels[:, 3] / 2  # x1
                labels_xyxy[:, 2] = img_labels[:, 2] - img_labels[:, 4] / 2  # y1
                labels_xyxy[:, 3] = img_labels[:, 1] + img_labels[:, 3] / 2  # x2
                labels_xyxy[:, 4] = img_labels[:, 2] + img_labels[:, 4] / 2  # y2
            else:
                labels_xyxy = img_labels.copy()
                
            # Scale to image size
            labels_xyxy[:, 1] *= img_w
            labels_xyxy[:, 2] *= img_h
            labels_xyxy[:, 3] *= img_w
            labels_xyxy[:, 4] *= img_h
            
            # Adjust coordinates for placement in mosaic
            labels_xyxy[:, [1, 3]] = labels_xyxy[:, [1, 3]] - x1b + x1a
            labels_xyxy[:, [2, 4]] = labels_xyxy[:, [2, 4]] - y1b + y1a
            
            # Filter out labels that are outside the mosaic
            valid_indices = ((labels_xyxy[:, 1] < x2a) & 
                             (labels_xyxy[:, 3] > x1a) & 
                             (labels_xyxy[:, 2] < y2a) & 
                             (labels_xyxy[:, 4] > y1a))
            
            valid_labels = labels_xyxy[valid_indices]
            
            # Clip coordinates to mosaic area
            valid_labels[:, 1] = np.clip(valid_labels[:, 1], x1a, x2a)
            valid_labels[:, 2] = np.clip(valid_labels[:, 2], y1a, y2a)
            valid_labels[:, 3] = np.clip(valid_labels[:, 3], x1a, x2a)
            valid_labels[:, 4] = np.clip(valid_labels[:, 4], y1a, y2a)
            
            # Convert back to normalized [class, x, y, w, h]
            valid_labels_xywh = valid_labels.copy()
            valid_labels_xywh[:, 1] = ((valid_labels[:, 1] + valid_labels[:, 3]) / 2) / w  # x center
            valid_labels_xywh[:, 2] = ((valid_labels[:, 2] + valid_labels[:, 4]) / 2) / h  # y center
            valid_labels_xywh[:, 3] = (valid_labels[:, 3] - valid_labels[:, 1]) / w  # width
            valid_labels_xywh[:, 4] = (valid_labels[:, 4] - valid_labels[:, 2]) / h  # height
            
            mosaic_labels.append(valid_labels_xywh)
            
    if len(mosaic_labels) > 0:
        mosaic_labels = np.concatenate(mosaic_labels, axis=0)
        
    # Apply random perspective to the mosaic
    mosaic_img, mosaic_labels = random_perspective(
        mosaic_img, 
        mosaic_labels, 
        degrees=0, 
        translate=0.1, 
        scale=0.5, 
        shear=0, 
        perspective=0.0, 
        border=(0, 0)
    )
        
    return mosaic_img, mosaic_labels


class TrainTransform:
    """
    Advanced transform pipeline for training data
    """
    def __init__(self, img_size=640, degrees=0.0, translate=0.2, scale=0.5, shear=0.0, 
                 perspective=0.0, hsv_h=0.015, hsv_s=0.7, hsv_v=0.4, 
                 flipud=0.0, fliplr=0.5, mosaic_prob=1.0, mixup_prob=0.0, cutout_prob=0.0):
        self.img_size = img_size
        self.degrees = degrees
        self.translate = translate
        self.scale = scale
        self.shear = shear
        self.perspective = perspective
        self.hsv_h = hsv_h
        self.hsv_s = hsv_s
        self.hsv_v = hsv_v
        self.flipud = flipud
        self.fliplr = fliplr
        self.mosaic_prob = mosaic_prob
        self.mixup_prob = mixup_prob
        self.cutout_prob = cutout_prob
        
    def __call__(self, img, labels=None, mosaic_imgs=None, mosaic_labels=None):
        """
        Apply transforms to image and labels
        Args:
            img: Input image (numpy array)
            labels: Labels [class, x, y, w, h]
            mosaic_imgs: Optional list of images for mosaic augmentation
            mosaic_labels: Optional list of labels for mosaic augmentation
        Returns:
            Transformed image (tensor)
            Transformed labels
        """
        if mosaic_imgs is not None and random.random() < self.mosaic_prob:
            # Apply mosaic augmentation
            img, labels = mosaic_augmentation(self.img_size, mosaic_imgs, mosaic_labels, p=1.0)
        
        # Resize and pad image to target size
        img, ratio, pad = letterbox(img, new_shape=self.img_size)
            
        # Convert format if necessary
        if labels is not None and len(labels) > 0:
            # Adjust boxes for resizing and padding
            if ratio[0] != 1 or ratio[1] != 1:
                if labels.shape[1] == 5:  # [class, x, y, w, h]
                    # Adjust center coordinates
                    labels[:, 1] = labels[:, 1] * ratio[0]
                    labels[:, 2] = labels[:, 2] * ratio[1]
                    # Adjust width and height
                    labels[:, 3] = labels[:, 3] * ratio[0]
                    labels[:, 4] = labels[:, 4] * ratio[1]
            
        # Apply HSV augmentation
        augment_hsv(img, h_gain=self.hsv_h, s_gain=self.hsv_s, v_gain=self.hsv_v)
        
        # Apply random perspective
        img, labels = random_perspective(
            img, labels, 
            degrees=self.degrees,
            translate=self.translate,
            scale=self.scale,
            shear=self.shear,
            perspective=self.perspective
        )
        
        # Random flips
        if random.random() < self.flipud:
            img = np.flipud(img)
            if labels is not None and len(labels) > 0:
                labels[:, 2] = 1 - labels[:, 2]  # Flip y coordinates
                
        if random.random() < self.fliplr:
            img = np.fliplr(img)
            if labels is not None and len(labels) > 0:
                labels[:, 1] = 1 - labels[:, 1]  # Flip x coordinates
        
        # Apply cutout augmentation
        img, labels = cutout(img, labels, p=self.cutout_prob)
        
        # Convert to tensor
        img = torch.from_numpy(img).float()
        img = img.permute(2, 0, 1)  # HWC to CHW
        
        # Normalize
        img = img / 255.0
            
        return img, labels


class ValTransform:
    """
    Simple transform pipeline for validation data
    """
    def __init__(self, img_size=640):
        self.img_size = img_size
        
    def __call__(self, img, labels=None):
        """
        Apply validation transforms to image and labels
        Args:
            img: Input image (numpy array)
            labels: Labels [class, x, y, w, h]
        Returns:
            Transformed image (tensor)
            Transformed labels
        """
        # Resize and pad image to target size
        img, ratio, pad = letterbox(img, new_shape=self.img_size, auto=False)
            
        # Convert format if necessary
        if labels is not None and len(labels) > 0:
            # Adjust boxes for resizing and padding
            if ratio[0] != 1 or ratio[1] != 1:
                if labels.shape[1] == 5:  # [class, x, y, w, h]
                    # Adjust center coordinates
                    labels[:, 1] = labels[:, 1] * ratio[0]
                    labels[:, 2] = labels[:, 2] * ratio[1]
                    # Adjust width and height
                    labels[:, 3] = labels[:, 3] * ratio[0]
                    labels[:, 4] = labels[:, 4] * ratio[1]
        
        # Convert to tensor
        img = torch.from_numpy(img).float()
        img = img.permute(2, 0, 1)  # HWC to CHW
        
        # Normalize
        img = img / 255.0
            
        return img, labels


def get_train_transforms(img_size=640, degrees=0.0, translate=0.2, scale=0.5, shear=0.0, 
                          perspective=0.0, hsv_h=0.015, hsv_s=0.7, hsv_v=0.4, 
                          flipud=0.0, fliplr=0.5, mosaic_prob=1.0, mixup_prob=0.0, 
                          cutout_prob=0.0):
    """
    Create training transforms
    Args:
        img_size: Target image size
        degrees: Rotation degrees
        translate: Translation fraction
        scale: Scale fraction
        shear: Shear degrees
        perspective: Perspective distortion factor
        hsv_h: HSV hue gain
        hsv_s: HSV saturation gain
        hsv_v: HSV value gain
        flipud: Vertical flip probability
        fliplr: Horizontal flip probability
        mosaic_prob: Mosaic augmentation probability
        mixup_prob: Mixup augmentation probability
        cutout_prob: Cutout augmentation probability
    Returns:
        TrainTransform object
    """
    return TrainTransform(
        img_size=img_size,
        degrees=degrees,
        translate=translate, 
        scale=scale,
        shear=shear,
        perspective=perspective,
        hsv_h=hsv_h,
        hsv_s=hsv_s,
        hsv_v=hsv_v,
        flipud=flipud,
        fliplr=fliplr,
        mosaic_prob=mosaic_prob,
        mixup_prob=mixup_prob,
        cutout_prob=cutout_prob
    )


def get_val_transforms(img_size=640):
    """
    Create validation transforms
    Args:
        img_size: Target image size
    Returns:
        ValTransform object
    """
    return ValTransform(img_size=img_size) 