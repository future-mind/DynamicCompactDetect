import os
import torch
import numpy as np
import cv2
from torch.utils.data import Dataset
from pycocotools.coco import COCO
import albumentations as A
from pathlib import Path


class COCODataset(Dataset):
    """
    Dataset class for MS COCO object detection dataset.
    """
    def __init__(self, path, img_size=640, augment=False, rect=False, batch_size=16):
        """
        Initialize COCO dataset.
        
        Args:
            path: Path to COCO dataset
            img_size: Input image size
            augment: Whether to apply data augmentation
            rect: Whether to use rectangular training (batch shape = [h, w])
            batch_size: Batch size for calculating rectangular shapes
        """
        self.path = Path(path)
        self.img_size = img_size
        self.augment = augment
        self.rect = rect
        self.batch_size = batch_size
        
        # The annotations directory is at the root of the COCO dataset, not in the image directories
        # Determine the root path (parent of the train path)
        root_dir = self.path.parent
        
        # Load COCO annotations
        annotations_path = root_dir / 'annotations' / 'instances_train2017.json'
        if self.path.name == 'val2017':
            annotations_path = root_dir / 'annotations' / 'instances_val2017.json'
        
        print(f"Loading COCO annotations from {annotations_path}")
        self.coco = COCO(annotations_path)
        self.ids = list(self.coco.imgs.keys())
        
        # Filter out images without annotations
        valid_ids = []
        for img_id in self.ids:
            ann_ids = self.coco.getAnnIds(imgIds=img_id)
            if len(ann_ids) > 0:
                valid_ids.append(img_id)
        self.ids = valid_ids
        
        # Calculate image shapes for rectangular training
        if self.rect:
            self.shapes = self._get_shapes()
            self.batch_shapes = self._get_batch_shapes()
        
        # Setup data augmentation
        self.transform = self._get_transform() if augment else None
    
    def __len__(self):
        """Return the number of images in the dataset."""
        return len(self.ids)
    
    def __getitem__(self, index):
        """
        Get image and annotations for the given index.
        
        Args:
            index: Index of the image
            
        Returns:
            tuple: (image, targets, image_id)
        """
        # Load image
        img_id = self.ids[index]
        img_info = self.coco.loadImgs(img_id)[0]
        # Images are directly in the train2017/val2017 directory, not in an 'images' subdirectory
        img_path = self.path / img_info['file_name']
        img = cv2.imread(str(img_path))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        # Get original image shape
        h, w = img.shape[:2]
        
        # Load annotations
        ann_ids = self.coco.getAnnIds(imgIds=img_id)
        anns = self.coco.loadAnns(ann_ids)
        
        # Extract bounding boxes and class labels
        boxes = []
        labels = []
        
        for ann in anns:
            if ann['bbox'][2] < 1 or ann['bbox'][3] < 1:
                continue
            
            # Convert COCO bbox [x, y, w, h] to [x1, y1, x2, y2]
            x1, y1, w_box, h_box = ann['bbox']
            x2, y2 = x1 + w_box, y1 + h_box
            
            # Normalize coordinates
            x1, y1, x2, y2 = x1 / w, y1 / h, x2 / w, y2 / h
            
            boxes.append([x1, y1, x2, y2])
            labels.append(ann['category_id'])
        
        # Convert to tensors
        boxes = torch.tensor(boxes, dtype=torch.float32)
        labels = torch.tensor(labels, dtype=torch.int64)
        
        # Create targets tensor [x1, y1, x2, y2, class_id]
        if len(boxes) > 0:
            targets = torch.cat([boxes, labels.unsqueeze(1)], dim=1)
        else:
            targets = torch.zeros((0, 5), dtype=torch.float32)
        
        # Apply data augmentation
        if self.transform and len(boxes) > 0:
            # Convert to albumentations format
            boxes_albu = boxes.clone()
            boxes_albu[:, 0] *= w  # x1
            boxes_albu[:, 1] *= h  # y1
            boxes_albu[:, 2] *= w  # x2
            boxes_albu[:, 3] *= h  # y2
            
            # Apply augmentations
            transformed = self.transform(
                image=img,
                bboxes=boxes_albu.tolist(),
                labels=labels.tolist()
            )
            
            img = transformed['image']
            
            if len(transformed['bboxes']) > 0:
                boxes_t = torch.tensor(transformed['bboxes'], dtype=torch.float32)
                labels_t = torch.tensor(transformed['labels'], dtype=torch.int64)
                
                # Normalize coordinates
                h_t, w_t = img.shape[:2]
                boxes_t[:, [0, 2]] /= w_t  # x1, x2
                boxes_t[:, [1, 3]] /= h_t  # y1, y2
                
                targets = torch.cat([boxes_t, labels_t.unsqueeze(1)], dim=1)
            else:
                targets = torch.zeros((0, 5), dtype=torch.float32)
        
        # Resize image
        if self.rect:
            # Get batch shape for this image
            batch_idx = index // self.batch_size
            batch_idx = min(batch_idx, len(self.batch_shapes) - 1)  # Ensure we don't go out of bounds
            target_h, target_w = self.batch_shapes[batch_idx]
        else:
            target_h, target_w = self.img_size, self.img_size
        
        # Resize image
        img = cv2.resize(img, (target_w, target_h))
        
        # Convert image to tensor and normalize
        img = torch.from_numpy(img).permute(2, 0, 1).float() / 255.0
        
        return img, targets, img_id
    
    def collate_fn(self, batch):
        """
        Custom collate function for batching.
        
        Args:
            batch: List of (image, targets, image_id) tuples
            
        Returns:
            tuple: (images, targets, image_ids)
        """
        imgs, targets, img_ids = zip(*batch)
        
        # Stack images
        imgs = torch.stack(imgs)
        
        # Concatenate targets with batch index
        for i, target in enumerate(targets):
            if len(target) > 0:
                target = torch.cat([torch.ones((len(target), 1), dtype=torch.float32) * i, target], dim=1)
            else:
                target = torch.zeros((0, 6), dtype=torch.float32)
            
            if i == 0:
                all_targets = target
            else:
                all_targets = torch.cat([all_targets, target], dim=0)
        
        if len(all_targets) == 0:
            all_targets = torch.zeros((0, 6), dtype=torch.float32)
        
        return imgs, all_targets, img_ids
    
    def _get_shapes(self):
        """Calculate image shapes for rectangular training."""
        shapes = []
        for img_id in self.ids:
            img_info = self.coco.loadImgs(img_id)[0]
            shapes.append((img_info['height'], img_info['width']))
        return np.array(shapes)
    
    def _get_batch_shapes(self):
        """Calculate batch shapes for rectangular training."""
        n = len(self.shapes)
        batch_shapes = np.zeros((n // self.batch_size + (n % self.batch_size > 0), 2), dtype=np.int64)
        
        # Sort by aspect ratio
        s = self.shapes  # wh
        ar = s[:, 1] / s[:, 0]  # aspect ratio
        irect = ar.argsort()
        self.ids = [self.ids[i] for i in irect]  # sort ids by aspect ratio
        s = s[irect]  # sort shapes by aspect ratio
        
        # Compute batch shapes
        for i in range(0, n, self.batch_size):
            batch_ids = range(i, min(i + self.batch_size, n))
            batch = s[batch_ids]
            
            # Compute max height and width for this batch
            max_height = batch[:, 0].max()
            max_width = batch[:, 1].max()
            
            # Scale to target_size while maintaining aspect ratio
            ratio = min(self.img_size / max_height, self.img_size / max_width)
            target_h = int(max_height * ratio)
            target_w = int(max_width * ratio)
            
            # Make divisible by 32
            target_h = (target_h // 32) * 32
            target_w = (target_w // 32) * 32
            
            # Make sure we don't go below 32x32
            target_h = max(32, target_h)
            target_w = max(32, target_w)
            
            batch_shapes[i // self.batch_size] = np.array([target_h, target_w])
        
        return batch_shapes
    
    def _get_transform(self):
        """Get data augmentation transforms."""
        return A.Compose([
            A.HorizontalFlip(p=0.5),
            A.RandomBrightnessContrast(p=0.2),
            A.RGBShift(p=0.2),
            A.HueSaturationValue(p=0.2),
            A.RandomScale(scale_limit=0.2, p=0.2),
            A.RandomRotate90(p=0.2),
            # Simplified CoarseDropout with only parameters that are supported
            A.CoarseDropout(p=0.2),
        ], bbox_params=A.BboxParams(format='pascal_voc', label_fields=['labels'])) 

def letterbox(img, new_shape=(640, 640), color=(114, 114, 114), auto=True, scaleFill=False, scaleup=True, stride=32):
    """Resize image to a 32-pixel-multiple rectangle https://github.com/ultralytics/yolov3/issues/232"""
    shape = img.shape[:2]  # current shape [height, width]
    if isinstance(new_shape, int):
        new_shape = (new_shape, new_shape)

    # Scale ratio (new / old)
    r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
    if not scaleup:  # only scale down, do not scale up (for better test mAP)
        r = min(r, 1.0)

    # Compute padding
    ratio = r, r  # width, height ratios
    new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
    dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]  # wh padding
    if auto:  # minimum rectangle
        dw, dh = np.mod(dw, stride), np.mod(dh, stride)  # wh padding
    elif scaleFill:  # stretch
        dw, dh = 0.0, 0.0
        new_unpad = (new_shape[1], new_shape[0])
        ratio = new_shape[1] / shape[1], new_shape[0] / shape[0]  # width, height ratios

    dw /= 2  # divide padding into 2 sides
    dh /= 2

    if shape[::-1] != new_unpad:  # resize
        img = cv2.resize(img, new_unpad, interpolation=cv2.INTER_LINEAR)
    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    img = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)  # add border
    return img, ratio, (dw, dh) 