import os
import json
import torch
import torch.nn.functional as F
import numpy as np
from PIL import Image
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as T
import torchvision.transforms.functional as TF
import random
import albumentations as A
from albumentations.pytorch import ToTensorV2
from pycocotools.coco import COCO

# COCO dataset class names
COCO_CLASSES = [
    'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat',
    'traffic light', 'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird', 'cat',
    'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe', 'backpack',
    'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball',
    'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard', 'tennis racket',
    'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple',
    'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair',
    'couch', 'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop', 'mouse', 'remote',
    'keyboard', 'cell phone', 'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'book',
    'clock', 'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush'
]

class COCODataset(Dataset):
    """Dataset class for COCO object detection dataset."""
    
    def __init__(self, img_dir, ann_file, input_size=(640, 640), transforms=None):
        """
        Initialize the COCO dataset.
        
        Args:
            img_dir: Directory with images
            ann_file: Path to annotation file
            input_size: Model input size (width, height)
            transforms: Data augmentation transforms
        """
        self.img_dir = img_dir
        self.input_size = input_size
        self.transforms = transforms
        
        # Load COCO annotations
        self.coco = COCO(ann_file)
        self.ids = list(sorted(self.coco.imgs.keys()))
        
        # Filter out images without annotations
        valid_ids = []
        for img_id in self.ids:
            ann_ids = self.coco.getAnnIds(imgIds=img_id)
            if len(ann_ids) > 0:
                valid_ids.append(img_id)
        self.ids = valid_ids
        
        # Category mapping
        self.categories = {cat['id']: i for i, cat in enumerate(self.coco.loadCats(self.coco.getCatIds()))}
    
    def __len__(self):
        return len(self.ids)
    
    def __getitem__(self, index):
        """
        Get dataset item.
        
        Args:
            index: Index of the item
            
        Returns:
            tuple: (image, target) where target is a dictionary of annotations
        """
        # Get image ID
        img_id = self.ids[index]
        
        # Load image
        img_info = self.coco.loadImgs(img_id)[0]
        image_path = os.path.join(self.img_dir, img_info['file_name'])
        image = Image.open(image_path).convert('RGB')
        
        # Load annotations
        ann_ids = self.coco.getAnnIds(imgIds=img_id)
        annotations = self.coco.loadAnns(ann_ids)
        
        # Extract bounding boxes and labels
        boxes = []
        labels = []
        areas = []
        
        for ann in annotations:
            # Skip crowd annotations
            if ann.get('iscrowd', 0) == 1:
                continue
                
            # Skip annotations with invalid category ID
            if ann['category_id'] not in self.categories:
                continue
                
            # Extract bbox and convert from [x, y, w, h] to [x1, y1, x2, y2]
            x, y, w, h = ann['bbox']
            if w <= 0 or h <= 0:
                continue
                
            x1 = x
            y1 = y
            x2 = x + w
            y2 = y + h
            
            boxes.append([x1, y1, x2, y2])
            labels.append(self.categories[ann['category_id']])
            areas.append(ann['area'])
        
        # Convert to tensors
        boxes = torch.as_tensor(boxes, dtype=torch.float32)
        labels = torch.as_tensor(labels, dtype=torch.int64)
        areas = torch.as_tensor(areas, dtype=torch.float32)
        
        # Create target dictionary
        target = {
            'boxes': boxes,
            'labels': labels,
            'image_id': torch.tensor([img_id]),
            'area': areas,
            'orig_size': torch.as_tensor([img_info['height'], img_info['width']])
        }
        
        # Apply transformations
        if self.transforms:
            image, target = self.transforms(image, target)
        
        return image, target

def collate_fn(batch):
    """
    Custom collate function for DataLoader.
    
    Args:
        batch: Batch of (image, target) tuples
        
    Returns:
        tuple: (images, targets) where images is a tensor and targets is a list
    """
    images, targets = zip(*batch)
    
    # Stack images into a tensor
    images = torch.stack(images, 0)
    
    return images, targets

def create_data_loaders(train_dataset, val_dataset, batch_size=8, num_workers=4, pin_memory=True):
    """
    Create data loaders for training and validation.
    
    Args:
        train_dataset: Training dataset
        val_dataset: Validation dataset
        batch_size: Batch size
        num_workers: Number of workers for data loading
        pin_memory: Whether to pin memory for faster GPU transfer
        
    Returns:
        tuple: (train_loader, val_loader)
    """
    train_loader = None
    val_loader = None
    
    if train_dataset:
        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=pin_memory,
            collate_fn=collate_fn,
            drop_last=True
        )
    
    if val_dataset:
        val_loader = DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=pin_memory,
            collate_fn=collate_fn
        )
    
    return train_loader, val_loader

def get_augmentations(cfg):
    """
    Create augmentation transforms for training and validation.
    
    Args:
        cfg: Configuration dictionary
        
    Returns:
        tuple: (train_transforms, val_transforms)
    """
    input_size = cfg['model']['input_size']
    
    # Basic transforms for both training and validation
    val_transforms = Compose([
        Resize(input_size),
        ToTensor(),
        Normalize(),
    ])
    
    # Additional augmentations for training
    train_transforms = Compose([
        RandomResize(scales=[0.8, 1.2], size=input_size),
        RandomHorizontalFlip(p=0.5),
        RandomBrightnesContrast(p=0.5),
        RandomHSV(
            p=0.5,
            h=cfg['augmentation'].get('hsv_h', 0.015),
            s=cfg['augmentation'].get('hsv_s', 0.7),
            v=cfg['augmentation'].get('hsv_v', 0.4)
        ),
        ToTensor(),
        Normalize(),
    ])
    
    # Add more augmentations if enabled
    if cfg['augmentation'].get('use_cutout', False):
        train_transforms.transforms.insert(-2, Cutout(p=0.3))
    
    if cfg['augmentation'].get('use_solarize', False):
        train_transforms.transforms.insert(-2, Solarize(p=0.2))
    
    if cfg['augmentation'].get('mosaic_prob', 0) > 0:
        train_transforms = MosaicTransform(
            train_transforms, 
            prob=cfg['augmentation'].get('mosaic_prob', 0.5),
            input_size=input_size
        )
    
    return train_transforms, val_transforms

# Base transform classes
class Compose:
    """Compose multiple transforms."""
    
    def __init__(self, transforms):
        self.transforms = transforms
    
    def __call__(self, image, target=None):
        for transform in self.transforms:
            image, target = transform(image, target)
        return image, target

class ToTensor:
    """Convert PIL Image to tensor."""
    
    def __call__(self, image, target=None):
        if isinstance(image, Image.Image):
            image = TF.to_tensor(image)
        elif isinstance(image, np.ndarray):
            image = torch.from_numpy(image.transpose(2, 0, 1)).float() / 255.0
        
        return image, target

class Normalize:
    """Normalize image using ImageNet mean and std."""
    
    def __init__(self, mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)):
        self.mean = mean
        self.std = std
    
    def __call__(self, image, target=None):
        if isinstance(image, torch.Tensor):
            image = TF.normalize(image, mean=self.mean, std=self.std)
        return image, target

class Resize:
    """Resize image and update bounding boxes."""
    
    def __init__(self, size):
        if isinstance(size, int):
            self.size = (size, size)
        else:
            # Ensure size is a tuple (width, height)
            self.size = tuple(size)  # Convert list to tuple
    
    def __call__(self, image, target=None):
        # Get original size
        if isinstance(image, Image.Image):
            width, height = image.size
            # Resize image
            image = image.resize(self.size, Image.BILINEAR)
        elif isinstance(image, torch.Tensor):
            _, height, width = image.shape
            # Resize image - convert size to tuple if needed
            image = F.interpolate(image.unsqueeze(0), size=tuple(self.size[::-1]), mode='bilinear', align_corners=False)[0]
        elif isinstance(image, np.ndarray):
            height, width, _ = image.shape
            # Resize image
            image = np.array(Image.fromarray(image).resize(self.size, Image.BILINEAR))
        
        # Update bounding boxes if target is provided
        if target is not None and 'boxes' in target and len(target['boxes']) > 0:
            # Calculate scale factors
            scale_x = self.size[0] / width
            scale_y = self.size[1] / height
            
            # Scale boxes
            boxes = target['boxes'].clone()
            boxes[:, 0] *= scale_x  # x1
            boxes[:, 1] *= scale_y  # y1
            boxes[:, 2] *= scale_x  # x2
            boxes[:, 3] *= scale_y  # y2
            
            # Update target
            target['boxes'] = boxes
        
        return image, target

# Data augmentation classes
class RandomHorizontalFlip:
    """Randomly flip image horizontally."""
    
    def __init__(self, p=0.5):
        self.p = p
    
    def __call__(self, image, target=None):
        if random.random() < self.p:
            # Flip image
            if isinstance(image, Image.Image):
                image = image.transpose(Image.FLIP_LEFT_RIGHT)
                width, _ = image.size
            elif isinstance(image, torch.Tensor):
                image = torch.flip(image, [2])
                _, _, width = image.shape
            elif isinstance(image, np.ndarray):
                image = np.fliplr(image).copy()
                width = image.shape[1]
            
            # Update bounding boxes if target is provided
            if target is not None and 'boxes' in target and len(target['boxes']) > 0:
                boxes = target['boxes'].clone()
                boxes[:, [0, 2]] = width - boxes[:, [2, 0]]
                target['boxes'] = boxes
        
        return image, target

class RandomResize:
    """Randomly resize image."""
    
    def __init__(self, scales, size):
        self.scales = scales
        if isinstance(size, int):
            self.size = (size, size)
        elif isinstance(size, list):
            self.size = tuple(size)  # Convert list to tuple
        else:
            self.size = size  # Already a tuple
    
    def __call__(self, image, target=None):
        # Choose random scale
        scale = random.uniform(self.scales[0], self.scales[1])
        
        if isinstance(image, Image.Image):
            width, height = image.size
            new_width = int(width * scale)
            new_height = int(height * scale)
            image = image.resize((new_width, new_height), Image.BILINEAR)
        elif isinstance(image, torch.Tensor):
            _, height, width = image.shape
            new_width = int(width * scale)
            new_height = int(height * scale)
            image = F.interpolate(image.unsqueeze(0), size=(new_height, new_width), mode='bilinear', align_corners=False)[0]
        elif isinstance(image, np.ndarray):
            height, width, _ = image.shape
            new_width = int(width * scale)
            new_height = int(height * scale)
            image = np.array(Image.fromarray(image).resize((new_width, new_height), Image.BILINEAR))
        
        # Update bounding boxes if target is provided
        if target is not None and 'boxes' in target and len(target['boxes']) > 0:
            boxes = target['boxes'].clone()
            boxes *= scale
            target['boxes'] = boxes
        
        # Then apply fixed resize
        resize_transform = Resize(self.size)
        return resize_transform(image, target)

class RandomBrightnesContrast:
    """Randomly adjust brightness and contrast."""
    
    def __init__(self, p=0.5, brightness=0.2, contrast=0.2):
        self.p = p
        self.brightness = brightness
        self.contrast = contrast
    
    def __call__(self, image, target=None):
        if random.random() < self.p:
            # Convert to numpy if tensor
            if isinstance(image, torch.Tensor):
                image_np = image.permute(1, 2, 0).numpy()
            elif isinstance(image, Image.Image):
                image_np = np.array(image) / 255.0
            else:
                image_np = image.copy() / 255.0
            
            # Apply transformations
            transform = A.Compose([
                A.RandomBrightnessContrast(
                    brightness_limit=self.brightness,
                    contrast_limit=self.contrast,
                    p=1.0
                )
            ])
            
            image_np = transform(image=image_np)['image']
            
            # Convert back to original format
            if isinstance(image, torch.Tensor):
                image = torch.from_numpy(image_np).permute(2, 0, 1)
            elif isinstance(image, Image.Image):
                image = Image.fromarray((image_np * 255).astype(np.uint8))
            else:
                image = image_np * 255
        
        return image, target

class RandomHSV:
    """Randomly adjust hue, saturation, and value."""
    
    def __init__(self, p=0.5, h=0.015, s=0.7, v=0.4):
        self.p = p
        self.h = h
        self.s = s
        self.v = v
    
    def __call__(self, image, target=None):
        if random.random() < self.p:
            # Convert to numpy as uint8
            if isinstance(image, torch.Tensor):
                # Convert tensor to uint8 numpy
                image_np = (image.permute(1, 2, 0).numpy() * 255).astype(np.uint8)
            elif isinstance(image, Image.Image):
                # Convert PIL to uint8 numpy
                image_np = np.array(image).astype(np.uint8)
            else:
                # Ensure numpy is uint8
                image_np = image.astype(np.uint8)
            
            # Apply transformations with uint8 input
            transform = A.Compose([
                A.HueSaturationValue(
                    hue_shift_limit=int(self.h * 180),
                    sat_shift_limit=int(self.s * 100),
                    val_shift_limit=int(self.v * 100),
                    p=1.0
                )
            ])
            
            # Apply transform
            image_np = transform(image=image_np)['image']
            
            # Convert back to original format
            if isinstance(image, torch.Tensor):
                # Convert back to tensor (0-1 range)
                image = torch.from_numpy(image_np.astype(np.float32) / 255.0).permute(2, 0, 1)
            elif isinstance(image, Image.Image):
                # Convert back to PIL
                image = Image.fromarray(image_np)
            else:
                # Leave as numpy array
                image = image_np
        
        return image, target

class Cutout:
    """Apply cutout augmentation."""
    
    def __init__(self, p=0.5, scale=(0.02, 0.2), ratio=(0.5, 2.0), value=0):
        self.p = p
        self.scale = scale
        self.ratio = ratio
        self.value = value
    
    def __call__(self, image, target=None):
        if random.random() < self.p:
            # Get image dimensions
            if isinstance(image, torch.Tensor):
                _, height, width = image.shape
            elif isinstance(image, Image.Image):
                width, height = image.size
            else:
                height, width, _ = image.shape
            
            # Calculate cutout size
            area = height * width
            target_area = random.uniform(self.scale[0], self.scale[1]) * area
            aspect_ratio = random.uniform(self.ratio[0], self.ratio[1])
            
            w = int(round(np.sqrt(target_area * aspect_ratio)))
            h = int(round(np.sqrt(target_area / aspect_ratio)))
            
            if w < width and h < height:
                # Choose random location for cutout
                x = random.randint(0, width - w)
                y = random.randint(0, height - h)
                
                # Apply cutout
                if isinstance(image, torch.Tensor):
                    image[:, y:y+h, x:x+w] = self.value
                elif isinstance(image, Image.Image):
                    img_arr = np.array(image)
                    img_arr[y:y+h, x:x+w] = self.value
                    image = Image.fromarray(img_arr)
                else:
                    image[y:y+h, x:x+w] = self.value
        
        return image, target

class Solarize:
    """Apply solarize augmentation."""
    
    def __init__(self, p=0.5, threshold=128):
        self.p = p
        self.threshold = threshold
    
    def __call__(self, image, target=None):
        if random.random() < self.p:
            # Apply solarize
            if isinstance(image, torch.Tensor):
                image_np = (image.permute(1, 2, 0).numpy() * 255).astype(np.uint8)
                image_np = np.where(image_np >= self.threshold, 255 - image_np, image_np)
                image = torch.from_numpy(image_np.astype(np.float32) / 255.0).permute(2, 0, 1)
            elif isinstance(image, Image.Image):
                image = TF.solarize(image, self.threshold)
            else:
                image = np.where(image >= self.threshold, 255 - image, image)
        
        return image, target

class MosaicTransform:
    """Apply mosaic augmentation (combines 4 images)."""
    
    def __init__(self, base_transform, prob=0.5, input_size=(640, 640)):
        self.base_transform = base_transform
        self.prob = prob
        self.input_size = input_size if isinstance(input_size, tuple) else (input_size, input_size)
    
    def __call__(self, image, target):
        # Apply base transform first
        image, target = self.base_transform(image, target)
        
        # We can't apply mosaic in __getitem__, so we just return the single image
        return image, target 