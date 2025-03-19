import os
import random
import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

import albumentations as A
from albumentations.pytorch import ToTensorV2
from pycocotools.coco import COCO

class COCODataset(Dataset):
    """Custom Dataset for loading COCO dataset."""
    def __init__(self, img_dir, ann_file, transforms=None, img_size=(640, 640)):
        self.img_dir = img_dir
        self.transforms = transforms
        self.img_size = img_size
        
        # Load COCO annotations
        self.coco = COCO(ann_file)
        self.ids = list(self.coco.imgs.keys())
        
        # Filter out images without annotations
        ids_with_anns = []
        for img_id in self.ids:
            ann_ids = self.coco.getAnnIds(imgIds=img_id)
            if len(ann_ids) > 0:
                ids_with_anns.append(img_id)
        self.ids = ids_with_anns
        
        # Class category mapping
        self.cat_ids = self.coco.getCatIds()
        self.cat_id_to_continuous_id = {
            v: i for i, v in enumerate(self.cat_ids)
        }
    
    def __len__(self):
        return len(self.ids)
    
    def __getitem__(self, index):
        # Load image
        img_id = self.ids[index]
        img_info = self.coco.loadImgs(img_id)[0]
        img_path = os.path.join(self.img_dir, img_info['file_name'])
        img = Image.open(img_path).convert('RGB')
        img = np.array(img)
        
        # Get annotations
        ann_ids = self.coco.getAnnIds(imgIds=img_id)
        anns = self.coco.loadAnns(ann_ids)
        
        # Extract bounding boxes and labels
        boxes = []
        labels = []
        
        for ann in anns:
            if ann['bbox'][2] < 1 or ann['bbox'][3] < 1:
                continue
                
            x, y, w, h = ann['bbox']
            # Convert to [x1, y1, x2, y2] format
            x1 = x
            y1 = y
            x2 = x + w
            y2 = y + h
            
            # Normalize to [0, 1] (will be converted to absolute coordinates for transforms)
            img_h, img_w = img.shape[:2]
            x1 /= img_w
            y1 /= img_h
            x2 /= img_w
            y2 /= img_h
            
            label = self.cat_id_to_continuous_id[ann['category_id']]
            
            boxes.append([x1, y1, x2, y2])
            labels.append(label)
        
        # Create target dictionary
        target = {
            'boxes': np.array(boxes, dtype=np.float32),
            'labels': np.array(labels, dtype=np.int64),
            'image_id': np.array([img_id], dtype=np.int64)
        }
        
        # Apply transforms
        if self.transforms:
            # Convert normalized to absolute coordinates for albumentations
            h, w = img.shape[:2]
            abs_boxes = target['boxes'].copy()
            abs_boxes[:, [0, 2]] *= w
            abs_boxes[:, [1, 3]] *= h
            
            # Albumentations expects [x_min, y_min, x_max, y_max]
            transformed = self.transforms(
                image=img,
                bboxes=abs_boxes,
                labels=target['labels']
            )
            
            img = transformed['image']
            
            # Re-normalize coordinates
            if len(transformed['bboxes']) > 0:
                boxes = np.array(transformed['bboxes'], dtype=np.float32)
                h, w = self.img_size
                boxes[:, [0, 2]] /= w
                boxes[:, [1, 3]] /= h
                target['boxes'] = boxes
                target['labels'] = np.array(transformed['labels'], dtype=np.int64)
            else:
                # If no boxes left after transform, create dummy box
                target['boxes'] = np.array([[0, 0, 1, 1]], dtype=np.float32)
                target['labels'] = np.array([0], dtype=np.int64)
        
        # Convert to tensors
        img = torch.as_tensor(img, dtype=torch.float32) / 255.0
        target['boxes'] = torch.as_tensor(target['boxes'], dtype=torch.float32)
        target['labels'] = torch.as_tensor(target['labels'], dtype=torch.int64)
        target['image_id'] = torch.as_tensor(target['image_id'], dtype=torch.int64)
        
        return img, target

def get_augmentations(cfg):
    """Get data augmentation transforms for training and validation."""
    img_size = cfg['model']['input_size']
    
    # Training transforms with augmentations
    train_transforms = A.Compose([
        A.RandomResizedCrop(height=img_size[1], width=img_size[0], scale=(0.5, 1.0)),
        A.HorizontalFlip(p=0.5),
        A.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1),
        A.ShiftScaleRotate(shift_limit=0.1, scale_limit=0.1, rotate_limit=10, p=0.5),
        
        # Mosaic augmentation would be implemented separately
        
        # Add more augmentations if specified in config
        A.Cutout(num_holes=8, max_h_size=img_size[1]//16, max_w_size=img_size[0]//16, p=0.5 if cfg['augmentation']['use_cutout'] else 0),
        A.Solarize(p=0.2 if cfg['augmentation']['use_solarize'] else 0),
        
        # Normalize and convert to tensor
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2(),
    ], bbox_params=A.BboxParams(format='pascal_voc', label_fields=['labels']))
    
    # Validation transforms
    val_transforms = A.Compose([
        A.Resize(height=img_size[1], width=img_size[0]),
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2(),
    ], bbox_params=A.BboxParams(format='pascal_voc', label_fields=['labels']))
    
    return train_transforms, val_transforms

def create_data_loaders(train_dataset, val_dataset, batch_size=16, num_workers=4, pin_memory=True):
    """Create data loaders for training and validation."""
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
        collate_fn=collate_fn,
        drop_last=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        collate_fn=collate_fn
    )
    
    return train_loader, val_loader

def collate_fn(batch):
    """Custom collate function for batching samples with varying number of objects."""
    images = []
    targets = []
    
    for img, target in batch:
        images.append(img)
        targets.append(target)
    
    # Stack images into a single tensor
    images = torch.stack(images, dim=0)
    
    return images, targets

def mixup(img1, target1, img2, target2, alpha=1.0):
    """Mixup augmentation: combines two images and their targets."""
    # Apply mixup to the images
    lambda_param = np.random.beta(alpha, alpha)
    mixed_img = lambda_param * img1 + (1 - lambda_param) * img2
    
    # For targets, we just concatenate and keep track of the lambda value
    target1['mixup_weight'] = lambda_param
    target2['mixup_weight'] = 1 - lambda_param
    
    # Concatenate boxes and labels
    boxes = torch.cat([target1['boxes'], target2['boxes']])
    labels = torch.cat([target1['labels'], target2['labels']])
    
    mixed_target = {
        'boxes': boxes,
        'labels': labels,
        'image_id': target1['image_id'],  # Keep the first image ID
        'mixup': torch.tensor([lambda_param], dtype=torch.float32)
    }
    
    return mixed_img, mixed_target

def mosaic_augmentation(images, targets, img_size):
    """Mosaic augmentation: combines 4 images into a single image."""
    # This is a simplified implementation
    # A full implementation would handle random placement and scaling properly
    
    # Number of images to combine (typically 4)
    n = len(images)
    if n != 4:
        # Pad with duplicates if we don't have exactly 4 images
        indices = list(range(n)) + list(range(n))[:4-n]
        images = [images[i] for i in indices[:4]]
        targets = [targets[i] for i in indices[:4]]
    
    # Create a new image of 2x size
    h, w = img_size
    mosaic_img = torch.zeros((3, h*2, w*2), dtype=torch.float32)
    
    # Coordinates for the 4 images
    positions = [
        (0, 0),      # Top-left
        (w, 0),      # Top-right
        (0, h),      # Bottom-left
        (w, h)       # Bottom-right
    ]
    
    combined_boxes = []
    combined_labels = []
    
    for i, ((x, y), img, target) in enumerate(zip(positions, images, targets)):
        # Place the image
        img_h, img_w = img.shape[1:]
        mosaic_img[:, y:y+img_h, x:x+img_w] = img
        
        # Adjust the bounding boxes
        if len(target['boxes']) > 0:
            boxes = target['boxes'].clone()
            boxes[:, [0, 2]] = boxes[:, [0, 2]] * img_w + x
            boxes[:, [1, 3]] = boxes[:, [1, 3]] * img_h + y
            
            # Normalize to [0, 1]
            boxes[:, [0, 2]] /= (w*2)
            boxes[:, [1, 3]] /= (h*2)
            
            combined_boxes.append(boxes)
            combined_labels.append(target['labels'])
    
    # Combine the boxes and labels
    combined_boxes = torch.cat(combined_boxes, dim=0)
    combined_labels = torch.cat(combined_labels, dim=0)
    
    # Resize the mosaic image to original size
    mosaic_img = F.interpolate(mosaic_img.unsqueeze(0), size=img_size, mode='bilinear', align_corners=False).squeeze(0)
    
    mosaic_target = {
        'boxes': combined_boxes,
        'labels': combined_labels,
        'image_id': targets[0]['image_id']  # Keep the first image ID
    }
    
    return mosaic_img, mosaic_target

class MosaicDataset(Dataset):
    """Dataset wrapper that applies mosaic augmentation."""
    def __init__(self, dataset, img_size, p=0.5):
        self.dataset = dataset
        self.img_size = img_size
        self.p = p
        self.indices = range(len(dataset))
    
    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, index):
        if random.random() < self.p:
            # Apply mosaic augmentation
            indices = [index] + random.sample(self.indices, 3)
            images = []
            targets = []
            
            for idx in indices:
                img, target = self.dataset[idx]
                images.append(img)
                targets.append(target)
            
            img, target = mosaic_augmentation(images, targets, self.img_size)
            return img, target
        else:
            # Return the original image
            return self.dataset[index] 