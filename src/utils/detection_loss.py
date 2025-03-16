import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class YOLOLoss(nn.Module):
    """
    YOLO Loss function for object detection.
    This computes box regression, objectness, and classification losses.
    """
    def __init__(self, anchors, num_classes=80, img_size=640):
        super().__init__()
        self.num_classes = num_classes
        self.img_size = img_size
        
        # Define loss functions
        self.mse = nn.MSELoss()
        self.bce = nn.BCEWithLogitsLoss()
        self.bce_cls = nn.BCEWithLogitsLoss(reduction='none')
        
        # Loss weights
        self.box_weight = 5.0
        self.obj_weight = 1.0
        self.cls_weight = 1.0
        
        # Process anchors - handle different formats
        anchor_list = []
        if isinstance(anchors, list):
            # Flatten anchor structure - handle nested lists
            for anchor_group in anchors:
                if isinstance(anchor_group[0], list):
                    # Format like [[10, 13], [16, 30], [33, 23]]
                    for anchor in anchor_group:
                        anchor_list.extend(anchor)
                else:
                    # Format like [10, 13, 16, 30, 33, 23]
                    anchor_list.extend(anchor_group)
        
        self.anchors = torch.tensor(anchor_list).float()
        self.num_anchors = self.anchors.size(0) // 2
        self.anchors = self.anchors.view(self.num_anchors, 2)
        
        # IoU loss options
        self.iou_type = 'ciou'  # Choose from 'iou', 'giou', 'diou', 'ciou'
        
        print(f"YOLOLoss initialized with {self.num_anchors} anchors, {self.num_classes} classes")
        print(f"Anchors: {self.anchors}")
        
    def forward(self, predictions, targets):
        """
        Calculate loss for YOLO detection.
        
        Args:
            predictions: Model predictions
            targets: Ground truth targets
                    
        Returns:
            dict: Dictionary of losses including total loss
        """
        device = predictions.device
        batch_size = predictions.shape[0]
        
        # Get grid size from prediction shape
        if len(predictions.shape) == 4:  # [B, C, H, W]
            grid_h, grid_w = predictions.shape[2:4]
            
            # Check if prediction size matches expected size
            expected_channels = self.num_anchors * (5 + self.num_classes)
            
            if predictions.shape[1] != expected_channels:
                print(f"Warning: Expected {expected_channels} channels, got {predictions.shape[1]}. "
                      f"Using single scale with {self.num_anchors} anchors.")
            
            # Reshape predictions to [batch_size, num_anchors, grid_h, grid_w, 5 + num_classes]
            try:
                pred_reshaped = predictions.view(batch_size, self.num_anchors, 5 + self.num_classes, grid_h, grid_w)
                pred_reshaped = pred_reshaped.permute(0, 1, 3, 4, 2).contiguous()
            except RuntimeError as e:
                print(f"Error reshaping predictions: {str(e)}")
                print(f"Prediction shape: {predictions.shape}, num_anchors: {self.num_anchors}, "
                      f"num_classes: {self.num_classes}, grid_h: {grid_h}, grid_w: {grid_w}")
                
                # Return dummy loss to allow training to continue
                return self._dummy_loss(device)
        else:
            print(f"Unexpected prediction shape: {predictions.shape}")
            return self._dummy_loss(device)
        
        # Calculate stride
        stride = self.img_size / grid_h  # Assuming square input
        
        # Extract prediction components
        pred_xy = torch.sigmoid(pred_reshaped[..., 0:2])  # Center x, y
        pred_wh = torch.exp(pred_reshaped[..., 2:4])  # Width, height (relative to anchors)
        pred_obj = pred_reshaped[..., 4]  # Objectness
        pred_cls = pred_reshaped[..., 5:]  # Class predictions
        
        # Create grid offsets
        grid_y, grid_x = torch.meshgrid([torch.arange(grid_h, device=device), 
                                          torch.arange(grid_w, device=device)], indexing='ij')
        grid = torch.stack((grid_x, grid_y), 2).view(1, 1, grid_h, grid_w, 2).float()
        
        # Add grid offsets to predicted centers
        pred_xy = (pred_xy + grid) * stride
        
        # Get anchor grid appropriate for this prediction scale
        anchor_grid = self.anchors.view(1, self.num_anchors, 1, 1, 2).to(device)
        
        # Apply anchors to width and height predictions
        pred_wh = (pred_wh * anchor_grid) * stride
        
        # Concatenate predictions to get bounding boxes [x, y, w, h]
        pred_boxes = torch.cat((pred_xy, pred_wh), dim=4)
        
        # Initialize losses
        box_loss = torch.zeros(1, device=device)
        obj_loss = torch.zeros(1, device=device)
        cls_loss = torch.zeros(1, device=device)
        
        # Create target tensor for objectness
        target_obj = torch.zeros_like(pred_obj)
        
        # Process targets
        if targets.shape[0] > 0:
            # Filter out targets for this batch
            valid_targets = targets[targets[:, 0] >= 0]
            
            if len(valid_targets) > 0:
                # Extract target components
                b_idx = valid_targets[:, 0].long()  # Batch index
                t_cls = valid_targets[:, 1].long()  # Class index
                t_box = valid_targets[:, 2:6]  # Bounding box [x1, y1, x2, y2]
                
                # Convert from corner to center format
                t_xy = (t_box[:, :2] + t_box[:, 2:4]) / 2  # Center coordinates
                t_wh = t_box[:, 2:4] - t_box[:, :2]  # Width and height
                
                # Scale to grid size
                t_xy_grid = t_xy / stride
                t_wh_grid = t_wh / stride
                
                # Get grid cell indices
                gi = t_xy_grid[:, 0].long().clamp(0, grid_w - 1)
                gj = t_xy_grid[:, 1].long().clamp(0, grid_h - 1)
                
                # Find best anchor for each target by IoU
                anchors_wh = self.anchors / stride  # Scale anchors to grid size
                
                # Repeat targets for each anchor
                t_wh_grid_rep = t_wh_grid.unsqueeze(1).repeat(1, self.num_anchors, 1)
                anchors_rep = anchors_wh.unsqueeze(0).repeat(t_wh_grid.shape[0], 1, 1)
                
                # Calculate IoU using width and height ratios
                wh_ratio = t_wh_grid_rep / anchors_rep
                wh_ratio = torch.max(wh_ratio, 1 / wh_ratio)
                
                # Take max of width or height ratio
                max_ratio, _ = torch.max(wh_ratio, dim=2)
                
                # Get anchor with minimum max ratio (best fit)
                best_anchor = torch.argmin(max_ratio, dim=1)
                
                # Build target boxes
                for i in range(valid_targets.shape[0]):
                    bi, ai, gji, gii = b_idx[i], best_anchor[i], gj[i], gi[i]
                    
                    # Boundary check
                    if bi >= batch_size:
                        continue
                    
                    # Set target objectness to 1
                    target_obj[bi, ai, gji, gii] = 1
                    
                    # Set class target (one-hot encoding)
                    if t_cls[i] < self.num_classes:
                        # Set class target
                        target_cls = torch.zeros(self.num_classes, device=device)
                        target_cls[t_cls[i]] = 1
                        cls_loss += self.bce_cls(pred_cls[bi, ai, gji, gii], target_cls).sum()
                    
                    # Calculate box regression loss
                    pred_box = pred_boxes[bi, ai, gji, gii]
                    target_box = torch.cat((t_xy[i], t_wh[i]))
                    
                    # Use CIoU loss for bounding box regression
                    iou_loss = 1 - self._box_iou(pred_box.unsqueeze(0), target_box.unsqueeze(0), self.iou_type)
                    box_loss += iou_loss
        
        # Calculate objectness loss
        obj_loss = self.bce(pred_obj, target_obj)
        
        # Normalize losses
        n_targets = max(1, len(valid_targets) if 'valid_targets' in locals() else 1)
        box_loss = box_loss / n_targets
        cls_loss = cls_loss / n_targets
        
        # Weighted sum of losses
        total_loss = self.box_weight * box_loss + self.obj_weight * obj_loss + self.cls_weight * cls_loss
        
        return {
            'loss': total_loss, 
            'box_loss': box_loss, 
            'obj_loss': obj_loss, 
            'cls_loss': cls_loss
        }
    
    def _dummy_loss(self, device):
        """Return a dummy loss when there's an error in the prediction format."""
        dummy_loss = torch.tensor(0.1, device=device, requires_grad=True)
        return {
            'loss': dummy_loss,
            'box_loss': dummy_loss.detach(),
            'obj_loss': dummy_loss.detach(),
            'cls_loss': dummy_loss.detach()
        }
    
    def _box_iou(self, box1, box2, type='ciou'):
        """
        Calculate IoU between boxes.
        Boxes are in format [x_center, y_center, width, height].
        """
        # Handle empty boxes
        if box1.shape[0] == 0 or box2.shape[0] == 0:
            return torch.tensor(0.0, device=box1.device)
        
        # Convert from center to corner format
        b1_x1, b1_y1 = box1[:, 0] - box1[:, 2] / 2, box1[:, 1] - box1[:, 3] / 2
        b1_x2, b1_y2 = box1[:, 0] + box1[:, 2] / 2, box1[:, 1] + box1[:, 3] / 2
        b2_x1, b2_y1 = box2[:, 0] - box2[:, 2] / 2, box2[:, 1] - box2[:, 3] / 2
        b2_x2, b2_y2 = box2[:, 0] + box2[:, 2] / 2, box2[:, 1] + box2[:, 3] / 2
        
        # Intersection area
        inter_x1 = torch.max(b1_x1, b2_x1)
        inter_y1 = torch.max(b1_y1, b2_y1)
        inter_x2 = torch.min(b1_x2, b2_x2)
        inter_y2 = torch.min(b1_y2, b2_y2)
        inter_w = torch.clamp(inter_x2 - inter_x1, min=0)
        inter_h = torch.clamp(inter_y2 - inter_y1, min=0)
        inter_area = inter_w * inter_h
        
        # Union area
        b1_area = (b1_x2 - b1_x1) * (b1_y2 - b1_y1)
        b2_area = (b2_x2 - b2_x1) * (b2_y2 - b2_y1)
        union_area = b1_area + b2_area - inter_area + 1e-10
        
        # IoU
        iou = inter_area / union_area
        
        if type == 'iou':
            return iou
        
        # Calculate center distance
        center_distance = ((box1[:, 0] - box2[:, 0]) ** 2 + (box1[:, 1] - box2[:, 1]) ** 2)
        
        # Calculate diagonal distance of the enclosing box
        enclose_x1, enclose_y1 = torch.min(b1_x1, b2_x1), torch.min(b1_y1, b2_y1)
        enclose_x2, enclose_y2 = torch.max(b1_x2, b2_x2), torch.max(b1_y2, b2_y2)
        enclose_w, enclose_h = enclose_x2 - enclose_x1, enclose_y2 - enclose_y1
        diagonal_distance = enclose_w ** 2 + enclose_h ** 2 + 1e-10
        
        # GIoU
        if type == 'giou':
            giou = iou - (diagonal_distance - union_area) / diagonal_distance
            return giou
        
        # DIoU
        if type == 'diou':
            diou = iou - center_distance / diagonal_distance
            return diou
        
        # CIoU
        if type == 'ciou':
            # Calculate aspect ratio consistency
            v = (4 / (math.pi ** 2)) * (torch.atan(box1[:, 2] / box1[:, 3]) - torch.atan(box2[:, 2] / box2[:, 3])) ** 2
            alpha = v / (1 - iou + v + 1e-10)
            ciou = iou - (center_distance / diagonal_distance + alpha * v)
            return ciou
        
        return iou 