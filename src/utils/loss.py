import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class ComposedLoss(nn.Module):
    """
    Composed loss function for object detection with knowledge distillation.
    
    This loss combines:
    1. Classification loss (BCE)
    2. Regression loss (CIoU)
    3. Knowledge distillation loss (KL Divergence)
    """
    def __init__(self, model, teacher_model=None, distill_weight=0.5):
        super().__init__()
        self.model = model
        self.teacher_model = teacher_model
        self.distill_weight = distill_weight
        
        # Loss weights
        self.cls_weight = 1.0
        self.box_weight = 5.0
        self.obj_weight = 1.0
        
        # Loss functions
        self.bce = nn.BCEWithLogitsLoss(reduction='none')
        self.kl_div = nn.KLDivLoss(reduction='batchmean')
    
    def forward(self, preds, targets, teacher_preds=None):
        """
        Calculate the combined loss.
        
        Args:
            preds: Model predictions
            targets: Ground truth targets
            teacher_preds: Teacher model predictions for knowledge distillation
            
        Returns:
            Tensor: Combined loss
        """
        # Unpack predictions
        pred_cls, pred_box, pred_obj = self._unpack_predictions(preds)
        
        # Calculate standard detection losses
        cls_loss = self._cls_loss(pred_cls, targets)
        box_loss = self._box_loss(pred_box, targets)
        obj_loss = self._obj_loss(pred_obj, targets)
        
        # Calculate knowledge distillation loss if teacher model is available
        distill_loss = torch.tensor(0.0, device=cls_loss.device)
        if self.teacher_model is not None and teacher_preds is not None:
            distill_loss = self._distillation_loss(preds, teacher_preds)
        
        # Combine losses
        total_loss = (
            self.cls_weight * cls_loss +
            self.box_weight * box_loss +
            self.obj_weight * obj_loss
        )
        
        # Add distillation loss if available
        if self.teacher_model is not None and teacher_preds is not None:
            total_loss = (1 - self.distill_weight) * total_loss + self.distill_weight * distill_loss
        
        return total_loss
    
    def _unpack_predictions(self, preds):
        """Unpack model predictions into classification, box, and objectness components."""
        # This is a placeholder implementation - actual implementation depends on model output format
        # For example, if preds is a list of tensors for each detection layer:
        if isinstance(preds, list):
            pred_cls = [p[..., 5:] for p in preds]
            pred_box = [p[..., :4] for p in preds]
            pred_obj = [p[..., 4:5] for p in preds]
        else:
            # If preds is a single tensor with shape [batch, anchors, grid_h, grid_w, nc+5]
            pred_cls = preds[..., 5:]
            pred_box = preds[..., :4]
            pred_obj = preds[..., 4:5]
        
        return pred_cls, pred_box, pred_obj
    
    def _cls_loss(self, pred_cls, targets):
        """Calculate classification loss."""
        # Extract target classes
        target_cls = self._get_target_classes(targets)
        
        # Apply BCE loss
        if isinstance(pred_cls, list):
            cls_loss = sum(self.bce(p, t).mean() for p, t in zip(pred_cls, target_cls))
        else:
            cls_loss = self.bce(pred_cls, target_cls).mean()
        
        return cls_loss
    
    def _box_loss(self, pred_box, targets):
        """Calculate box regression loss using Complete IoU (CIoU)."""
        # Extract target boxes
        target_box = self._get_target_boxes(targets)
        
        # Calculate CIoU loss
        if isinstance(pred_box, list):
            box_loss = sum(self._ciou_loss(p, t) for p, t in zip(pred_box, target_box))
        else:
            box_loss = self._ciou_loss(pred_box, target_box)
        
        return box_loss
    
    def _obj_loss(self, pred_obj, targets):
        """Calculate objectness loss."""
        # Extract target objectness
        target_obj = self._get_target_objectness(targets)
        
        # Apply BCE loss
        if isinstance(pred_obj, list):
            obj_loss = sum(self.bce(p, t).mean() for p, t in zip(pred_obj, target_obj))
        else:
            obj_loss = self.bce(pred_obj, target_obj).mean()
        
        return obj_loss
    
    def _distillation_loss(self, student_preds, teacher_preds):
        """Calculate knowledge distillation loss using KL divergence."""
        # Apply temperature scaling and KL divergence
        temperature = 2.0
        
        # Process predictions based on format
        if isinstance(student_preds, list) and isinstance(teacher_preds, list):
            distill_loss = 0
            for s_pred, t_pred in zip(student_preds, teacher_preds):
                # Apply softmax with temperature
                s_logits = s_pred[..., 5:] / temperature
                t_logits = t_pred[..., 5:] / temperature
                
                # Apply log_softmax to student and softmax to teacher
                s_log_probs = F.log_softmax(s_logits, dim=-1)
                t_probs = F.softmax(t_logits, dim=-1)
                
                # Calculate KL divergence
                distill_loss += self.kl_div(s_log_probs, t_probs) * (temperature ** 2)
        else:
            # Apply softmax with temperature
            s_logits = student_preds[..., 5:] / temperature
            t_logits = teacher_preds[..., 5:] / temperature
            
            # Apply log_softmax to student and softmax to teacher
            s_log_probs = F.log_softmax(s_logits, dim=-1)
            t_probs = F.softmax(t_logits, dim=-1)
            
            # Calculate KL divergence
            distill_loss = self.kl_div(s_log_probs, t_probs) * (temperature ** 2)
        
        return distill_loss
    
    def _ciou_loss(self, pred_box, target_box):
        """Calculate Complete IoU (CIoU) loss for bounding box regression."""
        # Convert boxes to [x1, y1, x2, y2] format if needed
        if pred_box.size(-1) == 4:
            # If boxes are in [x, y, w, h] format, convert to [x1, y1, x2, y2]
            pred_x1y1 = pred_box[..., :2] - pred_box[..., 2:] / 2
            pred_x2y2 = pred_box[..., :2] + pred_box[..., 2:] / 2
            pred_box_xyxy = torch.cat([pred_x1y1, pred_x2y2], dim=-1)
            
            target_x1y1 = target_box[..., :2] - target_box[..., 2:] / 2
            target_x2y2 = target_box[..., :2] + target_box[..., 2:] / 2
            target_box_xyxy = torch.cat([target_x1y1, target_x2y2], dim=-1)
        else:
            pred_box_xyxy = pred_box
            target_box_xyxy = target_box
        
        # Calculate IoU
        iou, ciou = self._box_iou_ciou(pred_box_xyxy, target_box_xyxy)
        
        # CIoU loss
        ciou_loss = (1 - ciou).mean()
        
        return ciou_loss
    
    def _box_iou_ciou(self, box1, box2):
        """
        Calculate IoU and CIoU between box1 and box2.
        
        Args:
            box1, box2: Boxes in [x1, y1, x2, y2] format
            
        Returns:
            Tuple: (IoU, CIoU)
        """
        # Calculate intersection area
        x1 = torch.max(box1[..., 0], box2[..., 0])
        y1 = torch.max(box1[..., 1], box2[..., 1])
        x2 = torch.min(box1[..., 2], box2[..., 2])
        y2 = torch.min(box1[..., 3], box2[..., 3])
        
        intersection = torch.clamp(x2 - x1, min=0) * torch.clamp(y2 - y1, min=0)
        
        # Calculate union area
        box1_area = (box1[..., 2] - box1[..., 0]) * (box1[..., 3] - box1[..., 1])
        box2_area = (box2[..., 2] - box2[..., 0]) * (box2[..., 3] - box2[..., 1])
        union = box1_area + box2_area - intersection
        
        # Calculate IoU
        iou = intersection / (union + 1e-7)
        
        # Calculate CIoU components
        # 1. Center distance
        c_x1 = (box1[..., 0] + box1[..., 2]) / 2
        c_y1 = (box1[..., 1] + box1[..., 3]) / 2
        c_x2 = (box2[..., 0] + box2[..., 2]) / 2
        c_y2 = (box2[..., 1] + box2[..., 3]) / 2
        
        center_dist_squared = (c_x1 - c_x2) ** 2 + (c_y1 - c_y2) ** 2
        
        # 2. Diagonal distance of the smallest enclosing box
        c_x_min = torch.min(box1[..., 0], box2[..., 0])
        c_y_min = torch.min(box1[..., 1], box2[..., 1])
        c_x_max = torch.max(box1[..., 2], box2[..., 2])
        c_y_max = torch.max(box1[..., 3], box2[..., 3])
        
        diagonal_dist_squared = (c_x_max - c_x_min) ** 2 + (c_y_max - c_y_min) ** 2
        
        # 3. Aspect ratio consistency
        w1 = box1[..., 2] - box1[..., 0]
        h1 = box1[..., 3] - box1[..., 1]
        w2 = box2[..., 2] - box2[..., 0]
        h2 = box2[..., 3] - box2[..., 1]
        
        v = (4 / (math.pi ** 2)) * (torch.atan(w1 / (h1 + 1e-7)) - torch.atan(w2 / (h2 + 1e-7))) ** 2
        alpha = v / (1 - iou + v + 1e-7)
        
        # Calculate CIoU
        ciou = iou - center_dist_squared / (diagonal_dist_squared + 1e-7) - alpha * v
        
        return iou, ciou
    
    def _get_target_classes(self, targets):
        """Extract target classes from ground truth targets."""
        # This is a placeholder - actual implementation depends on target format
        return targets[..., 5:]
    
    def _get_target_boxes(self, targets):
        """Extract target boxes from ground truth targets."""
        # This is a placeholder - actual implementation depends on target format
        return targets[..., :4]
    
    def _get_target_objectness(self, targets):
        """Extract target objectness from ground truth targets."""
        # This is a placeholder - actual implementation depends on target format
        return targets[..., 4:5] 