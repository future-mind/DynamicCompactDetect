import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import numpy as np


def smooth_BCE(eps=0.1):
    """
    Return positive, negative label smoothing BCE targets.
    
    Args:
        eps: epsilon value for label smoothing
        
    Returns:
        positive, negative label smoothing BCE targets
    """
    return 1.0 - 0.5 * eps, 0.5 * eps


class BCEBlurWithLogitsLoss(nn.Module):
    """
    BCE with logits loss with blurred labels.
    
    Args:
        alpha: weight for positive examples
        gamma: focal loss gamma (focusing parameter)
        blur: standard deviation for Gaussian blur
        eps: epsilon for numerical stability
    """
    def __init__(self, alpha=0.5, gamma=1.0, blur=0.05, eps=1e-7):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.blur = blur
        self.eps = eps
    
    def forward(self, pred, true):
        pred = pred.sigmoid()
        
        # Blur the labels
        if self.blur > 0:
            with torch.no_grad():
                true = true * (1 - self.blur) + 0.5 * self.blur
        
        # BCE loss
        loss = - (true * torch.log(pred + self.eps) + (1 - true) * torch.log(1 - pred + self.eps))
        
        # Focal loss
        if self.gamma:
            p_t = torch.exp(-loss)
            loss = loss * ((1 - p_t) ** self.gamma)
        
        # Alpha-balanced loss
        if self.alpha >= 0:
            alpha_t = self.alpha * true + (1 - self.alpha) * (1 - true)
            loss = alpha_t * loss
        
        return loss.mean()


class FocalLoss(nn.Module):
    """
    Focal Loss for dense object detection.
    
    Args:
        alpha: weight for positive examples
        gamma: focal loss gamma (focusing parameter)
        eps: epsilon for numerical stability
    """
    def __init__(self, alpha=0.25, gamma=2.0, eps=1e-7):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.eps = eps
    
    def forward(self, pred, true):
        # Sigmoid activation
        pred = pred.sigmoid()
        
        # BCE loss
        bce_loss = -true * torch.log(pred + self.eps) - (1 - true) * torch.log(1 - pred + self.eps)
        
        # Focal term
        pt = true * pred + (1 - true) * (1 - pred)
        focal_weight = (1 - pt) ** self.gamma
        
        # Alpha-balanced focal loss
        alpha_t = true * self.alpha + (1 - true) * (1 - self.alpha)
        
        # Combine all terms
        loss = alpha_t * focal_weight * bce_loss
        
        return loss.mean()


class VarifocalLoss(nn.Module):
    """
    Varifocal loss for bounding box regression and classification.
    
    Args:
        alpha: weight for positive examples
        gamma: focal loss gamma (focusing parameter)
        eps: epsilon for numerical stability
    """
    def __init__(self, alpha=0.75, gamma=2.0, eps=1e-7):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.eps = eps
    
    def forward(self, pred, true):
        # Convert pred to probability
        pred = pred.sigmoid()
        
        # Calculate focal weight
        alpha_factor = true * self.alpha + (1 - true) * (1 - self.alpha)
        focal_weight = torch.abs(true - pred) ** self.gamma
        
        # Varifocal loss
        loss = -alpha_factor * focal_weight * (true * torch.log(pred + self.eps) + (1 - true) * torch.log(1 - pred + self.eps))
        
        return loss.mean()


def bbox_iou(box1, box2, x1y1x2y2=True, GIoU=False, DIoU=False, CIoU=False, eps=1e-7):
    """
    Returns the IoU of box1 to box2. box1 is nx4, box2 is nx4.
    
    Args:
        box1: First box, can be nx4 or 4
        box2: Second box, can be nx4 or 4
        x1y1x2y2: Whether boxes are in x1y1x2y2 format
        GIoU: Whether to calculate GIoU
        DIoU: Whether to calculate DIoU
        CIoU: Whether to calculate CIoU
        eps: epsilon for numerical stability
        
    Returns:
        IoU, GIoU, DIoU, or CIoU
    """
    box2 = box2.T
    
    # Get the coordinates of bounding boxes
    if x1y1x2y2:  # x1, y1, x2, y2 = box1
        b1_x1, b1_y1, b1_x2, b1_y2 = box1[0], box1[1], box1[2], box1[3]
        b2_x1, b2_y1, b2_x2, b2_y2 = box2[0], box2[1], box2[2], box2[3]
    else:  # transform from xywh to xyxy
        b1_x1, b1_x2 = box1[0] - box1[2] / 2, box1[0] + box1[2] / 2
        b1_y1, b1_y2 = box1[1] - box1[3] / 2, box1[1] + box1[3] / 2
        b2_x1, b2_x2 = box2[0] - box2[2] / 2, box2[0] + box2[2] / 2
        b2_y1, b2_y2 = box2[1] - box2[3] / 2, box2[1] + box2[3] / 2
    
    # Intersection area
    inter = (torch.min(b1_x2, b2_x2) - torch.max(b1_x1, b2_x1)).clamp(0) * \
            (torch.min(b1_y2, b2_y2) - torch.max(b1_y1, b2_y1)).clamp(0)
    
    # Union Area
    w1, h1 = b1_x2 - b1_x1, b1_y2 - b1_y1 + eps
    w2, h2 = b2_x2 - b2_x1, b2_y2 - b2_y1 + eps
    union = w1 * h1 + w2 * h2 - inter + eps
    
    iou = inter / union
    
    if GIoU or DIoU or CIoU:
        # Convex (smallest enclosing box) width
        cw = torch.max(b1_x2, b2_x2) - torch.min(b1_x1, b2_x1)
        ch = torch.max(b1_y2, b2_y2) - torch.min(b1_y1, b2_y1)
        
        if CIoU or DIoU:  # Distance or Complete IoU
            c2 = cw ** 2 + ch ** 2 + eps  # convex diagonal squared
            rho2 = ((b2_x1 + b2_x2 - b1_x1 - b1_x2) ** 2 +
                    (b2_y1 + b2_y2 - b1_y1 - b1_y2) ** 2) / 4  # center distance squared
            
            if CIoU:  # Complete IoU
                v = (4 / math.pi ** 2) * \
                    torch.pow(torch.atan(w2 / h2) - torch.atan(w1 / h1), 2)
                alpha = v / (1 - iou + v + eps)
                return iou - (rho2 / c2 + v * alpha)  # CIoU
            
            return iou - rho2 / c2  # DIoU
        
        # GIoU https://arxiv.org/pdf/1902.09630.pdf
        c_area = cw * ch + eps  # convex area
        return iou - (c_area - union) / c_area  # GIoU
    
    return iou  # IoU


def box_iou(box1, box2):
    """
    Return intersection-over-union (Jaccard index) of boxes.
    
    Args:
        box1: Box 1, Tensor[N, 4]
        box2: Box 2, Tensor[M, 4]
        
    Returns:
        iou: Tensor[N, M]
    """
    # Both sets of boxes are expected to be in (x1, y1, x2, y2) format
    
    # Get the number of boxes in each set
    n = box1.shape[0]
    m = box2.shape[0]
    
    # Compute box areas
    area1 = (box1[:, 2] - box1[:, 0]) * (box1[:, 3] - box1[:, 1])  # [N]
    area2 = (box2[:, 2] - box2[:, 0]) * (box2[:, 3] - box2[:, 1])  # [M]
    
    # Compute intersection
    # Extend dimensions to compute intersection between all pairs of boxes
    box1 = box1.unsqueeze(1).expand(n, m, 4)  # [N, M, 4]
    box2 = box2.unsqueeze(0).expand(n, m, 4)  # [N, M, 4]
    
    # Compute coordinates of intersection box
    left = torch.max(box1[..., 0], box2[..., 0])  # [N, M]
    top = torch.max(box1[..., 1], box2[..., 1])  # [N, M]
    right = torch.min(box1[..., 2], box2[..., 2])  # [N, M]
    bottom = torch.min(box1[..., 3], box2[..., 3])  # [N, M]
    
    # Compute intersection area
    width = (right - left).clamp(min=0)  # [N, M]
    height = (bottom - top).clamp(min=0)  # [N, M]
    inter = width * height  # [N, M]
    
    # Compute union = area1 + area2 - inter
    union = area1.unsqueeze(1) + area2.unsqueeze(0) - inter  # [N, M]
    
    # Compute IoU
    iou = inter / union  # [N, M]
    
    return iou


class ComputeLoss:
    """
    Compute losses for YOLO.
    
    Args:
        model: YOLO model
        lambda_cls: weight for classification loss
        lambda_obj: weight for objectness loss
        lambda_box: weight for box regression loss
        box_loss_type: Type of box loss ('iou', 'giou', 'diou', 'ciou')
        cls_loss_type: Type of classification loss ('bce', 'focal', 'varifocal')
        focal_gamma: Gamma for focal loss
        focal_alpha: Alpha for focal loss
        use_fl_aux: Whether to use auxiliary focal loss
    """
    def __init__(
        self, 
        model, 
        lambda_cls=1.0, 
        lambda_obj=1.0, 
        lambda_box=0.05,
        box_loss_type='ciou',
        cls_loss_type='bce',
        focal_gamma=2.0,
        focal_alpha=0.25,
        use_fl_aux=False
    ):
        super().__init__()
        self.model = model
        self.lambda_cls = lambda_cls
        self.lambda_obj = lambda_obj
        self.lambda_box = lambda_box
        self.box_loss_type = box_loss_type
        self.cls_loss_type = cls_loss_type
        self.focal_gamma = focal_gamma
        self.focal_alpha = focal_alpha
        self.use_fl_aux = use_fl_aux
        
        # Define loss functions
        if cls_loss_type == 'bce':
            self.cls_loss = nn.BCEWithLogitsLoss(reduction='mean')
        elif cls_loss_type == 'focal':
            self.cls_loss = FocalLoss(alpha=focal_alpha, gamma=focal_gamma)
        elif cls_loss_type == 'varifocal':
            self.cls_loss = VarifocalLoss(alpha=focal_alpha, gamma=focal_gamma)
        else:
            raise ValueError(f"Unsupported classification loss: {cls_loss_type}")
        
        # Objectness loss is always BCE
        self.obj_loss = nn.BCEWithLogitsLoss(reduction='mean')
        
        # Get class and box weights if needed
        self.cls_pw = 1.0  # class positive weights
        self.obj_pw = 1.0  # obj positive weights
        
        # Set balance weights for different detection layers
        self.balance = [4.0, 1.0, 0.4] if len(model.head.heads) == 3 else [4.0, 1.0, 0.25, 0.06, 0.02]
    
    def __call__(self, outputs, targets):
        """
        Compute YOLO loss.
        
        Args:
            outputs: Model outputs, list of [batch_size, anchors, grid_h*grid_w, classes+5]
            targets: Target boxes and classes [num_targets, 6] where each row is [batch_idx, class, x, y, w, h]
            
        Returns:
            Total loss, loss items (box, obj, cls)
        """
        # Get device from outputs
        device = outputs[0].device
        loss = torch.zeros(3, device=device)  # box, obj, cls
        
        # Process targets to match model output format
        tcls, tbox, indices, anchors = self.build_targets(outputs, targets)
        
        # Flag to check if we found any valid targets
        found_targets = False
        
        # Calculate losses for each layer
        for i, (output, tcls_i, tbox_i, indices_i, anchors_i) in enumerate(zip(outputs, tcls, tbox, indices, anchors)):
            # Skip if no targets for this layer
            if indices_i.shape[0] == 0:
                continue
                
            found_targets = True
            b, a, gj, gi = indices_i  # image, anchor, gridy, gridx
            tobj = torch.zeros_like(output[..., 4])  # target obj
            
            n = b.shape[0]  # number of targets
            if n:
                # Extract predictions matching targets
                ps = output[b, a, gj, gi]  # prediction subset corresponding to targets
                
                # Regression loss
                pxy = ps[:, :2].sigmoid() * 2 - 0.5
                pwh = (ps[:, 2:4].sigmoid() * 2) ** 2 * anchors_i
                pbox = torch.cat((pxy, pwh), 1)  # predicted box
                
                # Calculate IoU loss
                if self.box_loss_type == 'iou':
                    iou = bbox_iou(pbox.T, tbox_i, x1y1x2y2=False)
                elif self.box_loss_type == 'giou':
                    iou = bbox_iou(pbox.T, tbox_i, x1y1x2y2=False, GIoU=True)
                elif self.box_loss_type == 'diou':
                    iou = bbox_iou(pbox.T, tbox_i, x1y1x2y2=False, DIoU=True)
                elif self.box_loss_type == 'ciou':
                    iou = bbox_iou(pbox.T, tbox_i, x1y1x2y2=False, CIoU=True)
                else:
                    raise ValueError(f"Unsupported box loss type: {self.box_loss_type}")
                
                # Box loss
                box_loss = (1.0 - iou).mean()
                
                # Objectness loss
                tobj[b, a, gj, gi] = iou.detach().clamp(0).type(tobj.dtype)  # iou ratio
                
                # Classification loss
                if self.model.num_classes > 1:  # cls loss (only if multiple classes)
                    t = torch.zeros_like(ps[:, 5:])  # targets
                    t[range(n), tcls_i] = 1.0
                    cls_loss = self.cls_loss(ps[:, 5:], t)
                else:
                    cls_loss = torch.tensor(0.0, device=device)
            else:
                box_loss = torch.tensor(0.0, device=device)
                cls_loss = torch.tensor(0.0, device=device)
            
            # Apply balance and lambda weights
            obj_loss = self.obj_loss(output[..., 4], tobj) * self.balance[i]
            
            # Sum all losses
            loss[0] += box_loss * self.lambda_box
            loss[1] += obj_loss * self.lambda_obj
            loss[2] += cls_loss * self.lambda_cls
        
        # If no targets were found, create a dummy loss based on the model outputs
        if not found_targets:
            # Create a dummy loss using the first output
            dummy_loss = outputs[0].sum() * 0.0
            # Return a small constant loss to avoid NaN gradients
            return dummy_loss + 0.1, {"box": 0.0, "obj": 0.0, "cls": 0.0}
        
        # Total loss
        total_loss = loss.sum()
        loss_items = {
            "box": loss[0].detach().item(),
            "obj": loss[1].detach().item(),
            "cls": loss[2].detach().item()
        }
        
        return total_loss, loss_items
    
    def build_targets(self, outputs, targets):
        """
        Build target arrays for computing loss.
        
        Args:
            outputs: Model outputs, list of [batch_size, anchors, grid_h*grid_w, classes+5]
            targets: Target boxes and classes [num_targets, 6] where each row is [batch_idx, class, x, y, w, h]
            
        Returns:
            tcls, tbox, indices, anch
        """
        na, nt = len(outputs), targets.shape[0]  # number of anchors, targets
        tcls, tbox, indices, anch = [], [], [], []
        
        # Initialize lists for each anchor
        for i in range(na):
            tcls.append([])
            tbox.append([])
            indices.append([])
            anch.append([])
        
        # Process targets
        for i in range(nt):
            if targets.sum(1)[i] == 0:  # Skip padded targets
                continue
            
            # Get batch index, class, box coordinates
            b, cls, tx, ty, tw, th = targets[i]
            b = b.long()
            
            # Create targets for all anchors in all layers
            for j, output in enumerate(outputs):
                batch_size, _, grid_h, grid_w = output.shape
                
                # Get grid coordinates
                gx, gy = tx * grid_w, ty * grid_h
                gi, gj = int(gx), int(gy)
                
                # Constrain to grid
                gi, gj = gi.clamp(0, grid_w - 1), gj.clamp(0, grid_h - 1)
                
                # Box coordinates
                tx, ty = tx * grid_w - gi, ty * grid_h - gj
                tw, th = tw, th
                
                # Append data
                tcls[j].append(cls)
                tbox[j].append(torch.stack([tx, ty, tw, th]))
                indices[j].append(torch.stack([b, 0, gj, gi]))  # img, anchor, grid indices
                anch[j].append(torch.tensor([1.0, 1.0], device=targets.device))
        
        # Convert lists to tensors
        for i in range(na):
            if len(tcls[i]) > 0:
                tcls[i] = torch.cat(tcls[i])
                tbox[i] = torch.cat(tbox[i])
                indices[i] = torch.cat(indices[i], 0)
                anch[i] = torch.cat(anch[i])
            else:
                tcls[i] = torch.zeros(0, device=targets.device, dtype=torch.long)
                tbox[i] = torch.zeros(0, 4, device=targets.device)
                indices[i] = torch.zeros(0, 4, device=targets.device, dtype=torch.long)
                anch[i] = torch.zeros(0, 2, device=targets.device)
        
        return tcls, tbox, indices, anch 