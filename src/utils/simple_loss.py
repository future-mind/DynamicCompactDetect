import torch
import torch.nn as nn
import torch.nn.functional as F


class SimpleLoss(nn.Module):
    """
    A simple loss function for object detection that works with our minimal model.
    """
    def __init__(self):
        super().__init__()
        self.mse = nn.MSELoss()
        self.bce = nn.BCEWithLogitsLoss()
        
        # Loss weights
        self.cls_weight = 1.0
        self.obj_weight = 1.0
        self.box_weight = 5.0
    
    def forward(self, preds, targets, teacher_preds=None):
        """
        Calculate a simple loss for object detection.
        
        Args:
            preds: Model predictions
            targets: Ground truth targets
            teacher_preds: Not used in this simple loss
            
        Returns:
            Tensor: Loss value
        """
        # For our minimal model, we just return a dummy loss
        # that allows training to proceed
        batch_size = preds.size(0)
        
        # Create a dummy target of the same shape as preds
        dummy_target = torch.zeros_like(preds)
        
        # Calculate a simple MSE loss
        loss = self.mse(preds, dummy_target)
        
        return loss 