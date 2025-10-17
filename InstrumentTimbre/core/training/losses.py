"""
Loss functions for InstrumentTimbre training
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Any, Optional

def get_loss_function(loss_config: str | Dict[str, Any]) -> nn.Module:
    """
    Get loss function based on configuration
    
    Args:
        loss_config: Loss configuration (string name or dict with parameters)
        
    Returns:
        PyTorch loss function
    """
    if isinstance(loss_config, str):
        loss_name = loss_config.lower()
        loss_params = {}
    else:
        loss_name = loss_config.get('name', 'crossentropy').lower()
        loss_params = loss_config.get('params', {})
    
    if loss_name == 'crossentropy':
        return nn.CrossEntropyLoss(**loss_params)
    elif loss_name == 'focal':
        return FocalLoss(**loss_params)
    elif loss_name == 'label_smoothing':
        return LabelSmoothingLoss(**loss_params)
    elif loss_name == 'weighted_crossentropy':
        return WeightedCrossEntropyLoss(**loss_params)
    else:
        raise ValueError(f"Unknown loss function: {loss_name}")

class FocalLoss(nn.Module):
    """
    Focal Loss for addressing class imbalance
    """
    
    def __init__(self, alpha: float = 1.0, gamma: float = 2.0, 
                 reduction: str = 'mean'):
        """
        Initialize Focal Loss
        
        Args:
            alpha: Weighting factor for rare class
            gamma: Focusing parameter
            reduction: Reduction method
        """
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
    
    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Forward pass
        
        Args:
            inputs: Model predictions (logits)
            targets: True labels
            
        Returns:
            Focal loss
        """
        ce_loss = F.cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * ce_loss
        
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss

class LabelSmoothingLoss(nn.Module):
    """
    Label Smoothing Loss to prevent overconfidence
    """
    
    def __init__(self, num_classes: int, smoothing: float = 0.1):
        """
        Initialize Label Smoothing Loss
        
        Args:
            num_classes: Number of classes
            smoothing: Smoothing factor
        """
        super().__init__()
        self.num_classes = num_classes
        self.smoothing = smoothing
        self.confidence = 1.0 - smoothing
    
    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Forward pass
        
        Args:
            inputs: Model predictions (logits)
            targets: True labels
            
        Returns:
            Label smoothing loss
        """
        log_probs = F.log_softmax(inputs, dim=1)
        
        # Create smooth labels
        smooth_targets = torch.zeros_like(log_probs)
        smooth_targets.fill_(self.smoothing / (self.num_classes - 1))
        smooth_targets.scatter_(1, targets.unsqueeze(1), self.confidence)
        
        loss = -torch.sum(smooth_targets * log_probs, dim=1)
        return loss.mean()

class WeightedCrossEntropyLoss(nn.Module):
    """
    Weighted Cross Entropy Loss for imbalanced datasets
    """
    
    def __init__(self, weights: Optional[torch.Tensor] = None):
        """
        Initialize Weighted Cross Entropy Loss
        
        Args:
            weights: Class weights tensor
        """
        super().__init__()
        self.weights = weights
    
    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Forward pass
        
        Args:
            inputs: Model predictions (logits)
            targets: True labels
            
        Returns:
            Weighted cross entropy loss
        """
        return F.cross_entropy(inputs, targets, weight=self.weights)