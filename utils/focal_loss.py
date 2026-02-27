"""Focal Loss implementation for addressing class imbalance.

Focal Loss reduces the relative loss for well-classified examples,
focusing training on hard negative and hard positive examples.
Reference: https://arxiv.org/abs/1708.02002
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class FocalLoss(nn.Module):
    """Focal Loss for multi-class classification.
    
    Reduces the relative loss for well-classified examples and focuses on hard examples.
    Particularly effective for imbalanced datasets where one class dominates predictions.
    """
    
    def __init__(
        self,
        alpha: float = 0.25,
        gamma: float = 2.0,
        weight: torch.Tensor = None,
        reduction: str = "mean",
        ignore_index: int = -100
    ):
        """Initialize Focal Loss.
        
        Args:
            alpha: Weighting factor in [0, 1] to balance positive/negative examples.
                   Higher values weight positive examples more heavily.
            gamma: Focusing parameter (concentration degree) in [0, 5+].
                   Higher values focus more on hard examples.
                   gamma=0 degenerates to CrossEntropyLoss.
                   Typically gamma=2.0 is used.
            weight: Manual rescaling weight given to the loss of each class.
                   Useful for handling class imbalance.
            reduction: Specifies reduction to apply: 'none' | 'mean' | 'sum'
            ignore_index: Class index to ignore in loss computation (-100 means no ignore)
        """
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.weight = weight
        self.reduction = reduction
        self.ignore_index = ignore_index
    
    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """Calculate focal loss.
        
        Args:
            inputs: model outputs - shape (N, C) where N=batch, C=num_classes
            targets: ground truth class indexes - shape (N,)
            
        Returns:
            torch.Tensor: Focal loss value
        """
        # Get softmax probabilities
        p = F.softmax(inputs, dim=1)
        
        # Get class probabilities
        ce_loss = F.cross_entropy(
            inputs, targets, weight=self.weight, reduction="none"
        )
        
        # Get probabilities of the true class
        p_t = p.gather(1, targets.view(-1, 1)).squeeze(1)
        
        # Apply focal loss formula: -alpha * (1 - p_t)^gamma * log(p_t)
        focal_weight = self.alpha * (1 - p_t) ** self.gamma
        
        focal_loss = focal_weight * ce_loss
        
        # Handle ignore_index
        if self.ignore_index >= 0:
            mask = targets != self.ignore_index
            focal_loss = focal_loss * mask.float()
        
        # Apply reduction
        if self.reduction == "mean":
            return focal_loss.mean()
        elif self.reduction == "sum":
            return focal_loss.sum()
        elif self.reduction == "none":
            return focal_loss
        else:
            raise ValueError(f"Invalid reduction mode: {self.reduction}")


class WeightedCrossEntropyLoss(nn.Module):
    """Weighted Cross-Entropy Loss with enhanced weighting for class imbalance.
    
    Combines class weights with hard example scaling to better handle imbalanced datasets.
    """
    
    def __init__(
        self,
        weight: torch.Tensor = None,
        label_smoothing: float = 0.0,
        reduction: str = "mean"
    ):
        """Initialize Weighted Cross-Entropy Loss.
        
        Args:
            weight: Manual rescaling weight given to the loss of each class.
            label_smoothing: Label smoothing regularization (0.0-1.0)
            reduction: Specifies reduction to apply: 'none' | 'mean' | 'sum'
        """
        super(WeightedCrossEntropyLoss, self).__init__()
        self.weight = weight
        self.label_smoothing = label_smoothing
        self.reduction = reduction
    
    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """Calculate weighted cross-entropy loss.
        
        Args:
            inputs: model outputs - shape (N, C)
            targets: ground truth class indexes - shape (N,)
            
        Returns:
            torch.Tensor: Loss value
        """
        return F.cross_entropy(
            inputs,
            targets,
            weight=self.weight,
            label_smoothing=self.label_smoothing,
            reduction=self.reduction
        )
