# models/efficientnet_model.py

import torch
import torch.nn as nn
from torchvision import models
from typing import Optional


class EfficientNetClassifier(nn.Module):
    """
    EfficientNet-based classifier for crop disease detection.
    
    Supports multiple EfficientNet variants with customizable head,
    dropout regularization, and optional backbone freezing for transfer learning.
    """
    
    def __init__(
        self,
        num_classes: int,
        model_variant: str = "b0",
        pretrained: bool = True,
        dropout_rate: float = 0.5,
        freeze_backbone: bool = False,
        device: str = "cpu"
    ):
        """
        Initialize EfficientNet classifier.
        
        Args:
            num_classes: Number of output classes (e.g., 4 for rice diseases)
            model_variant: EfficientNet variant ('b0', 'b1', 'b2', etc.)
            pretrained: Whether to use ImageNet pre-trained weights
            dropout_rate: Dropout rate for regularization (0.0-1.0)
            freeze_backbone: Whether to freeze backbone weights during training
            device: Device to place model on ('cpu' or 'cuda')
        """
        super(EfficientNetClassifier, self).__init__()
        
        self.num_classes = num_classes
        self.model_variant = model_variant
        self.pretrained = pretrained
        self.dropout_rate = dropout_rate
        self.device = device
        
        # Load EfficientNet backbone
        self.backbone = self._load_backbone()
        
        # Get backbone output features
        in_features = self.backbone.classifier[1].in_features
        
        # Custom classifier head with dropout and batch normalization
        self.classifier = nn.Sequential(
            nn.Dropout(p=dropout_rate),
            nn.BatchNorm1d(in_features),
            nn.Linear(in_features, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout_rate),
            nn.Linear(512, num_classes)
        )
        
        # Replace backbone classifier
        self.backbone.classifier = self.classifier
        
        # Optionally freeze backbone
        if freeze_backbone:
            self.freeze_backbone()
    
    def _load_backbone(self) -> nn.Module:
        """Load EfficientNet backbone based on variant."""
        model_name = f"efficientnet_{self.model_variant}"
        
        if self.model_variant == "b0":
            backbone = models.efficientnet_b0(pretrained=self.pretrained)
        elif self.model_variant == "b1":
            backbone = models.efficientnet_b1(pretrained=self.pretrained)
        elif self.model_variant == "b2":
            backbone = models.efficientnet_b2(pretrained=self.pretrained)
        elif self.model_variant == "b3":
            backbone = models.efficientnet_b3(pretrained=self.pretrained)
        elif self.model_variant == "b4":
            backbone = models.efficientnet_b4(pretrained=self.pretrained)
        else:
            raise ValueError(f"Unsupported EfficientNet variant: {self.model_variant}")
        
        return backbone
    
    def freeze_backbone(self):
        """Freeze backbone weights to prevent updates during training."""
        for param in self.backbone.features.parameters():
            param.requires_grad = False
    
    def unfreeze_backbone(self):
        """Unfreeze backbone weights to allow training."""
        for param in self.backbone.features.parameters():
            param.requires_grad = True
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through the model."""
        return self.backbone(x)
    
    def get_model_info(self) -> dict:
        """Return model configuration and parameter counts."""
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        
        return {
            "model_variant": self.model_variant,
            "num_classes": self.num_classes,
            "pretrained": self.pretrained,
            "dropout_rate": self.dropout_rate,
            "total_parameters": total_params,
            "trainable_parameters": trainable_params,
            "device": self.device
        }


def get_efficientnet_b0(num_classes: int, pretrained: bool = True) -> EfficientNetClassifier:
    """
    Create EfficientNet-B0 model with custom classification head.
    
    (Backward compatibility function)

    Args:
        num_classes: number of output classes
        pretrained: whether to use ImageNet weights

    Returns:
        EfficientNetClassifier model
    """
    return EfficientNetClassifier(
        num_classes=num_classes,
        model_variant="b0",
        pretrained=pretrained
    )


def create_model(
    num_classes: int,
    variant: str = "b0",
    pretrained: bool = True,
    dropout: float = 0.5,
    freeze_backbone: bool = False,
    device: str = "cpu"
) -> EfficientNetClassifier:
    """
    Factory function to create EfficientNet classifier.
    
    Args:
        num_classes: Number of output classes
        variant: EfficientNet variant
        pretrained: Use ImageNet pre-trained weights
        dropout: Dropout rate for regularization
        freeze_backbone: Freeze backbone during training
        device: Device placement
        
    Returns:
        Configured EfficientNetClassifier
    """
    model = EfficientNetClassifier(
        num_classes=num_classes,
        model_variant=variant,
        pretrained=pretrained,
        dropout_rate=dropout,
        freeze_backbone=freeze_backbone,
        device=device
    )
    return model.to(device)
