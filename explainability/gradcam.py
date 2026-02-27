# explainability/gradcam.py
"""
Grad-CAM implementation for interpreting EfficientNet crop disease predictions.

This module provides a production-ready Grad-CAM class for generating visual
explanations of model decisions through saliency maps that highlight the regions
contributing most to each prediction.

Classes:
    GradCAM: Main class for generating Grad-CAM visualizations.

References:
    - Selvaraju et al. (2017): "Grad-CAM: Visual Explanations from Deep Networks
      via Gradient-based Localization"
"""
import json
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import cv2
from pathlib import Path
from typing import Optional, Tuple
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class GradCAM:
    """
    Grad-CAM: Gradient-weighted Class Activation Mapping for model interpretability.
    
    Generates visual explanations of predictions by computing gradients of the
    output class with respect to intermediate feature maps. Works with any PyTorch
    model that has convolutional features.
    
    Attributes:
        model: PyTorch model instance.
        target_layer: Module reference to the layer for which to generate CAM.
        device: Computation device ('cuda' or 'cpu').
        gradients: Stored gradients from target layer.
        activations: Stored activations from target layer.
    """
    
    def __init__(
        self,
        model: nn.Module,
        target_layer: Optional[nn.Module] = None,
        device: Optional[str] = None
    ):
        """
        Initialize GradCAM.
        
        Args:
            model: PyTorch model instance (e.g., EfficientNetClassifier).
            target_layer: Target layer for gradient computation. If None,
                         automatically detects final convolutional layer
                         (backbone.features[-1]).
            device: Device to use ('cuda' or 'cpu'). If None, auto-detects.
        
        Raises:
            ValueError: If model is None or target layer cannot be detected.
        """
        if model is None:
            raise ValueError("Model cannot be None")
        
        self.model = model
        self.device = device or self._detect_device()
        
        # Detect target layer if not provided
        if target_layer is None:
            self.target_layer = self._detect_target_layer()
            if self.target_layer is None:
                raise ValueError(
                    "Could not auto-detect target layer. Ensure model has "
                    "backbone.features[-1] or provide target_layer explicitly."
                )
        else:
            self.target_layer = target_layer
        
        # Storage for activations and gradients
        self.activations = None
        self.gradients = None
        
        # Register hooks
        self._register_hooks()
        
        # Move model to device and set to eval mode
        self.model.to(self.device)
        self.model.eval()
        
        logger.info(f"GradCAM initialized on device: {self.device}")
    
    def _detect_device(self) -> str:
        """Auto-detect available device (CUDA or CPU)."""
        if torch.cuda.is_available():
            return "cuda"
        return "cpu"
    
    def _detect_target_layer(self) -> Optional[nn.Module]:
        """Auto-detect final convolutional layer (backbone.features[-1])."""
        try:
            if hasattr(self.model, "backbone") and hasattr(self.model.backbone, "features"):
                return self.model.backbone.features[-1]
        except (AttributeError, IndexError):
            pass
        return None
    
    def _register_hooks(self) -> None:
        """Register forward and backward hooks on target layer."""
        
        def forward_hook(module, input, output):
            """Hook to capture activations."""
            self.activations = output.detach()
        
        def backward_hook(module, grad_input, grad_output):
            """Hook to capture gradients."""
            self.gradients = grad_output[0].detach()
        
        self.target_layer.register_forward_hook(forward_hook)
        self.target_layer.register_full_backward_hook(backward_hook)
        
        logger.info("Hooks registered on target layer")
    
    def generate(
        self,
        input_tensor: torch.Tensor,
        target_class: Optional[int] = None
    ) -> Tuple[np.ndarray, int]:
        """
        Generate Grad-CAM heatmap for given input.
        
        Grad-CAM computation steps:
        1. Forward pass to get predictions and activations.
        2. Backward pass to compute gradients.
        3. Global average pooling of gradients.
        4. Weight activations by pooled gradients.
        5. Apply ReLU to keep only positive contributions.
        6. Normalize heatmap to [0, 1] range.
        
        Args:
            input_tensor: Input image tensor of shape (1, C, H, W),
                         values in [0, 1] or [-1, 1].
            target_class: Target class for gradient computation.
                         If None, uses predicted class.
        
        Returns:
            Tuple containing:
                - heatmap: Numpy array of shape (H, W) with values in [0, 1].
                - predicted_class: Predicted class index.
        
        Raises:
            ValueError: If input tensor has incorrect shape or device.
            RuntimeError: If hooks fail or model has unexpected structure.
        """
        # Validate input
        if input_tensor.dim() != 4:
            raise ValueError(
                f"Input tensor must have shape (1, C, H, W), "
                f"got {input_tensor.shape}"
            )
        
        if input_tensor.shape[0] != 1:
            raise ValueError(
                f"Batch size must be 1, got {input_tensor.shape[0]}"
            )
        
        # Move to device
        input_tensor = input_tensor.to(self.device).float()
        
        # Enable gradient computation
        input_tensor.requires_grad = True
        
        # Reset gradients before forward pass
        self.model.zero_grad()
        
        # Forward pass
        with torch.enable_grad():
            logits = self.model(input_tensor)
            probabilities = F.softmax(logits, dim=1)
        
        # Get predicted class if not provided
        if target_class is None:
            target_class = logits.argmax(dim=1).item()
        
        logger.info(f"Generating Grad-CAM for class: {target_class}")
        
        # Zero gradients
        self.model.zero_grad()
        
        # Backward pass
        target_score = logits[0, target_class]
        target_score.backward()
        
        # Check if gradients were captured
        if self.gradients is None or self.activations is None:
            raise RuntimeError("Failed to capture gradients or activations")
        
        # Extract and process activations and gradients
        gradients = self.gradients[0].cpu().numpy()  # Shape: (C, H, W)
        activations = self.activations[0].cpu().detach().numpy()  # Shape: (C, H, W)
        
        # Global average pooling of gradients
        weights = np.mean(gradients, axis=(1, 2))  # Shape: (C,)
        
        # Weighted combination of activation maps
        heatmap = np.zeros(activations.shape[1:], dtype=np.float32)  # Shape: (H, W)
        for i, weight in enumerate(weights):
            heatmap += weight * activations[i]
        
        # Apply ReLU to keep positive contributions
        heatmap = np.maximum(heatmap, 0)
        
        # Normalize to [0, 1]
        max_val = np.max(heatmap)
        if max_val > 0:
            heatmap = heatmap / max_val
        
        return heatmap, target_class
    
    def overlay_heatmap(
        self,
        original_image: np.ndarray,
        heatmap: np.ndarray,
        colormap: int = cv2.COLORMAP_JET,
        alpha: float = 0.4
    ) -> np.ndarray:
        """
        Overlay Grad-CAM heatmap on original image.
        
        Converts heatmap to RGB using a colormap, resizes to match original
        image dimensions, and blends with the original image.
        
        Args:
            original_image: Original image as numpy array (H, W, 3) with
                           values in [0, 255] as uint8.
            heatmap: Grad-CAM heatmap (H_heat, W_heat) with values in [0, 1].
            colormap: OpenCV colormap (default: cv2.COLORMAP_JET).
            alpha: Blending weight for heatmap (0.0-1.0).
        
        Returns:
            Overlay image as numpy array (H, W, 3) with values in [0, 255].
        
        Raises:
            ValueError: If input images have invalid shapes or types.
        """
        # Validate inputs
        if original_image is None or heatmap is None:
            raise ValueError("Original image and heatmap cannot be None")
        
        if original_image.ndim != 3 or original_image.shape[2] != 3:
            raise ValueError(
                f"Original image must have shape (H, W, 3), "
                f"got {original_image.shape}"
            )
        
        if heatmap.ndim != 2:
            raise ValueError(
                f"Heatmap must have shape (H, W), got {heatmap.shape}"
            )
        
        # Ensure original image is uint8
        if original_image.dtype != np.uint8:
            original_image = np.clip(original_image, 0, 255).astype(np.uint8)
        
        # Convert heatmap to uint8 (0-255)
        heatmap_uint8 = (heatmap * 255).astype(np.uint8)
        
        # Resize heatmap to match original image
        h_orig, w_orig = original_image.shape[:2]
        heatmap_resized = cv2.resize(
            heatmap_uint8,
            (w_orig, h_orig),
            interpolation=cv2.INTER_LINEAR
        )
        
        # Apply colormap
        heatmap_colored = cv2.applyColorMap(heatmap_resized, colormap)
        
        # Blend images
        overlay = cv2.addWeighted(
            original_image,
            1 - alpha,
            heatmap_colored,
            alpha,
            0
        )
        
        return overlay
    
    def __enter__(self):
        """Context manager entry."""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.model.train()


# ============================================================================
# CLI Testing Block
# ============================================================================

if __name__ == "__main__":
    import argparse
    import sys
    from PIL import Image
    
    def load_model_and_labels(
        model_path: str,
        labels_path: str,
        num_classes: int = 4,
        device: str = "cpu"
    ) -> Tuple[nn.Module, dict]:
        """
        Load trained model and class labels.
        
        Args:
            model_path: Path to saved model checkpoint.
            labels_path: Path to labels JSON file.
            num_classes: Number of output classes.
            device: Device to load model on.
        
        Returns:
            Tuple of (model, labels_dict).
        """
        # Import model class
        from models.efficientnet_model import EfficientNetClassifier
        
        # Initialize model
        model = EfficientNetClassifier(
            num_classes=num_classes,
            model_variant="b0",
            pretrained=False,
            device=device
        )
        
        # Load checkpoint
        checkpoint = torch.load(model_path, map_location=device)
        
        # Handle different checkpoint formats
        if isinstance(checkpoint, dict):
            if "model_state_dict" in checkpoint:
                model.load_state_dict(checkpoint["model_state_dict"])
            elif "state_dict" in checkpoint:
                model.load_state_dict(checkpoint["state_dict"])
            else:
                model.load_state_dict(checkpoint)
        else:
            model.load_state_dict(checkpoint)
        
        model = model.to(device)
        model.eval()
        
        # Load labels
        with open(labels_path, "r") as f:
            labels = json.load(f)
        
        labels_dict = {i: label for i, label in enumerate(labels)}
        
        logger.info(f"Model loaded from {model_path}")
        logger.info(f"Classes: {labels}")
        
        return model, labels_dict
    
    def load_and_preprocess_image(
        image_path: str,
        target_size: Tuple[int, int] = (224, 224)
    ) -> Tuple[torch.Tensor, np.ndarray]:
        """
        Load and preprocess image for model inference.
        
        Args:
            image_path: Path to input image.
            target_size: Target size for resizing (H, W).
        
        Returns:
            Tuple of (preprocessed_tensor, original_image_rgb).
        """
        # Load image
        image = Image.open(image_path).convert("RGB")
        original_image_pil = image.copy()
        
        # Resize
        image = image.resize(target_size, Image.Resampling.LANCZOS)
        
        # Convert to numpy and normalize
        image_np = np.array(image, dtype=np.float32) / 255.0
        
        # ImageNet normalization
        mean = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])
        image_normalized = (image_np - mean) / std
        
        # Convert to tensor (C, H, W)
        image_tensor = torch.from_numpy(image_normalized).permute(2, 0, 1)
        
        # Add batch dimension (1, C, H, W)
        image_tensor = image_tensor.unsqueeze(0)
        
        # Original image for overlay (resize to same size as model input)
        original_resized = np.array(
            original_image_pil.resize(target_size, Image.Resampling.LANCZOS),
            dtype=np.uint8
        )
        
        return image_tensor, original_resized
    
    def main():
        """Run Grad-CAM visualization from CLI."""
        parser = argparse.ArgumentParser(
            description="Generate Grad-CAM visualization for crop disease predictions"
        )
        parser.add_argument(
            "image_path",
            type=str,
            help="Path to input image"
        )
        parser.add_argument(
            "--model",
            type=str,
            default="models/rice_model.pt",
            help="Path to model checkpoint (default: models/rice_model.pt)"
        )
        parser.add_argument(
            "--labels",
            type=str,
            default="config/labels.json",
            help="Path to labels JSON (default: config/labels.json)"
        )
        parser.add_argument(
            "--output",
            type=str,
            default="gradcam_output.jpg",
            help="Path to save output overlay (default: gradcam_output.jpg)"
        )
        parser.add_argument(
            "--target-class",
            type=int,
            default=None,
            help="Target class index (if None, uses predicted class)"
        )
        parser.add_argument(
            "--device",
            type=str,
            choices=["cuda", "cpu"],
            default=None,
            help="Device to use (default: auto-detect)"
        )
        parser.add_argument(
            "--alpha",
            type=float,
            default=0.4,
            help="Blending alpha for overlay (0.0-1.0, default: 0.4)"
        )
        
        args = parser.parse_args()
        
        # Validate image path
        if not Path(args.image_path).exists():
            logger.error(f"Image file not found: {args.image_path}")
            sys.exit(1)
        
        # Validate model path
        if not Path(args.model).exists():
            logger.error(f"Model file not found: {args.model}")
            sys.exit(1)
        
        # Validate labels path
        if not Path(args.labels).exists():
            logger.error(f"Labels file not found: {args.labels}")
            sys.exit(1)
        
        # Auto-detect device
        device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"Using device: {device}")
        
        try:
            # Load model and labels
            model, labels_dict = load_model_and_labels(
                args.model,
                args.labels,
                device=device
            )
            
            # Load and preprocess image
            logger.info(f"Loading image: {args.image_path}")
            image_tensor, original_image = load_and_preprocess_image(args.image_path)
            
            # Initialize GradCAM
            gradcam = GradCAM(model, device=device)
            
            # Generate heatmap
            logger.info("Generating Grad-CAM heatmap...")
            heatmap, predicted_class = gradcam.generate(
                image_tensor,
                target_class=args.target_class
            )
            
            # Get class label
            class_label = labels_dict.get(predicted_class, f"Class {predicted_class}")
            logger.info(f"Predicted class: {predicted_class} ({class_label})")
            
            # Overlay heatmap on original image
            overlay = gradcam.overlay_heatmap(original_image, heatmap, alpha=args.alpha)
            
            # Save output
            output_path = Path(args.output)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            cv2.imwrite(str(output_path), cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR))
            logger.info(f"Overlay image saved: {args.output}")
            
        except Exception as e:
            logger.error(f"Error: {e}")
            import traceback
            traceback.print_exc()
            sys.exit(1)
    
    main()
