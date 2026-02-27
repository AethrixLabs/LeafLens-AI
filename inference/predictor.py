"""Inference helper for single-image classification.

Provides a reusable `Predictor` class that loads a PyTorch EfficientNet model
and performs single-image inference using the validation transform from
`preprocessing.image_transforms`.

Key features:
- Load model from a checkpoint path (supports full model or state_dict).
- Load labels from a JSON file (list or dict form supported).
- Accept image file path or raw image bytes for prediction.
- Returns predicted class name and softmax confidence score.

Designed for production use: clear errors, device handling, and small API.
"""
from __future__ import annotations

import io
import json
import logging
from typing import Any, Dict, Optional, Union

from PIL import Image
import torch
import torch.nn as nn
import torch.nn.functional as F

from preprocessing.image_transforms import preprocess_image, get_val_transform

# Try importing the project's model factory. If not available, fall back to torchvision.
try:
    from models.efficientnet_model import create_model
except Exception:
    create_model = None  # type: ignore

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


class Predictor:
    """Simple predictor for single-image classification.

    Example:
        p = Predictor(model_path="models/rice_model.pt", labels_path="config/labels.json")
        label, conf = p.predict_from_file("/path/to/image.jpg")
    """

    def __init__(
        self,
        model_path: str,
        labels_path: str,
        device: Optional[Union[str, torch.device]] = None,
        model_variant: str = "b0",
    ) -> None:
        self.model_path = model_path
        self.labels_path = labels_path
        self.device = torch.device("cuda" if torch.cuda.is_available() and device in (None, "cuda") else (device or "cpu"))
        self.model_variant = model_variant

        self.labels = self._load_labels(self.labels_path)
        self.num_classes = len(self.labels)

        self.model = self._load_model(self.model_path)
        self.model.eval()

    def _load_labels(self, path: str) -> list:
        """Load labels from JSON file.
        
        Supports multiple formats:
        - List of class names: ["class1", "class2", ...]
        - Dict mapping indices to names: {"0": "class1", "1": "class2", ...}
        - Dict mapping string keys to names: {"class1_key": "class1", ...}
        
        Args:
            path: Path to labels JSON file
            
        Returns:
            Sorted list of class labels
            
        Raises:
            FileNotFoundError: If labels file not found
            json.JSONDecodeError: If JSON is invalid
            TypeError: If content is not a list or dict
            ValueError: If labels list is empty
        """
        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
        except FileNotFoundError:
            logger.exception("Labels file not found: %s", path)
            raise
        except json.JSONDecodeError:
            logger.exception("Invalid JSON in labels file: %s", path)
            raise

        # Support either a list of names or a dict mapping indices/keys to names
        if isinstance(data, list):
            labels = data
        elif isinstance(data, dict):
            # Sort by key if keys look like indices, otherwise use insertion order
            try:
                # keys may be strings of ints (like "0", "1", etc.)
                sorted_items = sorted(data.items(), key=lambda kv: int(kv[0]))
                labels = [v for _, v in sorted_items]
            except (ValueError, TypeError):
                # keys are not numeric strings, use insertion order
                labels = list(data.values())
        else:
            raise TypeError("labels.json must contain a list or dict of class names")

        if not labels:
            raise ValueError("No labels found in labels file")

        logger.info(f"Loaded {len(labels)} class labels from {path}")
        return labels

    def _load_model(self, path: str) -> nn.Module:
        """Load model from checkpoint with robust format detection.
        
        Supports multiple checkpoint formats:
        1. Full nn.Module instance saved directly
        2. Checkpoint dict with 'model_state_dict' key
        3. Checkpoint dict with 'state_dict' key  
        4. Raw state_dict dict
        
        Dynamically rebuilds EfficientNet-B0 based on num_classes and loads
        weights from state_dict.
        
        Args:
            path: Path to model checkpoint (.pt or .pth file)
            
        Returns:
            PyTorch model on the configured device in eval mode
            
        Raises:
            FileNotFoundError: If model file not found
            RuntimeError: If model loading or instantiation fails
        """
        # Load checkpoint (be permissive — support full model, state_dict, or wrapped dict)
        try:
            checkpoint = torch.load(path, map_location=self.device)
            logger.info(f"Loaded checkpoint from {path} (device: {self.device})")
        except FileNotFoundError:
            logger.exception("Model file not found: %s", path)
            raise
        except Exception:
            logger.exception("Failed to load model checkpoint: %s", path)
            raise

        # Case 1: checkpoint is already an nn.Module (full model saved directly)
        if isinstance(checkpoint, nn.Module):
            model = checkpoint.to(self.device)
            logger.info(f"Loaded model as nn.Module instance")
            return model

        # Case 2 & 3: checkpoint is a dict with state_dict or model_state_dict keys
        # or raw state_dict
        state_dict = None
        idx_to_class = None
        
        if isinstance(checkpoint, dict):
            # Try common keys for state_dict
            for key in ("model_state_dict", "state_dict", "model"):
                if key in checkpoint:
                    state_dict = checkpoint[key]
                    logger.info(f"Found state_dict under key '{key}'")
                    break
            
            # Try to extract class mapping if available
            for class_key in ("idx_to_class", "class_names", "classes"):
                if class_key in checkpoint:
                    idx_to_class = checkpoint[class_key]
                    logger.info(f"Found class mapping under key '{class_key}'")
                    break
            
            # If no state_dict found, assume the entire dict is the state_dict
            if state_dict is None:
                state_dict = checkpoint
                logger.info("Treating entire checkpoint dict as state_dict")
        else:
            # Checkpoint is not a dict, treat it as state_dict
            state_dict = checkpoint

        # If we still don't have a state_dict, fail
        if state_dict is None:
            raise RuntimeError("Could not extract state_dict from checkpoint")

        # Use the project's create_model factory to construct model with correct num_classes
        if not create_model:
            raise RuntimeError(
                "Model factory (create_model) not available. "
                "Cannot construct model architecture from state_dict. "
                "Ensure models.efficientnet_model module is importable."
            )

        try:
            model = create_model(
                num_classes=self.num_classes,
                variant=self.model_variant,
                pretrained=False,
                device=str(self.device)
            )
            logger.info(f"Created EfficientNet-{self.model_variant.upper()} model with {self.num_classes} classes")
        except Exception:
            logger.exception("Failed to create model architecture")
            raise RuntimeError("Cannot instantiate model architecture")

        # Attempt to load state_dict into the model
        try:
            # Remove potential 'module.' prefixes from DataParallel models
            if isinstance(state_dict, dict):
                new_state = {}
                for k, v in state_dict.items():
                    new_key = k
                    if k.startswith("module."):
                        new_key = k[len("module."):]
                    new_state[new_key] = v
                model.load_state_dict(new_state, strict=False)
                logger.info("Loaded state_dict with prefix removal (DataParallel compatibility)")
            else:
                model.load_state_dict(state_dict)  # type: ignore
                logger.info("Loaded state_dict")
        except Exception:
            logger.exception("Failed to load state_dict into model, attempting non-strict load")
            try:
                model.load_state_dict(state_dict, strict=False)  # type: ignore
                logger.info("Loaded state_dict (non-strict mode)")
            except Exception:
                logger.exception("Non-strict state_dict load also failed")
                raise RuntimeError("Failed to load model weights from state_dict")

        return model.to(self.device)

    def _prepare_tensor(self, image: Union[str, bytes, Image.Image]) -> torch.Tensor:
        # Accept bytes, file path, or PIL image. Use preprocess_image helper which returns
        # a batched tensor on the requested device.
        if isinstance(image, bytes):
            try:
                img = Image.open(io.BytesIO(image)).convert("RGB")
            except Exception:
                logger.exception("Invalid image bytes provided")
                raise
            return preprocess_image(img, input_size=224, device=self.device)

        if isinstance(image, Image.Image):
            return preprocess_image(image, input_size=224, device=self.device)

        if isinstance(image, str):
            # Let preprocess_image open the path
            return preprocess_image(image, input_size=224, device=self.device)

        raise TypeError("image must be a file path (str), bytes, or PIL.Image.Image")

    def predict(self, image: Union[str, bytes, Image.Image]) -> Dict[str, Any]:
        """Run inference for a single image and return a JSON-serializable dict.

        Args:
            image: file path, raw image bytes, or PIL.Image

        Returns:
            dict: {"label": str, "confidence": float}
        """
        tensor = self._prepare_tensor(image)

        with torch.no_grad():
            outputs = self.model(tensor)

            # Some models return (logits,) or (logits, aux). Handle both.
            if isinstance(outputs, (list, tuple)):
                logits = outputs[0]
            else:
                logits = outputs

            if not isinstance(logits, torch.Tensor):
                raise TypeError("Model output is not a tensor")

            probs = F.softmax(logits, dim=1)
            top_prob, top_idx = torch.max(probs, dim=1)

            idx = int(top_idx.item())
            prob = float(top_prob.item())

            try:
                label = self.labels[idx]
            except Exception:
                logger.exception("Predicted index %s out of range for labels (len=%d)", idx, len(self.labels))
                raise IndexError("Predicted index out of range for labels")

            return {"label": label, "confidence": prob}

    # Convenience wrappers
    def predict_from_file(self, image_path: str) -> Dict[str, Any]:
        return self.predict(image_path)

    def predict_from_bytes(self, image_bytes: bytes) -> Dict[str, Any]:
        return self.predict(image_bytes)


if __name__ == "__main__":
    # Quick CLI for manual testing (not a replacement for production usage)
    import argparse

    parser = argparse.ArgumentParser(description="Run single-image inference with a trained EfficientNet model")
    parser.add_argument("image", help="Path to image file to classify")
    parser.add_argument("--model", default="models/rice_model.pt", help="Path to model checkpoint")
    parser.add_argument("--labels", default="config/labels.json", help="Path to labels JSON file")
    parser.add_argument("--device", default=None, help="Device (cpu or cuda)")
    args = parser.parse_args()

    p = Predictor(model_path=args.model, labels_path=args.labels, device=args.device)
    result = p.predict_from_file(args.image)
    print(json.dumps(result))
