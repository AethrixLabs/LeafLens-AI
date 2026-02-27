"""Production-ready model registry for multi-crop disease detection.

This module provides a centralized registry for loading and managing
multiple crop-specific models (Predictor + GradCAM) at application startup.
Ensures efficient model loading with proper device handling (CUDA/CPU),
caching to avoid reloading, and dynamic crop selection.

Features:
- Lazy model loading (models load only when accessed)
- Caching mechanism (each model loads only once)
- Dynamic crop configuration via CROP_CONFIG
- Multi-device support (CUDA/CPU auto-detection)
- Production-ready error handling and logging
- Scalable architecture for future crops

Usage:
    registry = ModelRegistry()
    registry.load_models()
    rice_models = registry.get("rice")
    prediction = rice_models["predictor"].predict_from_file("image.jpg")
    
    # List available crops
    crops = registry.list_available_crops()
    
    # Check if a crop is loaded
    is_loaded = registry.is_loaded("wheat")
"""
import logging
from typing import Dict, Any, Optional
from pathlib import Path

import torch

from inference.predictor import Predictor
from explainability.gradcam import GradCAM

# Configure logger
logger = logging.getLogger(__name__)


# Crop-specific model configuration
# Each crop defines its model checkpoint and labels configuration
# This is the central registry configuration - add new crops here to extend the system
CROP_CONFIG: Dict[str, Dict[str, str]] = {
    "rice": {
        "model_path": "models/rice_model.pt",
        "labels_path": "config/labels.json",
    },
    "wheat": {
        "model_path": "training/checkpoints/wheat/wheat_model.pth",
        "labels_path": "training/checkpoints/wheat/class_names.json",
    },
    "corn": {
        "model_path": "training/checkpoints/corn/best_model.pth",
        "labels_path": "training/checkpoints/corn/class_names.json",
    },
}

# Flag to validate checkpoint files exist before loading (development/production)
VALIDATE_CHECKPOINT_PATHS: bool = True


class ModelRegistry:
    """
    Production-ready registry for loading and managing multi-crop models.

    Loads Predictor and GradCAM instances for each crop at startup and
    provides a clean interface to retrieve model bundles by crop name.
    
    Key Features:
    - **Caching**: Each model loads only once and is cached for reuse
    - **Device Handling**: Auto-detects CUDA/CPU and places models appropriately
    - **Dynamic Loading**: Supports any crop defined in CROP_CONFIG
    - **Error Handling**: Comprehensive error logging and graceful failure
    - **Scalability**: Easy to add new crops by updating CROP_CONFIG
    
    The registry returns model bundles containing:
    - "predictor": Predictor instance for inference
    - "gradcam": GradCAM instance for visual explanations
    - "device": Computation device used
    - "config": Checkpoint and labels configuration

    Attributes:
        device: Computation device (torch.device instance).
        models: Dictionary mapping crop names to their model bundles
               (Predictor + GradCAM instances).

    Example:
        registry = ModelRegistry()
        registry.load_models()
        rice_bundle = registry.get("rice")
        # rice_bundle = {
        #     "predictor": Predictor(...),
        #     "gradcam": GradCAM(...),
        #     "device": torch.device("cuda"),
        #     "config": {...}
        # }
        
        # Make a prediction
        prediction = rice_bundle["predictor"].predict_from_file("image.jpg")
        
        # Generate explanation
        heatmap, class_name = rice_bundle["gradcam"].generate(tensor, target_class=0)
    """

    def __init__(self) -> None:
        """Initialize registry with device detection and empty models store.
        
        Auto-detects available computation device (CUDA if available, else CPU)
        and initializes empty model cache.
        """
        self.device = self._detect_device()
        self.models: Dict[str, Dict[str, Any]] = {}
        logger.info(f"ModelRegistry initialized on device: {self.device}")

    def _detect_device(self) -> torch.device:
        """
        Detect available computation device.

        Returns:
            torch.device: CUDA device if available, otherwise CPU.
        """
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"Device detection: {device}")
        if torch.cuda.is_available():
            try:
                logger.info(
                    f"CUDA available - GPU: {torch.cuda.get_device_name(0)}, "
                    f"Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB"
                )
            except Exception:
                logger.info("CUDA available but couldn't query device properties")
        return device
    
    def _validate_checkpoint_paths(self, crop: str, config: Dict[str, str]) -> bool:
        """
        Validate that checkpoint and labels files exist.
        
        Args:
            crop: Crop name (for logging)
            config: Configuration dict with model_path and labels_path
            
        Returns:
            True if paths are valid, False otherwise
        """
        if not VALIDATE_CHECKPOINT_PATHS:
            return True
            
        model_path = Path(config["model_path"])
        labels_path = Path(config["labels_path"])
        
        errors = []
        if not model_path.exists():
            errors.append(f"Model not found: {model_path.absolute()}")
        if not labels_path.exists():
            errors.append(f"Labels not found: {labels_path.absolute()}")
            
        if errors:
            for error in errors:
                logger.error(f"  ✗ {error}")
            return False
            
        return True

    def load_models(self) -> None:
        """
        Load all crops' models (Predictor + GradCAM) from CROP_CONFIG.

        For each crop in CROP_CONFIG:
        1. Validate checkpoint and labels files exist (if validation enabled)
        2. Instantiate a Predictor with model and labels paths
        3. Extract the underlying PyTorch model from Predictor
        4. Instantiate a GradCAM instance for explainability
        5. Store both in self.models[crop] along with metadata

        Model bundles are cached after loading, so subsequent calls to load_models()
        will skip reloading if models are already loaded.

        Raises:
            FileNotFoundError: If required model or labels files not found
            RuntimeError: If model loading or initialization fails
            ValueError: If crop configuration is invalid

        Example:
            registry = ModelRegistry()
            registry.load_models()  # Loads all crops in CROP_CONFIG
            # Subsequent calls skip reloading:
            registry.load_models()  # No-op if already loaded
        """
        if self.models:
            logger.warning("Models already loaded. Skipping reload.")
            return

        logger.info(f"Loading models for {len(CROP_CONFIG)} crop(s)...")
        logger.info(f"Available crops: {list(CROP_CONFIG.keys())}")

        for crop_name, config in CROP_CONFIG.items():
            try:
                logger.info(f"\n{'='*60}")
                logger.info(f"Loading {crop_name.upper()} model...")
                logger.info(f"{'='*60}")
                
                # Validate checkpoint paths
                if not self._validate_checkpoint_paths(crop_name, config):
                    raise FileNotFoundError(
                        f"Invalid checkpoint paths for {crop_name}. "
                        f"See logs above for details."
                    )
                
                model_path = config["model_path"]
                labels_path = config["labels_path"]

                logger.info(f"  Model checkpoint: {model_path}")
                logger.info(f"  Labels file:      {labels_path}")

                # Create Predictor instance
                # The Predictor handles:
                # 1. Loading labels from JSON
                # 2. Creating EfficientNet-B0 with correct num_classes
                # 3. Loading model weights from checkpoint
                # 4. Setting up device and eval mode
                predictor = Predictor(
                    model_path=model_path,
                    labels_path=labels_path,
                    device=self.device,
                )
                logger.info(
                    f"  ✓ {crop_name} Predictor loaded "
                    f"({predictor.num_classes} classes, {len(predictor.labels)} labels)"
                )

                # Extract model from Predictor for GradCAM
                underlying_model = predictor.model

                # Create GradCAM instance
                # GradCAM auto-detects the target layer and handles device placement
                gradcam = GradCAM(
                    model=underlying_model,
                    target_layer=None,  # Auto-detect
                    device=str(self.device),
                )
                logger.info(f"  ✓ {crop_name} GradCAM initialized")

                # Store model bundle with metadata
                self.models[crop_name] = {
                    "predictor": predictor,
                    "gradcam": gradcam,
                    "device": str(self.device),
                    "config": config,
                    "class_labels": predictor.labels,
                    "num_classes": predictor.num_classes,
                }

                logger.info(
                    f"  ✓ {crop_name} model bundle loaded successfully\n"
                )

            except FileNotFoundError as e:
                logger.error(f"✗ File not found for {crop_name}: {e}")
                raise
            except Exception as e:
                logger.error(f"✗ Failed to load {crop_name} model: {e}", exc_info=True)
                raise

        logger.info(
            f"\n{'='*60}\n"
            f"✓ All models loaded successfully\n"
            f"Available crops: {list(self.models.keys())}\n"
            f"{'='*60}\n"
        )

    def get(self, crop: str) -> Dict[str, Any]:
        """
        Retrieve model bundle for a specific crop.
        
        Returns a dictionary containing the Predictor, GradCAM, and metadata
        for efficient reuse without reloading.

        Args:
            crop: Crop name (e.g., "rice", "wheat").

        Returns:
            Dictionary containing:
            - "predictor": Predictor instance for inference
            - "gradcam": GradCAM instance for explainability
            - "device": Computation device used (string)
            - "config": Checkpoint and labels file paths
            - "class_labels": List of class names
            - "num_classes": Number of classes

        Raises:
            ValueError: If crop is not in the loaded models.

        Example:
            rice_bundle = registry.get("rice")
            predictor = rice_bundle["predictor"]
            gradcam = rice_bundle["gradcam"]
            class_labels = rice_bundle["class_labels"]
            
            # Make prediction
            prediction = predictor.predict_from_file("image.jpg")
            
            # Get visualization
            heatmap, class_name = gradcam.generate(tensor, target_class=0)
        """
        if crop not in self.models:
            available = list(self.models.keys())
            logger.error(
                f"Crop '{crop}' not found. Available crops: {available}"
            )
            raise ValueError(
                f"Unsupported crop: '{crop}'. "
                f"Available crops: {available}"
            )

        logger.debug(f"Retrieved model bundle for crop: {crop}")
        return self.models[crop]

    def list_available_crops(self) -> list:
        """
        Get list of currently loaded crops.

        Returns a sorted list of crop names that have been successfully loaded.

        Returns:
            List of crop names (e.g., ["rice", "wheat"]).
            
        Example:
            crops = registry.list_available_crops()
            # Returns: ["rice", "wheat"]
        """
        return sorted(list(self.models.keys()))

    def is_loaded(self, crop: Optional[str] = None) -> bool:
        """
        Check if models are loaded.

        Args:
            crop: Optional crop name. If provided, checks if that specific crop is loaded.
                 If None, checks if any models are loaded.

        Returns:
            True if models are loaded (or specific crop is loaded), False otherwise.
            
        Example:
            # Check if any models are loaded
            if registry.is_loaded():
                print("Models are ready")
            
            # Check if specific crop is loaded
            if registry.is_loaded("wheat"):
                wheat_bundle = registry.get("wheat")
        """
        if crop is None:
            return len(self.models) > 0
        return crop in self.models
    
    def get_model_info(self, crop: str) -> Dict[str, Any]:
        """
        Get metadata about a loaded model.
        
        Args:
            crop: Crop name
            
        Returns:
            Dictionary with model metadata including num_classes, device, config
            
        Raises:
            ValueError: If crop not loaded
        """
        bundle = self.get(crop)  # Raises ValueError if not found
        predictor = bundle["predictor"]
        
        return {
            "crop": crop,
            "num_classes": bundle["num_classes"],
            "class_labels": bundle["class_labels"],
            "device": bundle["device"],
            "model_path": bundle["config"]["model_path"],
            "labels_path": bundle["config"]["labels_path"],
            "model_variant": predictor.model_variant,
        }
