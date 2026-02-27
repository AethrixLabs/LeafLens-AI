#!/usr/bin/env python3
"""
Comprehensive test suite for multi-crop model registry.

Tests:
1. Model loading for Rice and Wheat crops
2. Model caching and reuse
3. Device handling (CPU/GPU)
4. Predictor functionality
5. GradCAM compatibility
6. API integration

Usage:
    python test_multi_crop_registry.py
"""

import sys
import logging
from pathlib import Path
import json
import torch
import numpy as np
from PIL import Image
import tempfile

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from app.registry import ModelRegistry, CROP_CONFIG
from inference.predictor import Predictor
from preprocessing.image_transforms import preprocess_image

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def create_dummy_image(size: int = 224) -> Image.Image:
    """Create a dummy RGB image for testing."""
    return Image.new('RGB', (size, size), color='red')


def test_crop_config():
    """Test 1: Verify CROP_CONFIG is properly configured."""
    logger.info("=" * 60)
    logger.info("TEST 1: CROP_CONFIG Validation")
    logger.info("=" * 60)
    
    assert len(CROP_CONFIG) >= 2, "CROP_CONFIG should have at least 2 crops"
    assert "rice" in CROP_CONFIG, "Rice crop must be in CROP_CONFIG"
    assert "wheat" in CROP_CONFIG, "Wheat crop must be in CROP_CONFIG"
    
    for crop_name, config in CROP_CONFIG.items():
        assert "model_path" in config, f"{crop_name}: missing model_path"
        assert "labels_path" in config, f"{crop_name}: missing labels_path"
        logger.info(f"  ✓ {crop_name}: {config['model_path']}")
    
    logger.info("  ✓ CROP_CONFIG is valid\n")


def test_device_detection():
    """Test 2: Device detection."""
    logger.info("=" * 60)
    logger.info("TEST 2: Device Detection")
    logger.info("=" * 60)
    
    registry = ModelRegistry()
    device = registry.device
    
    logger.info(f"  Device detected: {device}")
    logger.info(f"  Device type: {device.type}")
    
    if torch.cuda.is_available():
        logger.info(f"  CUDA GPU: {torch.cuda.get_device_name(0)}")
        assert device.type == "cuda", "Should detect CUDA"
    else:
        logger.info("  CUDA not available, using CPU")
        assert device.type == "cpu", "Should fall back to CPU"
    
    logger.info("  ✓ Device detection working\n")


def test_checkpoint_files_exist():
    """Test 3: Verify checkpoint files exist."""
    logger.info("=" * 60)
    logger.info("TEST 3: Checkpoint Files Validation")
    logger.info("=" * 60)
    
    for crop_name, config in CROP_CONFIG.items():
        model_path = Path(config["model_path"])
        labels_path = Path(config["labels_path"])
        
        logger.info(f"\n  {crop_name.upper()}:")
        
        if model_path.exists():
            logger.info(f"    ✓ Model exists: {model_path}")
            logger.info(f"      Size: {model_path.stat().st_size / 1e6:.1f} MB")
        else:
            logger.error(f"    ✗ Model NOT found: {model_path.absolute()}")
        
        if labels_path.exists():
            logger.info(f"    ✓ Labels exist: {labels_path}")
            with open(labels_path) as f:
                labels = json.load(f)
            logger.info(f"      Classes: {labels}")
        else:
            logger.error(f"    ✗ Labels NOT found: {labels_path.absolute()}")
    
    logger.info("")


def test_labels_loading():
    """Test 4: Load labels from JSON files."""
    logger.info("=" * 60)
    logger.info("TEST 4: Labels Loading")
    logger.info("=" * 60)
    
    for crop_name, config in CROP_CONFIG.items():
        labels_path = Path(config["labels_path"])
        
        if not labels_path.exists():
            logger.warning(f"  Skipping {crop_name} (labels file not found)")
            continue
        
        logger.info(f"\n  {crop_name.upper()}:")
        
        with open(labels_path) as f:
            data = json.load(f)
        
        logger.info(f"    Raw data type: {type(data).__name__}")
        
        if isinstance(data, list):
            logger.info(f"    Format: List of {len(data)} classes")
            logger.info(f"    Classes: {data}")
        elif isinstance(data, dict):
            logger.info(f"    Format: Dict with {len(data)} entries")
            logger.info(f"    Classes: {list(data.values())}")
    
    logger.info("")


def test_model_loading():
    """Test 5: Load individual models using Predictor."""
    logger.info("=" * 60)
    logger.info("TEST 5: Model Loading (Individual)")
    logger.info("=" * 60)
    
    for crop_name, config in CROP_CONFIG.items():
        model_path = Path(config["model_path"])
        labels_path = Path(config["labels_path"])
        
        if not model_path.exists() or not labels_path.exists():
            logger.warning(f"  Skipping {crop_name} (files not found)")
            continue
        
        logger.info(f"\n  Loading {crop_name.upper()}...")
        
        try:
            predictor = Predictor(
                model_path=str(model_path),
                labels_path=str(labels_path),
            )
            logger.info(f"    ✓ Model loaded")
            logger.info(f"    ✓ Classes: {predictor.num_classes}")
            logger.info(f"    ✓ Labels: {predictor.labels}")
            logger.info(f"    ✓ Device: {predictor.device}")
            logger.info(f"    ✓ Model type: {type(predictor.model).__name__}")
            
        except Exception as e:
            logger.error(f"    ✗ Failed to load {crop_name}: {e}", exc_info=True)
    
    logger.info("")


def test_registry_loading():
    """Test 6: Load all models via ModelRegistry."""
    logger.info("=" * 60)
    logger.info("TEST 6: ModelRegistry Loading")
    logger.info("=" * 60)
    
    registry = ModelRegistry()
    
    try:
        registry.load_models()
        logger.info(f"  ✓ Registry initialization successful")
    except Exception as e:
        logger.error(f"  ✗ Registry loading failed: {e}", exc_info=True)
        return False
    
    logger.info(f"  Available crops: {registry.list_available_crops()}")
    
    return True


def test_registry_functionality():
    """Test 7: Registry get, is_loaded, list methods."""
    logger.info("=" * 60)
    logger.info("TEST 7: Registry Functionality")
    logger.info("=" * 60)
    
    registry = ModelRegistry()
    
    try:
        registry.load_models()
    except Exception as e:
        logger.warning(f"  Skipping (model loading failed): {e}")
        return False
    
    # Test list_available_crops
    crops = registry.list_available_crops()
    logger.info(f"  Available crops: {crops}")
    assert len(crops) >= 1, "Should have at least 1 crop loaded"
    
    # Test is_loaded with no args (any model loaded)
    assert registry.is_loaded(), "Should return True if any model loaded"
    logger.info("  ✓ is_loaded() = True")
    
    # Test is_loaded with invalid crop
    assert not registry.is_loaded("invalid_crop"), "Should return False for invalid crop"
    logger.info("  ✓ is_loaded('invalid_crop') = False")
    
    # Test get for each loaded crop
    for crop in crops:
        assert registry.is_loaded(crop), f"Should return True for loaded crop: {crop}"
        logger.info(f"  ✓ is_loaded('{crop}') = True")
        
        bundle = registry.get(crop)
        assert "predictor" in bundle, f"{crop}: missing predictor in bundle"
        assert "gradcam" in bundle, f"{crop}: missing gradcam in bundle"
        assert "device" in bundle, f"{crop}: missing device in bundle"
        assert "class_labels" in bundle, f"{crop}: missing class_labels in bundle"
        logger.info(f"  ✓ get('{crop}') returned complete bundle")
    
    # Test get with invalid crop
    try:
        registry.get("invalid_crop")
        assert False, "Should raise ValueError for invalid crop"
    except ValueError as e:
        logger.info(f"  ✓ get('invalid_crop') raised ValueError: {e}")
    
    logger.info("")

def test_prediction():
    """Test 8: Make predictions with both models."""
    logger.info("=" * 60)
    logger.info("TEST 8: Prediction")
    logger.info("=" * 60)
    
    registry = ModelRegistry()
    
    try:
        registry.load_models()
    except Exception as e:
        logger.warning(f"  Skipping (model loading failed): {e}")
        return False
    
    # Create a dummy image
    dummy_img = create_dummy_image()
    
    for crop in registry.list_available_crops():
        logger.info(f"\n  {crop.upper()}:")
        
        try:
            bundle = registry.get(crop)
            predictor = bundle["predictor"]
            
            # Make prediction
            result = predictor.predict(dummy_img)
            
            assert "label" in result, "Missing 'label' in result"
            assert "confidence" in result, "Missing 'confidence' in result"
            
            logger.info(f"    ✓ Prediction successful")
            logger.info(f"    ✓ Label: {result['label']}")
            logger.info(f"    ✓ Confidence: {result['confidence']:.4f}")
            
        except Exception as e:
            logger.error(f"    ✗ Prediction failed: {e}", exc_info=True)
    
    logger.info("")


def test_gradcam_compatibility():
    """Test 9: GradCAM compatibility."""
    logger.info("=" * 60)
    logger.info("TEST 9: GradCAM Compatibility")
    logger.info("=" * 60)
    
    registry = ModelRegistry()
    
    try:
        registry.load_models()
    except Exception as e:
        logger.warning(f"  Skipping (model loading failed): {e}")
        return False
    
    # Create a dummy image and tensor
    dummy_img = create_dummy_image()
    
    for crop in registry.list_available_crops():
        logger.info(f"\n  {crop.upper()}:")
        
        try:
            bundle = registry.get(crop)
            predictor = bundle["predictor"]
            gradcam = bundle["gradcam"]
            
            # Prepare tensor
            device_str = str(predictor.device)
            tensor = preprocess_image(dummy_img, input_size=224, device=device_str)
            
            # Generate Grad-CAM
            heatmap, class_name = gradcam.generate(tensor, target_class=0)
            
            assert heatmap is not None, "Heatmap should not be None"
            assert isinstance(heatmap, np.ndarray), "Heatmap should be numpy array"
            
            logger.info(f"    ✓ GradCAM generation successful")
            logger.info(f"    ✓ Heatmap shape: {heatmap.shape}")
            logger.info(f"    ✓ Target class: {class_name}")
            
        except Exception as e:
            logger.error(f"    ✗ GradCAM failed: {e}", exc_info=True)
    
    logger.info("")


def test_caching():
    """Test 10: Model caching (models don't reload)."""
    logger.info("=" * 60)
    logger.info("TEST 10: Model Caching")
    logger.info("=" * 60)
    
    registry = ModelRegistry()
    
    try:
        registry.load_models()
        crops1 = registry.list_available_crops()
        crop1_predictor_id = id(registry.get(crops1[0])["predictor"])
        
        # Attempt to load again
        registry.load_models()  # Should skip (already loaded)
        crop1_predictor_id_2 = id(registry.get(crops1[0])["predictor"])
        
        # Should be same object (cached)
        assert crop1_predictor_id == crop1_predictor_id_2, \
            "Predictor should be reused from cache, not reloaded"
        
        logger.info(f"  ✓ Models cached (same object on 2nd access)")
        logger.info(f"  ✓ ID: {crop1_predictor_id}")
        
    except Exception as e:
        logger.error(f"  ✗ Caching test failed: {e}", exc_info=True)
    
    logger.info("")


def run_all_tests():
    """Run all tests and report results."""
    logger.info("\n")
    logger.info("╔" + "=" * 58 + "╗")
    logger.info("║" + " " * 10 + "MULTI-CROP MODEL REGISTRY TEST SUITE" + " " * 12 + "║")
    logger.info("╚" + "=" * 58 + "╝")
    logger.info("")
    
    tests = [
        ("CROP_CONFIG Validation", test_crop_config),
        ("Device Detection", test_device_detection),
        ("Checkpoint Files", test_checkpoint_files_exist),
        ("Labels Loading", test_labels_loading),
        ("Model Loading", test_model_loading),
        ("Registry Loading", test_registry_loading),
        ("Registry Functionality", test_registry_functionality),
        ("Prediction", test_prediction),
        ("GradCAM Compatibility", test_gradcam_compatibility),
        ("Model Caching", test_caching),
    ]
    
    passed = 0
    failed = 0
    
    for test_name, test_func in tests:
        try:
            test_func()
            passed += 1
        except Exception as e:
            logger.error(f"TEST FAILED: {test_name}: {e}", exc_info=True)
            failed += 1
        except AssertionError as e:
            logger.error(f"ASSERTION FAILED: {test_name}: {e}")
            failed += 1
    
    logger.info("\n")
    logger.info("╔" + "=" * 58 + "╗")
    logger.info(f"║ Results: {passed} passed, {failed} failed" + " " * (29 - len(str(passed)) - len(str(failed))) + "║")
    logger.info("╚" + "=" * 58 + "╝")
    logger.info("")
    
    return failed == 0


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)
