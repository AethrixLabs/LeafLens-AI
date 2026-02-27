"""Helpers to construct datasets and dataloaders using torchvision.datasets.ImageFolder.

Provides convenience wrappers that attach the transforms from `image_transforms.py` and
expose Dataset and DataLoader objects with class name mappings.
"""
from typing import Tuple, Dict, Optional
import os
import sys
from pathlib import Path
import torch
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
from torchvision import transforms as T
import numpy as np
from collections import Counter

# Ensure parent directory is in path for imports
_parent_dir = str(Path(__file__).parent.parent)
if _parent_dir not in sys.path:
    sys.path.insert(0, _parent_dir)

from preprocessing.image_transforms import get_train_transform, get_val_transform, get_tta_transforms
from utils.helpers import load_config, get_config_value


def create_image_datasets(train_dir: str,
                          val_dir: str,
                          input_size: int = 224,
                          augment: bool = True,
                          use_strong_aug: bool = False):
    """Create ImageFolder datasets for train and validation directories.

    Args:
        train_dir: directory pointing to training root (one subfolder per class)
        val_dir: directory pointing to validation root (one subfolder per class)
        input_size: transform input size
        augment: whether to apply data augmentation to training set
        use_strong_aug: whether to use strong augmentation (for fighting single-class bias)

    Returns:
        tuple: (train_dataset, val_dataset)
    """
    if not os.path.isdir(train_dir):
        raise FileNotFoundError(f"Train directory does not exist: {train_dir}")
    if not os.path.isdir(val_dir):
        raise FileNotFoundError(f"Val directory does not exist: {val_dir}")

    train_dataset = ImageFolder(root=train_dir, transform=get_train_transform(input_size, augment=augment, use_strong_aug=use_strong_aug))
    val_dataset = ImageFolder(root=val_dir, transform=get_val_transform(input_size))

    return train_dataset, val_dataset


def create_dataloaders(train_dir: str,
                       val_dir: str,
                       batch_size: int = 32,
                       num_workers: int = 4,
                       pin_memory: bool = True,
                       input_size: int = 224,
                       augment: bool = True,
                       persistent_workers: bool = True,
                       use_strong_aug: bool = False) -> Tuple[DataLoader, DataLoader, Dict[int, str]]:
    """Create train and val DataLoaders.

    Args:
        train_dir: training images root
        val_dir: validation images root
        batch_size: batch size
        num_workers: number of data loader workers
        pin_memory: set pin_memory on dataloaders
        input_size: model input size
        augment: whether to apply data augmentation
        persistent_workers: use persistent workers for better performance
        use_strong_aug: whether to use strong augmentation (for fighting single-class bias)

    Returns:
        train_loader, val_loader, idx_to_class
    """
    train_dataset, val_dataset = create_image_datasets(train_dir, val_dir, input_size=input_size, augment=augment, use_strong_aug=use_strong_aug)

    # Log class distribution for imbalance awareness
    class_counts = Counter([label for _, label in train_dataset.samples])
    print("Training class distribution:")
    for class_idx, count in sorted(class_counts.items()):
        class_name = train_dataset.classes[class_idx]
        print(f"  {class_name}: {count} samples")
    
    # Check for imbalance
    min_count = min(class_counts.values())
    max_count = max(class_counts.values())
    if max_count / min_count > 2.0:
        print(f"⚠️  Class imbalance detected (ratio: {max_count/min_count:.2f})")
        print("Consider using weighted loss or data augmentation strategies")

    train_loader = DataLoader(train_dataset,
                              batch_size=batch_size,
                              shuffle=True,
                              num_workers=num_workers,
                              pin_memory=pin_memory,
                              persistent_workers=persistent_workers if num_workers > 0 else False)

    val_loader = DataLoader(val_dataset,
                            batch_size=batch_size,
                            shuffle=False,
                            num_workers=num_workers,
                            pin_memory=pin_memory,
                            persistent_workers=persistent_workers if num_workers > 0 else False)

    # Mapping from class index to class name (useful for inference/metrics)
    idx_to_class = {v: k for k, v in train_dataset.class_to_idx.items()}

    return train_loader, val_loader, idx_to_class


def create_dataloaders_from_config(train_dir: str,
                                   val_dir: str,
                                   config_path: str = None) -> Tuple[DataLoader, DataLoader, Dict[int, str]]:
    """Create train and val DataLoaders using configuration from YAML file.

    Args:
        train_dir: training images root
        val_dir: validation images root
        config_path: path to config file (optional, uses default if None)

    Returns:
        train_loader, val_loader, idx_to_class
    """
    config = load_config(config_path)
    
    # Extract parameters from config
    batch_size = get_config_value(config, 'data.batch_size', 32)
    num_workers = get_config_value(config, 'data.num_workers', 4)
    pin_memory = get_config_value(config, 'data.pin_memory', True)
    input_size = get_config_value(config, 'model.input_size', 224)
    augment = get_config_value(config, 'data.augment', True)
    persistent_workers = get_config_value(config, 'data.persistent_workers', True)
    
    return create_dataloaders(
        train_dir=train_dir,
        val_dir=val_dir,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=pin_memory,
        input_size=input_size,
        augment=augment,
        persistent_workers=persistent_workers
    )


def compute_dataset_stats(dataset: ImageFolder) -> Tuple[Tuple[float, float, float], Tuple[float, float, float]]:
    """Compute mean and std statistics from a dataset.
    
    Args:
        dataset: ImageFolder dataset
        
    Returns:
        tuple: (mean, std) where each is a tuple of 3 floats
    """
    print("Computing dataset statistics (this may take a while)...")
    
    # Temporarily remove normalization to get raw pixel values
    original_transform = dataset.transform
    dataset.transform = T.Compose([
        T.Lambda(lambda img: img.convert("RGB")),
        T.ToTensor()
    ])
    
    loader = DataLoader(dataset, batch_size=32, shuffle=False, num_workers=0)
    
    mean = torch.zeros(3)
    std = torch.zeros(3)
    total_samples = 0
    
    for images, _ in loader:
        batch_samples = images.size(0)
        images = images.view(batch_samples, images.size(1), -1)
        mean += images.mean(2).sum(0)
        std += images.std(2).sum(0)
        total_samples += batch_samples
    
    mean /= total_samples
    std /= total_samples
    
    # Restore original transform
    dataset.transform = original_transform
    
    mean_tuple = tuple(mean.tolist())
    std_tuple = tuple(std.tolist())
    
    print(f"Dataset mean: {mean_tuple}")
    print(f"Dataset std: {std_tuple}")
    
    return mean_tuple, std_tuple


def set_seed(seed: int = 42):
    """Set random seed for reproducibility.
    
    Args:
        seed: random seed value
    """
    import random
    import torch
    import numpy as np
    
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed)
    random.seed(seed)
    
    print(f"Random seed set to {seed} for reproducibility")


__all__ = ["create_image_datasets", "create_dataloaders", "create_dataloaders_from_config", "compute_dataset_stats", "set_seed"]
