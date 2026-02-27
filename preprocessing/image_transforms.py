"""Image transformation helpers for training, validation and single-image preprocessing.

Uses torchvision transforms by default. Functions return torchvision.transforms.Compose instances
and a convenience helper to preprocess a single PIL image or image path into a batched tensor
ready for model inference.
"""
from typing import Tuple, Union
from PIL import Image
import torch
from torchvision import transforms as T

# Default normalization (ImageNet) — change if training from scratch with different stats
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

def get_train_transform(
    input_size: int = 224,
    mean: Tuple[float, float, float] = tuple(IMAGENET_MEAN),
    std: Tuple[float, float, float] = tuple(IMAGENET_STD),
    augment: bool = True,
    use_strong_aug: bool = True
    ) -> T.Compose:
    """
    Strong augmentation designed to break texture dominance
    (especially Brown_spot bias).
    """

    ops = [
        T.Lambda(lambda img: img.convert("RGB")),
        T.RandomResizedCrop(input_size, scale=(0.6, 1.0)),
    ]

    if augment and use_strong_aug:
        ops += [
            # Spatial distortions
            T.RandomHorizontalFlip(p=0.5),
            T.RandomAffine(
                degrees=20,
                translate=(0.1, 0.1),
                scale=(0.85, 1.15),
                shear=10
            ),

            # Color & texture destruction
            T.ColorJitter(
                brightness=0.4,
                contrast=0.4,
                saturation=0.3,
                hue=0.05
            ),

            # Blur weakens texture shortcuts
            T.GaussianBlur(kernel_size=3, sigma=(0.1, 2.0)),
        ]

    ops += [
        T.ToTensor(),

        # 🔥 Key anti-collapse augmentation
        T.RandomErasing(
            p=0.5,
            scale=(0.02, 0.25),
            ratio=(0.3, 3.3),
            value="random"
        ),

        T.Normalize(mean=mean, std=std),
    ]

    return T.Compose(ops)


def get_val_transform(input_size: int = 224,
                      mean: Tuple[float, float, float] = tuple(IMAGENET_MEAN),
                      std: Tuple[float, float, float] = tuple(IMAGENET_STD)) -> T.Compose:
    """Return a validation / inference transform (resize / center crop + normalization).

    Args:
        input_size: target crop size for model input
        mean: normalization mean
        std: normalization std

    Returns:
        torchvision.transforms.Compose
    """
    transform = T.Compose([
        T.Lambda(lambda img: img.convert("RGB")),  # Explicit RGB enforcement
        T.Resize(int(input_size * 1.14)),  # slightly larger for a clean center crop
        T.CenterCrop(input_size),
        T.ToTensor(),
        T.Normalize(mean=mean, std=std),
    ])
    return transform


def preprocess_image(image: Union[str, Image.Image],
                     input_size: int = 224,
                     device: Union[str, torch.device] = "cpu") -> torch.Tensor:
    """Load an image (path or PIL.Image) and return a 1 x C x H x W tensor ready for inference.

    Args:
        image: file path or PIL image
        input_size: resize / crop size
        device: destination device for tensor

    Returns:
        torch.Tensor: shape (1, C, H, W), dtype float32
    """
    if isinstance(image, str):
        img = Image.open(image).convert("RGB")
    elif isinstance(image, Image.Image):
        img = image.convert("RGB")
    else:
        raise TypeError("image must be a file path or PIL.Image.Image")

    transform = get_val_transform(input_size=input_size)
    tensor = transform(img).unsqueeze(0).to(device)
    return tensor


def get_tta_transforms(input_size: int = 224,
                       mean: Tuple[float, float, float] = tuple(IMAGENET_MEAN),
                       std: Tuple[float, float, float] = tuple(IMAGENET_STD)) -> list:
    """Return a list of transforms for test-time augmentation (TTA).
    
    Args:
        input_size: target crop size for model input
        mean: normalization mean
        std: normalization std
        
    Returns:
        list of torchvision.transforms.Compose for different augmentations
    """
    base_transform = T.Compose([
        T.Lambda(lambda img: img.convert("RGB")),
        T.Resize(int(input_size * 1.14)),
        T.CenterCrop(input_size),
        T.ToTensor(),
        T.Normalize(mean=mean, std=std),
    ])
    
    # Different TTA variations
    tta_transforms = [
        base_transform,  # Original
        T.Compose([
            T.Lambda(lambda img: img.convert("RGB")),
            T.Resize(int(input_size * 1.14)),
            T.CenterCrop(input_size),
            T.RandomHorizontalFlip(p=1.0),  # Always flip
            T.ToTensor(),
            T.Normalize(mean=mean, std=std),
        ]),
        T.Compose([
            T.Lambda(lambda img: img.convert("RGB")),
            T.Resize(int(input_size * 1.2)),  # Slightly larger resize
            T.CenterCrop(input_size),
            T.ToTensor(),
            T.Normalize(mean=mean, std=std),
        ]),
    ]
    
    return tta_transforms


__all__ = [
    "get_train_transform",
    "get_val_transform",
    "get_tta_transforms",
    "preprocess_image",
]
