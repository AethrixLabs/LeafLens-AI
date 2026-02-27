import os
import json
import sys
from pathlib import Path
from typing import Dict, Tuple, Optional

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from models.efficientnet_model import create_model
from preprocessing.dataset_loader import create_dataloaders, set_seed


class EarlyStopping:
    """Early stopping callback to stop training when validation loss stops improving.
    
    Monitors validation loss and stops training if no improvement is seen for
    a specified number of epochs (patience).
    """
    
    def __init__(self, patience: int = 5, min_delta: float = 0.0, mode: str = "min"):
        """Initialize early stopping.
        
        Args:
            patience: Number of epochs to wait before stopping
            min_delta: Minimum change to qualify as an improvement
            mode: 'min' for loss (lower is better), 'max' for accuracy (higher is better)
        """
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        
    def __call__(self, score: float) -> bool:
        """Check if training should stop.
        
        Args:
            score: Current validation score (loss or accuracy)
            
        Returns:
            True if training should stop, False otherwise
        """
        if self.best_score is None:
            self.best_score = score
            return False

        if self.mode == "min":
            improved = score < (self.best_score - self.min_delta)
        else:  # mode == "max"
            improved = score > (self.best_score + self.min_delta)

        if improved:
            self.best_score = score
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True

        return self.early_stop


def build_model(num_classes: int, device: torch.device) -> nn.Module:
    """Build EfficientNet-B0 model with custom classifier head.
    
    Args:
        num_classes: Number of output classes
        device: Device to place model on
        
    Returns:
        Configured EfficientNet model
    """
    model = create_model(
        num_classes=num_classes,
        variant="b0",
        pretrained=True,
        dropout=0.5,
        freeze_backbone=False,  # Will be handled by two-phase training
        device=str(device)
    )
    return model.to(device)


def train_one_epoch(
    model: nn.Module,
    train_loader: DataLoader,
    criterion: nn.Module,
    optimizer: optim.Optimizer,
    device: torch.device
) -> Dict[str, float]:
    """Train model for one epoch.
    
    Args:
        model: PyTorch model to train
        train_loader: Training DataLoader
        criterion: Loss function
        optimizer: Optimizer
        device: Device to train on
        
    Returns:
        Dictionary with training metrics (loss, accuracy)
    """
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0
    
    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)
        
        # Forward pass
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        
        # Backward pass
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        
        # Metrics
        total_loss += loss.item()
        _, predicted = outputs.max(1)
        correct += predicted.eq(labels).sum().item()
        total += labels.size(0)
    
    train_loss = total_loss / len(train_loader)
    train_acc = 100.0 * correct / total
    
    return {"loss": train_loss, "accuracy": train_acc}


def validate(
    model: nn.Module,
    val_loader: DataLoader,
    criterion: nn.Module,
    device: torch.device
) -> Dict[str, float]:
    """Validate model on validation set.
    
    Args:
        model: PyTorch model to validate
        val_loader: Validation DataLoader
        criterion: Loss function
        device: Device to validate on
        
    Returns:
        Dictionary with validation metrics (loss, accuracy)
    """
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(device), labels.to(device)
            
            outputs = model(images)
            loss = criterion(outputs, labels)
            
            total_loss += loss.item()
            _, predicted = outputs.max(1)
            correct += predicted.eq(labels).sum().item()
            total += labels.size(0)
    
    val_loss = total_loss / len(val_loader)
    val_acc = 100.0 * correct / total
    
    return {"loss": val_loss, "accuracy": val_acc}


def save_checkpoint(
    model: nn.Module,
    checkpoint_dir: Path,
    crop_name: str,
    epoch: int,
    metrics: Dict[str, float],
    idx_to_class: Dict[int, str],
    is_best: bool = False
) -> None:
    """Save model checkpoint and class names.
    
    Args:
        model: PyTorch model to save
        checkpoint_dir: Directory to save checkpoints
        crop_name: Name of the crop
        epoch: Current epoch number
        metrics: Current metrics dictionary
        idx_to_class: Mapping from class index to class name
        is_best: Whether this is the best model so far
    """
    crop_checkpoint_dir = checkpoint_dir / crop_name
    crop_checkpoint_dir.mkdir(parents=True, exist_ok=True)
    
    # Save checkpoint
    checkpoint = {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "metrics": metrics,
        "model_info": model.get_model_info(),
        "idx_to_class": idx_to_class
    }
    
    checkpoint_path = crop_checkpoint_dir / f"best_model.pth"
    if is_best:
        torch.save(checkpoint, checkpoint_path)
        print(f"✓ Best checkpoint saved: {checkpoint_path}")
    
    # Save class names as JSON (list format for inference compatibility)
    class_names = [idx_to_class[i] for i in sorted(idx_to_class.keys())]
    class_names_path = crop_checkpoint_dir / "class_names.json"
    with open(class_names_path, "w", encoding="utf-8") as f:
        json.dump(class_names, f, indent=2)
    print(f"✓ Class names saved: {class_names_path}")


def train_crop(
    crop_name: str,
    num_epochs: int = 20,
    batch_size: int = 16,
    learning_rate: float = 1e-4,
    checkpoint_dir: str = "checkpoints",
    seed: int = 42
) -> None:
    """Train a crop disease classification model using two-phase fine-tuning.
    
    This function implements a unified training pipeline for any crop:
    - Phase 1: Freeze backbone features, train classifier only (5 epochs)
    - Phase 2: Unfreeze last 2 layers of features, continue training
    
    Args:
        crop_name: Name of the crop (e.g., "rice", "wheat", "corn", "potato")
        num_epochs: Total number of training epochs
        batch_size: Batch size for training (default: 16)
        learning_rate: Learning rate for Adam optimizer (default: 1e-4)
        checkpoint_dir: Directory to save checkpoints (default: "checkpoints")
        seed: Random seed for reproducibility (default: 42)
    """
    # Set random seed
    set_seed(seed)
    
    # Device handling
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n{'='*70}")
    print(f"Training {crop_name.upper()} Disease Classification Model")
    print(f"{'='*70}")
    print(f"Device: {device}")
    print(f"Epochs: {num_epochs}")
    print(f"Batch size: {batch_size}")
    print(f"Learning rate: {learning_rate}")
    
    # Get project root directory (LeafLens-AI/)
    PROJECT_ROOT = Path(__file__).resolve().parent.parent
    
    # Data paths
    train_dir = PROJECT_ROOT / "Dataset_Crop" / "Train" / crop_name
    val_dir = PROJECT_ROOT / "Dataset_Crop" / "Val" / crop_name
    
    if not train_dir.exists():
        raise FileNotFoundError(f"Training directory not found: {train_dir}")
    if not val_dir.exists():
        raise FileNotFoundError(f"Validation directory not found: {val_dir}")
    
    print(f"\nLoading datasets...")
    print(f"  Train: {train_dir}")
    print(f"  Val: {val_dir}")
    
    # Create dataloaders
    train_loader, val_loader, idx_to_class = create_dataloaders(
        train_dir=str(train_dir),
        val_dir=str(val_dir),
        batch_size=batch_size,
        num_workers=0,  # Set to 0 for CPU compatibility
        input_size=224,
        augment=True,
        use_strong_aug=False,  # Can be customized per crop if needed
        pin_memory=False  # Set to False for CPU compatibility
    )
    
    print(f"\n✓ Datasets loaded successfully")
    print(f"  Classes: {list(idx_to_class.values())}")
    print(f"  Train batches: {len(train_loader)}")
    print(f"  Val batches: {len(val_loader)}")
    
    # Build model
    num_classes = len(idx_to_class)
    print(f"\nCreating EfficientNet-B0 model...")
    model = build_model(num_classes=num_classes, device=device)
    
    model_info = model.get_model_info()
    print(f"✓ Model created")
    print(f"  Parameters: {model_info['total_parameters']:,}")
    print(f"  Trainable: {model_info['trainable_parameters']:,}")
    
    # Loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=learning_rate
    )
    
    # Early stopping (monitor accuracy — higher is better)
    early_stopping = EarlyStopping(patience=5, mode="max")

    # Training metrics (use validation accuracy as primary metric)
    best_val_acc = 0.0
    best_epoch = 0
    checkpoint_path = Path(checkpoint_dir)

    # Helper to save checkpoint when validation accuracy improves
    def _maybe_save_best(val_metrics: Dict[str, float], epoch: int) -> None:
        nonlocal best_val_acc, best_epoch
        if val_metrics['accuracy'] > best_val_acc:
            best_val_acc = val_metrics['accuracy']
            best_epoch = epoch
            save_checkpoint(
                model=model,
                checkpoint_dir=checkpoint_path,
                crop_name=crop_name,
                epoch=epoch,
                metrics=val_metrics,
                idx_to_class=idx_to_class,
                is_best=True
            )
            print(f"✓ New best validation accuracy: {best_val_acc:.2f}%")
    
    # ========================================================================
    # PHASE 1: Freeze backbone features, train classifier only
    # ========================================================================
    phase1_epochs = min(5, num_epochs)  # Safety: don't exceed total epochs
    print(f"\n{'='*70}")
    print(f"PHASE 1: Frozen Backbone Training (Epochs 1-{phase1_epochs})")
    print(f"  - Backbone: FROZEN (pretrained features preserved)")
    print(f"  - Learning rate: {learning_rate}")
    print(f"  - Target: Warm up classifier head on new data")
    print(f"{'='*70}")
    
    # Freeze backbone features
    for param in model.backbone.features.parameters():
        param.requires_grad = False
    
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"✓ Backbone frozen: {trainable_params:,}/{total_params:,} params trainable")
    
    for epoch in range(phase1_epochs):
        print(f"\nPhase 1, Epoch {epoch + 1}/{phase1_epochs}")
        print("-" * 40)
        
        # Train
        train_metrics = train_one_epoch(model, train_loader, criterion, optimizer, device)
        print(f"Train Loss: {train_metrics['loss']:.4f}, Acc: {train_metrics['accuracy']:.2f}%")
        
        # Validate
        val_metrics = validate(model, val_loader, criterion, device)
        print(f"Val Loss: {val_metrics['loss']:.4f}, Acc: {val_metrics['accuracy']:.2f}%")
        
        # Save checkpoint if best (by validation accuracy)
        _maybe_save_best(val_metrics, epoch)
    
    # ========================================================================
    # PHASE 2: Unfreeze last 2 layers of features, continue training
    # ========================================================================
    phase2_start_epoch = phase1_epochs
    phase2_epochs = num_epochs - phase1_epochs
    
    print(f"\n{'='*70}")
    print(f"PHASE 2: Fine-tune Backbone (Epochs {phase2_start_epoch + 1}-{num_epochs})")
    print(f"  - Backbone: UNFROZEN (last 2 layers trainable)")
    print(f"  - Learning rate: {learning_rate}")
    print(f"  - Target: Learn subtle disease-specific features")
    print(f"{'='*70}")
    
    # Unfreeze last 2 layers of backbone features
    # EfficientNet features is a Sequential module containing:
    # - First conv layer
    # - Multiple MBConv blocks (stages)
    # - Last conv layer
    features = model.backbone.features
    feature_list = list(features.children())
    num_blocks = len(feature_list)
    
    # Unfreeze last 2 blocks (typically the last MBConv stage and final conv)
    # Freeze all except the last 2
    for i, block in enumerate(feature_list):
        if i >= num_blocks - 2:
            # Unfreeze last 2 blocks
            for param in block.parameters():
                param.requires_grad = True
        else:
            # Keep frozen
            for param in block.parameters():
                param.requires_grad = False
    
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"✓ Last 2 layers unfrozen: {trainable_params:,}/{total_params:,} params trainable")
    
    # Reinitialize optimizer for Phase 2 (optional, but helps with learning rate)
    optimizer = optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=learning_rate
    )
    
    for epoch in range(phase2_start_epoch, num_epochs):
        print(f"\nPhase 2, Epoch {epoch + 1}/{num_epochs}")
        print("-" * 40)
        
        # Train
        train_metrics = train_one_epoch(model, train_loader, criterion, optimizer, device)
        print(f"Train Loss: {train_metrics['loss']:.4f}, Acc: {train_metrics['accuracy']:.2f}%")
        
        # Validate
        val_metrics = validate(model, val_loader, criterion, device)
        print(f"Val Loss: {val_metrics['loss']:.4f}, Acc: {val_metrics['accuracy']:.2f}%")
        
        # Save checkpoint if best (by validation accuracy)
        _maybe_save_best(val_metrics, epoch)
        
        # Early stopping check (use validation accuracy)
        if early_stopping(val_metrics['accuracy']):
            print(f"\n⚠ Early stopping triggered after {early_stopping.patience} epochs without improvement")
            break
    
    # Print summary
    print(f"\n{'='*70}")
    print(f"✓ Training Complete!")
    print(f"  Best validation accuracy: {best_val_acc:.2f}% (Epoch {best_epoch + 1})")
    print(f"  Checkpoint saved to: {checkpoint_path / crop_name / 'best_model.pth'}")
    print(f"  Class names saved to: {checkpoint_path / crop_name / 'class_names.json'}")
    print(f"{'='*70}\n")


if __name__ == "__main__":
    # Example usage
    import argparse
    
    parser = argparse.ArgumentParser(description="Train crop disease classification model")
    parser.add_argument("--crop", type=str, required=True, choices=["rice", "wheat", "corn", "potato"],
                        help="Crop name to train")
    parser.add_argument("--epochs", type=int, default=20,
                        help="Total number of training epochs")
    parser.add_argument("--batch-size", type=int, default=16,
                        help="Batch size for training")
    parser.add_argument("--lr", type=float, default=1e-4,
                        help="Learning rate")
    parser.add_argument("--checkpoint-dir", type=str, default="checkpoints",
                        help="Directory to save checkpoints")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed")
    
    args = parser.parse_args()
    
    train_crop(
        crop_name=args.crop,
        num_epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        checkpoint_dir=args.checkpoint_dir,
        seed=args.seed
    )
