"""Training script for Rice disease classification model.

Uses EfficientNet-B0 backbone with custom classifier head.
Supports two-phase transfer learning:
  - Phase 1: Freeze backbone, train classifier only (5-10 epochs)
  - Phase 2: Unfreeze backbone, fine-tune features (15-30 epochs)

This two-phase approach helps prevent single-class bias collapse.
"""

import os
import sys
import json
import argparse
from pathlib import Path
from datetime import datetime
from collections import defaultdict

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR, ReduceLROnPlateau
from torch.utils.tensorboard import SummaryWriter
import numpy as np

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from models.efficientnet_model import create_model
from preprocessing.dataset_loader import (
    create_dataloaders,
    set_seed,
    compute_dataset_stats
)
from utils.helpers import load_config, get_config_value
from utils.focal_loss import FocalLoss, WeightedCrossEntropyLoss


class RiceModelTrainer:
    """Trainer class for Rice disease classification model.
    
    Supports two-phase transfer learning to prevent single-class bias:
      Phase 1: Freeze backbone, train classifier (5-10 epochs, lr=1e-3)
      Phase 2: Unfreeze backbone, fine-tune (15-30 epochs, lr=1e-4 or 1e-5)
    """
    
    def __init__(
        self,
        model: nn.Module,
        train_loader,
        val_loader,
        idx_to_class: dict,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
        learning_rate: float = 0.001,
        weight_decay: float = 1e-4,
        num_epochs: int = 50,
        checkpoint_dir: str = "checkpoints",
        tensorboard_dir: str = "runs/rice",
        model_name: str = "rice_efficientnet_b0",
        use_focal_loss: bool = True,
        focal_loss_gamma: float = 2.0,
        focal_loss_alpha: float = 0.25
    ):
        """Initialize trainer.
        
        Args:
            model: PyTorch model to train
            train_loader: Training DataLoader
            val_loader: Validation DataLoader
            idx_to_class: Mapping from class index to class name
            device: Device to train on
            learning_rate: Initial learning rate
            weight_decay: L2 regularization weight decay
            num_epochs: Number of training epochs
            checkpoint_dir: Directory to save checkpoints
            tensorboard_dir: Directory for TensorBoard logs
            model_name: Name identifier for the model
            use_focal_loss: Whether to use Focal Loss instead of CrossEntropyLoss
            focal_loss_gamma: Gamma parameter for Focal Loss (concentration on hard examples)
            focal_loss_alpha: Alpha parameter for Focal Loss (positive example weighting)
        """
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.idx_to_class = idx_to_class
        self.device = device
        self.num_epochs = num_epochs
        self.checkpoint_dir = Path(checkpoint_dir)
        self.model_name = model_name
        self.use_focal_loss = use_focal_loss
        
        # Create checkpoint directory
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        # Setup optimizer with weight decay for L2 regularization
        self.optimizer = optim.AdamW(
            self.model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay,
            betas=(0.9, 0.999)
        )
        
        # Loss function - use Focal Loss for better handling of class imbalance
        class_weights = self._compute_class_weights()
        
        if use_focal_loss:
            print(f"\n📌 Using Focal Loss (gamma={focal_loss_gamma}, alpha={focal_loss_alpha})")
            self.criterion = FocalLoss(
                alpha=focal_loss_alpha,
                gamma=focal_loss_gamma,
                weight=class_weights.to(device),
                reduction="mean"
            )
        else:
            print(f"\n📌 Using Weighted CrossEntropyLoss")
            self.criterion = WeightedCrossEntropyLoss(
                weight=class_weights.to(device),
                label_smoothing=0.1,
                reduction="mean"
            )
        
        # TensorBoard writer
        self.writer = SummaryWriter(f"{tensorboard_dir}/{model_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
        
        # Training metrics
        self.best_val_acc = 0.0
        self.best_epoch = 0
        self.patience_counter = 0
        self.early_stopping_patience = 15
        
        # Track per-class accuracies across epochs
        self.class_accuracies_history = defaultdict(list)
        
        print(f"✓ Trainer initialized for {model_name}")
        print(f"  Device: {device}")
        print(f"  Learning rate: {learning_rate}")
        print(f"  Epochs: {num_epochs}")
        
        
    def _compute_class_weights(self) -> torch.Tensor:
        """Compute class weights to handle imbalanced data.
        
        Returns:
            torch.Tensor: Class weights
        """
        class_counts = defaultdict(int)
        for _, labels in self.train_loader.dataset.samples:
            class_counts[labels] += 1
        
        total_samples = sum(class_counts.values())
        class_weights = torch.tensor(
            [total_samples / (len(class_counts) * class_counts[i]) 
             for i in range(len(class_counts))],
            dtype=torch.float32
        )
        
        print("Class weights (for handling imbalance):")
        for idx, weight in enumerate(class_weights):
            class_name = self.idx_to_class.get(idx, f"Class_{idx}")
            print(f"  {class_name}: {weight:.4f}")
        
        return class_weights
    
    def train_epoch(self, epoch: int) -> dict:
        """Train one epoch.
        
        Args:
            epoch: Current epoch number
            
        Returns:
            dict: Metrics for the epoch
        """
        self.model.train()
        total_loss = 0.0
        correct = 0
        total = 0
        
        for batch_idx, (images, labels) in enumerate(self.train_loader):
            images, labels = images.to(self.device), labels.to(self.device)
            
            # Forward pass
            self.optimizer.zero_grad()
            outputs = self.model(images)
            loss = self.criterion(outputs, labels)
            
            # Backward pass
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()
            
            # Metrics
            total_loss += loss.item()
            _, predicted = outputs.max(1)
            correct += predicted.eq(labels).sum().item()
            total += labels.size(0)
            
            # Progress
            if (batch_idx + 1) % max(1, len(self.train_loader) // 5) == 0:
                acc = 100.0 * correct / total
                avg_loss = total_loss / (batch_idx + 1)
                print(f"  Batch {batch_idx + 1}/{len(self.train_loader)} - Loss: {avg_loss:.4f}, Acc: {acc:.2f}%")
        
        train_loss = total_loss / len(self.train_loader)
        train_acc = 100.0 * correct / total
        
        return {"loss": train_loss, "accuracy": train_acc}
    
    def validate(self, epoch: int) -> dict:
        """Validate on validation set.
        
        Args:
            epoch: Current epoch number
            
        Returns:
            dict: Validation metrics
        """
        self.model.eval()
        total_loss = 0.0
        correct = 0
        total = 0
        class_correct = defaultdict(int)
        class_total = defaultdict(int)
        
        with torch.no_grad():
            for images, labels in self.val_loader:
                images, labels = images.to(self.device), labels.to(self.device)
                
                outputs = self.model(images)
                loss = self.criterion(outputs, labels) if not self.use_focal_loss else self.criterion(outputs, labels)
                
                total_loss += loss.item()
                _, predicted = outputs.max(1)
                correct += predicted.eq(labels).sum().item()
                total += labels.size(0)
                
                # Per-class accuracy
                for pred, label in zip(predicted, labels):
                    class_name = self.idx_to_class[label.item()]
                    class_correct[class_name] += (pred == label).item()
                    class_total[class_name] += 1
        
        val_loss = total_loss / len(self.val_loader)
        val_acc = 100.0 * correct / total
        
        # Per-class accuracy with collapse detection
        print(f"\n  Per-class accuracy:")
        class_accs = {}
        min_class_acc = 100.0
        max_class_acc = 0.0
        collapse_warning = False
        
        for class_name in sorted(class_correct.keys()):
            acc = 100.0 * class_correct[class_name] / class_total[class_name]
            class_accs[class_name] = acc
            self.class_accuracies_history[class_name].append(acc)
            
            print(f"    {class_name}: {acc:.2f}% ({class_correct[class_name]}/{class_total[class_name]})")
            
            min_class_acc = min(min_class_acc, acc)
            max_class_acc = max(max_class_acc, acc)
        
        # Detect single-class bias collapse
        imbalance_ratio = max_class_acc / (min_class_acc + 1e-6)
        if imbalance_ratio > 5.0:
            collapse_warning = True
            print(f"\n  ⚠️  COLLAPSE DETECTED: Class acc ratio = {imbalance_ratio:.2f}")
            print(f"     Max: {max_class_acc:.2f}%, Min: {min_class_acc:.2f}%")
            print(f"     Model may be collapsing into single-class bias!")
        
        return {
            "loss": val_loss,
            "accuracy": val_acc,
            "class_accuracy": class_accs,
            "collapse_warning": collapse_warning
        }
    
    def freeze_backbone(self):
        """Freeze backbone weights to prevent updates (Phase 1 training)."""
        for param in self.model.backbone.features.parameters():
            param.requires_grad = False
        
        # Count trainable params
        trainable = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        total = sum(p.numel() for p in self.model.parameters())
        print(f"✓ Backbone frozen: {trainable:,}/{total:,} params trainable")
    
    def unfreeze_backbone(self):
        """Unfreeze backbone weights for fine-tuning (Phase 2 training)."""
        for param in self.model.backbone.features.parameters():
            param.requires_grad = True
        
        # Count trainable params
        trainable = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        total = sum(p.numel() for p in self.model.parameters())
        print(f"✓ Backbone unfrozen: {trainable:,}/{total:,} params trainable")
    
    def setup_phase_optimizer(self, learning_rate: float, t_max: int):
        """
        Reinitialize optimizer AND schedulers for a new training phase.
        This prevents LR desynchronization.
        """
        self.optimizer = optim.AdamW(
            self.model.parameters(),
            lr=learning_rate,
            weight_decay=1e-4,
            betas=(0.9, 0.999)
        )
        self.scheduler_cosine = CosineAnnealingLR(
            self.optimizer,
            T_max=t_max,
            eta_min=1e-6
        )
        self.scheduler_plateau = ReduceLROnPlateau(
            self.optimizer,
            mode='max',
            factor=0.5,
            patience=5,
            min_lr=1e-6
        )
        
        print(f"✓ Optimizer & schedulers reset (lr={learning_rate})")
    
    def save_checkpoint(self, epoch: int, metrics: dict, is_best: bool = False):
        """Save model checkpoint.
        
        Args:
            epoch: Current epoch
            metrics: Current metrics
            is_best: Whether this is the best model so far
        """
        checkpoint = {
            "epoch": epoch,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scheduler_state_dict": self.scheduler_cosine.state_dict(),
            "metrics": metrics,
            "model_info": self.model.get_model_info(),
            "idx_to_class": self.idx_to_class
        }
        
        checkpoint_path = self.checkpoint_dir / f"{self.model_name}_epoch{epoch:03d}.pt"
        torch.save(checkpoint, checkpoint_path)
        
        if is_best:
            best_path = self.checkpoint_dir / f"{self.model_name}_best.pt"
            torch.save(checkpoint, best_path)
            print(f"✓ Best checkpoint saved: {best_path}")
    
    
    def load_checkpoint(self, checkpoint_path: str):
        """Load model from checkpoint.
        
        Args:
            checkpoint_path: Path to checkpoint file
        """
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        self.scheduler_cosine.load_state_dict(checkpoint["scheduler_state_dict"])
        print(f"✓ Checkpoint loaded from {checkpoint_path}")
        print(f"  Training from epoch {checkpoint['epoch'] + 1}")
    
    def train_two_phase(self, phase1_epochs: int = 10, resume_from: str = None):
        """Train model in two phases to prevent single-class bias collapse.
        
        Phase 1 (5-10 epochs):
          - Freeze backbone, train classifier head only
          - Higher learning rate (1e-3)
          - Prevents catastrophic forgetting of backbone features
        
        Phase 2 (15-30 epochs):
          - Unfreeze backbone, fine-tune all layers
          - Lower learning rate (1e-4 or 1e-5)
          - Learn disease-specific features
        
        Args:
            phase1_epochs: Number of epochs for Phase 1 (default 10)
            resume_from: Optional path to checkpoint to resume training from
        """
        print(f"\n{'='*70}")
        print(f"TWO-PHASE TRAINING FOR RICE DISEASE CLASSIFICATION")
        print(f"{'='*70}")
        
        print(f"\n📌 PHASE 1: Frozen Backbone Training (Epochs 1-{phase1_epochs})")
        print(f"   - Backbone: FROZEN (pretrained features preserved)")
        print(f"   - Learning rate: 1e-3 (learn fastest)")
        print(f"   - Target: Warm up classifier head on new data")
        print(f"   - Epochs: {phase1_epochs}")
        print("-" * 70)
        
        # PHASE 1: Freeze backbone, train only classifier head
        self.freeze_backbone()
        self.setup_phase_optimizer(
        learning_rate=1e-3,
        t_max=phase1_epochs
        ) 
        phase1_best_acc = 0.0
        for epoch in range(phase1_epochs):
            print(f"\nPhase 1, Epoch {epoch + 1}/{phase1_epochs}")
            print("-" * 40)
            
            # Train
            train_metrics = self.train_epoch(epoch)
            print(f"Train Loss: {train_metrics['loss']:.4f}, Acc: {train_metrics['accuracy']:.2f}%")
            
            # Validate
            val_metrics = self.validate(epoch)
            print(f"Val Loss: {val_metrics['loss']:.4f}, Acc: {val_metrics['accuracy']:.2f}%")
            
            # Learning rate scheduling
            self.scheduler_cosine.step()
            self.scheduler_plateau.step(val_metrics['accuracy'])
            
            # Log to TensorBoard
            self.writer.add_scalar("Phase1/Loss/train", train_metrics['loss'], epoch)
            self.writer.add_scalar("Phase1/Loss/val", val_metrics['loss'], epoch)
            self.writer.add_scalar("Phase1/Accuracy/train", train_metrics['accuracy'], epoch)
            self.writer.add_scalar("Phase1/Accuracy/val", val_metrics['accuracy'], epoch)
            self.writer.add_scalar("Phase1/LR", self.optimizer.param_groups[0]['lr'], epoch)
            
            # Checkpoint saving
            if val_metrics['accuracy'] > self.best_val_acc:
                self.best_val_acc = val_metrics['accuracy']
                self.best_epoch = epoch
                phase1_best_acc = val_metrics['accuracy']
                self.patience_counter = 0
                self.save_checkpoint(epoch, val_metrics, is_best=True)
                print(f"✓ New best validation accuracy: {self.best_val_acc:.2f}%")
            else:
                self.patience_counter += 1
                self.save_checkpoint(epoch, val_metrics)
        
        print(f"\n{'='*70}")
        print(f"📌 PHASE 2: Fine-tune Backbone (Epochs {phase1_epochs + 1}-{self.num_epochs})")
        print(f"   - Backbone: UNFROZEN (all layers trainable)")
        print(f"   - Learning rate: 1e-4 (learn carefully)")
        print(f"   - Target: Learn subtle disease-specific features")
        print(f"   - Epochs: {self.num_epochs - phase1_epochs}")
        print("-" * 70)
        
        # PHASE 2: Unfreeze backbone and fine-tune
        self.unfreeze_backbone()
        self.setup_phase_optimizer(
        learning_rate=1e-4,
        t_max=self.num_epochs - phase1_epochs
      )
        # Reset patience counter for Phase 2
        self.patience_counter = 0
        
        for epoch in range(phase1_epochs, self.num_epochs):
            print(f"\nPhase 2, Epoch {epoch + 1}/{self.num_epochs}")
            print("-" * 40)
            
            # Train
            train_metrics = self.train_epoch(epoch)
            print(f"Train Loss: {train_metrics['loss']:.4f}, Acc: {train_metrics['accuracy']:.2f}%")
            
            # Validate
            val_metrics = self.validate(epoch)
            print(f"Val Loss: {val_metrics['loss']:.4f}, Acc: {val_metrics['accuracy']:.2f}%")
            
            # Learning rate scheduling
            self.scheduler_cosine.step()
            self.scheduler_plateau.step(val_metrics['accuracy'])
            
            # Log to TensorBoard
            self.writer.add_scalar("Phase2/Loss/train", train_metrics['loss'], epoch)
            self.writer.add_scalar("Phase2/Loss/val", val_metrics['loss'], epoch)
            self.writer.add_scalar("Phase2/Accuracy/train", train_metrics['accuracy'], epoch)
            self.writer.add_scalar("Phase2/Accuracy/val", val_metrics['accuracy'], epoch)
            self.writer.add_scalar("Phase2/LR", self.optimizer.param_groups[0]['lr'], epoch)
            
            # Checkpoint saving
            if val_metrics['accuracy'] > self.best_val_acc:
                self.best_val_acc = val_metrics['accuracy']
                self.best_epoch = epoch
                self.patience_counter = 0
                self.save_checkpoint(epoch, val_metrics, is_best=True)
                print(f"✓ New best validation accuracy: {self.best_val_acc:.2f}%")
            else:
                self.patience_counter += 1
                self.save_checkpoint(epoch, val_metrics)
                if self.patience_counter % 5 == 0:
                    print(f"⚠ No improvement for {self.patience_counter} epochs")
            
            # Early stopping
            if self.patience_counter >= self.early_stopping_patience:
                print(f"\n⚠ Early stopping triggered after {self.early_stopping_patience} epochs without improvement")
                break
        
        # Print summary
        print(f"\n{'='*70}")
        print(f"✓ TWO-PHASE TRAINING COMPLETE!")
        print(f"  Phase 1 best accuracy: {phase1_best_acc:.2f}%")
        print(f"  Overall best accuracy: {self.best_val_acc:.2f}% (Epoch {self.best_epoch + 1})")
        print(f"{'='*70}\n")
        
        self.writer.close()
    
    
    def train(self, resume_from: str = None):
        """Train the model for specified number of epochs (single-phase - legacy).
        
        Note: Use train_two_phase() for better results to prevent single-class bias.
        
        Args:
            resume_from: Optional path to checkpoint to resume training from
        """
        start_epoch = 0
        
        if resume_from:
            self.load_checkpoint(resume_from)
            start_epoch = int(resume_from.split("_epoch")[1].split(".")[0]) + 1
        
        print(f"\n{'='*60}")
        print(f"Starting training: {self.model_name}")
        print(f"Epochs: {start_epoch} - {self.num_epochs - 1}")
        print(f"Device: {self.device}")
        print(f"{'='*60}\n")
        
        for epoch in range(start_epoch, self.num_epochs):
            print(f"Epoch {epoch + 1}/{self.num_epochs}")
            print("-" * 40)
            
            # Train
            train_metrics = self.train_epoch(epoch)
            print(f"Train Loss: {train_metrics['loss']:.4f}, Acc: {train_metrics['accuracy']:.2f}%")
            
            # Validate
            val_metrics = self.validate(epoch)
            print(f"Val Loss: {val_metrics['loss']:.4f}, Acc: {val_metrics['accuracy']:.2f}%")
            
            # Learning rate scheduling
            self.scheduler_cosine.step()
            self.scheduler_plateau.step(val_metrics['accuracy'])
            
            # Log to TensorBoard
            self.writer.add_scalar("Loss/train", train_metrics['loss'], epoch)
            self.writer.add_scalar("Loss/val", val_metrics['loss'], epoch)
            self.writer.add_scalar("Accuracy/train", train_metrics['accuracy'], epoch)
            self.writer.add_scalar("Accuracy/val", val_metrics['accuracy'], epoch)
            self.writer.add_scalar("LR", self.optimizer.param_groups[0]['lr'], epoch)
            
            # Checkpoint saving
            if val_metrics['accuracy'] > self.best_val_acc:
                self.best_val_acc = val_metrics['accuracy']
                self.best_epoch = epoch
                self.patience_counter = 0
                self.save_checkpoint(epoch, val_metrics, is_best=True)
                print(f"✓ New best validation accuracy: {self.best_val_acc:.2f}%\n")
            else:
                self.patience_counter += 1
                self.save_checkpoint(epoch, val_metrics)
                if self.patience_counter % 5 == 0:
                    print(f"⚠ No improvement for {self.patience_counter} epochs\n")
            
            # Early stopping
            if self.patience_counter >= self.early_stopping_patience:
                print(f"\n⚠ Early stopping triggered after {self.early_stopping_patience} epochs without improvement")
                break
        
        print(f"\n{'='*60}")
        print(f"Training complete!")
        print(f"Best validation accuracy: {self.best_val_acc:.2f}% (Epoch {self.best_epoch + 1})")
        print(f"{'='*60}\n")
        
        self.writer.close()


def main():
    """Main training script."""
    parser = argparse.ArgumentParser(description="Train Rice disease classification model")
    parser.add_argument("--data-dir", type=str, default="Dataset_Crop",
                        help="Root directory containing Train/Val subdirectories")
    parser.add_argument("--epochs", type=int, default=40,
                        help="Total number of training epochs (Phase 1 + Phase 2)")
    parser.add_argument("--phase1-epochs", type=int, default=10,
                        help="Number of epochs for Phase 1 (frozen backbone training)")
    parser.add_argument("--batch-size", type=int, default=32,
                        help="Batch size for training")
    parser.add_argument("--lr", type=float, default=0.001,
                        help="Initial learning rate")
    parser.add_argument("--weight-decay", type=float, default=1e-4,
                        help="Weight decay (L2 regularization)")
    parser.add_argument("--freeze-backbone", action="store_true",
                        help="Freeze backbone weights during training (legacy, not recommended)")
    parser.add_argument("--variant", type=str, default="b0",
                        help="EfficientNet variant (b0-b4)")
    parser.add_argument("--dropout", type=float, default=0.5,
                        help="Dropout rate for classifier")
    parser.add_argument("--checkpoint-dir", type=str, default="checkpoints",
                        help="Directory to save checkpoints")
    parser.add_argument("--resume", type=str, default=None,
                        help="Path to checkpoint to resume training from")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for reproducibility")
    parser.add_argument("--use-focal-loss", action="store_true", default=True,
                        help="Use Focal Loss instead of CrossEntropyLoss (default: True)")
    parser.add_argument("--focal-gamma", type=float, default=2.0,
                        help="Gamma parameter for Focal Loss")
    parser.add_argument("--focal-alpha", type=float, default=0.25,
                        help="Alpha parameter for Focal Loss")
    parser.add_argument("--strong-aug", action="store_true", default=True,
                        help="Use strong augmentation for rice disease training (default: True)")
    parser.add_argument("--single-phase", action="store_true", default=False,
                        help="Use single-phase training instead of two-phase (not recommended)")
    args = parser.parse_args()
    
    # Set random seed
    set_seed(args.seed)
    
    # Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Data paths
    train_dir = os.path.join(args.data_dir, "Train", "Rice")
    val_dir = os.path.join(args.data_dir, "Val", "Rice")
    
    if not os.path.exists(train_dir):
        raise FileNotFoundError(f"Training directory not found: {train_dir}")
    if not os.path.exists(val_dir):
        raise FileNotFoundError(f"Validation directory not found: {val_dir}")
    
    print(f"\nLoading datasets...")
    print(f"  Train: {train_dir}")
    print(f"  Val: {val_dir}")
    
    # Create dataloaders with strong augmentation for rice
    train_loader, val_loader, idx_to_class = create_dataloaders(
        train_dir=train_dir,
        val_dir=val_dir,
        batch_size=args.batch_size,
        num_workers=0,
        input_size=224,
        augment=True,
        use_strong_aug=args.strong_aug,
        pin_memory=False
    )
    
    print(f"\n✓ Datasets loaded successfully")
    print(f"  Classes: {list(idx_to_class.values())}")
    print(f"  Train batches: {len(train_loader)}")
    print(f"  Val batches: {len(val_loader)}")
    
    # Create model
    print(f"\nCreating model: EfficientNet-{args.variant}")
    model = create_model(
        num_classes=len(idx_to_class),
        variant=args.variant,
        pretrained=True,
        dropout=args.dropout,
        freeze_backbone=False,  # Will be handled by two-phase training
        device=str(device)
    )
    
    print(f"✓ Model created")
    model_info = model.get_model_info()
    print(f"  Parameters: {model_info['total_parameters']:,}")
    print(f"  Trainable: {model_info['trainable_parameters']:,}")
    
    # Initialize trainer
    trainer = RiceModelTrainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        idx_to_class=idx_to_class,
        device=str(device),
        learning_rate=args.lr,
        weight_decay=args.weight_decay,
        num_epochs=args.epochs,
        checkpoint_dir=args.checkpoint_dir,
        model_name=f"rice_efficientnet_{args.variant}",
        use_focal_loss=args.use_focal_loss,
        focal_loss_gamma=args.focal_gamma,
        focal_loss_alpha=args.focal_alpha
    )
    
    # Train using two-phase approach (or single-phase if requested)
    if args.single_phase:
        print(f"\n⚠️  Using single-phase training (not recommended for rice disease classification)")
        trainer.train(resume_from=args.resume)
    else:
        print(f"\n✅ Using TWO-PHASE training to prevent single-class bias collapse")
        trainer.train_two_phase(phase1_epochs=args.phase1_epochs, resume_from=args.resume)


if __name__ == "__main__":
    import argparse
    
    # Check if user wants to use the legacy trainer
    parser = argparse.ArgumentParser(description="Train Rice disease classification model")
    parser.add_argument("--use-legacy-trainer", action="store_true", default=False,
                        help="Use legacy RiceModelTrainer instead of unified core_trainer (for advanced features)")
    
    # Parse known args first to check for --use-legacy-trainer
    args, remaining_args = parser.parse_known_args()
    
    if args.use_legacy_trainer:
        # Use the original RiceModelTrainer (backward compatibility for advanced features)
        import sys
        sys.argv = [sys.argv[0]] + remaining_args
        main()
    else:
        # Use the new unified core trainer (default)
        from training.core_trainer import train_crop
        
        # Parse arguments for core trainer
        parser = argparse.ArgumentParser(description="Train Rice disease classification model")
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
            crop_name="rice",
            num_epochs=args.epochs,
            batch_size=args.batch_size,
            learning_rate=args.lr,
            checkpoint_dir=args.checkpoint_dir,
            seed=args.seed
        )
