"""Evaluation script for Rice disease classification model with confusion matrix.

Generates predictions on validation set and produces:
- Confusion matrix with visualization
- Per-class precision, recall, F1-score
- Overall accuracy and weighted metrics
- Classification report
"""

import os
import sys
import argparse
import json
from pathlib import Path
from datetime import datetime

# Add parent directory to path FIRST before any relative imports
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from torchvision.datasets import ImageFolder
from sklearn.metrics import (
    confusion_matrix,
    classification_report,
    accuracy_score,
    precision_recall_fscore_support,
    cohen_kappa_score,
    matthews_corrcoef
)

from models.efficientnet_model import create_model
from preprocessing.image_transforms import get_val_transform



class RiceModelEvaluator:
    """Evaluator class for Rice disease classification model."""
    
    def __init__(
        self,
        model: nn.Module,
        val_loader: DataLoader,
        idx_to_class: dict,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
        output_dir: str = "evaluation_results"
    ):
        """Initialize evaluator.
        
        Args:
            model: PyTorch model to evaluate
            val_loader: Validation DataLoader
            idx_to_class: Mapping from class index to class name
            device: Device to evaluate on
            output_dir: Directory to save evaluation results
        """
        self.model = model.to(device)
        self.val_loader = val_loader
        self.idx_to_class = idx_to_class
        self.class_to_idx = {v: k for k, v in idx_to_class.items()}
        self.device = device
        self.output_dir = Path(output_dir)
        
        # Create output directory
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Evaluation metrics storage
        self.all_predictions = []
        self.all_labels = []
        self.all_probabilities = []
        
        print(f"✓ Evaluator initialized for Rice model")
        print(f"  Device: {device}")
        print(f"  Number of classes: {len(idx_to_class)}")
        print(f"  Classes: {list(idx_to_class.values())}")
    
    def generate_predictions(self) -> dict:
        """Generate predictions on validation set.
        
        Returns:
            dict: Prediction results including statistics
        """
        print(f"\n{'='*60}")
        print(f"Generating predictions on validation set...")
        print(f"{'='*60}\n")
        
        self.model.eval()
        total = 0
        correct = 0
        
        with torch.no_grad():
            for batch_idx, (images, labels) in enumerate(self.val_loader):
                images, labels = images.to(self.device), labels.to(self.device)
                
                # Forward pass
                outputs = self.model(images)
                probabilities = torch.softmax(outputs, dim=1)
                _, predicted = outputs.max(1)
                
                # Store results
                self.all_predictions.extend(predicted.cpu().numpy())
                self.all_labels.extend(labels.cpu().numpy())
                self.all_probabilities.extend(probabilities.cpu().numpy())
                
                # Compute accuracy
                correct += predicted.eq(labels).sum().item()
                total += labels.size(0)
                
                # Progress
                if (batch_idx + 1) % max(1, len(self.val_loader) // 5) == 0:
                    acc = 100.0 * correct / total
                    print(f"  Batch {batch_idx + 1}/{len(self.val_loader)} - Acc: {acc:.2f}%")
        
        overall_accuracy = 100.0 * correct / total
        print(f"\n✓ Predictions generated")
        print(f"  Total samples: {total}")
        print(f"  Overall accuracy: {overall_accuracy:.2f}%")
        
        return {
            "total_samples": total,
            "correct_predictions": correct,
            "overall_accuracy": overall_accuracy
        }
    
    def compute_confusion_matrix(self) -> np.ndarray:
        """Compute confusion matrix.
        
        Returns:
            np.ndarray: Confusion matrix
        """
        cm = confusion_matrix(self.all_labels, self.all_predictions)
        return cm
    
    def compute_metrics(self) -> dict:
        """Compute detailed classification metrics.
        
        Returns:
            dict: Comprehensive metrics
        """
        # Basic accuracy metrics
        accuracy = accuracy_score(self.all_labels, self.all_predictions)
        kappa = cohen_kappa_score(self.all_labels, self.all_predictions)
        mcc = matthews_corrcoef(self.all_labels, self.all_predictions)
        
        # Per-class metrics
        precision, recall, f1, support = precision_recall_fscore_support(
            self.all_labels,
            self.all_predictions,
            average=None
        )
        
        # Weighted averages
        weighted_precision = np.average(precision, weights=support)
        weighted_recall = np.average(recall, weights=support)
        weighted_f1 = np.average(f1, weights=support)
        
        # Macro averages
        macro_precision = np.mean(precision)
        macro_recall = np.mean(recall)
        macro_f1 = np.mean(f1)
        
        metrics = {
            "overall": {
                "accuracy": float(accuracy),
                "cohen_kappa": float(kappa),
                "matthews_corrcoef": float(mcc),
                "macro_precision": float(macro_precision),
                "macro_recall": float(macro_recall),
                "macro_f1": float(macro_f1),
                "weighted_precision": float(weighted_precision),
                "weighted_recall": float(weighted_recall),
                "weighted_f1": float(weighted_f1)
            },
            "per_class": {}
        }
        
        # Per-class metrics
        for class_idx, class_name in self.idx_to_class.items():
            metrics["per_class"][class_name] = {
                "precision": float(precision[class_idx]),
                "recall": float(recall[class_idx]),
                "f1_score": float(f1[class_idx]),
                "support": int(support[class_idx])
            }
        
        return metrics
    
    def visualize_confusion_matrix(self, cm: np.ndarray, normalize: bool = True):
        """Visualize and save confusion matrix.
        
        Args:
            cm: Confusion matrix
            normalize: Whether to normalize the confusion matrix
        """
        # Create figure
        fig, axes = plt.subplots(1, 2, figsize=(16, 6))
        
        # Original confusion matrix
        sns.heatmap(
            cm,
            annot=True,
            fmt='d',
            cmap='Blues',
            xticklabels=list(self.idx_to_class.values()),
            yticklabels=list(self.idx_to_class.values()),
            ax=axes[0],
            cbar_kws={'label': 'Count'}
        )
        axes[0].set_title('Confusion Matrix (Raw Counts)', fontsize=14, fontweight='bold')
        axes[0].set_ylabel('True Label', fontsize=12)
        axes[0].set_xlabel('Predicted Label', fontsize=12)
        plt.setp(axes[0].get_xticklabels(), rotation=45, ha='right')
        plt.setp(axes[0].get_yticklabels(), rotation=0)
        
        # Normalized confusion matrix
        if normalize:
            cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
            sns.heatmap(
                cm_normalized,
                annot=True,
                fmt='.2%',
                cmap='Greens',
                xticklabels=list(self.idx_to_class.values()),
                yticklabels=list(self.idx_to_class.values()),
                ax=axes[1],
                cbar_kws={'label': 'Percentage'}
            )
            axes[1].set_title('Confusion Matrix (Normalized)', fontsize=14, fontweight='bold')
            axes[1].set_ylabel('True Label', fontsize=12)
            axes[1].set_xlabel('Predicted Label', fontsize=12)
            plt.setp(axes[1].get_xticklabels(), rotation=45, ha='right')
            plt.setp(axes[1].get_yticklabels(), rotation=0)
        
        plt.tight_layout()
        
        # Save figure
        output_path = self.output_dir / f"confusion_matrix_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"✓ Confusion matrix visualization saved: {output_path}")
        plt.close()
    
    def generate_classification_report(self) -> str:
        """Generate detailed classification report.
        
        Returns:
            str: Classification report
        """
        class_names = [self.idx_to_class[i] for i in range(len(self.idx_to_class))]
        report = classification_report(
            self.all_labels,
            self.all_predictions,
            target_names=class_names,
            digits=4
        )
        return report
    
    def generate_evaluation_report(self, metrics: dict, cm: np.ndarray) -> str:
        """Generate comprehensive evaluation report.
        
        Args:
            metrics: Metrics dictionary
            cm: Confusion matrix
            
        Returns:
            str: Formatted evaluation report
        """
        report = []
        report.append("\n" + "="*70)
        report.append("RICE DISEASE CLASSIFICATION MODEL - EVALUATION REPORT")
        report.append("="*70)
        report.append(f"Evaluation Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        
        # Overall metrics
        report.append("OVERALL METRICS:")
        report.append("-" * 70)
        overall = metrics["overall"]
        report.append(f"  Overall Accuracy:        {overall['accuracy']:.4f} ({overall['accuracy']*100:.2f}%)")
        report.append(f"  Cohen's Kappa:           {overall['cohen_kappa']:.4f}")
        report.append(f"  Matthews Correlation:    {overall['matthews_corrcoef']:.4f}")
        report.append(f"\n  Weighted Precision:      {overall['weighted_precision']:.4f}")
        report.append(f"  Weighted Recall:         {overall['weighted_recall']:.4f}")
        report.append(f"  Weighted F1-Score:       {overall['weighted_f1']:.4f}")
        report.append(f"\n  Macro Precision:         {overall['macro_precision']:.4f}")
        report.append(f"  Macro Recall:            {overall['macro_recall']:.4f}")
        report.append(f"  Macro F1-Score:          {overall['macro_f1']:.4f}\n")
        
        # Per-class metrics
        report.append("PER-CLASS METRICS:")
        report.append("-" * 70)
        report.append(f"{'Class':<30} {'Precision':<12} {'Recall':<12} {'F1-Score':<12} {'Support':<10}")
        report.append("-" * 70)
        
        for class_name, metrics_dict in metrics["per_class"].items():
            report.append(
                f"{class_name:<30} "
                f"{metrics_dict['precision']:<12.4f} "
                f"{metrics_dict['recall']:<12.4f} "
                f"{metrics_dict['f1_score']:<12.4f} "
                f"{metrics_dict['support']:<10}"
            )
        report.append("-" * 70 + "\n")
        
        # Confusion matrix details
        report.append("CONFUSION MATRIX DETAILS:")
        report.append("-" * 70)
        report.append("True positives per class:")
        for i, class_name in enumerate(self.idx_to_class.values()):
            tp = cm[i, i]
            total = cm[i].sum()
            report.append(f"  {class_name:<30} {tp:>4} / {total:<4} ({100*tp/total:.1f}%)")
        report.append("-" * 70 + "\n")
        
        return "\n".join(report)
    
    def save_results(self, metrics: dict, cm: np.ndarray, classification_report: str):
        """Save all evaluation results to files.
        
        Args:
            metrics: Metrics dictionary
            cm: Confusion matrix
            classification_report: Classification report string
        """
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        # Save metrics as JSON
        metrics_path = self.output_dir / f"metrics_{timestamp}.json"
        with open(metrics_path, 'w') as f:
            json.dump(metrics, f, indent=2)
        print(f"✓ Metrics saved: {metrics_path}")
        
        # Save confusion matrix as JSON
        cm_path = self.output_dir / f"confusion_matrix_{timestamp}.json"
        cm_dict = {
            "confusion_matrix": cm.tolist(),
            "class_names": list(self.idx_to_class.values()),
            "classes": list(range(len(self.idx_to_class)))
        }
        with open(cm_path, 'w') as f:
            json.dump(cm_dict, f, indent=2)
        print(f"✓ Confusion matrix saved: {cm_path}")
        
        # Save classification report
        report_path = self.output_dir / f"classification_report_{timestamp}.txt"
        with open(report_path, 'w') as f:
            f.write(classification_report)
        print(f"✓ Classification report saved: {report_path}")
    
    def evaluate(self):
        """Run full evaluation pipeline."""
        print(f"\n{'='*60}")
        print(f"RICE MODEL EVALUATION PIPELINE")
        print(f"{'='*60}")
        
        # Generate predictions
        prediction_stats = self.generate_predictions()
        
        # Compute confusion matrix
        cm = self.compute_confusion_matrix()
        
        # Compute detailed metrics
        metrics = self.compute_metrics()
        
        # Generate reports
        class_report = self.generate_classification_report()
        eval_report = self.generate_evaluation_report(metrics, cm)
        
        # Print reports
        print(eval_report)
        print("\nCLASSIFICATION REPORT:")
        print("-" * 70)
        print(class_report)
        
        # Visualize confusion matrix
        print(f"\nGenerating confusion matrix visualization...")
        self.visualize_confusion_matrix(cm, normalize=True)
        
        # Save results
        print(f"\nSaving evaluation results...")
        self.save_results(metrics, cm, class_report)
        
        print(f"\n{'='*60}")
        print(f"✓ Evaluation complete!")
        print(f"{'='*60}\n")
        
        return {
            "metrics": metrics,
            "confusion_matrix": cm,
            "classification_report": class_report,
            "evaluation_report": eval_report
        }


def main():
    import argparse
    from torchvision.datasets import ImageFolder
    from torch.utils.data import DataLoader
    from preprocessing.image_transforms import get_val_transform

    parser = argparse.ArgumentParser(description="Evaluate Rice model")
    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="Path to trained model checkpoint"
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        default="Dataset_Crop",
        help="Root dataset directory"
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=32
    )
    parser.add_argument(
        "--variant",
        type=str,
        default="b0"
    )
    parser.add_argument(
        "--dropout",
        type=float,
        default=0.5
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="evaluation_results/rice"
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu"
    )

    args = parser.parse_args()
    device = torch.device(args.device)

    print(f"Using device: {device}")

    # ----------------------------
    # Validation dataset ONLY
    # ----------------------------
    val_dir = os.path.join(args.data_dir, "Rice")

    val_dataset = ImageFolder(
        root=val_dir,
        transform=get_val_transform(input_size=224)
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,      # 🔥 CRITICAL
        num_workers=0
    )

    idx_to_class = {v: k for k, v in val_dataset.class_to_idx.items()}

    print("\n✓ Dataset loaded successfully")
    print(f"  Classes: {list(idx_to_class.values())}")
    print(f"  Val batches: {len(val_loader)}")

    # ----------------------------
    # Load model
    # ----------------------------
    print("\nLoading checkpoint...")
    checkpoint = torch.load(args.checkpoint, map_location=device)

    model = create_model(
        num_classes=len(idx_to_class),
        variant=args.variant,
        pretrained=False,
        dropout=args.dropout,
        freeze_backbone=False,
        device=str(device)
    )

    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()

    print("✓ Model loaded from checkpoint")

    # ----------------------------
    # Run evaluation
    # ----------------------------
    evaluator = RiceModelEvaluator(
        model=model,
        val_loader=val_loader,
        idx_to_class=idx_to_class,
        device=device,
        output_dir=args.output_dir
    )

    evaluator.evaluate()

if __name__ == "__main__":
    main()
