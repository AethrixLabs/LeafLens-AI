"""Create a dummy checkpoint for testing evaluation script"""
import torch
import sys
from pathlib import Path

sys.path.insert(0, str(Path.cwd()))

from models.efficientnet_model import create_model

# Create a dummy pretrained model checkpoint
print("Creating dummy checkpoint...")
model = create_model(
    num_classes=4,  # 4 rice disease classes
    variant="b0",
    pretrained=True,
    dropout=0.5,
    freeze_backbone=False,
    device="cpu"
)

checkpoint = {
    "epoch": 0,
    "model_state_dict": model.state_dict(),
    "optimizer_state_dict": {},
    "scheduler_state_dict": {},
    "metrics": {"accuracy": 0.75, "loss": 0.5},
    "model_info": model.get_model_info(),
    "idx_to_class": {
        0: "Bacterial_blight",
        1: "Blast",
        2: "Brown_spot",
        3: "Tungro"
    }
}

Path("checkpoints").mkdir(exist_ok=True)
torch.save(checkpoint, "checkpoints/rice_efficientnet_b0_best.pt")
print("✓ Checkpoint created: checkpoints/rice_efficientnet_b0_best.pt")
