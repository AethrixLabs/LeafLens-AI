import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from training.core_trainer import train_crop


if __name__ == "__main__":
    train_crop(
        crop_name="corn",
        num_epochs=25,
        batch_size=16,
        learning_rate=1e-4
    )