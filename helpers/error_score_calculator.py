import torch
import json
from pathlib import Path
from tqdm import tqdm
from typing import Dict
import sys
import os
from ultralytics import YOLO
from datetime import datetime

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from dataloader import create_dataloaders
from score_metrics import match_predictions


class ErrorScoreCalculator:
    """
    Calculator that computes error scores for batches of image pairs using YOLO predictions.
    This version includes timestamped output directories to track different analysis runs.
    """

    def __init__(self, model_path: str, device: torch.device):
        """
        Initialize the calculator with YOLO model

        Args:
            model_path: Path to the YOLO model weights
            device: Device to run calculations on (CPU or CUDA)
        """
        self.device = device
        # Initialize YOLO model for predictions
        self.yolo_model = YOLO(model_path, verbose=False)
        self.yolo_model.model.eval()

    def process_batch(
        self,
        gt_images: torch.Tensor,
        mod_images: torch.Tensor,
    ) -> torch.Tensor:
        """
        Process a batch of image pairs to get error scores

        Args:
            gt_images: Batch of ground truth images [B, C, H, W]
            mod_images: Batch of modified images [B, C, H, W]

        Returns:
            Tensor of error scores for the batch
        """
        with torch.no_grad():
            # Get YOLO predictions for error score calculation
            gt_predictions = self.yolo_model.predict(gt_images, verbose=False)
            mod_predictions = self.yolo_model.predict(mod_images, verbose=False)

            # Calculate error scores
            batch_matches = match_predictions(gt_predictions, mod_predictions)
            error_scores = torch.tensor(
                [match["error_score"] for match in batch_matches], device=self.device
            )

            return error_scores

    def process_split(
        self,
        dataloader: torch.utils.data.DataLoader,
        output_dir: Path,
    ) -> Dict[str, float]:
        """
        Process a dataset split and calculate error scores

        Args:
            dataloader: DataLoader containing image pairs
            output_dir: Directory to save error scores

        Returns:
            Dictionary mapping image names to their error scores
        """
        scores_dict = {}

        # Process all batches
        for batch in tqdm(dataloader, desc="Processing split"):
            gt_images = batch["gt"].to(self.device)
            mod_images = batch["compressed"].to(self.device)
            names = batch["name"]

            # Get scores
            batch_scores = self.process_batch(gt_images, mod_images)

            # Store scores
            for name, score in zip(names, batch_scores):
                scores_dict[name] = float(score)

        # Create output directory if it doesn't exist
        output_dir.mkdir(parents=True, exist_ok=True)

        # Save scores
        scores_file = output_dir / "error_scores.json"
        with open(scores_file, "w") as f:
            json.dump(scores_dict, f, indent=4)

        print(f"Saved scores to: {scores_file}")
        return scores_dict


def get_timestamp_dir() -> str:
    """
    Generate a formatted timestamp string for directory naming

    Returns:
        Formatted timestamp string (e.g., '2025_01_11_142159')
    """
    return datetime.now().strftime("%Y_%m_%d_%H%M%S")


def main():
    """
    Main function to calculate error scores for all splits with timestamped output
    """
    # Configuration
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    DATA_ROOT = "dataset_attention"
    OUTPUT_ROOT = "error_scores_analysis"
    BATCH_SIZE = 256
    MODEL_PATH = "../yolo11m.pt"

    # Create timestamped directory
    timestamp = get_timestamp_dir()
    output_root = Path(OUTPUT_ROOT) / timestamp

    print(f"Using device: {device}")
    print(f"Processing dataset from: {DATA_ROOT}")
    print(f"Saving scores to: {output_root}")
    print(f"Using YOLO model: {MODEL_PATH}")

    # Initialize calculator
    calculator = ErrorScoreCalculator(MODEL_PATH, device)

    # Create dataloaders
    train_loader, val_loader, test_loader = create_dataloaders(
        dataset_root=DATA_ROOT, batch_size=BATCH_SIZE
    )

    # Process each split
    for split, loader in [
        ("train", train_loader),
        ("val", val_loader),
        ("test", test_loader),
    ]:
        split_dir = output_root / split
        print(f"\nMaking prediction for {split} split...")
        scores = calculator.process_split(loader, split_dir)
        print(f"Completed {split} split: {len(scores)} images processed")

    print("\nProcessing complete!")


if __name__ == "__main__":
    main()
