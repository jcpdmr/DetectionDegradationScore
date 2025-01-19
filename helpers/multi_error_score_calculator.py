import torch
import json
from pathlib import Path
from tqdm import tqdm
from typing import Dict, List
import sys
import os
from ultralytics import YOLO
from datetime import datetime

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from dataloader import create_multi_compression_dataloaders
from score_metrics import match_predictions


class MultiCompressionErrorScoreCalculator:
    """
    Calculator that computes error scores for images with multiple compression levels using YOLO predictions.
    """

    def __init__(
        self,
        model_path: str,
        device: torch.device,
        quality_values: List[int],
    ):
        """
        Initialize the calculator with YOLO model

        Args:
            model_path: Path to the YOLO model weights
            device: Device to run calculations on (CPU or CUDA)
            quality_values: List of compression quality values to process
        """
        self.device = device
        self.quality_values = quality_values
        # Initialize YOLO model for predictions
        self.yolo_model = YOLO(model_path, verbose=False)
        self.yolo_model.to(device)
        self.yolo_model.model.eval()

    def process_batch(
        self,
        gt_images: torch.Tensor,
        compressed_images: Dict[int, torch.Tensor],
    ) -> Dict[int, torch.Tensor]:
        """
        Process a batch of images with their multiple compressed versions

        Args:
            gt_images: Batch of ground truth images [B, C, H, W]
            compressed_images: Dictionary mapping quality values to batches of compressed images

        Returns:
            Dictionary mapping quality values to error scores for the batch
        """
        with torch.no_grad():
            # Get predictions for ground truth images
            gt_predictions = self.yolo_model.predict(gt_images, verbose=False)

            # Process each compression quality
            quality_scores = {}
            for quality, mod_images in compressed_images.items():
                # Get predictions for modified images
                mod_predictions = self.yolo_model.predict(mod_images, verbose=False)

                # Calculate error scores
                batch_matches = match_predictions(gt_predictions, mod_predictions)
                error_scores = torch.tensor(
                    [match["error_score"] for match in batch_matches],
                    device=self.device,
                )
                quality_scores[quality] = error_scores

            return quality_scores

    def process_split(
        self,
        dataloader: torch.utils.data.DataLoader,
        output_dir: Path,
    ) -> Dict[str, Dict[int, float]]:
        """
        Process a dataset split and calculate error scores for all compression levels

        Args:
            dataloader: DataLoader containing image sets
            output_dir: Directory to save error scores

        Returns:
            Nested dictionary mapping image names to their error scores for each quality
        """
        scores_dict = {}

        # Process all batches
        for batch in tqdm(dataloader, desc="Processing split"):
            gt_images = batch["gt"].to(self.device)
            compressed_batch = {
                q: batch["compressed"][q].to(self.device) for q in self.quality_values
            }
            names = batch["name"]

            # Get scores for all qualities
            batch_scores = self.process_batch(gt_images, compressed_batch)

            # Store scores
            for name in names:
                scores_dict[name] = {
                    quality: float(batch_scores[quality][names.index(name)])
                    for quality in self.quality_values
                }

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
    Main function to calculate error scores for multiple compression levels
    """
    # Configuration
    device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
    DATA_ROOT = "unbalanced_dataset"
    OUTPUT_ROOT = "error_scores_analysis/mapping"
    BATCH_SIZE = 128
    MODEL_PATH = "../yolo11m.pt"
    QUALITY_VALUES = [40, 45, 50, 55, 60, 70]

    # Create timestamped directory
    timestamp = get_timestamp_dir()
    output_root = Path(OUTPUT_ROOT) / timestamp

    print(f"Using device: {device}")
    print(f"Processing dataset from: {DATA_ROOT}")
    print(f"Saving scores to: {output_root}")
    print(f"Using YOLO model: {MODEL_PATH}")
    print(f"Processing compression qualities: {QUALITY_VALUES}")

    # Initialize calculator
    calculator = MultiCompressionErrorScoreCalculator(
        MODEL_PATH, device, quality_values=QUALITY_VALUES
    )

    # Create dataloaders
    train_loader = create_multi_compression_dataloaders(
        dataset_root=DATA_ROOT, batch_size=BATCH_SIZE, quality_values=QUALITY_VALUES
    )

    # Process each split
    for split, loader in [
        ("total", train_loader),
        # ("val", val_loader),
        # ("test", test_loader),
    ]:
        split_dir = output_root / split
        print(f"\nMaking predictions for {split} split...")
        scores = calculator.process_split(loader, split_dir)
        print(f"Completed {split} split: {len(scores)} images processed")

    print("\nProcessing complete!")


if __name__ == "__main__":
    main()
