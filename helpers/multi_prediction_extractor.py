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


def get_timestamp_dir() -> str:
    """Generate a formatted timestamp string for directory naming"""
    return datetime.now().strftime("%Y_%m_%d_%H%M%S")


class YOLOPredictionExtractor:
    """Extracts and saves YOLO predictions for ground truth and compressed images"""

    def __init__(
        self,
        model_path: str,
        device: torch.device,
        quality_values: List[int],
    ):
        self.device = device
        self.quality_values = quality_values
        self.yolo_model = YOLO(model_path, verbose=False)
        self.yolo_model.to(device)
        self.yolo_model.model.eval()

    def extract_predictions(
        self,
        gt_images: torch.Tensor,
        compressed_images: Dict[int, torch.Tensor],
    ) -> Dict[str, List]:
        """Extract predictions for a batch of images"""
        with torch.no_grad():
            # Get predictions for ground truth images
            gt_preds = self.yolo_model.predict(gt_images, verbose=False)

            # Convert predictions to serializable format
            batch_predictions = {"gt": self._format_predictions(gt_preds)}

            # Get predictions for each compression quality
            for quality, comp_images in compressed_images.items():
                comp_preds = self.yolo_model.predict(comp_images, verbose=False)
                batch_predictions[f"comp_{quality}"] = self._format_predictions(
                    comp_preds
                )

            return batch_predictions

    def _format_predictions(self, predictions) -> List[Dict]:
        """Convert YOLO predictions to serializable format"""
        formatted_preds = []
        for pred in predictions:
            boxes = pred.boxes
            pred_dict = {
                "boxes": boxes.xyxy.cpu().numpy().tolist(),
                "scores": boxes.conf.cpu().numpy().tolist(),
                "classes": boxes.cls.cpu().numpy().tolist(),
            }
            formatted_preds.append(pred_dict)
        return formatted_preds

    def process_dataset(
        self,
        dataloader: torch.utils.data.DataLoader,
        output_dir: Path,
    ):
        """Process entire dataset and save predictions"""
        predictions_dict = {}

        for batch in tqdm(dataloader, desc="Extracting predictions"):
            gt_images = batch["gt"].to(self.device)
            compressed_batch = {
                q: batch["compressed"][q].to(self.device) for q in self.quality_values
            }
            names = batch["name"]

            # Get predictions for this batch
            batch_predictions = self.extract_predictions(gt_images, compressed_batch)

            # Store predictions for each image
            for idx, name in enumerate(names):
                predictions_dict[name] = {
                    k: v[idx] for k, v in batch_predictions.items()
                }

        # Save predictions
        output_dir.mkdir(parents=True, exist_ok=True)
        predictions_file = output_dir / "predictions.json"
        with open(predictions_file, "w") as f:
            json.dump(predictions_dict, f, indent=4)

        print(f"Saved predictions to: {predictions_file}")
        return predictions_dict


def main():
    # Configuration
    device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
    DATA_ROOT = "unbalanced_dataset"
    OUTPUT_ROOT = "predictions_analysis"
    BATCH_SIZE = 128
    MODEL_PATH = "../yolo11m.pt"
    QUALITY_VALUES = [20, 24, 28, 32, 36, 40, 50]

    # Create output directory with timestamp
    timestamp = get_timestamp_dir()
    output_root = Path(OUTPUT_ROOT) / timestamp

    print(f"Using device: {device}")
    print(f"Processing dataset from: {DATA_ROOT}")
    print(f"Saving predictions to: {output_root}")

    # Initialize extractor
    extractor = YOLOPredictionExtractor(MODEL_PATH, device, QUALITY_VALUES)

    # Create dataloaders
    train_loader = create_multi_compression_dataloaders(
        dataset_root=DATA_ROOT, batch_size=BATCH_SIZE, quality_values=QUALITY_VALUES
    )

    # Process dataset
    for split, loader in [("total", train_loader)]:
        split_dir = output_root / split
        print(f"\nExtracting predictions for {split} split...")
        predictions = extractor.process_dataset(loader, split_dir)
        print(f"Completed {split} split: {len(predictions)} images processed")

    print("\nPrediction extraction complete!")


if __name__ == "__main__":
    main()
