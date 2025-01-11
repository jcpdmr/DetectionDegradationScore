import torch
import json
import numpy as np
from pathlib import Path
from tqdm import tqdm
from typing import Dict, Tuple
import sys
import os
from ultralytics import YOLO
import shutil

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from dataloader import create_dataloaders
from score_metrics import match_predictions
from extractor import load_feature_extractor


class EnhancedBatchCalculator:
    """
    Enhanced calculator that extracts SPPF features and calculates error scores
    for batches of image pairs using YOLO predictions.
    """

    def __init__(self, model_path: str, device: torch.device):
        """
        Initialize the calculator with YOLO model and feature extractor

        Args:
            model_path: Path to the YOLO model weights
            device: Device to run calculations on (CPU or CUDA)
        """
        self.device = device
        # Initialize YOLO model for predictions
        self.yolo_model = YOLO(model_path, verbose=False)
        self.yolo_model.model.eval()

        # Initialize feature extractor
        self.feature_extractor = load_feature_extractor(model_path)
        self.feature_extractor.to(device)
        self.feature_extractor.eval()

    @staticmethod
    def clean_directory(directory: Path) -> None:
        """
        Safely removes and recreates a directory. The function ensures that we start
        with a clean state, preventing any mixing of old and new features.

        Args:
            directory: Path to the directory to clean
        """
        # Remove directory and its contents if it exists
        if directory.exists():
            shutil.rmtree(directory)

        # Recreate empty directory
        directory.mkdir(parents=True, exist_ok=True)

    def process_batch(
        self,
        gt_images: torch.Tensor,
        mod_images: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Process a batch of image pairs to get scores and features

        Args:
            gt_images: Batch of ground truth images [B, C, H, W]
            mod_images: Batch of modified images [B, C, H, W]

        Returns:
            Tuple of (error_scores, gt_features, mod_features)
        """
        with torch.no_grad():
            # Extract SPPF features
            gt_features, mod_features = self.feature_extractor.extract_features(
                gt_images, mod_images
            )

            # Get YOLO predictions for error score calculation
            gt_predictions = self.yolo_model.predict(gt_images, verbose=False)
            mod_predictions = self.yolo_model.predict(mod_images, verbose=False)

            # Calculate error scores
            batch_matches = match_predictions(gt_predictions, mod_predictions)
            error_scores = torch.tensor(
                [match["error_score"] for match in batch_matches], device=self.device
            )

            return error_scores, gt_features, mod_features

    def save_features(
        self, features: torch.Tensor, names: list, output_dir: Path
    ) -> None:
        """
        Save extracted features to numpy files

        Args:
            features: Batch of features [B, C, H, W]
            names: List of image names
            output_dir: Directory to save features
        """
        for feat, name in zip(features, names):
            # Convert feature tensor to numpy and save
            feat_np = feat.cpu().numpy()
            feat_path = output_dir / f"{Path(name).stem}.npy"
            np.save(feat_path, feat_np)
            # np.savez_compressed(feat_path, feat_np)

    def process_split(
        self,
        dataloader: torch.utils.data.DataLoader,
        split_path: Path,
        features_root: Path,
    ) -> Dict[str, float]:
        """
        Process a dataset split: calculate scores and extract features
        Now includes directory cleanup before processing.

        Args:
            dataloader: DataLoader containing image pairs
            split_path: Path to the split directory
            features_root: Root directory for saving features

        Returns:
            Dictionary mapping image names to their error scores
        """
        scores_dict = {}

        # Create feature directories for this split
        split_name = split_path.name
        gt_feat_dir = features_root / split_name / "extracted"
        mod_feat_dir = features_root / split_name / "compressed"

        # Clean directories before processing
        self.clean_directory(gt_feat_dir)
        self.clean_directory(mod_feat_dir)

        # Process all batches
        for batch in tqdm(dataloader, desc=f"Processing {split_name} split"):
            gt_images = batch["gt"].to(self.device)
            mod_images = batch["compressed"].to(self.device)
            names = batch["name"]

            # Get scores and features
            batch_scores, gt_features, mod_features = self.process_batch(
                gt_images, mod_images
            )

            # Save features
            self.save_features(gt_features, names, gt_feat_dir)
            self.save_features(mod_features, names, mod_feat_dir)

            # Store scores
            for name, score in zip(names, batch_scores):
                scores_dict[name] = float(score)

        # Save scores for this split
        scores_file = split_path / "error_scores.json"
        with open(scores_file, "w") as f:
            json.dump(scores_dict, f, indent=4)

        print(f"Saved scores to: {scores_file}")
        print(f"Saved features to: {gt_feat_dir} and {mod_feat_dir}")
        return scores_dict


def main():
    """
    Main function to process all splits: calculate scores and extract features
    """
    # Configuration
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    DATA_ROOT = "dataset_attention"
    FEATURES_ROOT = "feature_extracted_attention"
    BATCH_SIZE = 128
    MODEL_PATH = "../yolo11m.pt"

    print(f"Using device: {device}")
    print(f"Processing dataset from: {DATA_ROOT}")
    print(f"Saving features to: {FEATURES_ROOT}")
    print(f"Using YOLO model: {MODEL_PATH}")

    # Clean entire features directory at start
    features_root = Path(FEATURES_ROOT)
    if features_root.exists():
        print(f"Cleaning existing features directory: {FEATURES_ROOT}")
        shutil.rmtree(features_root)

    # Initialize enhanced calculator
    calculator = EnhancedBatchCalculator(MODEL_PATH, device)

    # Create dataloaders
    train_loader, val_loader, test_loader = create_dataloaders(
        dataset_root=DATA_ROOT, batch_size=BATCH_SIZE
    )

    # Process each split
    data_root = Path(DATA_ROOT)
    features_root = Path(FEATURES_ROOT)

    for split, loader in [
        ("train", train_loader),
        ("val", val_loader),
        ("test", test_loader),
    ]:
        split_path = data_root / split
        print(f"\nProcessing {split} split...")
        scores = calculator.process_split(loader, split_path, features_root)
        print(f"Completed {split} split: {len(scores)} images processed")

    print("\nProcessing complete!")


if __name__ == "__main__":
    main()
