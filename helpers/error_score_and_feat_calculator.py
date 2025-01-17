import torch
import json
import numpy as np
from pathlib import Path
from tqdm import tqdm
from typing import Dict, Tuple, Optional
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
    Enhanced calculator that extracts SPPF features and optionally calculates
    error scores for batches of image pairs using YOLO predictions.
    """

    def __init__(
        self, model_path: str, device: torch.device, extract_scores: bool = False
    ):
        """
        Initialize the calculator with YOLO model and feature extractor

        Args:
            model_path: Path to the YOLO model weights
            device: Device to run calculations on (CPU or CUDA)
            extract_scores: Whether to extract error scores
        """
        self.device = device
        self.extract_scores = extract_scores

        # Initialize feature extractor
        self.feature_extractor = load_feature_extractor(model_path)
        self.feature_extractor.to(device)
        self.feature_extractor.eval()

        # Initialize YOLO model only if needed
        if self.extract_scores:
            self.yolo_model = YOLO(model_path, verbose=False)
            self.yolo_model.model.eval()
        else:
            self.yolo_model = None

    @staticmethod
    def clean_directory(directory: Path) -> None:
        """
        Safely removes and recreates a directory.
        """
        if directory.exists():
            shutil.rmtree(directory)
        directory.mkdir(parents=True, exist_ok=True)

    def process_batch(
        self,
        gt_images: torch.Tensor,
        mod_images: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
        """
        Process a batch of image pairs to get features and optionally scores

        Args:
            gt_images: Batch of ground truth images [B, C, H, W]
            mod_images: Batch of modified images [B, C, H, W]

        Returns:
            Tuple of (gt_features, mod_features, error_scores)
            error_scores will be None if extract_scores is False
        """
        with torch.no_grad():
            # Extract SPPF features
            gt_features, mod_features = self.feature_extractor.extract_features(
                gt_images, mod_images
            )

            # Calculate error scores if requested
            if self.extract_scores:
                gt_predictions = self.yolo_model.predict(gt_images, verbose=False)
                mod_predictions = self.yolo_model.predict(mod_images, verbose=False)
                batch_matches = match_predictions(gt_predictions, mod_predictions)
                error_scores = torch.tensor(
                    [match["error_score"] for match in batch_matches],
                    device=self.device,
                )
            else:
                error_scores = None

            return gt_features, mod_features, error_scores

    def save_features(
        self, features: torch.Tensor, names: list, output_dir: Path
    ) -> None:
        """
        Save extracted features to numpy files
        """
        for feat, name in zip(features, names):
            feat_np = feat.cpu().numpy()
            feat_path = output_dir / f"{Path(name).stem}.npy"
            np.save(feat_path, feat_np)

    def process_split(
        self,
        dataloader: torch.utils.data.DataLoader,
        split_path: Path,
        features_root: Path,
    ) -> Optional[Dict[str, float]]:
        """
        Process a dataset split: extract features and optionally calculate scores

        Args:
            dataloader: DataLoader containing image pairs
            split_path: Path to the split directory
            features_root: Root directory for saving features

        Returns:
            Dictionary mapping image names to their error scores if extract_scores is True,
            None otherwise
        """
        scores_dict = {} if self.extract_scores else None

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

            # Get features and optionally scores
            gt_features, mod_features, batch_scores = self.process_batch(
                gt_images, mod_images
            )

            # Save features
            self.save_features(gt_features, names, gt_feat_dir)
            self.save_features(mod_features, names, mod_feat_dir)

            # Store scores if they were calculated
            if self.extract_scores and batch_scores is not None:
                for name, score in zip(names, batch_scores):
                    scores_dict[name] = float(score)

        # Save scores if they were calculated
        if self.extract_scores and scores_dict:
            scores_file = split_path / "error_scores.json"
            with open(scores_file, "w") as f:
                json.dump(scores_dict, f, indent=4)
            print(f"Saved scores to: {scores_file}")

        print(f"Saved features to: {gt_feat_dir} and {mod_feat_dir}")
        return scores_dict


def main():
    """
    Main function to process all splits: extract features and optionally calculate scores
    """
    # Configuration
    device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
    DATA_ROOT = "balanced_dataset"
    FEATURES_ROOT = "feature_extracted"
    BATCH_SIZE = 128
    MODEL_PATH = "../yolo11m.pt"
    EXTRACT_SCORES = False  # Set to True if you want to calculate error scores

    print(f"Using device: {device}")
    print(f"Processing dataset from: {DATA_ROOT}")
    print(f"Saving features to: {FEATURES_ROOT}")
    print(f"Using YOLO model: {MODEL_PATH}")
    print(f"Extracting error scores: {EXTRACT_SCORES}")

    # Clean entire features directory at start
    features_root = Path(FEATURES_ROOT)
    if features_root.exists():
        print(f"Cleaning existing features directory: {FEATURES_ROOT}")
        shutil.rmtree(features_root)

    # Initialize enhanced calculator
    calculator = EnhancedBatchCalculator(MODEL_PATH, device, EXTRACT_SCORES)

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
        calculator.process_split(loader, split_path, features_root)
        processed_count = len(loader.dataset)
        print(f"Completed {split} split: {processed_count} images processed")

    print("\nProcessing complete!")


if __name__ == "__main__":
    main()
