import os
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
from typing import Tuple, Dict, Union, Generator
from pathlib import Path
import json
import mmap
from contextlib import contextmanager


class FeatureMapsDataset(Dataset):
    """
    Dataset class that loads pre-extracted SPPF feature maps and error scores.
    Handles file name matching between feature files and scores.
    """

    def __init__(self, features_root: str, error_scores_root: str, split: str):
        """
        Initialize the dataset for loading feature maps and scores.

        Args:
            features_root: Root directory containing feature maps and scores
            split: Dataset split ('train', 'val', or 'test')
        """
        if split not in ["train", "val", "test"]:
            raise ValueError("Split must be one of: 'train', 'val', 'test'")

        # Setup paths
        self.split_path = Path(features_root) / split
        self.gt_features_path = self.split_path / "extracted"
        self.mod_features_path = self.split_path / "compressed"
        self.scores_path = Path(error_scores_root) / split / "error_scores.json"

        # Verify directories and load scores
        self._verify_directories()
        self._load_scores()

        # Get list of valid feature files
        self.feature_names = self._find_valid_features()

        if not self.feature_names:
            print("Debug info:")
            print(f"GT features path: {self.gt_features_path}")
            print(f"Available feature files: {os.listdir(self.gt_features_path)}")
            print(f"Available scores: {list(self.scores.keys())}")
            raise RuntimeError(
                f"No valid feature maps found in {self.gt_features_path}"
            )

    def _get_base_name(self, filename: str) -> str:
        """
        Extract base name without extension from filename.

        Args:
            filename: Input filename with extension

        Returns:
            Base filename without extension
        """
        return Path(filename).stem

    def _find_valid_features(self) -> list:
        """
        Find all valid feature files that have corresponding scores.
        Matches files based on their base names without extensions.

        Returns:
            List of valid feature filenames
        """
        valid_features = []

        # Convert score filenames to base names for easier matching
        score_base_names = {self._get_base_name(k) for k in self.scores.keys()}

        # Check each feature file
        for f in os.listdir(self.gt_features_path):
            if not f.endswith(".npy"):
                continue

            # Get base name of feature file
            feature_base_name = self._get_base_name(f)

            # Check if we have a matching score
            if feature_base_name in score_base_names:
                # Verify the corresponding modified feature exists
                if (self.mod_features_path / f).exists():
                    valid_features.append(f)
                else:
                    print(f"Warning: Missing modified feature for {f}")

        valid_features.sort()  # Ensure consistent ordering

        # Debug information about matching process
        print(f"Found {len(valid_features)} valid feature pairs")
        print("First few matches:")
        for f in valid_features[:3]:
            base_name = self._get_base_name(f)
            print(f"  Feature: {f} -> Score: {base_name}.jpg")

        return valid_features

    @staticmethod
    @contextmanager
    def _open_numpy_file(
        file_path: Union[str, Path],
    ) -> Generator[np.ndarray, None, None]:
        """
        Safely and efficiently open a numpy file using memory mapping.
        Uses proper type hints for the context manager that yields numpy arrays.

        Args:
            file_path: Path to the .npy file

        Yields:
            Memory-mapped numpy array with shape (512, 12, 12) and dtype float32
        """
        with open(file_path, "rb") as f:
            # Memory map the file for efficient reading
            mm = mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ)
            # Load the array using memory mapping
            array = np.load(mm, allow_pickle=False)
            try:
                yield array
            finally:
                # Ensure proper cleanup of resources
                mm.close()

    def _verify_directories(self) -> None:
        """Verify that all required directories and files exist."""
        if not self.gt_features_path.exists():
            raise RuntimeError(
                f"GT features directory not found: {self.gt_features_path}"
            )
        if not self.mod_features_path.exists():
            raise RuntimeError(
                f"Modified features directory not found: {self.mod_features_path}"
            )
        if not self.scores_path.exists():
            raise RuntimeError(
                f"Error scores file not found: {self.scores_path}\n"
                "Please run the score calculation script first."
            )

    def _load_scores(self) -> None:
        """Load error scores from the JSON file."""
        try:
            with open(self.scores_path, "r") as f:
                self.scores = json.load(f)
        except json.JSONDecodeError:
            raise RuntimeError(f"Invalid JSON file: {self.scores_path}")
        except Exception as e:
            raise RuntimeError(f"Error loading scores: {str(e)}")

    def __len__(self) -> int:
        """Return the number of valid feature map pairs"""
        return len(self.feature_names)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        Get a single pair of feature maps and their error score.

        Returns:
            Dictionary containing features and score
        """
        feat_name = self.feature_names[idx]
        base_name = self._get_base_name(feat_name)

        # Load features using memory mapping
        with self._open_numpy_file(self.gt_features_path / feat_name) as gt_array:
            gt_features = torch.from_numpy(gt_array.copy())

        with self._open_numpy_file(self.mod_features_path / feat_name) as mod_array:
            mod_features = torch.from_numpy(mod_array.copy())

        # Get corresponding score using the original image filename
        score = torch.tensor(self.scores[f"{base_name}.jpg"], dtype=torch.float32)

        return {
            "gt_features": gt_features,
            "mod_features": mod_features,
            "score": score,
            "name": feat_name,
        }


def create_feature_dataloaders(
    features_root: str,
    error_scores_root: str,
    batch_size: int,
    num_workers: int = 4,
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    Create dataloaders for feature maps across all splits.

    Args:
        features_root: Root directory containing the feature maps
        batch_size: Batch size for the dataloaders
        num_workers: Number of worker processes for loading

    Returns:
        Tuple of (train_loader, val_loader, test_loader)
    """
    loaders = {}

    for split in ["train", "val", "test"]:
        dataset = FeatureMapsDataset(
            features_root=features_root,
            error_scores_root=error_scores_root,
            split=split,
        )

        loaders[split] = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=(split == "train"),
            num_workers=num_workers,
            pin_memory=True,
        )

    return loaders["train"], loaders["val"], loaders["test"]
