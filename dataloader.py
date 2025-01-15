import os
import cv2
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from typing import Tuple, Dict
from pathlib import Path
import json


class ImagePairDataset(Dataset):
    """
    Dataset for loading pairs of original and compressed images
    """

    def __init__(self, root_path: str, split: str):
        """
        Initialize dataset

        Args:
            root_path: Root directory containing the dataset
            split: Dataset split ('train', 'val', 'test')
        """
        if split not in ["train", "val", "test"]:
            raise ValueError("Split must be one of: train, val, test")

        # Setup paths
        self.split_path = Path(root_path) / split
        self.gt_path = self.split_path / "extracted"
        self.compressed_path = self.split_path / "compressed"

        # Verify paths exist
        if not self.gt_path.exists():
            raise RuntimeError(f"Ground truth directory not found: {self.gt_path}")
        if not self.compressed_path.exists():
            raise RuntimeError(
                f"Compressed images directory not found: {self.compressed_path}"
            )

        # Get list of valid image files
        self.image_names = sorted(
            [
                f
                for f in os.listdir(self.gt_path)
                if f.lower().endswith((".jpg", ".jpeg", ".png"))
                and (self.compressed_path / f).exists()
            ]
        )

        if not self.image_names:
            raise RuntimeError(f"No valid image pairs found in {self.gt_path}")

        # Image transformations
        self.transform = transforms.ToTensor()

    def __len__(self) -> int:
        return len(self.image_names)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        Get image pair

        Returns:
            Dictionary containing:
            - gt: Ground truth image tensor [3, H, W]
            - compressed: Compressed image tensor [3, H, W]
            - name: Image filename
        """
        img_name = self.image_names[idx]

        # Load images
        gt_img = cv2.imread(str(self.gt_path / img_name))
        gt_img = cv2.cvtColor(gt_img, cv2.COLOR_BGR2RGB)
        gt_tensor = self.transform(gt_img)

        compressed_img = cv2.imread(str(self.compressed_path / img_name))
        compressed_img = cv2.cvtColor(compressed_img, cv2.COLOR_BGR2RGB)
        compressed_tensor = self.transform(compressed_img)

        return {"gt": gt_tensor, "compressed": compressed_tensor, "name": img_name}


class MultiCompressionDataset(Dataset):
    """
    Dataset for loading original images and their multiple compressed versions
    """

    def __init__(
        self, root_path: str, split: str, quality_values: list = [10, 20, 30, 40, 50]
    ):
        """
        Initialize dataset

        Args:
            root_path: Root directory containing the dataset
            split: Dataset split ('train', 'val', 'test')
            quality_values: List of compression quality values
        """
        if split not in ["train", "val", "test"]:
            raise ValueError("Split must be one of: train, val, test")

        # Setup paths
        self.split_path = Path(root_path) / split
        self.gt_path = self.split_path / "extracted"
        self.quality_values = quality_values

        # Create dictionary of compressed paths for each quality
        self.compressed_paths = {
            q: self.split_path / f"compressed{q}" for q in quality_values
        }

        # Verify paths exist
        if not self.gt_path.exists():
            raise RuntimeError(f"Ground truth directory not found: {self.gt_path}")

        for q, path in self.compressed_paths.items():
            if not path.exists():
                raise RuntimeError(
                    f"Compressed directory for quality {q} not found: {path}"
                )

        # Get list of valid image files (must exist in all directories)
        self.image_names = sorted(
            [
                f
                for f in os.listdir(self.gt_path)
                if f.lower().endswith((".jpg", ".jpeg", ".png"))
                and all((self.compressed_paths[q] / f).exists() for q in quality_values)
            ]
        )

        if not self.image_names:
            raise RuntimeError(f"No valid image sets found in {self.gt_path}")

        # Image transformations
        self.transform = transforms.ToTensor()

    def __len__(self) -> int:
        return len(self.image_names)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        Get image set (original + compressed versions)

        Returns:
            Dictionary containing:
            - gt: Ground truth image tensor [3, H, W]
            - compressed: Dictionary of compressed image tensors for each quality value
            - name: Image filename
        """
        img_name = self.image_names[idx]

        # Load ground truth image
        gt_img = cv2.imread(str(self.gt_path / img_name))
        gt_img = cv2.cvtColor(gt_img, cv2.COLOR_BGR2RGB)
        gt_tensor = self.transform(gt_img)

        # Load compressed versions
        compressed_tensors = {}
        for quality in self.quality_values:
            compressed_img = cv2.imread(str(self.compressed_paths[quality] / img_name))
            compressed_img = cv2.cvtColor(compressed_img, cv2.COLOR_BGR2RGB)
            compressed_tensors[quality] = self.transform(compressed_img)

        return {"gt": gt_tensor, "compressed": compressed_tensors, "name": img_name}


class FeatureMapDataset(Dataset):
    """
    Dataset for loading extracted feature maps and their error scores
    """

    def __init__(self, features_root: str, split: str):
        """
        Initialize dataset

        Args:
            features_root: Root directory containing feature maps
            split: Dataset split ('train', 'val', 'test')
        """
        if split not in ["train", "val", "test"]:
            raise ValueError("Split must be one of: train, val, test")

        # Setup paths
        self.split_path = Path(features_root) / split
        self.gt_path = self.split_path / "extracted"
        self.compressed_path = self.split_path / "compressed"
        self.scores_path = Path("dataset_attention") / split / "error_scores.json"

        # Verify paths exist
        if not self.gt_path.exists():
            raise RuntimeError(
                f"Ground truth features directory not found: {self.gt_path}"
            )
        if not self.compressed_path.exists():
            raise RuntimeError(
                f"Compressed features directory not found: {self.compressed_path}"
            )
        if not self.scores_path.exists():
            raise RuntimeError(f"Scores file not found: {self.scores_path}")

        # Load scores
        with open(self.scores_path) as f:
            self.scores = json.load(f)

        # Get list of valid feature files
        self.feature_names = sorted(
            [
                f
                for f in os.listdir(self.gt_path)
                if f.endswith(".npy")
                and Path(self.compressed_path / f).exists()
                and f"{Path(f).stem}.jpg" in self.scores
            ]
        )

        if not self.feature_names:
            raise RuntimeError(f"No valid feature maps found in {self.gt_path}")

    def __len__(self) -> int:
        return len(self.feature_names)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        Get feature map pair and score

        Returns:
            Dictionary containing:
            - gt_features: Ground truth features [512, 12, 12]
            - compressed_features: Compressed image features [512, 12, 12]
            - score: Error score [1]
            - name: Feature file name
        """
        feat_name = self.feature_names[idx]

        # Load feature maps
        gt_features = torch.from_numpy(np.load(self.gt_path / feat_name))
        compressed_features = torch.from_numpy(
            np.load(self.compressed_path / feat_name)
        )

        # Get score using original image name
        score = torch.tensor(
            self.scores[f"{Path(feat_name).stem}.jpg"], dtype=torch.float32
        )

        return {
            "gt_features": gt_features,
            "compressed_features": compressed_features,
            "score": score,
            "name": feat_name,
        }


def create_dataloaders(
    dataset_root: str, batch_size: int, num_workers: int = 4
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    Create dataloaders for image pairs

    Returns:
        Tuple of (train_loader, val_loader, test_loader)
    """
    loaders = {}

    for split in ["train", "val", "test"]:
        dataset = ImagePairDataset(dataset_root, split)

        loaders[split] = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=(split == "train"),
            num_workers=num_workers,
            pin_memory=True,
        )

    return loaders["train"], loaders["val"], loaders["test"]


def create_multi_compression_dataloaders(
    dataset_root: str,
    batch_size: int,
    quality_values: list = [10, 20, 30, 40, 50],
    num_workers: int = 4,
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    Create dataloaders for multi-compression image sets

    Args:
        dataset_root: Root directory containing the dataset
        batch_size: Batch size for the dataloaders
        quality_values: List of compression quality values
        num_workers: Number of workers for data loading

    Returns:
        Tuple of (train_loader, val_loader, test_loader)
    """
    loaders = {}

    for split in ["train"]:
        dataset = MultiCompressionDataset(
            dataset_root, split, quality_values=quality_values
        )

        loaders[split] = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=(split == "train"),
            num_workers=num_workers,
            pin_memory=True,
        )

    return loaders["train"]


def create_feature_dataloaders(
    features_root: str, batch_size: int, num_workers: int = 4
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    Create dataloaders for feature maps

    Returns:
        Tuple of (train_loader, val_loader, test_loader)
    """
    loaders = {}

    for split in ["train", "val", "test"]:
        dataset = FeatureMapDataset(features_root, split)

        loaders[split] = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=(split == "train"),
            num_workers=num_workers,
            pin_memory=True,
        )

    return loaders["train"], loaders["val"], loaders["test"]
