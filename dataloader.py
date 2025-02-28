import os
import cv2
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from typing import Tuple, Dict, Callable, Optional
from pathlib import Path
import json

from backbones import Backbone


class ImagePairDataset(Dataset):
    """
    Dataset for loading pairs of original and compressed images, with optional error scores and preprocessing.
    """

    def __init__(
        self,
        root_path: str,
        split: str,
        scores_root: str = None,
        preprocess: Optional[Callable] = None,
    ):
        """
        Initialize dataset.

        Args:
            root_path: Root directory containing the dataset
            split: Dataset split ('train', 'val', 'test')
            scores_root: Optional root directory containing error scores
            preprocess: Optional preprocessing transform to apply to images
        """
        if split not in ["train", "val", "test"]:
            raise ValueError("Split must be one of: train, val, test")

        # Setup paths
        self.split_path = Path(root_path) / split
        self.gt_path = self.split_path / "extracted"
        self.compressed_path = self.split_path / "compressed"

        # Setup scores if provided
        self.scores = None
        if scores_root:
            scores_path = Path(scores_root) / split / "error_scores.json"
            if not scores_path.exists():
                raise RuntimeError(f"Scores file not found: {scores_path}")
            with open(scores_path) as f:
                self.scores = json.load(f)

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
                and (not self.scores or f in self.scores)
            ]
        )

        if not self.image_names:
            raise RuntimeError(f"No valid image pairs found in {self.gt_path}")

        # Image transformations
        self.base_transform = transforms.ToTensor()  # Keep ToTensor as base
        self.preprocess = preprocess  # Store the provided preprocessing transform

    def __len__(self) -> int:
        """
        Returns the number of image pairs in the dataset.
        """
        return len(self.image_names)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        Get image pair and optional score.

        Returns:
            Dictionary containing:
            - gt: Ground truth image tensor [3, H, W] (preprocessed if preprocess is provided)
            - compressed: Compressed image tensor [3, H, W] (preprocessed if preprocess is provided)
            - name: Image filename
            - score: Error score [1] (if scores provided)
        """
        img_name = self.image_names[idx]

        # Load images
        gt_img = cv2.imread(str(self.gt_path / img_name))
        gt_img = cv2.cvtColor(gt_img, cv2.COLOR_BGR2RGB)
        gt_tensor = self.base_transform(gt_img)  # Use base_transform first

        compressed_img = cv2.imread(str(self.compressed_path / img_name))
        compressed_img = cv2.cvtColor(compressed_img, cv2.COLOR_BGR2RGB)
        compressed_tensor = self.base_transform(
            compressed_img
        )  # Use base_transform first

        # Apply preprocessing if provided
        if self.preprocess:
            gt_tensor = self.preprocess(gt_tensor)
            compressed_tensor = self.preprocess(compressed_tensor)

        result = {"gt": gt_tensor, "compressed": compressed_tensor, "name": img_name}

        # Add score if available
        if self.scores is not None:
            result["score"] = torch.tensor(self.scores[img_name], dtype=torch.float32)

        return result


class MultiCompressionDataset(Dataset):
    """
    Dataset for loading original images and their multiple compressed versions.
    """

    def __init__(
        self, root_path: str, split: str, quality_values: list = [10, 20, 30, 40, 50]
    ):
        """
        Initialize dataset.

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
        Get image set (original + compressed versions).

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


def create_dataloaders(
    dataset_root: str,
    batch_size: int,
    backbone_name: Backbone,
    error_scores_root: str = None,
    num_workers: int = 4,
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    Create dataloaders for image pairs with optional error scores and backbone-specific preprocessing.
    """
    loaders = {}
    imagenet_mean = [0.485, 0.456, 0.406]
    imagenet_std = [0.229, 0.224, 0.225]

    # Define default preprocessing (Resize + CenterCrop + Normalize)
    default_preprocess_transform = transforms.Compose(
        [
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.Normalize(mean=imagenet_mean, std=imagenet_std),
        ]
    )
    preprocess_transform = None  # Initialize to None, will be set conditionally

    if backbone_name in [
        Backbone.VGG_16,
        Backbone.MOBILENET_V3_L,
        Backbone.EFFICIENTNET_V2_M,
    ]:
        preprocess_transform = default_preprocess_transform
    elif backbone_name == Backbone.YOLO_V11_M:
        preprocess_transform = None  # No extra preprocessing for YOLO in dataloader
    else:
        raise ValueError(f"Unknown backbone '{backbone_name.value}'")

    for split in ["train", "val", "test"]:
        dataset = ImagePairDataset(
            dataset_root, split, error_scores_root, preprocess=preprocess_transform
        )  # Pass preprocess transform to dataset

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
) -> Dict[str, DataLoader]:
    """
    Create dataloaders for multi-compression image sets.

    Args:
        dataset_root: Root directory containing the dataset
        batch_size: Batch size for the dataloaders
        quality_values: List of compression quality values
        num_workers: Number of workers for data loading

    Returns:
        Dictionary of dataloaders for each split
    """
    loaders = {}

    for split in ["train", "val", "test"]:  # Now processing all splits
        try:
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
            print(f"Created dataloader for {split} split with {len(dataset)} images")
        except Exception as e:
            print(f"Warning: Could not create dataloader for {split} split: {str(e)}")
            loaders[split] = None

    return loaders
