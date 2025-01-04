import os
import random
import cv2
import torch
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
from typing import Optional, List, Tuple
from pathlib import Path


class PatchesDataset(Dataset):
    """
    Dataset class for loading image pairs from pre-split directories.
    Handles ground truth images and their modified versions (compressed/distorted).
    """

    def __init__(
        self,
        root_path: str,
        split: str,
        modification_types: Optional[List[str]] = None,
        transform: Optional[transforms.Compose] = None,
    ):
        """
        Initialize dataset from a specific split directory

        Args:
            root_path: Root directory containing all splits
            split: Dataset split to use ('train', 'val', or 'test')
            modification_types: List of modification types to include
            transform: Optional transforms to apply to images
        """
        # Validate split type
        if split not in ["train", "val", "test"]:
            raise ValueError("Split must be one of: 'train', 'val', 'test'")

        # Setup paths for this split
        self.split_path = Path(root_path) / split
        self.gt_path = self.split_path / "extracted"

        # Setup modification types (default to both if not specified)
        self.mod_types = modification_types or ["distorted", "compressed"]
        self.mod_paths = {
            mod_type: self.split_path / mod_type for mod_type in self.mod_types
        }

        # Store transform
        self.transform = transform

        # Verify directory structure
        self._verify_directories()

        # Get list of images (only need to check ground truth directory)
        self.image_names = sorted(
            [
                f
                for f in os.listdir(self.gt_path)
                if f.lower().endswith((".jpg", ".jpeg", ".png"))
            ]
        )

        if not self.image_names:
            raise RuntimeError(f"No images found in {self.gt_path}")

    def _verify_directories(self) -> None:
        """
        Verify that all required directories exist and contain matching files
        """
        if not self.gt_path.exists():
            raise RuntimeError(f"Ground truth directory not found: {self.gt_path}")

        for mod_type, path in self.mod_paths.items():
            if not path.exists():
                raise RuntimeError(f"{mod_type} directory not found: {path}")

    def __len__(self) -> int:
        """Return the total number of image pairs in this split"""
        return len(self.image_names)

    def __getitem__(self, idx: int) -> dict:
        """
        Get a single image pair (ground truth and modified)

        Args:
            idx: Index of the image pair to fetch

        Returns:
            Dictionary containing:
                - gt: Ground truth image tensor
                - modified: Modified image tensor
                - name: Image filename
                - mod_type: Type of modification applied
        """
        img_name = self.image_names[idx]

        # Load ground truth image
        gt_img = load_image_for_yolo(self.gt_path / img_name)

        # Randomly choose modification type (for training variety)
        mod_type = random.choice(self.mod_types)
        modified_img = load_image_for_yolo(self.mod_paths[mod_type] / img_name)

        # Apply any additional transforms if specified
        if self.transform:
            gt_img = self.transform(gt_img)
            modified_img = self.transform(modified_img)

        return {
            "gt": gt_img,
            "modified": modified_img,
            "name": img_name,
            "mod_type": mod_type,
        }


def create_dataloaders(
    root_path: str,
    batch_size: int,
    num_workers: int = 4,
    modification_types: Optional[List[str]] = None,
    transform: Optional[transforms.Compose] = None,
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    Create train, validation and test dataloaders

    Args:
        root_path: Root directory containing split folders
        batch_size: Batch size for the dataloaders
        num_workers: Number of worker processes for data loading
        modification_types: List of modification types to include
        transform: Optional transforms to apply to images

    Returns:
        Tuple of (train_loader, val_loader, test_loader)
    """
    # Create datasets for each split
    train_dataset = PatchesDataset(
        root_path, "train", modification_types=modification_types, transform=transform
    )

    val_dataset = PatchesDataset(
        root_path, "val", modification_types=modification_types, transform=transform
    )

    test_dataset = PatchesDataset(
        root_path, "test", modification_types=modification_types, transform=transform
    )

    # Create dataloaders with appropriate settings
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,  # Shuffle training data
        num_workers=num_workers,
        pin_memory=True,  # Faster data transfer to GPU
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,  # No need to shuffle validation
        num_workers=num_workers,
        pin_memory=True,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,  # No need to shuffle test
        num_workers=num_workers,
        pin_memory=True,
    )

    return train_loader, val_loader, test_loader


def load_image_for_yolo(image_path: str) -> torch.Tensor:
    """
    Load and preprocess image for YOLO model.
    Assumes images are already in a square format multiple of 32 pixels.

    Args:
        image_path: Path to the input image

    Returns:
        Preprocessed image tensor
    """
    image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError(f"Could not read image: {image_path}")

    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    return transforms.ToTensor()(image)
