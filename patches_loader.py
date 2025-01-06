import os
import cv2
import torch
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
from typing import Optional, Tuple, Dict
from pathlib import Path
import json


class EnhancedPatchesDataset(Dataset):
    """
    Dataset class that loads image pairs along with their precomputed error scores.
    Handles loading from the organized directory structure where each split has
    its own error_scores.json file.
    """

    def __init__(
        self,
        root_path: str,
        split: str,
        transform: Optional[transforms.Compose] = None,
    ):
        """
        Initialize the enhanced dataset with automatic score loading.
        Each split directory should contain an error_scores.json file.

        Args:
            root_path: Base directory containing split folders
            split: Dataset split ('train', 'val', or 'test')
            transform: Optional transforms to apply to the images
        """
        # Validate split name
        if split not in ["train", "val", "test"]:
            raise ValueError("Split must be one of: 'train', 'val', 'test'")

        # Setup paths
        self.split_path = Path(root_path) / split
        self.gt_path = self.split_path / "extracted"
        self.mod_path = self.split_path / "compressed"
        self.scores_path = self.split_path / "error_scores.json"
        self.transform = transform

        # Verify directory structure and load scores
        self._verify_directories()
        self._load_scores()

        # Get list of valid image files
        self.image_names = sorted(
            [
                f
                for f in os.listdir(self.gt_path)
                if f.lower().endswith((".jpg", ".jpeg", ".png"))
                and f in self.scores  # Only include images that have scores
            ]
        )

        if not self.image_names:
            raise RuntimeError(f"No valid images found in {self.gt_path}")

    def _verify_directories(self) -> None:
        """
        Verify that all required directories and files exist.
        Raises appropriate errors if anything is missing.
        """
        if not self.gt_path.exists():
            raise RuntimeError(f"Ground truth directory not found: {self.gt_path}")
        if not self.mod_path.exists():
            raise RuntimeError(f"Modified images directory not found: {self.mod_path}")
        if not self.scores_path.exists():
            raise RuntimeError(
                f"Error scores file not found: {self.scores_path}\n"
                "Please run the score calculation script first."
            )

    def _load_scores(self) -> None:
        """
        Load error scores from the JSON file.
        Stores them in a dictionary mapping image names to scores.
        """
        try:
            with open(self.scores_path, "r") as f:
                self.scores = json.load(f)
        except json.JSONDecodeError:
            raise RuntimeError(f"Invalid JSON file: {self.scores_path}")
        except Exception as e:
            raise RuntimeError(f"Error loading scores: {str(e)}")

    def __len__(self) -> int:
        """Return the number of valid image pairs with scores"""
        return len(self.image_names)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        Get a single image pair and its error score.

        Returns:
            Dictionary containing:
                - gt: Ground truth image tensor [3, H, W]
                - modified: Modified image tensor [3, H, W]
                - score: Error score tensor [1]
                - name: Image filename
        """
        img_name = self.image_names[idx]

        # Load images
        gt_img = load_image_for_yolo(self.gt_path / img_name)
        modified_img = load_image_for_yolo(self.mod_path / img_name)

        # Apply transforms if specified
        if self.transform:
            gt_img = self.transform(gt_img)
            modified_img = self.transform(modified_img)

        # Get score for this pair
        score = torch.tensor(self.scores[img_name], dtype=torch.float32)

        return {
            "gt": gt_img,
            "modified": modified_img,
            "score": score,
            "name": img_name,
        }


def create_dataloaders(
    root_path: str,
    batch_size: int,
    num_workers: int = 4,
    transform: Optional[transforms.Compose] = None,
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    Create dataloaders for train, validation, and test sets.
    Each split must have its corresponding error_scores.json file.

    Args:
        root_path: Root directory containing the split folders
        batch_size: Batch size for the dataloaders
        num_workers: Number of worker processes for loading
        transform: Optional transforms to apply to images

    Returns:
        Tuple of (train_loader, val_loader, test_loader)
    """
    loaders = {}

    for split in ["train", "val", "test"]:
        dataset = EnhancedPatchesDataset(
            root_path=root_path, split=split, transform=transform
        )

        loaders[split] = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=(split == "train"),  # Shuffle only training data
            num_workers=num_workers,
            pin_memory=True,
        )

    return loaders["train"], loaders["val"], loaders["test"]


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
