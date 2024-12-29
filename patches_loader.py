import os
import random
import cv2
import torch
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader

class PatchesDataset(Dataset):
    """
    Dataset for loading ground truth, distorted and compressed images pairs
    with train/validation split capability
    """
    def __init__(self, base_path: str, split: str = 'train', val_ratio: float = 0.2, seed: int = 42):
        """
        Initialize dataset with train/validation split
        
        Args:
            base_path: Base directory containing image folders
            split: Either 'train' or 'val'
            val_ratio: Proportion of data to use for validation
            seed: Random seed for reproducible splits
        """
        self.gt_path = os.path.join(base_path, "extracted")
        self.distorted_path = os.path.join(base_path, "distorted")
        self.compressed_path = os.path.join(base_path, "compressed")
        
        # Get all image names
        all_images = [f for f in os.listdir(self.gt_path) if f.endswith('.png')]
        
        # Create reproducible random split
        random.seed(seed)
        random.shuffle(all_images)
        
        # Calculate split index
        split_idx = int(len(all_images) * (1 - val_ratio))
        
        # Assign images based on split
        if split == 'train':
            self.image_names = all_images[:split_idx]
        elif split == 'val':
            self.image_names = all_images[split_idx:]
        else:
            raise ValueError("Split must be either 'train' or 'val'")
            
    def __len__(self):
        return len(self.image_names)
    
    def __getitem__(self, idx):
        img_name = self.image_names[idx]
        gt_img = load_image_for_yolo(os.path.join(self.gt_path, img_name))
        
        # Randomly choose between distorted or compressed
        if random.random() < 0.5:
            modified_img = load_image_for_yolo(os.path.join(self.distorted_path, img_name))
        else:
            modified_img = load_image_for_yolo(os.path.join(self.compressed_path, img_name))
            
        return {
            'gt': gt_img,
            'modified': modified_img,
            'name': img_name
        }

def create_dataloaders(base_path: str, batch_size: int, val_ratio: float = 0.2,
                      num_workers: int = 4, seed: int = 42):
    """
    Create train and validation dataloaders
    
    Args:
        base_path: Base directory containing image folders
        batch_size: Batch size for both loaders
        val_ratio: Proportion of data to use for validation
        num_workers: Number of workers for data loading
        seed: Random seed for reproducible splits
    
    Returns:
        train_loader, val_loader
    """
    # Create datasets
    train_dataset = PatchesDataset(base_path, split='train', val_ratio=val_ratio, seed=seed)
    val_dataset = PatchesDataset(base_path, split='val', val_ratio=val_ratio, seed=seed)
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,  # No need to shuffle validation data
        num_workers=num_workers,
        pin_memory=True
    )
    
    return train_loader, val_loader

def load_image_for_yolo(image_path: str) -> torch.Tensor:
    """
    Load and preprocess image for YOLO model.
    Assumes images are already 640x640.
    
    Args:
        image_path: Path to the input image
        
    Returns:
        Preprocessed image tensor on appropriate device (CPU/GPU)
    """
    # Read and convert image
    image = cv2.imread(image_path, cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError(f"Could not read image: {image_path}")
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # Convert to tensor
    image_tensor = transforms.ToTensor()(image)
    return image_tensor