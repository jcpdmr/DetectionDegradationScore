import os
import random
import cv2
import torch
from torchvision import transforms
from torch.utils.data import Dataset

class PatchesDataset(Dataset):
    """
    Dataset for loading ground truth, distorted and compressed images pairs
    """
    def __init__(self, base_path: str):
        self.gt_path = os.path.join(base_path, "extracted")
        self.distorted_path = os.path.join(base_path, "distorted")
        self.compressed_path = os.path.join(base_path, "compressed")
        
        # Get all image names from ground truth folder
        self.image_names = [f for f in os.listdir(self.gt_path) if f.endswith('.jpg')]
        
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
    
    # Add batch dimension and move to GPU if available
    image_tensor = image_tensor.unsqueeze(0)
    if torch.cuda.is_available():
        image_tensor = image_tensor.cuda()
    
    return image_tensor