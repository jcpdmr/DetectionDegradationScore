from ultralytics import YOLO
import torch
import torch.nn as nn
from torchvision import transforms
import cv2
from typing import Tuple, List
from dataclasses import dataclass
from collections import OrderedDict


@dataclass
class LayerConfig:
    """
    Configuration for a layer to extract features from
    
    Attributes:
        index: Index of the layer in YOLO model
        name: Human readable name for the layer
    """
    index: int
    name: str

class MultiFeatureExtractor:
    """
    Handles extraction of feature maps from multiple YOLO model layers
    """
    def __init__(self, layer_configs: List[LayerConfig]):
        """
        Initialize feature extractor with configurations for target layers
        
        Args:
            layer_configs: List of LayerConfig objects specifying which layers to extract from
        """
        self.layer_configs = layer_configs
        self.features = OrderedDict()  # Using OrderedDict to maintain layer order
        
    def hook_fn(self, layer_name: str):
        """
        Creates a hook function for a specific layer
        
        Args:
            layer_name: Name of the layer for identifying features
        
        Returns:
            Hook function that stores features in self.features
        """
        def hook(module, input, output):
            self.features[layer_name] = output.detach()
        return hook

    def register_hooks(self, model: YOLO) -> List[torch.utils.hooks.RemovableHandle]:
        """
        Register hooks on all specified layers
        
        Args:
            model: YOLO model instance
        
        Returns:
            List of hook handles that can be used to remove hooks later
        """
        hooks = []
        for config in self.layer_configs:
            layer = model.model.model[config.index]
            hook = layer.register_forward_hook(self.hook_fn(config.name))
            hooks.append(hook)
        return hooks

def extract_multiple_features(
    model: YOLO,
    image: torch.Tensor,
    layer_configs: List[LayerConfig]
) -> OrderedDict:
    """
    Extract features from multiple specified layers
    
    Args:
        model: YOLO model instance
        image: Input image tensor
        layer_configs: List of layer configurations
    
    Returns:
        OrderedDict containing feature maps from each specified layer
    """
    extractor = MultiFeatureExtractor(layer_configs)
    hooks = extractor.register_hooks(model)
    
    try:
        with torch.no_grad():
            model(image)
    finally:
        # Always remove hooks to prevent memory leaks
        for hook in hooks:
            hook.remove()
    
    return extractor.features

def load_image_for_yolo(image_path: str, target_size: Tuple[int, int] = (640, 640)) -> torch.Tensor:
    """
    Load and preprocess image for YOLO model
    Args:
        image_path: Path to the input image
        target_size: Target size for the image (width, height)
    Returns:
        Preprocessed image tensor
    """
    image = cv2.imread(image_path, cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError(f"Could not read image: {image_path}")
    
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    height, width = image.shape[:2]
    
    if width < target_size[0] or height < target_size[1]:
        raise ValueError(f"Image too small. Required size: {target_size}, "
                        f"Current size: {width}x{height}")
    
    # Calculate center crop coordinates
    start_x = (width - target_size[0]) // 2
    start_y = (height - target_size[1]) // 2
    
    # Extract center crop
    center_crop = image[start_y:start_y + target_size[1], 
                       start_x:start_x + target_size[0]]
    
    # Convert to tensor
    transform = transforms.Compose([transforms.ToTensor()])
    image_tensor = transform(center_crop)

    # Add batch dimension
    image_tensor = image_tensor.unsqueeze(0)
    
    # Move to GPU if available
    if torch.cuda.is_available():
        image_tensor = image_tensor.cuda()
    
    return image_tensor

def process_image_multiple_layers(
    model: YOLO,
    image_path: str,
    layer_configs: List[LayerConfig]
) -> OrderedDict:
    """
    Process an image and extract features from multiple layers
    
    Args:
        model: YOLO model instance
        image_path: Path to input image
        layer_configs: List of layer configurations
    
    Returns:
        OrderedDict containing feature maps from each layer
    """
    # Load and preprocess image
    image_tensor = load_image_for_yolo(image_path)
    
    # Extract features
    features = extract_multiple_features(model, image_tensor, layer_configs)
    
    # Print shapes for verification
    print(f"Input image shape: {image_tensor.shape}")
    for name, feature in features.items():
        print(f"Feature maps shape for {name}: {feature.shape}")
    
    return features

class YOLOPerceptualLoss(nn.Module):
    """
    Computes perceptual distance between feature maps extracted from two images using YOLO layers.
    Implements a custom loss similar to LPIPS (Learned Perceptual Image Patch Similarity).
    """
    def __init__(self):
        super(YOLOPerceptualLoss, self).__init__()
        
        # Initialize 1x1 convolutional layers for channel weighting
        # These reduce the channel dimension to 1 while learning weights
        # for the importance of each input channel
        self.lin_02 = nn.Conv2d(256, 1, 1, stride=1, padding=0, bias=False)  # Early features (layer 02)
        self.lin_09 = nn.Conv2d(512, 1, 1, stride=1, padding=0, bias=False)  # SPPF features (layer 09)
        self.lin_16 = nn.Conv2d(256, 1, 1, stride=1, padding=0, bias=False)  # Pre-detection features (layer 16)
        
        # Initialize learnable weights for each layer's contribution to final distance
        # These weights help balance the importance of different feature levels
        self.layer_weights = nn.Parameter(torch.ones(3))
        
        # Initialize all convolutional weights to 1.0
        # This ensures equal initial contribution from all channels
        self.lin_02.weight.data.fill_(1.0)
        self.lin_09.weight.data.fill_(1.0)
        self.lin_16.weight.data.fill_(1.0)

    def normalize_tensor(self, x: torch.Tensor, eps: float = 1e-10) -> torch.Tensor:
        """
        Applies L2 normalization along channel dimension.
        
        Args:
            x: Input tensor of shape [batch, channels, height, width]
            eps: Small constant to prevent division by zero
            
        Returns:
            Normalized tensor of same shape as input
        """
        norm_factor = torch.sqrt(torch.sum(x**2, dim=1, keepdim=True))
        return x / (norm_factor + eps)

    def spatial_average(self, x: torch.Tensor) -> torch.Tensor:
        """
        Computes mean across spatial dimensions (height and width).
        
        Args:
            x: Input tensor of shape [batch, channels, height, width]
            
        Returns:
            Tensor of shape [batch, channels, 1, 1]
        """
        return x.mean([2, 3], keepdim=True)

    def compute_layer_distance(self, feat_a: torch.Tensor, feat_b: torch.Tensor, 
                             linear_layer: nn.Module) -> torch.Tensor:
        """
        Computes weighted distance between feature maps from one layer.
        
        Args:
            feat_a: Feature maps from first image
            feat_b: Feature maps from second image
            linear_layer: Convolutional layer for channel weighting
            
        Returns:
            Distance tensor of shape [batch, 1, 1, 1]
        """
        # Normalize features
        norm_a = self.normalize_tensor(feat_a)
        norm_b = self.normalize_tensor(feat_b)
        
        # Compute squared difference
        diff = (norm_a - norm_b) ** 2
        
        # Apply channel weights and spatial averaging
        return self.spatial_average(linear_layer(diff))

    def forward(self, features_a: dict, features_b: dict) -> torch.Tensor:
        """
        Compute total perceptual distance between two sets of feature maps.
        
        Args:
            features_a: Dictionary of feature maps from first image
            features_b: Dictionary of feature maps from second image
            
        Returns:
            Scalar distance value for each image in batch
        """
        # Compute distances for each layer
        d_02 = self.compute_layer_distance(features_a['02_C3k2_early'], 
                                         features_b['02_C3k2_early'], 
                                         self.lin_02)
        
        d_09 = self.compute_layer_distance(features_a['09_SPPF'], 
                                         features_b['09_SPPF'], 
                                         self.lin_09)
        
        d_16 = self.compute_layer_distance(features_a['16_C3k2_pre_detect'], 
                                         features_b['16_C3k2_pre_detect'], 
                                         self.lin_16)
        
        # Combine distances using learned layer weights
        total_distance = (self.layer_weights[0] * d_02 + 
                        self.layer_weights[1] * d_09 + 
                        self.layer_weights[2] * d_16)
        
        # Remove singleton dimensions and return
        return total_distance.squeeze()