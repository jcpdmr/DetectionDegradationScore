from ultralytics import YOLO
import torch
import torch.nn as nn
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
    Supports batch processing
    """
    def __init__(self, layer_configs: List[LayerConfig]):
        """
        Initialize feature extractor with configurations for target layers
        
        Args:
            layer_configs: List of LayerConfig objects specifying which layers to extract from
        """
        self.layer_configs = layer_configs
        self.features = OrderedDict()
        
    def hook_fn(self, layer_name: str):
        """
        Creates a hook function for a specific layer
        Handles batched inputs
        
        Args:
            layer_name: Name of the layer for identifying features
        
        Returns:
            Hook function that stores features in self.features
        """
        def hook(module, input, output):
            # Clone and detach to prevent memory leaks and ensure proper batch handling
            self.features[layer_name] = output.clone().detach()
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

def extract_multiple_features_and_predictions(
    model: YOLO,
    images: torch.Tensor,
    layer_configs: List[LayerConfig],
    batch_size: int = 32
) -> Tuple[OrderedDict, List]:
    """
    Extract both feature maps and predictions from YOLO model
    Supports batch processing with memory-efficient batch handling
    
    Args:
        model: YOLO model instance
        images: Input image tensors of shape [batch_size, channels, height, width]
        layer_configs: List of layer configurations
        batch_size: Maximum batch size to process at once to manage memory
    
    Returns:
        Tuple containing:
        - OrderedDict with batched feature maps from each layer
        - List of model predictions for each batch
    """
    num_images = images.size(0)
    extractor = MultiFeatureExtractor(layer_configs)
    hooks = extractor.register_hooks(model)
    all_predictions = []
    
    try:
        with torch.no_grad():
            # Handle large batches by processing in chunks
            if num_images > batch_size:
                all_features = None
                
                for i in range(0, num_images, batch_size):
                    end_idx = min(i + batch_size, num_images)
                    batch = images[i:end_idx]
                    
                    # Forward pass to get both features and predictions
                    predictions = model(batch)
                    all_predictions.extend(predictions)
                    
                    # Initialize storage for features on first batch
                    if all_features is None:
                        all_features = OrderedDict({
                            name: torch.empty(
                                (num_images,) + feature.shape[1:],
                                dtype=feature.dtype,
                                device=feature.device
                            )
                            for name, feature in extractor.features.items()
                        })
                    
                    # Store features for current batch
                    for name, feature in extractor.features.items():
                        all_features[name][i:end_idx] = feature
                
                return all_features, all_predictions
            else:
                # For small batches, process all at once
                predictions = model(images)
                return extractor.features, predictions
    finally:
        # Ensure hooks are always removed
        for hook in hooks:
            hook.remove()

class YOLOSimilarity(nn.Module):
    """
    Computes perceptual distance between feature maps extracted from two images using YOLO layers.
    Implements a custom loss similar to LPIPS (Learned Perceptual Image Patch Similarity).
    """
    def __init__(self):
        super(YOLOSimilarity, self).__init__()
        
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