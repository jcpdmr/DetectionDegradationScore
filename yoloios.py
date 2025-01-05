from ultralytics import YOLO
import torch
import torch.nn as nn
from typing import Tuple, List
import torch.nn.functional as F
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


class LearnablePooling(nn.Module):
    """
    Combines max and average pooling with a learnable weight.
    The weight is passed through sigmoid to ensure it stays between 0 and 1,
    effectively controlling the contribution of each pooling operation.
    """

    def __init__(self):
        super().__init__()
        # Initialize weight at 0.5 to give equal importance to both poolings initially
        self.weight = nn.Parameter(torch.tensor([0.5]))

    def forward(self, x):
        # Compute both pooling operations
        avg_pool = x.mean([2, 3], keepdim=True)
        max_pool = torch.amax(x, dim=[2, 3], keepdim=True)

        # Get weight between 0 and 1 using sigmoid
        w = torch.sigmoid(self.weight)

        # Weighted combination of the two pooling results
        return w * max_pool + (1 - w) * avg_pool


def extract_multiple_features_and_predictions(
    model: YOLO,
    images: torch.Tensor,
    layer_configs: List[LayerConfig],
    batch_size: int = 32,
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
                    predictions = model.predict(batch, verbose=False)
                    all_predictions.extend(predictions)

                    # Initialize storage for features on first batch
                    if all_features is None:
                        all_features = OrderedDict(
                            {
                                name: torch.empty(
                                    (num_images,) + feature.shape[1:],
                                    dtype=feature.dtype,
                                    device=feature.device,
                                )
                                for name, feature in extractor.features.items()
                            }
                        )

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


class SpatialProcessingBlock(nn.Module):
    """
    Processes spatial information in the difference map through convolutions.
    Uses batch normalization and ReLU activations to improve training stability
    and introduce non-linearity.
    """

    def __init__(self, in_channels):
        super().__init__()
        # First conv-bn-relu block
        self.conv1 = nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(in_channels)

        # Second conv-bn-relu block
        self.conv2 = nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(in_channels)

        # Final 1x1 convolution to reduce channels to 1
        self.final_conv = nn.Conv2d(in_channels, 1, kernel_size=1)

    def forward(self, x):
        # First spatial processing
        x = F.relu(self.bn1(self.conv1(x)))

        # Second spatial processing
        x = F.relu(self.bn2(self.conv2(x)))

        # Channel reduction
        x = self.final_conv(x)
        return x


class YOLOSimilarity(nn.Module):
    """
    Computes perceptual distance between feature maps extracted from two images.
    Uses spatial convolutions to analyze difference patterns and learnable pooling
    to aggregate spatial information.
    """

    def __init__(self):
        super().__init__()

        # Spatial processing blocks for each YOLO layer
        self.process_02 = SpatialProcessingBlock(256)  # Early features
        self.process_09 = SpatialProcessingBlock(512)  # SPPF features
        self.process_16 = SpatialProcessingBlock(256)  # Pre-detect features

        # Learnable pooling modules for each layer
        self.pool_02 = LearnablePooling()
        self.pool_09 = LearnablePooling()
        self.pool_16 = LearnablePooling()

        # Weights for combining layer contributions
        self.layer_weights = nn.Parameter(
            torch.tensor([1 / 3, 1 / 3, 1 / 3], dtype=torch.float32)
            + torch.randn(3) * 0.1  # Small random noise
        )
        # self.layer_weights = nn.Parameter(torch.zeros(3))
        # nn.init.uniform_(self.layer_weights, a=-0.3, b=0.3)
        # nn.init.trunc_normal_(self.layer_weights, mean=1.0, std=0.2, a=0.6, b=1.4)

    def compute_layer_distance(self, feat_a, feat_b, process_block, pool_block):
        """
        Computes distance between feature maps from one layer using:
        1. Squared difference
        2. Spatial processing
        3. Learnable pooling
        """
        # Compute squared difference
        diff_map = (feat_a - feat_b) ** 2

        # Process difference map to analyze spatial patterns
        processed_diff = process_block(diff_map)

        # Apply learnable pooling to get final layer distance
        return pool_block(processed_diff)

    def forward(self, features_a, features_b):
        """
        Compute total perceptual distance between two sets of feature maps
        using learned weights to combine individual layer distances.
        """
        # Compute distances for each layer
        d_02 = self.compute_layer_distance(
            features_a["02_C3k2_early"],
            features_b["02_C3k2_early"],
            self.process_02,
            self.pool_02,
        )

        d_09 = self.compute_layer_distance(
            features_a["09_SPPF"], features_b["09_SPPF"], self.process_09, self.pool_09
        )

        d_16 = self.compute_layer_distance(
            features_a["16_C3k2_pre_detect"],
            features_b["16_C3k2_pre_detect"],
            self.process_16,
            self.pool_16,
        )

        # Normalize layer weights using softmax
        # normalized_weights = F.softmax(self.layer_weights, dim=0)
        normalized_weights = self.layer_weights / self.layer_weights.sum()

        # Weighted combination of layer distances
        total_distance = (
            normalized_weights[0] * d_02
            + normalized_weights[1] * d_09
            + normalized_weights[2] * d_16
        )

        return total_distance.squeeze()
