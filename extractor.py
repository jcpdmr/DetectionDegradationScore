from typing import Tuple, List, Dict
import torch
import torch.nn as nn
from ultralytics import YOLO


class YOLO11mExtractor(nn.Module):
    """
    Feature extractor for YOLO11m that gets outputs from specified layers.
    """

    def __init__(self, weights_path: str, layer_indices: List[int]):
        """
        Initialize the feature extractor.

        Args:
            weights_path: Path to YOLO11m weights file
            layer_indices: List of layer indices to extract features from
        """
        super().__init__()
        model = YOLO(weights_path, verbose=False)
        self.model = model.model
        self.layer_indices = layer_indices  # Store layer indices
        self.extracted_features = {}  # Dictionary to store extracted features, keyed by layer index
        self._register_hooks()

    def _register_hooks(self) -> None:
        """
        Register forward hooks on specified layers to capture their outputs.
        The hooks store the output in self.extracted_features dictionary.
        """

        def hook_fn(layer_index):  # Hook function needs to capture layer_index
            def actual_hook_fn(module, input, output):
                self.extracted_features[layer_index] = output

            return actual_hook_fn

        for layer_index in self.layer_indices:  # Iterate through specified indices
            layer = self.model.model[layer_index]  # Access layer directly by index
            layer.register_forward_hook(
                hook_fn(layer_index)
            )  # Register hook with layer index

    @torch.no_grad()
    def forward(self, x: torch.Tensor) -> Dict[int, torch.Tensor]:
        """
        Extract features from specified layers of the YOLO model.

        Args:
            x: Input tensor of shape [B, 3, H, W]

        Returns:
            Dictionary of extracted features, keyed by layer index.
        """
        self.extracted_features = {}  # Reset features at each forward pass
        self.model(x)
        return self.extracted_features  # Return the dictionary of features

    def extract_features(
        self, img_gt: torch.Tensor, img_mod: torch.Tensor
    ) -> Tuple[Dict[int, torch.Tensor], Dict[int, torch.Tensor]]:
        """
        Extract features from both GT and modified images from specified layers.

        Args:
            img_gt: GT image tensor [B, 3, 384, 384]
            img_mod: Modified image tensor [B, 3, 384, 384]

        Returns:
            Tuple of dictionaries (GT features, Modified features),
            each dictionary containing features keyed by layer index.
        """
        features_gt = self.forward(img_gt)
        features_mod = self.forward(img_mod)
        return features_gt, features_mod


def load_feature_extractor(
    weights_path: str, layer_indices: List[int]
) -> YOLO11mExtractor:
    """
    Create and initialize the YOLO11m feature extractor.

    Args:
        weights_path: Path to YOLO11m weights
        layer_indices: List of layer indices to extract features from

    Returns:
        Initialized feature extractor
    """
    return YOLO11mExtractor(weights_path, layer_indices)
