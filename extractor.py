from typing import Tuple
import torch
import torch.nn as nn
from ultralytics import YOLO


class YOLO11mExtractor(nn.Module):
    """
    Feature extractor for YOLO11m that gets SPPF outputs.
    Updated to handle the current YOLO model structure.
    """

    def __init__(self, weights_path: str):
        """
        Initialize the feature extractor.

        Args:
            weights_path: Path to YOLO11m weights file
        """
        super().__init__()
        # Load YOLO model with given weights
        model = YOLO(weights_path, verbose=False)

        # Get the underlying PyTorch model
        self.model = model.model

        # Find SPPF layer by iterating through model structure
        self.sppf_layer = None
        self._find_sppf_layer(self.model)

        if self.sppf_layer is None:
            raise ValueError("SPPF layer not found in model")

        # Set model in evaluation mode
        self.model.eval()

        # Register forward hook to get SPPF output
        self.sppf_features = None
        self._register_hooks()

    def _find_sppf_layer(self, module: nn.Module) -> None:
        """
        Recursively find the SPPF layer in the model structure.
        Updates self.sppf_layer when found.

        Args:
            module: Current module to search through
        """
        # Check if the current module's class name contains 'SPPF'
        if "SPPF" in module.__class__.__name__:
            self.sppf_layer = module
            return

        # Recursively search through child modules
        for child in module.children():
            self._find_sppf_layer(child)

    def _register_hooks(self) -> None:
        """
        Register forward hook on SPPF layer to capture its output.
        The hook stores the output in self.sppf_features.
        """

        def hook_fn(module, input, output):
            self.sppf_features = output

        self.sppf_layer.register_forward_hook(hook_fn)

    @torch.no_grad()
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Extract SPPF features from input image.

        Args:
            x: Input tensor of shape [B, 3, H, W]
               Expected input size is [B, 3, 384, 384]

        Returns:
            SPPF features of shape [B, 512, H/32, W/32]
            For 384x384 input, output will be [B, 512, 12, 12]
        """
        # Reset stored features
        self.sppf_features = None

        # Forward pass through the model
        self.model(x)

        # Return captured SPPF features
        if self.sppf_features is None:
            raise RuntimeError("Failed to capture SPPF features")

        return self.sppf_features

    def extract_features(
        self, img_gt: torch.Tensor, img_mod: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Extract SPPF features from both GT and modified images.

        Args:
            img_gt: GT image tensor [B, 3, 384, 384]
            img_mod: Modified image tensor [B, 3, 384, 384]

        Returns:
            Tuple of (GT features, Modified features)
            Each with shape [B, 512, 12, 12]
        """
        features_gt = self.forward(img_gt)
        features_mod = self.forward(img_mod)
        return features_gt, features_mod


def load_feature_extractor(weights_path: str) -> YOLO11mExtractor:
    """
    Create and initialize the YOLO11m feature extractor.

    Args:
        weights_path: Path to YOLO11m weights

    Returns:
        Initialized feature extractor
    """
    return YOLO11mExtractor(weights_path)
