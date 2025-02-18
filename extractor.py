from abc import ABC, abstractmethod
from typing import Dict, Tuple, List, Optional
import torch
import torch.nn as nn
from ultralytics import YOLO
import torchvision.models as models

from backbones import Backbone


class FeatureExtractor(ABC, nn.Module):
    """
    Abstract base class for feature extractors with common logic.
    Simplified to use only layer indices.
    """

    def __init__(self):
        super().__init__()
        self.model: Optional[nn.Module] = None  # To be set by subclasses
        self.layer_indices: List[int] = (
            None  # Layer indices, set by subclasses, now always List[int]
        )
        self.extracted_features: Dict[int, torch.Tensor] = {}  # Keys are now always int

    @abstractmethod
    def _load_model(self):
        """
        Abstract method to load the specific backbone model and set it to eval mode.
        Subclasses must implement this.
        """
        pass

    @abstractmethod
    def _get_layers_to_extract(self) -> List[int]:
        """
        Abstract method to define which layer indices to extract features from.
        Subclasses must implement this. Return layer indices.
        """
        pass

    def _register_hooks(self) -> None:
        """
        Registers forward hooks on specified layers (by index) to capture their outputs.
        Simplified to handle only layer indices.
        """

        def hook_fn(layer_index):  # layer_key is now always layer_index (int)
            def actual_hook_fn(module, input, output):
                self.extracted_features[layer_index] = output

            return actual_hook_fn

        if hasattr(
            self.model, "model"
        ):  # For YOLO-like models where layers are in model.model
            model_layers = self.model.model
        else:  # For sequential models or direct models
            model_layers = self.model

        for layer_index in self.layer_indices:
            try:
                layer = model_layers[layer_index]  # Access layer by index
            except IndexError:
                raise ValueError(
                    f"Layer index {layer_index} out of range for model with {len(model_layers)} layers."
                )
            layer.register_forward_hook(hook_fn(layer_index))

    @torch.no_grad()
    def forward(
        self, x: torch.Tensor
    ) -> Dict[int, torch.Tensor]:  # Return type is now always Dict[int, torch.Tensor]
        """
        Default forward method to extract features.
        """
        self.extracted_features = {}  # Reset features at each forward pass
        self.model(x)
        return self.extracted_features

    def extract_features(
        self, img_gt: torch.Tensor, img_mod: torch.Tensor
    ) -> Tuple[
        Dict[int, torch.Tensor], Dict[int, torch.Tensor]
    ]:  # Return type is now always Tuple[Dict[int, torch.Tensor], Dict[int, torch.Tensor]]
        """
        Default method to extract features for both GT and modified images.
        """
        features_gt = self.forward(img_gt)
        features_mod = self.forward(img_mod)
        return features_gt, features_mod


class YOLO11mExtractor(FeatureExtractor):
    """
    Feature extractor for YOLO11m.
    """

    def __init__(self, weights_path: str):  # layer_indices is still needed for YOLO
        self.weights_path = weights_path
        super().__init__()
        self._load_model()
        self.layer_indices = self._get_layers_to_extract()
        self._register_hooks()

    def _load_model(self):
        """Loads YOLOv11m model and sets it to eval mode."""
        model = YOLO(self.weights_path, verbose=False)
        self.model = model.model
        self.model.eval()  # Set YOLO model to eval mode

    def _get_layers_to_extract(self) -> List[int]:
        """Returns layer indices to extract from."""
        return Backbone.YOLO_V11_M.config.indices


class VGG16FeatureExtractor(FeatureExtractor):
    """
    Feature extractor for VGG16.
    """

    def __init__(self):
        super().__init__()
        self._load_model()
        self.layer_indices = self._get_layers_to_extract()
        self._register_hooks()

    def _load_model(self):
        """Loads VGG16 model and sets it to eval mode."""
        vgg16 = models.vgg16(weights=models.VGG16_Weights.DEFAULT).features
        self.model = vgg16
        self.model.eval()  # Set VGG16 model to eval mode

    def _get_layers_to_extract(self) -> List[int]:
        """Returns layer indices to extract from."""
        return Backbone.VGG_16.config.indices


class MobileNetV3LargeFeatureExtractor(FeatureExtractor):
    """
    Feature extractor for MobileNetV3-Large.
    """

    def __init__(self):
        super().__init__()
        self._load_model()
        self.layer_indices = self._get_layers_to_extract()
        self._register_hooks()

    def _load_model(self):
        """Loads MobileNetV3-Large model and sets it to eval mode."""
        mobilenet_v3_large = models.mobilenet_v3_large(
            weights=models.MobileNet_V3_Large_Weights.DEFAULT
        ).features
        self.model = mobilenet_v3_large
        self.model.eval()  # Set MobileNetV3-Large model to eval mode

    def _get_layers_to_extract(self) -> List[int]:
        """Returns layer indices to extract from."""
        return Backbone.MOBILENET_V3_L.config.indices


class EfficientNetV2MFeatureExtractor(FeatureExtractor):
    """
    Feature extractor for EfficientNetV2-Medium.
    """

    def __init__(self):
        super().__init__()
        self._load_model()
        self.layer_indices = self._get_layers_to_extract()
        self._register_hooks()

    def _load_model(self):
        """Loads EfficientNetV2-Medium model and sets it to eval mode."""
        efficientnet_v2_m = models.efficientnet_v2_m(
            weights=models.EfficientNet_V2_M_Weights.DEFAULT
        ).features
        self.model = efficientnet_v2_m
        self.model.eval()  # Set EfficientNetV2-Medium model to eval mode

    def _get_layers_to_extract(self) -> List[int]:
        """Returns layer indices to extract from."""
        return Backbone.EFFICIENTNET_V2_M.config.indices


def load_feature_extractor(
    backbone_name: Backbone,
    weights_path: str = None,
    # Removed layer_names and layer_indices parameters
) -> FeatureExtractor:
    """
    Factory function to load the feature extractor based on the backbone name.
    Simplified to use Backbone Enum for configuration.

    Args:
        weights_path: used only for Yolo backbone
    """

    if backbone_name == Backbone.YOLO_V11_M:
        return YOLO11mExtractor(weights_path=weights_path)
    elif backbone_name == Backbone.VGG_16:
        return VGG16FeatureExtractor()
    elif backbone_name == Backbone.MOBILENET_V3_L:
        return MobileNetV3LargeFeatureExtractor()
    elif backbone_name == Backbone.EFFICIENTNET_V2_M:
        return EfficientNetV2MFeatureExtractor()
    else:
        raise ValueError(
            f"Unsupported backbone: {backbone_name.value}. Supported backbones are: {[backbone.value for backbone in Backbone]}"
        )


def test_feature_extractors():
    """
    Test function to understand feature dimensions and layer names for each backbone.
    Simplified to use Backbone Enum configurations.
    """
    # Creiamo due set di dati dummy con dimensioni diverse
    batch_size = 16
    dummy_224 = torch.randn(
        batch_size, 3, 224, 224
    )  # Dimensione standard per la maggior parte dei modelli
    dummy_320 = torch.randn(batch_size, 3, 320, 320)  # Dimensione per YOLO

    # Configurazione per ogni backbone, simplified
    configs = {
        Backbone.YOLO_V11_M: {
            "input_data": dummy_320,
            "weights_path": "yolo11m.pt",  # Necessario per YOLO
        },
        Backbone.VGG_16: {
            "input_data": dummy_224,
        },
        Backbone.MOBILENET_V3_L: {
            "input_data": dummy_224,
        },
        Backbone.EFFICIENTNET_V2_M: {
            "input_data": dummy_224,
        },
    }

    for backbone in Backbone:
        print(f"\n=== Testing {backbone.value} Extractor ===")

        config = configs[backbone]
        try:
            # Initialize extractor, simplified call
            extractor = load_feature_extractor(
                backbone_name=backbone,
                weights_path=config.get("weights_path"),
            )

            # Extract features
            features = extractor.forward(config["input_data"])

            # Print info for each layer
            print("\nLayer information:")
            for layer_key, feature_map in features.items():
                print(f"\nLayer: {layer_key}")
                print(f"Feature map shape: {feature_map.shape}")
                print(f"Channels: {feature_map.shape[1]}")
                print(
                    f"Spatial dimensions: {feature_map.shape[2]}x{feature_map.shape[3]}"
                )

        except Exception as e:
            print(f"Error testing {backbone.value}: {str(e)}")


if __name__ == "__main__":
    test_feature_extractors()
