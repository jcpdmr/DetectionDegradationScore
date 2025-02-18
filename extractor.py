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
    """

    def __init__(self):
        super().__init__()
        self.model: Optional[nn.Module] = None  # To be set by subclasses
        self.layer_info: Optional[List[str] | List[int]] = (
            None  # Layer names or indices, set by subclasses
        )
        self.extracted_features: Dict[
            str | int, torch.Tensor
        ] = {}  # Keys can be str or int

    @abstractmethod
    def _load_model(self):
        """
        Abstract method to load the specific backbone model and set it to eval mode.
        Subclasses must implement this.
        """
        pass

    @abstractmethod
    def _get_layers_to_extract(self) -> List[str] | List[int]:
        """
        Abstract method to define which layers to extract features from.
        Subclasses must implement this. Return layer names or indices.
        """
        pass

    def _register_hooks(self) -> None:
        """
        Registers forward hooks on specified layers to capture their outputs.
        Handles both layer names (str) and layer indices (int).
        """

        def hook_fn(layer_key):  # layer_key can be name (str) or index (int)
            def actual_hook_fn(module, input, output):
                self.extracted_features[layer_key] = output

            return actual_hook_fn

        if isinstance(
            self.layer_info[0], str
        ):  # Check if layer_info is a list of names (strings)
            name_to_module = dict(
                self.model.named_children()
            )  # For named layers (like in VGG, ResNet, EfficientNet, MobileNet)
            for layer_name in self.layer_info:
                if layer_name not in name_to_module:
                    available_names = list(name_to_module.keys())
                    raise ValueError(
                        f"Layer name '{layer_name}' not found in model. Available layer names are: {available_names}"
                    )
                layer = name_to_module[layer_name]
                layer.register_forward_hook(hook_fn(layer_name))

        elif isinstance(
            self.layer_info[0], int
        ):  # Check if layer_info is a list of indices (integers)
            if hasattr(
                self.model, "model"
            ):  # For YOLO-like models where layers are in model.model
                model_layers = self.model.model
            else:  # For sequential models or direct models
                model_layers = self.model

            for layer_index in self.layer_info:
                try:
                    layer = model_layers[layer_index]  # Access layer by index
                except IndexError:
                    raise ValueError(
                        f"Layer index {layer_index} out of range for model with {len(model_layers)} layers."
                    )
                layer.register_forward_hook(hook_fn(layer_index))
        else:
            raise TypeError(
                "layer_info must be a list of layer names (str) or indices (int)."
            )

    @torch.no_grad()
    def forward(
        self, x: torch.Tensor
    ) -> Dict[str | int, torch.Tensor]:  # Return type allows str or int keys
        """
        Default forward method to extract features.
        """
        self.extracted_features = {}  # Reset features at each forward pass
        self.model(x)
        return self.extracted_features

    def extract_features(
        self, img_gt: torch.Tensor, img_mod: torch.Tensor
    ) -> Tuple[
        Dict[str | int, torch.Tensor], Dict[str | int, torch.Tensor]
    ]:  # Return type allows str or int keys
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

    def __init__(self, weights_path: str, layer_indices: List[int]):
        self.weights_path = weights_path
        self.layer_indices = layer_indices
        super().__init__()  # Initialize FeatureExtractor base class
        self._load_model()
        self.layer_info = (
            self._get_layers_to_extract()
        )  # Set layer_info for _register_hooks
        self._register_hooks()

    def _load_model(self):
        """Loads YOLOv11m model and sets it to eval mode."""
        model = YOLO(self.weights_path, verbose=False)
        self.model = model.model
        self.model.eval()  # Set YOLO model to eval mode

    def _get_layers_to_extract(self) -> List[int]:
        """Returns layer indices to extract from."""
        return self.layer_indices


class VGG16FeatureExtractor(FeatureExtractor):
    """
    Feature extractor for VGG16.
    """

    def __init__(self, layer_names: List[str]):
        self.layer_names = layer_names
        super().__init__()  # Initialize FeatureExtractor base class
        self._load_model()
        self.layer_info = (
            self._get_layers_to_extract()
        )  # Set layer_info for _register_hooks
        self._register_hooks()

    def _load_model(self):
        """Loads VGG16 model and sets it to eval mode."""
        vgg16 = models.vgg16(weights=models.VGG16_Weights.DEFAULT).features
        self.model = vgg16
        self.model.eval()  # Set VGG16 model to eval mode

    def _get_layers_to_extract(self) -> List[str]:
        """Returns layer names to extract from."""
        return self.layer_names


class MobileNetV3LargeFeatureExtractor(FeatureExtractor):
    """
    Feature extractor for MobileNetV3-Large.
    """

    def __init__(self, layer_names: List[str]):
        self.layer_names = layer_names
        super().__init__()  # Initialize FeatureExtractor base class
        self._load_model()
        self.layer_info = (
            self._get_layers_to_extract()
        )  # Set layer_info for _register_hooks
        self._register_hooks()

    def _load_model(self):
        """Loads MobileNetV3-Large model and sets it to eval mode."""
        mobilenet_v3_large = models.mobilenet_v3_large(
            weights=models.MobileNet_V3_Large_Weights.DEFAULT
        ).features
        self.model = mobilenet_v3_large
        self.model.eval()  # Set MobileNetV3-Large model to eval mode

    def _get_layers_to_extract(self) -> List[str]:
        """Returns layer names to extract from."""
        return self.layer_names


class EfficientNetV2MFeatureExtractor(FeatureExtractor):
    """
    Feature extractor for EfficientNetV2-Medium.
    """

    def __init__(self, layer_names: List[str]):
        self.layer_names = layer_names
        super().__init__()  # Initialize FeatureExtractor base class
        self._load_model()
        self.layer_info = (
            self._get_layers_to_extract()
        )  # Set layer_info for _register_hooks
        self._register_hooks()

    def _load_model(self):
        """Loads EfficientNetV2-Medium model and sets it to eval mode."""
        efficientnet_v2_m = models.efficientnet_v2_m(
            weights=models.EfficientNet_V2_M_Weights.DEFAULT
        ).features
        self.model = efficientnet_v2_m
        self.model.eval()  # Set EfficientNetV2-Medium model to eval mode

    def _get_layers_to_extract(self) -> List[str]:
        """Returns layer names to extract from."""
        return self.layer_names


def load_feature_extractor(
    backbone_name: Backbone,
    weights_path: str = None,
    layer_names: List[str] = None,
    layer_indices: List[int] = None,
) -> FeatureExtractor:
    """
    Factory function to load the feature extractor based on the backbone name.
    """

    if backbone_name == Backbone.YOLO_V11_M:
        return YOLO11mExtractor(weights_path=weights_path, layer_indices=layer_indices)
    elif backbone_name == Backbone.VGG_16:
        return VGG16FeatureExtractor(layer_names=layer_names)
    elif backbone_name == Backbone.MOBILENET_V3_L:
        return MobileNetV3LargeFeatureExtractor(layer_names=layer_names)
    elif backbone_name == Backbone.EFFICIENTNET_V2_M:
        return EfficientNetV2MFeatureExtractor(layer_names=layer_names)
    else:
        raise ValueError(
            f"Unsupported backbone: {backbone_name.value}. Supported backbones are: {[backbone.value for backbone in Backbone]}"
        )


def test_feature_extractors():
    """
    Test function to understand feature dimensions and layer names for each backbone.
    Creates dummy data and runs it through each feature extractor.
    """
    # Creiamo due set di dati dummy con dimensioni diverse
    batch_size = 16
    dummy_224 = torch.randn(
        batch_size, 3, 224, 224
    )  # Dimensione standard per la maggior parte dei modelli
    dummy_320 = torch.randn(batch_size, 3, 320, 320)  # Dimensione per YOLO

    # Configurazione per ogni backbone
    configs = {
        Backbone.YOLO_V11_M: {
            "layer_info": [9, 10],
            "input_data": dummy_320,
            "weights_path": "yolo11m.pt",  # Necessario per YOLO
        },
        Backbone.VGG_16: {
            "layer_info": ["16", "23"],
            "input_data": dummy_224,
        },
        Backbone.MOBILENET_V3_L: {
            "layer_info": ["5", "12"],
            "input_data": dummy_224,
        },
        Backbone.EFFICIENTNET_V2_M: {
            "layer_info": ["2", "5", "6"],
            "input_data": dummy_224,
        },
    }

    for backbone in Backbone:
        print(f"\n=== Testing {backbone.value} ===")

        config = configs[backbone]
        try:
            # Initialize extractor
            extractor = load_feature_extractor(
                backbone_name=backbone,
                weights_path=config.get("weights_path"),
                layer_names=config["layer_info"]
                if isinstance(config["layer_info"][0], str)
                else None,
                layer_indices=config["layer_info"]
                if isinstance(config["layer_info"][0], int)
                else None,
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
