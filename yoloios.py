from ultralytics import YOLO
import torch
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
    
    # To save image (using PIL)
    from torchvision.transforms import ToPILImage
    pil_image = ToPILImage()(image_tensor)
    pil_image.save('example_center_crop_640x640.jpeg')

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