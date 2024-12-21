from ultralytics import YOLO
from yoloios import process_image_multiple_layers, LayerConfig
from utils import analyze_features
import os

    

def main():
    output_dir = 'output'
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)


    # Define layers to extract features from
    layer_configs = [
        LayerConfig(2, "02_C3k2_early"),      # Early features
        LayerConfig(9, "09_SPPF"),            # Multi-scale features
        LayerConfig(16, "16_C3k2_pre_detect") # Pre-detection features
    ]
    
    # Load model and process image
    model = YOLO('yolo11m.pt')
    features = process_image_multiple_layers(
        model,
        "example.jpeg",
        layer_configs
    )

    # Analyze and visualize features
    for layer_name, feature_tensor in features.items():
        analyze_features(features=feature_tensor, name=layer_name, output_path=output_dir)

if __name__ == "__main__":
    main()