from ultralytics import YOLO
from yoloios import process_image_multiple_layers, LayerConfig, YOLOPerceptualLoss
# from utils import analyze_features
import torch


def main():

    # Define layers to extract features from
    layer_configs = [
        LayerConfig(2, "02_C3k2_early"),      # Early features
        LayerConfig(9, "09_SPPF"),            # Multi-scale features
        LayerConfig(16, "16_C3k2_pre_detect") # Pre-detection features
    ]
    
    # Load models and move them to GPU
    yolo = YOLO('yolo11m.pt')
    yoloios = YOLOPerceptualLoss()
    if torch.cuda.is_available():
        yoloios = yoloios.cuda()
        yolo = yolo.cuda()
    # Process images and extract feature maps
    featureMapsGT = process_image_multiple_layers(
        yolo,
        "patches/extracted/VIRAT_S_040103_08_001475_001512_patch_0.jpg",
        layer_configs
    )
    featureMapsDistorted = process_image_multiple_layers(
        yolo,
        "patches/distorted/VIRAT_S_040103_08_001475_001512_patch_0.jpg",
        layer_configs
    )
    featureMapsCompressed = process_image_multiple_layers(
        yolo,
        "patches/compressed/VIRAT_S_040103_08_001475_001512_patch_0.jpg",
        layer_configs
    )

    # for name, tensor in featuresDistorted.items():
    #     # Check if tensor is on CUDA
    #     print(f"Layer {name} on CUDA: {tensor.is_cuda}")

    # output_dir = 'feature_maps_output'
    # if not os.path.exists(output_dir):
    #     os.makedirs(output_dir)
    # # Analyze and visualize features
    # for layer_name, feature_tensor in features.items():
    #     analyze_features(features=feature_tensor, name=layer_name, output_path=output_dir)
    
    with torch.no_grad():
        distanceDistGT = yoloios(featureMapsDistorted, featureMapsGT)
        distanceCompGT = yoloios(featureMapsCompressed, featureMapsGT)
    print(f"Distance Distorted - GT: {distanceDistGT:.6f}")
    print(f"Distance Compressed - GT: {distanceCompGT:.6f}")


if __name__ == "__main__":
    main()