from ultralytics import YOLO
from train import test_perceptual_loss
from yoloios import LayerConfig
import os


def main():
    """
    Test a trained YOLOSimilarity model with results saved in a timestamp-specific directory.
    The timestamp is hardcoded for easy reference to specific training runs.
    """
    # Hardcoded timestamp from the training run we want to test
    model_timestamp = "20250103-213742"

    # Derive paths from timestamp
    base_output_dir = "output"
    model_dir = os.path.join(base_output_dir, model_timestamp)
    model_checkpoint = os.path.join(model_dir, "best_model.pth")

    # Configuration
    dataset_root = "dataset"
    batch_size = 128

    # Initialize YOLO model
    yolo = YOLO("yolo11m.pt", verbose=False)

    # Layer configurations
    layer_configs = [
        LayerConfig(2, "02_C3k2_early"),
        LayerConfig(9, "09_SPPF"),
        LayerConfig(16, "16_C3k2_pre_detect"),
    ]

    print(f"Testing model from run: {model_timestamp}")
    print("Results will be saved with '_manual' suffix")

    test_perceptual_loss(
        yolo_model=yolo,
        model_path=model_checkpoint,
        data_path=dataset_root,
        batch_size=batch_size,
        output_dir=model_dir,
        layer_configs=layer_configs,
        manual_test=True,
    )


if __name__ == "__main__":
    main()
