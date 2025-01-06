from ultralytics import YOLO
from train import train_perceptual_loss, test_perceptual_loss


def main():
    # Configuration
    config = {
        "num_epochs": 50,
        "batch_size": 64,
        "learning_rate": 1e-5,
        "data_path": "dataset",
        "val_frequency": 5,
        "patience": 4,
        "output_dir": "output",
        "seed": 100,
        "modification_types": ["compressed"],
    }

    # Load YOLO model
    yolo = YOLO("yolo11m.pt", verbose=False)

    # Training phase
    print("Starting training phase (compressed images only)...")
    training_info = train_perceptual_loss(yolo, **config)

    # Testing phase
    print("\nStarting testing phase...")
    test_results = test_perceptual_loss(
        yolo_model=yolo,
        model_path=training_info["model_path"],
        data_path=config["data_path"],
        batch_size=config["batch_size"],
        output_dir=training_info["output_dir"],
        layer_configs=training_info["layer_configs"],
        modification_types=training_info["modification_types"],
    )

    # Print final test results
    print("\nTest Results:")
    print(f"Correlation with error scores: {test_results['correlation']:.4f}")
    print(f"MAE(similarites-error scores): {test_results['mae']:.4f}")


if __name__ == "__main__":
    main()
