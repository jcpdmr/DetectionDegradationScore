import torch
import json
from pathlib import Path
from tqdm import tqdm
from dataloader import create_feature_dataloaders
from quality_estimator import create_quality_model


def predict_test_set(
    model_path: str,
    features_root: str,
    error_scores_root: str,
    output_path: str,
    batch_size: int = 64,
    device: str = "cuda:1",
):
    """
    Make predictions on test set and save results to JSON.

    Args:
        model_path: Path to saved model checkpoint
        features_root: Root directory containing feature maps
        output_path: Path where to save JSON results
        batch_size: Batch size for inference
        device: Device to run inference on
    """
    # Set device
    device = torch.device(device if torch.cuda.is_available() else "cpu")

    # Load model
    model = create_quality_model().to(device)
    checkpoint = torch.load(model_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    # Create test dataloader
    _, _, test_loader = create_feature_dataloaders(
        features_root=features_root,
        error_scores_root=error_scores_root,
        batch_size=batch_size,
    )

    # Store results
    results = []

    # Make predictions
    with torch.no_grad():
        for batch in tqdm(test_loader, desc="Making predictions"):
            # Get features and move to device
            gt_features = batch["gt_features"].to(device)
            compressed_features = batch["compressed_features"].to(device)
            names = batch["name"]
            scores = batch["score"]

            # Make predictions
            predictions = model(gt_features, compressed_features).squeeze()

            # Store results
            for name, pred, score in zip(names, predictions, scores):
                # Convert .npy extension to .jpg while keeping the same numeric name
                img_name = f"{Path(name).stem}.jpg"

                results.append(
                    {
                        "filename": img_name,
                        "distance": float(pred.cpu()),  # Predicted distance
                        "error_score": float(score),  # Ground truth error score
                    }
                )

    # Save results to JSON
    with open(output_path, "w") as f:
        json.dump(results, f, indent=4)

    print(f"Results saved to {output_path}")

    # Calculate and print some statistics
    predictions = [r["distance"] for r in results]
    scores = [r["error_score"] for r in results]
    correlation = torch.corrcoef(torch.tensor([predictions, scores]))[0, 1]
    print("\nStatistics:")
    print(f"Number of predictions: {len(results)}")
    print(f"Average predicted distance: {sum(predictions) / len(predictions):.4f}")
    print(f"Correlation with error scores: {correlation:.4f}")


def main():
    # Configuration
    MODEL_PATH = "checkpoints/best_model.pt"
    FEATURES_ROOT = "feature_extracted"
    ERROR_SCORES_ROOT = "balanced_dataset"
    OUTPUT_PATH = "test_predictions.json"

    predict_test_set(
        model_path=MODEL_PATH,
        features_root=FEATURES_ROOT,
        error_scores_root=ERROR_SCORES_ROOT,
        output_path=OUTPUT_PATH,
    )


if __name__ == "__main__":
    main()
