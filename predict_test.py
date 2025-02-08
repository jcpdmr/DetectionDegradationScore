import torch
import json
from pathlib import Path
from tqdm import tqdm
from dataloader import create_feature_dataloaders, create_dataloaders
from quality_estimator import create_quality_model
from extractor import load_feature_extractor, YOLO11mExtractor


def predict_test_set(
    model_path: str,
    features_root: str,
    error_scores_root: str,
    imgs_root: str,
    output_path: str,
    batch_size: int = 64,
    device: str = "cuda:0",
    yolo_weights_path: str = "yolo11m.pt",
):
    """
    Make predictions on test set and save results to JSON.

    Args:
        model_path: Path to saved model checkpoint
        features_root: Root directory containing feature maps
        imgs_root: Root directory containing images
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

    # Initialize feature extractor
    extractor: YOLO11mExtractor = load_feature_extractor(
        weights_path=yolo_weights_path
    ).to(device)

    # Create test dataloader
    _, _, test_loader = create_dataloaders(
        dataset_root=imgs_root,
        error_scores_root=error_scores_root,
        batch_size=batch_size,
    )

    # Store results
    results = []

    # Make predictions
    with torch.no_grad():
        for batch in tqdm(test_loader, desc="Making predictions"):
            gt = batch["gt"].to(device)
            compressed = batch["compressed"].to(device)
            # gt_features = batch["gt_features"].to(device)
            # mod_features = batch["compressed_features"].to(device)
            scores = batch["score"].to(device)
            names = batch["name"]
            gt_features, mod_features = extractor.extract_features(
                img_gt=gt, img_mod=compressed
            )

            # Make predictions
            predictions = model(gt_features, mod_features).squeeze()

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
    TRIAL = "attempt5_40bins_point8_06_visgen_coco17tr_openimagev7traine_320p_qual_20_24_28_32_36_40_50_smooth_2_subsam_444"
    MODEL_PATH = f"checkpoints/{TRIAL}/best_model.pt"
    FEATURES_ROOT = "feature_extracted"
    IMGS_ROOT = "balanced_dataset"
    ERROR_SCORES_ROOT = "balanced_dataset"
    OUTPUT_PATH = f"checkpoints/{TRIAL}/test_predictions.json"

    predict_test_set(
        model_path=MODEL_PATH,
        features_root=FEATURES_ROOT,
        imgs_root=IMGS_ROOT,
        error_scores_root=ERROR_SCORES_ROOT,
        output_path=OUTPUT_PATH,
        batch_size=128,
        yolo_weights_path="yolo11m.pt",
    )


if __name__ == "__main__":
    main()
