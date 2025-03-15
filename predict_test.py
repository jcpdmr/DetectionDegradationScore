import torch
import json
from pathlib import Path
from tqdm import tqdm
from dataloader import create_dataloaders
from quality_estimator import create_multifeature_baseline_quality_model
from extractor import load_feature_extractor, FeatureExtractor
import torch.nn.functional as F
from scipy.stats import spearmanr
import numpy as np
from scipy.stats import pearsonr
from backbones import Backbone


def predict_test_set(
    model_path: str,
    imgs_root: str,
    error_scores_root: str,
    output_path: str,
    backbone_name: Backbone,
    batch_size: int = 64,
    device: str = "cuda:0",
    weights_path: str = "yolo11m.pt",
):
    """
    Make predictions on test set, calculate metrics, and save results to JSON.

    Args:
        model_path: Path to saved model checkpoint
        imgs_root: Root directory containing images
        error_scores_root: Root directory containing error scores
        output_path: Path where to save JSON results
        backbone_name: the Backbone to use
        batch_size: Batch size for inference
        device: Device to run inference on
        weights_path: Path to YOLO11m weights file, needed only for YOLO backbone
    """
    # Set device
    device = torch.device(device if torch.cuda.is_available() else "cpu")

    layer_indices = backbone_name.config.indices
    feature_channels = backbone_name.config.channels

    # Load model
    model = create_multifeature_baseline_quality_model(
        feature_channels=feature_channels,
        layer_indices=layer_indices,
    ).to(device)

    checkpoint = torch.load(model_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    # Initialize feature extractor
    extractor: FeatureExtractor = load_feature_extractor(
        backbone_name=backbone_name,
        weights_path=weights_path,
    ).to(device)

    # Create test dataloader
    _, _, test_loader = create_dataloaders(
        backbone_name=backbone_name,
        dataset_root=imgs_root,
        error_scores_root=error_scores_root,
        batch_size=batch_size,
    )

    # Store results
    results = []
    all_predictions = []
    all_scores = []

    # Make predictions
    with torch.no_grad():
        for batch in tqdm(test_loader, desc="Making predictions"):
            gt = batch["gt"].to(device)
            compressed = batch["compressed"].to(device)
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
                        "error_score": float(score.cpu()),  # Ground truth error score
                    }
                )
                all_predictions.append(pred.cpu().numpy())
                all_scores.append(score.cpu().numpy())

    # Convert lists to numpy arrays
    all_predictions = np.array(all_predictions)
    all_scores = np.array(all_scores)

    # Calculate MAE
    mae = F.l1_loss(torch.tensor(all_predictions), torch.tensor(all_scores)).item()

    # Calculate Spearman's rank correlation
    spearman_corr, spearman_p = spearmanr(all_predictions, all_scores)

    # Calculate Pearson's correlation
    pearson_corr, pearson_p = pearsonr(all_predictions, all_scores)

    # Prepare statistics dictionary
    statistics = {
        "number_of_predictions": len(results),
        "average_predicted_distance": np.mean(all_predictions).item(),
        "MAE": mae,
        "Spearman_correlation": spearman_corr.item(),
        "Spearman_p_value": spearman_p.item(),
        "Pearson_correlation": pearson_corr.item(),
        "Pearson_p_value": pearson_p.item(),
    }

    # Save results to JSON
    output_data = {
        "statistics": statistics,
        "predictions": results,
    }

    with open(output_path, "w") as f:
        json.dump(output_data, f, indent=4)

    print(f"Results saved to {output_path}")

    # Print statistics to console
    print("\nStatistics:")
    for key, value in statistics.items():
        print(f"{key}: {value:.4f}" if isinstance(value, float) else f"{key}: {value}")


def main():
    # Configuration
    GPU_ID = 0
    DEVICE = torch.device(f"cuda:{GPU_ID}" if torch.cuda.is_available() else "cpu")
    ATTEMPT = 37
    DIR = "40bins_point8_07_coco17complete_320p_qual_20_25_30_35_40_45_50_subsamp_444"
    TRIAL = f"attempt{ATTEMPT}_{DIR}"
    MODEL_PATH = f"checkpoints/{TRIAL}/best_model.pt"
    IMGS_ROOT = "balanced_dataset_coco2017"
    ERROR_SCORES_ROOT = "balanced_dataset_coco2017"
    OUTPUT_PATH = f"checkpoints/{TRIAL}/test_predictions.json"
    BACKBONE = Backbone.EFFICIENTNET_V2_M

    predict_test_set(
        model_path=MODEL_PATH,
        imgs_root=IMGS_ROOT,
        error_scores_root=ERROR_SCORES_ROOT,
        output_path=OUTPUT_PATH,
        batch_size=128,
        weights_path="yolo11m.pt",
        backbone_name=BACKBONE,
        device=DEVICE,
    )


if __name__ == "__main__":
    main()
