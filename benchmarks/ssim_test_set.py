import os
import json
import numpy as np
import torch
from PIL import Image
from tqdm import tqdm
from skimage.metrics import structural_similarity as ssim
from scipy.stats import spearmanr, pearsonr
from typing import Dict, List, Tuple


def load_error_scores(json_path: str) -> Dict[str, float]:
    """
    Load error scores from JSON file.
    """
    with open(json_path, "r") as f:
        return json.load(f)


def get_image_pairs(
    extracted_dir: str, compressed_dir: str, error_scores: Dict[str, float]
) -> List[Tuple[str, str, float]]:
    """
    Get pairs of GT and compressed images with corresponding error scores.
    """
    pairs = []

    for img_name in error_scores.keys():
        gt_path = os.path.join(extracted_dir, img_name)
        compressed_path = os.path.join(compressed_dir, img_name)

        # Check if both images exist
        if os.path.exists(gt_path) and os.path.exists(compressed_path):
            pairs.append((gt_path, compressed_path, error_scores[img_name]))

    return pairs


def calculate_ssim_scores(
    image_pairs: List[Tuple[str, str, float]],
) -> Tuple[List[float], List[float]]:
    """
    Calculate SSIM scores for image pairs.
    """
    ssim_scores = []
    dds_targets = []

    for gt_path, compressed_path, target_score in tqdm(
        image_pairs, desc="Calculating SSIM"
    ):
        # Load images
        gt_img = np.array(Image.open(gt_path).convert("RGB"))
        compressed_img = np.array(Image.open(compressed_path).convert("RGB"))

        # Calculate SSIM (multichannel=True for RGB images)
        ssim_value = ssim(gt_img, compressed_img, multichannel=True, channel_axis=2)

        # Convert SSIM to dissimilarity score (1-SSIM) to align with DDS
        # DDS: higher value = more degradation
        # SSIM: higher value = more similarity
        dissimilarity = 1.0 - ssim_value

        ssim_scores.append(dissimilarity)
        dds_targets.append(target_score)

    return ssim_scores, dds_targets


def calculate_statistics(predictions: List[float], targets: List[float]) -> Dict:
    """
    Calculate statistical measures between predictions and targets.
    """
    predictions_array = np.array(predictions)
    targets_array = np.array(targets)

    # Calculate MAE
    mae = np.mean(np.abs(predictions_array - targets_array))

    # Calculate Spearman correlation
    spearman_corr, spearman_p = spearmanr(predictions_array, targets_array)

    # Calculate Pearson correlation
    pearson_corr, pearson_p = pearsonr(predictions_array, targets_array)

    return {
        "number_of_predictions": len(predictions),
        "average_predicted_distance": np.mean(predictions_array),
        "MAE": mae,
        "Spearman_correlation": spearman_corr,
        "Spearman_p_value": spearman_p,
        "Pearson_correlation": pearson_corr,
        "Pearson_p_value": pearson_p,
    }


def main():
    # Configuration
    base_dir = "balanced_dataset_coco2017/test"
    extracted_dir = os.path.join(base_dir, "extracted")
    compressed_dir = os.path.join(base_dir, "compressed")
    error_scores_path = os.path.join(base_dir, "error_scores.json")
    output_path = "ssim_vs_dds_results.json"

    # Check for GPU, but SSIM calculation is done on CPU
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device available: {device} (Note: SSIM calculation uses CPU)")

    # Load error scores
    print(f"Loading error scores from {error_scores_path}")
    error_scores = load_error_scores(error_scores_path)
    print(f"Loaded {len(error_scores)} error scores")

    # Get image pairs
    print("Finding valid image pairs...")
    image_pairs = get_image_pairs(extracted_dir, compressed_dir, error_scores)
    print(f"Found {len(image_pairs)} valid image pairs")

    # Calculate SSIM scores (as dissimilarity: 1-SSIM)
    print("Calculating SSIM scores...")
    ssim_scores, dds_targets = calculate_ssim_scores(image_pairs)

    # Calculate statistics
    print("Calculating statistics...")
    statistics = calculate_statistics(ssim_scores, dds_targets)

    # Print statistics
    print("\nSSIM (as 1-SSIM) vs DDS Statistics:")
    print(f"Number of predictions: {statistics['number_of_predictions']}")
    print(f"Average 1-SSIM score: {statistics['average_predicted_distance']:.4f}")
    print(f"MAE: {statistics['MAE']:.4f}")
    print(
        f"Spearman correlation: {statistics['Spearman_correlation']:.4f} (p-value: {statistics['Spearman_p_value']:.4e})"
    )
    print(
        f"Pearson correlation: {statistics['Pearson_correlation']:.4f} (p-value: {statistics['Pearson_p_value']:.4e})"
    )

    # Save results
    with open(output_path, "w") as f:
        # Convert numpy values to native Python types for JSON serialization
        statistics_json = {
            k: float(v)
            if isinstance(v, np.floating)
            else int(v)
            if isinstance(v, np.integer)
            else v
            for k, v in statistics.items()
        }
        json.dump(statistics_json, f, indent=4)

    print(f"Results saved to {output_path}")


if __name__ == "__main__":
    main()
