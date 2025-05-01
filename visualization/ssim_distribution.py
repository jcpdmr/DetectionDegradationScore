import os
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image
import torchvision.transforms as transforms
from tqdm import tqdm
from typing import Dict, List, Tuple
import json
import random
from pathlib import Path
from skimage.metrics import structural_similarity as ssim
import cv2


def calculate_ssim_score(img1_path: str, img2_path: str) -> float:
    """
    Calculate SSIM score between two images.

    Args:
        img1_path: Path to first image
        img2_path: Path to second image

    Returns:
        SSIM score (higher is better, range 0-1)
    """
    # Read images
    img1 = cv2.imread(img1_path)
    img2 = cv2.imread(img2_path)

    # Convert to grayscale
    gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

    # Calculate SSIM
    score, _ = ssim(gray1, gray2, full=True)
    return score


def calculate_ssim_for_dataset(
    base_dir: str, quality_factors: List[int], sample_size: int = None
) -> pd.DataFrame:
    """
    Calculate SSIM scores for image pairs across multiple quality factors.

    Args:
        base_dir: Base directory containing the dataset
        quality_factors: List of JPEG quality factors to process
        sample_size: Optional number of images to sample (for faster processing)

    Returns:
        DataFrame with columns: image, quality_factor, ssim_score
    """
    # Get list of GT images
    extracted_dir = os.path.join(base_dir, "extracted")
    gt_images = [
        f for f in os.listdir(extracted_dir) if f.endswith((".jpg", ".jpeg", ".png"))
    ]

    # Sample if requested
    if sample_size and sample_size < len(gt_images):
        print(f"Sampling {sample_size} images from {len(gt_images)} total")
        random.seed(42)  # For reproducibility
        gt_images = random.sample(gt_images, sample_size)

    # Prepare data collection
    results = []

    # Process each quality factor
    for qf in quality_factors:
        compressed_dir = os.path.join(base_dir, f"compressed{qf}")

        # Skip if directory doesn't exist
        if not os.path.exists(compressed_dir):
            print(f"Directory not found: {compressed_dir}")
            continue

        print(f"Processing quality factor {qf} with {len(gt_images)} images")

        # Process each image
        for img_name in tqdm(gt_images, desc=f"QF {qf}"):
            gt_path = os.path.join(extracted_dir, img_name)
            compressed_path = os.path.join(compressed_dir, img_name)

            # Skip if compressed image doesn't exist
            if not os.path.exists(compressed_path):
                continue

            try:
                # Calculate SSIM score
                ssim_score = calculate_ssim_score(gt_path, compressed_path)

                # Store result
                results.append(
                    {"image": img_name, "quality_factor": qf, "ssim_score": ssim_score}
                )
            except Exception as e:
                print(f"Error processing {img_name} with QF {qf}: {str(e)}")

    return pd.DataFrame(results)


def create_violin_plot(df: pd.DataFrame, output_path: str):
    """
    Create and save a violin plot of SSIM scores by quality factor.

    Args:
        df: DataFrame with columns: quality_factor, ssim_score
        output_path: Path to save the plot
    """
    plt.figure(figsize=(12, 8))

    # Create violin plot
    sns.violinplot(
        data=df,
        x="quality_factor",
        y="ssim_score",
        density_norm="area",  # Ensure equal areas
        bw_adjust=0.8,  # Adjust bandwidth (>1 smoother, <1 more detailed)
        inner="box",  # Show box plot inside violin
        width=1,  # Width of violins
    )

    # Set plot title and labels
    plt.title("SSIM Score Distribution by JPEG Quality Factor", fontsize=16)
    plt.xlabel("Quality Factor", fontsize=14)
    plt.ylabel("SSIM Score", fontsize=14)

    # Set y-axis limits for better visualization
    plt.ylim(0, 1.05)  # SSIM scores range from 0 to 1

    # Add grid for better readability
    plt.grid(True, alpha=0.3)

    # Save the plot
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    print(f"Violin plot saved to: {output_path}")

    # Show the plot
    plt.show()


def main():
    # Configuration
    BASE_DIR = "/andromeda/personal/jdamerini/unbalanced_dataset_coco2017/train"
    QUALITY_FACTORS = [20, 25, 30, 35, 40, 45, 50]
    OUTPUT_DIR = "ssim_analysis"
    SAMPLE_SIZE = None  # Set to None to process all images

    # Create output directory
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Calculate SSIM scores
    df = calculate_ssim_for_dataset(
        base_dir=BASE_DIR, quality_factors=QUALITY_FACTORS, sample_size=SAMPLE_SIZE
    )

    # Save raw data
    csv_path = os.path.join(OUTPUT_DIR, "ssim_scores.csv")
    df.to_csv(csv_path, index=False)
    print(f"Raw data saved to: {csv_path}")

    # Save basic statistics as JSON
    stats = (
        df.groupby("quality_factor")["ssim_score"]
        .agg(["count", "mean", "std", "min", "max", "median"])
        .reset_index()
    )

    stats_dict = {
        int(row["quality_factor"]): {
            "count": int(row["count"]),
            "mean": float(row["mean"]),
            "std": float(row["std"]),
            "min": float(row["min"]),
            "max": float(row["max"]),
            "median": float(row["median"]),
        }
        for _, row in stats.iterrows()
    }

    json_path = os.path.join(OUTPUT_DIR, "ssim_statistics.json")
    with open(json_path, "w") as f:
        json.dump(stats_dict, f, indent=4)
    print(f"Statistics saved to: {json_path}")

    # Create violin plot
    plot_path = os.path.join(OUTPUT_DIR, "ssim_violin_plot.png")
    create_violin_plot(df, plot_path)

    print("Analysis complete!")


if __name__ == "__main__":
    main()
