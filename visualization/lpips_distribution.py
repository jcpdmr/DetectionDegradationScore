import os
import torch
import lpips
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


def preprocess_image(img_path: str, transform) -> torch.Tensor:
    """
    Load and preprocess an image.
    """
    img = Image.open(img_path).convert("RGB")
    img_tensor = transform(img)
    return img_tensor.unsqueeze(0)  # Add batch dimension


def calculate_lpips_for_dataset(
    base_dir: str,
    quality_factors: List[int],
    sample_size: int = None,
    device: torch.device = torch.device("cuda"),
) -> pd.DataFrame:
    """
    Calculate LPIPS scores for image pairs across multiple quality factors.

    Args:
        base_dir: Base directory containing the dataset
        quality_factors: List of JPEG quality factors to process
        sample_size: Optional number of images to sample (for faster processing)
        device: Device to run calculations on

    Returns:
        DataFrame with columns: image, quality_factor, lpips_score
    """
    # Initialize LPIPS model
    lpips_model = lpips.LPIPS(net="alex").to(device)

    # Set up image transform
    transform = transforms.Compose(
        [
            transforms.ToTensor(),
        ]
    )

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
                # Load and preprocess images
                gt_tensor = preprocess_image(gt_path, transform).to(device)
                compressed_tensor = preprocess_image(compressed_path, transform).to(
                    device
                )

                # Calculate LPIPS score
                with torch.no_grad():
                    lpips_score = lpips_model(gt_tensor, compressed_tensor).item()

                # Store result
                results.append(
                    {
                        "image": img_name,
                        "quality_factor": qf,
                        "lpips_score": lpips_score,
                    }
                )
            except Exception as e:
                print(f"Error processing {img_name} with QF {qf}: {str(e)}")

    return pd.DataFrame(results)


def create_violin_plot(df: pd.DataFrame, output_path: str):
    """
    Create and save a violin plot of LPIPS scores by quality factor.

    Args:
        df: DataFrame with columns: quality_factor, lpips_score
        output_path: Path to save the plot
    """
    plt.figure(figsize=(12, 8))

    # Create violin plot
    sns.violinplot(
        data=df,
        x="quality_factor",
        y="lpips_score",
        density_norm="area",  # Ensure equal areas
        bw_adjust=0.8,  # Adjust bandwidth (>1 smoother, <1 more detailed)
        inner="box",  # Show box plot inside violin
        width=1,  # Width of violins
    )

    # Set plot title and labels
    plt.title("LPIPS Score Distribution by JPEG Quality Factor", fontsize=16)
    plt.xlabel("Quality Factor", fontsize=14)
    plt.ylabel("LPIPS Score", fontsize=14)

    # Set y-axis limits for better visualization
    plt.ylim(0, max(df["lpips_score"]) * 1.05)  # Add 5% margin at top

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
    OUTPUT_DIR = "lpips_analysis"
    SAMPLE_SIZE = None  # Set to None to process all images

    # Create output directory
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Calculate LPIPS scores
    df = calculate_lpips_for_dataset(
        base_dir=BASE_DIR,
        quality_factors=QUALITY_FACTORS,
        sample_size=SAMPLE_SIZE,
        device=device,
    )

    # Save raw data
    csv_path = os.path.join(OUTPUT_DIR, "lpips_scores.csv")
    df.to_csv(csv_path, index=False)
    print(f"Raw data saved to: {csv_path}")

    # Save basic statistics as JSON
    stats = (
        df.groupby("quality_factor")["lpips_score"]
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

    json_path = os.path.join(OUTPUT_DIR, "lpips_statistics.json")
    with open(json_path, "w") as f:
        json.dump(stats_dict, f, indent=4)
    print(f"Statistics saved to: {json_path}")

    # Create violin plot
    plot_path = os.path.join(OUTPUT_DIR, "lpips_violin_plot.png")
    create_violin_plot(df, plot_path)

    print("Analysis complete!")


if __name__ == "__main__":
    main()
