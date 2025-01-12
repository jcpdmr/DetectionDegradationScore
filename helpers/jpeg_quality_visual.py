from pathlib import Path
import json
import matplotlib.pyplot as plt
import numpy as np


def analyze_quality_distribution(base_path, split_name):
    """
    Analyzes and plots the distribution of compression quality values with precise bin control.
    This function creates a detailed histogram where each bin represents exactly one quality value,
    providing a clear view of how compression qualities are distributed across the entire possible range.

    Args:
        base_path: Base directory path for the dataset
        split_name: Name of the dataset split (train/val/test) to analyze
    """

    # Load the quality mapping from the JSON file that stores our compression settings
    quality_mapping_path = Path(base_path) / split_name / "quality_mapping.json"
    with open(quality_mapping_path, "r") as f:
        quality_mapping = json.load(f)

    # Extract all quality values from the mapping
    qualities = list(quality_mapping.values())

    # Create a figure with appropriate dimensions for detailed visualization
    plt.figure(figsize=(15, 6))

    # Create bins edges for the full quality range (0-100)
    # This ensures we see the complete distribution, even where no values exist
    bin_edges = np.arange(
        101
    )  # Creates exactly 100 bins, one for each possible quality value

    # Create the histogram with precise control over bin placement and appearance
    counts, edges, _ = plt.hist(
        qualities,
        bins=bin_edges,
        align="left",  # Aligns bars with their exact quality values
        edgecolor="black",
        rwidth=0.8,
    )  # Makes bars slightly narrower for better visibility

    # Set the axis limits to show the complete quality range
    plt.xlim(-1, 101)  # Slightly wider than 0-100 to prevent edge bars from being cut

    # Add a grid to make value tracking easier
    plt.grid(True, alpha=0.3, linestyle="--")

    # Set regular tick marks every 10 values for easy reading
    plt.xticks(np.arange(0, 101, 10))

    # Add clear, descriptive labels
    plt.title(
        f"Distribution of Compression Quality Values - {split_name} split",
        fontsize=12,
        pad=15,
    )
    plt.xlabel("Quality Value", fontsize=10)
    plt.ylabel("Count", fontsize=10)

    # Ensure the layout looks clean
    plt.tight_layout()

    # Save the visualization with high resolution
    output_path = Path(base_path) / split_name / "quality_distribution.png"
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()

    # Print helpful statistics about the distribution
    print(f"\nQuality Distribution Statistics for {split_name} split:")
    print(f"Total images processed: {len(qualities)}")
    print(f"Quality range: [{min(qualities)}, {max(qualities)}]")


if __name__ == "__main__":
    base_path = "dataset_attention"
    splits = ["train", "val", "test"]

    for split in splits:
        analyze_quality_distribution(base_path, split)
