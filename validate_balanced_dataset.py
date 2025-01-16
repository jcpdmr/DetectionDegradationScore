import json
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict


def validate_dataset(json_file):
    """
    Validate the dataset and create distribution visualization.

    Args:
        json_file (str): Path to JSON file containing the dataset
    """
    # Load the data
    with open(json_file, "r") as f:
        data = json.load(f)

    selected_items = data["selected_items"]

    # Check for duplicates
    image_count = defaultdict(int)
    for img_name in selected_items:
        image_count[img_name] += 1

    duplicates = {img: count for img, count in image_count.items() if count > 1}
    if duplicates:
        print("WARNING: Found duplicate images:")
        for img, count in duplicates.items():
            print(f"Image {img} appears {count} times")
    else:
        print("No duplicate images found.")

    # Create bin edges (21 bins from 0 to 20)
    n_bins = 21  # Changed to 21 bins
    bin_edges = np.linspace(0, 1.05, n_bins + 1)

    # Validate bin assignments and count distribution
    bin_distribution = np.zeros(n_bins, dtype=int)
    incorrect_bins = []

    for img_name, info in selected_items.items():
        score = info["score"]
        assigned_bin = info["bin"]

        # Find the correct bin
        correct_bin = None
        for i in range(len(bin_edges) - 1):
            if bin_edges[i] <= score < bin_edges[i + 1]:
                correct_bin = i
                break

        if correct_bin != assigned_bin:
            incorrect_bins.append(
                {
                    "image": img_name,
                    "score": score,
                    "assigned_bin": assigned_bin,
                    "correct_bin": correct_bin,
                    "bin_range": f"[{bin_edges[assigned_bin]:.3f}, {bin_edges[assigned_bin + 1]:.3f})",
                }
            )

        # Count for distribution
        if 0 <= assigned_bin < n_bins:  # Ensure bin index is valid
            bin_distribution[assigned_bin] += 1

    # Report incorrect bin assignments
    if incorrect_bins:
        print("\nWARNING: Found incorrect bin assignments:")
        for error in incorrect_bins:
            print(
                f"Image {error['image']} with score {error['score']:.4f} "
                f"is in bin {error['assigned_bin']} (range {error['bin_range']}) "
                f"but should be in bin {error['correct_bin']}"
            )
    else:
        print("\nAll bin assignments are correct.")

    # Create and show distribution plot
    plt.figure(figsize=(15, 7))
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    plt.bar(bin_centers, bin_distribution, width=0.04, alpha=0.7)
    plt.xlabel("Score Range")
    plt.ylabel("Number of Images")
    plt.title("Distribution of Images Across Bins (0-20)")
    plt.grid(True, alpha=0.3)

    # Add bin edges annotations
    for i, count in enumerate(bin_distribution):
        if count > 0:  # Only annotate non-empty bins
            plt.text(
                bin_centers[i], count, f"Bin {i}\n({count})", ha="center", va="bottom"
            )

    # Add x-axis ticks at bin edges
    plt.xticks(bin_edges[::2], [f"{x:.2f}" for x in bin_edges[::2]], rotation=45)

    plt.tight_layout()
    plt.savefig("balanced_dataset_distribution.png")

    return {
        "has_duplicates": bool(duplicates),
        "incorrect_bins": incorrect_bins,
        "distribution": bin_distribution.tolist(),
        "bin_edges": bin_edges.tolist(),
    }


# Example usage
results = validate_dataset("balanced_dataset_21bins.json")
