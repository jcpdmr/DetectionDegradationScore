import json
import numpy as np


def analyze_h_distributions(input_path):
    # Load data
    with open(input_path, "r") as f:
        data = json.load(f)

    # Extract h values
    h_types = [
        "h_diff",
        "h_ratio",
        "h_threshold",
        "h_hybrid",
        "h_diff_new",
        "h_ratio_new",
        "h_threshold_new",
        "h_hybrid_new",
    ]
    h_values = {h_type: [] for h_type in h_types}

    for img_name, scores in data.items():
        for h_type in h_types:
            h_values[h_type].append(scores[h_type])

    # Define bins
    bins = np.linspace(0, 1.01, 6)

    # Calculate histograms
    histograms = {}
    for h_type in h_types:
        hist, _ = np.histogram(h_values[h_type], bins=bins)
        histograms[h_type] = hist

        # Print distribution
        print(f"\nDistribution for {h_type}:")
        for i in range(len(hist)):
            print(f"[{bins[i]:.1f}, {bins[i + 1]:.1f}): {hist[i]} samples")
        print(f"Total samples: {sum(hist)}")
        print(f"Mean value: {np.mean(h_values[h_type]):.3f}")
        print(f"Std deviation: {np.std(h_values[h_type]):.3f}")

    return histograms, bins, h_values


if __name__ == "__main__":
    input_path = "error_scores_analysis/mapping/04_visual_genome_320p_qual_16_24_28_35_45_55/h_values_with_swap_v2.json"

    histograms, bins, h_values = analyze_h_distributions(input_path)
