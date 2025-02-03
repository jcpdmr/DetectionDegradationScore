import json
import numpy as np


def sigmoid(x, k=1):
    return 1 / (1 + np.exp(-k * x))


def load_error_scores(filepath):
    with open(filepath, "r") as f:
        return json.load(f)


def calculate_h_values(e0, e1):
    """Calculate h using different approaches
    e0: error score for quality 24
    e1: error score for quality 55
    """
    results = {}

    # 1. Simple difference normalized to [0,1]
    diff = e0 - e1
    results["diff(e24-e55)"] = diff
    results["h_diff"] = float(
        # the higher k, the steeper the sigmoid
        sigmoid(diff, k=15)
    )

    # 2. Ratio approach
    ratio = e0 / e1 if e1 != 0 else float("inf")
    results["h_ratio"] = float(sigmoid(np.log(ratio), k=2))  # log makes ratio symmetric

    # 3. Threshold approach
    threshold = 0.1
    if abs(diff) < threshold:
        h_threshold = 0.5
    else:
        h_threshold = 0.0 if diff < 0 else 1.0
    results["h_threshold"] = float(h_threshold)

    # 4. Hybrid approach
    # Combines smooth transition for small differences with more decisive values for large ones
    k = 10  # Controls steepness of transition
    threshold = 0.2
    normalized_diff = diff / threshold
    h_hybrid = sigmoid(k * (abs(normalized_diff) - 1)) * sigmoid(normalized_diff, k=2)
    results["h_hybrid"] = float(h_hybrid)

    # 5. Proportional difference
    h_prop_diff = e0 - e1

    results["h_prop_diff"] = float(h_prop_diff)

    return results


def process_error_scores(input_path, output_path):
    # Load data
    error_scores = load_error_scores(input_path)

    # Process each image
    results = {}
    for img_name, qualities in error_scores.items():
        # Get error scores for quality 24 and 55
        e0 = qualities["24"]
        e1 = qualities["55"]

        # Calculate h values using different approaches
        h_values = calculate_h_values(e0, e1)

        # Store results
        results[img_name] = {"e0": e0, "e1": e1, **h_values}

    # Save results
    with open(output_path, "w") as f:
        json.dump(results, f, indent=4)

    return results


if __name__ == "__main__":
    input_path = "error_scores_analysis/mapping/04_visual_genome_320p_qual_16_24_28_35_45_55/error_scores.json"
    output_path = "error_scores_analysis/mapping/04_visual_genome_320p_qual_16_24_28_35_45_55/h_values_v3.json"

    results = process_error_scores(input_path, output_path)

    # Print some statistics
    print("\nAnalysis complete. Sample results for first image:")
    first_img = next(iter(results))
    print(json.dumps(results[first_img], indent=4))
