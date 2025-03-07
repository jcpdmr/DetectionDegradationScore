import json
import numpy as np
from pathlib import Path


def sigmoid(x, k=1):
    return 1 / (1 + np.exp(-k * x))


def load_error_scores(filepath):
    with open(filepath, "r") as f:
        return json.load(f)


def calculate_h_values(e0, e1):
    """Calculate h using different approaches
    e0: error score for quality 25
    e1: error score for quality 50
    """
    results = {}

    # 1. Simple difference normalized to [0,1]
    diff = e0 - e1
    results["diff(e25-e50)"] = diff
    results["h_diff"] = float(
        # the higher k, the steeper the sigmoid
        sigmoid(diff, k=15)
    )

    # 2. Ratio approach - handle special cases
    if e1 == 0:
        if e0 == 0:
            # If both are zero, ratio is 1 (no difference)
            ratio = 1.0
        else:
            # If only denominator is zero, set to a large value
            ratio = 1000.0  # Large value to indicate very high ratio
    else:
        ratio = e0 / e1
    
    # Use safe log (handle values <= 0)
    log_ratio = np.log(max(ratio, 1e-10))
    results["h_ratio"] = float(sigmoid(log_ratio, k=2))

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


def process_split_error_scores(input_path, output_path):
    """Process error scores for a single split (train, val, or test)"""
    # Load data
    error_scores = load_error_scores(input_path)

    # Process each image
    results = {}
    for img_name, qualities in error_scores.items():
        # Get error scores for quality 25 and 50
        e0 = qualities["25"]
        e1 = qualities["50"]

        # Calculate h values using different approaches
        h_values = calculate_h_values(e0, e1)

        # Store results
        results[img_name] = {"e0": e0, "e1": e1, **h_values}

    # Save results
    with open(output_path, "w") as f:
        json.dump(results, f, indent=4)

    return results


def main():
    """Process error scores for all splits (train, val, test)"""
    # Base directory containing the error scores
    base_dir = Path("error_scores_analysis/mapping/07_coco17complete_320p_qual_20_25_30_35_40_45_50_subsamp_444")
    
    # Splits to process
    splits = ["train", "val", "test"]
    
    # Process each split
    for split in splits:
        split_dir = base_dir / split
        
        # Skip if directory doesn't exist
        if not split_dir.exists():
            print(f"Directory not found for {split} split, skipping: {split_dir}")
            continue
            
        input_path = split_dir / "error_scores.json"
        # Skip if file doesn't exist
        if not input_path.exists():
            print(f"Error scores file not found for {split} split, skipping: {input_path}")
            continue
            
        output_path = split_dir / "h_values_v3.json"
        
        print(f"Processing {split} split...")
        try:
            results = process_split_error_scores(input_path, output_path)
            print(f"  Processed {len(results)} images")
            print(f"  Saved results to: {output_path}")
            
            # Print some statistics for the first image
            if results:
                first_img = next(iter(results))
                print(f"  Sample results for first image ({first_img}):")
                print(f"  {json.dumps(results[first_img], indent=2)}")
        except Exception as e:
            print(f"  Error processing {split} split: {str(e)}")
    
    print("Analysis complete!")


if __name__ == "__main__":
    main()