import json
import numpy as np
from pathlib import Path


def analyze_h_distributions(input_path, h_types=None):
    """
    Analyze the distribution of h values in a dataset
    
    Args:
        input_path: Path to input h_values_with_swap json file
        h_types: List of h types to analyze (default: analyze all available)
        
    Returns:
        tuple: (histograms, bins, h_values)
    """
    # Load data
    with open(input_path, "r") as f:
        data = json.load(f)
    
    # Get sample to determine available h_types if not specified
    if h_types is None:
        first_img = next(iter(data))
        h_types = [key for key in data[first_img].keys() if key.startswith('h_')]
    
    # Extract h values
    h_values = {h_type: [] for h_type in h_types}

    for img_name, scores in data.items():
        for h_type in h_types:
            if h_type in scores:
                h_values[h_type].append(scores[h_type])
            else:
                print(f"Warning: {h_type} not found for image {img_name}")

    # Define bins
    bins = np.linspace(0, 1.01, 6)

    # Calculate histograms
    histograms = {}
    for h_type in h_types:
        hist, _ = np.histogram(h_values[h_type], bins=bins)
        # Convert NumPy array to list for JSON serialization
        histograms[h_type] = hist.tolist()

        # Print distribution
        print(f"\nDistribution for {h_type}:")
        for i in range(len(hist)):
            print(f"[{bins[i]:.1f}, {bins[i + 1]:.1f}): {hist[i]} samples")
        print(f"Total samples: {sum(hist)}")
        print(f"Mean value: {np.mean(h_values[h_type]):.3f}")
        print(f"Std deviation: {np.std(h_values[h_type]):.3f}")

    return histograms, bins, h_values


def main():
    """Analyze h distributions for all splits"""
    # Base directory containing the h_values files
    base_dir = Path("error_scores_analysis/mapping/07_coco17complete_320p_qual_20_25_30_35_40_45_50_subsamp_444")
    
    # Splits to analyze
    splits = ["train", "val", "test"]
    
    # h types to analyze
    h_types = [
        "h_diff",
        "h_ratio",
        "h_threshold",
        "h_hybrid",
        "h_prop_diff",
        "h_diff_new",
        "h_prop_diff_new",
    ]
    
    # Process each split
    results = {}
    for split in splits:
        split_dir = base_dir / split
        
        # Skip if directory doesn't exist
        if not split_dir.exists():
            print(f"Directory not found for {split} split, skipping: {split_dir}")
            continue
            
        input_path = split_dir / "h_values_with_swap_v3.json"
        # Skip if file doesn't exist
        if not input_path.exists():
            print(f"h_values file not found for {split} split, skipping: {input_path}")
            continue
        
        print(f"\n{'=' * 50}")
        print(f"Analyzing {split} split")
        print(f"{'=' * 50}")
        
        try:
            histograms, bins, h_values = analyze_h_distributions(input_path, h_types)
            
            # Convert NumPy arrays to lists for JSON serialization
            bins_list = bins.tolist()
            
            results[split] = {
                "histograms": histograms,  # Already converted to list in analyze_h_distributions
                "bins": bins_list,
                "statistics": {
                    h_type: {
                        "mean": float(np.mean(h_values[h_type])),
                        "std": float(np.std(h_values[h_type])),
                        "min": float(np.min(h_values[h_type])),
                        "max": float(np.max(h_values[h_type])),
                    }
                    for h_type in h_types if h_type in h_values and h_values[h_type]
                }
            }
            
            # Save split-specific results
            output_path = split_dir / "h_distribution_analysis.json"
            with open(output_path, "w") as f:
                json.dump(results[split], f, indent=4)
            print(f"\nSaved {split} analysis to: {output_path}")
            
        except Exception as e:
            print(f"Error analyzing {split} split: {str(e)}")
            import traceback
            traceback.print_exc()
    
    # Save combined results
    if results:
        combined_output_path = base_dir / "combined_h_distribution_analysis.json"
        with open(combined_output_path, "w") as f:
            json.dump(results, f, indent=4)
        print(f"\nSaved combined analysis to: {combined_output_path}")
    
    print("\nAnalysis complete!")


if __name__ == "__main__":
    main()