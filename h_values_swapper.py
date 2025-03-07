import json
import random
from pathlib import Path


def process_dataset_random_swap(input_path, output_path, seed=42):
    """
    Process a dataset by randomly swapping quality scores for each image
    
    Args:
        input_path: Path to input h_values json file
        output_path: Path to output file with swapped values
        seed: Random seed for reproducibility
    
    Returns:
        Dictionary containing processed results
    """
    # Set random seed for this specific processing
    random.seed(seed)
    
    # Load original h_values
    with open(input_path, "r") as f:
        h_values = json.load(f)

    # Process each image
    results = {}
    for img_name, scores in h_values.items():
        # Get all original scores
        result_dict = scores.copy()  # Keep all original scores

        # Randomly decide if we should swap
        should_swap = random.random() < 0.5

        # Add new information about swap and new h scores
        result_dict["swapped"] = should_swap

        # For each h_score type, calculate the new score if swapped
        h_types = ["h_diff", "h_prop_diff"]
        for h_type in h_types:
            result_dict[f"{h_type}_new"] = (
                1 - scores[h_type] if should_swap else scores[h_type]
            )
        result_dict["e0_new"] = scores["e1"] if should_swap else scores["e0"]
        result_dict["e1_new"] = scores["e0"] if should_swap else scores["e1"]
        # Store results
        results[img_name] = result_dict

    # Save results
    with open(output_path, "w") as f:
        json.dump(results, f, indent=4)

    # Calculate some statistics
    swap_count = sum(1 for img in results.values() if img["swapped"])
    total = len(results)

    print(f"Total images: {total}")
    print(f"Swapped images: {swap_count} ({swap_count / total * 100:.2f}%)")
    print(
        f"Non-swapped images: {total - swap_count} ({(total - swap_count) / total * 100:.2f}%)"
    )

    return results


def main():
    """Process all splits with random swap"""
    # Base directory containing the h_values files
    base_dir = Path("error_scores_analysis/mapping/07_coco17complete_320p_qual_20_25_30_35_40_45_50_subsamp_444")
    
    # Splits to process
    splits = ["train", "val", "test"]
    
    # Set master random seed for reproducibility
    master_seed = 42
    random.seed(master_seed)
    
    # Process each split
    for split in splits:
        split_dir = base_dir / split
        
        # Skip if directory doesn't exist
        if not split_dir.exists():
            print(f"Directory not found for {split} split, skipping: {split_dir}")
            continue
            
        input_path = split_dir / "h_values_v3.json"
        # Skip if file doesn't exist
        if not input_path.exists():
            print(f"h_values file not found for {split} split, skipping: {input_path}")
            continue
            
        output_path = split_dir / "h_values_with_swap_v3.json"
        
        print(f"\nProcessing {split} split...")
        try:
            # Use a different seed for each split, but derived from master_seed
            split_seed = master_seed + hash(split) % 1000
            results = process_dataset_random_swap(input_path, output_path, seed=split_seed)
            
            # Print example of one image
            if results:
                first_img = next(iter(results))
                print(f"\nExample of processed image from {split} split:")
                print(json.dumps({first_img: results[first_img]}, indent=2))
                
            print(f"Saved results to: {output_path}")
        except Exception as e:
            print(f"Error processing {split} split: {str(e)}")
    
    print("\nAll processing complete!")


if __name__ == "__main__":
    main()