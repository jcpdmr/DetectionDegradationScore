import json
import random


def process_dataset_random_swap(input_path, output_path):
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

    print("\nProcessing complete:")
    print(f"Total images: {total}")
    print(f"Swapped images: {swap_count} ({swap_count / total * 100:.2f}%)")
    print(
        f"Non-swapped images: {total - swap_count} ({(total - swap_count) / total * 100:.2f}%)"
    )

    return results


if __name__ == "__main__":
    input_path = "error_scores_analysis/mapping/04_visual_genome_320p_qual_16_24_28_35_45_55/h_values_v3.json"
    output_path = "error_scores_analysis/mapping/04_visual_genome_320p_qual_16_24_28_35_45_55/h_values_with_swap_v3.json"

    # Set random seed for reproducibility
    random.seed(42)

    results = process_dataset_random_swap(input_path, output_path)

    # Print example of one image before and after processing
    print("\nExample of processed image:")
    first_img = next(iter(results))
    print(json.dumps({first_img: results[first_img]}, indent=4))
