import json
from pathlib import Path

def check_negative_values(file_path):
    """
    Check for negative values in h_diff and h_diff_new fields
    
    Args:
        file_path: Path to the h_values_with_swap file
        
    Returns:
        tuple: (has_negative_h_diff, has_negative_h_diff_new, stats)
    """
    with open(file_path, 'r') as f:
        data = json.load(f)
    
    negative_h_diff = []
    negative_h_diff_new = []
    
    # Statistics
    stats = {
        "total_images": len(data),
        "h_diff_min": float('inf'),
        "h_diff_max": float('-inf'),
        "h_diff_new_min": float('inf'),
        "h_diff_new_max": float('-inf')
    }
    
    for img_name, values in data.items():
        # Check h_diff
        h_diff = values.get('h_diff', 0)
        stats["h_diff_min"] = min(stats["h_diff_min"], h_diff)
        stats["h_diff_max"] = max(stats["h_diff_max"], h_diff)
        
        if h_diff < 0:
            negative_h_diff.append((img_name, h_diff))
        
        # Check h_diff_new
        h_diff_new = values.get('h_diff_new', 0)
        stats["h_diff_new_min"] = min(stats["h_diff_new_min"], h_diff_new)
        stats["h_diff_new_max"] = max(stats["h_diff_new_max"], h_diff_new)
        
        if h_diff_new < 0:
            negative_h_diff_new.append((img_name, h_diff_new))
    
    return negative_h_diff, negative_h_diff_new, stats

def main():
    # Base directory containing the h_values files
    base_dir = Path("error_scores_analysis/mapping/07_coco17complete_320p_qual_20_25_30_35_40_45_50_subsamp_444")
    
    # Splits to check
    splits = ["train", "val", "test"]
    
    for split in splits:
        print(f"\n{'=' * 60}")
        print(f"Checking {split} split")
        print(f"{'=' * 60}")
        
        input_path = base_dir / split / "h_values_with_swap_v3.json"
        
        if not input_path.exists():
            print(f"File not found: {input_path}")
            continue
        
        try:
            negative_h_diff, negative_h_diff_new, stats = check_negative_values(input_path)
            
            print(f"Total images in {split}: {stats['total_images']}")
            print(f"h_diff range: [{stats['h_diff_min']:.6f}, {stats['h_diff_max']:.6f}]")
            print(f"h_diff_new range: [{stats['h_diff_new_min']:.6f}, {stats['h_diff_new_max']:.6f}]")
            
            if negative_h_diff:
                print(f"\nFound {len(negative_h_diff)} images with negative h_diff values:")
                for img_name, value in negative_h_diff[:5]:  # Show only first 5
                    print(f"  {img_name}: {value}")
                if len(negative_h_diff) > 5:
                    print(f"  ... and {len(negative_h_diff) - 5} more")
            else:
                print("\nNo negative h_diff values found")
            
            if negative_h_diff_new:
                print(f"\nFound {len(negative_h_diff_new)} images with negative h_diff_new values:")
                for img_name, value in negative_h_diff_new[:5]:  # Show only first 5
                    print(f"  {img_name}: {value}")
                if len(negative_h_diff_new) > 5:
                    print(f"  ... and {len(negative_h_diff_new) - 5} more")
            else:
                print("\nNo negative h_diff_new values found")
            
        except Exception as e:
            print(f"Error checking {split}: {e}")

if __name__ == "__main__":
    main()