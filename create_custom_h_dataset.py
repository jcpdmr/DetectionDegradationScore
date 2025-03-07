import json
import os
import shutil
import numpy as np
from pathlib import Path
from PIL import Image
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor
import sys


def create_directory_structure(base_path, splits, clean=True):
    """
    Create the required directory structure for all splits (train, val, test)
    Cleans individual split directories if requested, rather than the entire base directory

    Args:
        base_path (str): Base path where to create the directory structure (already includes 2afc)
        splits (list): List of splits to create directories for
        clean (bool): If True, removes existing split directories before creating new ones
    """
    # Define the base directory
    base_dir = Path(base_path)
    
    # Make sure the base directory exists
    base_dir.mkdir(parents=True, exist_ok=True)
    
    # Create directories for each split
    for split in splits:
        split_dir = base_dir / split / "custom"
        
        # Clean this specific split directory if requested
        if clean and split_dir.exists():
            print(f"Cleaning directory structure for split: {split}")
            shutil.rmtree(split_dir)
        
        # Create subdirectories for this split
        for subdir in ["judge", "p0", "p1", "ref", "e0", "e1"]:
            dir_path = split_dir / subdir
            dir_path.mkdir(parents=True, exist_ok=True)
            print(f"Created directory: {dir_path}")


def load_and_convert_image(src_path, dst_path):
    """Load JPG image and save as PNG"""
    try:
        if not os.path.exists(src_path):
            print(f"Error: Source image not found: {src_path}")
            return False
            
        img = Image.open(src_path)
        img.save(dst_path, "PNG")
        
        # Verify that the destination file was created
        if not os.path.exists(dst_path):
            print(f"Error: Failed to create destination image: {dst_path}")
            return False
            
        return True
    except Exception as e:
        print(f"Error converting image {src_path} to {dst_path}: {str(e)}")
        return False


def process_single_image(args):
    """Process a single image (to be used with multiprocessing)"""
    img_name, img_data, split_name, src_base_path, dst_base_path = args

    try:
        # Verify the required keys exist in img_data
        required_keys = ["h_diff_new", "e0_new", "e1_new", "swapped"]
        for key in required_keys:
            if key not in img_data:
                print(f"Error: Missing required key '{key}' for image {img_name}")
                return False
        
        base_name = os.path.splitext(img_name)[0]

        # Save h_diff_new as numpy array
        h_value = np.array(img_data["h_diff_new"]).reshape(1, 1, 1)
        h_path = f"{dst_base_path}/{split_name}/custom/judge/{base_name}.npy"
        np.save(h_path, h_value)
        
        # Verify the judge file was created
        if not os.path.exists(h_path):
            print(f"Error: Failed to create judge file: {h_path}")
            return False
        
        # Save e0_new and e1_new as numpy array
        e0_value = np.array(img_data["e0_new"]).reshape(1, 1, 1)
        e0_path = f"{dst_base_path}/{split_name}/custom/e0/{base_name}.npy"
        np.save(e0_path, e0_value)
        
        # Verify the e0 file was created
        if not os.path.exists(e0_path):
            print(f"Error: Failed to create e0 file: {e0_path}")
            return False
            
        e1_value = np.array(img_data["e1_new"]).reshape(1, 1, 1)
        e1_path = f"{dst_base_path}/{split_name}/custom/e1/{base_name}.npy"
        np.save(e1_path, e1_value)
        
        # Verify the e1 file was created
        if not os.path.exists(e1_path):
            print(f"Error: Failed to create e1 file: {e1_path}")
            return False

        # Define paths for source images
        gt_path = f"{src_base_path}/{split_name}/extracted/{img_name}"
        comp25_path = f"{src_base_path}/{split_name}/compressed25/{img_name}"
        comp50_path = f"{src_base_path}/{split_name}/compressed50/{img_name}"
        
        # Verify source paths exist
        for path in [gt_path, comp25_path, comp50_path]:
            if not os.path.exists(path):
                print(f"Error: Source image not found: {path}")
                return False

        # Define paths for destination images
        ref_dst = f"{dst_base_path}/{split_name}/custom/ref/{base_name}.png"
        p0_dst = f"{dst_base_path}/{split_name}/custom/p0/{base_name}.png"
        p1_dst = f"{dst_base_path}/{split_name}/custom/p1/{base_name}.png"

        # Copy and convert images
        if not load_and_convert_image(gt_path, ref_dst):
            return False

        if img_data["swapped"]:
            if not load_and_convert_image(comp50_path, p0_dst):
                return False
            if not load_and_convert_image(comp25_path, p1_dst):
                return False
        else:
            if not load_and_convert_image(comp25_path, p0_dst):
                return False
            if not load_and_convert_image(comp50_path, p1_dst):
                return False

        return True
    except Exception as e:
        print(f"Error processing {img_name}: {str(e)}")
        return False


def verify_source_dataset(src_base_path, splits, quality_values):
    """
    Verify that the source dataset structure is valid
    
    Args:
        src_base_path: Base path for source images
        splits: List of splits to verify
        quality_values: List of compression quality values to verify
        
    Returns:
        bool: True if dataset structure is valid, False otherwise
    """
    print("\nVerifying source dataset structure...")
    
    # Check base path exists
    if not os.path.exists(src_base_path):
        print(f"Error: Source base path not found: {src_base_path}")
        return False
    
    # Check each split
    for split in splits:
        split_path = os.path.join(src_base_path, split)
        if not os.path.exists(split_path):
            print(f"Error: Split directory not found: {split_path}")
            return False
            
        # Check extracted directory
        extracted_path = os.path.join(split_path, "extracted")
        if not os.path.exists(extracted_path):
            print(f"Error: Extracted directory not found: {extracted_path}")
            return False
            
        # Check compression quality directories
        for quality in quality_values:
            quality_path = os.path.join(split_path, f"compressed{quality}")
            if not os.path.exists(quality_path):
                print(f"Error: Compressed directory not found: {quality_path}")
                return False
    
    print("Source dataset structure verification passed.")
    return True


def verify_h_values_files(mapping_dir, splits):
    """
    Verify that h_values files exist for all splits
    
    Args:
        mapping_dir: Base directory for h_values files
        splits: List of splits to verify
        
    Returns:
        bool: True if all h_values files exist, False otherwise
    """
    print("\nVerifying h_values files...")
    
    all_files_exist = True
    
    for split in splits:
        h_values_path = f"{mapping_dir}/{split}/h_values_with_swap_v3.json"
        if not os.path.exists(h_values_path):
            print(f"Error: h_values file not found: {h_values_path}")
            all_files_exist = False
            
    if all_files_exist:
        print("h_values files verification passed.")
    
    return all_files_exist


def prepare_split_dataset(
    h_values_path, src_base_path, dst_base_path, split, num_workers=None, max_images=None
):
    """
    Prepare dataset for a single split (train, val, or test)
    
    Args:
        h_values_path: Path to h_values_with_swap JSON file
        src_base_path: Base path for source images
        dst_base_path: Base path for destination dataset
        split: Split name (train, val, test)
        num_workers: Number of workers for parallel processing
        max_images: Maximum number of images to process (for testing)
        
    Returns:
        tuple: (total_images, successful_images)
    """
    if not os.path.exists(h_values_path):
        print(f"Error: h_values file not found at {h_values_path}")
        return 0, 0
        
    print(f"\nProcessing {split} split...")
    
    # Load h_values with swap information
    try:
        with open(h_values_path, "r") as f:
            h_values = json.load(f)
    except json.JSONDecodeError as e:
        print(f"Error parsing JSON file {h_values_path}: {str(e)}")
        return 0, 0
    
    print(f"Loaded {len(h_values)} images from {h_values_path}")
    
    # Limit number of images if requested (for testing)
    image_list = list(h_values.items())
    if max_images and max_images < len(image_list):
        print(f"Limiting to {max_images} images for testing")
        image_list = image_list[:max_images]
    
    # Prepare arguments for parallel processing
    process_args = [
        (img_name, img_data, split, src_base_path, dst_base_path)
        for img_name, img_data in image_list
    ]
    
    # Process images in parallel with progress bar
    with ThreadPoolExecutor(max_workers=num_workers) as executor:
        results = list(
            tqdm(
                executor.map(process_single_image, process_args),
                total=len(process_args),
                desc=f"Processing {split} split"
            )
        )
    
    # Count successful processing
    successful = sum(1 for r in results if r)
    failed = len(results) - successful
    
    print(f"Successfully processed {successful}/{len(results)} images in {split} split")
    if failed > 0:
        print(f"Failed to process {failed} images")
    
    return len(results), successful


def prepare_dataset(src_base_path, dst_base_path, num_workers=None, max_images=None):
    """
    Prepare complete dataset for all splits
    
    Args:
        src_base_path: Base path for source dataset
        dst_base_path: Base path for destination dataset
        num_workers: Number of workers for parallel processing
        max_images: Maximum number of images to process per split (for testing)
    """
    if num_workers is None:
        num_workers = os.cpu_count()  # Use all available CPUs

    print(f"Using {num_workers} workers")

    # Define splits to process
    splits = ["train", "val", "test"]
    
    # Define quality values to verify
    quality_values = [25, 50]
    
    # Verify source dataset structure
    if not verify_source_dataset(src_base_path, splits, quality_values):
        print("Error: Source dataset verification failed. Aborting.")
        sys.exit(1)
    
    # Define mapping directory
    mapping_dir = "error_scores_analysis/mapping/07_coco17complete_320p_qual_20_25_30_35_40_45_50_subsamp_444"
    
    # Verify h_values files
    if not verify_h_values_files(mapping_dir, splits):
        print("Error: h_values files verification failed. Aborting.")
        sys.exit(1)
    
    # Create directory structure
    create_directory_structure(dst_base_path, splits)
    
    # Process each split
    split_stats = {}
    
    for split in splits:
        h_values_path = f"{mapping_dir}/{split}/h_values_with_swap_v3.json"
        
        # Process this split
        total, successful = prepare_split_dataset(
            h_values_path=h_values_path,
            src_base_path=src_base_path,
            dst_base_path=dst_base_path,
            split=split,
            num_workers=num_workers,
            max_images=max_images
        )
        
        split_stats[split] = {"total": total, "successful": successful}
    
    # Verify destination dataset
    print("\nVerifying destination dataset...")
    for split in splits:
        successful = split_stats[split]["successful"]
        
        # Check that files were created (sampling a few directories)
        subdirs = ["judge", "p0", "p1", "ref", "e0", "e1"]
        for subdir in subdirs:
            dir_path = f"{dst_base_path}/{split}/custom/{subdir}"
            file_count = len(os.listdir(dir_path))
            print(f"{split}/{subdir}: {file_count} files")
            
            if file_count != successful:
                print(f"Warning: Expected {successful} files but found {file_count} in {dir_path}")
    
    # Print summary
    print("\nDataset preparation complete:")
    total_successful = 0
    total_images = 0
    
    for split, stats in split_stats.items():
        total = stats["total"]
        successful = stats["successful"]
        total_images += total
        total_successful += successful
        
        print(f"{split.capitalize()} set: {successful}/{total} successfully processed ({successful/total*100:.1f}%)")
    
    print(f"\nOverall: {total_successful}/{total_images} successfully processed ({total_successful/total_images*100:.1f}%)")
    
    if total_successful < total_images:
        print("Warning: Some images were not processed successfully. Check the logs for details.")


if __name__ == "__main__":
    # Define paths
    src_base_path = "/andromeda/personal/jdamerini/unbalanced_dataset_coco2017"
    dst_base_path = "../PerceptualSimilarity/LPIPS_dataset_coco2017/2afc"
    
    # For testing, uncomment the next line to process only a few images per split
    # max_images = 10
    max_images = None

    # Prepare the dataset
    prepare_dataset(src_base_path, dst_base_path, max_images=max_images)