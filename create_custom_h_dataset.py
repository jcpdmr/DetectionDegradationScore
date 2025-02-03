import json
import os
import shutil
import numpy as np
from pathlib import Path
from PIL import Image
import random
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor


def create_directory_structure(base_path, clean=True):
    """
    Create the required directory structure for train and val sets

    Args:
        base_path (str): Base path where to create the directory structure
        clean (bool): If True, removes existing directories before creating new ones
    """
    # Define the base directory for 2afc
    afc_dir = Path(f"{base_path}/2afc")

    # Clean if requested and directory exists
    if clean and afc_dir.exists():
        print(f"Cleaning existing directory structure at {afc_dir}")
        shutil.rmtree(afc_dir)

    # Create directories
    for split in ["train", "val"]:
        for subdir in ["judge", "p0", "p1", "ref", "e0", "e1"]:
            dir_path = Path(f"{base_path}/2afc/{split}/custom/{subdir}")
            dir_path.mkdir(parents=True, exist_ok=True)
            print(f"Created directory: {dir_path}")


def load_and_convert_image(src_path, dst_path):
    """Load JPG image and save as PNG"""
    img = Image.open(src_path)
    img.save(dst_path, "PNG")


def process_single_image(args):
    """Process a single image (to be used with multiprocessing)"""
    img_name, img_data, split_name, src_base_path, dst_base_path = args

    try:
        base_name = os.path.splitext(img_name)[0]

        # Save h_diff_new as numpy array
        h_value = np.array(img_data["h_diff_new"]).reshape(1, 1, 1)
        np.save(
            f"{dst_base_path}/2afc/{split_name}/custom/judge/{base_name}.npy", h_value
        )
        # Save e0_new and e1_new as numpy array
        e0_value = np.array(img_data["e0_new"]).reshape(1, 1, 1)
        np.save(
            f"{dst_base_path}/2afc/{split_name}/custom/e0/{base_name}.npy", e0_value
        )
        e1_value = np.array(img_data["e1_new"]).reshape(1, 1, 1)
        np.save(
            f"{dst_base_path}/2afc/{split_name}/custom/e1/{base_name}.npy", e1_value
        )

        # Define paths
        gt_path = f"{src_base_path}/train/extracted/{img_name}"
        comp24_path = f"{src_base_path}/train/compressed24/{img_name}"
        comp55_path = f"{src_base_path}/train/compressed55/{img_name}"

        ref_dst = f"{dst_base_path}/2afc/{split_name}/custom/ref/{base_name}.png"
        p0_dst = f"{dst_base_path}/2afc/{split_name}/custom/p0/{base_name}.png"
        p1_dst = f"{dst_base_path}/2afc/{split_name}/custom/p1/{base_name}.png"

        # Copy and convert images
        load_and_convert_image(gt_path, ref_dst)

        if img_data["swapped"]:
            load_and_convert_image(comp55_path, p0_dst)
            load_and_convert_image(comp24_path, p1_dst)
        else:
            load_and_convert_image(comp24_path, p0_dst)
            load_and_convert_image(comp55_path, p1_dst)

        return True
    except Exception as e:
        print(f"Error processing {img_name}: {str(e)}")
        return False


def prepare_dataset(
    h_values_path, src_base_path, dst_base_path, val_ratio=0.1, num_workers=None
):
    if num_workers is None:
        num_workers = os.cpu_count()  # Use all available CPUs

    print(f"Using {num_workers} workers")

    # Load h_values with swap information
    with open(h_values_path, "r") as f:
        h_values = json.load(f)

    # Create directory structure
    create_directory_structure(dst_base_path)

    # Get list of all images and shuffle
    all_images = list(h_values.keys())
    random.shuffle(all_images)

    # Split into train and val
    split_idx = int(len(all_images) * val_ratio)
    val_images = all_images[:split_idx]
    train_images = all_images[split_idx:]

    # Process both splits
    for split_name, image_list in [("val", val_images), ("train", train_images)]:
        print(f"\nProcessing {split_name} set...")

        # Prepare arguments for parallel processing
        process_args = [
            (img_name, h_values[img_name], split_name, src_base_path, dst_base_path)
            for img_name in image_list
        ]

        # Process images in parallel with progress bar
        with ThreadPoolExecutor(max_workers=num_workers) as executor:
            results = list(
                tqdm(
                    executor.map(process_single_image, process_args),
                    total=len(image_list),
                    desc=f"Processing {split_name} set",
                )
            )

        # Count successful processing
        successful = sum(results)
        print(
            f"Successfully processed {successful}/{len(image_list)} images in {split_name} set"
        )

    print("\nDataset preparation complete:")
    print(f"Train set size: {len(train_images)}")
    print(f"Val set size: {len(val_images)}")


if __name__ == "__main__":
    # Set random seed for reproducibility
    random.seed(42)

    # Define paths
    h_values_path = "error_scores_analysis/mapping/04_visual_genome_320p_qual_16_24_28_35_45_55/h_values_with_swap_v3.json"
    src_base_path = "unbalanced_dataset"
    dst_base_path = "../PerceptualSimilarity/dataset"

    # You can specify number of workers or let it use all available CPUs
    prepare_dataset(h_values_path, src_base_path, dst_base_path)
