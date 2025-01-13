import os
import shutil
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
import random
import cv2
import numpy as np
from PIL import Image
from tqdm import tqdm
import json


def clean_and_create_directory_structure(base_path):
    """
    Creates or cleans the directory structure for organizing processed images.
    The structure includes:
    - patches/extracted (input directory - must exist)
    - patches/compressed (for JPEG compressed versions)
    - patches/distorted (for images with visual artifacts)
    """

    base_path = Path(base_path)
    base_dirs = [
        base_path / "train" / "compressed",
        base_path / "val" / "compressed",
        base_path / "test" / "compressed",
    ]

    # Create or clean output directories
    for dir_path in base_dirs:
        if os.path.exists(dir_path):
            print(f"Cleaning directory: {dir_path}")
            shutil.rmtree(dir_path)
        os.makedirs(dir_path)
        print(f"Creating directory: {dir_path}")


def generate_quality_values(image_files, quality_range=(40, 50)):
    """
    Generates a dictionary mapping image names to their randomly assigned
    compression quality values.

    Args:
        image_files: List of Path objects representing the images
        quality_range: Tuple of (min_quality, max_quality)

    Returns:
        Dictionary mapping image names to quality values
    """
    # Create a dictionary with image names as keys and random qualities as values
    quality_mapping = {
        img_path.name: random.randint(quality_range[0], quality_range[1])
        for img_path in image_files
    }

    return quality_mapping


def save_quality_mapping(quality_mapping, split_name, base_path):
    """
    Saves the quality mapping to a JSON file.

    Args:
        quality_mapping: Dictionary mapping image names to quality values
        split_name: Name of the dataset split (train/val/test)
        base_path: Base directory path
    """
    output_path = Path(base_path) / split_name / "quality_mapping.json"
    with open(output_path, "w") as f:
        json.dump(quality_mapping, f, indent=4)


def apply_compression_artifacts(img_path, output_path, quality):
    """
    Applies JPEG compression.
    Uses PIL for efficient JPEG compression.
    """

    with Image.open(img_path) as img:
        if img.mode != "RGB":
            img = img.convert("RGB")
        img.save(output_path, "JPEG", quality=quality, optimize=True)


def apply_distortions(img_path, output_path):
    """
    Applies random visual distortions to the image.
    Uses OpenCV for efficient image processing.
    """
    img = cv2.imread(img_path)

    # Randomly choose distortion type
    distortion_type = random.choice(["color_shift", "gaussian_blur"])

    if distortion_type == "gaussian_blur":
        kernel_size = random.choice([3, 5, 7])
        sigma = random.uniform(0.3, 2.0)
        distorted = cv2.GaussianBlur(img, (kernel_size, kernel_size), sigma)

    else:  # color_shift
        b, g, r = cv2.split(img)
        shift_x = random.randint(1, 5)
        b_shifted = np.roll(b, -shift_x, axis=1)
        r_shifted = np.roll(r, shift_x, axis=1)
        distorted = cv2.merge([b_shifted, g, r_shifted])

    cv2.imwrite(output_path, distorted)


def process_single_image(args):
    """
    Processes a single image by applying both compression and distortion.
    This function is designed to be used with parallel processing.
    """
    input_path, compressed_path, distorted_path, quality = args

    try:
        # Apply both transformations
        apply_compression_artifacts(input_path, compressed_path, quality)
        # apply_distortions(input_path, distorted_path)
        return True
    except Exception as e:
        print(f"Error processing {input_path}: {str(e)}")
        return False


def process_split(split_name, base_path):
    """
    Process all images in a specific dataset split

    Args:
        split_name: Name of the split (train/val/test)
        base_path: Root directory of the dataset
    """
    input_dir = Path(base_path) / split_name / "extracted"
    if not input_dir.exists():
        raise ValueError(f"Input directory '{input_dir}' not found!")

    image_files = (
        list(input_dir.glob("*.jpg"))
        + list(input_dir.glob("*.jpeg"))
        + list(input_dir.glob("*.png"))
    )

    # Generate quality mapping before parallel processing
    quality_mapping = generate_quality_values(image_files)

    # Save the quality mapping
    save_quality_mapping(quality_mapping, split_name, base_path)

    # Prepare arguments for parallel processing
    process_args = [
        (
            str(img_path),
            str(Path(base_path) / split_name / "compressed" / img_path.name),
            str(Path(base_path) / split_name / "distorted" / img_path.name),
            quality_mapping[img_path.name],  # Pass the pre-determined quality value
        )
        for img_path in image_files
    ]

    # Process images in parallel with progress bar
    num_workers = os.cpu_count()
    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        results = list(
            tqdm(
                executor.map(process_single_image, process_args),
                total=len(process_args),
                desc=f"Processing {split_name} split",
            )
        )

    return sum(results), len(results) - sum(results)


def main():
    BASE_PATH = "dataset_attention"
    splits = ["train", "val", "test"]

    total_successful = 0
    total_failed = 0

    clean_and_create_directory_structure(base_path="dataset_attention")

    for split in splits:
        print(f"\nProcessing {split} split...")
        successful, failed = process_split(split, BASE_PATH)
        total_successful += successful
        total_failed += failed
        print(f"{split} split complete:")
        print(f"Successfully processed: {successful} images")
        print(f"Failed: {failed} images")

    print("\nTotal processing complete:")
    print(f"Total successfully processed: {total_successful} images")
    print(f"Total failed: {total_failed} images")


if __name__ == "__main__":
    main()
