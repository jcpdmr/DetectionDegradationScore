import os
import shutil
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
import random
import cv2
import numpy as np
from PIL import Image
from tqdm import tqdm


def create_directory_structure():
    """
    Creates or cleans the directory structure for organizing processed images.
    The structure includes:
    - patches/extracted (input directory - must exist)
    - patches/compressed (for JPEG compressed versions)
    - patches/distorted (for images with visual artifacts)
    """
    base_dirs = ["patches/compressed", "patches/distorted"]

    # Create or clean output directories
    for dir_path in base_dirs:
        if os.path.exists(dir_path):
            shutil.rmtree(dir_path)
        os.makedirs(dir_path)


def apply_compression_artifacts(img_path, output_path, quality_range=(20, 50)):
    """
    Applies JPEG compression with random quality factor.
    Uses PIL for efficient JPEG compression.
    """
    quality = random.randint(quality_range[0], quality_range[1])

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
    input_path, compressed_path, distorted_path = args

    try:
        # Apply both transformations
        apply_compression_artifacts(input_path, compressed_path)
        apply_distortions(input_path, distorted_path)
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

    # Prepare arguments for parallel processing
    process_args = [
        (
            str(img_path),
            str(Path(base_path) / split_name / "compressed" / img_path.name),
            str(Path(base_path) / split_name / "distorted" / img_path.name),
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
