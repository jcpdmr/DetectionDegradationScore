import os
import shutil
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from PIL import Image
from tqdm import tqdm
from typing import List


def clean_and_create_directory_structure(
    base_path, quality_values: List[int], splits: List[str]
):
    """
    Creates or cleans the directory structure for GT and compressed images.

    Args:
        base_path: Base directory of the dataset
        quality_values: List of quality values to create directories for
        splits: List of dataset splits to process
    """
    base_path = Path(base_path)

    # Create compressed directories for each quality value and split
    for split in splits:
        for quality in quality_values:
            compressed_dir = base_path / split / f"compressed{quality}"
            if os.path.exists(compressed_dir):
                print(f"Cleaning directory: {compressed_dir}")
                shutil.rmtree(compressed_dir)
            os.makedirs(compressed_dir)
            print(f"Creating directory: {compressed_dir}")


def apply_compression_artifacts(img_path, output_path, quality):
    """
    Applies JPEG compression with specified quality.
    """
    with Image.open(img_path) as img:
        if img.mode != "RGB":
            img = img.convert("RGB")
        img.save(
            output_path,
            "JPEG",
            quality=quality,
            optimize=True,
            subsampling="4:4:4",
        )


def process_single_image(args):
    """
    Processes a single image by applying compression at different quality levels.
    """
    input_path, base_output_dir, quality_values = args
    results = {}

    try:
        input_path = Path(input_path)
        for quality in quality_values:
            output_dir = base_output_dir / f"compressed{quality}"
            output_path = output_dir / input_path.name
            apply_compression_artifacts(input_path, output_path, quality)
            results[quality] = True
        return input_path.name, results
    except Exception as e:
        print(f"Error processing {input_path}: {str(e)}")
        return input_path.name, {q: False for q in quality_values}


def process_split(split_name, base_path, quality_values: List[int]):
    """
    Process all images in a specific dataset split with multiple compression qualities
    """
    base_path = Path(base_path)
    input_dir = base_path / split_name / "extracted"
    if not input_dir.exists():
        raise ValueError(f"Input directory '{input_dir}' not found!")

    # Get all image files
    image_files = (
        list(input_dir.glob("*.jpg"))
        + list(input_dir.glob("*.jpeg"))
        + list(input_dir.glob("*.png"))
    )

    # Prepare arguments for parallel processing
    process_args = [
        (str(img_path), base_path / split_name, quality_values)
        for img_path in image_files
    ]

    # Process images in parallel with progress bar
    results = {}
    num_workers = os.cpu_count()
    with ThreadPoolExecutor(max_workers=num_workers) as executor:
        for img_name, img_results in tqdm(
            executor.map(process_single_image, process_args),
            total=len(process_args),
            desc=f"Processing {split_name} split",
        ):
            results[img_name] = img_results

    # Calculate statistics
    success_count = {q: 0 for q in quality_values}
    fail_count = {q: 0 for q in quality_values}

    for img_results in results.values():
        for quality, success in img_results.items():
            if success:
                success_count[quality] += 1
            else:
                fail_count[quality] += 1

    return success_count, fail_count


def main():
    BASE_PATH = (
        "/andromeda/personal/jdamerini/unbalanced_dataset_coco2017"  # Updated path
    )
    splits = ["train", "val", "test"]  # Now processing all splits
    quality_values = [20, 25, 30, 35, 40, 45, 50]  # You can adjust these quality values

    # Create directory structure
    clean_and_create_directory_structure(
        base_path=BASE_PATH, quality_values=quality_values, splits=splits
    )

    for split in splits:
        print(f"\nProcessing {split} split...")
        success_count, fail_count = process_split(
            split_name=split, base_path=BASE_PATH, quality_values=quality_values
        )

        print(f"\n{split} split complete:")
        for quality in success_count.keys():
            print(f"Quality {quality}:")
            print(f"Successfully processed: {success_count[quality]} images")
            print(f"Failed: {fail_count[quality]} images")
            print()


if __name__ == "__main__":
    main()
