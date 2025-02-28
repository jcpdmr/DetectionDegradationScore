import cv2
import os
import shutil
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor
from tqdm import tqdm


def create_split_directories(base_path, splits=["train", "val", "test"]):
    """
    Create the complete directory structure for the dataset splits

    Args:
        base_path: Root directory for the dataset
        splits: List of dataset splits to create
    """
    modifications = ["extracted"]

    for split in splits:
        for mod in modifications:
            path = Path(base_path) / split / mod
            if path.exists():
                print(f"Cleaning {path}")
                shutil.rmtree(path)
            print(f"Created {path}")
            path.mkdir(parents=True, exist_ok=True)


def process_image(args):
    """
    Process a single image: resize preserving aspect ratio, then center crop to get 320x320.
    Smaller images are discarded.

    Args:
        args (tuple): (image_path, output_path, target_size)
    """
    image_path, output_path, target_size = args

    # Read image
    img = cv2.imread(str(image_path))
    if img is None:
        return False

    height, width = img.shape[:2]

    # Skip if image is smaller than target size in either dimension
    if height < target_size or width < target_size:
        return False

    # Calculate new dimensions while preserving aspect ratio
    aspect_ratio = width / height

    if aspect_ratio > 1:  # Width > Height (landscape)
        new_width = int(target_size * aspect_ratio)
        new_height = target_size
    else:  # Height >= Width (portrait or square)
        new_width = target_size
        new_height = int(target_size / aspect_ratio)

    # Resize maintaining aspect ratio
    resized = cv2.resize(img, (new_width, new_height), interpolation=cv2.INTER_AREA)

    # Calculate crop coordinates to get center crop
    y_start = (new_height - target_size) // 2
    x_start = (new_width - target_size) // 2

    # Perform center crop
    cropped = resized[y_start : y_start + target_size, x_start : x_start + target_size]

    # Save processed image
    cv2.imwrite(str(output_path), cropped)
    return True


def process_split(input_dir, output_dir, split_name, target_size=320, num_workers=None):
    """
    Process all images from a specific split directory and save to output directory

    Args:
        input_dir: Input directory containing images
        output_dir: Output base directory
        split_name: Name of the split (train, val, test)
        target_size: Target size for the output images
        num_workers: Number of parallel workers
    """
    if num_workers is None:
        num_workers = os.cpu_count()

    input_path = Path(input_dir)
    if not input_path.exists():
        print(f"Warning: Directory {input_dir} does not exist, skipping...")
        return

    # Get image files
    image_files = (
        list(input_path.glob("*.jpg"))
        + list(input_path.glob("*.jpeg"))
        + list(input_path.glob("*.png"))
    )
    print(f"Found {len(image_files)} images in {input_dir}")

    # Prepare arguments for processing
    process_args = [
        (
            img_path,
            Path(output_dir) / split_name / "extracted" / img_path.name,
            target_size,
        )
        for img_path in image_files
    ]

    # Process images in parallel with progress bar
    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        results = list(
            tqdm(
                executor.map(process_image, process_args),
                total=len(process_args),
                desc=f"Processing {split_name} images",
            )
        )

    # Print split statistics
    processed = sum(1 for r in results if r)
    skipped = len(results) - processed
    print(f"{split_name} split complete:")
    print(f"Successfully processed: {processed} images")
    print(f"Skipped (too small or invalid): {skipped} images")


def main():
    # Configuration
    INPUT_DIRS = {
        "train": "/andromeda/personal/jdamerini/train2017",
        "val": "/andromeda/personal/jdamerini/val2017",
        "test": "/andromeda/personal/jdamerini/test2017",
    }
    OUTPUT_DIR = "/andromeda/personal/jdamerini/unbalanced_dataset_coco2017"
    TARGET_SIZE = 320  # Target size for the square output images
    NUM_WORKERS = os.cpu_count()  # Use all available CPU cores

    # Create directory structure
    create_split_directories(OUTPUT_DIR)

    # Process each split
    for split_name, input_dir in INPUT_DIRS.items():
        process_split(
            input_dir=input_dir,
            output_dir=OUTPUT_DIR,
            split_name=split_name,
            target_size=TARGET_SIZE,
            num_workers=NUM_WORKERS,
        )


if __name__ == "__main__":
    main()
