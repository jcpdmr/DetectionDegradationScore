import cv2
import os
import shutil
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor
from tqdm import tqdm
import random  # Added for random sampling


def create_split_directories(base_path, splits=["train", "val", "test"]):
    """
    Create the complete directory structure for the dataset splits

    Args:
        base_path: Root directory for the dataset
        splits: List of dataset splits to create
    """
    modifications = ["extracted", "compressed", "distorted"]

    for split in splits:
        for mod in modifications:
            path = Path(base_path) / split / mod
            if path.exists():
                print(f"Cleaning {path}")
                shutil.rmtree(path)
            print(f"Created {path}")
            path.mkdir(parents=True, exist_ok=True)


def split_images(image_files, val_ratio=0.001, test_ratio=0.001, seed=42):
    """
    Split image files into train, validation and test sets

    Args:
        image_files: List of image paths
        val_ratio: Proportion of validation set
        test_ratio: Proportion of test set
        seed: Random seed for reproducibility

    Returns:
        Dictionary containing split image lists
    """
    random.seed(seed)
    random.shuffle(image_files)

    total = len(image_files)
    test_idx = int(total * (1 - test_ratio))
    val_idx = int(test_idx * (1 - val_ratio))

    return {
        "train": image_files[:val_idx],
        "val": image_files[val_idx:test_idx],
        "test": image_files[test_idx:],
    }


def get_center_crop_coordinates(height, width, crop_size):
    """
    Calculate coordinates for center cropping.

    Args:
        height (int): Image height
        width (int): Image width
        crop_size (int): Size of the square crop

    Returns:
        tuple: (y1, y2, x1, x2) coordinates for cropping
    """
    y1 = (height - crop_size) // 2
    y2 = y1 + crop_size
    x1 = (width - crop_size) // 2
    x2 = x1 + crop_size
    return y1, y2, x1, x2


def process_image(args):
    """
    Process a single image: load, crop, resize, and save.
    Designed to be used with multiprocessing.

    Args:
        args (tuple): (image_path, output_path, target_size, min_acceptable_size)
    """
    image_path, output_path, target_size, min_acceptable_size = args

    # Read image
    img = cv2.imread(str(image_path))
    if img is None:
        return False

    height, width = img.shape[:2]
    min_dim = min(height, width)

    # Skip if image is too small
    if min_dim < min_acceptable_size:
        return False

    # Determine crop size (use the smaller dimension)
    crop_size = min_dim

    # Get crop coordinates
    y1, y2, x1, x2 = get_center_crop_coordinates(height, width, crop_size)

    # Perform center crop
    cropped = img[y1:y2, x1:x2]

    # Resize if necessary (only downscaling)
    if crop_size > target_size:
        cropped = cv2.resize(
            cropped, (target_size, target_size), interpolation=cv2.INTER_AREA
        )

    # Save processed image
    cv2.imwrite(str(output_path), cropped)
    return True


def main():
    # Configuration
    INPUT_DIR = "../visual_genome"
    OUTPUT_DIR = "dataset_attention"
    TARGET_SIZE = 320  # Target size for the square patches
    MIN_ACCEPTABLE_SIZE = TARGET_SIZE  # Skip images smaller than this
    NUM_WORKERS = os.cpu_count()  # Use all available CPU cores
    MAX_IMAGES = 100000  # Maximum number of images to process

    # Create directory structure
    create_split_directories(OUTPUT_DIR)
    # Get and split image files
    input_path = Path(INPUT_DIR)
    image_files = (
        list(input_path.glob("*.jpg"))
        + list(input_path.glob("*.jpeg"))
        + list(input_path.glob("*.png"))
    )

    if MAX_IMAGES and MAX_IMAGES < len(image_files):
        random.seed(42)
        image_files = random.sample(image_files, MAX_IMAGES)

    # Split datasets
    splits = split_images(image_files)

    # Process each split
    for split_name, split_files in splits.items():
        print(f"\nProcessing {split_name} split...")

        # Prepare arguments for processing
        process_args = [
            (
                img_path,
                Path(OUTPUT_DIR) / split_name / "extracted" / img_path.name,
                TARGET_SIZE,
                MIN_ACCEPTABLE_SIZE,
            )
            for img_path in split_files
        ]

        # Process images in parallel with progress bar
        with ProcessPoolExecutor(max_workers=NUM_WORKERS) as executor:
            results = list(
                tqdm(
                    executor.map(process_image, process_args),
                    total=len(process_args),
                    desc=f"Extracting {split_name} images",
                )
            )

        # Print split statistics
        processed = sum(1 for r in results if r)
        skipped = len(results) - processed
        print(f"{split_name} split complete:")
        print(f"Successfully processed: {processed} images")
        print(f"Skipped (too small): {skipped} images")


if __name__ == "__main__":
    main()
