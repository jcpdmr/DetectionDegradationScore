import cv2
import os
import shutil
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor
from tqdm import tqdm
import random  # Added for random sampling


def create_clean_directory(path):
    """
    Create a directory if it doesn't exist, or clean it if it does.

    Args:
        path (str): Directory path to create or clean
    """
    directory = Path(path)
    if directory.exists():
        shutil.rmtree(directory)
    directory.mkdir(parents=True)


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
    INPUT_DIR = "../train2017"
    OUTPUT_DIR = "patches/extracted"
    TARGET_SIZE = 384  # Target size for the square patches
    MIN_ACCEPTABLE_SIZE = TARGET_SIZE  # Skip images smaller than this
    NUM_WORKERS = os.cpu_count()  # Use all available CPU cores
    MAX_IMAGES = 16000  # Maximum number of images to process

    # Create or clean output directory
    create_clean_directory(OUTPUT_DIR)

    # Get list of all images
    input_path = Path(INPUT_DIR)
    image_files = (
        list(input_path.glob("*.jpg"))
        + list(input_path.glob("*.jpeg"))
        + list(input_path.glob("*.png"))
    )

    # If we want to limit the number of images, randomly sample from the list
    if MAX_IMAGES and MAX_IMAGES < len(image_files):
        # Set random seed for reproducibility
        random.seed(42)
        image_files = random.sample(image_files, MAX_IMAGES)

    # Prepare arguments for processing
    process_args = [
        (img_path, Path(OUTPUT_DIR) / img_path.name, TARGET_SIZE, MIN_ACCEPTABLE_SIZE)
        for img_path in image_files
    ]

    # Process images in parallel with progress bar
    with ProcessPoolExecutor(max_workers=NUM_WORKERS) as executor:
        results = list(
            tqdm(
                executor.map(process_image, process_args),
                total=len(process_args),
                desc="Processing images",
            )
        )

    # Print statistics
    processed = sum(1 for r in results if r)
    skipped = len(results) - processed
    print("\nProcessing complete:")
    print(f"Successfully processed: {processed} images")
    print(f"Skipped (too small): {skipped} images")


if __name__ == "__main__":
    main()
