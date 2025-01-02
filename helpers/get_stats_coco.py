from pycocotools.coco import COCO
import numpy as np

# Initialize COCO dataset with the annotations file
annotations_file = "../annotations/instances_train2017.json"
coco = COCO(annotations_file)

# Lists to store the dimensions and aspect ratios of all images
dimensions = []
aspect_ratios = []
square_sizes = []  # We'll store all possible square sizes here

print("Starting image analysis...")

# Variables to track the smallest square size and its corresponding image
min_square_size = float("inf")  # Start with infinity
min_square_image_id = None

# Iterate through all images in the dataset
for img_id in coco.getImgIds():
    # Load metadata for current image
    img_info = coco.loadImgs(img_id)[0]
    width = img_info["width"]
    height = img_info["height"]

    # Calculate the maximum possible square size for this image
    square_size = min(width, height)

    # Update minimum square size if we found a smaller one
    if square_size < min_square_size:
        min_square_size = square_size
        min_square_image_id = img_id

    # Store dimensions and aspect ratio for statistical analysis
    dimensions.append((width, height))
    aspect_ratios.append(width / height)
    square_sizes.append(square_size)

# Convert lists to numpy arrays for easier statistical calculations
dimensions = np.array(dimensions)
aspect_ratios = np.array(aspect_ratios)
square_sizes = np.array(square_sizes)

# Print comprehensive statistics about the dataset
print("\nDataset Statistics:")
print(f"Total number of images: {len(dimensions)}")

print("\nDimension Statistics:")
print(f"Average width: {dimensions[:,0].mean():.2f}px")
print(f"Average height: {dimensions[:,1].mean():.2f}px")
print(f"Maximum possible square size (average): {np.mean(square_sizes):.2f}px")

print("\nSquare Size Analysis:")
print(f"Minimum square size in dataset: {min_square_size}px")
print("\nDetails of the smallest image:")
smallest_img = coco.loadImgs(min_square_image_id)[0]
print(f"Image ID: {min_square_image_id}")
print(f"File name: {smallest_img['file_name']}")
print(f"Original dimensions: {smallest_img['width']}x{smallest_img['height']}px")

print("\nSquare Size Percentiles:")
for p in [25, 50, 75, 90]:
    print(f"{p}th percentile: {np.percentile(square_sizes, p):.2f}px")

print("\nAspect Ratio Statistics:")
print(f"Mean aspect ratio: {aspect_ratios.mean():.2f}")
print(f"Median aspect ratio: {np.median(aspect_ratios):.2f}")
print(f"Standard deviation: {aspect_ratios.std():.2f}")

# Print recommendations based on the analysis
print("\nRecommendations for Square Cropping:")
print(f"Minimum possible size (no upscaling): {min_square_size}x{min_square_size}")
print(
    f"Conservative size (25th percentile): {np.percentile(square_sizes, 25):.0f}x{np.percentile(square_sizes, 25):.0f}"
)
print(
    f"Balanced size (median): {np.median(square_sizes):.0f}x{np.median(square_sizes):.0f}"
)

# Count how many images would need upscaling for each common target size
common_sizes = [416, 512, 640, 768]
print("\nUpscaling Analysis for Common Sizes:")
for target_size in common_sizes:
    needs_upscale = np.sum(square_sizes < target_size)
    percentage = (needs_upscale / len(square_sizes)) * 100
    print(
        f"{target_size}x{target_size}: {percentage:.1f}% of images would need upscaling"
    )
