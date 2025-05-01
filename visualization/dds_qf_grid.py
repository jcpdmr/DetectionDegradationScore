import os
import cv2
import matplotlib.pyplot as plt
import sys
import random
from pathlib import Path
from ultralytics import YOLO
from tqdm import tqdm

# Import for DDS calculation
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
try:
    from dds_metric import match_predictions
except ImportError:
    print("Warning: Cannot import match_predictions. DDS calculation may fail.")


def load_image(path):
    """Load image in RGB format from path."""
    img = cv2.imread(str(path))
    if img is None:
        raise ValueError(f"Failed to load image from {path}")
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return img


def detect_and_draw(model, img):
    """Run detection and return annotated image and results."""
    results = model.predict(img, verbose=False)
    annotated_img = img.copy()

    for box in results[0].boxes:
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        class_id = int(box.cls[0])
        class_name = results[0].names[class_id]
        confidence = float(box.conf[0])

        # Draw rectangle
        cv2.rectangle(annotated_img, (x1, y1), (x2, y2), (0, 255, 0), 2)

        # Add text
        label = f"{class_name} {confidence:.2f}"
        text_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)[0]
        text_x = x1
        text_y = y1 - 5 if y1 > text_size[1] + 5 else y1 + text_size[1] + 5
        cv2.putText(
            annotated_img,
            label,
            (text_x, text_y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 255, 0),
            2,
        )

    return annotated_img, results[0]


def calculate_dds(gt_results, comp_results):
    """Calculate Detection Degradation Score."""
    try:
        matches = match_predictions([gt_results], [comp_results])
        return float(matches[0]["error_score"])
    except Exception as e:
        print(f"Error calculating DDS: {str(e)}")
        return -1.0


def main():
    # Configuration
    BASE_PATH = Path("/andromeda/personal/jdamerini/unbalanced_dataset_coco2017/train")
    GT_DIR = BASE_PATH / "extracted"
    QUALITY_FACTORS = [20, 25, 30, 35, 40, 45, 50]
    COMPRESSED_DIRS = {qf: BASE_PATH / f"compressed{qf}" for qf in QUALITY_FACTORS}
    NUM_IMAGES_TO_FIND = 100  # Number of valid images to find
    NUM_IMAGES_TO_PROCESS = 100  # Number of images to actually process
    OUTPUT_DIR = Path("grid_visualization_output")
    RANDOM_SEED = 41

    # Create output directory
    OUTPUT_DIR.mkdir(exist_ok=True, parents=True)

    # Set random seed
    random.seed(RANDOM_SEED)

    # Initialize model
    print("Loading YOLO model...")
    model = YOLO("yolo11m.pt")

    # Find valid images
    print(f"Finding {NUM_IMAGES_TO_FIND} valid images...")
    valid_images = []

    for img_path in GT_DIR.glob("*.jpg"):
        img_name = img_path.name

        # Check if image exists in all compressed directories
        if all(
            COMPRESSED_DIRS[qf].joinpath(img_name).exists() for qf in QUALITY_FACTORS
        ):
            valid_images.append(img_name)

            # Exit loop when we reach the desired number of images
            if len(valid_images) >= NUM_IMAGES_TO_FIND:
                break

    # Sort valid images for reproducibility
    valid_images.sort()

    print(f"Found {len(valid_images)} valid images")

    # Select images
    if len(valid_images) > NUM_IMAGES_TO_PROCESS:
        selected_images = random.sample(valid_images, NUM_IMAGES_TO_PROCESS)
    else:
        selected_images = valid_images

    print(f"Selected {len(selected_images)} images for processing")

    # Process each selected image
    for img_name in tqdm(selected_images, desc="Processing images"):
        # Load GT image
        gt_path = GT_DIR / img_name
        try:
            gt_img = load_image(gt_path)
        except Exception as e:
            print(f"Error loading GT image {gt_path}: {str(e)}")
            continue

        # Run detection on GT
        gt_annotated, gt_results = detect_and_draw(model, gt_img)

        # Create figure with 2 rows and 4 columns
        fig, axes = plt.subplots(2, 4, figsize=(16, 8))

        # Process all QF images
        for i, qf in enumerate(QUALITY_FACTORS):
            comp_path = COMPRESSED_DIRS[qf] / img_name

            try:
                comp_img = load_image(comp_path)

                # Run detection
                comp_annotated, comp_results = detect_and_draw(model, comp_img)

                # Calculate DDS
                dds = calculate_dds(gt_results, comp_results)

                # Determine position in the grid
                if i < 4:  # First row: QF20, QF25, QF30, QF35
                    row, col = 0, i
                else:  # Second row: QF40, QF45, QF50
                    row, col = 1, i - 4

                # Plot image
                axes[row, col].imshow(comp_annotated)
                axes[row, col].set_title(f"QF{qf}")

                # Add only DDS score
                axes[row, col].text(
                    0.5,
                    -0.1,
                    f"DDS Score: {dds:.3f}",
                    transform=axes[row, col].transAxes,
                    ha="center",
                    fontsize=12,
                )

                axes[row, col].axis("off")

            except Exception as e:
                print(f"Error processing {comp_path}: {str(e)}")
                axes[row, col].text(
                    0.5,
                    0.5,
                    f"Error processing QF{qf}",
                    transform=axes[row, col].transAxes,
                    ha="center",
                    va="center",
                )
                axes[row, col].axis("off")

        # Add GT image in the last position (bottom right)
        axes[1, 3].imshow(gt_annotated)
        axes[1, 3].set_title("Ground Truth")
        axes[1, 3].text(
            0.5,
            -0.1,
            "DDS Score: 0.000",  # GT vs GT would be 0
            transform=axes[1, 3].transAxes,
            ha="center",
            fontsize=12,
        )
        axes[1, 3].axis("off")

        plt.subplots_adjust(hspace=0.3)

        # Adjust layout
        plt.tight_layout()

        # Save figure
        output_path = OUTPUT_DIR / f"grid_{img_name.split('.')[0]}.png"
        plt.savefig(output_path, dpi=300, bbox_inches="tight")
        plt.close(fig)

        print(f"Saved grid to {output_path}")


if __name__ == "__main__":
    main()
