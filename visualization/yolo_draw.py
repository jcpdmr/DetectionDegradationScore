import os
import shutil
from ultralytics import YOLO
import cv2
import json
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from dds_metric import match_predictions


def get_image_names(attempt: str, type_files: list, n_images: int = 20) -> dict:
    """
    Read first N image names from each type file.

    Returns:
        Dictionary with type names as keys and lists of image names as values
    """
    image_names = {}
    base_path = f"ddscores_analysis/mapping/{attempt}/total"

    for type_file in type_files:
        type_name = type_file.split("_")[0]
        file_path = os.path.join(base_path, type_file)
        try:
            with open(file_path, "r") as f:
                names = [line.strip() for line in f.readlines()[:n_images]]
                image_names[type_name] = names
        except FileNotFoundError:
            print(f"Warning: File {file_path} not found")

    return image_names


def setup_draw_directory(type_files):
    """
    Clean and recreate the draw directory with subdirectories for each type.
    """
    draw_dir = "draw"
    if os.path.exists(draw_dir):
        shutil.rmtree(draw_dir)

    # Create main draw directory
    os.makedirs(draw_dir)

    # Create subdirectory for each type
    for type_file in type_files:
        # Get type name without extension (e.g., 'robust' from 'robust_images.txt')
        type_name = type_file.split("_")[0]
        os.makedirs(os.path.join(draw_dir, type_name))


def draw_predictions(image, results, save_path):
    """
    Draw bounding boxes, class labels and scores on the image and save it.
    """
    annotated_img = image.copy()

    boxes = results[0].boxes
    for box in boxes:
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        class_id = int(box.cls[0])
        class_name = results[0].names[class_id]
        confidence = float(box.conf[0])

        cv2.rectangle(annotated_img, (x1, y1), (x2, y2), (0, 255, 0), 1)

        label = f"{class_name} {confidence:.2f}"
        (text_width, text_height), _ = cv2.getTextSize(
            label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1
        )
        text_x = x1 + (x2 - x1 - text_width) // 2
        text_y = y1 + (y2 - y1 + text_height) // 2
        cv2.putText(
            annotated_img,
            label,
            (text_x, text_y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 255, 0),
            1,
        )

    cv2.imwrite(save_path, annotated_img)
    return results


def process_images(model, image_names_dict, base_path, compression_values):
    """
    Process images, draw predictions and calculate dd scores for each compression level.
    """
    metrics = {
        type_name: {f"compressed{qf}": [] for qf in compression_values}
        for type_name in image_names_dict.keys()
    }

    for type_name, image_names in image_names_dict.items():
        for img_name in image_names:
            # Process GT (extracted) image
            gt_path = os.path.join(base_path, "extracted", img_name)
            if not os.path.exists(gt_path):
                print(f"Warning: GT image not found: {gt_path}")
                continue

            gt_image = cv2.imread(gt_path)
            gt_results = model(gt_image, imgsz=320, verbose=False)
            save_name = f"{os.path.splitext(img_name)[0]}_gt.jpg"
            gt_results = draw_predictions(
                gt_image, gt_results, os.path.join("draw", type_name, save_name)
            )

            # Process compressed images
            for qf in compression_values:
                comp_path = os.path.join(base_path, f"compressed{qf}", img_name)
                if not os.path.exists(comp_path):
                    print(f"Warning: Compressed image not found: {comp_path}")
                    continue

                comp_image = cv2.imread(comp_path)
                comp_results = model(comp_image, imgsz=320)
                save_name = f"{os.path.splitext(img_name)[0]}_comp{qf}.jpg"
                comp_results = draw_predictions(
                    comp_image, comp_results, os.path.join("draw", type_name, save_name)
                )

                # Calculate quality score
                matches = match_predictions([gt_results[0]], [comp_results[0]])
                metrics[type_name][f"compressed{qf}"].append(
                    {
                        "image": img_name,
                        "ddscore": float(matches[0]["ddscore"]),
                        "num_matches": len(matches[0]["matches"]),
                        "num_gt": int(matches[0]["num_gt"]),
                        "num_pred": int(matches[0]["num_mod"]),
                        "matches": [
                            {
                                "mod_idx": int(match["mod_idx"]),
                                "gt_idx": int(match["gt_idx"]),
                                "iou": float(match["iou"]),
                                "class": int(match["class"]),
                                "gt_score": float(match["gt_score"]),
                                "mod_score": float(match["mod_score"]),
                            }
                            for match in matches[0]["matches"]
                        ],
                    }
                )

    return metrics


def analyze_metrics(metrics):
    """
    Analyze and print summary statistics for the quality scores.
    """
    for type_name, type_metrics in metrics.items():
        print(f"\n=== {type_name.upper()} Analysis ===")
        for category, data in type_metrics.items():
            if data:
                scores = [float(m["ddscore"]) for m in data]
                print(f"\n{category} analysis:")
                print(f"Average score: {sum(scores) / len(scores):.3f}")
                print(f"Best score: {min(scores):.3f}")
                print(f"Worst score: {max(scores):.3f}")

                # Print worst cases
                worst_cases = sorted(
                    data, key=lambda x: float(x["ddscore"]), reverse=True
                )[:3]
                print("\nImages with worst performance:")
                for case in worst_cases:
                    print(f"Image: {case['image']}")
                    print(f"Score: {float(case['ddscore']):.3f}")
                    print(
                        f"Match: {case['num_matches']} (GT: {case['num_gt']}, Pred: {case['num_pred']})"
                    )


def reorganize_metrics_by_image(metrics, type_name):
    """
    Reorganize metrics from QF-based to image-based organization
    """
    image_metrics = {}

    # Get all images from all compression levels
    all_images = set()
    for comp_data in metrics[type_name].values():
        for entry in comp_data:
            all_images.add(entry["image"])

    # Reorganize data by image
    for image_name in all_images:
        image_metrics[image_name] = {}
        for qf, comp_data in metrics[type_name].items():
            # Find data for this image at this compression level
            for entry in comp_data:
                if entry["image"] == image_name:
                    image_metrics[image_name][qf] = entry
                    break

    return image_metrics


def create_comparison_grid(image_metrics, type_name, base_path, compression_values):
    """
    Create a 2-row grid of images with borders between them.
    """
    import numpy as np

    draw_dir = os.path.join("draw", type_name)
    grid_dir = os.path.join(draw_dir, "comparisons")
    os.makedirs(grid_dir, exist_ok=True)

    # Sort compression values and add "GT" at the end
    all_qualities = [f"compressed{qf}" for qf in compression_values] + ["GT"]

    # Calculate grid layout
    images_per_row = (len(all_qualities) + 1) // 2  # Divide images across 2 rows
    border_size = 10  # Size of the border between images

    for img_name, results in image_metrics.items():
        # Calculate dimensions
        sample_img = cv2.imread(os.path.join(base_path, "extracted", img_name))
        img_height, img_width = sample_img.shape[:2]

        text_height = 30  # Space for dd score text
        total_height = (
            img_height + text_height
        ) * 2 + border_size  # 2 rows + middle border
        total_width = (
            img_width * images_per_row + (images_per_row - 1) * border_size
        )  # Add borders between columns

        # Create white canvas for the grid
        grid = np.ones((total_height, total_width, 3), dtype=np.uint8) * 255

        # For each quality level
        for idx, quality in enumerate(all_qualities):
            # Calculate position in grid
            row = idx // images_per_row
            col = idx % images_per_row

            # Calculate offsets including borders
            y_offset = row * (img_height + text_height + border_size)
            x_offset = col * (img_width + border_size)

            # Read and place image
            if quality == "GT":
                img_path = os.path.join(
                    draw_dir, f"{os.path.splitext(img_name)[0]}_gt.jpg"
                )
                ddscore = 0.0
            else:
                qf = quality.replace("compressed", "")
                img_path = os.path.join(
                    draw_dir, f"{os.path.splitext(img_name)[0]}_comp{qf}.jpg"
                )
                ddscore = results[quality]["ddscore"]

            if os.path.exists(img_path):
                img = cv2.imread(img_path)
                grid[
                    y_offset : y_offset + img_height, x_offset : x_offset + img_width
                ] = img

                # Add dd score text
                text = f"QF: {quality.replace('compressed', '')} Score: {ddscore:.3f}"
                cv2.putText(
                    grid,
                    text,
                    (x_offset + 10, y_offset + img_height + 20),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (0, 0, 0),
                    1,
                )

        # Save the comparison grid
        cv2.imwrite(os.path.join(grid_dir, f"{img_name}_comparison.jpg"), grid)


def main():
    # Configuration
    ATTEMPT = "06_visgen_coco17tr_openimagev7traine_320p_qual_20_24_28_32_36_40_50_smooth_2_subsam_444"
    TYPE = ["random_pick.txt"]
    BASE_PATH = "/andromeda/personal/jdamerini/unbalanced_dataset/train"
    COMPRESSION_VALUES = [20, 24, 28, 32, 36, 40, 50]
    N_IMAGES = 100

    # Setup draw directory with type subdirectories
    setup_draw_directory(TYPE)

    # Get image names for each type
    image_names_dict = get_image_names(ATTEMPT, TYPE, N_IMAGES)
    print(f"Processing images for types: {list(image_names_dict.keys())}")

    # Load YOLO model
    model = YOLO("yolo11m.pt")

    # Process images and get metrics
    metrics = process_images(model, image_names_dict, BASE_PATH, COMPRESSION_VALUES)

    # Reorganize metrics by image
    type_name = TYPE[0].split("_")[0]  # Get type name without extension
    image_metrics = reorganize_metrics_by_image(metrics, type_name)

    # Create comparison grids
    create_comparison_grid(image_metrics, type_name, BASE_PATH, COMPRESSION_VALUES)

    # Save reorganized metrics to file
    with open("draw/quality_metrics.json", "w") as f:
        json.dump(image_metrics, f, indent=4)


if __name__ == "__main__":
    main()
