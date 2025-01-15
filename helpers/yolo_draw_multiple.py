import os
import shutil
from ultralytics import YOLO
import cv2
import json
import sys
from pathlib import Path
from torchvision import transforms

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from score_metrics import match_predictions

QUALITY_VALUES = [10, 20, 30, 40, 50]


def setup_output_directories(base_dir="draw"):
    """
    Create output directories for all compression levels.
    """
    base_dir = Path(base_dir)

    # Create list of all required directories
    dirs_to_setup = [base_dir / "extracted"]
    dirs_to_setup.extend(
        [base_dir / f"compressed{quality}" for quality in QUALITY_VALUES]
    )

    # Setup each directory
    for dir_path in dirs_to_setup:
        if not dir_path.exists():
            dir_path.mkdir(parents=True)
        else:
            for file in os.listdir(dir_path):
                file_path = dir_path / file
                if file_path.is_file():
                    file_path.unlink()
                elif file_path.is_dir():
                    shutil.rmtree(file_path)


def draw_predictions(image_tensor, results, save_path):
    """
    Draw bounding boxes on image and save it.
    Now accepts tensor input and converts back for visualization.
    """
    # Convert tensor back to image format for visualization
    image_np = (image_tensor.permute(1, 2, 0).numpy() * 255).astype("uint8")
    annotated_img = image_np.copy()

    boxes = results[0].boxes
    for box in boxes:
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        class_id = int(box.cls[0])
        class_name = results[0].names[class_id]
        confidence = float(box.conf[0])

        cv2.rectangle(annotated_img, (x1, y1), (x2, y2), (0, 255, 0), 2)

        label = f"{class_name} {confidence:.2f}"
        (text_width, text_height), _ = cv2.getTextSize(
            label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2
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
            2,
        )

    # Save in BGR format for OpenCV
    cv2.imwrite(str(save_path), cv2.cvtColor(annotated_img, cv2.COLOR_RGB2BGR))
    return results


def process_image_sets(model, image_list, base_dirs, output_dirs):
    """
    Process sets of images with consistent image loading.
    """
    metrics = {f"quality_{q}": [] for q in QUALITY_VALUES}

    for img_name in image_list:
        # Load and process GT image
        gt_path = base_dirs["extracted"] / img_name
        gt_tensor = load_and_transform_image(gt_path)  # Now returns (1,C,H,W)
        gt_results = model(gt_tensor)
        draw_predictions(
            gt_tensor.squeeze(0),  # Remove batch dim for visualization
            gt_results,
            output_dirs["extracted"] / img_name,
        )

        # Process each compression quality
        for quality in QUALITY_VALUES:
            comp_path = base_dirs[f"compressed{quality}"] / img_name
            if comp_path.exists():
                comp_tensor = load_and_transform_image(
                    comp_path
                )  # Now returns (1,C,H,W)
                comp_results = model(comp_tensor)
                draw_predictions(
                    comp_tensor.squeeze(0),  # Remove batch dim for visualization
                    comp_results,
                    output_dirs[f"compressed{quality}"] / img_name,
                )

                # Calculate quality score
                comp_matches = match_predictions([gt_results[0]], [comp_results[0]])
                metrics[f"quality_{quality}"].append(
                    {
                        "image": img_name,
                        "error_score": float(comp_matches[0]["error_score"]),
                        "num_matches": len(comp_matches[0]["matches"]),
                        "num_gt": int(comp_matches[0]["num_gt"]),
                        "num_pred": int(comp_matches[0]["num_mod"]),
                        "matches": [
                            {
                                "mod_idx": int(match["mod_idx"]),
                                "gt_idx": int(match["gt_idx"]),
                                "iou": float(match["iou"]),
                                "class": int(match["class"]),
                                "gt_score": float(match["gt_score"]),
                                "mod_score": float(match["mod_score"]),
                            }
                            for match in comp_matches[0]["matches"]
                        ],
                    }
                )

    return metrics


def load_and_transform_image(image_path):
    """
    Load and transform image consistently with MultiCompressionErrorScoreCalculator
    and add batch dimension for YOLO
    """
    # Read image and convert to RGB
    img = cv2.imread(str(image_path))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # Apply transform
    transform = transforms.ToTensor()
    tensor = transform(img)

    # Add batch dimension
    tensor = tensor.unsqueeze(0)  # Add batch dimension: (C,H,W) -> (1,C,H,W)

    return tensor


def analyze_metrics(metrics):
    """
    Analyze and print summary statistics for each compression quality.
    """
    for quality in QUALITY_VALUES:
        category = f"quality_{quality}"
        if metrics[category]:
            scores = [float(m["error_score"]) for m in metrics[category]]
            print(f"\nQuality {quality} analysis:")
            print(f"Average score: {sum(scores) / len(scores):.3f}")
            print(f"Best score: {min(scores):.3f}")
            print(f"Worst score: {max(scores):.3f}")

            # Print worst cases
            worst_cases = sorted(
                metrics[category], key=lambda x: float(x["error_score"]), reverse=True
            )[:3]
            print(f"\nWorst performing images for quality {quality}:")
            for case in worst_cases:
                print(f"Image: {case['image']}")
                print(f"Score: {float(case['error_score']):.3f}")
                print(
                    f"Matches: {case['num_matches']} "
                    f"(GT: {case['num_gt']}, Pred: {case['num_pred']})"
                )


def main():
    # Define image list to process
    image_list = [
        "2335669.jpg",
        "2417841.jpg",
        "2337436.jpg",
        "2378415.jpg",
    ]

    # Define directories using Path
    base_dirs = {
        "extracted": Path("unbalanced_dataset/train/extracted"),
        **{
            f"compressed{q}": Path(f"unbalanced_dataset/train/compressed{q}")
            for q in QUALITY_VALUES
        },
    }

    output_dirs = {
        "extracted": Path("draw/extracted"),
        **{f"compressed{q}": Path(f"draw/compressed{q}") for q in QUALITY_VALUES},
    }

    # Setup output directories
    setup_output_directories()

    # Load YOLO model
    model = YOLO("yolo11m.pt")
    model.eval()

    # Process images and get metrics
    metrics = process_image_sets(model, image_list, base_dirs, output_dirs)

    # Analyze and print results
    analyze_metrics(metrics)

    # Save metrics to file
    with open("draw/quality_metrics.json", "w") as f:
        json.dump(metrics, f, indent=4)


if __name__ == "__main__":
    main()
