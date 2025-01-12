import os
import shutil
from ultralytics import YOLO
import cv2
import json
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from score_metrics import match_predictions

PATCH_SIZE = 384


def setup_output_directories(base_dirs):
    """
    Create output directories if they don't exist and clean them if they do.
    Args:
        base_dirs: List of directory paths to setup
    """
    for dir_path in base_dirs:
        if not os.path.exists(dir_path):
            os.makedirs(dir_path)
        else:
            for file in os.listdir(dir_path):
                file_path = os.path.join(dir_path, file)
                if os.path.isfile(file_path):
                    os.unlink(file_path)
                elif os.path.isdir(file_path):
                    shutil.rmtree(file_path)


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

    cv2.imwrite(save_path, annotated_img)
    return results


def process_image_pairs(model, image_list, base_dirs, output_dirs):
    """
    Process pairs of images (GT vs compressed/distorted) and calculate quality scores.
    Returns dictionary with quality scores and metrics for analysis.
    """
    metrics = {"compressed": [], "distorted": []}

    for img_name in image_list:
        # Process GT (extracted) image
        gt_path = os.path.join(base_dirs["extracted"], img_name)
        gt_image = cv2.imread(gt_path)
        gt_results = model(gt_image)
        draw_predictions(
            gt_image, gt_results, os.path.join(output_dirs["extracted"], img_name)
        )

        # Process and compare compressed image
        comp_path = os.path.join(base_dirs["compressed"], img_name)
        if os.path.exists(comp_path):
            comp_image = cv2.imread(comp_path)
            comp_results = model(comp_image)
            draw_predictions(
                comp_image,
                comp_results,
                os.path.join(output_dirs["compressed"], img_name),
            )

            # Calculate quality score for compressed
            comp_matches = match_predictions([gt_results[0]], [comp_results[0]])
            metrics["compressed"].append(
                {
                    "image": img_name,
                    "error_score": float(
                        comp_matches[0]["error_score"]
                    ),  # Convert tensor to float
                    "num_matches": len(comp_matches[0]["matches"]),
                    "num_gt": int(comp_matches[0]["num_gt"]),  # Convert to int
                    "num_pred": int(comp_matches[0]["num_mod"]),  # Convert to int
                    "matches": [  # Convert match details to standard Python types
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

        # # Process and compare distorted image
        # dist_path = os.path.join(base_dirs["distorted"], img_name)
        # if os.path.exists(dist_path):
        #     dist_image = cv2.imread(dist_path)
        #     dist_results = model(dist_image)
        #     draw_predictions(
        #         dist_image,
        #         dist_results,
        #         os.path.join(output_dirs["distorted"], img_name),
        #     )

        #     # Calculate quality score for distorted
        #     dist_matches = match_predictions([gt_results[0]], [dist_results[0]])
        #     metrics["distorted"].append(
        #         {
        #             "image": img_name,
        #             "error_score": float(
        #                 dist_matches[0]["error_score"]
        #             ),  # Convert tensor to float
        #             "num_matches": len(dist_matches[0]["matches"]),
        #             "num_gt": int(dist_matches[0]["num_gt"]),  # Convert to int
        #             "num_pred": int(dist_matches[0]["num_mod"]),  # Convert to int
        #             "matches": [  # Convert match details to standard Python types
        #                 {
        #                     "mod_idx": int(match["mod_idx"]),
        #                     "gt_idx": int(match["gt_idx"]),
        #                     "iou": float(match["iou"]),
        #                     "class": int(match["class"]),
        #                     "gt_score": float(match["gt_score"]),
        #                     "mod_score": float(match["mod_score"]),
        #                 }
        #                 for match in dist_matches[0]["matches"]
        #             ],
        #         }
        #     )

    return metrics


def analyze_metrics(metrics):
    """
    Analyze and print summary statistics for the quality scores.
    """
    for category in ["compressed"]:
        if metrics[category]:
            # Convert to float
            scores = [float(m["error_score"]) for m in metrics[category]]
            print(f"\n{category} images analysis:")
            print(f"Average score: {sum(scores) / len(scores):.3f}")
            print(f"Best score: {min(scores):.3f}")
            print(f"Worst score: {max(scores):.3f}")

            # Print worst cases
            worst_cases = sorted(
                metrics[category], key=lambda x: float(x["error_score"]), reverse=True
            )[:3]
            print("\nImages with worst performance:")
            for case in worst_cases:
                print(f"Image: {case['image']}")
                print(f"Score: {float(case['error_score']):.3f}")
                print(
                    f"Match: {case['num_matches']} (GT: {case['num_gt']}, Pred: {case['num_pred']})"
                )


def main():
    # Define image list to process
    image_list = [
        "000000000025.jpg",
        "000000000078.jpg",
        "000000000321.jpg",
        "000000000400.jpg",
        "000000000419.jpg",
        "000000447588.jpg",
        # Add more images as needed
    ]

    # Define directories
    base_dirs = {
        "extracted": "dataset_attention/train/extracted",
        "compressed": "dataset_attention/train//compressed",
        # "distorted": "patches/distorted",
    }

    output_dirs = {
        "extracted": "draw/extracted",
        "compressed": "draw/compressed",
        # "distorted": "draw/distorted",
    }

    # Setup output directories
    setup_output_directories(output_dirs.values())

    # Load YOLO model
    model = YOLO("yolo11m.pt")

    # Process images and get metrics
    metrics = process_image_pairs(model, image_list, base_dirs, output_dirs)

    # Analyze and print results
    analyze_metrics(metrics)

    # Save metrics to file for later analysis
    with open("draw/quality_metrics.json", "w") as f:
        json.dump(metrics, f, indent=4)


if __name__ == "__main__":
    main()
