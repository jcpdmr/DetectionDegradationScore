import torch
from typing import List, Dict
from torchvision import transforms
from PIL import Image
import torch
from ultralytics import YOLO


def format_yolo_predictions(yolo_predictions, is_ground_truth=False):
    """
    Convert YOLO predictions to the format expected by torchmetrics

    Args:
        yolo_predictions: List of predictions from YOLO model
        is_ground_truth: Boolean flag indicating if these are ground truth predictions
                        When True, sets all confidence scores to 1.0
                        When False, uses the model's confidence scores

    Returns:
        List of dictionaries, each containing:
        - 'boxes': tensor of bounding boxes in [x1, y1, x2, y2] format
        - 'labels': tensor of class labels
        - 'scores': tensor of confidence scores (all 1.0 for ground truth)
    """
    formatted_predictions = []
    for pred in yolo_predictions:
        # Extract boxes and labels which are needed for both GT and predictions
        boxes = pred.boxes.xyxy  # Gets boxes in [x1, y1, x2, y2] format
        labels = (
            pred.boxes.cls.long()
        )  # Gets class labels and convert from floats to long tensors

        # Handle confidence scores differently based on whether this is GT or not
        if is_ground_truth:
            # For ground truth, create tensor of 1.0s matching the number of boxes
            scores = torch.ones_like(labels)
        else:
            # For model predictions, use the actual confidence scores
            scores = pred.boxes.conf

        formatted_predictions.append(
            {"boxes": boxes, "labels": labels, "scores": scores}
        )
    return formatted_predictions


def match_predictions(
    gt_predictions: List[Dict], mod_predictions: List[Dict], iou_threshold: float = 0.5
) -> List[Dict]:
    """
    Match predictions following torchmetrics approach:
    - First group by class
    - Then match within each class using IoU
    """
    batch_matches = []

    # Process each image in the batch
    for gt_pred, mod_pred in zip(gt_predictions, mod_predictions):
        gt_boxes = gt_pred.boxes.xyxy.clone().detach()
        gt_scores = gt_pred.boxes.conf.clone().detach()
        gt_classes = gt_pred.boxes.cls.clone().detach()

        mod_boxes = mod_pred.boxes.xyxy.clone().detach()
        mod_scores = mod_pred.boxes.conf.clone().detach()
        mod_classes = mod_pred.boxes.cls.clone().detach()

        # Get unique classes from both predictions
        unique_classes = torch.unique(torch.cat([gt_classes, mod_classes]))

        matches = []
        matched_gt_indices = set()

        # Process one class at a time
        for class_id in unique_classes:
            # Filter boxes for current class
            gt_mask = gt_classes == class_id
            mod_mask = mod_classes == class_id

            class_gt_boxes = gt_boxes[gt_mask]
            class_gt_scores = gt_scores[gt_mask]
            class_gt_indices = torch.where(gt_mask)[0]

            class_mod_boxes = mod_boxes[mod_mask]
            class_mod_scores = mod_scores[mod_mask]
            class_mod_indices = torch.where(mod_mask)[0]

            # Sort modified predictions by confidence (within this class)
            mod_sort_indices = torch.argsort(class_mod_scores, descending=True)
            class_mod_boxes = class_mod_boxes[mod_sort_indices]
            class_mod_scores = class_mod_scores[mod_sort_indices]
            class_mod_indices = class_mod_indices[mod_sort_indices]

            # Match within this class
            for mod_idx in range(len(class_mod_boxes)):
                best_iou = iou_threshold
                best_gt_idx = -1

                # Find best matching ground truth box (of same class)
                for gt_idx in range(len(class_gt_boxes)):
                    if class_gt_indices[gt_idx].item() in matched_gt_indices:
                        continue

                    iou = box_iou(
                        class_mod_boxes[mod_idx : mod_idx + 1],
                        class_gt_boxes[gt_idx : gt_idx + 1],
                    )

                    if iou > best_iou:
                        best_iou = iou
                        best_gt_idx = gt_idx

                # If we found a match
                if best_gt_idx >= 0:
                    matches.append(
                        {
                            "mod_idx": class_mod_indices[mod_idx].item(),
                            "gt_idx": class_gt_indices[best_gt_idx].item(),
                            "iou": best_iou,
                            "class": class_id.item(),
                            "gt_score": class_gt_scores[best_gt_idx].item(),
                            "mod_score": class_mod_scores[mod_idx].item(),
                        }
                    )
                    matched_gt_indices.add(class_gt_indices[best_gt_idx].item())

        # Calculate metrics for this image
        image_metrics = {
            "matches": matches,
            "num_gt": len(gt_boxes),
            "num_mod": len(mod_boxes),
            "ddscore": calculate_dds(matches, len(gt_boxes), len(mod_boxes)),
        }

        batch_matches.append(image_metrics)

    return batch_matches


def box_iou(boxes1: torch.Tensor, boxes2: torch.Tensor) -> torch.Tensor:
    """
    Calculate IoU between two boxes in (x1, y1, x2, y2) format
    """
    # Calculate intersection areas
    x1 = torch.max(boxes1[:, 0], boxes2[:, 0])
    y1 = torch.max(boxes1[:, 1], boxes2[:, 1])
    x2 = torch.min(boxes1[:, 2], boxes2[:, 2])
    y2 = torch.min(boxes1[:, 3], boxes2[:, 3])

    intersection = torch.clamp(x2 - x1, min=0) * torch.clamp(y2 - y1, min=0)

    # Calculate union areas
    area1 = (boxes1[:, 2] - boxes1[:, 0]) * (boxes1[:, 3] - boxes1[:, 1])
    area2 = (boxes2[:, 2] - boxes2[:, 0]) * (boxes2[:, 3] - boxes2[:, 1])
    union = area1 + area2 - intersection

    return intersection / union


def calculate_dds(matches: List[Dict], num_gt: int, num_mod: int) -> float:
    """
    Calculate Detection Degradation Score based on matches and predictions.
    Since matches are already filtered by class (following torchmetrics approach),
    we only need to consider IoU and confidence ratios.

    Args:
        matches: List of dictionaries containing match information
        num_gt: Total number of ground truth predictions
        num_mod: Total number of modified image predictions

    Returns:
        DDS between 0 (perfect match) and 1 (complete mismatch)
    """
    # Handle edge cases
    if num_gt == 0 and num_mod == 0:
        return 0.0  # Perfect match - no objects in either image

    if len(matches) == 0:
        return 1.0  # Worst case - no valid matches found

    # Calculate quality for each match
    match_qualities = []
    for match in matches:
        # IoU quality remains unchanged
        iou_quality = match["iou"]

        # Calculate confidence quality using the ratio
        # Note: we know classes match because of the new matching system
        conf_quality = min(match["mod_score"] / match["gt_score"], 1.0)

        # Combine IoU and confidence qualities
        match_quality = iou_quality * conf_quality
        match_qualities.append(match_quality)

    # Calculate final detection quality score
    # Normalize by max number of predictions to penalize both
    # missing detections and false positives
    detection_quality_score = sum(match_qualities) / max(num_gt, num_mod)

    # Convert to dd score where:
    # 0 = perfect match (all objects detected with perfect IoU and confidence)
    # 1 = complete mismatch (no valid matches found)
    return 1.0 - detection_quality_score

def DDS(ref_img_path, degraded_img_path, device='cuda:0'):
    """
    Compute the Detection Degradation Score (DDS) between a reference and a degraded image.

    The DDS measures the quality degradation in object detection performance between two images.
    A score of 0.0 indicates perfect similarity (identical detections), while a score of 1.0 
    indicates complete mismatch (no matching detections).

    Args:
        ref_img_path (str): File path to the reference image.
        degraded_img_path (str): File path to the degraded (e.g., compressed, blurred) image.
        device (str): Device to run inference on (e.g., 'cuda:0' or 'cpu').

    Returns:
        float: Detection Degradation Score (DDS) between 0.0 (best) and 1.0 (worst).
    """

    detector = YOLO("yolo11m.pt").to('cuda:0')
    # Load and preprocess images
    transform = transforms.Compose([transforms.ToTensor()])
    img_ref = transform(Image.open(ref_img_path).convert("RGB")).unsqueeze(0).to(device)
    img_deg = transform(Image.open(degraded_img_path).convert("RGB")).unsqueeze(0).to(device)

    # Run inference
    with torch.no_grad():
        ref_preds = detector.predict(img_ref, verbose=False)
        deg_preds = detector.predict(img_deg, verbose=False)

    # Extract single image predictions
    ref_result = ref_preds[0]
    deg_result = deg_preds[0]

    # Compute DDS
    batch_results = match_predictions([ref_result], [deg_result])
    dds_score = batch_results[0]['ddscore']

    dds_score = dds_score.item() if isinstance(dds_score, torch.Tensor) else dds_score

    return dds_score
