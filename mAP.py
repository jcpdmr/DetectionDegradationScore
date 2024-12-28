from torchmetrics.detection.mean_ap import MeanAveragePrecision
import torch

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
        boxes = pred.boxes.xyxy    # Gets boxes in [x1, y1, x2, y2] format
        labels = pred.boxes.cls.long()    # Gets class labels and convert from floats to long tensors
        
        # Handle confidence scores differently based on whether this is GT or not
        if is_ground_truth:
            # For ground truth, create tensor of 1.0s matching the number of boxes
            scores = torch.ones_like(labels)
        else:
            # For model predictions, use the actual confidence scores
            scores = pred.boxes.conf
        
        formatted_predictions.append({
            'boxes': boxes,
            'labels': labels,
            'scores': scores
        })
    return formatted_predictions

def calculate_batch_mAP(gt_predictions, mod_predictions):
    """
    Calculate mean Average Precision between ground truth and modified predictions
    
    Args:
        gt_predictions: Ground truth predictions from YOLO
        mod_predictions: Predictions on modified images
        
    Returns:
        Scalar tensor containing mAP score
    
    Note:
        Ground truth predictions are treated differently by setting their
        confidence scores to 1.0, as they represent the "true" detections.
        Modified predictions keep their original confidence scores as they
        represent the model's uncertainty in its predictions.
    """
    metric = MeanAveragePrecision()
    
    # Format predictions, specifying which ones are ground truth
    gt_formatted = format_yolo_predictions(gt_predictions, is_ground_truth=True)
    mod_formatted = format_yolo_predictions(mod_predictions, is_ground_truth=False)
    
    # Update metric with formatted predictions
    metric.update(mod_formatted, gt_formatted)
    
    # Compute and return mAP
    results = metric.compute()
    return results['map']