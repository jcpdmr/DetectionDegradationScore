import numpy as np
def calculate_batch_map(gt_predictions, distorted_predictions, iou_threshold=0.5):
    """
    Calculate mean Average Precision between two batches of YOLO predictions
    
    Args:
        gt_predictions: List of ground truth predictions for each image in batch
                       Each prediction contains bounding boxes and class labels
        distorted_predictions: List of predictions on distorted images
                             Each prediction contains bounding boxes, class labels and confidence scores
        iou_threshold: IoU threshold to consider a detection as correct (default 0.5)
    
    Returns:
        mAP score between 0 and 1
    """

def process_by_class(gt_predictions, distorted_predictions):
    """
    Organize predictions by class across the batch
    
    Returns dictionary with:
    - All ground truth boxes for each class
    - All predicted boxes with confidence scores for each class
    - Count of ground truth objects per class
    """
    classes_data = {}
    
    # Process ground truth predictions
    for img_idx, gt_pred in enumerate(gt_predictions):
        for box, class_id in gt_pred:
            if class_id not in classes_data:
                classes_data[class_id] = {
                    'gt_boxes': [], 
                    'pred_boxes': [],
                    'gt_count': 0
                }
            classes_data[class_id]['gt_boxes'].append((img_idx, box))
            classes_data[class_id]['gt_count'] += 1
            
    # Process predictions on distorted images
    for img_idx, dist_pred in enumerate(distorted_predictions):
        for box, class_id, confidence in dist_pred:
            if class_id in classes_data:  # Only consider classes that appear in GT
                classes_data[class_id]['pred_boxes'].append(
                    (img_idx, box, confidence)
                )
    
    return classes_data

def calculate_ap(gt_boxes, pred_boxes, n_gt, iou_threshold):
    """
    Calculate Average Precision for a single class
    
    Args:
        gt_boxes: List of (image_idx, box) for ground truth
        pred_boxes: List of (image_idx, box, confidence) for predictions
        n_gt: Total number of ground truth objects
        iou_threshold: Threshold to consider a detection as correct
    
    Returns:
        AP value for this class
    """
    # Sort predictions by confidence
    pred_boxes = sorted(pred_boxes, key=lambda x: x[2], reverse=True)
    
    tp = np.zeros(len(pred_boxes))
    fp = np.zeros(len(pred_boxes))
    gt_used = {img_idx: [] for img_idx, _ in gt_boxes}
    
    # For each prediction, check if it's correct
    for pred_idx, (img_idx, pred_box, _) in enumerate(pred_boxes):
        # Get ground truth boxes for this image
        img_gt_boxes = [(box, i) for i, (idx, box) in enumerate(gt_boxes) 
                       if idx == img_idx]
        
        max_iou = 0
        best_gt_idx = None
        
        # Find best matching ground truth box
        for gt_box, gt_idx in img_gt_boxes:
            if gt_idx in gt_used[img_idx]:
                continue
            iou = calculate_iou(pred_box, gt_box)  # You'll need to implement this
            if iou > max_iou:
                max_iou = iou
                best_gt_idx = gt_idx
        
        if max_iou >= iou_threshold:
            tp[pred_idx] = 1
            gt_used[img_idx].append(best_gt_idx)
        else:
            fp[pred_idx] = 1
    
    # Calculate precision and recall
    tp_cumsum = np.cumsum(tp)
    fp_cumsum = np.cumsum(fp)
    recalls = tp_cumsum / n_gt
    precisions = tp_cumsum / (tp_cumsum + fp_cumsum)
    
    # Calculate area under precision-recall curve
    ap = 0
    for i in range(len(recalls)-1):
        ap += (recalls[i+1] - recalls[i]) * precisions[i+1]
    
    return ap

def calculate_iou(box1, box2):
    """
    Calculate Intersection over Union (IoU) between two bounding boxes
    
    Args:
        box1: First bounding box coordinates [x1, y1, x2, y2]
              where (x1,y1) is the top-left corner and (x2,y2) is the bottom-right corner
        box2: Second bounding box coordinates in the same format
    
    Returns:
        IoU score between 0 and 1
    """
    # Calculate coordinates of intersection rectangle
    x_left = max(box1[0], box2[0])     # Rightmost left corner
    y_top = max(box1[1], box2[1])      # Bottommost top corner
    x_right = min(box1[2], box2[2])    # Leftmost right corner
    y_bottom = min(box1[3], box2[3])   # Topmost bottom corner

    # If there is no intersection, return 0
    if x_right < x_left or y_bottom < y_top:
        return 0.0

    # Calculate intersection area
    intersection_area = (x_right - x_left) * (y_bottom - y_top)

    # Calculate area of both boxes
    box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
    box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])

    # Calculate union area: sum of both areas minus intersection
    union_area = box1_area + box2_area - intersection_area

    # Handle edge case of zero union area
    if union_area == 0:
        return 0.0

    # Calculate IoU
    iou = intersection_area / union_area

    return iou

def calculate_map(gt_predictions, distorted_predictions, iou_threshold=0.5):
    """
    Calculate mAP for entire batch
    """
    # Organize data by class
    classes_data = process_by_class(gt_predictions, distorted_predictions)
    
    # Calculate AP for each class
    aps = []
    for class_id, data in classes_data.items():
        ap = calculate_ap(
            data['gt_boxes'],
            data['pred_boxes'], 
            data['gt_count'],
            iou_threshold
        )
        aps.append(ap)
    
    # Calculate mean AP
    mAP = sum(aps) / len(aps)
    return mAP