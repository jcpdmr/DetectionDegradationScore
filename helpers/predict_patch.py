from ultralytics import YOLO
import cv2
import os
import glob
import torch
from pathlib import Path
import shutil

PATCH_SIZE = 640
BATCH_SIZE = 32  # Change this value based on the available GPU memory

# Load YOLO model
model = YOLO('yolo11m.pt')

# Enable GPU if available
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)

def create_directory_structure():
    """
    Create the directory structure for organizing prediction results
    """
    base_dir = 'prediction_results'
    subdirs = ['extracted', 'compressed', 'distorted']
    
    # Create base directory if it doesn't exist
    if not os.path.exists(base_dir):
        os.makedirs(base_dir)
    
    # Create subdirectories
    for subdir in subdirs:
        path = os.path.join(base_dir, subdir)
        os.makedirs(path, exist_ok=True)

def clear_output_folder(clear=True):
    """
    Ask user if they want to clear the prediction_results folder
    """
    if clear:
        if os.path.exists('prediction_results'):
                shutil.rmtree('prediction_results')
    # while True:
    #     response = input("Do you want to clear the 'prediction_results' folder? (yes/no): ").lower()
    #     if response in ['yes', 'y']:
    #         if os.path.exists('prediction_results'):
    #             shutil.rmtree('prediction_results')
    #         return True
    #     elif response in ['no', 'n']:
    #         return False
    #     print("Please enter 'yes' or 'no'")

def format_class_name(class_name):
    """
    Format class name by replacing spaces with hyphens, needed for mAP evaluation
    Args:
        class_name: Original class name
    Returns:
        Formatted class name with spaces replaced by hyphens
    """
    return class_name.replace(' ', '-')

def process_batch(image_paths, patch_type):
    """
    Process a batch of images and save predictions for each image
    Args:
        image_paths: List of paths to images
        patch_type: Type of patch (extracted/compressed/distorted)
    """
    # Prepare batch of images
    images = []
    valid_paths = []
    for img_path in image_paths:
        img = cv2.imread(img_path)
        if img is not None:
            images.append(img)
            valid_paths.append(img_path)
    
    if not images:
        return
    
    # Perform batch inference
    with torch.no_grad():
        results = model(images, imgsz=PATCH_SIZE)
    
    # Process results for each image
    for img_path, result in zip(valid_paths, results):
        # Create output filename
        filename = Path(img_path).stem
        output_path = Path('prediction_results') / patch_type / f'{filename}.txt'
        
        # Save predictions to file
        with open(output_path, 'w') as f:
            for box in result.boxes:
                # Extract prediction details
                class_id = int(box.cls[0])
                conf = float(box.conf[0])
                class_name = format_class_name(result.names[class_id])
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                
                # Different format for extracted patches (ground truth)
                if patch_type == 'extracted':
                    f.write(f"{class_name} {int(x1)} {int(y1)} {int(x2)} {int(y2)}\n")
                else:
                    f.write(f"{class_name} {conf:.6f} {int(x1)} {int(y1)} {int(x2)} {int(y2)}\n")
        
        
        print(f"Saved predictions for {filename} in {patch_type}")


def main():
    # Define patch types and their directories
    patch_types = ['extracted', 'compressed', 'distorted']
    
    # Process each type of patch
    for patch_type in patch_types:
        print(f"\nProcessing {patch_type} patches...")
        input_files = glob.glob(f'patches/{patch_type}/*.jpg')
        
        # Process images in batches
        for i in range(0, len(input_files), BATCH_SIZE):
            batch_paths = input_files[i:i + BATCH_SIZE]
            process_batch(batch_paths, patch_type)
            print(f"Processed batch {i//BATCH_SIZE + 1}/{(len(input_files) + BATCH_SIZE - 1)//BATCH_SIZE}")

if __name__ == '__main__':
    clear_output_folder(clear=True)
    create_directory_structure()
    main()
    print("Completed!")