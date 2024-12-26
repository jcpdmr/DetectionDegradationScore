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


def clear_output_folder():
    """
    Ask user if they want to clear the output folder
    Returns:
        bool: True if folder should be cleared, False otherwise
    """
    while True:
        response = input("Do you want to clear the 'prediction_results' folder? (yes/no): ").lower()
        if response in ['yes', 'y']:
            if os.path.exists('prediction_results'):
                shutil.rmtree('prediction_results')
            return True
        elif response in ['no', 'n']:
            return False
        print("Please enter 'yes' or 'no'")

def process_batch(image_paths):
    """
    Process a batch of images and save predictions for each image
    """
    # Prepare batch of images
    images = []
    for img_path in image_paths:
        img = cv2.imread(img_path)
        if img is not None:
            images.append(img)
    
    if not images:
        return
    
    # Perform batch inference
    with torch.no_grad():
        results = model(images, imgsz=PATCH_SIZE)
    
    # Process results for each image
    for img_path, result in zip(image_paths, results):
        # Create output filename
        filename = Path(img_path).stem
        output_path = Path('prediction_results') / f'{filename}.txt'
        
        # Save predictions to file
        with open(output_path, 'w') as f:
            for box in result.boxes:
                # Extract prediction details
                class_id = int(box.cls[0])
                conf = float(box.conf[0])
                class_name = result.names[class_id]
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                
                # Write prediction in required format
                f.write(f"{class_name} {conf:.6f} {int(x1)} {int(y1)} {int(x2)} {int(y2)}\n")
        
        print(f"Saved predictions for {filename}")

def main():


    # Get all patch files
    input_files = glob.glob('extracted_patches/*_patch_*.jpg')
    
    # Process images in batches
    for i in range(0, len(input_files), BATCH_SIZE):
        batch_paths = input_files[i:i + BATCH_SIZE]
        process_batch(batch_paths)
        print(f"Processed batch {i//BATCH_SIZE + 1}/{(len(input_files) + BATCH_SIZE - 1)//BATCH_SIZE}")

if __name__ == '__main__':
    clear_output_folder()
    # Create output directory if it doesn't exist 
    os.makedirs('prediction_results', exist_ok=True)
    main()
    print("Completed!")