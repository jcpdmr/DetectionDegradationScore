import os
import shutil

def clear_directory(directory):
    """
    Remove all files in the specified directory
    """
    print(f"Attempting to clear directory: {directory}")
    if os.path.exists(directory):
        print(f"Directory exists: {directory}")
        for file in os.listdir(directory):
            file_path = os.path.join(directory, file)
            if os.path.isfile(file_path):
                os.remove(file_path)
    else:
        print(f"Directory does not exist: {directory}")

def copy_files(src_dir, dst_dir):
    """
    Copy all files from source directory to destination directory
    """
    print(f"\nAttempting to copy from {src_dir} to {dst_dir}")
    
    if not os.path.exists(dst_dir):
        print(f"Creating destination directory: {dst_dir}")
        os.makedirs(dst_dir)
    
    if os.path.exists(src_dir):
        print(f"Source directory exists: {src_dir}")
        files = os.listdir(src_dir)
        # print(f"Files found in source: {files}")
        for file in files:
            src_file = os.path.join(src_dir, file)
            dst_file = os.path.join(dst_dir, file)
            if os.path.isfile(src_file):
                shutil.copy2(src_file, dst_file)
                # print(f"Copied: {file}")
    else:
        print(f"Source directory does not exist: {src_dir}")

def copy_for_mAP_evaluation(prediction_type):
    """
    Copy files to appropriate directories for mAP evaluation
    prediction_type: either 'compressed' or 'distorted'
    """
    # Get current working directory
    current_dir = os.getcwd()
    print(f"Current working directory: {current_dir}")
    
    # Define directories relative to /yoloios
    patches_dir = "patches/extracted"
    ground_truth_dir = "prediction_results/extracted"
    predictions_dir = f"prediction_results/{prediction_type}"
    
    mAP_images_dir = "../mAP/input/images-optional"
    mAP_ground_truth_dir = "../mAP/input/ground-truth"
    mAP_predictions_dir = "../mAP/input/detection-results"
    
    # Print absolute paths
    print("\nAbsolute paths:")
    print(f"Patches dir: {os.path.abspath(patches_dir)}")
    print(f"Ground truth dir: {os.path.abspath(ground_truth_dir)}")
    print(f"Predictions dir: {os.path.abspath(predictions_dir)}")
    print(f"mAP images dir: {os.path.abspath(mAP_images_dir)}")
    print(f"mAP ground truth dir: {os.path.abspath(mAP_ground_truth_dir)}")
    print(f"mAP predictions dir: {os.path.abspath(mAP_predictions_dir)}")
    
    # Clear and copy files for images
    print("\nClearing and copying image files...")
    clear_directory(mAP_images_dir)
    copy_files(patches_dir, mAP_images_dir)
    
    # Copy ground truth files
    print("\nCopying ground truth files...")
    copy_files(ground_truth_dir, mAP_ground_truth_dir)
    
    # Copy prediction files
    print("\nCopying prediction files...")
    copy_files(predictions_dir, mAP_predictions_dir)
    
    print("\nFile copying completed!")

if __name__ == "__main__":
    # while True:
    #     pred_type = input("Enter prediction type (compressed/distorted): ").lower()
    #     if pred_type in ['compressed', 'distorted']:
    #         break
    #     print("Invalid input. Please enter either 'compressed' or 'distorted'")
    pred_type = 'compressed'
    copy_for_mAP_evaluation(pred_type)