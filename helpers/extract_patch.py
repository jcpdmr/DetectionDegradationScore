import subprocess
import random
import json
import os
import shutil
import cv2
import numpy as np
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import timedelta
from functools import partial
from PIL import Image

def create_directory_structure():
    """
    Create the directory structure for organizing different types of patches
    """
    base_dir = 'patches'
    subdirs = ['extracted', 'compressed', 'distorted']

    # Create base directory if it doesn't exist
    if not os.path.exists(base_dir):
        os.makedirs(base_dir)

    # Create subdirectories
    for subdir in subdirs:
        path = os.path.join(base_dir, subdir)
        os.makedirs(path, exist_ok=True)

def clear_output_folders(clear=True):
    """
    Ask user if they want to clear the output folders
    Returns:
        bool: True if folders should be cleared, False otherwise
    """
    if clear:
        if os.path.exists('patches'):
                shutil.rmtree('patches')
    # while True:
    #     response = input("Do you want to clear all output folders? (yes/no): ").lower()
    #     if response in ['yes', 'y']:
    #         if os.path.exists('patches'):
    #             shutil.rmtree('patches')
    #         return True
    #     elif response in ['no', 'n']:
    #         return False
    #     print("Please enter 'yes' or 'no'")

def get_video_info(video_path):
    """
    Extract video metadata using ffprobe
    Args:
        video_path: Path to the video file
    Returns:
        width: Video width in pixels
        height: Video height in pixels 
        duration: Video duration in seconds
    """
    cmd = ['ffprobe', '-v', 'quiet', '-print_format', 'json', 
           '-show_format', '-show_streams', video_path]
    result = subprocess.run(cmd, capture_output=True, text=True)
    info = json.loads(result.stdout)
    
    width = int(info['streams'][0]['width'])
    height = int(info['streams'][0]['height'])
    duration = float(info['format']['duration'])
    return width, height, duration

def apply_compression_artifacts(input_path, output_path, quality_range=(20, 50)):
    """
    Apply JPEG compression artifacts to an image with random quality
    
    Args:
        input_path (str): Path to input image
        output_path (str): Path to save compressed image
        quality_range (tuple): Range of quality values (min, max)
    
    Returns:
        str: Path to compressed image
    """
    try:
        # Generate random quality value
        quality = random.randint(quality_range[0], quality_range[1])
        
        # Open and convert image to RGB
        with Image.open(input_path) as img:
            if img.mode != 'RGB':
                img = img.convert('RGB')
            
            # Save with JPEG compression
            img.save(output_path, 'JPEG', quality=quality, optimize=True)
            
        return output_path
    
    except Exception as e:
        print(f"Error compressing image: {str(e)}")
        return None

def apply_distortions(input_path, output_path):
    """
    Apply distortions to the input image
    """
    img = cv2.imread(input_path)
    
    # Randomly select distortion type
    distortion_type = random.choice(['color_shift'])
    # distortion_type = random.choice(['gaussian_blur'])
    
    if distortion_type == 'gaussian_noise':
        std_dev = random.uniform(1, 8)
        # Convert to float32 for better precision in noise addition
        img_float = img.astype(np.float32) / 255.0
        noise = np.random.normal(0, std_dev/255.0, img.shape)
        noisy = img_float + noise
        # Clip values to maintain valid pixel range
        noisy = np.clip(noisy, 0, 1)
        # Convert back to uint8
        distorted = (noisy * 255).astype(np.uint8)
    
    elif distortion_type == 'gaussian_blur':
        # Kernel sizes for slight to moderate blur
        # Simulating motion blur or out-of-focus effects
        kernel_size = random.choice([3, 5, 7])
        # Sigma values between 0.3 and 2.0 for natural blur appearance
        sigma = random.uniform(0.3, 2.0)
        distorted = cv2.GaussianBlur(img, (kernel_size, kernel_size), sigma)
    
    else:  # color_shift
        # Split into BGR channels
        b, g, r = cv2.split(img)
        
        # Define random shift amount (can be adjusted)
        shift_x = np.random.randint(1, 5)
        
        # Shift blue channel left
        b_shifted = np.roll(b, -shift_x, axis=1)
        
        # Shift red channel right 
        r_shifted = np.roll(r, shift_x, axis=1)
        
        # Merge channels back
        distorted = cv2.merge([b_shifted, g, r_shifted])
    
    cv2.imwrite(output_path, distorted)
    return output_path

def process_single_patch(extracted_patch_path):
    """
    Process a single extracted patch to create compressed and distorted versions
    """
    base_name = os.path.basename(extracted_patch_path)
    name_without_ext = os.path.splitext(base_name)[0]
    
    # Define output paths
    compressed_path = f'patches/compressed/{name_without_ext}.jpg'
    distorted_path = f'patches/distorted/{name_without_ext}.png'
    
    # Apply both transformations
    apply_compression_artifacts(extracted_patch_path, compressed_path)
    apply_distortions(extracted_patch_path, distorted_path)
    
    return compressed_path, distorted_path

def extract_single_patch(video_path, patch_size, output_dir, video_name, patch_info):
    """
    Extract a single patch from a video frame using ffmpeg
    Args:
        video_path: Path to the source video
        patch_size: Size of the patch to extract
        output_dir: Directory to save the extracted patch
        video_name: Name of the video being processed
        patch_info: Tuple containing (index, x, y, timestamp)
    """
    i, x, y, timestamp = patch_info
    output_filename = f'{output_dir}/{video_name}_patch_{i}.png'
    
    # Use faster codec and optimize FFmpeg settings
    cmd = [
        'ffmpeg', '-ss', str(timestamp),
        '-i', video_path,
        '-vf', f'crop={patch_size}:{patch_size}:{x}:{y}',
        '-frames:v', '1',
        '-c:v', 'png',     # Using PNG format for lossless compression
        '-threads', '1',    # Single-threaded for parallel processing
        output_filename
    ]
    subprocess.run(cmd, capture_output=True)
    return i, x, y, timestamp

def extract_patches(video_path, n_patches=50, patch_size=640, max_workers=4):
    """
    Extract random patches from video frames using parallel processing
    Args:
        video_path: Path to input video
        n_patches: Number of patches to extract
        patch_size: Size of square patches in pixels
        max_workers: Maximum number of parallel workers
    """
    # Get video dimensions and duration
    width, height, duration = get_video_info(video_path)
    video_name = os.path.splitext(os.path.basename(video_path))[0]
    output_dir = 'patches/extracted'

    # Pre-generate all random coordinates and timestamps
    patch_infos = [
        (i,
         random.randint(0, width - patch_size),
         random.randint(0, height - patch_size),
         random.uniform(0, duration))
        for i in range(n_patches)
    ]

    # Create partial function with fixed arguments
    extract_fn = partial(
        extract_single_patch,
        video_path,
        patch_size,
        output_dir,
        video_name
    )

    # Extract patches in parallel
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_patch = {
            executor.submit(extract_fn, patch_info): patch_info 
            for patch_info in patch_infos
        }
        
        for future in as_completed(future_to_patch):
            i, x, y, timestamp = future.result()
            # print(f'Extracted patch {i} from position ({x},{y}) at time {timestamp:.2f}s')

def process_videos(video_folder, n_patches=64, max_workers=4):
    """
    Process all videos and create compressed/distorted versions in parallel
    """
    start_time = time.time()
    print("Starting patch extraction and processing...")

    # Get video files
    video_files = [
        os.path.join(video_folder, f) 
        for f in os.listdir(video_folder) 
        if f.endswith(('.mp4', '.avi', '.mov'))
    ]
    n_videos = len(video_files)
    expected_patches = n_videos * n_patches

    # Extract patches from all videos
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = [
            executor.submit(extract_patches, video_path, n_patches)
            for video_path in video_files
        ]
        
        for future in as_completed(futures):
            future.result()

    # Get list of extracted patches
    extracted_patches = [
        os.path.join('patches/extracted', f)
        for f in os.listdir('patches/extracted')
        if f.endswith('.png')
    ]

    # Process extracted patches in parallel
    print("\nStarting compression and distortion processing...")
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = [
            executor.submit(process_single_patch, patch_path)
            for patch_path in extracted_patches
        ]
        
        for future in as_completed(futures):
            future.result()

    # Verify results
    extracted_count = len(extracted_patches)
    compressed_count = len([f for f in os.listdir('patches/compressed') if f.endswith('.jpg')])
    distorted_count = len([f for f in os.listdir('patches/distorted') if f.endswith('.png')])
    
    # Calculate elapsed time
    elapsed_time = time.time() - start_time
    elapsed_formatted = str(timedelta(seconds=int(elapsed_time)))

    # Print verification results
    print("\n-----Processing Results-----")
    print(f"Number of videos processed: {n_videos}")
    print(f"Patches per video: {n_patches}")
    print(f"Expected patches: {expected_patches}")
    print(f"Extracted patches: {extracted_count}")
    print(f"Compressed patches: {compressed_count}")
    print(f"Distorted patches: {distorted_count}")
    print(f"Total execution time: {elapsed_formatted} (HH:MM:SS)")
    print("-------------------------\n")
    
    if extracted_count == compressed_count == distorted_count == expected_patches:
        print("✓ Processing completed successfully")
    else:
        raise Exception("Processing incomplete: Mismatch in number of processed patches")
if __name__ == '__main__':
    clear_output_folders(clear=True)
    create_directory_structure()
    
    # Process all videos with parallel execution
    process_videos('videos/', n_patches=64, max_workers=6)
