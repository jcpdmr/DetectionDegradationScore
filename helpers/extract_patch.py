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

# def apply_compression_artifacts(input_path, output_path, quality=30):
#     """
#     Apply compression artifacts using FFmpeg with a low quality setting
#     Args:
#         input_path: Path to input image
#         output_path: Path to save compressed image
#         quality: JPEG quality (0-100, where 20-30 simulates typical web video compression)
#                         Lower values create more visible artifacts but stay in realistic range
#     """
#     cmd = [
#         'ffmpeg', '-i', input_path,
#         '-qscale:v', str(quality),  # Set quality level
#         '-codec:v', 'mjpeg',   # Use JPEG compression
#         output_path
#     ]
#     subprocess.run(cmd, capture_output=True)

# def apply_lpips_style_distortions(image_path):
#     """
#     Apply distortions similar to those used in LPIPS training
#     Args:
#         image_path: Path to input image
#     Returns:
#         Distorted image as numpy array
#     """
#     # Read image
#     img = cv2.imread(image_path)
    
#     # Randomly select distortion type
#     # ['gaussian_noise', 'gaussian_blur', 'color_shift']
#     distortion_type = random.choice(['gaussian_blur']) 
    
#     if distortion_type == 'gaussian_noise':
#         # Add Gaussian noise
#         noise = np.random.normal(0, 25, img.shape).astype(np.uint8)
#         distorted = cv2.add(img, noise)
    
#     elif distortion_type == 'gaussian_blur':
#         # Apply Gaussian blur
#         kernel_size = random.choice([3, 5, 7])
#         distorted = cv2.GaussianBlur(img, (kernel_size, kernel_size), 0)
    
#     else:  # color_shift
#         # Randomly shift color channels
#         channels = cv2.split(img)
#         shift_amount = random.randint(10, 30)
#         shifted_channels = [np.roll(channel, shift_amount) for channel in channels]
#         distorted = cv2.merge(shifted_channels)
    
#     return distorted

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
    Process all videos in parallel using a thread pool
    Args:
        video_folder: Folder containing input videos
        n_patches: Number of patches to extract per video
        max_workers: Maximum number of parallel workers
    """

    # Start timing the entire process
    start_time = time.time()
    print("Starting patch extraction...")

    video_files = [
        os.path.join(video_folder, f) 
        for f in os.listdir(video_folder) 
        if f.endswith(('.mp4', '.avi', '.mov'))
    ]
    n_videos = len(video_files)
    expected_patches = n_videos * n_patches

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = [
            executor.submit(extract_patches, video_path, n_patches)
            for video_path in video_files
        ]
        
        for future in as_completed(futures):
            future.result()

    # Verify the number of extracted patches
    extracted_patches = len([
        f for f in os.listdir('patches/extracted') 
        if f.endswith('.png')
    ])
    
    # Calculate elapsed time
    elapsed_time = time.time() - start_time
    # Format times for better readability
    elapsed_formatted = str(timedelta(seconds=int(elapsed_time)))

    # Print verification results
    print("\n-----Extraction Verification Results-----")
    print(f"Number of videos processed: {n_videos}")
    print(f"Patches per video: {n_patches}")
    print(f"Expected total patches: {expected_patches}")
    print(f"Actually extracted patches: {extracted_patches}")
    print(f"Total execution time: {elapsed_formatted} (HH:MM:SS)")
    print("----------------------------------------\n")
    
    if extracted_patches == expected_patches:
        print("✓ Extraction completed successfully: All patches were extracted")
    else:
        raise Exception(f"Extraction incomplete: {expected_patches - extracted_patches} patches are missing")
if __name__ == '__main__':
    clear_output_folders(clear=True)
    create_directory_structure()
    
    # Process all videos with parallel execution
    process_videos('videos/', n_patches=64, max_workers=6)
