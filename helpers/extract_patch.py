import subprocess
import random
import json
import os
import shutil
import cv2
import numpy as np

def clear_output_folder():
    """
    Ask user if they want to clear the output folder
    Returns:
        bool: True if folder should be cleared, False otherwise
    """
    while True:
        response = input("Do you want to clear the 'extracted_patches' folder? (yes/no): ").lower()
        if response in ['yes', 'y']:
            if os.path.exists('extracted_patches'):
                shutil.rmtree('extracted_patches')
            return True
        elif response in ['no', 'n']:
            return False
        print("Please enter 'yes' or 'no'")

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

def apply_compression_artifacts(input_path, output_path, quality=20):
    """
    Apply compression artifacts using FFmpeg with a low quality setting
    Args:
        input_path: Path to input image
        output_path: Path to save compressed image
        quality: JPEG quality (0-100, where 20-30 simulates typical web video compression)
                        Lower values create more visible artifacts but stay in realistic range
    """
    cmd = [
        'ffmpeg', '-i', input_path,
        '-qscale:v', str(quality),  # Set quality level
        '-codec:v', 'mjpeg',   # Use JPEG compression
        output_path
    ]
    subprocess.run(cmd, capture_output=True)

def apply_lpips_style_distortions(image_path):
    """
    Apply distortions similar to those used in LPIPS training
    Args:
        image_path: Path to input image
    Returns:
        Distorted image as numpy array
    """
    # Read image
    img = cv2.imread(image_path)
    
    # Randomly select distortion type
    # ['gaussian_noise', 'gaussian_blur', 'color_shift']
    distortion_type = random.choice(['gaussian_blur']) 
    
    if distortion_type == 'gaussian_noise':
        # Add Gaussian noise
        noise = np.random.normal(0, 25, img.shape).astype(np.uint8)
        distorted = cv2.add(img, noise)
    
    elif distortion_type == 'gaussian_blur':
        # Apply Gaussian blur
        kernel_size = random.choice([3, 5, 7])
        distorted = cv2.GaussianBlur(img, (kernel_size, kernel_size), 0)
    
    else:  # color_shift
        # Randomly shift color channels
        channels = cv2.split(img)
        shift_amount = random.randint(10, 30)
        shifted_channels = [np.roll(channel, shift_amount) for channel in channels]
        distorted = cv2.merge(shifted_channels)
    
    return distorted

def extract_patches(video_path, n_patches=50, patch_size=640):
    """
    Extract random patches from video frames and create distorted versions
    Args:
        video_path: Path to input video
        n_patches: Number of patches to extract
        patch_size: Size of square patches in pixels
    """
    # Get video dimensions and duration
    width, height, duration = get_video_info(video_path)
    video_name = os.path.splitext(os.path.basename(video_path))[0]

    for i in range(n_patches):
        # Generate random coordinates and timestamp
        x = random.randint(0, width - patch_size)
        y = random.randint(0, height - patch_size)
        timestamp = random.uniform(0, duration)
        
        # Extract original patch
        original_filename = f'extracted_patches/{video_name}_patch_{i}.jpg'
        cmd = [
            'ffmpeg', '-ss', str(timestamp),
            '-i', video_path,
            '-vf', f'crop={patch_size}:{patch_size}:{x}:{y}',
            '-frames:v', '1',
            original_filename
        ]
        subprocess.run(cmd)

        # Create compression artifacts version
        compressed_filename = f'extracted_patches/{video_name}_patch_{i}_compressed.jpg'
        apply_compression_artifacts(original_filename, compressed_filename)

        # Create LPIPS-style distorted version
        distorted = apply_lpips_style_distortions(original_filename)
        distorted_filename = f'extracted_patches/{video_name}_patch_{i}_distorted.jpg'
        cv2.imwrite(distorted_filename, distorted)

        print(f'Extracted patch {i} from position ({x},{y}) at time {timestamp:.2f}s')


if __name__ == '__main__':
    # Check if user wants to clear output folder and create it if needed
    clear_output_folder()
    os.makedirs('extracted_patches', exist_ok=True)

    # Process all videos in the folder
    video_folder = 'videos/'
    video_files = [f for f in os.listdir(video_folder) if f.endswith(('.mp4', '.avi', '.mov'))]

    for video_name in video_files:
        print(f'Processing video: {video_name}')
        video_path = os.path.join(video_folder, video_name)
        extract_patches(video_path, n_patches=20)