import subprocess
import random
import json
import os
import shutil

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

def extract_patches(video_path, n_patches=50, patch_size=640):
    """
    Extract random patches from video frames
    Args:
        video_path: Path to input video
        n_patches: Number of patches to extract
        patch_size: Size of square patches in pixels
    """
    # Get video dimensions and duration
    width, height, duration = get_video_info(video_path)

    # Get video name without extension
    video_name = os.path.splitext(os.path.basename(video_path))[0]

    # Extract each patch
    for i in range(n_patches):
        # Generate random coordinates within video dimensions
        x = random.randint(0, width - patch_size)
        y = random.randint(0, height - patch_size)
        
        # Generate random timestamp
        timestamp = random.uniform(0, duration)
        
        # FFmpeg command to extract the patch
        output_filename = f'extracted_patches/{video_name}_patch_{i}.jpg'
        cmd = [
            'ffmpeg', '-ss', str(timestamp),
            '-i', video_path,
            '-vf', f'crop={patch_size}:{patch_size}:{x}:{y}',
            '-frames:v', '1',  # extract single frame
            output_filename
        ]
        
        # Execute FFmpeg command
        subprocess.run(cmd)
        print(f'Extracted patch {i} from position ({x},{y}) at time {timestamp:.2f}s')


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