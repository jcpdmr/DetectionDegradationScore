import numpy as np
import skimage
from skimage import io, metrics, img_as_float
from skimage.filters import gaussian
from skimage.transform import warp, AffineTransform
import matplotlib.pyplot as plt
import skimage.util

# For LPIPS:
import torch
import lpips

# Load a ground truth image (replace 'gt_image.png' with your image path)
gt_image = img_as_float(io.imread('example_center_crop_640x640.jpeg'))

# Initialize LPIPS (requires CUDA if available)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
lpips_alex = lpips.LPIPS(net='alex').to(device)  # You can also use 'vgg'

# 1. Translation
translated_image = np.roll(gt_image, shift=5, axis=1)  # Shift 5 pixels right

# 2. Blur
blurred_image = gaussian(gt_image, sigma=2, channel_axis=-1) # Ensure blur works on RGB

# 3. Contrast Adjustment (example - simple contrast reduction)
contrast_adjusted_image = np.clip(gt_image * 0.7 + 0.1, 0, 1)

# 4. Rotation (example - small rotation)
transform = AffineTransform(scale=1, rotation=np.deg2rad(3))
rotated_image = warp(gt_image, transform, order=1, preserve_range=True)

# 5. Additive Gaussian Noise
noisy_image = skimage.util.random_noise(gt_image, mode='gaussian', var=0.01)

# Dictionary of distorted images
distorted_images = {
    "Original": gt_image,
    "Translation": translated_image,
    "Blur": blurred_image,
    "Contrast Reduction": contrast_adjusted_image,
    "Rotation": rotated_image,
    "Gaussian Noise": noisy_image,
}

# Calculate MSE, PSNR, SSIM, and LPIPS for each distortion
metrics_values = {}
for distortion_name, distorted_img in distorted_images.items():
    mse_value = metrics.mean_squared_error(gt_image, distorted_img)
    psnr_value = metrics.peak_signal_noise_ratio(gt_image, distorted_img, data_range=1.0)
    ssim_value = metrics.structural_similarity(gt_image, distorted_img, channel_axis=-1, data_range=1.0)  # Added SSIM
    # Calculate LPIPS
    img1_tensor = torch.from_numpy(gt_image).permute(2, 0, 1).unsqueeze(0).to(device).float() # C, H, W, CONVERT TO FLOAT32
    img2_tensor = torch.from_numpy(distorted_img).permute(2, 0, 1).unsqueeze(0).to(device).float() # C, H, W, CONVERT TO FLOAT32
    lpips_value = lpips_alex(img1_tensor, img2_tensor).item()

    metrics_values[distortion_name] = {"MSE": mse_value, "PSNR": psnr_value, "SSIM": ssim_value, "LPIPS": lpips_value}

    # Print the metrics for this distortion
    print(f"Metrics for {distortion_name}:")
    print(f"  MSE: {mse_value:.4f}")
    print(f"  PSNR: {psnr_value:.2f} dB")
    print(f"  SSIM: {ssim_value:.4f}")
    print(f"  LPIPS: {lpips_value:.4f}")
    print("-" * 30) # Separator

# Create the grid for display
num_images = len(distorted_images)
num_cols = 3  # Adjust for desired grid layout
num_rows = (num_images + num_cols - 1) // num_cols  # Calculate number of rows

fig, axes = plt.subplots(num_rows, num_cols, figsize=(15, 5 * num_rows))
axes = axes.ravel()  # Flatten the axes array for easy indexing

for i, (distortion_name, distorted_img) in enumerate(distorted_images.items()):
    ax = axes[i]
    ax.imshow(distorted_img)  # No cmap='gray' for RGB
    ax.set_title(distortion_name)
    ax.axis('off')

    # Add metrics as text below the image
    mse_text = f"MSE: {metrics_values[distortion_name]['MSE']:.4f}"
    psnr_text = f"PSNR: {metrics_values[distortion_name]['PSNR']:.2f} dB"
    ssim_text = f"SSIM: {metrics_values[distortion_name]['SSIM']:.4f}"
    lpips_text = f"LPIPS: {metrics_values[distortion_name]['LPIPS']:.4f}"
    ax.text(0.5, -0.15, f"{mse_text}\n{psnr_text}\n{ssim_text}\n{lpips_text}", size=8, ha="center", transform=ax.transAxes) # Added LPIPS

# Hide any unused subplots
for j in range(i + 1, len(axes)):
    axes[j].axis('off')

# Adjust spacing between subplots
plt.subplots_adjust(wspace=0.1, hspace=0.4) # Reduce horizontal space, increase vertical space

# Remove whitespace around the plot
plt.tight_layout()

# Save the figure with higher DPI
plt.savefig("distorted_images_grid_rgb.png", dpi=300) # Set DPI here
plt.show()