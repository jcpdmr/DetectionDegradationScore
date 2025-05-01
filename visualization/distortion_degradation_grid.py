import numpy as np
import skimage
from skimage import io, metrics, img_as_float
from skimage.filters import gaussian
import matplotlib.pyplot as plt
import skimage.util

# For LPIPS:
import torch
import lpips

# Load a ground truth image (replace 'gt_image.png' with your image path)
gt_image = img_as_float(io.imread("example_center_crop_640x640.jpeg"))

# Initialize LPIPS (requires CUDA if available)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
lpips_alex = lpips.LPIPS(net="alex").to(device)  # You can also use 'vgg'


# Define artifact functions
def pixel_shift(image, dx=1, dy=0):
    """Sposta l'immagine di dx pixel in orizzontale e dy pixel in verticale."""
    shifted = np.roll(image, shift=(dy, dx), axis=(0, 1))
    return shifted


def channel_misalignment(image, dx_r=0, dy_r=0, dx_g=0, dy_g=0, dx_b=0, dy_b=0):
    """
    Sposta individualmente i canali R, G e B di dx pixel in orizzontale e dy pixel in verticale.
    """
    shifted_image = np.zeros_like(
        image
    )  # Crea un array vuoto con la stessa forma dell'immagine originale

    # Sposta il canale rosso
    shifted_image[:, :, 0] = np.roll(
        image[:, :, 0], shift=(dy_r, dx_r), axis=(0, 1)
    )  # Applica lo shift
    # Sposta il canale verde
    shifted_image[:, :, 1] = np.roll(
        image[:, :, 1], shift=(dy_g, dx_g), axis=(0, 1)
    )  # Applica lo shift
    # Sposta il canale blu
    shifted_image[:, :, 2] = np.roll(
        image[:, :, 2], shift=(dy_b, dx_b), axis=(0, 1)
    )  # Applica lo shift

    return shifted_image


def quantize(image, levels=32):
    """Quantizza l'immagine a un numero specificato di livelli."""
    return (np.floor(image * levels) / levels).astype(image.dtype)


def localized_corruption(image, x, y, width, height, corruption_level=0.5):
    """Corrompe una regione rettangolare dell'immagine."""
    corrupted = image.copy()
    corrupted[y : y + height, x : x + width] = (
        corruption_level  # Sostituisce con un livello
    )
    return corrupted


# 1. Translation
translated_image = np.roll(gt_image, shift=5, axis=1)  # Shift 5 pixels right

# 2. Blur
blurred_image = gaussian(gt_image, sigma=2, channel_axis=-1)  # Ensure blur works on RGB

# 3. Contrast Adjustment (example - simple contrast reduction)
contrast_adjusted_image = np.clip(gt_image * 0.7 + 0.1, 0, 1)

# 4. Chromatic Aberration
chromatic_aberration_image = channel_misalignment(
    gt_image, dx_r=1, dy_r=1, dx_g=-1, dy_g=-1, dx_b=0, dy_b=0
)

# 5. Quantization
quantized_image = quantize(
    gt_image, levels=16
)  # Riduce a 16 livelli di colore per canale

# 6. Localized Corruption
corruption_width = int(gt_image.shape[1] * 0.2)  # 20% della larghezza dell'immagine
corruption_height = int(gt_image.shape[0] * 0.2)  # 20% dell'altezza dell'immagine
corruption_x = int(
    gt_image.shape[1] * 0.5 - corruption_width // 2 + gt_image.shape[1] * 0.1
)  # Centro + 10% a destra
corruption_y = int(gt_image.shape[0] * 0.5 - corruption_height // 2)  # Centro
corrupted_image = localized_corruption(
    gt_image,
    corruption_x,
    corruption_y,
    corruption_width,
    corruption_height,
    corruption_level=0.2,
)

# 7. Additive Gaussian Noise
noisy_image = skimage.util.random_noise(gt_image, mode="gaussian", var=0.01)

# 8. Salt and Pepper Noise
salt_pepper_noise = skimage.util.random_noise(gt_image, mode="s&p", amount=0.05)


# Dictionary of distorted images
distorted_images = {
    "Original": gt_image,
    "Translation": translated_image,
    "Blur": blurred_image,
    "Contrast Reduction": contrast_adjusted_image,
    "Chromatic Aberration": chromatic_aberration_image,
    "Quantization": quantized_image,
    "Localized Corruption": corrupted_image,
    "Gaussian Noise": noisy_image,
    "Salt & Pepper Noise": salt_pepper_noise,
}

# Calculate MSE, PSNR, SSIM, and LPIPS for each distortion
metrics_values = {}
for distortion_name, distorted_img in distorted_images.items():
    mse_value = metrics.mean_squared_error(gt_image, distorted_img)
    psnr_value = metrics.peak_signal_noise_ratio(
        gt_image, distorted_img, data_range=1.0
    )
    ssim_value = metrics.structural_similarity(
        gt_image, distorted_img, channel_axis=-1, data_range=1.0
    )  # Added SSIM
    # Calculate LPIPS
    img1_tensor = (
        torch.from_numpy(gt_image).permute(2, 0, 1).unsqueeze(0).to(device).float()
    )  # C, H, W, CONVERT TO FLOAT32
    img2_tensor = (
        torch.from_numpy(distorted_img).permute(2, 0, 1).unsqueeze(0).to(device).float()
    )  # C, H, W, CONVERT TO FLOAT32
    lpips_value = lpips_alex(img1_tensor, img2_tensor).item()

    metrics_values[distortion_name] = {
        "MSE": mse_value,
        "PSNR": psnr_value,
        "SSIM": ssim_value,
        "LPIPS": lpips_value,
    }

    # Print the metrics for this distortion
    print(f"Metrics for {distortion_name}:")
    print(f"  MSE: {mse_value:.4f}")
    print(f"  PSNR: {psnr_value:.2f} dB")
    print(f"  SSIM: {ssim_value:.4f}")
    print(f"  LPIPS: {lpips_value:.4f}")
    print("-" * 30)  # Separator


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
    ax.axis("off")

    # Add metrics as text below the image
    mse_text = f"MSE: {metrics_values[distortion_name]['MSE']:.4f}"
    psnr_text = f"PSNR: {metrics_values[distortion_name]['PSNR']:.2f} dB"
    ssim_text = f"SSIM: {metrics_values[distortion_name]['SSIM']:.4f}"
    lpips_text = f"LPIPS: {metrics_values[distortion_name]['LPIPS']:.4f}"
    ax.text(
        0.02,
        0.98,
        f"{mse_text}\n{psnr_text}\n{ssim_text}\n{lpips_text}",
        size=11,
        ha="left",
        va="top",
        transform=ax.transAxes,
        color="black",
        # bbox=dict(facecolor="black", alpha=0.5),
    )


# Hide any unused subplots
for j in range(i + 1, len(axes)):
    axes[j].axis("off")


# Adjust spacing between subplots
plt.subplots_adjust(
    wspace=0.3, hspace=0.5
)  # Reduce horizontal space, increase vertical space

# Remove whitespace around the plot
plt.tight_layout()

# Save the figure with higher DPI
plt.savefig("distorted_images_grid.png", dpi=300)  # Set DPI here
plt.show()
