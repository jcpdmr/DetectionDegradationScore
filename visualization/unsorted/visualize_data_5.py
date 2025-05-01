import numpy as np
import skimage
from skimage import io, img_as_float
import matplotlib.pyplot as plt

# Load a ground truth image (replace 'gt_image.png' with your image path)
gt_image = img_as_float(io.imread("example_center_crop_640x640.jpeg"))


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


# Define different channel misalignment amounts (dx_r, dy_r, dx_g, dy_g, dx_b, dy_b)
misalignment_amounts = [
    (0, 0, 0, 0, 0, 0),  # No misalignment (original)
    (1, 0, -1, 0, 0, 0),  # Red right, Green left
    (0, 1, 0, -1, 0, 0),  # Red down, Green up
    (1, 1, -1, -1, 0, 0),  # Red right/down, Green left/up
    (0, 0, 1, 0, -1, 0),  # Green right, Blue left
    (0, 0, 0, 1, 0, -1),  # Green down, Blue up
]

# Create and save the channel misaligned images
for i, (dx_r, dy_r, dx_g, dy_g, dx_b, dy_b) in enumerate(misalignment_amounts):
    misaligned_image = channel_misalignment(
        gt_image, dx_r, dy_r, dx_g, dy_g, dx_b, dy_b
    )
    filename = f"channel_misalignment_{dx_r}_{dy_r}_{dx_g}_{dy_g}_{dx_b}_{dy_b}.png"  # Create a filename
    plt.imsave(
        filename, misaligned_image, format="png"
    )  # Save the image using matplotlib
    print(f"Saved image: {filename}")

# Optional: Display the images in a grid (for quick visual inspection)
num_images = len(misalignment_amounts)
num_cols = 3
num_rows = (num_images + num_cols - 1) // num_cols

fig, axes = plt.subplots(num_rows, num_cols, figsize=(12, 4 * num_rows))
axes = axes.ravel()

for i, (dx_r, dy_r, dx_g, dy_g, dx_b, dy_b) in enumerate(misalignment_amounts):
    misaligned_image = channel_misalignment(
        gt_image, dx_r, dy_r, dx_g, dy_g, dx_b, dy_b
    )
    axes[i].imshow(misaligned_image)
    axes[i].set_title(f"R({dx_r},{dy_r}) G({dx_g},{dy_g}) B({dx_b},{dy_b})")
    axes[i].axis("off")

# Hide unused subplots
for j in range(i + 1, len(axes)):
    axes[j].axis("off")

plt.tight_layout()
plt.show()
