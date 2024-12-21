import numpy as np
import matplotlib.pyplot as plt
import torch
import os
from typing import Tuple


def visualize_feature_maps(features: torch.Tensor,
                           name: str,
                           num_features: int = 16,
                           figsize: Tuple[int, int] = (30, 15),
                           output_path: str = 'feature_maps.png',
                           ):
    """
    Visualize and save feature maps
    Args:
        features: Extracted feature maps
        num_features: Number of feature maps to visualize
        figsize: Figure size for the plot
        output_path: Path to save the visualization
    """
    features = features.cpu().numpy()
    fig = plt.figure(figsize=figsize)

    # 1. Mean of all channels
    ax1 = fig.add_subplot(131)
    mean_activation = np.mean(features[0], axis=0)
    im1 = ax1.imshow(mean_activation, cmap='viridis')
    ax1.set_title('Mean of all channels')
    plt.colorbar(im1, ax=ax1)

    # 2. Grid of first n feature maps
    ax2 = fig.add_subplot(132)
    grid_size = int(np.ceil(np.sqrt(num_features)))
    grid = np.zeros(
        (grid_size * features.shape[2], grid_size * features.shape[3]))

    for idx in range(min(num_features, features.shape[1])):
        i = idx // grid_size
        j = idx % grid_size
        grid[i*features.shape[2]:(i+1)*features.shape[2],
             j*features.shape[3]:(j+1)*features.shape[3]] = features[0, idx]

    im2 = ax2.imshow(grid, cmap='viridis')
    ax2.set_title(f'First {num_features} feature maps')
    plt.colorbar(im2, ax=ax2)

    # 3. Maximum activation map
    ax3 = fig.add_subplot(133)
    max_activation = np.max(features[0], axis=0)
    im3 = ax3.imshow(max_activation, cmap='viridis')
    ax3.set_title('Maximum activation')
    plt.colorbar(im3, ax=ax3)

    plt.tight_layout()
    plt.savefig(os.path.join(output_path, f'feature_maps_{name}.png'))
    plt.close()

    # Create and save activation distribution histogram
    plt.figure(figsize=(10, 5))
    plt.hist(features[0].flatten(), bins=50)
    plt.title('Activation Distribution')
    plt.xlabel('Activation Value')
    plt.ylabel('Frequency')
    plt.savefig(os.path.join(
        output_path, f'activation_distribution_{name}.png'))
    plt.close()


def analyze_features(features: torch.Tensor, name: str, output_path: str):
    """
    Analyze feature maps and print statistics
    Args:
        features: Extracted feature maps
        name: Name of the feature maps channel
        output_path: Path to save visualizations
    """
    subfolder_path = os.path.join(output_path, f'{name}')
    if not os.path.exists(subfolder_path):
        os.makedirs(subfolder_path)

    with open(os.path.join(subfolder_path, f'features_stats_{name}.txt'), 'w', encoding="utf-8") as f:
        f.write(f"Feature maps {name} statistics:\n")
        f.write(f"- Shape: {features.shape}\n")
        f.write(f"- Minimum value: {features.min():.4f}\n")
        f.write(f"- Maximum value: {features.max():.4f}\n")
        f.write(f"- Mean: {features.mean():.4f}\n")
        f.write(f"- Standard deviation: {features.std():.4f}\n")

    visualize_feature_maps(features, name=name, output_path=subfolder_path)
    print("Visualizations saved successfully!")
