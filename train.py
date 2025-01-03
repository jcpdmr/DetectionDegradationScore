from ultralytics import YOLO
import os
import time
import torch
import torch.nn as nn
from tqdm import tqdm
from patches_loader import create_dataloaders
from typing import Tuple, Dict
from score_metrics import match_predictions
from yoloios import (
    extract_multiple_features_and_predictions,
    LayerConfig,
    YOLOSimilarity,
)


def train_perceptual_loss(
    yolo_model: YOLO,
    num_epochs: int,
    batch_size: int,
    learning_rate: float,
    data_path: str,
    val_frequency: int = 1,
    patience: int = 5,
    output_dir: str = "output",
    seed: int = 42,
):
    """
    Train the perceptual loss model with validation and early stopping
    """
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)

    timestamp = time.strftime("%Y%m%d-%H%M%S")
    # Create run-specific output directory
    run_output_dir = os.path.join(output_dir, timestamp)
    os.makedirs(run_output_dir, exist_ok=True)

    # Create log files with timestamp
    train_log_path = os.path.join(run_output_dir, f"log_{timestamp}_trn.csv")
    val_log_path = os.path.join(run_output_dir, f"log_{timestamp}_val.csv")
    weights_log_path = os.path.join(run_output_dir, f"log_{timestamp}_weights.csv")

    with open(train_log_path, "w") as log_file:
        # Write header
        log_file.write("epoch,loss,avg_error_score,avg_distance,normalized_distance\n")
    with open(val_log_path, "w") as log_file:
        # Write header
        log_file.write("epoch,loss,avg_error_score,avg_distance,normalized_distance\n")
    with open(weights_log_path, "w") as log_file:
        log_file.write("epoch,feature_weight_02,feature_weight_09,feature_weight_16,")
        log_file.write("lin_02_min,lin_02_max,lin_02_mean,lin_02_std,")
        log_file.write("lin_09_min,lin_09_max,lin_09_mean,lin_09_std,")
        log_file.write("lin_16_min,lin_16_max,lin_16_mean,lin_16_std\n")

    # Initialize models
    yolo_model.eval()  # Freeze YOLO weights
    yolo_similarity_model = YOLOSimilarity()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if torch.cuda.is_available():
        yolo_model = yolo_model.cuda()
        yolo_similarity_model = yolo_similarity_model.cuda()

    # Create dataloaders
    train_loader, val_loader = create_dataloaders(data_path, batch_size, seed=seed)

    # Setup training
    optimizer = torch.optim.Adam(yolo_similarity_model.parameters(), lr=learning_rate)
    mse_criterion = nn.MSELoss()
    layer_configs = [
        LayerConfig(2, "02_C3k2_early"),
        LayerConfig(9, "09_SPPF"),
        LayerConfig(16, "16_C3k2_pre_detect"),
    ]

    # Training state
    best_val_loss = float("inf")
    patience_counter = 0

    epoch_prog_bar = tqdm(range(num_epochs), desc="Training", unit="epoch")

    for epoch in epoch_prog_bar:
        # Training phase
        yolo_similarity_model.train()
        train_losses = []
        train_mean_error_score = []
        train_distances = []
        train_norm_distances = []

        train_prog_bar = tqdm(
            train_loader,
            total=len(train_loader),
            desc=f"Epoch {epoch+1}/{num_epochs} (Train)",
            leave=False,
        )

        for batch in train_prog_bar:
            loss, error_score, distance = process_batch(
                yolo_model,
                yolo_similarity_model,
                batch,
                layer_configs,
                mse_criterion,
                device,
                optimizer,
                training=True,
            )
            train_losses.append(loss)
            train_mean_error_score.append(error_score)
            train_distances.append(distance)
            train_norm_distances.append(torch.sigmoid(torch.tensor(distance)).item())

        # Log training metrics
        avg_train_metrics = {
            "loss": sum(train_losses) / len(train_losses),
            "mean_error_score": sum(train_mean_error_score)
            / len(train_mean_error_score),
            "distance": sum(train_distances) / len(train_distances),
            "norm_distance": sum(train_norm_distances) / len(train_norm_distances),
        }

        with open(train_log_path, "a") as log_file:
            log_file.write(
                f"{epoch+1},{avg_train_metrics['loss']:.6f},"
                f"{avg_train_metrics['mean_error_score']:.6f},{avg_train_metrics['distance']:.6f},"
                f"{avg_train_metrics['norm_distance']:.6f}\n"
            )

        with open(weights_log_path, "a") as log_file:
            feature_weights = yolo_similarity_model.layer_weights.detach().cpu().numpy()

            lin_02_weights = yolo_similarity_model.lin_02.weight.detach().cpu().numpy()
            lin_09_weights = yolo_similarity_model.lin_09.weight.detach().cpu().numpy()
            lin_16_weights = yolo_similarity_model.lin_16.weight.detach().cpu().numpy()

            log_file.write(f"{epoch+1},")
            log_file.write(
                f"{feature_weights[0]:.6f},{feature_weights[1]:.6f},{feature_weights[2]:.6f},"
            )

            log_file.write(f"{lin_02_weights.min():.6f},{lin_02_weights.max():.6f},")
            log_file.write(f"{lin_02_weights.mean():.6f},{lin_02_weights.std():.6f},")

            log_file.write(f"{lin_09_weights.min():.6f},{lin_09_weights.max():.6f},")
            log_file.write(f"{lin_09_weights.mean():.6f},{lin_09_weights.std():.6f},")

            log_file.write(f"{lin_16_weights.min():.6f},{lin_16_weights.max():.6f},")
            log_file.write(f"{lin_16_weights.mean():.6f},{lin_16_weights.std():.6f}\n")

        # Validation phase if needed
        if (epoch + 1) % val_frequency == 0:
            yolo_similarity_model.eval()
            val_losses = []
            val_mean_error_score = []
            val_distances = []
            val_norm_distances = []

            val_prog_bar = tqdm(
                val_loader, desc=f"Epoch {epoch+1}/{num_epochs} (Val)", leave=False
            )

            with torch.no_grad():
                for batch in val_prog_bar:
                    loss, error_score, distance = process_batch(
                        yolo_model,
                        yolo_similarity_model,
                        batch,
                        layer_configs,
                        mse_criterion,
                        device,
                        training=False,
                    )
                    val_losses.append(loss)
                    val_mean_error_score.append(error_score)
                    val_distances.append(distance)
                    val_norm_distances.append(
                        torch.sigmoid(torch.tensor(distance)).item()
                    )

            # Log validation metrics
            avg_val_metrics = {
                "loss": sum(val_losses) / len(val_losses),
                "mean_error_score": sum(val_mean_error_score)
                / len(val_mean_error_score),
                "distance": sum(val_distances) / len(val_distances),
                "norm_distance": sum(val_norm_distances) / len(val_norm_distances),
            }

            with open(val_log_path, "a") as log_file:
                log_file.write(
                    f"{epoch+1},{avg_val_metrics['loss']:.6f},"
                    f"{avg_val_metrics['mean_error_score']:.6f},{avg_val_metrics['distance']:.6f},"
                    f"{avg_val_metrics['norm_distance']:.6f}\n"
                )

            # Print current metrics
            print(f"\nEpoch {epoch+1}/{num_epochs}")
            print(
                f"Train - Loss: {avg_train_metrics['loss']:.4f}, Error Score: {avg_train_metrics['mean_error_score']:.4f}, "
                f"Norm Distance: {avg_train_metrics['norm_distance']:.4f}"
            )
            print(
                f"Val   - Loss: {avg_val_metrics['loss']:.4f}, Error Score: {avg_val_metrics['mean_error_score']:.4f}, "
                f"Norm Distance: {avg_val_metrics['norm_distance']:.4f}\n"
            )

            # Save best model and check for early stopping
            if avg_val_metrics["loss"] < best_val_loss:
                best_val_loss = avg_val_metrics["loss"]
                model_save_path = os.path.join(run_output_dir, "best_model.pth")
                torch.save(
                    {
                        "epoch": epoch,
                        "model_state_dict": yolo_similarity_model.state_dict(),
                        "optimizer_state_dict": optimizer.state_dict(),
                        "best_val_loss": best_val_loss,
                        "layer_configs": layer_configs,
                    },
                    model_save_path,
                )
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    print(f"Early stopping triggered after {epoch+1} epochs")
                    break
        else:
            # Print only training metrics
            print(f"\nEpoch {epoch+1}/{num_epochs}")
            print(
                f"Train - Loss: {avg_train_metrics['loss']:.4f}, Error Score: {avg_train_metrics['mean_error_score']:.4f}, "
                f"Norm Distance: {avg_train_metrics['norm_distance']:.4f}\n"
            )


def process_batch(
    yolo_model: YOLO,
    yolo_similarity_model: YOLOSimilarity,
    batch: Dict[str, torch.Tensor],
    layer_configs: list,
    mse_criterion: nn.Module,
    device: torch.device,
    optimizer: torch.optim.Optimizer = None,
    training: bool = True,
) -> Tuple[float, float, float]:
    """
    Process a single batch (training or validation)
    Returns loss, error_score and normalized distance for monitoring
    """
    gt_batch = batch["gt"].to(device)
    modified_batch = batch["modified"].to(device)

    # Extract features and predictions
    gt_features, gt_predictions = extract_multiple_features_and_predictions(
        yolo_model, gt_batch, layer_configs
    )
    mod_features, mod_predictions = extract_multiple_features_and_predictions(
        yolo_model, modified_batch, layer_configs
    )

    # Calculate distances between feature maps
    distances = yolo_similarity_model(gt_features, mod_features)

    # Get error scores for each image pair in the batch
    matches = match_predictions(gt_predictions, mod_predictions)
    error_scores = torch.as_tensor([m["error_score"] for m in matches]).to(device)

    # Compute loss directly between distances and error scores
    loss = mse_criterion(distances, error_scores)

    # Backpropagation if in training mode
    if training and optimizer is not None:
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    return loss.item(), error_scores.mean().item(), distances.mean().item()


def visualize_batch(batch, save_path="batch_visualization.png"):
    """
    Visualize a batch of image pairs

    Args:
        batch: Batch from dataloader
        save_path: Where to save the visualization
    """
    import matplotlib.pyplot as plt

    gt_batch = batch["gt"]
    modified_batch = batch["modified"]
    names = batch["name"]

    batch_size = 4
    fig, axes = plt.subplots(batch_size, 2, figsize=(10, 5 * batch_size))

    for idx in range(batch_size):
        # Convert tensors to numpy arrays and transpose to HWC format
        gt_img = gt_batch[idx].permute(1, 2, 0).numpy()
        mod_img = modified_batch[idx].permute(1, 2, 0).numpy()

        axes[idx, 0].imshow(gt_img)
        axes[idx, 0].set_title(f"GT: {names[idx]}")
        axes[idx, 1].imshow(mod_img)
        axes[idx, 1].set_title(f"Modified: {names[idx]}")

    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
