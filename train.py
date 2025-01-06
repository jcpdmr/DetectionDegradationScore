from ultralytics import YOLO
import os
import time
import torch
import math
import torch.nn as nn
import numpy as np
from tqdm import tqdm
from patches_loader import create_dataloaders
from typing import Tuple, Dict, Optional, List
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
    modification_types: Optional[List[str]] = None,
):
    """
    Train the perceptual loss model with validation and early stopping

    Args:
        modification_types: List of modification types to train on. If None, uses all available types.
    """
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)

    timestamp = time.strftime("%Y%m%d-%H%M%S")

    # Add modification type to the output directory name if specific types are selected
    if modification_types and len(modification_types) == 1:
        timestamp += f"_{modification_types[0]}"

    # Create run-specific output directory
    run_output_dir = os.path.join(output_dir, timestamp)
    os.makedirs(run_output_dir, exist_ok=True)

    # Create log files with timestamp
    train_log_path = os.path.join(run_output_dir, f"log_{timestamp}_trn.csv")
    val_log_path = os.path.join(run_output_dir, f"log_{timestamp}_val.csv")
    weights_log_path = os.path.join(run_output_dir, f"log_{timestamp}_weights.csv")
    lr_log_path = os.path.join(run_output_dir, f"log_{timestamp}_lr.csv")

    with open(train_log_path, "w") as log_file:
        # Write header
        log_file.write("epoch,loss,avg_error_score,avg_distance\n")
    with open(val_log_path, "w") as log_file:
        # Write header
        log_file.write("epoch,loss,avg_error_score,avg_distance\n")
    with open(weights_log_path, "w") as log_file:
        # Weights header
        header = [
            "epoch",
            "layer_weight_raw_02",
            "layer_weight_raw_09",
            "layer_weight_raw_16",
            "layer_weight_softmax_02",
            "layer_weight_softmax_09",
            "layer_weight_softmax_16",
            "pool_02_weight_sigmoid",
            "pool_09_weight_sigmoid",
            "pool_16_weight_sigmoid",
        ]

        # conv1 and final_conv statistics
        for layer_name in ["02", "09", "16"]:
            header.extend(
                [
                    f"conv1_{layer_name}_min",
                    f"conv1_{layer_name}_max",
                    f"conv1_{layer_name}_mean",
                    f"conv1_{layer_name}_std",
                    f"final_conv_{layer_name}_min",
                    f"final_conv_{layer_name}_max",
                    f"final_conv_{layer_name}_mean",
                    f"final_conv_{layer_name}_std",
                ]
            )

        # Write header with csv join to avoid extra commas
        log_file.write(",".join(header) + "\n")
    with open(lr_log_path, "w") as log_file:  # New LR log initialization
        log_file.write("epoch,learning_rate\n")

    # Initialize models
    yolo_model.eval()  # Freeze YOLO weights
    yolo_similarity_model = YOLOSimilarity()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if torch.cuda.is_available():
        yolo_model = yolo_model.cuda()
        yolo_similarity_model = yolo_similarity_model.cuda()

    # Create dataloaders
    train_loader, val_loader, _ = create_dataloaders(
        data_path,
        batch_size,
        num_workers=os.cpu_count(),
        # seed=seed,
        modification_types=modification_types,
    )

    # Log the training configuration
    with open(os.path.join(run_output_dir, "training_config.txt"), "w") as f:
        f.write("Training Configuration\n")
        f.write("=" * 50 + "\n\n")
        f.write(f"Timestamp: {timestamp}\n")
        f.write(f"Modification types: {modification_types or 'all'}\n")
        f.write(f"Batch size: {batch_size}\n")
        f.write(f"Learning rate: {learning_rate}\n")
        f.write(f"Number of epochs: {num_epochs}\n")
        f.write(f"Validation frequency: {val_frequency}\n")
        f.write(f"Early stopping patience: {patience}\n")

    # Calculate warmup steps and total steps
    num_warmup_epochs = int(num_epochs * 0.1)  # 10% of total epochs for warmup

    # Setup optimizer with weight decay separation
    no_decay = ["bias", "LayerNorm.weight"]
    optimizer_grouped_parameters = [
        {
            "params": [
                p
                for n, p in yolo_similarity_model.named_parameters()
                if not any(nd in n for nd in no_decay)
            ],
            "weight_decay": 0.01,
        },
        {
            "params": [
                p
                for n, p in yolo_similarity_model.named_parameters()
                if any(nd in n for nd in no_decay)
            ],
            "weight_decay": 0.0,
        },
    ]

    optimizer = torch.optim.AdamW(
        optimizer_grouped_parameters, lr=learning_rate, betas=(0.9, 0.999), eps=1e-8
    )

    # Learning rate scheduler with warmup
    class WarmupCosineSchedule:
        def __init__(self, optimizer, warmup_epochs, total_epochs):
            self.optimizer = optimizer
            self.warmup_epochs = warmup_epochs
            self.total_epochs = total_epochs
            self.current_epoch = 0

        def step(self):
            self.current_epoch += 1
            if self.current_epoch <= self.warmup_epochs:
                # Linear warmup
                lr_scale = self.current_epoch / max(1, self.warmup_epochs)
            else:
                # Cosine decay
                progress = (self.current_epoch - self.warmup_epochs) / max(
                    1, self.total_epochs - self.warmup_epochs
                )
                lr_scale = max(0.0, 0.5 * (1.0 + math.cos(math.pi * progress)))

            # Update learning rate
            for param_group in self.optimizer.param_groups:
                param_group["lr"] = learning_rate * lr_scale

        def get_last_lr(self):
            return [group["lr"] for group in self.optimizer.param_groups]

    scheduler = WarmupCosineSchedule(optimizer, num_warmup_epochs, num_epochs)

    # Setup training
    loss_criterion = nn.MSELoss()
    # loss_criterion = nn.L1Loss()
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
                loss_criterion,
                device,
                optimizer,
                training=True,
            )
            train_losses.append(loss)
            train_mean_error_score.append(error_score)
            train_distances.append(distance)
        # Step the scheduler after each epoch
        scheduler.step()

        # Log the learning rate
        current_lr = scheduler.get_last_lr()[0]
        with open(lr_log_path, "a") as log_file:
            log_file.write(f"{epoch+1},{current_lr:.8f}\n")

        # Log training metrics
        avg_train_metrics = {
            "loss": sum(train_losses) / len(train_losses),
            "mean_error_score": sum(train_mean_error_score)
            / len(train_mean_error_score),
            "distance": sum(train_distances) / len(train_distances),
        }

        with open(train_log_path, "a") as log_file:
            log_file.write(
                f"{epoch+1},{avg_train_metrics['loss']:.6f},"
                f"{avg_train_metrics['mean_error_score']:.6f},{avg_train_metrics['distance']:.6f}\n"
            )

        with open(weights_log_path, "a") as log_file:
            log_weights_and_stats(log_file, epoch, yolo_similarity_model)

        # Validation phase if needed
        if (epoch + 1) % val_frequency == 0:
            yolo_similarity_model.eval()
            val_losses = []
            val_mean_error_score = []
            val_distances = []

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
                        loss_criterion,
                        device,
                        training=False,
                    )
                    val_losses.append(loss)
                    val_mean_error_score.append(error_score)
                    val_distances.append(distance)

            # Log validation metrics
            avg_val_metrics = {
                "loss": sum(val_losses) / len(val_losses),
                "mean_error_score": sum(val_mean_error_score)
                / len(val_mean_error_score),
                "distance": sum(val_distances) / len(val_distances),
            }

            with open(val_log_path, "a") as log_file:
                log_file.write(
                    f"{epoch+1},{avg_val_metrics['loss']:.6f},"
                    f"{avg_val_metrics['mean_error_score']:.6f},{avg_val_metrics['distance']:.6f}\n"
                )

            # Print current metrics
            print(f"\nEpoch {epoch+1}/{num_epochs}")
            print(
                f"Train - Loss: {avg_train_metrics['loss']:.4f}, Error Score: {avg_train_metrics['mean_error_score']:.4f}, "
            )
            print(
                f"Val   - Loss: {avg_val_metrics['loss']:.4f}, Error Score: {avg_val_metrics['mean_error_score']:.4f}, "
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
            print(f"Learning Rate: {current_lr:.8f}")
            print(
                f"Train - Loss: {avg_train_metrics['loss']:.4f}, Error Score: {avg_train_metrics['mean_error_score']:.4f}"
            )
    return {
        "model_path": os.path.join(run_output_dir, "best_model.pth"),
        "output_dir": run_output_dir,
        "layer_configs": layer_configs,
        "modification_types": modification_types,
    }


def process_batch(
    yolo_model: YOLO,
    yolo_similarity_model: YOLOSimilarity,
    batch: Dict[str, torch.Tensor],
    layer_configs: list,
    loss_criterion: nn.Module,
    device: torch.device,
    optimizer: torch.optim.Optimizer = None,
    training: bool = True,
) -> Tuple[float, float, float]:
    """
    Process a single batch (training or validation)
    Returns loss, error_score and distance for monitoring
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
    loss = loss_criterion(distances, error_scores)

    # Backpropagation if in training mode
    if training and optimizer is not None:
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    return loss.item(), error_scores.mean().item(), distances.mean().item()


def test_perceptual_loss(
    yolo_model: YOLO,
    model_path: str,
    data_path: str,
    batch_size: int,
    output_dir: str,
    layer_configs: list,
    manual_test: bool = False,
    modification_types: Optional[List[str]] = None,
):
    """
    Test the trained perceptual loss model on the test set.

    Args:
        yolo_model: Base YOLO model for feature extraction
        model_path: Path to the trained model weights
        data_path: Path to dataset root directory
        batch_size: Batch size for testing
        output_dir: Directory to save test results
        layer_configs: Layer configurations for feature extraction
        manual_test: If True, saves results with '_manual' suffix to differentiate from automatic test results
        modification_types: List of modification types to test on. If None, uses all available types.
    """
    # Initialize models
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    yolo_model.eval()
    yolo_similarity_model = YOLOSimilarity().to(device)

    # Load trained weights
    checkpoint = torch.load(model_path)
    yolo_similarity_model.load_state_dict(checkpoint["model_state_dict"])
    yolo_similarity_model.eval()

    # Create test dataloader
    _, _, test_loader = create_dataloaders(
        data_path,
        batch_size,
        num_workers=os.cpu_count(),
        modification_types=modification_types,
    )

    # Lists for storing results
    similarities = []
    error_scores = []
    image_pairs = []
    mod_types = []

    # Test loop
    with torch.no_grad():
        test_progress = tqdm(test_loader, desc="Testing")
        for batch in test_progress:
            gt_batch = batch["gt"].to(device)
            modified_batch = batch["modified"].to(device)

            # Extract features and predictions
            gt_features, gt_predictions = extract_multiple_features_and_predictions(
                yolo_model, gt_batch, layer_configs
            )
            mod_features, mod_predictions = extract_multiple_features_and_predictions(
                yolo_model, modified_batch, layer_configs
            )

            # Calculate similarity scores
            batch_similarities = yolo_similarity_model(gt_features, mod_features)

            # Calculate error scores
            matches = match_predictions(gt_predictions, mod_predictions)
            batch_error_scores = torch.tensor([m["error_score"] for m in matches]).to(
                device
            )

            # Store results
            similarities.extend(batch_similarities.cpu().numpy())
            error_scores.extend(batch_error_scores.cpu().numpy())
            image_pairs.extend(zip(batch["name"], batch["name"]))
            mod_types.extend(batch["mod_type"])

    # Initialize the suffix as empty string
    suffix = ""

    # Add manual suffix if this is a manual test
    if manual_test:
        suffix += "_manual"

    # Add modification type to suffix if we're testing a specific type
    if modification_types and len(modification_types) == 1:
        suffix += f"_{modification_types[0]}"

    # If no suffix was added (not manual and no specific modification),
    # use the default filenames
    if suffix:
        results_filename = f"test_results{suffix}.csv"
        stats_filename = f"test_statistics{suffix}.txt"
    else:
        results_filename = "test_results.csv"
        stats_filename = "test_statistics.txt"

    # Save detailed results
    results_path = os.path.join(output_dir, results_filename)
    with open(results_path, "w") as f:
        f.write("image_name,modification_type,similarity_score,error_score\n")
        for (img_name, _), mod_type, sim, err in zip(
            image_pairs, mod_types, similarities, error_scores
        ):
            f.write(f"{img_name},{mod_type},{sim:.6f},{err:.6f}\n")

    # Calculate and save statistics
    stats_path = os.path.join(output_dir, stats_filename)
    with open(stats_path, "w") as f:
        f.write("Test Statistics\n")
        f.write("=" * 50 + "\n\n")

        # Overall statistics
        correlation = np.corrcoef(similarities, error_scores)[0, 1]
        mse = np.mean((np.array(similarities) - np.array(error_scores)) ** 2)
        # mae = np.mean(np.abs(np.array(similarities) - np.array(error_scores)))

        f.write("Overall Statistics:\n")
        f.write(f"Total images tested: {len(similarities)}\n")
        f.write(f"Correlation coefficient: {correlation:.4f}\n")
        f.write(f"MSE(similarites-error scores): {mse:.4f}\n\n")

        # Per-modification type statistics
        f.write("Statistics by Modification Type:\n")
        for mod_type in set(mod_types):
            mask = np.array(mod_types) == mod_type
            mod_sims = np.array(similarities)[mask]
            mod_errs = np.array(error_scores)[mask]

            # mod_mae = np.mean(np.abs(mod_sims - mod_errs))
            mod_mse = np.mean((mod_sims - mod_errs) ** 2)

            f.write(f"\n{mod_type.upper()}:\n")
            f.write(f"Count: {len(mod_sims)}\n")
            f.write(f"Mean similarity_score: {np.mean(mod_sims):.4f}\n")
            f.write(f"Mean error_score: {np.mean(mod_errs):.4f}\n")
            f.write(
                f"Correlation coefficient: {np.corrcoef(mod_sims, mod_errs)[0,1]:.4f}\n"
            )
            f.write(f"MSE(similarites-error scores): {mod_mse:.4f}\n")

    return {
        "similarities": similarities,
        "error_scores": error_scores,
        "correlation": correlation,
        "mse": mse,
    }


def log_conv_stats(conv_layer):
    weights = conv_layer.weight.detach().cpu().numpy()
    return {
        "min": weights.min(),
        "max": weights.max(),
        "mean": weights.mean(),
        "std": weights.std(),
    }


def log_weights_and_stats(log_file, epoch, yolo_similarity_model):
    """
    Log weights and statistics to CSV, ensuring proper CSV formatting.
    Each line should end without a comma.
    """
    # Prepare all values in a list first
    values = []

    # Add epoch
    values.append(f"{epoch+1}")

    # Raw layer weights
    feature_weights = yolo_similarity_model.layer_weights.detach().cpu().numpy()
    values.extend([f"{w:.6f}" for w in feature_weights])

    # Normalized weights
    normalized_weights = feature_weights / feature_weights.sum()
    values.extend([f"{w:.6f}" for w in normalized_weights])

    # Sigmoid pooling weights
    pool_weights = [
        torch.sigmoid(yolo_similarity_model.pool_02.weight).item(),
        torch.sigmoid(yolo_similarity_model.pool_09.weight).item(),
        torch.sigmoid(yolo_similarity_model.pool_16.weight).item(),
    ]
    values.extend([f"{w:.6f}" for w in pool_weights])

    # Conv layers stats
    for process_block, layer_name in [
        (yolo_similarity_model.process_02, "02"),
        (yolo_similarity_model.process_09, "09"),
        (yolo_similarity_model.process_16, "16"),
    ]:
        # conv1
        conv1_stats = log_conv_stats(process_block.conv1)
        values.extend(
            [
                f"{conv1_stats['min']:.6f}",
                f"{conv1_stats['max']:.6f}",
                f"{conv1_stats['mean']:.6f}",
                f"{conv1_stats['std']:.6f}",
            ]
        )

        # final_conv
        final_conv_stats = log_conv_stats(process_block.final_conv)
        values.extend(
            [
                f"{final_conv_stats['min']:.6f}",
                f"{final_conv_stats['max']:.6f}",
                f"{final_conv_stats['mean']:.6f}",
                f"{final_conv_stats['std']:.6f}",
            ]
        )

    # Join all values with commas and write the line
    log_file.write(",".join(values) + "\n")


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
