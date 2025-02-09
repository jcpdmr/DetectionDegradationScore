import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from pathlib import Path
from tqdm import tqdm
import wandb
from typing import Dict, Optional, List
import math
import os
from itertools import islice

from quality_estimator import create_quality_model, create_baseline_quality_model
from extractor import load_feature_extractor, YOLO11mExtractor


class BinDistributionVisualizer:
    """Class to visualize the distribution of predictions across bins."""

    def __init__(self, n_bins: int = 40, max_score: float = 0.8):
        """
        Initialize the bin distribution visualizer.

        Args:
            n_bins (int): Number of bins to divide the predictions into
            max_score (float): Maximum score value to consider
        """
        self.n_bins = n_bins
        self.max_score = max_score
        self.bin_edges = np.linspace(0, max_score, n_bins + 1)

    def visualize(self, predictions: List[float]) -> None:
        """
        Visualize the distribution of predictions across bins.

        Args:
            predictions (List[float]): List of prediction values
        """
        # Calculate histogram
        counts, _ = np.histogram(predictions, bins=self.bin_edges)
        max_count = max(counts)

        print("\nPrediction Distribution:")
        print("-" * 80)
        
        for bin_ in range(self.n_bins):
            count = counts[bin_]
            bar_length = int((count / max_count) * 50) if max_count > 0 else 0
            print(
                f"Bin {bin_:2d} [{self.bin_edges[bin_]:.3f}-{self.bin_edges[bin_ + 1]:.3f}]: "
                f"{'#' * bar_length} ({count})"
            )
        print("-" * 80)

class WarmupScheduler:
    def __init__(self, optimizer, warmup_epochs, total_epochs, initial_lr):
        self.optimizer = optimizer
        self.warmup_epochs = warmup_epochs
        self.total_epochs = total_epochs
        self.initial_lr = initial_lr
        self.current_epoch = 0

    def step(self):
        self.current_epoch += 1
        if self.current_epoch <= self.warmup_epochs:
            # Linear warmup
            lr_scale = self.current_epoch / max(1, self.warmup_epochs)
        else:
            # Cosine decay after warmpu
            progress = (self.current_epoch - self.warmup_epochs) / max(
                1, self.total_epochs - self.warmup_epochs
            )
            lr_scale = max(0.0, 0.5 * (1.0 + math.cos(math.pi * progress)))

        for param_group in self.optimizer.param_groups:
            param_group["lr"] = (
                self.initial_lr * lr_scale * param_group["initial_lr_scale"]
            )

    def state_dict(self):
        """Save the current state of the scheduler."""
        return {
            "current_epoch": self.current_epoch,
            "warmup_epochs": self.warmup_epochs,
            "total_epochs": self.total_epochs,
            "initial_lr": self.initial_lr,
        }


class QualityTrainer:
    """
    Trainer for the quality assessment model using MSE loss only.
    Implements training techniques including:
    - Monte Carlo dropout for validation
    - Plateau detection and learning rate adaptation
    """

    def __init__(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
        device: torch.device,
        learning_rate: float = 1e-4,
        checkpoint_dir: Optional[str] = None,
        num_epochs: int = 100,
        yolo_weights_path: str = "yolo11m.pt",
        try_run: bool = False,
        use_online_wandb=True,
    ):
        """
        Initialize the trainer with all necessary components.

        Args:
            train_loader: DataLoader for training data
            val_loader: DataLoader for validation data
            device: Device to run training on
            learning_rate: Initial learning rate
            checkpoint_dir: Directory to save checkpoints
            try_run: Whether to run a quick test
        """
        self.device = device
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.try_run = try_run
        self.use_online_wandb = use_online_wandb

        # Initialize model
        # self.model = create_quality_model().to(device)
        self.model = create_baseline_quality_model().to(device)

        # Initialize feature extractor
        self.extractor: YOLO11mExtractor = load_feature_extractor(
            weights_path=yolo_weights_path
        ).to(device)

        # Initialize loss
        # self.loss = nn.MSELoss()
        # self.loss = nn.SmoothL1Loss(beta=0.2)
        self.loss = nn.L1Loss()

        # Setup parameter groups for different learning rates
        # attention_params = []
        # other_params = []
        # for name, param in self.model.named_parameters():
        #     if "cross_attention" in name:
        #         attention_params.append(param)
        #     else:
        #         other_params.append(param)

        # Configure optimizer
        # self.optimizer = optim.AdamW(
        #     [
        #         {
        #             "params": attention_params,
        #             "lr": learning_rate * 0.1,
        #             "initial_lr_scale": 0.1,
        #         },
        #         {
        #             "params": other_params,
        #             "lr": learning_rate,
        #             "initial_lr_scale": 1.0,
        #         },
        #     ],
        #     weight_decay=0.01,
        # )
        # Invece di separare i parametri per attention, mettiamo tutti i parametri nello stesso gruppo
        params = list(self.model.parameters())

        # Configure optimizer mantenendo la stessa struttura di prima
        self.optimizer = optim.AdamW(
            [
                {
                    "params": params,
                    "lr": learning_rate,
                    "initial_lr_scale": 1.0,  # Manteniamo initial_lr_scale per compatibilità
                }
            ],
            weight_decay=0.1,
        )

        # Warmup scheduler, 10% of total epochs
        num_warmup_epochs = int(num_epochs * 0.1)
        self.scheduler = WarmupScheduler(
            self.optimizer,
            warmup_epochs=num_warmup_epochs,
            total_epochs=num_epochs,
            initial_lr=learning_rate,
        )

        # # Setup learning rate scheduler
        # self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        #     self.optimizer, mode="min", factor=0.5, patience=5, verbose=True
        # )

        # Setup checkpointing
        self.checkpoint_dir = Path(checkpoint_dir) if checkpoint_dir else None
        if self.checkpoint_dir:
            self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

        # Training configuration
        self.gradient_clip_value = 1.0
        self.mc_dropout_samples = 5

    def train_epoch(self) -> Dict[str, float]:
        """
        Train for one epoch.

        Returns:
            Dictionary of training metrics
        """
        self.model.train()
        running_loss = 0.0

        # Determine number of batches to run
        num_batches = 5 if self.try_run else len(self.train_loader)

        # Prepare iterator with the first 5 batches if try_run is enabled
        train_iterator = (
            islice(self.train_loader, 5) if self.try_run else self.train_loader
        )

        for i, batch in enumerate(
            tqdm(train_iterator, total=num_batches, desc="Training", ncols=120)
        ):
            # Process batch
            gt = batch["gt"].to(self.device)
            compressed = batch["compressed"].to(self.device)
            # gt_features = batch["gt_features"].to(self.device)
            # mod_features = batch["compressed_features"].to(self.device)
            scores = batch["score"].to(self.device)
            gt_features, mod_features = self.extractor.extract_features(
                img_gt=gt, img_mod=compressed
            )

            # Forward pass
            predictions = self.model(gt_features, mod_features).squeeze()

            # Calculate loss
            loss = self.loss(predictions, scores)

            # Backward pass
            loss.backward()

            # # Gradient clipping
            # torch.nn.utils.clip_grad_norm_(
            #     self.model.parameters(), max_norm=self.gradient_clip_value
            # )

            # Update weights
            self.optimizer.step()
            self.optimizer.zero_grad()

            # Update metrics
            running_loss += loss.item()

        print(f"Train loss: {running_loss / num_batches}")
        # Calculate epoch metrics
        return {"train_loss": running_loss / num_batches}

    @torch.no_grad()
    def validate(self) -> Dict[str, float]:
        """
        Validate model.

        Returns:
            Dictionary of validation metrics
        """
        running_loss = 0.0
        all_preds = []
        all_targets = []

        # Determine number of batches to run
        num_batches = 5 if self.try_run else len(self.val_loader)

        # Prepare iterator with the first 5 batches if try_run is enabled
        val_iterator = islice(self.val_loader, 5) if self.try_run else self.val_loader

        # Set model to eval mode
        self.model.eval()

        # Usa il numero corretto di batch per tqdm
        for batch in tqdm(
            val_iterator, total=num_batches, desc="Validating", ncols=120
        ):
            gt = batch["gt"].to(self.device)
            compressed = batch["compressed"].to(self.device)
            scores = batch["score"].to(self.device)

            gt_features, mod_features = self.extractor.extract_features(
                img_gt=gt, img_mod=compressed
            )

            # Single forward pass
            predictions = self.model(gt_features, mod_features).squeeze()

            # Calculate loss
            loss = self.loss(predictions, scores)
            running_loss += loss.item()

            # Store predictions for analysis
            all_preds.extend(predictions.cpu().numpy())
            all_targets.extend(scores.cpu().numpy())

        visualizer = BinDistributionVisualizer(n_bins=40, max_score=0.8)
        visualizer.visualize(all_preds)

        print(f"Val loss: {running_loss / num_batches}")

        # Calculate metrics
        metrics = {
            "val_loss": running_loss / num_batches,
            "val_mean_pred": np.mean(all_preds),
            "val_std_pred": np.std(all_preds),
        }

        wandb.log(
            {
                "predictions_dist": wandb.Histogram(all_preds),
            }
        )

        return metrics

    def save_checkpoint(
        self, epoch: int, metrics: Dict[str, float], is_best: bool = False
    ) -> None:
        """
        Save model checkpoint.

        Args:
            epoch: Current epoch number
            metrics: Current metrics
            is_best: Whether this is the best model so far
        """
        if not self.checkpoint_dir:
            return

        checkpoint = {
            "epoch": epoch,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scheduler_state_dict": self.scheduler.state_dict(),
            "metrics": metrics,
        }

        # Save regular checkpoint
        path = self.checkpoint_dir / f"checkpoint_epoch_{epoch}.pt"
        torch.save(checkpoint, path)

        # Save best model separately
        if is_best:
            best_path = self.checkpoint_dir / "best_model.pt"
            torch.save(checkpoint, best_path)
            print(f"Saved best model to {best_path}")

    def train(
        self,
        num_epochs: int,
        early_stopping_patience: int = 10,
    ) -> None:
        """
        Complete training loop with validation and early stopping.

        Args:
            num_epochs: Number of epochs to train
            early_stopping_patience: Epochs to wait before early stopping
        """
        wandb.init(
            project="quality-assessment",
            mode="offline" if (self.try_run or not self.use_online_wandb) else "online",
            config={
                "learning_rate": self.optimizer.param_groups[0]["lr"],
                "batch_size": self.train_loader.batch_size,
                "model_type": self.model.__class__.__name__,
                "gradient_clipping": self.gradient_clip_value,
            },
        )

        best_val_loss = float("inf")
        patience_counter = 0

        for epoch in range(num_epochs):
            print(f"\nEpoch {epoch + 1}/{num_epochs}")

            # Training and validation
            train_metrics = self.train_epoch()
            val_metrics = self.validate()

            # Update learning rate scheduler
            self.scheduler.step()

            # Log metrics
            current_lr = self.optimizer.param_groups[0]["lr"]
            wandb.log({"learning_rate": current_lr, **train_metrics, **val_metrics})

            # Early stopping and checkpointing
            if val_metrics["val_loss"] < best_val_loss:
                best_val_loss = val_metrics["val_loss"]
                patience_counter = 0
                self.save_checkpoint(epoch, val_metrics, is_best=True)
            else:
                patience_counter += 1
                self.save_checkpoint(epoch, val_metrics, is_best=False)

            if patience_counter >= early_stopping_patience:
                print(f"\nEarly stopping triggered after {epoch + 1} epochs")
                break

        wandb.finish()


def main():
    """
    Main training script.
    """
    # Configuration
    DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    FEATURES_ROOT = "feature_extracted"
    ERROR_SCORES_ROOT = "balanced_dataset"
    BATCH_SIZE = 230
    NUM_EPOCHS = 30
    LEARNING_RATE = 1e-4
    CHECKPOINT_DIR = "checkpoints/attempt10_40bins_point8_06_visgen_coco17tr_openimagev7traine_320p_qual_20_24_28_32_36_40_50_smooth_2_subsam_444"
    TRY_RUN = False
    USE_ONLINE_WANDB = False

    # Create dataloaders
    # from dataloader import create_feature_dataloaders

    # train_loader, val_loader, _ = create_feature_dataloaders(
    #     features_root=FEATURES_ROOT,
    #     error_scores_root=ERROR_SCORES_ROOT,
    #     batch_size=BATCH_SIZE,
    # )

    from dataloader import create_dataloaders

    train_loader, val_loader, _ = create_dataloaders(
        dataset_root=ERROR_SCORES_ROOT,
        error_scores_root=ERROR_SCORES_ROOT,
        batch_size=BATCH_SIZE,
        num_workers=os.cpu_count(),
    )

    # Initialize and run trainer
    trainer = QualityTrainer(
        train_loader=train_loader,
        val_loader=val_loader,
        device=DEVICE,
        learning_rate=LEARNING_RATE,
        checkpoint_dir=CHECKPOINT_DIR,
        num_epochs=NUM_EPOCHS,
        yolo_weights_path="yolo11m.pt",
        try_run=TRY_RUN,
        use_online_wandb=USE_ONLINE_WANDB,
    )

    trainer.train(num_epochs=NUM_EPOCHS)


if __name__ == "__main__":
    main()
