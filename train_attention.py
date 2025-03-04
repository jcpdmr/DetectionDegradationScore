import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from pathlib import Path
from tqdm import tqdm
import wandb
from typing import Dict, Optional, List
import os
from itertools import islice

from quality_estimator import create_multifeature_baseline_quality_model
from extractor import load_feature_extractor, FeatureExtractor
from backbones import Backbone


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

    def visualize(
        self, predictions: List[float], epoch: int, total_epochs: int, output_file: str
    ) -> None:
        """
        Write prediction distribution to file with epoch information.

        Args:
            predictions (List[float]): List of prediction values
            epoch (int): Current epoch number
            total_epochs (int): Total number of epochs
            output_file (str): Path to output file
        """
        # Calculate histogram
        counts, _ = np.histogram(predictions, bins=self.bin_edges)
        max_count = max(counts)

        # Calculate predictions in range
        in_range = sum(1 for p in predictions if 0 <= p <= self.max_score)
        total_preds = len(predictions)

        with open(output_file, "a") as f:
            f.write(f"Epoch: {epoch}/{total_epochs}")
            f.write(
                f"\nInside Range: {in_range} / {total_preds} ({(in_range / total_preds) * 100:.2f}%)"
            )
            f.write("\nPrediction Distribution\n")
            f.write("-" * 80 + "\n")

            for bin_ in range(self.n_bins):
                count = counts[bin_]
                bar_length = int((count / max_count) * 50) if max_count > 0 else 0
                f.write(
                    f"Bin {bin_:2d} [{self.bin_edges[bin_]:.3f}-{self.bin_edges[bin_ + 1]:.3f}]: "
                    f"{'#' * bar_length} ({count})\n"
                )
            f.write("-" * 80 + "\n\n\n")


class QualityTrainer:
    """
    Trainer for the quality assessment model.
    """

    def __init__(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
        device: torch.device,
        backbone_name: Backbone,
        learning_rate: float = 1e-4,
        checkpoint_dir: Optional[str] = None,
        num_epochs: int = 100,
        yolo_weights_path: str = "yolo11m.pt",
        try_run: bool = False,
        use_online_wandb=True,
        attempt: int = 0,
        batch_size: int = 128,
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
            backbone_name: Name of the backbone model [yolov11m, efficientnet-v2, mobilenet-v3, ...]
        """
        self.device = device
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.try_run = try_run
        self.use_online_wandb = use_online_wandb
        self.attempt = attempt
        self.batch_size = batch_size
        self.total_epochs = num_epochs
        self.current_epoch = 0
        self.backbone_name = backbone_name

        # Initialize feature extractor
        layer_config = self.backbone_name.config
        layer_indices = layer_config.indices
        feature_channels = layer_config.channels
        # For YOLO, weights_path is needed, for others, it's None
        weights_path_extractor = (
            yolo_weights_path if self.backbone_name == Backbone.YOLO_V11_M else None
        )

        # Initialize feature extractor
        self.extractor: FeatureExtractor = load_feature_extractor(
            backbone_name=self.backbone_name,
            weights_path=weights_path_extractor,
        ).to(device)

        # Initialize model
        self.model = create_multifeature_baseline_quality_model(
            feature_channels=feature_channels, layer_indices=layer_indices
        ).to(device)

        # Initialize loss
        self.loss = nn.MSELoss()
        # self.loss = nn.SmoothL1Loss(beta=0.2)
        # self.loss = nn.L1Loss()
        print(f"Loss: {self.loss}")

        params = list(self.model.parameters())
        trainable_params = sum(
            p.numel() for p in self.model.parameters() if p.requires_grad
        )
        total_params = sum(p.numel() for p in self.model.parameters())
        print(f"Trainable Parameters: {trainable_params}")
        print(f"Total Parameters: {total_params}")

        # Configure optimizer mantenendo la stessa struttura di prima
        self.optimizer = optim.AdamW(
            [
                {
                    "params": params,
                    "lr": learning_rate,
                }
            ],
            weight_decay=1e-3,
        )
        print(f"Optimizer: {self.optimizer}")

        # Setup learning rate scheduler
        steps_per_epoch = 50 if try_run else len(train_loader)
        self.scheduler = torch.optim.lr_scheduler.OneCycleLR(
            optimizer=self.optimizer,
            max_lr=learning_rate,  # usa il learning_rate passato al costruttore
            epochs=num_epochs,
            steps_per_epoch=steps_per_epoch,
            pct_start=0.20,  # 20% degli step per arrivare al picco
            div_factor=2,  # Initial LR = max_lr / div_factor
            final_div_factor=1.2,  # Final LR = initial LR / final_div_factor
            three_phase=False,  # usa two-phase policy
            anneal_strategy="cos",  # transizione smooth
        )
        print(f"Scheduler: {self.scheduler}")

        # Setup checkpointing
        self.checkpoint_dir = Path(checkpoint_dir) if checkpoint_dir else None
        if self.checkpoint_dir:
            self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

    def train_epoch(self) -> Dict[str, float]:
        """
        Train for one epoch.

        Returns:
            Dictionary of training metrics
        """
        self.model.train()
        running_loss = 0.0
        all_preds = []

        # Determine number of batches to run
        num_batches = 50 if self.try_run else len(self.train_loader)

        # Prepare iterator with the first 50 batches if try_run is enabled
        train_iterator = (
            islice(self.train_loader, 50) if self.try_run else self.train_loader
        )

        for i, batch in enumerate(
            tqdm(train_iterator, total=num_batches, desc="Training", ncols=120)
        ):
            # Process batch
            gt = batch["gt"].to(self.device)
            compressed = batch["compressed"].to(self.device)
            scores = batch["score"].to(self.device)
            gt_features, mod_features = self.extractor.extract_features(
                img_gt=gt, img_mod=compressed
            )

            # Forward pass
            predictions = self.model(gt_features, mod_features).squeeze()

            # Store predictions for analysis
            all_preds.extend(predictions.detach().cpu().numpy())

            # Calculate loss
            loss = self.loss(predictions, scores)

            # Backward pass
            loss.backward()

            # Update weights
            self.optimizer.step()
            self.scheduler.step()  # Aggiornamento del learning rate dopo ogni batch
            self.optimizer.zero_grad()

            # Update metrics
            running_loss += loss.item()

        visualizer = BinDistributionVisualizer(n_bins=40, max_score=0.8)
        visualizer.visualize(
            predictions=all_preds,
            epoch=self.current_epoch,
            total_epochs=self.total_epochs,
            output_file=f"{self.checkpoint_dir}/train_log.txt",
        )

        wandb.log(
            {
                "train_predictions_dist": wandb.Histogram(all_preds),
            }
        )

        print(f"Train loss: {(running_loss / num_batches):.6f}")
        # Calculate epoch metrics
        return {"train_loss": running_loss / num_batches}

    @torch.no_grad()
    def validate(self, current_epoch, val_log_file) -> Dict[str, float]:
        """
        Validate model.

        Returns:
            Dictionary of validation metrics
        """
        running_loss = 0.0

        all_preds = []
        all_targets = []

        # Determine number of batches to run
        num_batches = 15 if self.try_run else len(self.val_loader)

        # Prepare iterator with the first 15 batches if try_run is enabled
        val_iterator = islice(self.val_loader, 15) if self.try_run else self.val_loader

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
        visualizer.visualize(
            predictions=all_preds,
            epoch=current_epoch,
            total_epochs=self.total_epochs,
            output_file=val_log_file,
        )

        print(f"Val loss: {(running_loss / num_batches):.6f}")

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
        # path = self.checkpoint_dir / f"checkpoint_epoch_{epoch}.pt"
        # torch.save(checkpoint, path)

        # Save best model separately
        if is_best:
            best_path = self.checkpoint_dir / "best_model.pt"
            torch.save(checkpoint, best_path)
            print(f"Saved best model to {best_path}")

    def train(
        self,
        num_epochs: int,
        early_stopping_patience: int = 15,
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
            name=f"attempt{self.attempt}",
            config={
                "learning_rate": self.optimizer.param_groups[0]["lr"],
                "batch_size": self.train_loader.batch_size,
                "model_type": self.model.__class__.__name__,
            },
        )

        trainable_params = sum(
            p.numel() for p in self.model.parameters() if p.requires_grad
        )
        total_params = sum(p.numel() for p in self.model.parameters())
        print(f"Trainable Parameters: {trainable_params}")
        print(f"Total Parameters: {total_params}")
        wandb.log(
            {"TrainableParameters": trainable_params, "TotalParameters": total_params}
        )
        wandb.log({"LossObject": self.loss})
        wandb.log({"Batch Size": self.batch_size})

        best_val_loss = float("inf")
        patience_counter = 0

        for epoch in range(num_epochs):
            self.current_epoch = epoch + 1
            current_lr = self.optimizer.param_groups[0]["lr"]
            print(f"\nEpoch {epoch + 1}/{num_epochs},  Learning Rate: {current_lr:.9f}")

            # Training and validation
            train_metrics = self.train_epoch()
            val_metrics = self.validate(
                current_epoch=epoch + 1,
                val_log_file=f"{self.checkpoint_dir}/val_log.txt",
            )

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
                # self.save_checkpoint(epoch, val_metrics, is_best=False)

            if patience_counter >= early_stopping_patience:
                print(f"\nEarly stopping triggered after {epoch + 1} epochs")
                break

        wandb.finish()


def main():
    """
    Main training script.
    """
    # Configuration
    GPU_ID = 1
    DEVICE = torch.device(f"cuda:{GPU_ID}" if torch.cuda.is_available() else "cpu")
    ERROR_SCORES_ROOT = "balanced_dataset_coco2017"
    BATCH_SIZE = 210
    NUM_EPOCHS = 50
    LEARNING_RATE = 1e-3
    ATTEMPT = 35
    DIR = "07_coco17complete_320p_qual_20_25_30_35_40_45_50_subsamp_444"
    CHECKPOINT_DIR = f"checkpoints/attempt{ATTEMPT}_40bins_point8_{DIR}"
    TRY_RUN = False
    USE_ONLINE_WANDB = True
    BACKBONE = Backbone.YOLO_V11_M

    from dataloader import create_dataloaders

    train_loader, val_loader, _ = create_dataloaders(
        dataset_root=ERROR_SCORES_ROOT,
        error_scores_root=ERROR_SCORES_ROOT,
        batch_size=BATCH_SIZE,
        num_workers=os.cpu_count(),
        backbone_name=BACKBONE,
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
        attempt=ATTEMPT,
        batch_size=BATCH_SIZE,
        backbone_name=BACKBONE,
    )

    trainer.train(num_epochs=NUM_EPOCHS)


if __name__ == "__main__":
    main()
