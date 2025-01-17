import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from pathlib import Path
from tqdm import tqdm
import wandb
from typing import Dict, Optional

from quality_estimator import create_quality_model


class QualityTrainer:
    """
    Trainer for the quality assessment model using MSE loss only.
    Implements training techniques including:
    - Gradient accumulation
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
    ):
        """
        Initialize the trainer with all necessary components.

        Args:
            train_loader: DataLoader for training data
            val_loader: DataLoader for validation data
            device: Device to run training on
            learning_rate: Initial learning rate
            checkpoint_dir: Directory to save checkpoints
        """
        self.device = device
        self.train_loader = train_loader
        self.val_loader = val_loader

        # Initialize model
        self.model = create_quality_model().to(device)

        # Initialize loss
        self.mae_loss = nn.L1Loss()

        # Setup parameter groups for different learning rates
        attention_params = []
        other_params = []
        for name, param in self.model.named_parameters():
            if "cross_attention" in name:
                attention_params.append(param)
            else:
                other_params.append(param)

        # Configure optimizer
        self.optimizer = optim.AdamW(
            [
                {"params": attention_params, "lr": learning_rate * 0.1},
                {"params": other_params, "lr": learning_rate},
            ],
            weight_decay=0.01,
        )

        # Setup learning rate scheduler
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode="min", factor=0.5, patience=5, verbose=True
        )

        # Setup checkpointing
        self.checkpoint_dir = Path(checkpoint_dir) if checkpoint_dir else None
        if self.checkpoint_dir:
            self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

        # Training configuration
        self.accumulation_steps = 4
        self.gradient_clip_value = 1.0
        self.mc_dropout_samples = 5

    def train_epoch(self) -> Dict[str, float]:
        """
        Train for one epoch with gradient accumulation.

        Returns:
            Dictionary of training metrics
        """
        self.model.train()
        running_loss = 0.0

        for idx, batch in enumerate(tqdm(self.train_loader, desc="Training")):
            # Process batch
            gt_features = batch["gt_features"].to(self.device)
            mod_features = batch["compressed_features"].to(self.device)
            scores = batch["score"].to(self.device)

            # Forward pass
            predictions = self.model(gt_features, mod_features).squeeze()

            # Calculate MSE loss
            loss = self.mae_loss(predictions, scores)

            # Scale loss for accumulation
            loss = loss / self.accumulation_steps

            # Backward pass with gradient accumulation
            loss.backward()

            # # Gradient clipping
            # torch.nn.utils.clip_grad_norm_(
            #     self.model.parameters(), max_norm=self.gradient_clip_value
            # )

            # Update weights after accumulation
            if (idx + 1) % self.accumulation_steps == 0:
                self.optimizer.step()
                self.optimizer.zero_grad()

            # Update metrics
            running_loss += loss.item() * self.accumulation_steps

        # Calculate epoch metrics
        return {"train_loss": running_loss / len(self.train_loader)}

    @torch.no_grad()
    def validate(self) -> Dict[str, float]:
        """
        Validate model with Monte Carlo dropout averaging.

        Returns:
            Dictionary of validation metrics
        """
        running_loss = 0.0
        all_preds = []
        all_targets = []

        for batch in tqdm(self.val_loader, desc="Validating"):
            gt_features = batch["gt_features"].to(self.device)
            mod_features = batch["compressed_features"].to(self.device)
            scores = batch["score"].to(self.device)

            # Monte Carlo dropout predictions
            predictions = torch.zeros_like(scores)
            for _ in range(self.mc_dropout_samples):
                self.model.train()  # Enable dropout
                predictions += self.model(gt_features, mod_features).squeeze()
            predictions /= self.mc_dropout_samples

            # Calculate MSE loss
            loss = self.mae_loss(predictions, scores)
            running_loss += loss.item()

            # Store predictions for analysis
            all_preds.extend(predictions.cpu().numpy())
            all_targets.extend(scores.cpu().numpy())

        # Calculate metrics
        metrics = {
            "val_loss": running_loss / len(self.val_loader),
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
            config={
                "learning_rate": self.optimizer.param_groups[0]["lr"],
                "batch_size": self.train_loader.batch_size,
                "model_type": self.model.__class__.__name__,
                "accumulation_steps": self.accumulation_steps,
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
            self.scheduler.step(val_metrics["val_loss"])

            # Log metrics
            wandb.log({**train_metrics, **val_metrics})

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
    DEVICE = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
    FEATURES_ROOT = "feature_extracted"
    ERROR_SCORES_ROOT = "balanced_dataset"
    BATCH_SIZE = 128
    NUM_EPOCHS = 100
    LEARNING_RATE = 5e-5
    CHECKPOINT_DIR = "checkpoints"

    # Create dataloaders
    from dataloader import create_feature_dataloaders

    train_loader, val_loader, _ = create_feature_dataloaders(
        features_root=FEATURES_ROOT,
        error_scores_root=ERROR_SCORES_ROOT,
        batch_size=BATCH_SIZE,
    )

    # Initialize and run trainer
    trainer = QualityTrainer(
        train_loader=train_loader,
        val_loader=val_loader,
        device=DEVICE,
        learning_rate=LEARNING_RATE,
        checkpoint_dir=CHECKPOINT_DIR,
    )

    trainer.train(num_epochs=NUM_EPOCHS)


if __name__ == "__main__":
    main()
