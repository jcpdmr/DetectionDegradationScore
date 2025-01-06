import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from pathlib import Path
from tqdm import tqdm
import wandb
from typing import Dict, Optional
import numpy as np
from torchmetrics.regression import PearsonCorrCoef

from quality_estimator import create_quality_model


class QualityTrainer:
    """
    Trainer class for the quality assessment model. Handles the complete training process
    including model initialization, optimization, logging, and evaluation.
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
            learning_rate: Initial learning rate for optimization
            checkpoint_dir: Directory to save model checkpoints
        """
        self.device = device
        self.train_loader = train_loader
        self.val_loader = val_loader

        # Initialize model and move to device
        self.model = create_quality_model()
        self.model.to(device)

        # Setup optimization
        self.optimizer = optim.AdamW(
            self.model.parameters(),
            lr=learning_rate,
            weight_decay=0.01,  # L2 regularization to prevent overfitting
        )

        # Learning rate scheduler for better convergence
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode="min", factor=0.5, patience=5, verbose=True
        )

        # Loss function combines MSE for regression and a correlation term
        self.mse_loss = nn.MSELoss()
        self.pearson = PearsonCorrCoef().to(device)

        # Setup checkpoint directory
        self.checkpoint_dir = Path(checkpoint_dir) if checkpoint_dir else None
        if self.checkpoint_dir:
            self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

    def correlation_loss(
        self, pred: torch.Tensor, target: torch.Tensor
    ) -> torch.Tensor:
        """
        Calculate correlation loss using torchmetrics' PearsonCorrCoef.
        Returns 1 - correlation for minimization.

        Args:
            pred: Predicted scores [B]
            target: Target scores [B]

        Returns:
            Correlation loss (1 - correlation coefficient)
        """
        correlation = self.pearson(pred, target)
        return 1 - correlation.abs()  # Using abs to handle negative correlations

    def train_epoch(self) -> Dict[str, float]:
        """
        Train the model for one epoch.

        Returns:
            Dictionary containing training metrics
        """
        self.model.train()
        running_loss = 0.0
        running_mse = 0.0
        running_corr = 0.0

        # Process all batches with progress tracking
        for batch in tqdm(self.train_loader, desc="Training"):
            # Move data to device
            gt_features = batch["gt_features"].to(self.device)
            mod_features = batch["mod_features"].to(self.device)
            scores = batch["score"].to(self.device)

            # Forward pass
            self.optimizer.zero_grad()
            predictions = self.model(gt_features, mod_features).squeeze()

            # Calculate losses
            mse_loss = self.mse_loss(predictions, scores)
            corr_loss = self.correlation_loss(predictions, scores)
            total_loss = mse_loss + 0.5 * corr_loss  # Weighted combination

            # Backward pass and optimization
            total_loss.backward()
            # Gradient clipping to prevent exploding gradients
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()

            # Update running metrics
            running_loss += total_loss.item()
            running_mse += mse_loss.item()
            running_corr += corr_loss.item()

        # Calculate epoch metrics
        num_batches = len(self.train_loader)
        metrics = {
            "train_loss": running_loss / num_batches,
            "train_mse": running_mse / num_batches,
            "train_correlation_loss": running_corr / num_batches,
        }
        return metrics

    @torch.no_grad()
    def validate(self) -> Dict[str, float]:
        """
        Evaluate the model on validation data.

        Returns:
            Dictionary containing validation metrics
        """
        self.model.eval()
        running_loss = 0.0
        running_mse = 0.0
        running_corr = 0.0
        all_preds = []
        all_targets = []

        for batch in tqdm(self.val_loader, desc="Validating"):
            # Move data to device
            gt_features = batch["gt_features"].to(self.device)
            mod_features = batch["mod_features"].to(self.device)
            scores = batch["score"].to(self.device)

            # Forward pass
            predictions = self.model(gt_features, mod_features).squeeze()

            # Calculate losses
            mse_loss = self.mse_loss(predictions, scores)
            corr_loss = self.correlation_loss(predictions, scores)
            total_loss = mse_loss + 0.5 * corr_loss

            # Update running metrics
            running_loss += total_loss.item()
            running_mse += mse_loss.item()
            running_corr += corr_loss.item()

            # Store predictions for overall correlation
            all_preds.extend(predictions.cpu().numpy())
            all_targets.extend(scores.cpu().numpy())

        # Calculate overall correlation
        overall_corr = np.corrcoef(all_preds, all_targets)[0, 1]

        # Calculate validation metrics
        num_batches = len(self.val_loader)
        metrics = {
            "val_loss": running_loss / num_batches,
            "val_mse": running_mse / num_batches,
            "val_correlation_loss": running_corr / num_batches,
            "val_pearson_correlation": overall_corr,
        }
        return metrics

    def save_checkpoint(self, epoch: int, metrics: Dict[str, float]) -> None:
        """
        Save model checkpoint including state dict and metrics.

        Args:
            epoch: Current epoch number
            metrics: Dictionary of current metrics
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

        path = self.checkpoint_dir / f"checkpoint_epoch_{epoch}.pt"
        torch.save(checkpoint, path)
        print(f"Saved checkpoint to {path}")

    def train(self, num_epochs: int, early_stopping_patience: int = 10) -> None:
        """
        Complete training loop with validation and early stopping.

        Args:
            num_epochs: Number of epochs to train for
            early_stopping_patience: Number of epochs to wait for improvement
        """
        # Initialize wandb for experiment tracking
        wandb.init(
            project="quality-assessment",
            config={
                "learning_rate": self.optimizer.param_groups[0]["lr"],
                "batch_size": self.train_loader.batch_size,
                "model_type": self.model.__class__.__name__,
            },
        )

        best_val_loss = float("inf")
        patience_counter = 0

        for epoch in range(num_epochs):
            print(f"\nEpoch {epoch+1}/{num_epochs}")

            # Training phase
            train_metrics = self.train_epoch()

            # Validation phase
            val_metrics = self.validate()

            # Update learning rate scheduler
            self.scheduler.step(val_metrics["val_loss"])

            # Log metrics
            wandb.log({**train_metrics, **val_metrics})

            # Save checkpoint if validation loss improved
            if val_metrics["val_loss"] < best_val_loss:
                best_val_loss = val_metrics["val_loss"]
                patience_counter = 0
                self.save_checkpoint(epoch, val_metrics)
            else:
                patience_counter += 1

            # Early stopping check
            if patience_counter >= early_stopping_patience:
                print(f"\nEarly stopping triggered after {epoch+1} epochs")
                break

        wandb.finish()


def main():
    """
    Main training script.
    """
    # Configuration
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    FEATURES_ROOT = "feature_extracted_attention"
    BATCH_SIZE = 128
    NUM_EPOCHS = 100
    LEARNING_RATE = 1e-4
    CHECKPOINT_DIR = "checkpoints"

    # Create dataloaders
    from features_and_scores_loader import create_feature_dataloaders

    train_loader, val_loader, _ = create_feature_dataloaders(
        features_root=FEATURES_ROOT, batch_size=BATCH_SIZE
    )

    # Initialize trainer
    trainer = QualityTrainer(
        train_loader=train_loader,
        val_loader=val_loader,
        device=DEVICE,
        learning_rate=LEARNING_RATE,
        checkpoint_dir=CHECKPOINT_DIR,
    )

    # Start training
    trainer.train(num_epochs=NUM_EPOCHS)


if __name__ == "__main__":
    main()
