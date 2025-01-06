from collections import defaultdict
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from pathlib import Path
from tqdm import tqdm
import wandb
from typing import Dict, Optional, List
from torchmetrics.regression import PearsonCorrCoef

from quality_estimator import create_quality_model


class QualityTrainer:
    """
    Comprehensive trainer for the quality assessment model.
    Implements advanced training techniques including:
    - Cyclical learning rates
    - Gradient accumulation
    - Dynamic loss weighting
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

        # Initialize metrics
        self.pearson = PearsonCorrCoef().to(device)
        self.mse_loss = nn.MSELoss()

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
        self.scheduler = self._create_cyclic_scheduler(
            base_lr=learning_rate * 0.1,
            max_lr=learning_rate,
            step_size=4 * len(train_loader),
        )

        # Setup checkpointing
        self.checkpoint_dir = Path(checkpoint_dir) if checkpoint_dir else None
        if self.checkpoint_dir:
            self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

        # Training configuration
        self.accumulation_steps = 4
        self.gradient_clip_value = 1.0
        self.mc_dropout_samples = 5

    def correlation_loss(
        self, pred: torch.Tensor, target: torch.Tensor
    ) -> torch.Tensor:
        """
        Calculate correlation-based loss component.

        Args:
            pred: Predicted scores
            target: Target scores

        Returns:
            Correlation loss value
        """
        pred = pred.view(-1)
        target = target.view(-1)
        correlation = self.pearson(pred, target)
        return 1 - correlation.abs()

    def _create_cyclic_scheduler(
        self, base_lr: float, max_lr: float, step_size: int
    ) -> torch.optim.lr_scheduler.CyclicLR:
        """
        Create cyclical learning rate scheduler.

        Args:
            base_lr: Minimum learning rate
            max_lr: Maximum learning rate
            step_size: Steps per half cycle

        Returns:
            Configured CyclicLR scheduler
        """
        return torch.optim.lr_scheduler.CyclicLR(
            self.optimizer,
            base_lr=base_lr,
            max_lr=max_lr,
            step_size_up=step_size // 2,
            mode="triangular2",
            cycle_momentum=False,
        )

    def _calculate_dynamic_weight(self, batch_idx: int, total_batches: int) -> float:
        """
        Calculate dynamic weight for loss components based on training progress.

        Args:
            batch_idx: Current batch index
            total_batches: Total number of batches

        Returns:
            Weight factor between 0 and 1
        """
        return 0.5 * (1 + np.cos(np.pi * (batch_idx / total_batches)))

    def train_epoch(self) -> Dict[str, float]:
        """
        Train for one epoch with gradient accumulation and dynamic loss weighting.

        Returns:
            Dictionary of training metrics
        """
        self.model.train()
        running_metrics = defaultdict(float)

        for idx, batch in enumerate(tqdm(self.train_loader, desc="Training")):
            # Process batch
            gt_features = batch["gt_features"].to(self.device)
            mod_features = batch["mod_features"].to(self.device)
            scores = batch["score"].to(self.device)

            # Forward pass
            predictions = self.model(gt_features, mod_features).squeeze()

            # Calculate losses
            mse_loss = self.mse_loss(predictions, scores)
            corr_loss = self.correlation_loss(predictions, scores)

            # Dynamic loss weighting
            alpha = self._calculate_dynamic_weight(idx, len(self.train_loader))
            total_loss = mse_loss + alpha * corr_loss

            # Scale loss for accumulation
            total_loss = total_loss / self.accumulation_steps

            # Backward pass with gradient accumulation
            total_loss.backward()

            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(
                self.model.parameters(), max_norm=self.gradient_clip_value
            )

            # Update weights after accumulation
            if (idx + 1) % self.accumulation_steps == 0:
                self.optimizer.step()
                self.scheduler.step()
                self.optimizer.zero_grad()

            # Update metrics
            running_metrics["train_loss"] += total_loss.item() * self.accumulation_steps
            running_metrics["train_mse"] += mse_loss.item()
            running_metrics["train_correlation_loss"] += corr_loss.item()

        # Calculate epoch metrics
        num_batches = len(self.train_loader)
        return {k: v / num_batches for k, v in running_metrics.items()}

    @torch.no_grad()
    def validate(self) -> Dict[str, float]:
        """
        Validate model with Monte Carlo dropout averaging.

        Returns:
            Dictionary of validation metrics
        """
        running_metrics = defaultdict(float)
        all_preds = []
        all_targets = []

        for batch in tqdm(self.val_loader, desc="Validating"):
            gt_features = batch["gt_features"].to(self.device)
            mod_features = batch["mod_features"].to(self.device)
            scores = batch["score"].to(self.device)

            # Monte Carlo dropout predictions
            predictions = torch.zeros_like(scores)
            for _ in range(self.mc_dropout_samples):
                self.model.train()  # Enable dropout
                predictions += self.model(gt_features, mod_features).squeeze()
            predictions /= self.mc_dropout_samples

            # Calculate losses
            mse_loss = self.mse_loss(predictions, scores)
            corr_loss = self.correlation_loss(predictions, scores)
            total_loss = mse_loss + 0.5 * corr_loss

            # Update metrics
            running_metrics["val_loss"] += total_loss.item()
            running_metrics["val_mse"] += mse_loss.item()
            running_metrics["val_correlation_loss"] += corr_loss.item()

            # Store predictions for correlation
            all_preds.extend(predictions.cpu().numpy())
            all_targets.extend(scores.cpu().numpy())

        # Calculate final metrics
        num_batches = len(self.val_loader)
        metrics = {k: v / num_batches for k, v in running_metrics.items()}

        # Add overall correlation
        metrics["val_pearson_correlation"] = np.corrcoef(all_preds, all_targets)[0, 1]

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
        early_stopping_patience: int = 15,
        plateau_patience: int = 5,
    ) -> None:
        """
        Complete training loop with validation and early stopping.

        Args:
            num_epochs: Number of epochs to train
            early_stopping_patience: Epochs to wait before early stopping
            plateau_patience: Epochs to wait before reducing LR on plateau
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
        plateau_counter = 0
        previous_val_losses: List[float] = []

        for epoch in range(num_epochs):
            print(f"\nEpoch {epoch+1}/{num_epochs}")

            # Training and validation
            train_metrics = self.train_epoch()
            val_metrics = self.validate()

            # Log metrics
            wandb.log({**train_metrics, **val_metrics})

            # Check for plateau
            previous_val_losses.append(val_metrics["val_loss"])
            if len(previous_val_losses) > plateau_patience:
                recent_std = np.std(previous_val_losses[-plateau_patience:])
                if recent_std < 1e-4:
                    plateau_counter += 1
                    if plateau_counter >= 3:
                        # Reduce learning rate
                        for param_group in self.optimizer.param_groups:
                            param_group["lr"] *= 0.5
                        plateau_counter = 0
                        print("Detected plateau, reducing learning rate")
                else:
                    plateau_counter = 0

            # Early stopping and checkpointing
            if val_metrics["val_loss"] < best_val_loss:
                best_val_loss = val_metrics["val_loss"]
                patience_counter = 0
                self.save_checkpoint(epoch, val_metrics, is_best=True)
            else:
                patience_counter += 1
                self.save_checkpoint(epoch, val_metrics, is_best=False)

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
    BATCH_SIZE = 64  # Reduced batch size for better generalization
    NUM_EPOCHS = 100
    LEARNING_RATE = 5e-5  # Reduced initial learning rate
    CHECKPOINT_DIR = "checkpoints"

    # Create dataloaders
    from features_and_scores_loader import create_feature_dataloaders

    train_loader, val_loader, _ = create_feature_dataloaders(
        features_root=FEATURES_ROOT, batch_size=BATCH_SIZE
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
