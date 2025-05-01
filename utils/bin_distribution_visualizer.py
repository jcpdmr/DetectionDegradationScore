import numpy as np
from typing import List


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
