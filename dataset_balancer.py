import json
import numpy as np
from collections import defaultdict
from typing import Dict, List, Tuple, Set
import pandas as pd


class DatasetBalancer:
    def __init__(
        self,
        error_scores_path: str,
        n_bins: int = 20,
        critical_range: Tuple[float, float] = (0.8, 0.99),
        recalc_interval: int = 100,
    ):
        """
        Initialize the dataset balancer.

        Args:
            error_scores_path: Path to the JSON file containing error scores
            n_bins: Number of bins to divide the error scores into
            critical_range: Range of scores considered critical
            recalc_interval: Number of images after which to recalculate critical bins
        """
        self.n_bins = n_bins
        self.bin_edges = np.linspace(0, 1, n_bins + 1)
        self.critical_range = critical_range
        self.recalc_interval = recalc_interval

        # Load and process error scores
        with open(error_scores_path, "r") as f:
            self.error_scores = json.load(f)

        # Dataset statistics
        self.total_images = len(self.error_scores)
        self.scores_per_image = {
            img_id: len(scores) for img_id, scores in self.error_scores.items()
        }
        self.total_scores = sum(self.scores_per_image.values())

        # Create initial availability map
        self.availability_map = self._create_availability_map()

        # Initialize statistics dictionary
        self.stats = {
            "initial_stats": self._calculate_initial_stats(),
            "final_stats": {},
            "bin_stats": {},
        }

    def _calculate_initial_stats(self) -> Dict:
        """
        Calculate initial dataset statistics.
        """
        # Count how many images can potentially contribute to each bin
        images_per_bin = defaultdict(set)
        for img_id, scores in self.error_scores.items():
            for score in scores.values():
                bin_idx = np.digitize(score, self.bin_edges) - 1
                if bin_idx < self.n_bins:
                    images_per_bin[bin_idx].add(img_id)

        critical_images = sum(
            1
            for img_id, scores in self.error_scores.items()
            if any(self._is_in_critical_range(score) for score in scores.values())
        )

        return {
            "total_images": self.total_images,
            "total_scores": self.total_scores,
            "avg_scores_per_image": self.total_scores / self.total_images,
            "images_per_bin": {
                bin_idx: len(images) for bin_idx, images in images_per_bin.items()
            },
            "critical_range_images": critical_images,
            "score_distribution": {
                f"{self.bin_edges[i]:.2f}-{self.bin_edges[i + 1]:.2f}": len(
                    images_per_bin[i]
                )
                for i in range(self.n_bins)
            },
        }

    def _count_scores_in_bin(self, img_id: str, bin_idx: int) -> int:
        """
        Count how many error scores of an image fall into a specific bin.

        Args:
            img_id: Image identifier
            bin_idx: Index of the bin

        Returns:
            Number of scores in the bin
        """
        bin_start = self.bin_edges[bin_idx]
        bin_end = self.bin_edges[bin_idx + 1]
        return sum(
            1
            for score in self.error_scores[img_id].values()
            if bin_start <= score < bin_end
        )

    def _update_availability(self, used_images: Set[str]) -> Dict:
        """
        Update availability map excluding already used images.

        Args:
            used_images: Set of image IDs that have already been used

        Returns:
            Updated availability map
        """
        current_availability = defaultdict(list)

        for img_id, scores in self.error_scores.items():
            if img_id not in used_images:
                for quality, score in scores.items():
                    bin_idx = np.digitize(score, self.bin_edges) - 1
                    if bin_idx < self.n_bins:
                        current_availability[bin_idx].append(
                            {"img_id": img_id, "quality": quality, "score": score}
                        )

        return current_availability

    def _get_total_valid_bins(self, img_id: str) -> int:
        """
        Count in how many different bins an image can be allocated.

        Args:
            img_id: Image identifier

        Returns:
            Number of different bins where the image can be allocated
        """
        scores = self.error_scores[img_id].values()
        return len(set(np.digitize(list(scores), self.bin_edges) - 1))

    def _create_availability_map(self) -> Dict:
        """
        Create a map showing which images can contribute to which bins.
        """
        availability = defaultdict(list)

        for img_id, scores in self.error_scores.items():
            for quality, score in scores.items():
                bin_idx = np.digitize(score, self.bin_edges) - 1
                if bin_idx < self.n_bins:  # Ensure we don't exceed bin range
                    availability[bin_idx].append(
                        {"img_id": img_id, "quality": quality, "score": score}
                    )

        return availability

    def _is_in_critical_range(self, score: float) -> bool:
        """Check if a score falls within the critical range."""
        return self.critical_range[0] <= score <= self.critical_range[1]

    def _find_critical_bins(self, current_availability: Dict) -> List[int]:
        """
        Identify bins with fewest available images.

        Args:
            current_availability: Current availability map

        Returns:
            List of critical bin indices
        """
        bin_counts = {i: len(imgs) for i, imgs in current_availability.items()}
        if not bin_counts:
            return []

        min_count = min(bin_counts.values())

        # Sort bins by count and identify critical ones
        critical_bins = []
        for bin_idx in range(self.n_bins):
            if bin_idx in bin_counts and bin_counts[bin_idx] <= min_count * 1.2:
                critical_bins.append(bin_idx)

        # Sort by count to prioritize most critical
        return sorted(critical_bins, key=lambda x: bin_counts[x])

    def _sort_available_images(self, available: List[Dict], bin_idx: int) -> List[Dict]:
        """
        Sort available images for a specific bin based on optimization criteria.

        Args:
            available: List of available images for the bin
            bin_idx: Index of the bin being processed

        Returns:
            Sorted list of available images
        """
        return sorted(
            available,
            key=lambda x: (
                self._count_scores_in_bin(
                    x["img_id"], bin_idx
                ),  # Primary: options in current bin
                -self._get_total_valid_bins(
                    x["img_id"]
                ),  # Secondary: total bins available
            ),
        )

    def _update_final_stats(self, selected_items: Dict, bin_counts: Dict):
        """
        Update final statistics after dataset creation.
        """
        self.stats["final_stats"] = {
            "images_used": len(selected_items),
            "usage_percentage": (len(selected_items) / self.total_images) * 100,
            "unused_images": self.total_images - len(selected_items),
            "bins_filled": len(bin_counts),
            "average_images_per_bin": len(selected_items) / self.n_bins,
        }

        target_count = len(selected_items) / self.n_bins

        self.stats["bin_stats"] = {
            "bin_counts": dict(bin_counts),
            "bin_percentages": {
                bin_: (count / len(selected_items)) * 100
                for bin_, count in bin_counts.items()
            },
            "deviation_from_target": {
                bin_: count - target_count for bin_, count in bin_counts.items()
            },
            "unfilled_bins": [i for i in range(self.n_bins) if i not in bin_counts],
            "overfilled_bins": [
                bin_ for bin_, count in bin_counts.items() if count > target_count * 1.1
            ],
            "underfilled_bins": [
                bin_ for bin_, count in bin_counts.items() if count < target_count * 0.9
            ],
        }

    def _convert_numpy_types(self, data):
        """
        Convert numpy types to native Python types for JSON serialization.
        """
        if isinstance(data, dict):
            return {str(k): self._convert_numpy_types(v) for k, v in data.items()}
        elif isinstance(data, list):
            return [self._convert_numpy_types(x) for x in data]
        elif isinstance(data, np.integer):
            return int(data)
        elif isinstance(data, np.floating):
            return float(data)
        else:
            return data

    def create_balanced_dataset(self) -> Dict:
        """
        Create a balanced dataset with maximum 5% deviation between bins.
        """
        used_images = set()
        selected_items = {}
        bin_counts = defaultdict(int)

        # Initialize availability per bin
        bins_availability = {}
        for bin_idx in range(self.n_bins):
            bins_availability[bin_idx] = [
                item
                for items in self.availability_map[bin_idx]
                for item in [items]
                if item["img_id"] not in used_images
            ]

        # Find the minimum number of available images among all bins
        min_available = min(len(bins_availability[i]) for i in range(self.n_bins))

        # Target per bin with 1% tolerance
        base_target = min_available
        max_target = int(base_target * 1.01)  # 1% maximum tolerance

        # Sort bins by initial availability (from most critical)
        bins_order = sorted(range(self.n_bins), key=lambda x: len(bins_availability[x]))

        # Process bin by bin
        for bin_idx in bins_order:
            available = bins_availability[bin_idx]

            # Sort available images to optimize selection
            available.sort(key=lambda x: self._get_total_valid_bins(x["img_id"]))

            for item in available:
                if (
                    item["img_id"] not in used_images
                    and bin_counts[bin_idx] < max_target
                ):
                    selected_items[item["img_id"]] = {
                        "quality": item["quality"],
                        "score": item["score"],
                        "bin": bin_idx,
                    }
                    used_images.add(item["img_id"])
                    bin_counts[bin_idx] += 1

                    # If we have reached the base target for this bin,
                    # continue only if the fullest bin does not exceed 5% difference
                    if (
                        bin_counts[bin_idx] >= base_target
                        and max(bin_counts.values()) - bin_counts[bin_idx]
                        > base_target * 0.05
                    ):
                        break

        self._update_final_stats(selected_items, bin_counts)
        return selected_items

    def analyze_distribution(self, selected_items: Dict) -> pd.DataFrame:
        """
        Analyze the distribution of selected items across bins.
        """
        bin_counts = defaultdict(int)
        for item in selected_items.values():
            bin_counts[item["bin"]] += 1

        # Create DataFrame for analysis
        df = pd.DataFrame(
            {
                "bin": range(self.n_bins),
                "count": [bin_counts[i] for i in range(self.n_bins)],
                "range": [
                    f"{self.bin_edges[i]:.2f}-{self.bin_edges[i + 1]:.2f}"
                    for i in range(self.n_bins)
                ],
            }
        )

        # Add statistics
        df["percentage"] = (df["count"] / df["count"].sum() * 100).round(2)
        df["diff_from_mean"] = (df["count"] - df["count"].mean()).round(2)

        return df

    def print_statistics(self):
        """
        Print comprehensive statistics about the dataset and balancing process.
        """
        print("\n=== Initial Dataset Statistics ===")
        print(f"Total images: {self.stats['initial_stats']['total_images']}")
        print(f"Total error scores: {self.stats['initial_stats']['total_scores']}")
        print(
            f"Average scores per image: {self.stats['initial_stats']['avg_scores_per_image']:.2f}"
        )
        print(
            f"Images in critical range ({self.critical_range[0]}-{self.critical_range[1]}): "
            f"{self.stats['initial_stats']['critical_range_images']}"
        )

        if self.stats["final_stats"]:
            print("\n=== Final Dataset Statistics ===")
            print(f"Images used: {self.stats['final_stats']['images_used']}")
            print(
                f"Usage percentage: {self.stats['final_stats']['usage_percentage']:.2f}%"
            )
            print(f"Unused images: {self.stats['final_stats']['unused_images']}")
            print(
                f"Average images per bin: {self.stats['final_stats']['average_images_per_bin']:.2f}"
            )

            print("\n=== Bin Statistics ===")
            print(f"Unfilled bins: {len(self.stats['bin_stats']['unfilled_bins'])}")
            print(f"Overfilled bins: {len(self.stats['bin_stats']['overfilled_bins'])}")
            print(
                f"Underfilled bins: {len(self.stats['bin_stats']['underfilled_bins'])}"
            )

            # Print distribution visualization
            print("\nBin Distribution:")
            max_count = max(self.stats["bin_stats"]["bin_counts"].values())
            for bin_ in range(self.n_bins):
                count = self.stats["bin_stats"]["bin_counts"].get(bin_, 0)
                bar_length = int((count / max_count) * 50)
                print(
                    f"Bin {bin_:2d} [{self.bin_edges[bin_]:.2f}-{self.bin_edges[bin_ + 1]:.2f}]: "
                    f"{'#' * bar_length} ({count})"
                )


balancer = DatasetBalancer(
    "error_scores_analysis/mapping/2025_01_15_192915/total/error_scores.json",
    n_bins=20,
    critical_range=(0.8, 1),
)
selected_items = balancer.create_balanced_dataset()
balancer.print_statistics()

# Save results
with open("balanced_dataset.json", "w") as f:
    json.dump(
        {
            "selected_items": selected_items,
            "statistics": balancer._convert_numpy_types(balancer.stats),
        },
        f,
        indent=4,
    )
