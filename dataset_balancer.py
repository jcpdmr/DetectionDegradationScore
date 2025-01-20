import json
import numpy as np
from collections import defaultdict
from typing import Dict, List, Tuple, Set
import pandas as pd
import matplotlib.pyplot as plt


class DatasetBalancer:
    def __init__(
        self,
        error_scores_path: str,
        n_bins: int = 40,
        max_score: float = 0.8,  # New parameter
        critical_range: Tuple[float, float] = (0.6, 0.8),  # Adjusted critical range
        recalc_interval: int = 100,
    ):
        """
        Initialize the dataset balancer.

        Args:
            error_scores_path: Path to the JSON file containing error scores
            n_bins: Number of bins to divide the error scores into
            max_score: Maximum error score to consider
            critical_range: Range of scores considered critical
            recalc_interval: Number of images after which to recalculate critical bins
        """
        self.n_bins = n_bins
        self.max_score = max_score
        # Create bin edges from 0 to max_score
        self.bin_edges = np.linspace(0, max_score, n_bins + 1)
        self.critical_range = critical_range
        self.recalc_interval = recalc_interval

        # Load and process error scores
        with open(error_scores_path, "r") as f:
            error_scores_full = json.load(f)

        # Filter out scores above max_score
        self.error_scores = {}
        for img_id, scores in error_scores_full.items():
            valid_scores = {
                quality: score
                for quality, score in scores.items()
                if score <= self.max_score
            }
            if valid_scores:  # Only include images that have valid scores
                self.error_scores[img_id] = valid_scores

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
        # Count available images per bin
        bin_counts = {i: len(imgs) for i, imgs in current_availability.items()}

        if not bin_counts:
            return []

        # Calculate statistics
        mean_count = np.mean(list(bin_counts.values()))
        min_count = min(bin_counts.values())

        # Consider a bin critical if:
        # 1. It has less than mean_count * 0.8 images OR
        # 2. It has less than min_count * 1.2 images
        critical_bins = []
        for bin_idx in range(self.n_bins):
            if bin_idx in bin_counts:
                count = bin_counts[bin_idx]
                if count < mean_count * 0.8 or count <= min_count * 1.2:
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
        if not selected_items:
            self.stats["final_stats"] = {
                "images_used": 0,
                "usage_percentage": 0,
                "unused_images": self.total_images,
                "bins_filled": 0,
                "average_images_per_bin": 0,
            }
            self.stats["bin_stats"] = {
                "bin_counts": {},
                "bin_percentages": {},
                "deviation_from_target": {},
                "unfilled_bins": list(range(self.n_bins)),
                "overfilled_bins": [],
                "underfilled_bins": [],
            }
            return

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
        Create a balanced dataset with faster selection strategy and safe removal.
        """
        print("\nInitializing dataset balancer...")
        selected_items = {}

        # Pre-process data structures
        image_to_bins = defaultdict(
            list
        )  # image_id -> list of (bin_idx, score, quality)
        bin_to_images = defaultdict(set)  # bin_idx -> set of available image_ids
        bin_counts = defaultdict(int)  # bin_idx -> number of selected images

        # Single pass to organize data
        print("Building initial data structures...")
        for bin_idx, items in self.availability_map.items():
            for item in items:
                img_id = item["img_id"]
                image_to_bins[img_id].append((bin_idx, item["score"], item["quality"]))
                bin_to_images[bin_idx].add(img_id)

        print("Initial availability:")
        for bin_idx in range(self.n_bins):
            print(f"Bin {bin_idx}: {len(bin_to_images[bin_idx])} images available")

        while True:
            # Calculate minimum available images across all bins
            min_available = min(len(images) for images in bin_to_images.values())
            if min_available == 0:
                break

            # Find the most empty bin (based on what we've already selected)
            current_counts = [bin_counts[i] for i in range(self.n_bins)]
            min_filled = min(current_counts)
            emptiest_bin = current_counts.index(min_filled)

            # Determine how many images to select in this iteration
            n_select = max(1, int(min_available * 0.01))

            print("\nIteration status:")
            print(f"Min available: {min_available}")
            print(f"Most empty bin: {emptiest_bin} (contains {min_filled} images)")
            print(f"Will select {n_select} images")

            # Select images from the emptiest bin
            selected_count = 0
            available_images = list(bin_to_images[emptiest_bin])

            for img_id in available_images:
                if selected_count >= n_select:
                    break

                # Get all bin assignments for this image
                assignments = image_to_bins[img_id]

                # Add to selected items
                bin_assignment = next(a for a in assignments if a[0] == emptiest_bin)
                selected_items[img_id] = {
                    "quality": bin_assignment[2],
                    "score": bin_assignment[1],
                    "bin": emptiest_bin,
                }
                bin_counts[emptiest_bin] += 1
                selected_count += 1

                # Safely remove this image from all bins it was available in
                for bin_idx, _, _ in assignments:
                    if (
                        img_id in bin_to_images[bin_idx]
                    ):  # Check if image is still in the bin
                        bin_to_images[bin_idx].remove(img_id)

                # Remove from image_to_bins
                del image_to_bins[img_id]

            if selected_count == 0:
                print(f"Warning: Could not select any images for bin {emptiest_bin}")
                break

            # Print current distribution every iteration
            print("\nCurrent distribution:")
            counts = [bin_counts[i] for i in range(self.n_bins)]
            print(
                f"Min: {min(counts)}, Max: {max(counts)}, Diff: {max(counts) - min(counts)}"
            )

        print("\nFinal distribution:")
        final_counts = [bin_counts[i] for i in range(self.n_bins)]
        print(f"Min: {min(final_counts)}, Max: {max(final_counts)}")
        print(f"Total images selected: {len(selected_items)}")

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


def validate_dataset(json_file, img_file):
    """
    Validate the dataset and create distribution visualization.
    """
    # Load the data
    with open(json_file, "r") as f:
        data = json.load(f)

    selected_items = data["selected_items"]

    # Check for duplicates
    image_count = defaultdict(int)
    for img_name in selected_items:
        image_count[img_name] += 1

    duplicates = {img: count for img, count in image_count.items() if count > 1}
    if duplicates:
        print("WARNING: Found duplicate images:")
        for img, count in duplicates.items():
            print(f"Image {img} appears {count} times")
    else:
        print("No duplicate images found.")

    # Create bin edges (40 bins from 0 to 0.8)
    n_bins = 40
    bin_edges = np.linspace(0, 0.8, n_bins + 1)

    # Validate bin assignments and count distribution
    bin_distribution = np.zeros(n_bins, dtype=int)
    incorrect_bins = []

    for img_name, info in selected_items.items():
        score = info["score"]
        assigned_bin = info["bin"]

        # Find the correct bin
        correct_bin = None
        for i in range(len(bin_edges) - 1):
            if bin_edges[i] <= score < bin_edges[i + 1]:
                correct_bin = i
                break

        if correct_bin != assigned_bin:
            incorrect_bins.append(
                {
                    "image": img_name,
                    "score": score,
                    "assigned_bin": assigned_bin,
                    "correct_bin": correct_bin,
                    "bin_range": f"[{bin_edges[assigned_bin]:.3f}, {bin_edges[assigned_bin + 1]:.3f})",
                }
            )

        # Count for distribution
        if 0 <= assigned_bin < n_bins:  # Ensure bin index is valid
            bin_distribution[assigned_bin] += 1

    # Report incorrect bin assignments
    if incorrect_bins:
        print("\nWARNING: Found incorrect bin assignments:")
        for error in incorrect_bins:
            print(
                f"Image {error['image']} with score {error['score']:.4f} "
                f"is in bin {error['assigned_bin']} (range {error['bin_range']}) "
                f"but should be in bin {error['correct_bin']}"
            )
    else:
        print("\nAll bin assignments are correct.")

    # Modify plot
    plt.figure(figsize=(15, 7))
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    plt.bar(bin_centers, bin_distribution, width=0.4 / n_bins, alpha=0.7)
    plt.xlabel("Score Range (0-0.8)")
    plt.ylabel("Number of Images")
    plt.title("Distribution of Images Across Bins (0-0.8)")
    plt.grid(True, alpha=0.3)

    # Add bin edges annotations
    for i, count in enumerate(bin_distribution):
        if count > 0:  # Only annotate non-empty bins
            plt.text(
                bin_centers[i], count, f"Bin {i}\n({count})", ha="center", va="bottom"
            )

    # Add x-axis ticks at bin edges
    plt.xticks(bin_edges[::2], [f"{x:.2f}" for x in bin_edges[::2]], rotation=45)

    plt.tight_layout()
    plt.savefig(img_file)

    return {
        "has_duplicates": bool(duplicates),
        "incorrect_bins": incorrect_bins,
        "distribution": bin_distribution.tolist(),
        "bin_edges": bin_edges.tolist(),
    }


if __name__ == "__main__":
    N_BINS = 40
    MAX_SCORE = 0.8  # New parameter
    MULTI_ERROR_SCORES_PATH = "error_scores_analysis/mapping/05_visual_genome_cocotrain17_320p_qual_20_24_28_32_36_40_50_smooth_2_subsam_444/total/error_scores.json"
    OUTPUT_BAL_DATASET_JSON = "balanced_dataset_40bins_point8_05_visual_genome_cocotrain17_320p_qual_20_24_28_32_36_40_50_smooth_2_subsam_444.json"
    OUTPUT_DISTRIBUTION_IMG = "balanced_dataset_40bins_point8_05_visual_genome_cocotrain17_320p_qual_20_24_28_32_36_40_50_smooth_2_subsam_444.png"

    balancer = DatasetBalancer(
        error_scores_path=MULTI_ERROR_SCORES_PATH,
        n_bins=N_BINS,
        max_score=MAX_SCORE,
        critical_range=(0.6, 0.8),
    )
    selected_items = balancer.create_balanced_dataset()
    balancer.print_statistics()

    # Save results
    with open(OUTPUT_BAL_DATASET_JSON, "w") as f:
        json.dump(
            {
                "selected_items": balancer._convert_numpy_types(selected_items),
                "statistics": balancer._convert_numpy_types(balancer.stats),
            },
            f,
            indent=4,
        )

    # Validate the dataset
    results = validate_dataset(OUTPUT_BAL_DATASET_JSON, OUTPUT_DISTRIBUTION_IMG)
