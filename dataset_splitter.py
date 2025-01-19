import json
import random
from collections import defaultdict
from typing import Dict, List, Tuple, Union
import shutil
from pathlib import Path


class DatasetSplitter:
    """
    A class to split a balanced dataset into train, validation and test sets
    while maintaining score distribution.
    """

    def __init__(
        self,
        input_path: str,
        output_path: str,
        val_split: float,
        test_split: float,
        seed: int,
    ):
        """
        Initialize the splitter with the provided parameters.

        Args:
            input_path: Path to input JSON file
            output_path: Path to output JSON file
            val_split: Fraction of data for validation
            test_split: Fraction of data for testing
            seed: Random seed for reproducibility
        """
        self.input_file = input_path
        self.output_file = output_path
        self.val_split = val_split
        self.test_split = test_split
        self.train_split = 1.0 - (val_split + test_split)
        self.seed = seed
        random.seed(self.seed)

    def load_data(self) -> Dict:
        """Load and return the data from the input JSON file."""
        with open(self.input_file, "r") as f:
            return json.load(f)

    def organize_by_bins(self, data: Dict) -> Dict[int, List[Tuple[str, Dict]]]:
        """
        Organize images by their score bins.

        Args:
            data: The loaded JSON data

        Returns:
            Dictionary with bins as keys and lists of (image_name, image_data) as values
        """
        bins_dict = defaultdict(list)
        for img_name, img_data in data["selected_items"].items():
            bins_dict[img_data["bin"]].append((img_name, img_data))
        return bins_dict

    def split_bin_data(
        self, bin_data: List[Tuple[str, Dict]]
    ) -> Tuple[List, List, List]:
        """
        Split a single bin's data into train, validation and test sets.
        """
        n_samples = len(bin_data)
        val_size = int(n_samples * self.val_split)
        test_size = int(n_samples * self.test_split)

        # Shuffle the data
        shuffled_data = bin_data.copy()
        random.shuffle(shuffled_data)

        # Split the data
        test_data = shuffled_data[:test_size]
        val_data = shuffled_data[test_size : test_size + val_size]
        train_data = shuffled_data[test_size + val_size :]

        return train_data, val_data, test_data

    def create_splits(self) -> Dict:
        """
        Create the train, validation and test splits while maintaining bin distribution.
        """
        # Load and organize data
        data = self.load_data()
        bins_dict = self.organize_by_bins(data)

        # Initialize split info
        split_info = {
            "metadata": {
                "val_split": self.val_split,
                "test_split": self.test_split,
                "train_split": self.train_split,
                "seed": self.seed,
            },
            "train": {},
            "val": {},
            "test": {},
        }

        # Process each bin
        for bin_num, bin_data in bins_dict.items():
            train_data, val_data, test_data = self.split_bin_data(bin_data)

            # Add to train split
            for img_name, img_data in train_data:
                split_info["train"][img_name] = {
                    "quality": img_data["quality"],
                    "score": img_data["score"],
                    "bin": img_data["bin"],
                }

            # Add to validation split
            for img_name, img_data in val_data:
                split_info["val"][img_name] = {
                    "quality": img_data["quality"],
                    "score": img_data["score"],
                    "bin": img_data["bin"],
                }

            # Add to test split
            for img_name, img_data in test_data:
                split_info["test"][img_name] = {
                    "quality": img_data["quality"],
                    "score": img_data["score"],
                    "bin": img_data["bin"],
                }

        return split_info

    def save_splits(self, split_info: Dict):
        """Save the split information to a JSON file."""
        with open(self.output_file, "w") as f:
            json.dump(split_info, f, indent=4)

    def process(self):
        """Execute the complete splitting process and analyze distribution."""
        split_info = self.create_splits()
        self.save_splits(split_info)

        # Analyze and print distribution
        distribution = self.analyze_distribution(split_info)
        self.print_distribution_analysis(distribution)
        return distribution

    def analyze_distribution(self, split_info: Dict) -> Dict:
        """
        Analyze and return the distribution of bins in each split.

        Returns:
            Dictionary containing bin distributions for each split
        """
        distributions = {
            "train": defaultdict(int),
            "val": defaultdict(int),
            "test": defaultdict(int),
        }

        # Count bins in each split
        for img_name, data in split_info["train"].items():
            distributions["train"][data["bin"]] += 1

        for img_name, data in split_info["val"].items():
            distributions["val"][data["bin"]] += 1

        for img_name, data in split_info["test"].items():
            distributions["test"][data["bin"]] += 1

        # Convert defaultdict to regular dict and sort bins
        for split in distributions:
            distributions[split] = dict(sorted(distributions[split].items()))

        # Calculate percentages and total counts
        total_counts = {
            "train": sum(distributions["train"].values()),
            "val": sum(distributions["val"].values()),
            "test": sum(distributions["test"].values()),
        }

        # Add percentage information
        distribution_with_percentages = {
            "summary": {
                "total_images": sum(total_counts.values()),
                "split_sizes": total_counts,
            },
            "distributions": {
                split: {
                    bin_num: {
                        "count": count,
                        "percentage": (count / total_counts[split]) * 100,
                    }
                    for bin_num, count in bins.items()
                }
                for split, bins in distributions.items()
            },
        }

        return distribution_with_percentages

    def print_distribution_analysis(self, distribution: Dict):
        """
        Print a formatted analysis of the distribution.
        """
        print("\nDataset Distribution Analysis")
        print("=" * 50)

        # Print summary
        print("\nTotal Images:", distribution["summary"]["total_images"])
        print("\nSplit Sizes:")
        for split, count in distribution["summary"]["split_sizes"].items():
            print(f"{split}: {count} images")

        # Print distribution for each split
        for split, bins in distribution["distributions"].items():
            print(f"\n{split.upper()} Split Distribution:")
            print("-" * 30)
            print(f"{'Bin':>5} {'Count':>8} {'Percentage':>12}")
            print("-" * 30)
            for bin_num, data in bins.items():
                print(f"{bin_num:>5} {data['count']:>8} {data['percentage']:>11.2f}%")


class DatasetValidator:
    """Validates the organized dataset and creates error scores files"""

    def __init__(self, balanced_path: Path, split_info: Dict):
        self.balanced_path = balanced_path
        self.split_info = split_info

    def validate_split(self, split: str) -> List[str]:
        """
        Validate a specific split and return any missing files
        """
        missing_files = []
        split_path = self.balanced_path / split

        # Get list of expected images
        expected_images = set(self.split_info[split].keys())

        # Check extracted folder
        extracted_images = set(f.name for f in (split_path / "extracted").glob("*.jpg"))
        missing_extracted = expected_images - extracted_images

        # Check compressed folder
        compressed_images = set(
            f.name for f in (split_path / "compressed").glob("*.jpg")
        )
        missing_compressed = expected_images - compressed_images

        # Combine missing files
        missing_files.extend(f"{split}/extracted/{img}" for img in missing_extracted)
        missing_files.extend(f"{split}/compressed/{img}" for img in missing_compressed)

        return missing_files

    def create_error_scores(self, split: str):
        """Create error_scores.json for a specific split"""
        error_scores = {
            img_name: info["score"] for img_name, info in self.split_info[split].items()
        }

        output_file = self.balanced_path / split / "error_scores.json"
        with open(output_file, "w") as f:
            json.dump(error_scores, f, indent=4)


class DatasetOrganizer:
    """
    Organizes dataset by copying files from unbalanced to balanced structure
    according to split information
    """

    def __init__(
        self,
        split_file: Union[str, Path],
        unbalanced_path: Union[str, Path],
        balanced_path: Union[str, Path],
        clean_existing: bool = True,
    ):
        self.split_file = Path(split_file)
        self.unbalanced_path = Path(unbalanced_path)
        self.balanced_path = Path(balanced_path)
        self.clean_existing = clean_existing

    def load_split_info(self) -> Dict:
        """Load the split information from JSON file"""
        with open(self.split_file, "r") as f:
            return json.load(f)

    def create_directory_structure(self):
        """Create or clean the directory structure for the balanced dataset"""
        splits = ["train", "val", "test"]
        modifications = ["extracted", "compressed"]

        for split in splits:
            for mod in modifications:
                path = self.balanced_path / split / mod

                if path.exists() and self.clean_existing:
                    print(f"Cleaning {path}")
                    shutil.rmtree(path)

                path.mkdir(parents=True, exist_ok=True)
                print(f"Created {path}")

    def copy_image(self, img_name: str, quality: str, split: str):
        """
        Copy both extracted and compressed versions of an image to their respective locations
        """
        # Copy extracted image
        src_extracted = self.unbalanced_path / "train" / "extracted" / img_name
        dst_extracted = self.balanced_path / split / "extracted" / img_name

        # Copy compressed image (from specific quality folder)
        src_compressed = (
            self.unbalanced_path / "train" / f"compressed{quality}" / img_name
        )
        dst_compressed = self.balanced_path / split / "compressed" / img_name

        try:
            shutil.copy2(src_extracted, dst_extracted)
            shutil.copy2(src_compressed, dst_compressed)
        except FileNotFoundError as e:
            print(f"Error copying {img_name}: {str(e)}")
        except Exception as e:
            print(f"Unexpected error copying {img_name}: {str(e)}")

    def organize_dataset(self):
        """Main method to organize the dataset according to split information"""
        # Load split information
        split_info = self.load_split_info()

        # Create directory structure
        self.create_directory_structure()

        # Process each split
        total_processed = 0
        for split in ["train", "val", "test"]:
            print(f"Processing {split} split...")

            for img_name, info in split_info[split].items():
                self.copy_image(img_name, info["quality"], split)
                total_processed += 1

                if total_processed % 10000 == 0:
                    print(f"Processed {total_processed} images")

        print(
            f"Dataset organization completed. Total images processed: {total_processed}"
        )

        # Validate dataset and create error scores
        validator = DatasetValidator(self.balanced_path, split_info)

        # Validate all splits
        all_missing_files = []
        for split in ["train", "val", "test"]:
            missing_files = validator.validate_split(split)
            if missing_files:
                all_missing_files.extend(missing_files)

            # Create error_scores.json for each split
            validator.create_error_scores(split)

        # Report validation results
        if all_missing_files:
            print("\nValidation Error: Missing files detected:")
            for file in all_missing_files:
                print(f"  - {file}")
            raise Exception("Dataset validation failed: Missing files detected")
        else:
            print("\nValidation successful: All files present")


if __name__ == "__main__":
    # Hardcoded configuration
    INPUT_PATH = "balanced_dataset_20bins_point8_qual_40_45_50_55_60_70.json"
    OUTPUT_PATH = "split.json"
    VAL_SPLIT = 0.1
    TEST_SPLIT = 0.1
    SEED = 42

    # Create and run splitter
    splitter = DatasetSplitter(
        input_path=INPUT_PATH,
        output_path=OUTPUT_PATH,
        val_split=VAL_SPLIT,
        test_split=TEST_SPLIT,
        seed=SEED,
    )
    splitter.process()

    # Hardcoded configuration
    CONFIG = {
        "split_file": "split.json",
        "unbalanced_path": "unbalanced_dataset",
        "balanced_path": "balanced_dataset",
        "clean_existing": True,
    }

    # Create and run organizer
    organizer = DatasetOrganizer(**CONFIG)
    organizer.organize_dataset()
