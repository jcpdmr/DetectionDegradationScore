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
    according to balanced dataset information from three separate files
    """

    def __init__(
        self,
        balanced_files: Dict[str, Path],
        unbalanced_path: Union[str, Path],
        balanced_path: Union[str, Path],
        clean_existing: bool = True,
    ):
        """
        Initialize the dataset organizer.
        
        Args:
            balanced_files: Dictionary mapping split names to their balanced dataset files
            unbalanced_path: Path to the unbalanced dataset
            balanced_path: Path to store the balanced dataset
            clean_existing: Whether to clean existing directories
        """
        self.balanced_files = {k: Path(v) for k, v in balanced_files.items()}
        self.unbalanced_path = Path(unbalanced_path)
        self.balanced_path = Path(balanced_path)
        self.clean_existing = clean_existing
        self.split_info = {}

    def load_split_info(self) -> Dict:
        """Load the balanced dataset information from JSON files"""
        for split, file_path in self.balanced_files.items():
            with open(file_path, "r") as f:
                data = json.load(f)
                self.split_info[split] = data.get("selected_items", {})
                
            print(f"Loaded {len(self.split_info[split])} items from {split} balanced dataset")
        return self.split_info

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

    def copy_image(self, img_name: str, quality: str, source_split: str, target_split: str):
        """
        Copy both extracted and compressed versions of an image to their respective locations
        
        Args:
            img_name: Image filename
            quality: JPEG quality value
            source_split: Source split (train, val, test)
            target_split: Target split (train, val, test)
        """
        # Copy extracted image
        src_extracted = self.unbalanced_path / source_split / "extracted" / img_name
        dst_extracted = self.balanced_path / target_split / "extracted" / img_name

        # Copy compressed image (from specific quality folder)
        src_compressed = (
            self.unbalanced_path / source_split / f"compressed{quality}" / img_name
        )
        dst_compressed = self.balanced_path / target_split / "compressed" / img_name

        try:
            shutil.copy2(src_extracted, dst_extracted)
            shutil.copy2(src_compressed, dst_compressed)
            return True
        except FileNotFoundError as e:
            print(f"Error copying {img_name} from {source_split}: {str(e)}")
            return False
        except Exception as e:
            print(f"Unexpected error copying {img_name} from {source_split}: {str(e)}")
            return False

    def organize_dataset(self):
        """Main method to organize the dataset according to balanced dataset information"""
        # Load balanced dataset information
        self.load_split_info()

        # Create directory structure
        self.create_directory_structure()

        # Process each split
        total_processed = 0
        success_count = 0
        fail_count = 0

        for target_split, items in self.split_info.items():
            print(f"Processing {target_split} split...")
            
            for img_name, info in items.items():
                # Use same split for source and target (train->train, val->val, test->test)
                source_split = target_split
                
                success = self.copy_image(img_name, info["quality"], source_split, target_split)
                total_processed += 1
                
                if success:
                    success_count += 1
                else:
                    fail_count += 1

                if total_processed % 10000 == 0:
                    print(f"Processed {total_processed} images ({success_count} successful, {fail_count} failed)")

        print(
            f"Dataset organization completed. Total: {total_processed}, Successful: {success_count}, Failed: {fail_count}"
        )

        # Create error scores for each split
        self.create_error_scores()
        
        print("Error scores created for all splits")

    def create_error_scores(self):
        """Create error_scores.json for each split"""
        for split, items in self.split_info.items():
            error_scores = {
                img_name: info["score"] for img_name, info in items.items()
            }

            output_file = self.balanced_path / split / "error_scores.json"
            with open(output_file, "w") as f:
                json.dump(error_scores, f, indent=4)
                
            print(f"Created error_scores.json for {split} with {len(error_scores)} items")


if __name__ == "__main__":

    ATTEMPT = "07_coco17complete_320p_qual_20_25_30_35_40_45_50_subsamp_444"
    BASE_DIR = f"error_scores_analysis/mapping/{ATTEMPT}"

    # OUTPUT_PATH = "split.json"
    # VAL_SPLIT = 0.05
    # TEST_SPLIT = 0.05
    # SEED = 42

    # # Create and run splitter
    # splitter = DatasetSplitter(
    #     input_path=INPUT_PATH,
    #     output_path=OUTPUT_PATH,
    #     val_split=VAL_SPLIT,
    #     test_split=TEST_SPLIT,
    #     seed=SEED,
    # )
    # splitter.process()

    CONFIG = {
        "balanced_files": {
            "train": f"{BASE_DIR}/train/balanced_dataset.json",
            "val": f"{BASE_DIR}/val/balanced_dataset.json",
            "test": f"{BASE_DIR}/test/balanced_dataset.json",
        },
        "unbalanced_path": "/andromeda/personal/jdamerini/unbalanced_dataset_coco2017",
        "balanced_path": "balanced_dataset_coco2017",
        "clean_existing": True,
    }

    # Create and run organizer
    organizer = DatasetOrganizer(**CONFIG)
    organizer.organize_dataset()
