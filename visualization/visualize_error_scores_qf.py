import plotly.graph_objects as go
import numpy as np
import json
from pathlib import Path
import pandas as pd
from typing import List, Tuple


class CompressionAnalyzer:
    def __init__(self, json_path: str):
        """Initialize analyzer with path to JSON data file."""
        self.json_path = json_path
        self.data = self._load_data()
        self.quality_factors = sorted(
            [int(qf) for qf in list(list(self.data.values())[0].keys())]
        )
        self.df = self._create_dataframe()

    def _load_data(self) -> dict:
        """Load and validate JSON data."""
        with open(self.json_path, "r") as f:
            return json.load(f)

    def _create_dataframe(self) -> pd.DataFrame:
        """Convert JSON data to pandas DataFrame for easier analysis."""
        records = []
        for img_name, scores in self.data.items():
            for qf, score in scores.items():
                records.append(
                    {
                        "image": img_name,
                        "quality_factor": int(qf),
                        "ddscore": float(score),
                    }
                )
        return pd.DataFrame(records)

    def plot_degradation_analysis(self, output_dir: Path) -> None:
        """Create and save degradation analysis plots."""
        output_dir.mkdir(parents=True, exist_ok=True)

        # 1. Mean degradation curve
        mean_scores = self.df.groupby("quality_factor")["ddscore"].mean()
        std_scores = self.df.groupby("quality_factor")["ddscore"].std()

        fig_mean = go.Figure()
        fig_mean.add_trace(
            go.Scatter(
                x=list(range(len(self.quality_factors))),  # Equispaced x-axis
                y=mean_scores.values,
                mode="lines+markers",
                name="Mean DDS",
                error_y=dict(type="data", array=std_scores.values, visible=True),
            )
        )

        # Update x-axis to show actual QF values as labels
        fig_mean.update_layout(
            xaxis=dict(
                title="Quality Factor",
                tickmode="array",
                ticktext=self.quality_factors,
                tickvals=list(range(len(self.quality_factors))),
                showline=True,
                linewidth=1,
                linecolor="black",
                mirror=True,
            ),
            yaxis=dict(
                title="Detection Degradation Score",
                gridcolor="lightgray",
                showline=True,
                linewidth=1,
                linecolor="black",
                mirror=True,
            ),
            plot_bgcolor="white",
            height=600,
            width=800,
        )
        fig_mean.write_image(output_dir / "mean_degradation.png")

        # 2. Box plot
        fig_box = go.Figure()

        # Create mapping between actual QF and equispaced positions
        qf_positions = {qf: i for i, qf in enumerate(self.quality_factors)}

        # Create a new column in dataframe with equispaced positions
        temp_df = self.df.copy()
        temp_df["equispaced_pos"] = temp_df["quality_factor"].map(qf_positions)

        fig_box.add_trace(
            go.Box(
                x=temp_df["equispaced_pos"],  # Use equispaced positions
                y=temp_df["ddscore"],
                name="Score Distribution",
            )
        )

        fig_box.update_layout(
            xaxis=dict(
                title="Quality Factor",
                tickmode="array",
                ticktext=self.quality_factors,  # Show actual QF values as labels
                tickvals=list(
                    range(len(self.quality_factors))
                ),  # Use equispaced positions
                showline=True,
                linewidth=1,
                linecolor="black",
                mirror=True,
            ),
            yaxis=dict(
                title="Detection Degradation Score",
                gridcolor="lightgray",
                showline=True,
                linewidth=1,
                linecolor="black",
                mirror=True,
            ),
            plot_bgcolor="white",
            height=600,
            width=800,
        )
        fig_box.write_image(output_dir / "error_distribution.png")

        # Violin Plot
        import seaborn as sns
        import matplotlib.pyplot as plt

        plt.figure(figsize=(12, 8))

        sns.violinplot(
            data=self.df,
            x="quality_factor",
            y="ddscore",
            density_norm="area",  # Ensure equal areas
            bw_adjust=0.8,  # Adjust bandwidth (>1 smoother, <1 more detailed)
            inner="box",
            width=1,
        )

        plt.ylim(0, 1)
        plt.xlabel("Quality Factor")
        plt.ylabel("Detection Degradation Score")
        plt.grid(True, alpha=0.3)

        # Save plot
        plt.savefig(output_dir / "violin_plot.png", bbox_inches="tight", dpi=300)
        plt.close()

        # Count ddscore = 1 for each QF
        ones_count = {}
        total_count = {}

        for qf in self.quality_factors:
            mask = self.df["quality_factor"] == qf
            ones_count[qf] = len(self.df[mask & (self.df["ddscore"] > 0.95)])
            total_count[qf] = len(self.df[mask])

        # Create absolute count plot
        fig_abs = go.Figure()
        fig_abs.add_trace(
            go.Bar(
                x=list(range(len(self.quality_factors))),
                y=[ones_count[qf] for qf in self.quality_factors],
                text=[ones_count[qf] for qf in self.quality_factors],
                textposition="auto",
                opacity=0.9,
            )
        )

        fig_abs.update_layout(
            # title="Number of DDS close to 1 (e > 0.95) per Quality Factor",
            xaxis=dict(
                title="Quality Factor",
                tickmode="array",
                ticktext=self.quality_factors,
                tickvals=list(range(len(self.quality_factors))),
                showline=True,
                linewidth=1,
                linecolor="black",
                mirror=True,
            ),
            yaxis=dict(
                title="Count",
                gridcolor="lightgray",
                showline=True,
                linewidth=1,
                linecolor="black",
                mirror=True,
            ),
            plot_bgcolor="white",
            height=600,
            width=800,
        )

        fig_abs.write_image(output_dir / "error_ones_absolute.png")

        # Create percentage plot
        percentages = [
            ones_count[qf] / total_count[qf] * 100 for qf in self.quality_factors
        ]

        fig_perc = go.Figure()
        fig_perc.add_trace(
            go.Bar(
                x=list(range(len(self.quality_factors))),
                y=percentages,
                text=[f"{p:.1f}%" for p in percentages],
                textposition="auto",
                opacity=0.9,
            )
        )

        fig_perc.update_layout(
            # title="Percentage of DDS close to 1 (e > 0.95) per Quality Factor",
            xaxis=dict(
                title="Quality Factor",
                tickmode="array",
                ticktext=self.quality_factors,
                tickvals=list(range(len(self.quality_factors))),
                showline=True,
                linewidth=1,
                linecolor="black",
                mirror=True,
            ),
            yaxis=dict(
                title="Percentage",
                gridcolor="lightgray",
                showline=True,
                linewidth=1,
                linecolor="black",
                mirror=True,
            ),
            plot_bgcolor="white",
            height=600,
            width=800,
        )

        fig_perc.write_image(output_dir / "error_ones_percentage.png")

        # Count ddscore = 0 for each QF
        zeros_count = {}
        total_count = {}

        for qf in self.quality_factors:
            mask = self.df["quality_factor"] == qf
            zeros_count[qf] = len(self.df[mask & (self.df["ddscore"] < 0.05)])
            total_count[qf] = len(self.df[mask])

        # Create percentage plot
        percentages = [
            zeros_count[qf] / total_count[qf] * 100 for qf in self.quality_factors
        ]

        fig_perc = go.Figure()
        fig_perc.add_trace(
            go.Bar(
                x=list(range(len(self.quality_factors))),
                y=percentages,
                text=[f"{p:.1f}%" for p in percentages],
                textposition="auto",
                opacity=0.9,
            )
        )

        fig_perc.update_layout(
            # title="Percentage of DDS close to 0 (e < 0.05) per Quality Factor",
            xaxis=dict(
                title="Quality Factor",
                tickmode="array",
                ticktext=self.quality_factors,
                tickvals=list(range(len(self.quality_factors))),
                showline=True,
                linewidth=1,
                linecolor="black",
                mirror=True,
            ),
            yaxis=dict(
                title="Percentage",
                gridcolor="lightgray",
                showline=True,
                linewidth=1,
                linecolor="black",
                mirror=True,
            ),
            plot_bgcolor="white",
            height=600,
            width=800,
        )
        fig_perc.write_image(output_dir / "error_zeros_percentage.png")

    def analyze_image_robustness(self) -> Tuple[List[str], List[str]]:
        """
        Identify robust and sensitive images to compression.
        Returns tuple of (robust_images, sensitive_images)
        """
        # Calculate degradation slope for each image
        slopes = []
        for img_name in self.data.keys():
            scores = [self.data[img_name][str(qf)] for qf in self.quality_factors]
            # Use equispaced x values for slope calculation
            x_values = list(range(len(self.quality_factors)))
            slope = np.polyfit(x_values, scores, 1)[0]
            slopes.append((img_name, slope))

        # Sort by slope
        slopes.sort(key=lambda x: x[1])

        # Get 1000 most robust and sensitive images
        robust_images = [img for img, _ in slopes[:1000]]
        sensitive_images = [img for img, _ in slopes[-1000:]]

        return robust_images, sensitive_images

    def get_random_images(self, n_images: int = 1000, seed: int = 42) -> List[str]:
        """
        Pick n_images random images from the dataset.

        Args:
            n_images: Number of images to select
            seed: Random seed for reproducibility

        Returns:
            List of randomly selected image names
        """
        np.random.seed(seed)
        all_images = list(self.data.keys())
        selected_images = np.random.choice(all_images, size=n_images, replace=False)
        return selected_images.tolist()


def main():
    # Configuration
    BASE_PATH = Path("ddscores_analysis/mapping")
    ATTEMPT = "07_coco17complete_320p_qual_20_25_30_35_40_45_50_subsamp_444"
    SPLITS = ["train", "val", "test"]  # Process all splits

    for split in SPLITS:
        print(f"\n\n{'=' * 50}")
        print(f"Visualizing error-QF for {split.upper()} split")
        print(f"{'=' * 50}\n")

        OUTPUT_DIR = BASE_PATH / ATTEMPT / split
        JSON_PATH = OUTPUT_DIR / "ddscores.json"

        if not JSON_PATH.exists():
            print(f"Warning: No dd scores found for {split} split at {JSON_PATH}")
            continue

        print(f"Loading dd scores from: {JSON_PATH}")

        # Initialize analyzer for this split
        analyzer = CompressionAnalyzer(JSON_PATH)

        # Generate plots
        analyzer.plot_degradation_analysis(OUTPUT_DIR)

        # # Get robust and sensitive images
        # robust_images, sensitive_images = analyzer.analyze_image_robustness()
        # # Save image lists
        # with open(OUTPUT_DIR / "robust_images.txt", "w") as f:
        #     f.write("\n".join(robust_images))
        # with open(OUTPUT_DIR / "sensitive_images.txt", "w") as f:
        #     f.write("\n".join(sensitive_images))

        # Get random images
        random_images = analyzer.get_random_images(n_images=1000, seed=42)

        # Save random images list
        with open(OUTPUT_DIR / "random_pick.txt", "w") as f:
            f.write("\n".join(random_images))

        print(f"Analysis complete for {split} split")
        print(f"Graphs and results saved to {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
