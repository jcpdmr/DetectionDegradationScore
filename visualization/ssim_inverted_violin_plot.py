import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


def create_inverted_ssim_plot(input_csv: str, output_path: str):
    """
    Create and save a violin plot of 1-SSIM scores by quality factor.

    Args:
        input_csv: Path to CSV file containing SSIM scores
        output_path: Path to save the plot
    """
    # Load data
    df = pd.read_csv(input_csv)

    # Calculate 1-SSIM
    df["inverted_ssim"] = 1 - df["ssim_score"]

    # Create plot
    plt.figure(figsize=(12, 8))

    # Create violin plot
    sns.violinplot(
        data=df,
        x="quality_factor",
        y="inverted_ssim",
        density_norm="area",  # Ensure equal areas
        bw_adjust=0.8,  # Adjust bandwidth
        inner="box",  # Show box plot inside violin
        width=1,  # Width of violins
    )

    # Set plot title and labels
    plt.title("1-SSIM Distribution by JPEG Quality Factor", fontsize=16)
    plt.xlabel("Quality Factor", fontsize=14)
    plt.ylabel("1-SSIM Score (higher = more distortion)", fontsize=14)

    # Set y-axis limits for better visualization
    plt.ylim(0, 1.05)

    # Add grid for better readability
    plt.grid(True, alpha=0.3)

    # Save the plot
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    print(f"Inverted SSIM violin plot saved to: {output_path}")

    # Save inverted data
    inverted_csv = output_path.replace(".png", "_data.csv")
    df[["image", "quality_factor", "inverted_ssim"]].to_csv(inverted_csv, index=False)
    print(f"Inverted SSIM data saved to: {inverted_csv}")

    # Also save statistics
    stats = (
        df.groupby("quality_factor")["inverted_ssim"]
        .agg(["count", "mean", "std", "min", "max", "median"])
        .reset_index()
    )

    stats_dict = {
        int(row["quality_factor"]): {
            "count": int(row["count"]),
            "mean": float(row["mean"]),
            "std": float(row["std"]),
            "min": float(row["min"]),
            "max": float(row["max"]),
            "median": float(row["median"]),
        }
        for _, row in stats.iterrows()
    }

    json_path = output_path.replace(".png", "_stats.json")
    import json

    with open(json_path, "w") as f:
        json.dump(stats_dict, f, indent=4)
    print(f"Inverted SSIM statistics saved to: {json_path}")

    # Show the plot
    plt.show()


def main():
    # Configuration
    INPUT_CSV = "ssim_analysis/ssim_scores.csv"
    OUTPUT_PATH = "ssim_analysis/inverted_ssim_violin_plot.png"

    # Check if input file exists
    if not os.path.exists(INPUT_CSV):
        print(f"Error: Input CSV file not found: {INPUT_CSV}")
        return

    # Create the plot
    create_inverted_ssim_plot(INPUT_CSV, OUTPUT_PATH)

    print("Analysis complete!")


if __name__ == "__main__":
    main()
