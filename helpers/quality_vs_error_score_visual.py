import json
import plotly.graph_objects as go
import numpy as np
from scipy import stats
import pandas as pd
from plotly.subplots import make_subplots
import os


def load_data(quality_path, error_scores_path):
    """
    Load quality and error score data from respective JSON files and combine them into a DataFrame.

    Args:
        quality_path: path to the JSON file containing the quality mapping
        error_scores_path: path to the JSON file containing the error scores

    Returns:
        DataFrame with columns: image, quality, error_score
    """
    with open(quality_path, "r") as f:
        quality_mapping = json.load(f)

    with open(error_scores_path, "r") as f:
        error_scores = json.load(f)

    data = [
        {
            "image": img_name,
            "quality": quality_mapping[img_name],
            "error_score": error_scores[img_name],
        }
        for img_name in set(quality_mapping.keys()) & set(error_scores.keys())
    ]
    return pd.DataFrame(data)


def get_bin_labels(bins):
    """
    Create readable labels for bins with a numeric prefix to ensure correct ordering.

    Args:
        bins: array of values defining the bin edges

    Returns:
        List of strings formatted as "XX. start-end"
    """
    return [f"{i:02d}. {bins[i]:.1f}-{bins[i + 1]:.1f}" for i in range(len(bins) - 1)]


def create_range_stats(df, column, bins, value_column):
    """
    Calculate statistics for ranges of values.

    Args:
        df: DataFrame with the data
        column: name of the column to be binned
        bins: array of bin edges
        value_column: name of the column on which to calculate statistics

    Returns:
        Dictionary with statistics for each bin
    """
    df_copy = df.copy()
    labels = get_bin_labels(bins)
    df_copy["bin"] = pd.cut(df_copy[column], bins, labels=labels)

    stats = (
        df_copy.groupby("bin", observed=True)[value_column]
        .agg(["mean", "std", "count"])
        .round(3)
    )
    return {idx: row.to_dict() for idx, row in stats.iterrows()}


def create_box_trace(df, bin_column, value_column, name):
    """
    Create a box plot with correctly ordered categories.

    Args:
        df: DataFrame with the data
        bin_column: name of the column containing the bins
        value_column: name of the column containing the values
        name: name to assign to the trace

    Returns:
        Configured go.Box object
    """
    df_valid = df[df[bin_column].notna()].copy()

    return go.Box(x=df_valid[bin_column], y=df_valid[value_column], name=name)


def create_analysis_report(df, output_path):
    """
    Create a complete HTML report with interactive visualizations and statistics
    analyzing the relationship between quality and error score.

    Args:
        df: DataFrame containing the data
        output_path: path to save the HTML report
    """
    # Create the report structure with subplots
    fig = make_subplots(
        rows=3,
        cols=2,
        subplot_titles=(
            "Scatter Plot: Quality vs Error Score",
            "Quality Distribution by Error Score Ranges",
            "Error Score Distribution by Quality Ranges",
            "Distribution of Quality Values for Error Score = 1",
            "Correlation Analysis",
            "Summary Statistics",
        ),
        specs=[
            [{"type": "scatter"}, {"type": "box"}],
            [{"type": "box"}, {"type": "histogram"}],
            [{"type": "table"}, {"type": "table"}],
        ],
        vertical_spacing=0.12,
        horizontal_spacing=0.1,
    )

    # 1. Scatter plot with trend line
    fig.add_trace(
        go.Scatter(
            x=df["quality"],
            y=df["error_score"],
            mode="markers",
            name="Data Points",
            marker=dict(size=5, opacity=0.5),
        ),
        row=1,
        col=1,
    )

    # Calculate and add the trend line
    z = np.polyfit(df["quality"], df["error_score"], 1)
    p = np.poly1d(z)
    fig.add_trace(
        go.Scatter(
            x=df["quality"],
            y=p(df["quality"]),
            mode="lines",
            name="Trend Line",
            line=dict(color="red"),
        ),
        row=1,
        col=1,
    )

    # 2. Create bins for error score (range 0-1)
    error_bins = np.arange(0, 1.1, 0.1)
    error_labels = get_bin_labels(error_bins)
    df["error_bin"] = pd.cut(df["error_score"], bins=error_bins, labels=error_labels)

    # 3. Create bins for quality (range 0-100)
    quality_bins = np.arange(0, 101, 5)
    quality_labels = get_bin_labels(quality_bins)
    df["quality_bin"] = pd.cut(df["quality"], bins=quality_bins, labels=quality_labels)

    # Add box plots
    fig.add_trace(
        create_box_trace(df, "error_bin", "quality", "Quality by Error Range"),
        row=1,
        col=2,
    )

    fig.add_trace(
        create_box_trace(df, "quality_bin", "error_score", "Error by Quality Range"),
        row=2,
        col=1,
    )

    # 4. Analyze cases with error_score = 1
    perfect_errors = df[df["error_score"] == 1.0]
    fig.add_trace(
        go.Histogram(x=perfect_errors["quality"], nbinsx=20, name="Error Score = 1"),
        row=2,
        col=2,
    )

    # 5. Calculate and display correlations
    pearson_corr, p_value_pearson = stats.pearsonr(df["quality"], df["error_score"])
    spearman_corr, p_value_spearman = stats.spearmanr(df["quality"], df["error_score"])

    correlation_data = [
        ["Pearson", f"{pearson_corr:.3f}", f"{p_value_pearson:.3e}"],
        ["Spearman", f"{spearman_corr:.3f}", f"{p_value_spearman:.3e}"],
    ]

    fig.add_trace(
        go.Table(
            header=dict(values=["Correlation Type", "Coefficient", "P-value"]),
            cells=dict(values=list(zip(*correlation_data))),
        ),
        row=3,
        col=1,
    )

    # 6. Statistics for cases with error_score = 1
    error_1_stats = {
        "Total Count": len(perfect_errors),
        "Percentage": f"{(len(perfect_errors) / len(df)) * 100:.2f}%",
        "Mean Quality": f"{perfect_errors['quality'].mean():.2f}",
        "Std Quality": f"{perfect_errors['quality'].std():.2f}",
        "Min Quality": f"{perfect_errors['quality'].min():.2f}",
        "Max Quality": f"{perfect_errors['quality'].max():.2f}",
    }

    fig.add_trace(
        go.Table(
            header=dict(values=["Metric", "Value"]),
            cells=dict(
                values=[list(error_1_stats.keys()), list(error_1_stats.values())]
            ),
        ),
        row=3,
        col=2,
    )

    # Update layout
    fig.update_layout(
        height=1200,
        width=1600,
        showlegend=False,
        title_text="Analysis of Quality vs Error Score Relationship",
    )

    # Improve label readability
    fig.update_xaxes(tickangle=45, row=1, col=2)
    fig.update_xaxes(tickangle=45, row=2, col=1)
    fig.update_xaxes(
        categoryorder="array",
        categoryarray=sorted(df["error_bin"].dropna().unique()),
        row=1,
        col=2,
    )
    fig.update_xaxes(
        categoryorder="array",
        categoryarray=sorted(df["quality_bin"].dropna().unique()),
        row=2,
        col=1,
    )
    # Create output directory if it doesn't exist
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    # Save the HTML report
    fig.write_html(output_path)
    fig.show()

    # Create and save statistics in JSON format
    stats_dict = {
        "correlations": {
            "pearson": {"coefficient": pearson_corr, "p_value": float(p_value_pearson)},
            "spearman": {
                "coefficient": spearman_corr,
                "p_value": float(p_value_spearman),
            },
        },
        "error_score_1_statistics": error_1_stats,
        "error_score_ranges": create_range_stats(
            df, "error_score", error_bins, "quality"
        ),
        "quality_ranges": create_range_stats(
            df, "quality", quality_bins, "error_score"
        ),
    }

    stats_path = os.path.join(os.path.dirname(output_path), "analysis_stats.json")
    with open(stats_path, "w") as f:
        json.dump(stats_dict, f, indent=4)


# Define file paths
error_scores_run_folder = "09_quality_range_uniform_10_30_visgen"
quality_path = "dataset_attention/train/quality_mapping.json"
error_scores_path = (
    f"error_scores_analysis/{error_scores_run_folder}/train/error_scores.json"
)
output_path = (
    f"error_scores_analysis/{error_scores_run_folder}/train/quality_error_analysis.html"
)

# Run the analysis
df = load_data(quality_path, error_scores_path)
create_analysis_report(df, output_path)
