import json
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np
from scipy import stats as scipy_stats


def load_error_scores(filepath):
    """
    Load error scores from a JSON file and return them as a list
    """
    with open(filepath, "r") as f:
        data = json.load(f)
    return list(data.values())


def create_error_score_analysis(error_scores, output_path, title_prefix):
    """
    Create comprehensive error score analysis with multiple visualizations

    Args:
        error_scores: List of error scores
        output_path: Where to save the HTML file
        title_prefix: 'Training' or 'Validation' prefix for plots

    Returns:
        fig: The plotly figure object for display
        stats: Dictionary containing basic statistics
    """
    # Calculate basic statistics to characterize the distribution
    mean_score = np.mean(error_scores)
    median_score = np.median(error_scores)
    std_score = np.std(error_scores)
    percentiles = np.percentile(error_scores, [25, 50, 75])
    iqr = percentiles[2] - percentiles[0]

    # Create a figure with four subplots that will give us different perspectives on the distribution
    fig = make_subplots(
        rows=2,
        cols=2,
        subplot_titles=(
            "Distribution of Error Scores",
            "Box Plot of Error Scores",
            "Q-Q Plot",
            "Cumulative Distribution",
        ),
        vertical_spacing=0.15,
        horizontal_spacing=0.12,
    )

    # 1. Simple histogram (top left)
    # This directly shows the distribution of scores
    fig.add_trace(
        go.Histogram(x=error_scores, nbinsx=20, name="Error Scores", showlegend=False),
        row=1,
        col=1,
    )

    # Add vertical lines for mean and median as references
    fig.add_vline(
        x=mean_score,
        line_dash="dash",
        line_color="red",
        annotation_text=f"Mean: {mean_score:.3f}",
        row=1,
        col=1,
    )
    fig.add_vline(
        x=median_score,
        line_dash="dash",
        line_color="green",
        annotation_text=f"Median: {median_score:.3f}",
        row=1,
        col=1,
    )

    # 2. Box Plot (top right)
    # This gives us a synthetic view of the distribution and outliers
    fig.add_trace(
        go.Box(
            y=error_scores,
            name="Error Scores",
            boxpoints="all",  # show all points
            jitter=0.3,
            pointpos=-1.8,
            boxmean=True,  # show the mean as well
        ),
        row=1,
        col=2,
    )

    # 3. Q-Q Plot (bottom left)
    # This helps us evaluate the normality of the distribution
    qq = scipy_stats.probplot(error_scores)
    fig.add_trace(
        go.Scatter(
            x=qq[0][0],
            y=qq[0][1],
            mode="markers",
            name="Q-Q Plot Points",
            showlegend=False,
        ),
        row=2,
        col=1,
    )

    # Add reference line for the Q-Q plot
    z = np.polyfit(qq[0][0], qq[0][1], 1)
    p = np.poly1d(z)
    fig.add_trace(
        go.Scatter(
            x=qq[0][0],
            y=p(qq[0][0]),
            name="Reference Line",
            line=dict(color="red", dash="dash"),
            showlegend=False,
        ),
        row=2,
        col=1,
    )

    # 4. Cumulative distribution (bottom right)
    # This shows how values accumulate
    sorted_scores = np.sort(error_scores)
    cumulative = np.arange(1, len(sorted_scores) + 1) / len(sorted_scores)

    fig.add_trace(
        go.Scatter(
            x=sorted_scores, y=cumulative, mode="lines", name="ECDF", showlegend=False
        ),
        row=2,
        col=2,
    )

    # Update layout for optimal visualization
    fig.update_layout(
        height=1000,
        width=1200,
        title_text=f"{title_prefix} Error Score Analysis",
        template="plotly_white",
    )

    # Update axis labels for clarity
    fig.update_xaxes(title_text="Error Score", row=1, col=1)
    fig.update_yaxes(title_text="Count", row=1, col=1)
    fig.update_yaxes(title_text="Error Score", row=1, col=2)
    fig.update_xaxes(title_text="Theoretical Quantiles", row=2, col=1)
    fig.update_yaxes(title_text="Sample Quantiles", row=2, col=1)
    fig.update_xaxes(title_text="Error Score", row=2, col=2)
    fig.update_yaxes(title_text="Cumulative Probability", row=2, col=2)

    # Save the figure
    fig.write_html(output_path)

    # Return figure and statistics
    stats = {
        "mean": mean_score,
        "median": median_score,
        "std": std_score,
        "q1": percentiles[0],
        "q3": percentiles[2],
        "iqr": iqr,
    }

    return fig, stats


# Process both datasets
train_scores = load_error_scores("balanced_dataset/train/error_scores.json")
val_scores = load_error_scores("balanced_dataset/val/error_scores.json")

# Create visualizations and get statistics
train_fig, train_stats = create_error_score_analysis(
    train_scores,
    "balanced_dataset/train/error_scores_visual.html",
    "Training",
)

val_fig, val_stats = create_error_score_analysis(
    val_scores,
    "balanced_dataset/val/error_scores_visual.html",
    "Validation",
)

# Print statistical summary
print(
    """
Statistical Summary:
-------------------
Training Set:
    Mean: {:.3f}
    Median: {:.3f}
    Standard Deviation: {:.3f}
    Q1: {:.3f}
    Q3: {:.3f}
    IQR: {:.3f}

Validation Set:
    Mean: {:.3f}
    Median: {:.3f}
    Standard Deviation: {:.3f}
    Q1: {:.3f}
    Q3: {:.3f}
    IQR: {:.3f}
""".format(
        train_stats["mean"],
        train_stats["median"],
        train_stats["std"],
        train_stats["q1"],
        train_stats["q3"],
        train_stats["iqr"],
        val_stats["mean"],
        val_stats["median"],
        val_stats["std"],
        val_stats["q1"],
        val_stats["q3"],
        val_stats["iqr"],
    )
)

# Show figures
train_fig.show()
val_fig.show()
