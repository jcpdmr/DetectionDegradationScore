import json
import numpy as np
import plotly.graph_objects as go
from plotly.offline import plot
import plotly.io as pio


def create_calibration_curve(
    path: str,
    json_file: str,
    with_errors: bool = False,
    n_bins: int = 40,
    max_value: float = 1.0,
) -> str:
    """
    Creates a calibration curve from JSON data and saves it as an HTML file.

    Args:
        json_file: Path to the JSON file containing predictions and dd scores.
        n_bins: Number of bins to divide the predictions into.
        max_value: Maximum value for the x-axis.

    Returns:
        Path to the generated HTML file.
    """
    # Load data from JSON
    with open(f"{path}/{json_file}", "r") as f:
        data = json.load(f)

    predictions = [item["pred_ddscore"] for item in data["predictions"]]
    ddscores = [item["ddscore"] for item in data["predictions"]]

    # Divide predictions into bins
    bin_edges = np.linspace(0, 1, n_bins + 1)
    bin_indices = np.digitize(predictions, bin_edges)

    # Calculate average prediction, fraction of positives, and std dev for each bin
    bin_means = []
    bin_fractions = []
    bin_std_devs = []
    for i in range(1, n_bins + 1):
        bin_mask = bin_indices == i
        if np.any(bin_mask):
            bin_predictions = [predictions[j] for j in np.where(bin_mask)[0]]
            bin_ddscores = [ddscores[j] for j in np.where(bin_mask)[0]]

            bin_means.append(np.mean(bin_predictions))
            bin_fractions.append(np.mean(bin_ddscores))
            bin_std_devs.append(
                np.std(bin_ddscores)
            )  # Calculate standard deviation of dd scores
        else:
            bin_means.append(None)  # No data in this bin
            bin_fractions.append(None)
            bin_std_devs.append(None)

    # Create plot
    fig = go.Figure()

    # Add calibration curve with error bars
    fig.add_trace(
        go.Scatter(
            x=bin_means,
            y=bin_fractions,
            mode="markers+lines",
            name="Calibration Curve",
            marker=dict(size=4),
            error_y=dict(
                type="data",
                array=bin_std_devs,
                visible=True,
            )
            if with_errors
            else dict(),
        )
    )

    # Add perfect calibration line
    fig.add_trace(
        go.Scatter(
            x=np.linspace(0, 1, 100),
            y=np.linspace(0, 1, 100),
            mode="lines",
            name="Perfectly Calibrated",
            line=dict(color="gray", dash="dash"),
        )
    )

    # Set layout
    fig.update_layout(
        xaxis_title="Binned Mean Predicted DDS",
        yaxis_title="Binned Mean Target DDS",
        xaxis=dict(
            range=[0, max_value],
            dtick=0.2,
            minor=dict(dtick=0.05),
            gridcolor="lightgray",
            showline=True,
            linewidth=1,
            linecolor="black",
            mirror=True,
        ),
        yaxis=dict(
            range=[0, max_value],
            dtick=0.2,
            minor=dict(dtick=0.05),
            gridcolor="lightgray",
            showline=True,
            linewidth=1,
            linecolor="black",
            mirror=True,
        ),
        plot_bgcolor="white",
    )

    # Save to HTML and PNG
    output_file_html = (
        f"{path}/calibration_curve{'_with_error_bars' if with_errors else ''}.html"
    )
    plot(fig, filename=output_file_html, auto_open=False)

    output_file_png = (
        f"{path}/calibration_curve{'_with_error_bars' if with_errors else ''}.png"
    )
    pio.write_image(fig, output_file_png, scale=8)


if __name__ == "__main__":
    ATTEMPT = 28
    CHECKPOINT_DIR = f"checkpoints/attempt{ATTEMPT}_40bins_point8_07_coco17complete_320p_qual_20_25_30_35_40_45_50_subsamp_444"
    JSON_FILE = "test_predictions.json"
    WITH_ERRORS = False
    MAX_VALUE = 0.8
    html_file_path = create_calibration_curve(
        path=CHECKPOINT_DIR,
        json_file=JSON_FILE,
        with_errors=WITH_ERRORS,
        max_value=MAX_VALUE,
    )
    print(f"Calibration curve saved, use error bars: {WITH_ERRORS}")
