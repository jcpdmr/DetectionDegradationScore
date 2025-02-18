import json
import numpy as np
import plotly.graph_objects as go
from plotly.offline import plot
import plotly.io as pio


def create_calibration_curve(json_file: str, n_bins: int = 40) -> str:
    """
    Creates a calibration curve from JSON data and saves it as an HTML file.

    Args:
        json_file: Path to the JSON file containing predictions and error scores.
        n_bins: Number of bins to divide the predictions into.

    Returns:
        Path to the generated HTML file.
    """
    # Load data from JSON
    with open(json_file, "r") as f:
        data = json.load(f)

    predictions = [item["distance"] for item in data["predictions"]]
    error_scores = [item["error_score"] for item in data["predictions"]]

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
            bin_error_scores = [error_scores[j] for j in np.where(bin_mask)[0]]

            bin_means.append(np.mean(bin_predictions))
            bin_fractions.append(np.mean(bin_error_scores))
            bin_std_devs.append(
                np.std(bin_error_scores)
            )  # Calculate standard deviation of error scores
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
            marker=dict(size=10),
            # error_y=dict(
            #     type="data",
            #     array=bin_std_devs,
            #     visible=True,
            # ),
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
        title="Calibration Curve with Error Bars",
        xaxis_title="Mean Predicted Distance",
        yaxis_title="Fraction of Expected Error Score",
        xaxis=dict(range=[0, 1], dtick=0.2, minor=dict(dtick=0.05)),
        yaxis=dict(range=[0, 1], dtick=0.2, minor=dict(dtick=0.05)),
    )

    # Save to HTML and PNG
    output_file_html = "calibration_curve_with_error_bars.html"
    plot(fig, filename=output_file_html, auto_open=False)

    output_file_png = "calibration_curve.png"
    pio.write_image(fig, output_file_png)

    return output_file_html


if __name__ == "__main__":
    json_file_path = "checkpoints/attempt24_40bins_point8_06_visgen_coco17tr_openimagev7traine_320p_qual_20_24_28_32_36_40_50_smooth_2_subsam_444/test_predictions.json"
    html_file_path = create_calibration_curve(json_file_path)
    print(f"Calibration curve saved to {html_file_path}")
