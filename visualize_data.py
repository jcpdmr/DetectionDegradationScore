import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np
import os
import json


def create_interactive_analysis(json_path):
    """
    Create an interactive visualization for comparing similarity and error scores from JSON data.

    Args:
        json_path (str): Path to JSON file containing the scores
    """
    # Read and process the JSON data
    with open(json_path, "r") as f:
        data = json.load(f)

    # Convert to DataFrame and rename columns
    df = pd.DataFrame(data)
    df = df.rename(columns={"distance": "similarity_score", "filename": "image_name"})

    # Create figure with secondary y-axis
    fig = make_subplots(
        rows=2,
        cols=2,
        subplot_titles=(
            "Scatter Plot Similarity vs Error Score",
            "Score Distributions",
            "Difference Distribution (Error - Similarity)",
            "Cumulative Distribution",
        ),
        specs=[[{}, {}], [{}, {}]],
    )

    # 1. Main scatter plot with y=x reference line
    fig.add_trace(
        go.Scatter(
            x=df["similarity_score"],
            y=df["error_score"],
            mode="markers",
            name="Data Points",
            hovertemplate=(
                "Image: %{customdata}<br>"
                + "Similarity: %{x:.3f}<br>"
                + "Error: %{y:.3f}"
            ),
            customdata=df["image_name"],
            marker=dict(size=8, opacity=0.6, color="blue"),
        ),
        row=1,
        col=1,
    )

    # Add y=x reference line
    fig.add_trace(
        go.Scatter(
            x=[0, 1],
            y=[0, 1],
            mode="lines",
            name="y=x (Perfect Prediction)",
            line=dict(color="red", dash="dash"),
        ),
        row=1,
        col=1,
    )

    # 2. Distribution plot
    fig.add_trace(
        go.Histogram(
            x=df["similarity_score"],
            name="Similarity Score",
            opacity=0.75,
            xbins=dict(start=0, end=1.05, size=0.05),
            autobinx=False,
        ),
        row=1,
        col=2,
    )

    fig.add_trace(
        go.Histogram(x=df["error_score"], name="Error Score", opacity=0.75, nbinsx=30),
        row=1,
        col=2,
    )

    # 3. Difference distribution
    differences = df["error_score"] - df["similarity_score"]
    fig.add_trace(
        go.Histogram(x=differences, name="Error - Similarity", opacity=0.75, nbinsx=30),
        row=2,
        col=1,
    )

    # 4. Cumulative distribution
    fig.add_trace(
        go.Scatter(
            x=sorted(df["similarity_score"]),
            y=np.linspace(0, 1, len(df)),
            name="Similarity CDF",
            mode="lines",
        ),
        row=2,
        col=2,
    )

    fig.add_trace(
        go.Scatter(
            x=sorted(df["error_score"]),
            y=np.linspace(0, 1, len(df)),
            name="Error CDF",
            mode="lines",
        ),
        row=2,
        col=2,
    )

    # Update layout with titles and axis labels
    fig.update_layout(
        title="Interactive Analysis of Similarity vs Error Scores",
        showlegend=True,
        height=1000,
        width=1200,
        template="plotly_white",
        hovermode="closest",
    )

    # Update axes labels
    fig.update_xaxes(title_text="Similarity Score", row=1, col=1)
    fig.update_yaxes(title_text="Error Score", row=1, col=1)
    fig.update_xaxes(title_text="Score Value", row=1, col=2)
    fig.update_yaxes(title_text="Count", row=1, col=2)
    fig.update_xaxes(title_text="Difference (Error - Similarity)", row=2, col=1)
    fig.update_yaxes(title_text="Count", row=2, col=1)
    fig.update_xaxes(title_text="Score Value", row=2, col=2)
    fig.update_yaxes(title_text="Cumulative Probability", row=2, col=2)

    # Calculate and display statistics
    stats = {
        "Correlation": df["similarity_score"].corr(df["error_score"]),
        "Mean Absolute Error": np.mean(
            np.abs(df["similarity_score"] - df["error_score"])
        ),
        "Mean Similarity": df["similarity_score"].mean(),
        "Mean Error": df["error_score"].mean(),
        "Std Similarity": df["similarity_score"].std(),
        "Std Error": df["error_score"].std(),
    }

    print("\nStatistics:")
    for key, value in stats.items():
        print(f"{key}: {value:.4f}")

    return fig


def main():
    # Specify the path to your JSON file
    base_path = ""  # Modifica questo percorso
    file_name = "test_predictions.json"  # Modifica questo nome file

    fig = create_interactive_analysis(os.path.join(base_path, file_name))
    fig.show()

    # Optional: save as HTML file
    fig.write_html(os.path.join(base_path, "interactive_analysis.html"))


if __name__ == "__main__":
    main()
