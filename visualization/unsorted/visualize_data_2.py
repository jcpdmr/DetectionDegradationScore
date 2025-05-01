import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np
from scipy import stats
import json

# Load and prepare the data
TRIAL = "attempt5_40bins_point8_06_visgen_coco17tr_openimagev7traine_320p_qual_20_24_28_32_36_40_50_smooth_2_subsam_444"

with open(f"checkpoints/{TRIAL}/test_predictions.json", "r") as f:
    data = json.load(f)
df = pd.DataFrame(data)

# Calculate statistical measures for our analysis
differences = df["pred_ddscore"] - df["ddscore"]
means = (df["pred_ddscore"] + df["ddscore"]) / 2
mean_diff = np.mean(differences)
std_diff = np.std(differences)

# Calculate Spearman correlation coefficient
spearman_corr, p_value = stats.spearmanr(df["ddscore"], df["pred_ddscore"])

# Create subplot layout with 3 rows and 2 columns for better space utilization
fig = make_subplots(
    rows=3,
    cols=2,
    subplot_titles=(
        f"Distance vs DDS (Spearman ρ: {spearman_corr:.3f}, p: {p_value:.3e})",
        "Bland-Altman Plot",
        "Normal Q-Q Plot of Differences",
        "Distribution of DD scores",
        "Distribution of Distances",
        "Distribution of Differences",
    ),
    vertical_spacing=0.15,  # Increased spacing between rows
    horizontal_spacing=0.12,  # Added spacing between columns
)

# 1. Enhanced Scatter Plot (top left)
fig.add_trace(
    go.Scatter(
        x=df["ddscore"],
        y=df["pred_ddscore"],
        mode="markers",
        name="Observations",
        text=df["filename"],
        marker=dict(size=2),
        hovertemplate="<b>File:</b> %{text}<br>"
        + "<b>DDS:</b> %{x:.3f}<br>"
        + "<b>Distance:</b> %{y:.3f}<br>",
    ),
    row=1,
    col=1,
)

# Add regression line with equation
z = np.polyfit(df["ddscore"], df["pred_ddscore"], 3)
p = np.poly1d(z)
x_range = np.linspace(df["ddscore"].min(), df["ddscore"].max(), 100)
fig.add_trace(
    go.Scatter(
        x=x_range,
        y=p(x_range),
        name=f"Regression (y={z[0]:.3f}x+{z[1]:.3f})",
        line=dict(color="red", dash="dash"),
    ),
    row=1,
    col=1,
)

# Add ideal y=x line
fig.add_trace(
    go.Scatter(
        x=[df["ddscore"].min(), df["ddscore"].max()],
        y=[df["ddscore"].min(), df["ddscore"].max()],
        name="y=x (ideal)",
        line=dict(color="black", dash="dot"),
    ),
    row=1,
    col=1,
)

# 2. Bland-Altman Plot (top right)
fig.add_trace(
    go.Scatter(
        x=means,
        y=differences,
        mode="markers",
        name="Differences",
        text=df["filename"],
        hovertemplate="<b>File:</b> %{text}<br>"
        + "<b>Mean:</b> %{x:.3f}<br>"
        + "<b>Difference:</b> %{y:.3f}<br>",
    ),
    row=1,
    col=2,
)

# Add reference lines with annotations
fig.add_hline(
    y=mean_diff,
    line_dash="dash",
    line_color="red",
    annotation_text=f"Mean: {mean_diff:.3f}",
    row=1,
    col=2,
)
fig.add_hline(
    y=mean_diff + 1.96 * std_diff,
    line_dash="dot",
    line_color="gray",
    annotation_text=f"+1.96 SD: {(mean_diff + 1.96 * std_diff):.3f}",
    row=1,
    col=2,
)
fig.add_hline(
    y=mean_diff - 1.96 * std_diff,
    line_dash="dot",
    line_color="gray",
    annotation_text=f"-1.96 SD: {(mean_diff - 1.96 * std_diff):.3f}",
    row=1,
    col=2,
)

# 3. QQ Plot (middle left)
qq = stats.probplot(differences)
fig.add_trace(
    go.Scatter(x=qq[0][0], y=qq[0][1], mode="markers", name="QQ Plot Points"),
    row=2,
    col=1,
)

# Add reference line
z = np.polyfit(qq[0][0], qq[0][1], 1)
p = np.poly1d(z)
fig.add_trace(
    go.Scatter(
        x=qq[0][0],
        y=p(qq[0][0]),
        name="Reference Line",
        line=dict(color="red", dash="dash"),
    ),
    row=2,
    col=1,
)

# 4. Distribution Histograms
# DDS distribution
fig.add_trace(
    go.Histogram(
        x=df["ddscore"],
        nbinsx=100,
        name="DDS",
    ),
    row=2,
    col=2,
)

# Distance distribution
fig.add_trace(
    go.Histogram(
        x=df["pred_ddscore"],
        nbinsx=100,
        name="Distance",
    ),
    row=3,
    col=1,
)

# Differences distribution
fig.add_trace(
    go.Histogram(
        x=differences,
        nbinsx=100,
        name="Differences",
    ),
    row=3,
    col=2,
)

# Update layout for better visualization
fig.update_layout(
    height=1800,  # Increased height
    width=1600,  # Increased width for two columns
    showlegend=True,
    legend=dict(yanchor="top", y=0.99, xanchor="left", x=1.02),
    title_text="Comprehensive Analysis of Distance vs DDS",
    template="plotly_white",
)

# Update axes labels with detailed descriptions
fig.update_xaxes(title_text="DDS", row=1, col=1)
fig.update_yaxes(title_text="Distance", row=1, col=1)
fig.update_xaxes(title_text="Mean of Measurements", row=1, col=2)
fig.update_yaxes(title_text="Difference (Distance - DDS)", row=1, col=2)
fig.update_xaxes(title_text="Theoretical Quantiles", row=2, col=1)
fig.update_yaxes(title_text="Sample Quantiles", row=2, col=1)
fig.update_xaxes(title_text="DDS", row=2, col=2)
fig.update_yaxes(title_text="Density", row=2, col=2)
fig.update_xaxes(title_text="Distance", row=3, col=1)
fig.update_yaxes(title_text="Density", row=3, col=1)
fig.update_xaxes(title_text="Difference", row=3, col=2)
fig.update_yaxes(title_text="Density", row=3, col=2)

# Save both interactive and static versions
fig.write_html(f"checkpoints/{TRIAL}/analysis_report_enhanced.html")

# Display the figure
fig.show()

# Print detailed statistical summary
print(f"""
Statistical Summary:
-------------------
Spearman Correlation: ρ = {spearman_corr:.3f} (p-value: {p_value:.3e})
Mean Difference: {mean_diff:.3f}
Standard Deviation of Differences: {std_diff:.3f}
95% Limits of Agreement: [{mean_diff - 1.96 * std_diff:.3f}, {mean_diff + 1.96 * std_diff:.3f}]
""")
