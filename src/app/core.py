"""Theil-Sen non-parametric anomaly detector for time series data.

Uses local Theil-Sen regression and robust statistics to detect anomalies in time series,
adapting to different distributions and patterns without parametric assumptions.
"""

# %%
import os
from os import path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from scipy import stats


# %%
def estimate_threshold(z_scores: np.ndarray, max_anomaly_ratio: float = 0.05) -> float:
    """Calculate adaptive threshold for anomaly detection based on z-score distribution.

    Args:
        z_scores: Array of z-scores from local regression residuals
        max_anomaly_ratio: Maximum proportion of points to be flagged as anomalies

    Returns:
        Threshold value, minimum of 3.0 MADs or data-driven estimate
    """
    sorted_scores = np.sort(np.abs(z_scores))
    n = len(sorted_scores)
    max_outliers = int(n * max_anomaly_ratio)

    if max_outliers == 0:
        return float(max(3.0, sorted_scores[-1]))

    return float(max(3.0, sorted_scores[-(max_outliers + 1)]))


# %%
def detect_local_anomalies(
    df: pd.DataFrame, window_size: int = 20, max_anomaly_ratio: float = 0.05
) -> pd.DataFrame:
    """Detect anomalies using sliding window Theil-Sen regression.

    Args:
        df: DataFrame with 'date' and 'daily_total' columns
        window_size: Number of points to use for local trend estimation
        max_anomaly_ratio: Maximum proportion of anomalies to detect

    Returns:
        DataFrame with columns: date, daily_total, expected, z_score, is_anomaly
    """
    df = df.copy()
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date").reset_index(drop=True)

    z_scores: list[float] = []
    expecteds: list[float] = []

    # First pass: calculate z-scores
    for i in range(len(df)):
        start_idx = max(0, i - window_size)
        end_idx = min(len(df), i + window_size + 1)
        window = df.iloc[start_idx:end_idx]

        X = np.arange(len(window))
        y = np.array(window["daily_total"].values, dtype=np.float64)
        slope, intercept = stats.theilslopes(y, X)[:2]
        expected = float(slope * window_size + intercept)

        trend = slope * X + intercept
        deviations = y - trend
        mad = float(stats.median_abs_deviation(deviations, scale="normal"))

        curr_dev = float(df.iloc[i]["daily_total"]) - expected
        z_score = float(curr_dev / mad if mad > 0 else 0)

        z_scores.append(z_score)
        expecteds.append(expected)

    # Calculate adaptive threshold
    threshold = estimate_threshold(np.array(z_scores), max_anomaly_ratio)

    # Create results
    results = [
        {
            "date": df.iloc[i]["date"],
            "daily_total": float(df.iloc[i]["daily_total"]),
            "expected": expecteds[i],
            "z_score": z_scores[i],
            "is_anomaly": bool(abs(z_scores[i]) > threshold),
        }
        for i in range(len(df))
    ]

    return pd.DataFrame(results)


# %%
def plot_anomalies(results: pd.DataFrame, title: str = "Anomaly Detection") -> None:
    """Generate interactive plotly visualization of anomalies.

    Args:
        results: DataFrame from detect_local_anomalies()
        title: Plot title, also used for output filename

    Creates HTML file in plots/ directory named from sanitized title
    """
    os.makedirs("plots", exist_ok=True)

    fig = go.Figure()

    fig.add_trace(
        go.Scatter(
            x=results["date"],
            y=results["daily_total"],
            mode="lines",
            name="Values",
            line=dict(color="blue", width=1),
            hovertemplate="Date: %{x}<br>Value: %{y:.2f}<extra></extra>",
        )
    )

    fig.add_trace(
        go.Scatter(
            x=results["date"],
            y=results["expected"],
            mode="lines",
            name="Local Trend",
            line=dict(color="green", width=1),
            hovertemplate="Date: %{x}<br>Expected: %{y:.2f}<extra></extra>",
        )
    )

    anomalies = results[results["is_anomaly"]]
    fig.add_trace(
        go.Scatter(
            x=anomalies["date"],
            y=anomalies["daily_total"],
            mode="markers",
            name="Anomalies",
            marker=dict(color="red", size=8),
            hovertemplate=(
                "Date: %{x}<br>" "Value: %{y:.2f}<br>" "Z-score: %{text:.3f}<extra></extra>"
            ),
            text=anomalies["z_score"],
        )
    )

    fig.update_layout(
        title=title,
        xaxis_title="Date",
        yaxis_title="Daily Total",
        hovermode="x unified",
        template="plotly_white",
    )

    filename = f"plots/{title.lower().replace(' ', '_')}.html"
    fig.write_html(filename)


# %%
def analyse_dataset(filepath: str) -> dict[str, pd.DataFrame]:
    """Process all sheets in Excel file for anomalies.

    Args:
        filepath: Path to Excel file with multiple sheets

    Returns:
        Dict mapping sheet names to detection results

    Also generates plots and prints summary statistics
    """
    sheets = pd.read_excel(filepath, sheet_name=None)
    results = {}

    for sheet_name, df in sheets.items():
        results[sheet_name] = detect_local_anomalies(df)
        plot_anomalies(results[sheet_name], f"Anomalies in {sheet_name}")

        anomalies = results[sheet_name][results[sheet_name]["is_anomaly"]]
        print(f"\nDataset {sheet_name}: {len(anomalies)} anomalies detected")
        print(
            anomalies.sort_values("z_score", ascending=False)[
                ["date", "daily_total", "z_score"]
            ].head()
        )

    return results


# %%
if __name__ == "__main__":
    # Example usage
    home = path.expanduser("~")
    filepath = home + "/Downloads/test.xlsx"
    results = analyse_dataset(filepath)
