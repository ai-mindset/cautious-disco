"""Adaptive Isolation Forest Detection"""

# %%
from os import path

import pandas as pd
import plotly.graph_objects as go
from sklearn.ensemble import IsolationForest


# %%
def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """Engineer features capturing temporal patterns and variations."""
    df = df.copy()
    df["date"] = pd.to_datetime(df["date"])

    # Sort by date
    df = df.sort_values("date").reset_index(drop=True)

    features = pd.DataFrame()

    # Basic time features
    features["day_of_month"] = df["date"].dt.day
    features["month"] = df["date"].dt.month
    features["value"] = df["daily_total"]

    # Rolling statistics (multiple windows)
    for window in [5, 10, 20]:
        features[f"rolling_med_{window}"] = (
            df["daily_total"].rolling(window=window, center=True, min_periods=1).median()
        )
        features[f"rolling_std_{window}"] = (
            df["daily_total"].rolling(window=window, center=True, min_periods=1).std()
        )
        features[f"value_to_med_{window}"] = df["daily_total"] / features[f"rolling_med_{window}"]

    # Local variation
    features["diff_prev"] = df["daily_total"].diff()
    features["diff_next"] = df["daily_total"].diff(-1)
    features["pct_change"] = df["daily_total"].pct_change()

    # Month start/end indicators
    features["is_month_start"] = df["date"].dt.is_month_start.astype(int)
    features["is_month_end"] = df["date"].dt.is_month_end.astype(int)

    # Fill any NaN values with 0
    features = features.fillna(0)

    return features


# %%
def detect_anomalies(
    df: pd.DataFrame, contamination: float = 0.05
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Detect anomalies using Isolation Forest with engineered features."""
    # Prepare features
    features = engineer_features(df)

    # Train Isolation Forest
    clf = IsolationForest(
        contamination=contamination, random_state=42, n_estimators=200, max_samples="auto"
    )

    # Predict anomalies
    anomaly_labels = clf.fit_predict(features)
    scores = clf.score_samples(features)

    # Add results to original dataframe
    results = df.copy()
    results["is_anomaly"] = anomaly_labels == -1
    results["anomaly_score"] = scores

    return results, features


# %%
def plot_anomalies(results: pd.DataFrame, title: str = "Anomaly Detection") -> None:
    """Create and save interactive plot of anomalies."""
    import os

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

    anomalies = results[results["is_anomaly"]]
    fig.add_trace(
        go.Scatter(
            x=anomalies["date"],
            y=anomalies["daily_total"],
            mode="markers",
            name="Anomalies",
            marker=dict(color="red", size=8),
            hovertemplate=(
                "Date: %{x}<br>" "Value: %{y:.2f}<br>" "Score: %{text:.3f}<extra></extra>"
            ),
            text=anomalies["anomaly_score"],
        )
    )

    fig.update_layout(
        title=title,
        xaxis_title="Date",
        yaxis_title="Daily Total",
        hovermode="x unified",
        template="plotly_white",
    )

    # Create plots directory if it doesn't exist
    os.makedirs("plots", exist_ok=True)

    # Generate filename from title
    filename = f"plots/{title.lower().replace(' ', '_')}.html"
    fig.write_html(filename)


# %%
def analyze_dataset(filepath: str) -> dict[str, pd.DataFrame]:
    """Analyze all sheets in Excel file."""
    sheets = pd.read_excel(filepath, sheet_name=None)
    results = {}

    for sheet_name, df in sheets.items():
        results_df, _ = detect_anomalies(df)
        results[sheet_name] = results_df
        plot_anomalies(results_df, f"Anomalies in {sheet_name}")

        anomalies = results_df[results_df["is_anomaly"]]
        print(f"\nDataset {sheet_name}: {len(anomalies)} anomalies detected")
        print(
            anomalies.sort_values("anomaly_score")[["date", "daily_total", "anomaly_score"]].head()
        )

    return results


# %%
if __name__ == "__main__":
    # Example usage
    home = path.expanduser("~")
    filepath = home + "/Downloads/test.xlsx"
    results = analyze_dataset(filepath)
