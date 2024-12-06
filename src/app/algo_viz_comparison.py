"""Visual comparison of the outlier detection performance of RPCA, MCD, FFT, and SST
on a sample univariate time series dataset."""

# %%
# Imports

from os import path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from sklearn.preprocessing import StandardScaler
from xgboost import XGBRegressor

# %% [markdown]
# 1. Load the sample univariate time series data.


# %%
def generate_time_series() -> dict[str, pd.DataFrame]:
    """Load the sample univariate time series data from an Excel file.

    Args:
        None

    Returns:
        results: Dictionary containing the sample time series data
    """

    home = path.expanduser("~")
    filepath = home + "/Downloads/test.xlsx"

    sheets = pd.read_excel(filepath, sheet_name=None)
    results = {}

    for sheet_name, df in sheets.items():
        df["date"] = pd.to_datetime(df["date"])
        df["daily_total"] = df["daily_total"].astype(float)
        df_clean = df.dropna(subset=["daily_total"])
        results[sheet_name] = df_clean

    return results


# %% [markdown]
# 2. Apply XGBoost-based outlier detection methods to the univariate time series data.


# %%
def detect_outliers(df: pd.DataFrame) -> pd.DataFrame:
    """
    Detect outliers in a univariate time series using XGBoost.

    Args:
        df (pd.DataFrame): Input DataFrame with columns 'dataset', 'date', and 'daily_total'.

    Returns:
        pd.DataFrame: Original DataFrame with an additional 'is_outlier' column indicating
        whether each data point is an outlier.
    """
    # Prepare the data
    X = df["date"].values.reshape(-1, 1)
    y = df["daily_total"].values
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Train the XGBoost model
    model = XGBRegressor(objective="reg:squarederror", random_state=42)
    model.fit(X_scaled, y)

    # Predict on the scaled input and calculate the residuals
    y_pred = model.predict(X_scaled)
    residuals = np.abs(y - y_pred)

    # Identify outliers based on the residuals
    threshold = np.mean(residuals) + 3 * np.std(residuals)
    df["is_outlier"] = residuals > threshold

    return df


# %%

# %% [markdown]
# Plot the univariate time series data with or without outliers


# %%
def plot_time_series(
    df: pd.DataFrame,
    title: str,
    sheet_name: str,
    outliers: np.ndarray | None = None,
) -> go.Figure:
    """Plot the univariate time series data with optional outlier overlay.

    Args:
        df: DataFrame with columns 'date', 'daily_total'
        title: Title of the plot
        sheet_name: Name of the dataset being plotted
        outliers: Optional boolean array indicating outlier positions

    Returns:
        fig: Plotly figure object with time series and optional outliers
    """
    fig = go.Figure()

    if not df.empty:
        # Plot main time series line
        fig.add_trace(
            go.Scatter(
                x=df["date"],
                y=df["daily_total"],
                mode="lines",
                name=f"Time Series for {sheet_name}",
            )
        )

        # Add outlier markers if provided
        if outliers is not None:
            outlier_dates = df["date"][outliers]
            outlier_values = df["daily_total"][outliers]

            fig.add_trace(
                go.Scatter(
                    x=outlier_dates,
                    y=outlier_values,
                    mode="markers",
                    name="Outliers",
                    marker=dict(
                        symbol="x",
                        size=10,
                        color="orange",
                        line=dict(width=2),
                    ),
                )
            )

        # Add hovertemplate to both traces
        fig.add_trace(
            go.Scatter(
                x=df["date"],
                y=df["daily_total"],
                mode="lines",
                name=f"Time Series for {sheet_name}",
                hovertemplate="Date: %{x|%B %d, %Y}<br>" + "Value: %{y:.2f}<br><extra></extra>",
            )
        )

        # Add outlier markers if provided
        if outliers is not None:
            outlier_dates = df["date"][outliers]
            outlier_values = df["daily_total"][outliers]

            fig.add_trace(
                go.Scatter(
                    x=outlier_dates,
                    y=outlier_values,
                    mode="markers",
                    name="Outliers",
                    marker=dict(
                        symbol="x",
                        size=10,
                        color="orange",
                        line=dict(width=2),
                    ),
                    hovertemplate="Date: %{x|%B %d, %Y}<br>" + "Value: %{y:.2f}<br><extra></extra>",
                )
            )

        fig.update_layout(
            title=title,
            xaxis_title="Date",
            yaxis_title="Daily Total",
            width=1200,
            height=600,
            showlegend=True,
        )

    return fig


# %% [markdown]
# 4. Main guard to run the script

# %%
if __name__ == "__main__":
    # Load the sample univariate time series data
    df = generate_time_series()
    sheet_name = "dataset_c"
    time_series = df[sheet_name]["daily_total"].values

    # Example usage
    outlier_df = detect_outliers(df[sheet_name])
    print(f"Number of outliers detected: {outlier_df['is_outlier'].sum()}")

    # Plot the time series data with or without outliers
    fig = plot_time_series(
        outlier_df, "Time Series with Outliers", sheet_name, outlier_df["is_outlier"]
    )
    fig.show()
    fig.write_html("plots/time_series_outliers.html")
