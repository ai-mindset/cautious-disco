"""Detect outliers in univariate time series data using XGBoost."""

# %%
# Imports

from os import path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from sklearn.preprocessing import StandardScaler
from xgboost import XGBRegressor

import ipdb

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
# # 2. Apply XGBoost-based outlier detection methods to the univariate time series data.
# How to handle periodic spikes in the time series:

# 1. **Added Periodic Features**
# ```python
# df_features["is_month_start"] = df_features["date"].dt.is_month_start.astype(int)
# df_features["is_year_start"] = df_features["date"].dt.is_year_start.astype(int)
# ```
# These binary indicators explicitly capture the first days of months and years where spikes
# may occur.

# 2. **Dynamic Threshold System**
# ```python
# base_threshold = np.mean(residuals) + 3 * np.std(residuals)
# threshold_multiplier = np.ones(len(df))
# ```
# Instead of a single fixed threshold, we now have a base threshold that gets adjusted dynamically.

# 3. **Adaptive Threshold Adjustment**
# ```python
# if df[month_start_mask]["daily_total"].mean() > df["daily_total"].mean() * 1.5:
#     threshold_multiplier[month_start_mask] = 2.0
# if df[year_start_mask]["daily_total"].mean() > df["daily_total"].mean() * 2.0:
#     threshold_multiplier[year_start_mask] = 3.0
# ```
# This is the key change. The code:
# - Checks if month starts show significantly higher values (>50% higher than overall mean)
# - Checks if year starts show significantly higher values (>100% higher than overall mean)
# - Adjusts thresholds accordingly by multiplying them by 2.0 or 3.0

# 4. **Final Outlier Detection**
# ```python
# dynamic_thresholds = base_threshold * threshold_multiplier
# df["is_outlier"] = residuals > dynamic_thresholds
# ```
# Applies the customized thresholds to each point based on its temporal characteristics.

# This approach maintains the algorithm's effectiveness for regular time series while
# preventing false positives on legitimate periodic spikes.


# %%
def detect_outliers(df: pd.DataFrame) -> pd.DataFrame:
    """
    Detect outliers in a univariate time series using XGBoost, accounting for periodic patterns.

    Args:
        df (pd.DataFrame): Input DataFrame with columns 'date' and 'daily_total'.

    Returns:
        pd.DataFrame: Original DataFrame with an additional 'is_outlier' column indicating
        whether each data point is an outlier.
    """
    # Create periodic features
    df["date"] = pd.to_datetime(df["date"], format="mixed")
    df_features = df.copy()
    df_features["day_of_month"] = df_features["date"].dt.day
    df_features["day_of_year"] = df_features["date"].dt.day_of_year
    df_features["is_month_start"] = df_features["date"].dt.is_month_start.astype(int)
    df_features["is_year_start"] = df_features["date"].dt.is_year_start.astype(int)

    # Prepare features for XGBoost
    feature_cols = ["day_of_month", "day_of_year", "is_month_start", "is_year_start"]
    X = df_features[feature_cols].values
    y = df_features["daily_total"].values

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Train XGBoost model with periodic features
    model = XGBRegressor(
        objective="reg:squarederror", random_state=42, n_estimators=100, learning_rate=0.1
    )
    model.fit(X_scaled, y)

    # Predict and calculate residuals
    y_pred = model.predict(X_scaled)
    residuals = np.abs(y - y_pred)

    # Calculate dynamic thresholds based on date characteristics
    base_threshold = np.mean(residuals) + 3 * np.std(residuals)

    # Adjust threshold for month/year starts
    threshold_multiplier = np.ones(len(df))
    month_start_mask = df_features["is_month_start"] == 1
    year_start_mask = df_features["is_year_start"] == 1

    # Increase threshold for month/year starts if periodic pattern exists
    if df[month_start_mask]["daily_total"].mean() > df["daily_total"].mean() * 1.5:
        threshold_multiplier[month_start_mask] = 2.0
    if df[year_start_mask]["daily_total"].mean() > df["daily_total"].mean() * 2.0:
        threshold_multiplier[year_start_mask] = 3.0

    # Apply dynamic thresholds
    dynamic_thresholds = base_threshold * threshold_multiplier
    df["is_outlier"] = residuals > dynamic_thresholds

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
                mode="markers",
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
    # Load synthetic example
    df = pd.read_csv("./synthetic_data.csv")
    time_series = df["daily_total"].to_numpy() if "daily_total" in df.columns else None
    sheet_name = "synthetic_dataset"
    outlier_df = detect_outliers(df)

    # Plot the time series data with or without outliers
    fig = plot_time_series(
        outlier_df,
        f"Time Series with Outliers for {sheet_name}",
        sheet_name,
        outlier_df["is_outlier"],
    )
    fig.show()
    fig.write_html(f"plots/time_series_outliers_{sheet_name}.html")

    # Load the sample univariate time series data
    # df = generate_time_series()
    # sheet_name = "dataset_c"
    # time_series = df[sheet_name]["daily_total"].values

    # for _, df_sheet_name in df.items():
    #     outlier_df_sheet_name = detect_outliers(df_sheet_name)
    #     print(f"Number of outliers detected: {outlier_df_sheet_name['is_outlier'].sum()}")

    #     # Plot the time series data with or without outliers
    #     fig = plot_time_series(
    #         outlier_df_sheet_name,
    #         f"Time Series with Outliers for {sheet_name}",
    #         sheet_name,
    #         outlier_df_sheet_name["is_outlier"],
    #     )
    #     fig.show()
    #     fig.write_html(f"plots/time_series_outliers_{sheet_name}.html")
