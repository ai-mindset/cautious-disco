# XGBoost Time Series Outlier Detection

A system for detecting outliers in time series data using XGBoost regression with dynamic thresholding, optimised for handling periodic patterns.

## Requirements

- Python 3.13
- uv (for dependency management)
- Node.js/Deno (for synthetic data generation)

### Dependencies
See [pyproject.toml](https://github.com/ai-mindset/xgboost_outlier_detection/blob/d8807841e082cc9e4bf27b0cd60cd8da3bd51f78/pyproject.toml#L6) for more details

## Components

### 1. Data Generator
Run using Deno:
```bash
deno run gen_synth_data.js > synthetic_data.csv
```

Generates time series data with:
- Daily observations (100 points from 2023-01-01)
- Upward trend (+0.5/day)
- Two step changes (+20 at day 30, +15 at day 60)
- Cyclical patterns (10-point amplitude)
- Random noise (±1.5 points)

### 2. Outlier Detector

```python
from outlier_detection import detect_outliers, plot_time_series

# Load data
df = pd.read_csv("synthetic_data.csv")

# Detect outliers
df_with_outliers = detect_outliers(df)

# Visualize (saves to plots/time_series_outliers_{sheet_name}.html)
fig = plot_time_series(
    df_with_outliers,
    "Time Series Analysis",
    "dataset_name",
    df_with_outliers["is_outlier"]
)
fig.show()
```

Key Features:
- Temporal feature engineering (day/month/year patterns)
- Dynamic thresholding for periodic spikes
- Interactive Plotly visualizations
- Excel and CSV support

