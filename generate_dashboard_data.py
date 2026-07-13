"""
generate_dashboard_data.py

Run this ONCE, in the same folder as train.csv, after finishing Task 6.
It reuses your Task 1/3/5/6 logic exactly, adds real 1-3 month future
forecasting (recursive XGBoost), and saves everything the Streamlit
dashboard needs into a data/ folder as clean CSVs.

Usage (from Command Prompt, in your project folder):
    python generate_dashboard_data.py
"""

import pandas as pd
import numpy as np
import os

os.makedirs("data", exist_ok=True)

# ============================================
# Task 1 — Load & Clean (same as analysis.ipynb)
# ============================================
df = pd.read_csv("train.csv")
df["Order Date"] = pd.to_datetime(df["Order Date"], format="%d/%m/%Y")
df["Ship Date"] = pd.to_datetime(df["Ship Date"], format="%d/%m/%Y")
df["Year"] = df["Order Date"].dt.year
df["Month"] = df["Order Date"].dt.month
df["Month Name"] = df["Order Date"].dt.month_name()
df["Quarter"] = df["Order Date"].dt.quarter

df.to_csv("data/clean_sales.csv", index=False)
print("Saved data/clean_sales.csv")

# ============================================
# Task 3/7 — Best Model: XGBoost, Recursive Future Forecast
# ============================================
from xgboost import XGBRegressor


def build_monthly_series(data):
    return data.groupby(pd.Grouper(key="Order Date", freq="M"))["Sales"].sum()


def make_features(ts):
    fdf = ts.to_frame(name="Sales")
    fdf["Lag_1"] = fdf["Sales"].shift(1)
    fdf["Lag_2"] = fdf["Sales"].shift(2)
    fdf["Lag_3"] = fdf["Sales"].shift(3)
    fdf["Rolling_Mean_3"] = fdf["Sales"].rolling(3).mean()
    fdf["Month"] = fdf.index.month
    fdf["Year"] = fdf.index.year
    return fdf.dropna()


def recursive_forecast(ts, horizon=3):
    """
    Trains XGBoost on the FULL known monthly series (same hyperparameters
    and features as Task 3), then forecasts `horizon` months forward by
    feeding each prediction back in as the next lag/rolling mean.
    """
    history = ts.copy()
    forecasts = []

    feat_df = make_features(history)
    X_train = feat_df.drop(columns="Sales")
    y_train = feat_df["Sales"]

    model = XGBRegressor(
        n_estimators=100,
        learning_rate=0.1,
        max_depth=3,
        random_state=42
    )
    model.fit(X_train, y_train)

    for _ in range(horizon):
        next_date = history.index[-1] + pd.DateOffset(months=1)
        lag_1 = history.iloc[-1]
        lag_2 = history.iloc[-2]
        lag_3 = history.iloc[-3]
        rolling_mean_3 = history.iloc[-3:].mean()

        X_next = pd.DataFrame([{
            "Lag_1": lag_1,
            "Lag_2": lag_2,
            "Lag_3": lag_3,
            "Rolling_Mean_3": rolling_mean_3,
            "Month": next_date.month,
            "Year": next_date.year
        }])

        pred = model.predict(X_next)[0]
        forecasts.append({"Month": next_date.strftime("%Y-%m"), "Forecast": round(float(pred), 2)})
        history.loc[next_date] = pred

    return forecasts


# ---- Overall sales forecast ----
overall_ts = build_monthly_series(df)
overall_forecast = recursive_forecast(overall_ts, horizon=3)
overall_df = pd.DataFrame(overall_forecast)
overall_df["Segment"] = "Overall"
overall_df["Type"] = "Overall"

# ---- Category & Region level forecasts (all categories/regions, not just 5) ----
segment_rows = []
for cat in df["Category"].unique():
    seg_ts = build_monthly_series(df[df["Category"] == cat])
    for row in recursive_forecast(seg_ts, horizon=3):
        row["Segment"] = cat
        row["Type"] = "Category"
        segment_rows.append(row)

for reg in df["Region"].unique():
    seg_ts = build_monthly_series(df[df["Region"] == reg])
    for row in recursive_forecast(seg_ts, horizon=3):
        row["Segment"] = reg
        row["Type"] = "Region"
        segment_rows.append(row)

segment_df = pd.DataFrame(segment_rows)
all_forecasts = pd.concat([overall_df, segment_df], ignore_index=True)
all_forecasts.to_csv("data/forecasts.csv", index=False)
print("Saved data/forecasts.csv")

# ---- Model metrics (from your Task 3 test evaluation) ----
model_metrics = pd.DataFrame([
    {"Model": "SARIMA",  "MAE": 51585.634271, "RMSE": 55573.613822, "MAPE": 63.538688},
    {"Model": "Prophet", "MAE": 14309.986950, "RMSE": 18954.579462, "MAPE": 17.469222},
    {"Model": "XGBoost", "MAE": 9812.411198,  "RMSE": 14124.749140, "MAPE": 11.180919},
])
model_metrics.to_csv("data/model_metrics.csv", index=False)
print("Saved data/model_metrics.csv")

# ============================================
# Task 5 — Anomaly Detection (Isolation Forest + Z-Score)
# ============================================
from sklearn.ensemble import IsolationForest

weekly_sales = (
    df.groupby(pd.Grouper(key="Order Date", freq="W"))["Sales"]
      .sum()
      .reset_index()
)

iso = IsolationForest(contamination=0.05, random_state=42)
weekly_sales["Anomaly"] = iso.fit_predict(weekly_sales[["Sales"]])

rolling_mean = weekly_sales["Sales"].rolling(window=4).mean()
rolling_std = weekly_sales["Sales"].rolling(window=4).std()
weekly_sales["Z_Score"] = (weekly_sales["Sales"] - rolling_mean) / rolling_std
weekly_sales["Z_Anomaly"] = weekly_sales["Z_Score"].abs() > 2

weekly_sales.to_csv("data/weekly_anomalies.csv", index=False)
print("Saved data/weekly_anomalies.csv")

# ============================================
# Task 6 — Product Demand Clustering
# ============================================
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA

monthly_subcat = (
    df.groupby(["Sub-Category", pd.Grouper(key="Order Date", freq="M")])["Sales"]
      .sum()
      .reset_index()
)

feature_df = monthly_subcat.groupby("Sub-Category").agg(
    Total_Sales=("Sales", "sum"),
    Avg_Order_Value=("Sales", "mean"),
    Sales_Volatility=("Sales", "std"),
    Growth_Rate=("Sales", lambda x: (x.iloc[-1] - x.iloc[0]) / x.iloc[0] if x.iloc[0] != 0 else 0)
).fillna(0)

scaler = StandardScaler()
scaled_features = scaler.fit_transform(feature_df)

kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
feature_df["Cluster"] = kmeans.fit_predict(scaled_features)

pca = PCA(n_components=2)
components = pca.fit_transform(scaled_features)
feature_df["PC1"] = components[:, 0]
feature_df["PC2"] = components[:, 1]

# Matches your Cluster Labels markdown in Task 6
cluster_labels = {
    0: "Low Volume, Stable Demand",
    1: "High Volume, Stable Demand",
    2: "Growing Demand"
}
feature_df["Cluster_Label"] = feature_df["Cluster"].map(cluster_labels)

feature_df.reset_index().to_csv("data/clusters.csv", index=False)
print("Saved data/clusters.csv")

print("\nAll done! Check the data/ folder — you should see 5 CSV files:")
print(" - clean_sales.csv")
print(" - forecasts.csv")
print(" - model_metrics.csv")
print(" - weekly_anomalies.csv")
print(" - clusters.csv")
