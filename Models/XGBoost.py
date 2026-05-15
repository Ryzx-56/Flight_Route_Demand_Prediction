import os
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    mean_absolute_error, mean_squared_error, r2_score,
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix
)
from xgboost import XGBRegressor
 

MERGED_DATA_PATH = "DATA/International_Report_Merged_Data.csv"
AIRPORTS_PATH    = "DATA/airports.csv"
 
# LOAD DATA
df       = pd.read_csv(MERGED_DATA_PATH)
airports = pd.read_csv(AIRPORTS_PATH)
 
# FEATURE ENGINEERING
df["route"]    = df["us_airport"] + "_" + df["foreign_airport"]
df["route_id"] = df["route"].astype("category").cat.codes
 
airports = airports.dropna(subset=["iata_code"])
airports = airports.drop_duplicates("iata_code")
airports = airports.set_index("iata_code")
 
df["lat_us"]      = df["us_airport"].map(airports["latitude_deg"])
df["lon_us"]      = df["us_airport"].map(airports["longitude_deg"])
df["lat_foreign"] = df["foreign_airport"].map(airports["latitude_deg"])
df["lon_foreign"] = df["foreign_airport"].map(airports["longitude_deg"])
 
df["distance"] = (
    ((df["lat_us"] - df["lat_foreign"])**2 +
     (df["lon_us"] - df["lon_foreign"])**2)**0.5
) * 111
 
df["distance_bin"] = pd.cut(
    df["distance"],
    bins=[0, 2000, 5000, np.inf],
    labels=[1, 2, 3]
)
 
df["distance_bin_label"] = df["distance_bin"].map({
    1: "Short",
    2: "Medium",
    3: "Long"
})
 
print(df[["route_id", "distance", "distance_bin_label"]].head())
 
# BASELINE MODEL (no lag features)
X = df[["route_id", "distance", "distance_bin", "Month"]]
y = df["Total_Passengers"]
 
print(X.head().assign(Total_Passengers=y.head().values).to_string())
 
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
print("Training and test sets are ready")
 
model = XGBRegressor(random_state=42)
model.fit(X_train, y_train)
print("Model finished training")
 
y_pred = model.predict(X_test)
print("Predictions are ready")
 
mae  = mean_absolute_error(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
r2   = r2_score(y_test, y_pred)
 
print("\nBaseline model results")
print(f"MAE:  {mae:.2f}")
print(f"RMSE: {rmse:.2f}")
print(f"R²:   {r2:.4f}")
 
results = pd.DataFrame({"Actual": y_test.values, "Predicted": y_pred})
print(results.head().to_string())
 
# IMPROVED MODEL (with lag features)
df = df.sort_values(by=["route_id", "Year", "Month"])
 
df["lag_1"]  = df.groupby("route_id")["Total_Passengers"].shift(1)
df["lag_12"] = df.groupby("route_id")["Total_Passengers"].shift(12)
 
df_lag = df.dropna(subset=[
    "route_id", "distance", "distance_bin", "Month", "Year",
    "lag_1", "lag_12", "Total_Passengers"
])
 
X_lag_year = df_lag[["route_id", "distance", "distance_bin", "Month", "Year", "lag_1", "lag_12"]]
y_lag_year = df_lag["Total_Passengers"]
 
X_train_year, X_test_year, y_train_year, y_test_year = train_test_split(
    X_lag_year, y_lag_year, test_size=0.2, random_state=42
)
 
model_year = XGBRegressor(random_state=42)
model_year.fit(X_train_year, y_train_year)
 
y_pred_year = model_year.predict(X_test_year)
 
mae_year  = mean_absolute_error(y_test_year, y_pred_year)
rmse_year = np.sqrt(mean_squared_error(y_test_year, y_pred_year))
r2_year   = r2_score(y_test_year, y_pred_year)
 
print("\nImproved model results")
print(f"MAE:  {mae_year:.2f}")
print(f"RMSE: {rmse_year:.2f}")
print(f"R²:   {r2_year:.4f}")
 
# KEEP / REMOVE DECISION (66th percentile threshold)
final_results = pd.DataFrame({
    "Actual": y_test_year.values,
    "Predicted": y_pred_year
})
 
final_results["Demand_Percentage"] = final_results["Predicted"].rank(pct=True) * 100
final_results["Decision"] = np.where(
    final_results["Demand_Percentage"] >= 66, "Fly", "No Fly"
)
 
print(final_results.head(7).to_string())
 
decision_summary = final_results["Decision"].value_counts().reset_index()
decision_summary.columns = ["Decision", "Count"]
decision_summary["Percentage"] = (
    decision_summary["Count"] / decision_summary["Count"].sum() * 100
).round(2)
 
print(decision_summary.to_string())
print("Decision summary is ready")
 
# CLASSIFICATION METRICS
final_results["Actual_Demand_Percentage"] = final_results["Actual"].rank(pct=True) * 100
final_results["Actual_Decision"]          = np.where(
    final_results["Actual_Demand_Percentage"] >= 66, "Fly", "No Fly"
)
final_results["Predicted_Decision"] = final_results["Decision"]
 
accuracy  = accuracy_score(final_results["Actual_Decision"],  final_results["Predicted_Decision"])
precision = precision_score(final_results["Actual_Decision"], final_results["Predicted_Decision"], pos_label="Fly")
recall    = recall_score(final_results["Actual_Decision"],    final_results["Predicted_Decision"], pos_label="Fly")
f1        = f1_score(final_results["Actual_Decision"],        final_results["Predicted_Decision"], pos_label="Fly")
 
metrics_table = pd.DataFrame({
    "Metric": ["Accuracy", "Precision", "Recall", "F1-score"],
    "Value":  [accuracy, precision, recall, f1]
})
 
print(metrics_table.to_string())
print("Classification metrics are ready")
 
# CONFUSION MATRIX
cm = confusion_matrix(
    final_results["Actual_Decision"],
    final_results["Predicted_Decision"],
    labels=["Fly", "No Fly"]
)
 
cm_table = pd.DataFrame(
    cm,
    index=["Actual Fly", "Actual No Fly"],
    columns=["Predicted Fly", "Predicted No Fly"]
)
 
print(cm_table.to_string())
print("Confusion matrix is ready")
 
# VISUALIZATIONS
 
# Keep vs Remove bar chart
plot_data = decision_summary.copy()
plot_data["Route Status"] = plot_data["Decision"].replace({"Fly": "Keep", "No Fly": "Remove"})
 
plt.figure(figsize=(7, 5))
bars = plt.bar(plot_data["Route Status"], plot_data["Count"], color=["steelblue", "tomato"])
 
for bar, count, percentage in zip(bars, plot_data["Count"], plot_data["Percentage"]):
    plt.text(
        bar.get_x() + bar.get_width() / 2,
        bar.get_height(),
        f"{count:,}\n({percentage}%)",
        ha="center", va="bottom"
    )
 
plt.title("Route Status: Keep vs Remove")
plt.xlabel("Route Status")
plt.ylabel("Number of Routes")
plt.ylim(0, plot_data["Count"].max() * 1.15)
plt.tight_layout()
plt.show()
print("Keep vs Remove chart is ready")
 
# Performance metrics bar chart
plt.figure(figsize=(7, 5))
plt.bar(metrics_table["Metric"], metrics_table["Value"])
plt.title("XGBoost Performance Metrics")
plt.xlabel("Metric")
plt.ylabel("Score")
plt.ylim(0, 1)
 
for i, value in enumerate(metrics_table["Value"]):
    plt.text(i, value + 0.02, f"{value:.2f}", ha="center")
 
plt.tight_layout()
plt.show()
print("Metrics chart is ready")
 
# Confusion matrix heatmap
names = [["Correct", "Missed Fly"],
         ["False Fly", "Correct"]]
 
plt.figure(figsize=(6, 5))
plt.imshow(cm)
plt.title("XGBoost Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.xticks([0, 1], ["Fly", "No Fly"])
plt.yticks([0, 1], ["Fly", "No Fly"])
 
for i in range(2):
    for j in range(2):
        plt.text(j, i, f"{cm[i,j]:,}\n{names[i][j]}", ha="center", va="center", fontsize=12)
 
plt.colorbar(label="Number of Routes")
plt.tight_layout()
plt.show()
 
# TIMING
model_time = XGBRegressor(random_state=42)
 
start = time.perf_counter()
model_time.fit(X_train_year, y_train_year)
train_time = time.perf_counter() - start
 
start = time.perf_counter()
model_time.predict(X_test_year)
inference_time = (time.perf_counter() - start) / len(X_test_year) * 1000
 
print(f"\nTraining Time:              {train_time:.2f} seconds")
print(f"Inference Time per sample:  {inference_time:.4f} ms")