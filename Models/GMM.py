from FeatureEngineering import feature_engineering
import numpy as np
import pandas as pd

# Load data
df = feature_engineering(
    "DATA/International_Report_Departures.csv",
    "DATA/International_Report_Passengers.csv"
)

# Features
features = [
    'Total_Passengers',
    'avg_demand_per_route',
    'distance'
]

df_model = df[features].copy()

df_model['distance'] = df_model['distance'].fillna(df_model['distance'].median())
df_model = df_model.dropna()

df_model['Total_Passengers'] = np.log1p(df_model['Total_Passengers'])


# Scaling
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
X_scaled = scaler.fit_transform(df_model)

# Train GMM
from sklearn.mixture import GaussianMixture

gmm = GaussianMixture(
    n_components=3,
    covariance_type='full',
    random_state=42,
    n_init=10
)

gmm.fit(X_scaled)

labels = gmm.predict(X_scaled)
probs = gmm.predict_proba(X_scaled)

df_model['cluster'] = labels
df_model['confidence'] = probs.max(axis=1)

# Map clusters → Low / Medium / High
cluster_means = df_model.groupby('cluster')['avg_demand_per_route'].mean()
sorted_clusters = cluster_means.sort_values()

label_map = {
    sorted_clusters.index[0]: "Low",
    sorted_clusters.index[1]: "Medium",
    sorted_clusters.index[2]: "High"
}

df_model['demand_label'] = df_model['cluster'].map(label_map)

df_model['Route_Status'] = df_model['demand_label'].map({
    'High': 'Keep',
    'Medium': 'Remove',
    'Low': 'Remove'
})

# PRINT OUTPUT

print("\nDEMAND DISTRIBUTION")
print(df_model['demand_label'].value_counts())

print("\nROUTE STATUS DISTRIBUTION")
print(df_model['Route_Status'].value_counts())

print("\nSAMPLE OUTPUT")
print(df_model[['Total_Passengers', 'demand_label', 'Route_Status', 'confidence']].head(10))


# Model Metrics
from sklearn.metrics import silhouette_score

print("\nMODEL METRICS")
print(f"BIC: {gmm.bic(X_scaled):,.2f}")
print(f"AIC: {gmm.aic(X_scaled):,.2f}")

sample_size = min(10000, len(X_scaled))
sample_idx = np.random.choice(len(X_scaled), sample_size, replace=False)

sil_score = silhouette_score(X_scaled[sample_idx], labels[sample_idx])
print(f"Silhouette Score: {sil_score:.4f}")


print("\nCONFIDENCE STATS PER ROUTE STATUS")
print(df_model.groupby('Route_Status')['confidence'].describe())


# VISUALIZATION
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)

sample_size = min(5000, len(X_pca))
sample_idx = np.random.choice(len(X_pca), sample_size, replace=False)

X_plot = X_pca[sample_idx]
labels_plot = df_model['demand_label'].values[sample_idx]

plt.figure(figsize=(8,6))

for label in ['Low', 'Medium', 'High']:
    mask = labels_plot == label
    plt.scatter(
        X_plot[mask, 0],
        X_plot[mask, 1],
        label=label,
        alpha=0.3,
        s=10
    )

plt.title("Final Stable GMM Clusters")
plt.xlabel("PC1")
plt.ylabel("PC2")
plt.legend()
plt.show()

# ROUTE STATUS VISUALIZATION
status_counts = df_model['Route_Status'].value_counts()

plt.figure(figsize=(6, 5))
bars = plt.bar(status_counts.index, status_counts.values, color=['steelblue', 'tomato'], width=0.5)

for bar, count in zip(bars, status_counts.values):
    plt.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 5000,
             f'{count:,}\n({count/len(df_model)*100:.1f}%)',
             ha='center', va='bottom', fontsize=11)

plt.title("Route Status: Keep vs Remove")
plt.xlabel("Route Status")
plt.ylabel("Number of Routes")
plt.tight_layout()
plt.show()

# CLUSTER PROBABILITY HEATMAP
import seaborn as sns

prob_df = pd.DataFrame(probs, columns=['Cluster 0', 'Cluster 1', 'Cluster 2'])
prob_df['demand_label'] = df_model['demand_label'].values

heatmap_data = prob_df.groupby('demand_label')[['Cluster 0', 'Cluster 1', 'Cluster 2']].mean()

plt.figure(figsize=(7, 4))
sns.heatmap(
    heatmap_data,
    annot=True,
    fmt='.3f',
    cmap='Blues',
    linewidths=0.5
)
plt.title("GMM Cluster Probability Heatmap")
plt.ylabel("Demand Label")
plt.xlabel("Cluster")
plt.tight_layout()
plt.show()

# BIC / AIC CHECK

bic_scores = []
aic_scores = []
k_range = range(2, 7)

for k in k_range:
    g = GaussianMixture(n_components=k, covariance_type='diag', random_state=42)
    g.fit(X_scaled)
    bic_scores.append(g.bic(X_scaled))
    aic_scores.append(g.aic(X_scaled))

plt.plot(k_range, bic_scores, label='BIC', marker='o')
plt.plot(k_range, aic_scores, label='AIC', marker='s', linestyle='--')
plt.xlabel("n_components")
plt.title("BIC / AIC vs Number of Components")
plt.legend()
plt.show()

# CLUSTER PROFILES
profile = df_model.groupby('demand_label')[features].mean()
print("\nCLUSTER PROFILES")
print(profile)

# TRAINING TIME CALCULATION
import time

start = time.time()
gmm.fit(X_scaled)
train_time = time.time() - start
print(f"Training Time: {train_time:.2f} seconds")


# INFERENCE TIME
start = time.time()
labels = gmm.predict(X_scaled)
inference_time = time.time() - start
ms_per_sample = (inference_time / len(X_scaled)) * 1000
print(f"Inference Time: {ms_per_sample:.4f} ms per sample")


# ─────────────────────────────────────────
# CLASSIFICATION METRICS + CONFUSION MATRIX
# ─────────────────────────────────────────
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, classification_report, confusion_matrix
)
import seaborn as sns
import matplotlib.pyplot as plt

# Step 1: Create ground truth labels by binning avg_demand_per_route
df_model['y_true'] = pd.qcut(
    df_model['avg_demand_per_route'],
    q=3,
    labels=['Low', 'Medium', 'High']
)

# Step 2: Encode both to numbers for metric functions
label_encoder = {'Low': 0, 'Medium': 1, 'High': 2}
y_true = df_model['y_true'].map(label_encoder).values
y_pred = df_model['demand_label'].map(label_encoder).values

# Step 3: Print all metrics
print("\nCLASSIFICATION METRICS")
print(f"Accuracy:  {accuracy_score(y_true, y_pred):.4f}")
print(f"Precision: {precision_score(y_true, y_pred, average='weighted'):.4f}")
print(f"Recall:    {recall_score(y_true, y_pred, average='weighted'):.4f}")
print(f"F1 Score:  {f1_score(y_true, y_pred, average='weighted'):.4f}")

print("\nDETAILED REPORT")
print(classification_report(y_true, y_pred, target_names=['Low', 'Medium', 'High']))

# Step 4: Confusion Matrix
cm = confusion_matrix(y_true, y_pred)

plt.figure(figsize=(7, 5))
sns.heatmap(
    cm,
    annot=True,
    fmt='d',
    cmap='Blues',
    xticklabels=['Low', 'Medium', 'High'],
    yticklabels=['Low', 'Medium', 'High'],
    linewidths=0.5
)
plt.title("Confusion Matrix - GMM Demand Classification")
plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.tight_layout()
plt.show()