# ✈️ Flight Route Demand Prediction
 
Four ML models trained on 880K+ BTS flight records to classify airline route demand and recommend which routes to keep or cut. Built as a comparative study across unsupervised clustering, supervised learning, and deep learning — benchmarked against a hybrid LSTM-Dense state-of-the-art baseline.
 
---
 
## The Problem
 
Airlines commit aircraft and crew to routes months in advance. By the time a route proves unprofitable, the damage is done. Most flights need a 75–80% load factor just to break even, and the industry's average net margin sits around 3.6% — a handful of underloaded routes can erase it.
 
This project trains models on historical passenger data to flag routes as High, Medium, or Low demand, then outputs a simple operational decision: **Keep** or **Remove**.
 
---
 
## Dataset
 
**BTS US International Air Travel Statistics**
 
Three source files, merged and cleaned into a single pipeline:
 
| File | Contents |
|---|---|
| `International_Report_Departures.csv` | Scheduled and charter departure counts per route and month |
| `International_Report_Passengers.csv` | Scheduled and charter passenger counts per route and month |
| `airports.csv` | Airport coordinates used to calculate route distances |
 
- Raw records: 1.5M+
- After preprocessing: ~880K rows
- Date range: Multi-year US international flight history
The dataset was sourced and assembled by **Abdulmalik Hawsawi**.
 
---
 
## Team
 
| Name | Role |
|---|---|
| **Abdulmalik Y. Hawsawi** | Team Lead · Dataset sourcing · Full preprocessing pipeline · Data merge · Feature engineering (avg_demand_per_route) · GMM model · Overall comparison presenter |
| **Hamed H. Al-Ansari** | GRU model · Feature engineering (demand sequence construction) |
| **Nasser H. Qahhat** | XGBoost model · Feature engineering (distance calculation and binning) |
| **Waleed A. Al-Jaser** | SOA benchmark (Hybrid LSTM-Dense) |
 
---
 
## Solutions
 
| Solution | Type | Author |
|---|---|---|
| **GMM** — Gaussian Mixture Model | Unsupervised ML | Abdulmalik Y. Hawsawi |
| **GRU** — Gated Recurrent Unit | Supervised Deep Learning | Hamed H. Al-Ansari |
| **XGBoost** — Gradient Boosted Trees | Supervised ML | Nasser H. Qahhat |
| **Hybrid LSTM-Dense** | SOA Benchmark | Waleed A. Al-Jaser |
 
Each model was built independently and evaluated on the same cleaned dataset, then compared head-to-head.
 
---
 
## Results
 
### Performance Metrics
 
| Model | Accuracy | Precision | Recall | F1-Score | Train Time | Inference |
|---|---|---|---|---|---|---|
| GMM | 69% | 73% | 69% | 68% | 39.88s | 0.0001ms/sample |
| GRU | 94.8% | 93.0% | 94.0% | 93.0% | 36m 30s | 6s |
| XGBoost | 85.6% | 78.8% | 78.8% | 78.8% | 1.32s | 0.0008ms/sample |
| SOA (LSTM-Dense) | 98.0% | 99.0% | 98.0% | 98.0% | 21m 2s | 12.9s |
 
> ¹ GMM is unsupervised — standard classification metrics don't apply. It's validated through Silhouette Score, BIC/AIC, and cluster profile separation (see below).
 
### GMM Cluster Metrics
 
| Metric | Value |
|---|---|
| Silhouette Score | 0.3857 |
| BIC | 4,783,161.45 |
| AIC | 4,782,822.35 |
| Low demand cluster avg passengers | 651 |
| High demand cluster avg passengers | 6,999 |
 
The ~10x passenger gap between Low and High clusters was found without any labels — purely from the data's structure.
 
### Keep vs. Remove Distribution
 
| Model | Keep % | Remove % |
|---|---|---|
| SOA | 3.4% | 96.6% |
| GMM | 17.3% | 82.7% |
| GRU | 24.7% | 75.3% |
| XGBoost | 34.0% | 66.0% |
 
---
 
## Visualizations
 
> Add your output images to an `images/` folder in the repo root and they will render here automatically.
 
### GMM Demand Clusters (PCA)
![GMM Clusters](images/gmm_clusters.png)
 
### BIC / AIC vs Number of Components
![BIC AIC Curve](images/bic_aic_curve.png)
 
### GMM Cluster Probability Heatmap
![GMM Heatmap](images/gmm_heatmap.png)
 
### Keep vs. Remove — GMM
![Keep vs Remove GMM](images/keep_vs_remove_gmm.png)
 
### Keep vs. Remove — GRU
![Keep vs Remove GRU](images/keep_vs_remove_gru.png)
 
### GRU Confusion Matrix
![GRU Confusion Matrix](images/gru_confusion_matrix.png)
 
### Keep vs. Remove — XGBoost
![Keep vs Remove XGBoost](images/keep_vs_remove_xgboost.png)
 
### XGBoost Confusion Matrix
![XGBoost Confusion Matrix](images/xgboost_confusion_matrix.png)
 
### Keep vs. Remove — SOA (LSTM-Dense)
![Keep vs Remove SOA](images/keep_vs_remove_soa.png)
 
---
 
## Project Structure
 
```
flight-route-demand-prediction/
│
├── README.md
├── requirements.txt
│
├── Preprocessing.py          # Data loading, merging, and cascading imputation
├── FeatureEngineering.py     # Shared feature pipeline (distance, demand, lag features)
│
├── GMM.py                    # Solution 1 — Gaussian Mixture Model
├── GRU_project.py            # Solution 2 — GRU sequence classifier
├── XGBoost.py                # Solution 3 — XGBoost regressor + decision threshold
├── LSTM_Dense22.py           # SOA Benchmark — Hybrid LSTM-Dense
│
├── images/                   # Visualization outputs (add your plots here)
│
└── DATA/
    ├── International_Report_Departures.csv
    ├── International_Report_Passengers.csv
    └── airports.csv
```
 
---
 
## Setup
 
Python 3.9–3.11 required. TensorFlow does not support Python 3.12+.
 
```bash
# Create and activate a virtual environment
python -m venv venv
source venv/bin/activate        # macOS / Linux
venv\Scripts\activate           # Windows
 
# Install dependencies
pip install -r requirements.txt
```
 
**`requirements.txt`**
```
numpy
pandas
scikit-learn
matplotlib
seaborn
xgboost
tensorflow
```
 
### Data Setup
 
Place the three dataset files inside a `DATA/` folder in the project root:
 
```
DATA/
├── International_Report_Departures.csv
├── International_Report_Passengers.csv
└── airports.csv
```
 
---
 
## Running the Models
 
All scripts depend on `Preprocessing.py` and `FeatureEngineering.py` being in the same directory.
 
```bash
# GMM (unsupervised clustering)
python GMM.py
 
# GRU (sequence-based deep learning)
python GRU_project.py
 
# XGBoost (gradient boosted trees)
python XGBoost.py
 
# SOA Benchmark (Hybrid LSTM-Dense)
python LSTM_Dense22.py
```
 
Each script handles its own training, evaluation, and visualization. Results print to the console and plots open via matplotlib.
 
> **XGBoost note:** Update the two path variables at the top of `XGBoost.py` to point to your local data files before running.
 
---
 
## What We Found
 
**GRU beat XGBoost by 9 points, which surprised everyone.** XGBoost dominates tabular benchmarks — it was the expected winner. But airline demand is sequential, not tabular. GRU's hidden state carries 12 months of context automatically. XGBoost needed lag_12 engineered manually, and even with that it couldn't replicate what GRU learned from the sequence directly.
 
**GMM found a real structure without any labels.** The Low and High clusters differ by roughly 10x in average passenger volume. That gap wasn't given to the model — it found it on its own. A Silhouette Score of 0.3857 confirms the clusters are meaningfully distinct, not arbitrary, which is a solid result for overlapping real-world airline data.
 
**Every model learned a different definition of "worthwhile."** Keep rates range from 3.4% (SOA) to 34% (XGBoost) on the same dataset. The threshold choice matters as much as the model choice.
 
**The preprocessing pipeline recovered ~207K rows that would otherwise have been dropped.** Cascading grouped-median imputation — first by route+airline+month, then by route+month, then by route — filled missing passenger values at three levels of granularity before a final drop of ~43K rows that couldn't be recovered. Full details are in `Preprocessing.py`.
 
---
 
## Limitations
 
- No cost or revenue data in the dataset. Models classify demand, not profitability. A route can be high-demand and still lose money.
- XGBoost's 34% Keep rate suggests its threshold may be too lenient for real operational use.
- GRU requires at least 12 months of route history to build a sequence — new routes can't be scored.
---
 
*University of Jeddah — College of Computer Science & Engineering*  
*CCAI323 Machine Learning | Fall 2026*
