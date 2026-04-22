# 📡 ChurnScope — SaaS Customer Churn Prediction

A full-stack machine learning project that predicts customer churn for a SaaS streaming platform. Built with a custom sklearn pipeline, SHAP-based explainability, and deployed as an interactive Streamlit dashboard.

🔗 **Live App:** [saas-churn-prediction-withshap.streamlit.app](https://saas-churn-prediction-withshap.streamlit.app/)

---

## 📁 Project Structure

```
├── dashboard.py                    # Streamlit app (ChurnScope)
├── SaaS_Churning_Prediction.ipynb  # End-to-end ML notebook
├── feature_engineering.py          # Custom sklearn transformers
├── churn_model.pkl                 # Serialized pipeline + threshold
├── train.csv                       # Training dataset
├── sample_data.csv                 # Sample CSV for batch/insights demo
└── requirements.txt                # Python dependencies
```

---

## 🌐 Live Dashboard — ChurnScope

The deployed app has three pages:

### 🔍 Single Prediction
Enter any customer's attributes through a form (account info, usage patterns, profile) and instantly get:
- **Churn probability** with a visual progress bar
- **Risk category** — Low / Medium / High Risk
- **Model decision** — Will Retain or Will Churn
- **SHAP waterfall plot** — explains which features pushed the prediction up or down for that individual customer
- **Retention insights** — rule-based flags like high support volume, low engagement, new user, or premium underuse

### 📂 Batch CSV Upload
Upload a CSV of customer records to score in bulk:
- Summary metrics: total customers, high/medium risk counts, average churn probability
- Probability distribution histogram with the decision threshold marked
- Risk breakdown bar (stacked Low / Medium / High)
- Sortable scored table with `Churn_Probability`, `Churn_Prediction`, and `Risk_Category` columns
- Downloadable results as `churn_predictions.csv`

### 📊 Model Insights
Upload a sample dataset to generate global SHAP explanations across three tabs:
- **Bar plot** — top features ranked by mean absolute SHAP value
- **Beeswarm** — distribution of SHAP impacts per feature across all samples
- **Heatmap** — per-sample SHAP intensity across the top features

A feature engineering summary is also shown on this page explaining every derived feature.

---

## 📊 Dataset

**Source:** [Predictive Analytics for Customer Churn Dataset](https://www.kaggle.com/datasets/safrin03/predictive-analytics-for-customer-churn-dataset) — Kaggle (by safrin03)

| Stat | Value |
|---|---|
| Total rows | 243,787 |
| Features | 21 (20 inputs + 1 target) |
| Churn rate | ~18.1% (class imbalance handled with SMOTE) |

### Feature Overview

| Category | Columns |
|---|---|
| **Account** | `CustomerID`, `AccountAge`, `SubscriptionType`, `MonthlyCharges`, `TotalCharges` |
| **Engagement** | `ViewingHoursPerWeek`, `AverageViewingDuration`, `ContentDownloadsPerMonth`, `WatchlistSize` |
| **Content** | `ContentType`, `GenrePreference` |
| **Device** | `MultiDeviceAccess`, `DeviceRegistered` |
| **Support** | `SupportTicketsPerMonth`, `UserRating` |
| **Demographics** | `Gender`, `ParentalControl`, `SubtitlesEnabled`, `PaperlessBilling`, `PaymentMethod` |
| **Target** | `Churn` (0 = retained, 1 = churned) |

---

## ⚙️ Feature Engineering (`feature_engineering.py`)

Two custom sklearn-compatible transformers handle all preprocessing:

### `FeatureConstructor`

Derives 11 behavioral and financial signal features from the raw inputs:

| Feature | Formula / Logic | Signal |
|---|---|---|
| `engagement_score` | `ViewingHours + AvgDuration + Downloads` | Overall usage intensity |
| `cost_per_hour` | `MonthlyCharges / (ViewingHours + 1)` | Value perception |
| `watch_intensity` | `ViewingHours / (AccountAge + 1)` | Engagement relative to tenure |
| `support_to_usage` | `SupportTickets / (ViewingHours + 1)` | Frustration-to-use ratio |
| `log_total_charges` | `log1p(TotalCharges)` | Stabilises right-skewed distribution |
| `watchlist_to_watch_ratio` | `WatchlistSize / (ViewingHours + 1)` | Intent vs actual viewing |
| `session_depth` | `AvgDuration / (ViewingHours + 1)` | Session quality |
| `premium_underuse` | Premium subscriber with low watch hours (< 30th percentile) | Churn risk flag |
| `frustration_index` | `SupportTickets × (6 − UserRating)` | Combined dissatisfaction signal |
| `loyalty_tier` | `AccountAge` binned into 5 ordinal tiers | Retention signal |
| `is_new_user` | `AccountAge ≤ 3 months` | Early churn detection flag |

> `low_watch_threshold` (30th percentile of `ViewingHoursPerWeek`) is learned during `fit()` to prevent data leakage.

### `OutlierHandler`

A per-column outlier clipper that auto-selects its strategy based on skewness:
- **Z-score clipping** (±2σ) for near-normal distributions (skew between -0.5 and 0.5)
- **IQR clipping** (1.5× IQR) for skewed distributions

---

## 🔁 ML Pipeline

```
FeatureConstructor → OutlierHandler (TotalCharges) → ColumnTransformer → SMOTE → Logistic Regression
```

| Step | Detail |
|---|---|
| `FeatureConstructor` | Adds 11 engineered features, drops `CustomerID` |
| `OutlierHandler` | Clips `TotalCharges` outliers using auto-selected method |
| `ColumnTransformer` | OneHotEncoder for nominal categoricals; OrdinalEncoder for `SubscriptionType` & `loyalty_tier` |
| `SMOTE` | Oversamples minority class inside the pipeline (no leakage) |
| `LogisticRegression` | `C=10`, `max_iter=1000` |

The full pipeline is built with `imblearn.pipeline.Pipeline` so SMOTE is applied only to training folds.

---

## 🎯 Threshold Tuning

The default 0.5 threshold was replaced with **0.35**, selected by scanning thresholds from 0.05 to 1.0 and evaluating confusion matrices at each step. This trades a small number of false positives for significantly better recall on churned customers — prioritising catching at-risk users over minimising false alarms.

---

## 🔍 Explainability (SHAP)

`shap.LinearExplainer` is used to interpret the model:

- **Waterfall plot** — per-prediction explanation showing how each feature pushed the probability up or down from the base value
- **Bar plot** — global feature importance ranked by mean absolute SHAP value
- **Beeswarm plot** — distribution of feature impacts across the full dataset
- **Heatmap** — per-sample SHAP intensity across top features

---

## 💾 Model Artifact

The fitted pipeline and tuned threshold are saved together in `churn_model.pkl`:

```python
import joblib

# Save
joblib.dump({"model": pipeline, "threshold": 0.35}, "churn_model.pkl")

# Load and predict
artifact  = joblib.load("churn_model.pkl")
model     = artifact["model"]
threshold = artifact["threshold"]

y_proba = model.predict_proba(X_new)[:, 1]
y_pred  = (y_proba >= threshold).astype(int)
```

---

## 🚀 Run Locally

### 1. Clone the repository

```bash
git clone https://github.com/your-username/saas-churn-prediction.git
cd saas-churn-prediction
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Launch the Streamlit app

```bash
streamlit run dashboard.py
```

Make sure `churn_model.pkl` and `feature_engineering.py` are in the same directory as `dashboard.py`.

### 4. Run the notebook (optional)

```bash
jupyter notebook SaaS_Churning_Prediction.ipynb
```

Place `train.csv` under a `data/` folder, or update the path in the first data-loading cell.

---

## 📦 Dependencies

| Package | Version |
|---|---|
| `streamlit` | 1.56.0 |
| `scikit-learn` | 1.8.0 |
| `imbalanced-learn` | 0.14.1 |
| `shap` | 0.50.0 |
| `pandas` | 3.0.1 |
| `numpy` | 2.4.2 |
| `matplotlib` | 3.10.8 |
| `seaborn` | 0.13.2 |
| `joblib` | 1.5.3 |

---

## ⚠️ Notes

- `feature_engineering.py` must be importable from wherever `churn_model.pkl` is loaded — keep it in the same directory or on your `PYTHONPATH`.
- The `FeatureConstructor` learns `low_watch_threshold` from training data during `fit()`. Never call `transform()` before `fit()` on a fresh instance.
- `CustomerID` is automatically dropped inside `FeatureConstructor.transform()`.
- For batch predictions, the input CSV must have the same column names as the training data (minus `Churn`). Use `sample_data.csv` as a reference template.

---

## 📄 License

This project is for educational purposes. Dataset credit: [safrin03 on Kaggle](https://www.kaggle.com/datasets/safrin03/predictive-analytics-for-customer-churn-dataset).
