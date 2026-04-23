# 📡 ChurnScope — SaaS Customer Churn Prediction

A full-stack machine learning project that predicts customer churn for a SaaS streaming platform. Built with a custom sklearn pipeline, SHAP-based explainability, and deployed as an interactive Streamlit dashboard.

🔗 **Live App:** [saas-churn-prediction-withshap.streamlit.app](https://saas-churn-prediction-withshap.streamlit.app/)

---

## 📁 Project Structure

```
├── dashboard/                      # Streamlit app (ChurnScope)
├── Notebook/                       # End-to-end ML notebook
├── Images/                         # Dashboard screenshots
├── sample data/                    # Sample CSV for batch/insights demo
├── feature_engineering.py          # Custom sklearn transformers
├── churn_model.pkl                 # Serialized pipeline + threshold
└── requirements.txt                # Python dependencies
```

---

## 🌐 Live Dashboard — ChurnScope

### 🔍 Single Prediction

Enter any customer's attributes and instantly get churn probability, risk category, model decision, a SHAP waterfall explanation, and rule-based retention insights.

![Single Prediction](https://raw.githubusercontent.com/ShazanRafi/saas-churn-prediction/main/Images/image1.png)

---

### 📂 Batch CSV Upload

Upload a CSV of customer records to score in bulk. The dashboard shows a probability distribution histogram with the decision threshold marked, a stacked risk breakdown bar (Low / Medium / High), and a sortable scored table — all downloadable as `churn_predictions.csv`.

![Batch Probability Distribution](https://raw.githubusercontent.com/ShazanRafi/saas-churn-prediction/main/Images/image2.png)

![Batch Scored Records](https://raw.githubusercontent.com/ShazanRafi/saas-churn-prediction/main/Images/image3.png)

---

### 📊 Model Insights — SHAP

Upload a sample dataset to generate global SHAP explanations. The bar plot ranks features by mean absolute SHAP value; the beeswarm shows how feature values drive predictions across all customers.

![SHAP Bar Plot](https://raw.githubusercontent.com/ShazanRafi/saas-churn-prediction/main/Images/image4.png)

![SHAP Beeswarm Plot](https://raw.githubusercontent.com/ShazanRafi/saas-churn-prediction/main/Images/image5.png)

---

## 📊 Dataset

**Source:** [Predictive Analytics for Customer Churn Dataset](https://www.kaggle.com/datasets/safrin03/predictive-analytics-for-customer-churn-dataset) — Kaggle (by safrin03)

| Stat | Value |
|---|---|
| Total rows | 243,787 |
| Features | 21 (20 inputs + 1 target) |
| Churn rate | ~18.1% |
| Train / Test split | 80% / 20% (`random_state=42`) |

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

## 📈 Model Performance

### Key Metrics

| Metric | Value |
|---|---|
| **ROC-AUC** | **0.7527** |
| Decision Threshold | 0.35 (tuned) |
| Test Set Size | 48,758 customers |

### Classification Report (Threshold = 0.35)

| Class | Precision | Recall | F1-Score | Support |
|---|---|---|---|---|
| **0 — Retained** | 0.94 | 0.45 | 0.61 | 39,968 |
| **1 — Churned** | 0.26 | 0.87 | 0.40 | 8,790 |
| **Weighted Avg** | 0.82 | 0.53 | 0.57 | 48,758 |

### Confusion Matrix (Threshold = 0.35)

```
                   Predicted: Retain   Predicted: Churn
Actual: Retain         18,123              21,845
Actual: Churn           1,151               7,639
```

### Why Threshold 0.35?

The default 0.5 threshold was replaced with **0.35** after scanning confusion matrices from 0.05 to 1.0. At 0.35 the model achieves **87% recall on churned customers** — catching the vast majority of at-risk users. The trade-off is lower precision (more false positives), which is acceptable since missing a churner costs far more than a wrongly flagged retained customer.

### Top Features by SHAP Importance

| Rank | Feature | Mean \|SHAP\| |
|---|---|---|
| 1 | `engagement_score` | +1.01 |
| 2 | `AccountAge` | +0.63 |
| 3 | `AverageViewingDuration` | +0.54 |
| 4 | `SupportTicketsPerMonth` | +0.32 |
| 5 | `log_total_charges` | +0.29 |
| 6 | `ViewingHoursPerWeek` | +0.23 |
| 7 | `MonthlyCharges` | +0.20 |
| 8 | `loyalty_tier` | +0.11 |

---

## ⚙️ Feature Engineering (`feature_engineering.py`)

### `FeatureConstructor`

Derives 11 behavioral and financial signal features from raw inputs:

| Feature | Formula / Logic | Signal |
|---|---|---|
| `engagement_score` | `ViewingHours + AvgDuration + Downloads` | Overall usage intensity |
| `cost_per_hour` | `MonthlyCharges / (ViewingHours + 1)` | Value perception |
| `watch_intensity` | `ViewingHours / (AccountAge + 1)` | Engagement relative to tenure |
| `support_to_usage` | `SupportTickets / (ViewingHours + 1)` | Frustration-to-use ratio |
| `log_total_charges` | `log1p(TotalCharges)` | Stabilises right-skewed distribution |
| `watchlist_to_watch_ratio` | `WatchlistSize / (ViewingHours + 1)` | Intent vs actual viewing |
| `session_depth` | `AvgDuration / (ViewingHours + 1)` | Session quality |
| `premium_underuse` | Premium subscriber with watch hours < 30th percentile | Churn risk flag |
| `frustration_index` | `SupportTickets × (6 − UserRating)` | Combined dissatisfaction signal |
| `loyalty_tier` | `AccountAge` binned into 5 ordinal tiers | Retention signal |
| `is_new_user` | `AccountAge ≤ 3 months` | Early churn detection flag |

> `low_watch_threshold` (30th percentile of `ViewingHoursPerWeek`) is learned during `fit()` to prevent data leakage.

### `OutlierHandler`

Auto-selects clipping strategy based on feature skewness — **Z-score** (±2σ) for near-normal distributions, **IQR** (1.5×) for skewed ones.

---

## 🔁 ML Pipeline

```
FeatureConstructor → OutlierHandler → ColumnTransformer → SMOTE → Logistic Regression
```

| Step | Detail |
|---|---|
| `FeatureConstructor` | Adds 11 engineered features, drops `CustomerID` |
| `OutlierHandler` | Clips `TotalCharges` outliers using auto-selected method |
| `ColumnTransformer` | OneHotEncoder for nominal categoricals; OrdinalEncoder for `SubscriptionType` & `loyalty_tier` |
| `SMOTE` | Oversamples minority class inside the pipeline (no data leakage) |
| `LogisticRegression` | `C=10`, `max_iter=1000` |

---

## 💾 Load & Predict

```python
import joblib

artifact  = joblib.load("churn_model.pkl")
model     = artifact["model"]
threshold = artifact["threshold"]  # 0.35

y_proba = model.predict_proba(X_new)[:, 1]
y_pred  = (y_proba >= threshold).astype(int)
```

---

## 🚀 Run Locally

> The app is already live — run locally only if you want to retrain or modify the code.

```bash
git clone https://github.com/ShazanRafi/saas-churn-prediction.git
cd saas-churn-prediction
pip install -r requirements.txt
streamlit run dashboard/dashboard.py
```

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

- `feature_engineering.py` must be in the same directory as `churn_model.pkl` or on your `PYTHONPATH`.
- `FeatureConstructor` learns `low_watch_threshold` during `fit()` — never call `transform()` before `fit()` on a fresh instance.
- `CustomerID` is automatically dropped inside `FeatureConstructor.transform()`.
- For batch predictions, input CSV must have the same column names as training data (minus `Churn`). Use the files in `sample data/` as a reference template.

---

## 📄 License

This project is for educational purposes. Dataset credit: [safrin03 on Kaggle](https://www.kaggle.com/datasets/safrin03/predictive-analytics-for-customer-churn-dataset).
