import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import streamlit as st
import pandas as pd
import numpy as np
import joblib
import shap
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import io
import warnings
warnings.filterwarnings('ignore')
from feature_engineering import FeatureConstructor, OutlierHandler

st.set_page_config(
    page_title="ChurnScope",
    page_icon="📡",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Mono:wght@400;500&family=Syne:wght@400;600;700;800&display=swap');

html, body, [class*="css"] {
    font-family: 'Syne', sans-serif;
    background-color: #0c0f1a;
    color: #e8e6df;
}

.stApp { background-color: #0c0f1a; }

/* Sidebar */
section[data-testid="stSidebar"] {
    background: #111627;
    border-right: 1px solid #1e2540;
}

/* Header */
.brand-header {
    padding: 1.5rem 0 1rem 0;
    border-bottom: 1px solid #1e2540;
    margin-bottom: 1.5rem;
}
.brand-title {
    font-family: 'Syne', sans-serif;
    font-size: 2rem;
    font-weight: 800;
    letter-spacing: -0.03em;
    color: #f0ede6;
    margin: 0;
}
.brand-subtitle {
    font-family: 'DM Mono', monospace;
    font-size: 0.7rem;
    color: #5a6080;
    letter-spacing: 0.12em;
    text-transform: uppercase;
    margin: 0.2rem 0 0 0;
}

/* Metric cards */
.metric-card {
    background: #111627;
    border: 1px solid #1e2540;
    border-radius: 8px;
    padding: 1.2rem 1.4rem;
    margin-bottom: 0.8rem;
}
.metric-label {
    font-family: 'DM Mono', monospace;
    font-size: 0.65rem;
    letter-spacing: 0.12em;
    text-transform: uppercase;
    color: #5a6080;
    margin-bottom: 0.3rem;
}
.metric-value {
    font-family: 'Syne', sans-serif;
    font-size: 2rem;
    font-weight: 800;
    color: #f0ede6;
    line-height: 1;
}
.metric-sub {
    font-family: 'DM Mono', monospace;
    font-size: 0.7rem;
    color: #5a6080;
    margin-top: 0.2rem;
}

/* Risk badge */
.risk-high {
    background: #2d0f0f;
    border: 1px solid #8b1a1a;
    color: #ff6b6b;
    border-radius: 4px;
    padding: 0.3rem 0.8rem;
    font-family: 'DM Mono', monospace;
    font-size: 0.75rem;
    font-weight: 500;
    letter-spacing: 0.1em;
    display: inline-block;
}
.risk-medium {
    background: #2d1f0f;
    border: 1px solid #8b5a1a;
    color: #ffaa6b;
    border-radius: 4px;
    padding: 0.3rem 0.8rem;
    font-family: 'DM Mono', monospace;
    font-size: 0.75rem;
    font-weight: 500;
    letter-spacing: 0.1em;
    display: inline-block;
}
.risk-low {
    background: #0f2d1a;
    border: 1px solid #1a8b4a;
    color: #6bffaa;
    border-radius: 4px;
    padding: 0.3rem 0.8rem;
    font-family: 'DM Mono', monospace;
    font-size: 0.75rem;
    font-weight: 500;
    letter-spacing: 0.1em;
    display: inline-block;
}

/* Section headers */
.section-header {
    font-family: 'DM Mono', monospace;
    font-size: 0.65rem;
    letter-spacing: 0.15em;
    text-transform: uppercase;
    color: #5a6080;
    border-bottom: 1px solid #1e2540;
    padding-bottom: 0.5rem;
    margin: 1.5rem 0 1rem 0;
}

/* Prob bar */
.prob-bar-container {
    background: #1e2540;
    border-radius: 4px;
    height: 8px;
    margin: 0.5rem 0;
    overflow: hidden;
}
.prob-bar-fill {
    height: 100%;
    border-radius: 4px;
    transition: width 0.5s ease;
}

/* Insight card */
.insight-card {
    background: #0f1422;
    border: 1px solid #1e2540;
    border-left: 3px solid #4a7af5;
    border-radius: 0 8px 8px 0;
    padding: 0.8rem 1rem;
    margin: 0.5rem 0;
    font-family: 'DM Mono', monospace;
    font-size: 0.75rem;
    color: #9aa5cc;
    line-height: 1.5;
}

/* Streamlit overrides */
.stSelectbox label, .stSlider label, .stNumberInput label, .stTextInput label {
    font-family: 'DM Mono', monospace !important;
    font-size: 0.7rem !important;
    letter-spacing: 0.08em !important;
    text-transform: uppercase !important;
    color: #5a6080 !important;
}
.stButton > button {
    background: #4a7af5;
    color: #ffffff;
    border: none;
    border-radius: 6px;
    font-family: 'Syne', sans-serif;
    font-weight: 600;
    font-size: 0.9rem;
    padding: 0.6rem 2rem;
    width: 100%;
    transition: background 0.2s;
}
.stButton > button:hover {
    background: #3a6ae5;
    color: #ffffff;
}
div[data-testid="stDataFrame"] {
    border: 1px solid #1e2540;
    border-radius: 8px;
    overflow: hidden;
}
</style>
""", unsafe_allow_html=True)

@st.cache_resource
def load_model():
    try:
        bundle = joblib.load("churn_model.pkl")
        return bundle["model"], bundle["threshold"]
    except FileNotFoundError:
        return None, 0.35

model, THRESHOLD = load_model()

def get_risk(prob):
    if prob >= 0.65:
        return "HIGH RISK", "risk-high"
    elif prob >= 0.35:
        return "MEDIUM RISK", "risk-medium"
    else:
        return "LOW RISK", "risk-low"

_OHE_COLS = [
    "PaymentMethod", "PaperlessBilling", "ContentType", "MultiDeviceAccess",
    "DeviceRegistered", "GenrePreference", "Gender", "ParentalControl", "SubtitlesEnabled", "premium_underuse", "is_new_user"
]
_NUM_COLS = [
    "AccountAge", "MonthlyCharges", "TotalCharges", "ViewingHoursPerWeek",
    "AverageViewingDuration", "ContentDownloadsPerMonth", "WatchlistSize",
    "SupportTicketsPerMonth", "UserRating",
]

def _coerce_dtypes(df):
    
    df = df.copy()
    for col in _OHE_COLS:
        if col in df.columns:
            df[col] = df[col].astype(str)
    for col in _NUM_COLS:
        if col in df.columns:
            try:
                df[col] = pd.to_numeric(df[col], errors="raise")
            except (ValueError, TypeError):
                pass
    return df

def predict_single(input_df):
    prob = model.predict_proba(_coerce_dtypes(input_df))[:, 1][0]
    pred = int(prob >= THRESHOLD)
    return prob, pred

def predict_batch(df):
    probs = model.predict_proba(_coerce_dtypes(df))[:, 1]
    preds = (probs >= THRESHOLD).astype(int)
    return probs, preds

def get_shap_values(input_df):
    try:
        fc   = model.named_steps['feature_constructor']
        oh   = model.named_steps['outlier_handling']
        tr   = model.named_steps['transformer']
        clf  = model.named_steps['model']
        X_fc = fc.transform(input_df)
        X_oh = oh.transform(X_fc)
        X_tr = tr.transform(X_oh)
        feat_names = tr.get_feature_names_out()
        X_tr_df    = pd.DataFrame(X_tr, columns=feat_names)
        explainer  = shap.Explainer(clf, feature_names=feat_names)
        sv = explainer(X_tr_df)
        return sv, feat_names
    except Exception:
        return None, None

def bar_color(prob):
    if prob >= 0.65:
        return "#ff6b6b"
    elif prob >= 0.35:
        return "#ffaa6b"
    return "#6bffaa"

with st.sidebar:
    st.markdown("""
        <div class="brand-header">
            <p class="brand-title">ChurnScope</p>
            <p class="brand-subtitle">SaaS Retention Intelligence</p>
        </div>
    """, unsafe_allow_html=True)

    page = st.radio(
        "Navigate",
        ["🔍  Single Prediction", "📂  Batch CSV Upload", "📊  Model Insights"],
        label_visibility="collapsed"
    )

    st.markdown("<div class='section-header'>Model Status</div>", unsafe_allow_html=True)
    if model is not None:
        st.markdown("""
            <div class="insight-card">
            ✅ Model loaded<br>
            📐 Threshold: <strong>{:.2f}</strong>
            </div>
        """.format(THRESHOLD), unsafe_allow_html=True)
    else:
        st.markdown("""
            <div class="insight-card" style="border-left-color:#ff6b6b;">
            ⚠️ churn_model.pkl not found.<br>
            Place the model file in the same directory.
            </div>
        """, unsafe_allow_html=True)

if page == "🔍  Single Prediction":
    st.markdown("## Single Customer Prediction")
    st.markdown("<p style='color:#5a6080;font-family:DM Mono,monospace;font-size:0.75rem;'>Fill in customer attributes to get a real-time churn probability with SHAP explanation.</p>", unsafe_allow_html=True)

    with st.form("prediction_form"):
        col1, col2, col3 = st.columns(3)

        with col1:
            st.markdown("<div class='section-header'>Account Info</div>", unsafe_allow_html=True)
            customer_id    = st.text_input("Customer ID", value="CUST_001")
            subscription   = st.selectbox("Subscription Type", ["Basic", "Standard", "Premium"])
            account_age    = st.number_input("Account Age (months)", 1, 200, 24)
            monthly_charge = st.number_input("Monthly Charges ($)", 5.0, 200.0, 49.99, step=0.01)
            total_charges  = st.number_input("Total Charges ($)", 0.0, 20000.0, 1200.0, step=1.0)

        with col2:
            st.markdown("<div class='section-header'>Usage Patterns</div>", unsafe_allow_html=True)
            viewing_hours     = st.number_input("Viewing Hours / Week", 0.0, 80.0, 12.0, step=0.5)
            avg_duration      = st.number_input("Avg Viewing Duration (min)", 1.0, 300.0, 45.0)
            downloads         = st.number_input("Content Downloads / Month", 0, 200, 10)
            watchlist         = st.number_input("Watchlist Size", 0, 300, 20)
            support_tickets   = st.number_input("Support Tickets / Month", 0, 20, 1)
            user_rating       = st.slider("User Rating", 1, 5, 4)

        with col3:
            st.markdown("<div class='section-header'>Profile</div>", unsafe_allow_html=True)
            gender            = st.selectbox("Gender", ["Male", "Female"])
            payment_method    = st.selectbox("Payment Method", ["Credit Card", "Bank Transfer", "Electronic Check", "Mailed Check"])
            paperless         = st.selectbox("Paperless Billing", ["Yes", "No"])
            content_type      = st.selectbox("Content Type", ["Movies", "TV Shows", "Both"])
            multi_device      = st.selectbox("Multi Device Access", ["Yes", "No"])
            device_registered = st.number_input("Devices Registered", 1, 10, 2)
            genre_pref        = st.selectbox("Genre Preference", ["Drama", "Comedy", "Action", "Sci-Fi", "Horror", "Romance", "Documentary"])
            parental          = st.selectbox("Parental Control", ["Yes", "No"])
            subtitles         = st.selectbox("Subtitles Enabled", ["Yes", "No"])

        submitted = st.form_submit_button("▶  Run Prediction")

    if submitted and model is not None:
        input_data = pd.DataFrame([{
            "CustomerID":               customer_id,
            "SubscriptionType":         subscription,
            "AccountAge":               account_age,
            "MonthlyCharges":           monthly_charge,
            "TotalCharges":             total_charges,
            "ViewingHoursPerWeek":      viewing_hours,
            "AverageViewingDuration":   avg_duration,
            "ContentDownloadsPerMonth": downloads,
            "WatchlistSize":            watchlist,
            "SupportTicketsPerMonth":   support_tickets,
            "UserRating":               user_rating,
            "Gender":                   gender,
            "PaymentMethod":            payment_method,
            "PaperlessBilling":         paperless,
            "ContentType":              content_type,
            "MultiDeviceAccess":        multi_device,
            "DeviceRegistered":         device_registered,
            "GenrePreference":          genre_pref,
            "ParentalControl":          parental,
            "SubtitlesEnabled":         subtitles,
        }])

        prob, pred = predict_single(input_data)
        risk_label, risk_css = get_risk(prob)
        color = bar_color(prob)

        st.markdown("---")
        r1, r2, r3 = st.columns(3)

        with r1:
            st.markdown(f"""
                <div class="metric-card">
                    <div class="metric-label">Churn Probability</div>
                    <div class="metric-value">{prob:.1%}</div>
                    <div class="prob-bar-container">
                        <div class="prob-bar-fill" style="width:{prob*100:.1f}%;background:{color};"></div>
                    </div>
                </div>
            """, unsafe_allow_html=True)

        with r2:
            st.markdown(f"""
                <div class="metric-card">
                    <div class="metric-label">Risk Category</div>
                    <div style="margin-top:0.6rem;"><span class="{risk_css}">{risk_label}</span></div>
                    <div class="metric-sub" style="margin-top:0.6rem;">Threshold: {THRESHOLD:.2f}</div>
                </div>
            """, unsafe_allow_html=True)

        with r3:
            decision = "Will Churn" if pred == 1 else "Will Retain"
            d_color  = "#ff6b6b" if pred == 1 else "#6bffaa"
            st.markdown(f"""
                <div class="metric-card">
                    <div class="metric-label">Model Decision</div>
                    <div class="metric-value" style="color:{d_color};font-size:1.4rem;">{decision}</div>
                    <div class="metric-sub">Customer: {customer_id}</div>
                </div>
            """, unsafe_allow_html=True)

        st.markdown("<div class='section-header'>SHAP Feature Explanation</div>", unsafe_allow_html=True)
        with st.spinner("Computing SHAP values..."):
            sv, feat_names = get_shap_values(input_data)

        if sv is not None:
            fig, ax = plt.subplots(figsize=(10, 5))
            fig.patch.set_facecolor('#0c0f1a')
            ax.set_facecolor('#0c0f1a')
            shap.plots.waterfall(sv[0], show=False, max_display=15)
            plt.gcf().set_facecolor('#0c0f1a')
            for text in plt.gcf().findobj(plt.Text):
                text.set_color('#e8e6df')
            plt.tight_layout()
            st.pyplot(fig, use_container_width=True)
            plt.close()
        else:
            st.info("SHAP explanation not available for this model configuration.")

        st.markdown("<div class='section-header'>Retention Insights</div>", unsafe_allow_html=True)
        insights = []
        if support_tickets >= 3:
            insights.append("⚠️  High support ticket volume — proactive outreach recommended.")
        if viewing_hours < 5:
            insights.append("📉  Low viewing engagement — consider content recommendation nudges.")
        if subscription == "Premium" and viewing_hours < 10:
            insights.append("💸  Premium subscriber with low usage — at-risk of downgrade or churn.")
        if account_age <= 3:
            insights.append("🆕  New user — early-stage churn is common, consider onboarding campaign.")
        if user_rating <= 2:
            insights.append("😞  Low user rating — satisfaction issue needs follow-up.")
        if not insights:
            insights.append("✅  No strong churn signals detected. Customer appears stable.")
        for ins in insights:
            st.markdown(f"<div class='insight-card'>{ins}</div>", unsafe_allow_html=True)

    elif submitted and model is None:
        st.error("Model not loaded. Place `churn_model.pkl` in the working directory.")

elif page == "📂  Batch CSV Upload":
    st.markdown("## Batch Churn Prediction")
    st.markdown("<p style='color:#5a6080;font-family:DM Mono,monospace;font-size:0.75rem;'>Upload a CSV with customer records. Get churn probabilities + downloadable results.</p>", unsafe_allow_html=True)

    uploaded = st.file_uploader("Drop your CSV here", type=["csv"])

    if uploaded and model is not None:
        df = pd.read_csv(uploaded)
        st.markdown(f"<div class='insight-card'>📄 Loaded <strong>{len(df):,}</strong> records · <strong>{df.shape[1]}</strong> columns</div>", unsafe_allow_html=True)

        with st.expander("Preview raw data"):
            st.dataframe(df.head(10), use_container_width=True)

        with st.spinner("Running predictions..."):
            probs, preds = predict_batch(df)

        result_df = df.copy()
        result_df["Churn_Probability"] = probs.round(4)
        result_df["Churn_Prediction"]  = preds
        result_df["Risk_Category"]     = pd.cut(
            probs,
            bins=[-0.001, 0.35, 0.65, 1.001],
            labels=["Low Risk", "Medium Risk", "High Risk"]
        )

        total     = len(result_df)
        high_risk = (result_df["Risk_Category"] == "High Risk").sum()
        med_risk  = (result_df["Risk_Category"] == "Medium Risk").sum()
        low_risk  = (result_df["Risk_Category"] == "Low Risk").sum()
        avg_prob  = probs.mean()

        m1, m2, m3, m4 = st.columns(4)
        for col, label, value, sub in [
            (m1, "Total Customers",  f"{total:,}",          "in batch"),
            (m2, "High Risk",        f"{high_risk:,}",      f"{high_risk/total:.1%} of batch"),
            (m3, "Medium Risk",      f"{med_risk:,}",       f"{med_risk/total:.1%} of batch"),
            (m4, "Avg Churn Prob",   f"{avg_prob:.1%}",     "across all customers"),
        ]:
            col.markdown(f"""
                <div class="metric-card">
                    <div class="metric-label">{label}</div>
                    <div class="metric-value">{value}</div>
                    <div class="metric-sub">{sub}</div>
                </div>
            """, unsafe_allow_html=True)

        st.markdown("<div class='section-header'>Probability Distribution</div>", unsafe_allow_html=True)
        fig, ax = plt.subplots(figsize=(10, 3))
        fig.patch.set_facecolor('#0c0f1a')
        ax.set_facecolor('#111627')
        ax.hist(probs, bins=40, color='#4a7af5', edgecolor='#0c0f1a', linewidth=0.5, alpha=0.9)
        ax.axvline(THRESHOLD, color='#ff6b6b', linewidth=1.5, linestyle='--', label=f'Threshold ({THRESHOLD})')
        ax.set_xlabel("Churn Probability", color='#5a6080', fontsize=9)
        ax.set_ylabel("Count", color='#5a6080', fontsize=9)
        ax.tick_params(colors='#5a6080', labelsize=8)
        for spine in ax.spines.values():
            spine.set_edgecolor('#1e2540')
        ax.legend(facecolor='#111627', edgecolor='#1e2540', labelcolor='#e8e6df', fontsize=8)
        plt.tight_layout()
        st.pyplot(fig, use_container_width=True)
        plt.close()

        st.markdown("<div class='section-header'>Risk Category Breakdown</div>", unsafe_allow_html=True)
        fig2, ax2 = plt.subplots(figsize=(10, 1.5))
        fig2.patch.set_facecolor('#0c0f1a')
        ax2.set_facecolor('#0c0f1a')
        segments = [
            (low_risk / total,  "#6bffaa", "Low Risk"),
            (med_risk / total,  "#ffaa6b", "Medium Risk"),
            (high_risk / total, "#ff6b6b", "High Risk"),
        ]
        left = 0
        for frac, color, label in segments:
            ax2.barh(0, frac, left=left, color=color, height=0.6, label=f"{label} ({frac:.1%})")
            if frac > 0.05:
                ax2.text(left + frac / 2, 0, f"{frac:.1%}", ha='center', va='center',
                         color='#0c0f1a', fontsize=8, fontweight='bold')
            left += frac
        ax2.set_xlim(0, 1)
        ax2.axis('off')
        ax2.legend(loc='lower center', bbox_to_anchor=(0.5, -1.5), ncol=3,
                   facecolor='#111627', edgecolor='#1e2540', labelcolor='#e8e6df', fontsize=8)
        plt.tight_layout()
        st.pyplot(fig2, use_container_width=True)
        plt.close()

        st.markdown("<div class='section-header'>Scored Records (sorted by risk)</div>", unsafe_allow_html=True)
        display_cols = ["CustomerID", "Churn_Probability", "Churn_Prediction", "Risk_Category"] if "CustomerID" in result_df.columns else ["Churn_Probability", "Churn_Prediction", "Risk_Category"]
        st.dataframe(
            result_df[display_cols].sort_values("Churn_Probability", ascending=False).reset_index(drop=True),
            use_container_width=True,
            height=350
        )

        csv_out = result_df.to_csv(index=False).encode("utf-8")
        st.download_button(
            label="⬇  Download Scored CSV",
            data=csv_out,
            file_name="churn_predictions.csv",
            mime="text/csv"
        )

    elif uploaded and model is None:
        st.error("Model not loaded. Place `churn_model.pkl` in the working directory.")
    else:
        st.markdown("""
            <div class="insight-card" style="text-align:center;padding:2rem;">
            Upload a CSV file with the same columns used during training.<br>
            The model will score each row and return churn probabilities.
            </div>
        """, unsafe_allow_html=True)

elif page == "📊  Model Insights":
    st.markdown("## Model Insights")
    st.markdown("<p style='color:#5a6080;font-family:DM Mono,monospace;font-size:0.75rem;'>Global SHAP feature importance — what drives churn across all customers.</p>", unsafe_allow_html=True)

    st.markdown("""
        <div class="insight-card">
        To generate SHAP plots, upload a sample of your training data below (100–500 rows is enough).
        </div>
    """, unsafe_allow_html=True)

    sample_file = st.file_uploader("Upload a sample dataset (CSV)", type=["csv"], key="insights")

    if sample_file and model is not None:
        df_sample = pd.read_csv(sample_file)
        if "Churn" in df_sample.columns:
            df_sample = df_sample.drop(columns=["Churn"])

        n = min(300, len(df_sample))
        df_sample = df_sample.sample(n, random_state=42)

        with st.spinner(f"Computing SHAP on {n} rows…"):
            try:
                fc          = model.named_steps['feature_constructor']
                oh          = model.named_steps['outlier_handling']
                tr          = model.named_steps['transformer']
                clf         = model.named_steps['model']
                X_fc        = fc.transform(df_sample)
                X_oh        = oh.transform(X_fc)
                X_tr        = tr.transform(X_oh)
                feat_names  = tr.get_feature_names_out()
                X_tr_df     = pd.DataFrame(X_tr, columns=feat_names)
                background = X_tr_df.sample(min(100, len(X_tr_df)), random_state=42)
                explainer = shap.LinearExplainer(clf, background)
                sv = explainer(X_tr_df)
                


                tab1, tab2, tab3 = st.tabs(["Bar (Mean |SHAP|)", "Beeswarm", "Heatmap"])

                with tab1:
                    st.markdown("<div class='section-header'>Top Features by Mean Absolute SHAP</div>", unsafe_allow_html=True)
                    fig, ax = plt.subplots(figsize=(10, 6))
                    fig.patch.set_facecolor('#0c0f1a')
                    ax.set_facecolor('#111627')
                    shap.plots.bar(sv, show=False, max_display=15, ax=ax)
                    ax.tick_params(colors='#9aa5cc', labelsize=8)
                    for spine in ax.spines.values():
                        spine.set_edgecolor('#1e2540')
                    ax.set_xlabel(ax.get_xlabel(), color='#5a6080', fontsize=9)
                    for text in fig.findobj(plt.Text):
                        text.set_color('#e8e6df')
                    plt.tight_layout()
                    st.pyplot(fig, use_container_width=True)
                    plt.close()

                with tab2:
                    st.markdown("<div class='section-header'>Beeswarm — Feature Impact Distribution</div>", unsafe_allow_html=True)
                    fig, ax = plt.subplots(figsize=(10, 6))
                    fig.patch.set_facecolor('#0c0f1a')
                    ax.set_facecolor('#111627')
                    shap.plots.beeswarm(sv, show=False, max_display=15)
                    plt.gcf().set_facecolor('#0c0f1a')
                    for text in plt.gcf().findobj(plt.Text):
                        text.set_color('#e8e6df')
                    plt.tight_layout()
                    st.pyplot(plt.gcf(), use_container_width=True)
                    plt.close()

                with tab3:
                    st.markdown("<div class='section-header'>SHAP Heatmap</div>", unsafe_allow_html=True)
                    fig, ax = plt.subplots(figsize=(12, 10))
                    fig.patch.set_facecolor('#0c0f1a')
                    shap.plots.heatmap(sv, show=False, max_display=10)
                    plt.gcf().set_facecolor('#0c0f1a')
                    for text in plt.gcf().findobj(plt.Text):
                        text.set_color('#e8e6df')
                    plt.tight_layout()
                    st.pyplot(plt.gcf(), use_container_width=True)
                    plt.close()

            except Exception as e:
                st.error(f"SHAP computation failed: {e}")

    elif sample_file and model is None:
        st.error("Model not loaded.")

    st.markdown("<div class='section-header'>Feature Engineering Summary</div>", unsafe_allow_html=True)
    features_info = [
        ("engagement_score",        "ViewingHours + AvgDuration + Downloads — overall usage intensity"),
        ("cost_per_hour",           "MonthlyCharges / (ViewingHoursPerWeek + 1) — value perception"),
        ("watch_intensity",         "ViewingHours / (AccountAge + 1) — engagement relative to tenure"),
        ("support_to_usage",        "SupportTickets / (ViewingHours + 1) — frustration-to-use ratio"),
        ("log_total_charges",       "Log-transformed TotalCharges — stabilises right-skewed distribution"),
        ("frustration_index",       "SupportTickets × (6 − UserRating) — combined dissatisfaction signal"),
        ("premium_underuse",        "Premium subscriber with low viewing hours — churn risk flag"),
        ("loyalty_tier",            "AccountAge bucketed: very_new → veteran — ordinal retention signal"),
        ("is_new_user",             "AccountAge ≤ 3 months — early churn detection flag"),
    ]
    for name, desc in features_info:
        st.markdown(f"""
            <div class="insight-card">
            <span style="color:#4a7af5;font-weight:500;">{name}</span><br>
            {desc}
            </div>
        """, unsafe_allow_html=True)