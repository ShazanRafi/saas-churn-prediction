import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
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

st.set_page_config( page_title="ChurnScope", page_icon="📡", layout="wide", initial_sidebar_state="expanded" )

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
    
