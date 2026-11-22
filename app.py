"""
House Price Prediction - Streamlit Dashboard
----------------------------------------------
CIE 2 Activity 1: Multiple Linear Regression Worksheet
Professional 11-page Streamlit web application.
"""

import sys
import os

# Ensure project root is on path
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from src.data_loader import (
    load_dataset, get_dataset_info, TARGET_COL, FEATURE_COLS, FEATURE_INFO,
)
from src.model_training import (
    train_all_models, split_data, load_model, compute_metrics,
    MODEL_PATH, METADATA_PATH,
)
from src.manual_calculation import manual_regression
from src.multicollinearity import compute_vif, interpret_vif, get_high_correlations
from src.residual_analysis import compute_residual_stats

# ── Page Config ──────────────────────────────────────────────
st.set_page_config(
    page_title="House Price Prediction | MLR Dashboard",
    page_icon="🏠",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ───────────────────────────────────────────────
st.markdown("""
<style>
    .main .block-container {padding-top: 1.5rem; padding-bottom: 1.5rem;}
    [data-testid="stSidebar"] {background: linear-gradient(180deg, #0f172a 0%, #1e293b 100%);}
    [data-testid="stSidebar"] * {color: #e2e8f0 !important;}
    [data-testid="stSidebar"] .stRadio label:hover {color: #93c5fd !important;}
    .metric-card {
        background: linear-gradient(135deg, #1e3a5f 0%, #0f172a 100%);
        padding: 1.2rem; border-radius: 12px; color: white; text-align: center;
        box-shadow: 0 4px 15px rgba(0,0,0,0.15);
    }
    .metric-card h3 {font-size: 0.85rem; color: #93c5fd; margin-bottom: 4px;}
    .metric-card .value {font-size: 1.6rem; font-weight: 800; color: #ffffff;}
    .metric-card .info {font-size: 0.72rem; color: #94a3b8; margin-top: 4px;}
    .section-header {
        background: linear-gradient(90deg, #1e3a5f, #0f172a);
        color: white; padding: 0.8rem 1.2rem; border-radius: 8px;
        margin-bottom: 1rem; font-size: 1.1rem; font-weight: 700;
    }
    .info-box {
        background: #eff6ff; border-left: 4px solid #3b82f6;
        padding: 0.8rem 1rem; border-radius: 0 8px 8px 0; margin: 0.5rem 0;
        font-size: 0.88rem; color: #1e3a5f;
    }
    .warn-box {
        background: #fef3c7; border-left: 4px solid #f59e0b;
        padding: 0.8rem 1rem; border-radius: 0 8px 8px 0; margin: 0.5rem 0;
        font-size: 0.88rem; color: #92400e;
    }
</style>
""", unsafe_allow_html=True)


def metric_card(title, value, info=""):
    return f"""<div class="metric-card">
        <h3>{title}</h3>
        <div class="value">{value}</div>
        <div class="info">{info}</div>
    </div>"""


def section_header(text):
    st.markdown(f'<div class="section-header">📌 {text}</div>', unsafe_allow_html=True)


# ── Data Loading (Cached) ───────────────────────────────────
@st.cache_data
def cached_load():
    df = load_dataset()
    info = get_dataset_info(df)
    return df, info


@st.cache_data
def cached_train():
    return train_all_models()


# ── Sidebar Navigation ──────────────────────────────────────
st.sidebar.markdown("## 🏠 House Price Prediction")
st.sidebar.markdown("**Multiple Linear Regression**")
st.sidebar.markdown("---")

PAGES = [
    "1. Overview",
    "2. Dataset Explorer",
    "3. Data Visualization",
    "4. Manual Calculation",
    "5. Single vs Multiple Regression",
    "6. Model Training",
    "7. Model Evaluation",
    "8. Residual Analysis",
    "9. Multicollinearity & Assumptions",
    "10. House Price Prediction",
    "11. About Project",
]
page = st.sidebar.radio("Navigate", PAGES, label_visibility="collapsed")

st.sidebar.markdown("---")
st.sidebar.markdown("*CIE 2 — Activity 1*")


# ══════════════════════════════════════════════════════════════
# PAGE 1: OVERVIEW
# ══════════════════════════════════════════════════════════════
if page == PAGES[0]:
    st.title("🏠 House Price Prediction Using Multiple Linear Regression")
    st.markdown("**CIE 2 — Activity 1: Multiple Linear Regression Worksheet**")

    section_header("Problem Statement")
    st.markdown("""
    **Objective:** Predict house prices using multiple input features with Multiple Linear Regression.

    **Modelling Goal:** Build a supervised regression model that learns the relationship
    between house characteristics (features) and their market price (target), so it can
    estimate the price of a new, unseen house.

    **Mathematical Model:**
    """)
    st.latex(r"\hat{y} = \beta_0 + \beta_1 x_1 + \beta_2 x_2 + \cdots + \beta_p x_p")
    st.markdown("Where *y&#770;* is the predicted house price, *β₀* is the intercept, and *β₁…βₚ* are the coefficients for each feature.")

    section_header("Feature Descriptions")
    feat_rows = []
    for feat in FEATURE_COLS:
        fi = FEATURE_INFO[feat]
        feat_rows.append({
            "Feature": feat,
            "Description": fi["description"],
            "Unit": fi["unit"],
            "Why It Matters": fi["why"],
        })
    st.dataframe(pd.DataFrame(feat_rows), use_container_width=True, hide_index=True)

    st.markdown("""
    - **Target Variable (y):** `House_Price` — the market price of the house in US dollars.
    - **One Row =** One individual house listing with its characteristics and sale price.
    - **Dataset Source:** Synthetic dataset for academic ML demonstration.
    """)

    # Quick metrics if model is trained
    pipeline, metadata = load_model()
    if metadata and "evaluation_metrics" in metadata:
        section_header("Current Model Performance")
        m = metadata["evaluation_metrics"]
        c1, c2, c3, c4 = st.columns(4)
        c1.markdown(metric_card("R² Score", f"{m['R2']:.4f}", "Variance explained"), unsafe_allow_html=True)
        c2.markdown(metric_card("MAE", f"${m['MAE']:,.2f}", "Avg absolute error"), unsafe_allow_html=True)
        c3.markdown(metric_card("RMSE", f"${m['RMSE']:,.2f}", "Typical deviation"), unsafe_allow_html=True)
        adj = m.get("Adjusted_R2")
        c4.markdown(metric_card("Adj R²", f"{adj:.4f}" if adj else "N/A", "Penalizes extra features"), unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════
# PAGE 2: DATASET EXPLORER
# ══════════════════════════════════════════════════════════════
elif page == PAGES[1]:
    st.title("📊 Dataset Explorer")
    df, info = cached_load()

    section_header("Dataset Overview")
    c1, c2, c3, c4 = st.columns(4)
    c1.markdown(metric_card("Rows", str(info["rows"]), "Total observations"), unsafe_allow_html=True)
    c2.markdown(metric_card("Columns", str(info["cols"]), "Total features + target"), unsafe_allow_html=True)
    c3.markdown(metric_card("Missing Values", str(info["total_missing"]), "Across all columns"), unsafe_allow_html=True)
    c4.markdown(metric_card("Duplicates", str(info["duplicates"]), "Duplicate rows"), unsafe_allow_html=True)

    st.markdown(f"""
    <div class="info-box">
    <strong>Dataset:</strong> {info['name']} &nbsp;|&nbsp;
    <strong>Source:</strong> {info['source']} &nbsp;|&nbsp;
    <strong>Numerical Features:</strong> {len(info['numerical'])} &nbsp;|&nbsp;
    <strong>Categorical Features:</strong> {len(info['categorical'])}
    </div>""", unsafe_allow_html=True)

    section_header("First 5 Records")
    st.dataframe(df.head(), use_container_width=True, hide_index=True)

    section_header("Data Types & Missing Values")
    type_df = pd.DataFrame({
        "Column": df.columns,
        "Data Type": df.dtypes.astype(str).values,
        "Missing": df.isnull().sum().values,
        "Missing %": (df.isnull().sum() / len(df) * 100).round(2).values,
        "Unique": df.nunique().values,
    })
    st.dataframe(type_df, use_container_width=True, hide_index=True)

    section_header("Summary Statistics")
    st.dataframe(df.describe().round(2), use_container_width=True)

    section_header("Feature Units & Ranges")
    range_rows = []
    for feat in FEATURE_COLS:
        fi = FEATURE_INFO[feat]
        range_rows.append({
            "Feature": feat,
            "Unit": fi["unit"],
            "Min": f"{df[feat].min():.2f}",
            "Max": f"{df[feat].max():.2f}",
            "Mean": f"{df[feat].mean():.2f}",
        })
    st.dataframe(pd.DataFrame(range_rows), use_container_width=True, hide_index=True)

    section_header("Data Cleaning Notes")
    st.markdown("""
    - **Missing values:** No missing values detected — no imputation needed.
    - **Duplicate records:** No duplicates found.
    - **Invalid values:** All columns contain valid numerical values.
    - **Outliers:** Not automatically removed. Outlier visualizations are available in the Data Visualization page.
    - **Categorical variables:** None present — all features are numerical.
    - **Original dataset preserved:** The original CSV file is never modified.
    """)


# ══════════════════════════════════════════════════════════════
# PAGE 3: DATA VISUALIZATION
# ══════════════════════════════════════════════════════════════
elif page == PAGES[2]:
    st.title("📈 Data Exploration & Visualization")
    df, _ = cached_load()

    section_header("1. House Price Distribution")
    fig = px.histogram(df, x=TARGET_COL, nbins=30, color_discrete_sequence=["#1e3a5f"],
                       title="Distribution of House Prices", labels={TARGET_COL: "House Price ($)"})
    fig.add_vline(x=df[TARGET_COL].mean(), line_dash="dash", line_color="red",
                  annotation_text=f"Mean: ${df[TARGET_COL].mean():,.0f}")
    fig.add_vline(x=df[TARGET_COL].median(), line_dash="dot", line_color="green",
                  annotation_text=f"Median: ${df[TARGET_COL].median():,.0f}")
    fig.update_layout(xaxis_title="House Price ($)", yaxis_title="Frequency")
    st.plotly_chart(fig, use_container_width=True)
    st.markdown('<div class="info-box"><strong>Interpretation:</strong> The house price distribution appears approximately bell-shaped. The mean and median are close, suggesting a roughly symmetric distribution with no extreme skew.</div>', unsafe_allow_html=True)

    section_header("2. Feature vs House Price Scatter Plots")
    selected_feat = st.selectbox("Select feature:", FEATURE_COLS)
    fig2 = px.scatter(df, x=selected_feat, y=TARGET_COL, trendline="ols",
                      color_discrete_sequence=["#1e3a5f"],
                      title=f"{selected_feat} vs House Price",
                      labels={selected_feat: f"{selected_feat} ({FEATURE_INFO[selected_feat]['unit']})",
                              TARGET_COL: "House Price ($)"})
    st.plotly_chart(fig2, use_container_width=True)
    corr_val = df[selected_feat].corr(df[TARGET_COL])
    direction = "positive" if corr_val > 0 else "negative"
    strength = "strong" if abs(corr_val) > 0.7 else "moderate" if abs(corr_val) > 0.4 else "weak"
    st.markdown(f'<div class="info-box"><strong>Interpretation:</strong> {selected_feat} shows a <strong>{strength} {direction}</strong> correlation (r = {corr_val:.4f}) with House Price. {"This feature is a strong predictor." if abs(corr_val) > 0.7 else "Other features may add predictive value."}</div>', unsafe_allow_html=True)

    section_header("3. Feature Correlation Heatmap")
    corr_matrix = df.corr()
    fig3 = px.imshow(corr_matrix.round(3), text_auto=True, color_continuous_scale="RdBu_r",
                     aspect="auto", title="Feature Correlation Heatmap",
                     zmin=-1, zmax=1)
    fig3.update_layout(width=800, height=600)
    st.plotly_chart(fig3, use_container_width=True)
    st.markdown("""<div class="info-box"><strong>Interpretation:</strong> The heatmap shows pairwise Pearson correlations.
    Strong positive correlations (close to +1) appear in red. Features highly correlated with House_Price
    are strong predictors. Features highly correlated with each other may indicate multicollinearity.<br>
    <strong>⚠️ Note:</strong> Correlation does not prove causation.</div>""", unsafe_allow_html=True)

    section_header("4. Feature Distributions (Box Plots)")
    feat_choice = st.selectbox("Feature to visualize:", FEATURE_COLS, key="box_feat")
    fig4 = px.box(df, y=feat_choice, title=f"Distribution of {feat_choice}",
                  labels={feat_choice: f"{feat_choice} ({FEATURE_INFO[feat_choice]['unit']})"})
    fig4.update_traces(marker_color="#1e3a5f")
    st.plotly_chart(fig4, use_container_width=True)
    st.markdown(f'<div class="info-box"><strong>Interpretation:</strong> The box plot shows the spread of {feat_choice}. Points beyond the whiskers are potential outliers. Outliers are NOT automatically removed — they may represent valid extreme observations.</div>', unsafe_allow_html=True)

    section_header("5. Pairwise Relationships")
    pair_feats = st.multiselect("Select features for pair plot:", FEATURE_COLS, default=FEATURE_COLS[:3])
    if pair_feats:
        fig5 = px.scatter_matrix(df[pair_feats + [TARGET_COL]], dimensions=pair_feats + [TARGET_COL],
                                 color_discrete_sequence=["#1e3a5f"], title="Pairwise Scatter Matrix",
                                 opacity=0.5)
        fig5.update_layout(height=600)
        st.plotly_chart(fig5, use_container_width=True)
        st.markdown('<div class="info-box"><strong>Note:</strong> Correlation does not imply causation. These plots show associations only.</div>', unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════
# PAGE 4: MANUAL CALCULATION
# ══════════════════════════════════════════════════════════════
elif page == PAGES[3]:
    st.title("🧮 Manual Calculation of Multiple Linear Regression")
    st.markdown("**Educational section: Normal Equation step-by-step using a small sample**")

    df, _ = cached_load()
    c1, c2, c3 = st.columns(3)
    feat1 = c1.selectbox("Feature X₁:", FEATURE_COLS, index=0)
    remaining = [f for f in FEATURE_COLS if f != feat1]
    feat2 = c2.selectbox("Feature X₂:", remaining, index=0)
    n_obs = c3.slider("Number of observations:", 5, 8, 6)

    result = manual_regression(df, feat1, feat2, n_obs)

    section_header("Step 1: Selected Observations")
    st.dataframe(result["sample_df"][[feat1, feat2, TARGET_COL]], use_container_width=True, hide_index=True)

    section_header("Step 2: Design Matrix X (with ones column)")
    design_df = pd.DataFrame(result["X_design"], columns=["1 (Intercept)", f"X₁ ({feat1})", f"X₂ ({feat2})"])
    st.dataframe(design_df.round(4), use_container_width=True, hide_index=True)

    section_header("Step 3: Target Vector y")
    st.dataframe(pd.DataFrame({"y (House_Price)": result["y"]}), use_container_width=True, hide_index=True)

    section_header("Step 4: Transpose of X (Xᵀ)")
    xt_df = pd.DataFrame(result["Xt"].round(4))
    st.dataframe(xt_df, use_container_width=True, hide_index=True)

    section_header("Step 5: Xᵀ × X")
    xtx_df = pd.DataFrame(result["XtX"].round(4),
                          columns=["Intercept", feat1, feat2],
                          index=["Intercept", feat1, feat2])
    st.dataframe(xtx_df, use_container_width=True)

    section_header("Step 6: Xᵀ × y")
    xty_df = pd.DataFrame({"Xᵀy": result["Xty"].round(4)}, index=["Intercept", feat1, feat2])
    st.dataframe(xty_df, use_container_width=True)

    section_header("Step 7: Normal Equation")
    st.latex(r"(X^T X) \mathbf{b} = X^T \mathbf{y}")
    st.markdown(f"**Solution method:** {result['solve_method']}")
    st.markdown(f"**Condition number of XᵀX:** {result['condition_number']:.2f}")

    section_header("Step 8: Calculated Coefficients")
    st.markdown(f"""
    | Coefficient | Symbol | Value |
    |---|---|---|
    | Intercept | b₀ | {result['b0']:,.4f} |
    | {feat1} | b₁ | {result['b1']:,.6f} |
    | {feat2} | b₂ | {result['b2']:,.6f} |
    """)

    section_header("Step 9: Fitted Equation")
    st.latex(rf"\hat{{y}} = {result['b0']:,.2f} + ({result['b1']:,.4f}) \cdot X_1 + ({result['b2']:,.4f}) \cdot X_2")

    section_header("Step 10: Prediction for Observation 1")
    st.markdown(f"""
    - **{feat1} =** {result['X_raw'][0, 0]:.4f}, **{feat2} =** {result['X_raw'][0, 1]:.4f}
    - **Predicted Price =** ${result['prediction_obs1']:,.2f}
    - **Actual Price =** ${result['actual_obs1']:,.2f}
    """)

    section_header("Step 11: Residual for Observation 1")
    st.markdown(f"**Residual = Actual − Predicted = ${result['residual_obs1']:,.2f}**")

    section_header("Verification: Sklearn Comparison")
    comp_df = pd.DataFrame({
        "Coefficient": ["b₀ (Intercept)", f"b₁ ({feat1})", f"b₂ ({feat2})"],
        "Manual (Normal Eq)": [f"{result['b0']:.6f}", f"{result['b1']:.6f}", f"{result['b2']:.6f}"],
        "Sklearn": [f"{result['sklearn_b0']:.6f}", f"{result['sklearn_b1']:.6f}", f"{result['sklearn_b2']:.6f}"],
        "Absolute Difference": [f"{result['diff_b0']:.2e}", f"{result['diff_b1']:.2e}", f"{result['diff_b2']:.2e}"],
    })
    st.dataframe(comp_df, use_container_width=True, hide_index=True)
    st.markdown("""<div class="info-box"><strong>Why are they approximately equal?</strong>
    Both methods solve the same Ordinary Least Squares problem. The Normal Equation directly
    computes the closed-form solution, while Sklearn internally uses an optimised solver.
    Tiny differences arise from floating-point arithmetic precision.</div>""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════
# PAGE 5: SINGLE vs MULTIPLE REGRESSION
# ══════════════════════════════════════════════════════════════
elif page == PAGES[4]:
    st.title("📐 Single vs Multiple Regression Comparison")

    results = cached_train()
    baseline = results["baseline"]
    single = results["single"]
    multiple = results["multiple"]

    section_header("Baseline Model (Mean Prediction)")
    st.markdown(f"""
    **Method:** Predict the training set mean (${baseline['mean_price']:,.2f}) for every test observation.

    **Why a baseline?** A baseline provides a reference point. If a model cannot beat the baseline,
    it adds no value. The baseline R² is ≤ 0 by definition (a constant prediction explains none
    of the variance relative to the test set mean).
    """)
    bc1, bc2, bc3, bc4 = st.columns(4)
    bm = baseline["metrics"]
    bc1.markdown(metric_card("R²", f"{bm['R2']:.4f}", "Should be ≈ 0"), unsafe_allow_html=True)
    bc2.markdown(metric_card("MAE", f"${bm['MAE']:,.2f}", ""), unsafe_allow_html=True)
    bc3.markdown(metric_card("RMSE", f"${bm['RMSE']:,.2f}", ""), unsafe_allow_html=True)
    bc4.markdown(metric_card("MSE", f"{bm['MSE']:,.0f}", ""), unsafe_allow_html=True)

    section_header(f"Single-Feature Regression: {single['feature']}")
    st.markdown(f"""
    **Selected Feature:** `{single['feature']}` (strongest correlation with training target)

    **Regression Equation:**
    """)
    st.latex(rf"\hat{{y}} = {single['intercept']:,.2f} + {single['coefficient']:,.4f} \times \text{{{single['feature']}}}")
    st.markdown(f"""
    - **Intercept ({single['intercept']:,.2f}):** Theoretical price when {single['feature']} = 0.
    - **Coefficient ({single['coefficient']:,.4f}):** For every 1-unit increase in {single['feature']},
      the predicted price changes by ${single['coefficient']:,.2f}.
    """)
    sc1, sc2, sc3, sc4 = st.columns(4)
    sm = single["metrics"]
    sc1.markdown(metric_card("R²", f"{sm['R2']:.4f}", ""), unsafe_allow_html=True)
    sc2.markdown(metric_card("MAE", f"${sm['MAE']:,.2f}", ""), unsafe_allow_html=True)
    sc3.markdown(metric_card("RMSE", f"${sm['RMSE']:,.2f}", ""), unsafe_allow_html=True)
    sc4.markdown(metric_card("Adj R²", f"{sm['Adjusted_R2']:.4f}" if sm.get("Adjusted_R2") else "N/A", ""), unsafe_allow_html=True)

    section_header("Multiple Linear Regression (All Features)")
    mm = multiple["metrics"]
    mc1, mc2, mc3, mc4 = st.columns(4)
    mc1.markdown(metric_card("R²", f"{mm['R2']:.4f}", ""), unsafe_allow_html=True)
    mc2.markdown(metric_card("MAE", f"${mm['MAE']:,.2f}", ""), unsafe_allow_html=True)
    mc3.markdown(metric_card("RMSE", f"${mm['RMSE']:,.2f}", ""), unsafe_allow_html=True)
    mc4.markdown(metric_card("Adj R²", f"{mm['Adjusted_R2']:.4f}" if mm.get("Adjusted_R2") else "N/A", ""), unsafe_allow_html=True)

    section_header("Model Comparison Table")
    comp_df = pd.DataFrame({
        "Model": ["Mean Baseline", f"Single ({single['feature']})", "Multiple LR"],
        "MAE ($)": [f"{bm['MAE']:,.2f}", f"{sm['MAE']:,.2f}", f"{mm['MAE']:,.2f}"],
        "MSE": [f"{bm['MSE']:,.0f}", f"{sm['MSE']:,.0f}", f"{mm['MSE']:,.0f}"],
        "RMSE ($)": [f"{bm['RMSE']:,.2f}", f"{sm['RMSE']:,.2f}", f"{mm['RMSE']:,.2f}"],
        "R²": [f"{bm['R2']:.4f}", f"{sm['R2']:.4f}", f"{mm['R2']:.4f}"],
    })
    st.dataframe(comp_df, use_container_width=True, hide_index=True)

    section_header("Model Comparison Bar Chart")
    bar_df = pd.DataFrame({
        "Model": ["Baseline", f"Single ({single['feature']})", "Multiple LR"] * 2,
        "Metric": ["R²"] * 3 + ["RMSE ($)"] * 3,
        "Value": [bm["R2"], sm["R2"], mm["R2"], bm["RMSE"], sm["RMSE"], mm["RMSE"]],
    })
    fig = px.bar(bar_df, x="Model", y="Value", color="Model", facet_col="Metric",
                 title="Model Comparison", color_discrete_sequence=["#94a3b8", "#3b82f6", "#0f172a"])
    fig.update_yaxes(matches=None)
    fig.for_each_annotation(lambda a: a.update(text=a.text.split("=")[-1]))
    st.plotly_chart(fig, use_container_width=True)

    st.markdown(f"""<div class="info-box"><strong>Analysis:</strong>
    <ul>
    <li>Multiple LR (R² = {mm['R2']:.4f}) substantially outperforms the baseline (R² = {bm['R2']:.4f}).</li>
    <li>Adding more features beyond {single['feature']} {"significantly improves" if mm['R2'] - sm['R2'] > 0.01 else "marginally improves"} prediction accuracy.</li>
    <li>The added complexity of multiple features is {"justified" if mm['R2'] > sm['R2'] + 0.005 else "marginal"} given the improvement in R² and reduction in RMSE.</li>
    </ul></div>""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════
# PAGE 6: MODEL TRAINING
# ══════════════════════════════════════════════════════════════
elif page == PAGES[5]:
    st.title("⚙️ Model Training")

    section_header("Training Configuration")
    st.markdown(f"""
    | Setting | Value |
    |---|---|
    | Algorithm | Multiple Linear Regression (OLS) |
    | Train/Test Split | 80% / 20% |
    | Random State | 42 |
    | Features | {len(FEATURE_COLS)} numerical |
    | Target | {TARGET_COL} |
    """)

    if st.button("🚀 Train All Models", type="primary"):
        with st.spinner("Training baseline, single-feature, and multiple regression models..."):
            results = train_all_models()
            st.cache_data.clear()
        st.success("✅ All models trained and saved successfully!")
        multiple = results["multiple"]
        mm = multiple["metrics"]
        st.markdown(f"""
        **Multiple LR Results:**
        - Training samples: {len(results['X_train'])}
        - Testing samples: {len(results['X_test'])}
        - R² = {mm['R2']:.4f} | MAE = ${mm['MAE']:,.2f} | RMSE = ${mm['RMSE']:,.2f}
        """)

    # Show current model info
    pipeline, metadata = load_model()
    if metadata:
        section_header("Saved Model Metadata")
        st.json(metadata)
    else:
        st.info("No saved model found. Click 'Train All Models' to train.")

    if pipeline:
        section_header("Model Coefficients")
        reg = pipeline.named_steps["regressor"]
        coef_df = pd.DataFrame({
            "Feature": FEATURE_COLS,
            "Coefficient": reg.coef_,
            "Impact": [f"${abs(c):,.2f} per unit" for c in reg.coef_],
        })
        st.dataframe(coef_df, use_container_width=True, hide_index=True)
        st.markdown(f"**Intercept:** ${reg.intercept_:,.2f}")

        section_header("Coefficient Interpretation")
        for feat, coef in zip(FEATURE_COLS, reg.coef_):
            direction = "increases" if coef > 0 else "decreases"
            st.markdown(f"- **{feat}:** When {feat} increases by 1 {FEATURE_INFO[feat]['unit']}, "
                       f"the predicted price {direction} by ${abs(coef):,.2f}, holding other features constant.")
        st.markdown(f"""
        **Intercept (${reg.intercept_:,.2f}):** This is the theoretical price when ALL features are zero.
        Since features like Year_Built cannot realistically be zero, the intercept is a mathematical
        artifact of the linear model and should not be interpreted literally.

        ⚠️ **Caution:** Coefficients show statistical associations, not proof of causation.
        """)

        section_header("Sample Predictions (Test Set)")
        df, _ = cached_load()
        X_train, X_test, y_train, y_test = split_data(df)
        y_pred = pipeline.predict(X_test)
        sample_df = X_test.head(10).copy()
        sample_df["Actual Price"] = y_test.head(10).values
        sample_df["Predicted Price"] = y_pred[:10]
        sample_df["Error"] = sample_df["Actual Price"] - sample_df["Predicted Price"]
        st.dataframe(sample_df.round(2), use_container_width=True, hide_index=True)


# ══════════════════════════════════════════════════════════════
# PAGE 7: MODEL EVALUATION
# ══════════════════════════════════════════════════════════════
elif page == PAGES[6]:
    st.title("📏 Model Evaluation Metrics")

    results = cached_train()
    mm = results["multiple"]["metrics"]

    section_header("Evaluation Metrics with Formulas")

    # MAE
    col1, col2 = st.columns([1, 2])
    with col1:
        st.markdown(metric_card("MAE", f"${mm['MAE']:,.2f}", "Mean Absolute Error"), unsafe_allow_html=True)
    with col2:
        st.latex(r"MAE = \frac{1}{n} \sum_{i=1}^{n} |y_i - \hat{y}_i|")
        st.markdown(f"**Meaning:** Average magnitude of prediction errors, without considering direction. On average, the model's prediction is off by **${mm['MAE']:,.2f}**.")

    st.markdown("---")
    # MSE
    col1, col2 = st.columns([1, 2])
    with col1:
        st.markdown(metric_card("MSE", f"{mm['MSE']:,.2f}", "Mean Squared Error"), unsafe_allow_html=True)
    with col2:
        st.latex(r"MSE = \frac{1}{n} \sum_{i=1}^{n} (y_i - \hat{y}_i)^2")
        st.markdown(f"**Meaning:** Average of squared errors. Penalises large errors more heavily due to squaring. Value: **{mm['MSE']:,.2f}**.")

    st.markdown("---")
    # RMSE
    col1, col2 = st.columns([1, 2])
    with col1:
        st.markdown(metric_card("RMSE", f"${mm['RMSE']:,.2f}", "Root Mean Squared Error"), unsafe_allow_html=True)
    with col2:
        st.latex(r"RMSE = \sqrt{\frac{1}{n} \sum_{i=1}^{n} (y_i - \hat{y}_i)^2}")
        st.markdown(f"**Meaning:** Square root of MSE, bringing the error back to dollar units. The typical prediction deviation is **${mm['RMSE']:,.2f}**.")

    st.markdown("---")
    # R²
    col1, col2 = st.columns([1, 2])
    with col1:
        st.markdown(metric_card("R²", f"{mm['R2']:.4f}", f"{mm['R2']*100:.2f}% variance explained"), unsafe_allow_html=True)
    with col2:
        st.latex(r"R^2 = 1 - \frac{\sum (y_i - \hat{y}_i)^2}{\sum (y_i - \bar{y})^2}")
        st.markdown(f"**Meaning:** Proportion of variance in house prices explained by the model. R² = {mm['R2']:.4f} means **{mm['R2']*100:.2f}%** of price variation is captured.")

    st.markdown("---")
    # Adjusted R²
    col1, col2 = st.columns([1, 2])
    adj = mm.get("Adjusted_R2")
    with col1:
        st.markdown(metric_card("Adjusted R²", f"{adj:.4f}" if adj else "N/A", "Penalised for extra features"), unsafe_allow_html=True)
    with col2:
        st.latex(r"R^2_{adj} = 1 - \frac{(1 - R^2)(n - 1)}{n - p - 1}")
        n = results["metadata"]["test_samples"] + results["metadata"]["train_samples"]
        p = len(FEATURE_COLS)
        st.markdown(f"**Meaning:** Adjusts R² for the number of predictors (p = {p}) and observations (n = {n}). "
                   f"Prevents overestimation when adding features that do not improve the model.")


# ══════════════════════════════════════════════════════════════
# PAGE 8: RESIDUAL ANALYSIS
# ══════════════════════════════════════════════════════════════
elif page == PAGES[7]:
    st.title("🔍 Residual Analysis")

    results = cached_train()
    y_test = results["y_test"]
    y_pred = results["multiple"]["predictions"]
    res_stats = compute_residual_stats(y_test, y_pred)
    residuals = res_stats["residuals"]

    section_header("What Are Residuals?")
    st.markdown("**Residual = Actual House Price − Predicted House Price**")
    st.latex(r"e_i = y_i - \hat{y}_i")
    st.markdown("Residuals represent the prediction errors. Analysing their distribution and patterns helps assess model quality.")

    section_header("Residual Statistics")
    c1, c2, c3, c4 = st.columns(4)
    c1.markdown(metric_card("Mean", f"${res_stats['mean']:,.2f}", "Should be ≈ 0"), unsafe_allow_html=True)
    c2.markdown(metric_card("Std Dev", f"${res_stats['std']:,.2f}", "Spread of errors"), unsafe_allow_html=True)
    c3.markdown(metric_card("Skewness", f"{res_stats['skewness']:.4f}", "0 = symmetric"), unsafe_allow_html=True)
    c4.markdown(metric_card("Kurtosis", f"{res_stats['kurtosis']:.4f}", "0 = normal-like"), unsafe_allow_html=True)

    section_header("1. Residuals vs Predicted Values")
    fig1 = go.Figure()
    fig1.add_trace(go.Scatter(x=y_pred, y=residuals, mode="markers",
                              marker=dict(color="#1e3a5f", opacity=0.6),
                              name="Residuals"))
    fig1.add_hline(y=0, line_dash="dash", line_color="red")
    fig1.update_layout(title="Residuals vs Predicted Values",
                       xaxis_title="Predicted House Price ($)",
                       yaxis_title="Residual ($)")
    st.plotly_chart(fig1, use_container_width=True)
    st.markdown('<div class="info-box"><strong>Interpretation:</strong> If residuals are randomly scattered around the zero line without a clear pattern (no funnel, curve, or clustering), the model assumptions of linearity and constant variance appear reasonable.</div>', unsafe_allow_html=True)

    section_header("2. Residual Distribution Histogram")
    fig2 = px.histogram(x=residuals, nbins=30, color_discrete_sequence=["#3b82f6"],
                        title="Distribution of Residuals",
                        labels={"x": "Residual ($)", "count": "Frequency"})
    fig2.add_vline(x=0, line_dash="dash", line_color="red")
    st.plotly_chart(fig2, use_container_width=True)
    st.markdown(f'<div class="info-box"><strong>Interpretation:</strong> The distribution appears {"approximately symmetric" if abs(res_stats["skewness"]) < 1 else "skewed"} (skewness = {res_stats["skewness"]:.4f}). {"This supports the normality assumption." if abs(res_stats["skewness"]) < 1 else "The normality assumption may be questionable."}</div>', unsafe_allow_html=True)

    section_header("3. Predicted vs Actual House Prices")
    fig3 = go.Figure()
    fig3.add_trace(go.Scatter(x=np.array(y_test), y=y_pred, mode="markers",
                              marker=dict(color="#1e3a5f", opacity=0.6),
                              name="Predictions"))
    min_val = min(np.min(y_test), np.min(y_pred))
    max_val = max(np.max(y_test), np.max(y_pred))
    fig3.add_trace(go.Scatter(x=[min_val, max_val], y=[min_val, max_val],
                              mode="lines", line=dict(color="red", dash="dash"),
                              name="Ideal (y = x)"))
    fig3.update_layout(title="Predicted vs Actual House Prices",
                       xaxis_title="Actual House Price ($)",
                       yaxis_title="Predicted House Price ($)")
    st.plotly_chart(fig3, use_container_width=True)
    st.markdown('<div class="info-box"><strong>Interpretation:</strong> Points close to the red 45° line indicate accurate predictions. The tighter the clustering around this line, the better the model performs.</div>', unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════
# PAGE 9: MULTICOLLINEARITY & ASSUMPTIONS
# ══════════════════════════════════════════════════════════════
elif page == PAGES[8]:
    st.title("🔗 Multicollinearity & Regression Assumptions")

    df, _ = cached_load()
    results = cached_train()
    y_test = results["y_test"]
    y_pred = results["multiple"]["predictions"]

    # ── Multicollinearity ──
    section_header("Multicollinearity Analysis")
    st.markdown("""
    **What is multicollinearity?** When input features are highly correlated with each other,
    individual coefficient estimates become unstable — small changes in data can produce large
    swings in coefficients. The model's overall predictions may still be good, but interpreting
    individual coefficients becomes unreliable.
    """)

    st.markdown("**Correlation Heatmap (Input Features Only)**")
    corr_feat = df[FEATURE_COLS].corr()
    fig_corr = px.imshow(corr_feat.round(3), text_auto=True, color_continuous_scale="RdBu_r",
                         aspect="auto", zmin=-1, zmax=1,
                         title="Feature-to-Feature Correlation Matrix")
    st.plotly_chart(fig_corr, use_container_width=True)

    st.markdown("**Variance Inflation Factor (VIF)**")
    vif_df = compute_vif(df)
    vif_df["Interpretation"] = vif_df["VIF"].apply(interpret_vif)
    st.dataframe(vif_df, use_container_width=True, hide_index=True)

    high_corr = get_high_correlations(df)
    if len(high_corr) > 0:
        st.markdown("**Highly Correlated Feature Pairs (|r| ≥ 0.7):**")
        st.dataframe(high_corr, use_container_width=True, hide_index=True)
    else:
        st.markdown("**No feature pairs have |correlation| ≥ 0.7.** Multicollinearity risk is low.")

    st.markdown("""<div class="info-box"><strong>Decision:</strong> Even if some features show moderate VIF,
    we retain all features because: (1) the model's predictive accuracy is high, (2) removing features
    would reduce information, and (3) no feature has been automatically dropped based on a threshold alone.
    Feature changes, if any, are documented here.</div>""", unsafe_allow_html=True)

    # ── Regression Assumptions ──
    section_header("Five Assumptions of Linear Regression")
    res_stats = compute_residual_stats(y_test, y_pred)

    st.markdown("### 1. Linearity")
    st.markdown("""The relationship between features and target should be approximately linear.
    **Evidence:** The scatter plots on the Data Visualization page show roughly linear trends
    for key features like Square_Footage. The Residuals vs Predicted plot should show no
    systematic curve.""")

    st.markdown("### 2. Independence of Errors")
    st.markdown("""Residuals should not be correlated with each other.
    **Evidence:** This is a cross-sectional dataset (not time-series), so serial autocorrelation
    is unlikely. Each observation represents an independent house.""")

    st.markdown("### 3. Constant Variance of Errors (Homoscedasticity)")
    st.markdown(f"""The spread of residuals should remain roughly constant across predicted values.
    **Evidence:** Check the Residuals vs Predicted plot on the Residual Analysis page.
    {"The residuals appear randomly scattered." if abs(res_stats['skewness']) < 1 else "Some pattern may be present."}""")

    st.markdown("### 4. Approximately Normal Errors")
    st.markdown(f"""Residuals should follow an approximately normal distribution.
    **Evidence:** Skewness = {res_stats['skewness']:.4f}, Kurtosis = {res_stats['kurtosis']:.4f}.
    Shapiro-Wilk p-value = {res_stats['shapiro_p']:.4f}.
    {"The normality assumption appears reasonable." if res_stats['shapiro_p'] > 0.05 else "The Shapiro-Wilk test suggests deviation from normality, though for large samples this test can be overly sensitive."}""")

    st.markdown("### 5. No Severe Multicollinearity")
    max_vif = vif_df["VIF"].max()
    st.markdown(f"""Features should not be excessively correlated with each other.
    **Evidence:** Maximum VIF = {max_vif:.2f}.
    {"All VIF values are below 10, suggesting no severe multicollinearity." if max_vif < 10 else "Some features have VIF ≥ 10, indicating potential multicollinearity."}""")

    st.markdown("""<div class="warn-box"><strong>Important:</strong> No assumption should be claimed as
    perfectly satisfied without thorough statistical evidence. The diagnostics above provide evidence
    for or against each assumption. In practice, minor violations are common and may not invalidate
    the model's usefulness.</div>""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════
# PAGE 10: HOUSE PRICE PREDICTION
# ══════════════════════════════════════════════════════════════
elif page == PAGES[9]:
    st.title("🏡 Predict a New House Price")

    pipeline, metadata = load_model()
    if pipeline is None:
        st.error("⚠️ No trained model found. Please go to 'Model Training' and train the model first.")
        st.stop()

    train_ranges = metadata.get("train_ranges", {}) if metadata else {}

    section_header("Enter House Details")
    col1, col2 = st.columns(2)

    with col1:
        sqft = st.number_input("Square Footage (sq ft)", min_value=100, max_value=15000, value=2500, step=50)
        bedrooms = st.number_input("Number of Bedrooms", min_value=1, max_value=10, value=3, step=1)
        bathrooms = st.number_input("Number of Bathrooms", min_value=1, max_value=10, value=2, step=1)
        year_built = st.number_input("Year Built", min_value=1800, max_value=2030, value=2000, step=1)

    with col2:
        lot_size = st.number_input("Lot Size (acres)", min_value=0.01, max_value=20.0, value=2.5, step=0.1)
        garage = st.number_input("Garage Size (cars)", min_value=0, max_value=5, value=2, step=1)
        quality = st.slider("Neighborhood Quality (1-10)", min_value=1, max_value=10, value=7)

    # Extrapolation warnings
    input_values = {
        "Square_Footage": sqft, "Num_Bedrooms": bedrooms, "Num_Bathrooms": bathrooms,
        "Year_Built": year_built, "Lot_Size": lot_size, "Garage_Size": garage,
        "Neighborhood_Quality": quality,
    }
    warnings = []
    for feat, val in input_values.items():
        if feat in train_ranges:
            r = train_ranges[feat]
            if val < r["min"] or val > r["max"]:
                warnings.append(f"**{feat}** = {val} is outside training range [{r['min']:.2f}, {r['max']:.2f}]")

    if warnings:
        st.markdown(f"""<div class="warn-box"><strong>⚠️ Extrapolation Warning:</strong> The following inputs
        are outside the observed training data range. Predictions may be unreliable:<br>
        {'<br>'.join(warnings)}<br><br>
        Predictions outside the training data range may be unreliable because the model has not seen
        such values during training.</div>""", unsafe_allow_html=True)

    if st.button("🔮 Predict House Price", type="primary"):
        import pandas as _pd
        input_df = _pd.DataFrame([input_values])
        predicted = pipeline.predict(input_df)[0]

        st.markdown(f"""
        <div style="background: linear-gradient(135deg, #065f46, #064e3b); padding: 2rem;
             border-radius: 12px; text-align: center; color: white; margin: 1rem 0;">
            <div style="font-size: 0.9rem; color: #a7f3d0;">Estimated House Price</div>
            <div style="font-size: 3rem; font-weight: 800; margin: 0.5rem 0;">${predicted:,.2f}</div>
            <div style="font-size: 0.8rem; color: #6ee7b7;">USD — Predicted by Multiple Linear Regression</div>
        </div>
        """, unsafe_allow_html=True)

        section_header("Your House Details")
        detail_df = pd.DataFrame([{
            "Feature": feat,
            "Value": f"{val:,.2f}" if isinstance(val, float) else str(val),
            "Unit": FEATURE_INFO[feat]["unit"],
        } for feat, val in input_values.items()])
        st.dataframe(detail_df, use_container_width=True, hide_index=True)


# ══════════════════════════════════════════════════════════════
# PAGE 11: ABOUT PROJECT
# ══════════════════════════════════════════════════════════════
elif page == PAGES[10]:
    st.title("ℹ️ About This Project")

    section_header("Project Information")
    st.markdown("""
    | Item | Detail |
    |---|---|
    | **Project Title** | House Price Prediction Using Multiple Linear Regression |
    | **Academic Context** | CIE 2 — Activity 1: Multiple Linear Regression Worksheet |
    | **Algorithm** | Multiple Linear Regression (Ordinary Least Squares) |
    | **Framework** | Streamlit (Python Web Application) |
    | **ML Library** | Scikit-learn |
    | **Visualization** | Plotly (interactive), Matplotlib, Seaborn |
    | **Statistics** | Statsmodels (VIF), SciPy (normality tests) |
    | **Model Persistence** | Joblib |
    | **Testing** | Pytest |
    """)

    section_header("Dashboard Pages")
    st.markdown("""
    1. **Overview** — Problem statement, features, and current model performance.
    2. **Dataset Explorer** — Data inspection, types, missing values, statistics.
    3. **Data Visualization** — Interactive EDA charts with interpretations.
    4. **Manual Calculation** — Step-by-step Normal Equation with small sample.
    5. **Single vs Multiple Regression** — Baseline, single, and multiple model comparison.
    6. **Model Training** — Train models, view coefficients and interpretations.
    7. **Model Evaluation** — MAE, MSE, RMSE, R², Adjusted R² with formulas.
    8. **Residual Analysis** — Residual plots and diagnostics.
    9. **Multicollinearity & Assumptions** — VIF, correlation analysis, 5 assumptions.
    10. **House Price Prediction** — Interactive prediction form with validation.
    11. **About Project** — This page.
    """)

    section_header("How to Run Locally")
    st.code("""
# 1. Navigate to project folder
cd D:\\project

# 2. Activate virtual environment
.\\venv\\Scripts\\activate

# 3. Run the Streamlit app
streamlit run app.py
    """, language="bash")

    section_header("Deployment (Streamlit Community Cloud)")
    st.markdown("""
    1. Push the project to a GitHub repository.
    2. Go to [share.streamlit.io](https://share.streamlit.io).
    3. Connect your GitHub repository.
    4. Set the main file path to `app.py`.
    5. Ensure `requirements.txt` is in the repository root.
    6. The dataset must be included in the repository.
    7. No secrets or credentials are needed.
    """)

    section_header("Project Structure")
    st.code("""
project/
├── app.py                  # Streamlit dashboard (this app)
├── requirements.txt        # Python dependencies
├── .gitignore              # Git ignore rules
├── .streamlit/
│   └── config.toml         # Streamlit theme configuration
├── dataset/
│   └── house_price_regression_dataset.csv
├── model/
│   ├── house_price_model.pkl    # Saved pipeline
│   └── metadata.json            # Model metadata
├── src/
│   ├── __init__.py
│   ├── data_loader.py           # Dataset loading & constants
│   ├── model_training.py        # Model training & evaluation
│   ├── manual_calculation.py    # Normal Equation calculations
│   ├── multicollinearity.py     # VIF & correlation analysis
│   └── residual_analysis.py     # Residual diagnostics
├── tests/
│   ├── __init__.py
│   └── test_project.py          # Pytest automated tests
├── visualizations/              # Saved PNG charts
├── preprocessing.py             # Original preprocessing module
├── train.py                     # Original training script
├── predict.py                   # Original prediction module
├── main.py                      # Original CLI entry point
└── web_app.py                   # Original HTML dashboard
    """, language="text")
