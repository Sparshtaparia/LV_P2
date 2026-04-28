"""
Streamlit Multi-Page Dashboard – Customer Churn Intelligence Platform
Pages: Overview | Customer Risk Table | Segment Explorer | What-If Simulator | Campaign Export
"""
import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

import warnings; warnings.filterwarnings("ignore")
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import joblib, json
from io import StringIO

from config.config import (
    PROC_DIR, PLOTS_DIR, CALIBRATED_MODEL, FEATURE_COLS_PATH,
    CLUSTER_MODEL_PATH, MODELS_DIR
)

# ─── Page config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Churn Intelligence Platform",
    page_icon="🔮",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─── Custom CSS ───────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;600;700&family=Inter:wght@300;400;500;600&display=swap');

html, body, [class*="css"] { 
    font-family: 'Inter', sans-serif; 
    color: #e2e8f0;
}
h1, h2, h3, h4, h5, h6 {
    font-family: 'Outfit', sans-serif !important;
    font-weight: 600 !important;
}

/* Base backgrounds */
.main { background: #0b0f19; }
.stApp { 
    background-color: #0b0f19;
    background-image: 
        radial-gradient(at 0% 0%, rgba(99, 102, 241, 0.15) 0px, transparent 50%),
        radial-gradient(at 100% 0%, rgba(168, 85, 247, 0.15) 0px, transparent 50%),
        radial-gradient(at 100% 100%, rgba(236, 72, 153, 0.1) 0px, transparent 50%);
    background-attachment: fixed;
}

/* Glassmorphism Metric Cards */
.metric-card {
    background: rgba(30, 41, 59, 0.4);
    backdrop-filter: blur(12px);
    -webkit-backdrop-filter: blur(12px);
    border: 1px solid rgba(255, 255, 255, 0.08);
    border-radius: 16px;
    padding: 24px;
    text-align: center;
    box-shadow: 0 4px 30px rgba(0, 0, 0, 0.1);
    transition: transform 0.3s ease, border-color 0.3s ease;
}
.metric-card:hover {
    transform: translateY(-5px);
    border-color: rgba(99, 102, 241, 0.4);
}
.metric-card h1 { 
    font-size: 2.5rem; 
    margin: 0; 
    font-weight: 700;
    background: linear-gradient(to right, #818cf8, #c084fc);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    font-family: 'Outfit', sans-serif;
}
.metric-card p { 
    color: #94a3b8; 
    margin: 8px 0 0; 
    font-size: 0.9rem; 
    font-weight: 500;
    text-transform: uppercase;
    letter-spacing: 1px; 
}

/* Risk colors overriding gradient text */
.metric-card h1.risk-val-high { background: linear-gradient(to right, #ef4444, #f87171); -webkit-background-clip: text; -webkit-text-fill-color: transparent; }
.metric-card h1.risk-val-med { background: linear-gradient(to right, #f59e0b, #fbbf24); -webkit-background-clip: text; -webkit-text-fill-color: transparent; }
.metric-card h1.risk-val-low { background: linear-gradient(to right, #10b981, #34d399); -webkit-background-clip: text; -webkit-text-fill-color: transparent; }

/* Status Badges */
.risk-high  { color: #ef4444; font-weight: 700; background: rgba(239, 68, 68, 0.1); padding: 2px 8px; border-radius: 4px; }
.risk-med   { color: #f59e0b; font-weight: 700; background: rgba(245, 158, 11, 0.1); padding: 2px 8px; border-radius: 4px; }
.risk-low   { color: #10b981; font-weight: 700; background: rgba(16, 185, 129, 0.1); padding: 2px 8px; border-radius: 4px; }

/* Hero Section */
.hero {
    position: relative;
    overflow: hidden;
    background: linear-gradient(135deg, rgba(99, 102, 241, 0.9), rgba(168, 85, 247, 0.9));
    padding: 48px 40px;
    border-radius: 24px;
    margin-bottom: 36px;
    box-shadow: 0 20px 40px rgba(99, 102, 241, 0.2);
    border: 1px solid rgba(255, 255, 255, 0.1);
}
.hero-content {
    position: relative;
    z-index: 1;
}
.hero h1 { 
    font-size: 3rem !important; 
    color: white !important; 
    margin: 0; 
    line-height: 1.2;
    text-shadow: 0 2px 10px rgba(0,0,0,0.2);
}
.hero p  { 
    color: rgba(255, 255, 255, 0.9); 
    margin: 12px 0 0; 
    font-size: 1.2rem; 
    font-weight: 400;
}

/* Sidebar styling */
[data-testid="stSidebar"] {
    background-color: #0f172a !important;
    border-right: 1px solid rgba(255, 255, 255, 0.05);
}
.sidebar-logo { 
    font-family: 'Outfit', sans-serif;
    font-size: 1.8rem; 
    font-weight: 700; 
    background: linear-gradient(to right, #818cf8, #c084fc);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    padding: 12px 0 24px; 
    display: flex;
    align-items: center;
    gap: 12px;
}

/* Inputs / Native elements */
.stSelectbox label, .stSlider label { color: #cbd5e1 !important; font-weight: 500 !important; }
div[data-testid="stMetric"] { 
    background: rgba(30, 41, 59, 0.4); 
    backdrop-filter: blur(10px);
    border: 1px solid rgba(255, 255, 255, 0.05);
    border-radius: 12px; 
    padding: 16px; 
}
[data-testid="stDataFrame"] {
    border-radius: 12px;
    overflow: hidden;
    border: 1px solid rgba(255,255,255,0.05);
}
div.stButton > button:first-child {
    background: linear-gradient(135deg, #6366f1, #a855f7);
    color: white;
    border: none;
    border-radius: 8px;
    padding: 8px 24px;
    font-weight: 600;
    transition: all 0.3s ease;
}
div.stButton > button:first-child:hover {
    transform: translateY(-2px);
    box-shadow: 0 8px 20px rgba(99, 102, 241, 0.4);
}
</style>
""", unsafe_allow_html=True)

# ─── Data loading (cached) ────────────────────────────────────────────────────
@st.cache_data(ttl=300)
def load_scored():
    path = PROC_DIR / "customers_scored.csv"
    if path.exists():
        return pd.read_csv(path)
    return None

@st.cache_data(ttl=300)
def load_sim_results():
    path = PLOTS_DIR / "retention_simulation.csv"
    if path.exists():
        return pd.read_csv(path)
    return None

@st.cache_data(ttl=300)
def load_model_comparison():
    path = PLOTS_DIR / "model_comparison.csv"
    if path.exists():
        return pd.read_csv(path, index_col=0)
    return None

@st.cache_resource
def load_model():
    try:
        m = joblib.load(CALIBRATED_MODEL)
        f = joblib.load(FEATURE_COLS_PATH)
        return m, f
    except:
        return None, None

# ─── Sidebar navigation ───────────────────────────────────────────────────────
with st.sidebar:
    st.markdown('<div class="sidebar-logo">🔮 Churn Intel</div>', unsafe_allow_html=True)
    page = st.radio(
        "Navigate",
        ["📊 Overview", "🔍 Customer Risk Table",
         "🗂️ Segment Explorer", "🔬 What-If Simulator",
         "📤 Campaign Export"],
        label_visibility="collapsed"
    )
    st.markdown("---")
    st.markdown("**Model Status**")
    model, feat_cols = load_model()
    if model:
        st.success("✅ Model loaded")
    else:
        st.error("❌ Run pipeline first")
    df = load_scored()
    if df is not None:
        st.info(f"📦 {len(df):,} customers scored")
    st.markdown("---")
    st.caption("LogicVeda · Churn Platform v4.0")

# ═══════════════════════════════════════════════════════════════════════════════
# PAGE 1: OVERVIEW
# ═══════════════════════════════════════════════════════════════════════════════
if page == "📊 Overview":
    st.markdown("""
    <div class="hero">
      <div class="hero-content">
        <h1>🔮 Churn Intelligence Platform</h1>
        <p>Identify who's leaving, why — and exactly what to do about it.</p>
      </div>
    </div>
    """, unsafe_allow_html=True)

    if df is None:
        st.warning("⚠️ No scored data found. Run `master_pipeline.py` first.")
        st.code("venv\\Scripts\\python.exe master_pipeline.py", language="bash")
        st.stop()

    churn_col = "Churn" if "Churn" in df.columns else "churn_label"
    total    = len(df)
    n_churn  = int(df["churn_label"].sum())
    churn_r  = n_churn / total
    avg_risk = df["churn_probability"].mean()
    high_risk_n = int((df["churn_probability"] >= 0.7).sum())

    # KPI row
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.markdown(f'<div class="metric-card"><h1>{total:,}</h1><p>TOTAL CUSTOMERS</p></div>', unsafe_allow_html=True)
    with c2:
        st.markdown(f'<div class="metric-card"><h1 class="risk-val-high">{n_churn:,}</h1><p>PREDICTED TO CHURN</p></div>', unsafe_allow_html=True)
    with c3:
        st.markdown(f'<div class="metric-card"><h1 class="risk-val-med">{churn_r*100:.1f}%</h1><p>CHURN RATE</p></div>', unsafe_allow_html=True)
    with c4:
        st.markdown(f'<div class="metric-card"><h1 class="risk-val-low">{high_risk_n:,}</h1><p>HIGH-RISK (≥70%)</p></div>', unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    col_l, col_r = st.columns([1.2, 1])

    with col_l:
        st.subheader("📈 Churn Probability Distribution")
        fig = px.histogram(df, x="churn_probability", nbins=50,
                           color_discrete_sequence=["#667eea"],
                           labels={"churn_probability": "Churn Probability"})
        fig.add_vline(x=0.5, line_dash="dash", line_color="#FF4B4B",
                      annotation_text="Decision threshold")
        fig.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                          font_color="white", margin=dict(t=20))
        st.plotly_chart(fig, use_container_width=True)

    with col_r:
        st.subheader("🗂️ Segment Distribution")
        seg_counts = df["segment_name"].value_counts().reset_index()
        seg_counts.columns = ["Segment", "Count"]
        fig2 = px.pie(seg_counts, names="Segment", values="Count",
                      color_discrete_sequence=px.colors.qualitative.Bold, hole=0.4)
        fig2.update_layout(paper_bgcolor="rgba(0,0,0,0)", font_color="white",
                           margin=dict(t=20))
        st.plotly_chart(fig2, use_container_width=True)

    # Model performance
    cmp = load_model_comparison()
    if cmp is not None:
        st.subheader("🏆 Model Performance Comparison")
        st.dataframe(
            cmp.style.highlight_max(color="#2a4a2a", axis=0)
                     .format("{:.4f}"),
            use_container_width=True
        )

    # Retention simulation
    sim = load_sim_results()
    if sim is not None:
        st.subheader("💡 Retention Simulation – Expected Churn Reduction")
        fig3 = go.Figure()
        colors = ["#2196F3","#4CAF50","#FF9800","#9C27B0"]
        for i, row in sim.iterrows():
            fig3.add_trace(go.Bar(
                name=row["intervention"],
                x=[row["intervention"]],
                y=[row["mc_mean_reduction"] * 100],
                error_y=dict(
                    type="data",
                    symmetric=False,
                    array=[(row["mc_p95"] - row["mc_mean_reduction"]) * 100],
                    arrayminus=[(row["mc_mean_reduction"] - row["mc_p5"]) * 100],
                ),
                marker_color=colors[i % len(colors)],
            ))
        fig3.update_layout(
            yaxis_title="Churn Reduction (%)",
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
            font_color="white", showlegend=False, margin=dict(t=20)
        )
        st.plotly_chart(fig3, use_container_width=True)

# ═══════════════════════════════════════════════════════════════════════════════
# PAGE 2: CUSTOMER RISK TABLE
# ═══════════════════════════════════════════════════════════════════════════════
elif page == "🔍 Customer Risk Table":
    st.title("🔍 Customer Risk Table")
    if df is None:
        st.warning("Run `master_pipeline.py` first."); st.stop()

    # Filters
    c1, c2, c3 = st.columns(3)
    with c1:
        risk_thresh = st.slider("Min Churn Risk", 0.0, 1.0, 0.5, 0.05)
    with c2:
        seg_filter = st.multiselect("Segment", df["segment_name"].unique(),
                                     default=list(df["segment_name"].unique()))
    with c3:
        contract_f = st.multiselect("Contract", df["Contract"].unique(),
                                     default=list(df["Contract"].unique())) \
                     if "Contract" in df.columns else None

    view = df[df["churn_probability"] >= risk_thresh].copy()
    if seg_filter:
        view = view[view["segment_name"].isin(seg_filter)]
    if contract_f and "Contract" in view.columns:
        view = view[view["Contract"].isin(contract_f)]

    def risk_badge(p):
        if p >= 0.7:   return f'<span class="risk-high">🔴 {p:.2f}</span>'
        elif p >= 0.4: return f'<span class="risk-med">🟡 {p:.2f}</span>'
        else:          return f'<span class="risk-low">🟢 {p:.2f}</span>'

    display_cols = [c for c in ["customerID","tenure","MonthlyCharges","TotalCharges",
                                 "Contract","churn_probability","segment_name"]
                    if c in view.columns]
    st.info(f"Showing **{len(view):,}** customers")
    st.dataframe(
        view[display_cols].sort_values("churn_probability", ascending=False)
                          .reset_index(drop=True),
        use_container_width=True
    )

# ═══════════════════════════════════════════════════════════════════════════════
# PAGE 3: SEGMENT EXPLORER
# ═══════════════════════════════════════════════════════════════════════════════
elif page == "🗂️ Segment Explorer":
    st.title("🗂️ Segment Explorer")
    if df is None:
        st.warning("Run `master_pipeline.py` first."); st.stop()

    selected_seg = st.selectbox("Select Segment", ["All"] + list(df["segment_name"].unique()))
    view = df if selected_seg == "All" else df[df["segment_name"] == selected_seg]

    col1, col2 = st.columns(2)
    with col1:
        fig = px.scatter(view, x="tenure" if "tenure" in view.columns else view.index,
                         y="churn_probability", color="segment_name",
                         color_discrete_sequence=px.colors.qualitative.Bold,
                         labels={"churn_probability":"Churn Risk","tenure":"Tenure (months)"},
                         opacity=0.6, title="Tenure vs Churn Risk by Segment")
        fig.update_layout(paper_bgcolor="rgba(0,0,0,0)",plot_bgcolor="rgba(0,0,0,0)",
                          font_color="white", margin=dict(t=40))
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        if "MonthlyCharges" in view.columns:
            fig2 = px.box(view, x="segment_name", y="MonthlyCharges",
                          color="segment_name",
                          color_discrete_sequence=px.colors.qualitative.Bold,
                          title="Monthly Charges by Segment")
            fig2.update_layout(paper_bgcolor="rgba(0,0,0,0)",plot_bgcolor="rgba(0,0,0,0)",
                               font_color="white", showlegend=False, margin=dict(t=40))
            st.plotly_chart(fig2, use_container_width=True)

    # Profile table
    profile_cols = [c for c in ["segment_name","churn_probability","tenure",
                                  "MonthlyCharges","TotalCharges"] if c in view.columns]
    profile = view[profile_cols].groupby("segment_name").agg(
        Count   = ("churn_probability","count"),
        Avg_Risk= ("churn_probability","mean"),
        Avg_Tenure = ("tenure","mean") if "tenure" in view.columns else ("churn_probability","count"),
        Avg_Monthly= ("MonthlyCharges","mean") if "MonthlyCharges" in view.columns else ("churn_probability","count"),
    ).round(3).reset_index()
    st.subheader("Segment Profiles")
    st.dataframe(profile, use_container_width=True)

# ═══════════════════════════════════════════════════════════════════════════════
# PAGE 4: WHAT-IF SIMULATOR
# ═══════════════════════════════════════════════════════════════════════════════
elif page == "🔬 What-If Simulator":
    st.title("🔬 What-If Churn Simulator")
    st.markdown("Adjust customer attributes and see how churn probability changes in real-time.")

    if model is None:
        st.error("Model not loaded. Run `master_pipeline.py` first."); st.stop()

    col1, col2 = st.columns([1, 1])
    with col1:
        st.subheader("Customer Attributes")
        tenure          = st.slider("Tenure (months)", 0, 72, 12)
        monthly_charges = st.slider("Monthly Charges ($)", 18.0, 120.0, 65.0, 0.5)
        total_charges   = monthly_charges * tenure
        contract        = st.selectbox("Contract", ["Month-to-month","One year","Two year"])
        internet        = st.selectbox("Internet Service", ["Fiber optic","DSL","No"])
        online_sec      = st.selectbox("Online Security", ["Yes","No","No internet service"])
        tech_support    = st.selectbox("Tech Support", ["Yes","No","No internet service"])
        paperless       = st.selectbox("Paperless Billing", ["Yes","No"])
        payment_method  = st.selectbox("Payment Method",
                                        ["Electronic check","Mailed check",
                                         "Bank transfer (automatic)","Credit card (automatic)"])

    with col2:
        st.subheader("Churn Risk Prediction")

        def build_input():
            row = {c: 0 for c in feat_cols}
            row["tenure"]         = (tenure - 35) / 25
            row["MonthlyCharges"] = (monthly_charges - 65) / 30
            row["TotalCharges"]   = (total_charges - 2000) / 2500
            row["Is_Month_to_Month_Yes"] = 1 if contract == "Month-to-month" else 0
            row["Is_Two_Year_Yes"]       = 1 if contract == "Two year" else 0
            row["Has_Fiber"]             = 1 if internet == "Fiber optic" else 0
            row["Has_DSL"]               = 1 if internet == "DSL" else 0
            row["Is_Paperless"]          = 1 if paperless == "Yes" else 0
            for k in [f"OnlineSecurity_{online_sec}", f"TechSupport_{tech_support}",
                      f"PaymentMethod_{payment_method}"]:
                if k in row: row[k] = 1
            return pd.DataFrame([row])

        inp   = build_input()
        proba = float(model.predict_proba(inp)[0, 1])
        label = "🔴 HIGH RISK" if proba >= 0.7 else ("🟡 MEDIUM RISK" if proba >= 0.4 else "🟢 LOW RISK")

        # Gauge
        fig = go.Figure(go.Indicator(
            mode="gauge+number",
            value=proba * 100,
            title={"text": "Churn Probability (%)", "font": {"color": "white"}},
            gauge={
                "axis": {"range": [0, 100], "tickcolor": "white"},
                "bar":  {"color": "#667eea"},
                "steps": [
                    {"range": [0, 40],  "color": "#1a2a1a"},
                    {"range": [40, 70], "color": "#2a2a1a"},
                    {"range": [70, 100],"color": "#2a1a1a"},
                ],
                "threshold": {"line":{"color":"#FF4B4B","width":4},"value":50}
            },
            number={"suffix": "%", "font": {"color": "white", "size": 36}},
        ))
        fig.update_layout(paper_bgcolor="rgba(0,0,0,0)", font_color="white",
                          height=280, margin=dict(t=30, b=0))
        st.plotly_chart(fig, use_container_width=True)
        st.markdown(f"<h2 style='text-align:center;'>{label}</h2>", unsafe_allow_html=True)

        # Key drivers
        st.markdown("**Key Risk Factors:**")
        drivers = []
        if contract == "Month-to-month":   drivers.append("📛 Month-to-month contract (+↑)")
        if internet == "Fiber optic":      drivers.append("🌐 Fiber optic service (+↑)")
        if paperless == "Yes":             drivers.append("📄 Paperless billing (+↑)")
        if tenure < 12:                    drivers.append(f"⏱️ Short tenure ({tenure}mo) (+↑)")
        if monthly_charges > 80:           drivers.append(f"💸 High charges (${monthly_charges:.0f}) (+↑)")
        if online_sec == "No":             drivers.append("🔒 No online security (+↑)")
        if not drivers:                    drivers.append("✅ Profile looks stable")
        for d in drivers:
            st.write(d)

# ═══════════════════════════════════════════════════════════════════════════════
# PAGE 5: CAMPAIGN EXPORT
# ═══════════════════════════════════════════════════════════════════════════════
elif page == "📤 Campaign Export":
    st.title("📤 Campaign List Export")
    st.markdown("Export targeted customer lists for marketing campaigns.")

    if df is None:
        st.warning("Run `master_pipeline.py` first."); st.stop()

    c1, c2 = st.columns(2)
    with c1:
        risk_min  = st.slider("Min Churn Risk", 0.0, 1.0, 0.5, 0.05)
        seg_sel   = st.multiselect("Segments", df["segment_name"].unique(),
                                    default=list(df["segment_name"].unique()))
    with c2:
        intervention = st.selectbox("Intervention Type",
            ["Discount (10% off)","Support Outreach","Plan Upgrade Offer","Loyalty Reward"])
        max_customers = st.number_input("Max customers in export", 100, 10000, 1000, 100)

    export_df = df[df["churn_probability"] >= risk_min].copy()
    if seg_sel:
        export_df = export_df[export_df["segment_name"].isin(seg_sel)]
    export_df = export_df.sort_values("churn_probability", ascending=False).head(int(max_customers))
    export_df["recommended_action"] = intervention

    st.info(f"**{len(export_df):,}** customers selected")

    exp_cols = [c for c in ["customerID","segment_name","churn_probability",
                              "tenure","MonthlyCharges","Contract",
                              "recommended_action"] if c in export_df.columns]
    st.dataframe(export_df[exp_cols], use_container_width=True)

    col_csv, col_json = st.columns(2)
    with col_csv:
        csv = export_df[exp_cols].to_csv(index=False)
        st.download_button("⬇️ Download CSV", csv,
                           "campaign_list.csv", "text/csv",
                           use_container_width=True)
    with col_json:
        js = export_df[exp_cols].to_json(orient="records", indent=2)
        st.download_button("⬇️ Download JSON", js,
                           "campaign_list.json", "application/json",
                           use_container_width=True)
