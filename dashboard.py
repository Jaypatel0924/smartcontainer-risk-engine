"""
SmartContainer Risk Engine - Dashboard
Pixel-perfect dark-themed Streamlit dashboard for HackaMINEd-2026
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
import pickle, os, io
from datetime import datetime
from predict import RiskPredictor
from utils import load_data
import warnings
warnings.filterwarnings('ignore')

# ── Page config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="SmartContainer Risk Engine",
    page_icon="",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Color palette (GitHub-dark inspired) ─────────────────────────────────────
BG           = "#0d1117"
CARD         = "#161b22"
BORDER       = "#30363d"
TEXT         = "#e6edf3"
TEXT2        = "#8b949e"
RED          = "#f85149"
YELLOW       = "#d29922"
GREEN        = "#3fb950"
BLUE         = "#58a6ff"
ACCENT       = "#1f6feb"
DARK_SURFACE = "#010409"

# ── Global CSS ───────────────────────────────────────────────────────────────
st.markdown(f"""
<style>
/* ---- root / body ---- */
html, body, [data-testid="stAppViewContainer"], .main, .block-container {{
    background-color: {BG} !important;
    color: {TEXT} !important;
}}
[data-testid="stAppViewBlockContainer"] {{
    padding-top: 1rem !important;
}}
/* ---- sidebar ---- */
[data-testid="stSidebar"] {{
    background-color: {DARK_SURFACE} !important;
    border-right: 1px solid {BORDER};
}}
[data-testid="stSidebar"] * {{
    color: {TEXT2} !important;
}}
[data-testid="stSidebar"] .stRadio label {{
    color: {TEXT} !important;
    font-size: 0.88rem;
    padding: 6px 0;
}}
[data-testid="stSidebar"] .stRadio label:hover {{
    color: {BLUE} !important;
}}
[data-testid="stSidebar"] .stRadio div[role="radiogroup"] {{
    gap: 0rem;
}}
/* inject section headers between radio items */
[data-testid="stSidebar"] .stRadio div[role="radiogroup"] > label:nth-child(3)::before {{
    content: "INTELLIGENCE";
    display: block;
    font-size: 0.58rem;
    font-weight: 700;
    letter-spacing: 0.12em;
    color: {TEXT2} !important;
    padding: 14px 0 4px 0;
    margin-bottom: 2px;
}}
[data-testid="stSidebar"] .stRadio div[role="radiogroup"] > label:nth-child(5)::before {{
    content: "DATA";
    display: block;
    font-size: 0.58rem;
    font-weight: 700;
    letter-spacing: 0.12em;
    color: {TEXT2} !important;
    padding: 14px 0 4px 0;
    margin-bottom: 2px;
}}

/* ---- hide streamlit chrome ---- */
#MainMenu, footer, header {{visibility: hidden;}}

/* ---- metric cards ---- */
[data-testid="stMetric"] {{
    background: {CARD};
    border: 1px solid {BORDER};
    border-radius: 10px;
    padding: 16px 20px;
}}
[data-testid="stMetric"] label {{
    color: {TEXT2} !important;
    font-size: 0.75rem !important;
    text-transform: uppercase;
    letter-spacing: 0.05em;
}}
[data-testid="stMetric"] [data-testid="stMetricValue"] {{
    color: {TEXT} !important;
    font-size: 2rem !important;
    font-weight: 700;
}}

/* ---- custom cards ---- */
.sc-card {{
    background: {CARD};
    border: 1px solid {BORDER};
    border-radius: 10px;
    padding: 20px 24px;
    margin-bottom: 16px;
}}
.sc-card-title {{
    font-size: 0.7rem;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: 0.08em;
    color: {TEXT};
    margin-bottom: 14px;
    display: flex;
    justify-content: space-between;
    align-items: center;
}}
.sc-badge {{
    font-size: 0.6rem;
    padding: 2px 8px;
    border-radius: 4px;
    background: {BORDER};
    color: {TEXT2};
}}
.sc-badge-green {{background: #238636; color: #fff;}}
.sc-badge-yellow {{background: #9e6a03; color: #fff;}}
.sc-badge-red {{background: #da3633; color: #fff;}}
.sc-badge-blue {{background: {ACCENT}; color: #fff;}}

/* kpi big number */
.kpi-value {{font-size: 2.4rem; font-weight: 700; color: {TEXT}; line-height:1.1;}}
.kpi-sub {{font-size: 0.75rem; color: {TEXT2}; margin-top:2px;}}
.kpi-pct {{font-size: 0.85rem; font-weight: 600;}}
.kpi-pct-red {{color: {RED};}}
.kpi-pct-yellow {{color: {YELLOW};}}
.kpi-pct-green {{color: {GREEN};}}

/* bar rows */
.bar-row {{
    display: flex; align-items: center; margin: 6px 0; font-size:0.82rem;
}}
.bar-label {{width: 120px; color: {TEXT}; flex-shrink:0;}}
.bar-track {{flex:1; height:8px; background:{BORDER}; border-radius:4px; margin:0 10px; position:relative;}}
.bar-fill {{height:100%; border-radius:4px; position:absolute; left:0; top:0;}}
.bar-val {{width:50px; text-align:right; color:{TEXT2}; flex-shrink:0;}}

/* port bar rows */
.port-row {{
    display: flex; align-items: center; margin: 5px 0; font-size:0.8rem;
}}
.port-label {{width:70px; color:{TEXT}; flex-shrink:0; font-weight:600;}}
.port-bar-track {{flex:1; height:6px; background:{BORDER}; border-radius:3px; margin:0 8px;}}
.port-bar-fill {{height:100%; border-radius:3px;}}
.port-val {{width:40px; text-align:right; color:{TEXT2}; flex-shrink:0;}}

/* perf grid */
.perf-grid {{display:grid; grid-template-columns:1fr 1fr; gap:8px; margin-bottom:12px;}}
.perf-box {{text-align:center; padding:10px 0;}}
.perf-num {{font-size:1.6rem; font-weight:700;}}
.perf-lbl {{font-size:0.65rem; text-transform:uppercase; color:{TEXT2}; letter-spacing:0.05em;}}

/* score-band */
.band-row {{display:flex; align-items:center; margin:5px 0; font-size:0.8rem;}}
.band-color {{width:36px; text-align:center; padding:2px 4px; border-radius:4px; font-weight:700; font-size:0.7rem; margin-right:8px; flex-shrink:0;}}
.band-count {{width:110px; color:{TEXT2}; flex-shrink:0;}}
.band-track {{flex:1; height:10px; background:{BORDER}; border-radius:5px; margin:0 10px;}}
.band-fill {{height:100%; border-radius:5px;}}
.band-pct {{width:40px; text-align:right; color:{TEXT2}; flex-shrink:0;}}

/* top-bar */
.topbar {{
    display:flex; justify-content:space-between; align-items:center;
    padding:8px 0 12px 0; border-bottom:1px solid {BORDER}; margin-bottom:18px;
}}
.topbar-left h2 {{margin:0; font-size:1.3rem; font-weight:700; color:{TEXT};}}
.topbar-left p {{margin:0; font-size:0.72rem; color:{TEXT2};}}
.topbar-right {{display:flex; gap:10px; align-items:center; font-size:0.72rem;}}

/* footer status */
.status-bar {{
    position:fixed; bottom:0; left:0; width:100%; padding:6px 24px;
    background:{DARK_SURFACE}; border-top:1px solid {BORDER};
    font-size:0.72rem; color:{TEXT2}; z-index:100;
}}
.status-dot {{display:inline-block; width:8px; height:8px; border-radius:50%;
              background:{GREEN}; margin-right:6px; vertical-align:middle;}}

/* prediction table */
.pred-table {{width:100%; border-collapse:collapse; font-size:0.78rem;}}
.pred-table th {{
    text-align:left; padding:10px 12px; color:{TEXT2}; border-bottom:1px solid {BORDER};
    font-size:0.68rem; text-transform:uppercase; letter-spacing:0.05em; font-weight:600;
}}
.pred-table td {{padding:9px 12px; border-bottom:1px solid #21262d; color:{TEXT};}}
.pred-table tr:hover {{background:#1c2128;}}
.level-badge {{
    padding:3px 10px; border-radius:4px; font-size:0.68rem; font-weight:700;
    text-transform:uppercase; letter-spacing:0.04em; display:inline-block;
}}
.level-critical {{background:#da363322; color:{RED}; border:1px solid #da363366;}}
.level-low {{background:#9e6a0322; color:{YELLOW}; border:1px solid #9e6a0366;}}
.level-clear {{background:#23863622; color:{GREEN}; border:1px solid #23863666;}}

/* score bar in table */
.score-bar-wrap {{display:flex; align-items:center; gap:6px;}}
.score-num {{font-weight:700; width:30px;}}
.score-track {{width:80px; height:4px; background:{BORDER}; border-radius:2px;}}
.score-fill {{height:100%; border-radius:2px; background:{RED};}}

/* filter buttons */
.filter-btn {{
    display:inline-block; padding:5px 14px; border-radius:6px; font-size:0.72rem;
    font-weight:600; cursor:pointer; margin-right:4px; border:1px solid {BORDER};
}}
.filter-active {{background:{ACCENT}; color:#fff; border-color:{ACCENT};}}
.filter-inactive {{background:transparent; color:{TEXT2};}}

/* import area */
.import-zone {{
    border:2px dashed {BORDER}; border-radius:12px; padding:60px 40px;
    text-align:center; background:{CARD}; margin:20px 0;
}}
.import-zone h3 {{color:{TEXT}; margin:12px 0 4px 0;}}
.import-zone p {{color:{TEXT2}; font-size:0.8rem;}}

/* config cards */
.cfg-card {{
    background:{CARD}; border:1px solid {BORDER}; border-radius:8px;
    padding:18px 20px; text-align:left;
}}
.cfg-label {{font-size:0.6rem; text-transform:uppercase; letter-spacing:0.08em; color:{TEXT2}; margin-bottom:6px;}}
.cfg-value {{font-size:1.1rem; font-weight:700; color:{TEXT};}}
.cfg-sub {{font-size:0.72rem; color:{TEXT2};}}
.cfg-row {{
    display:flex; justify-content:space-between; align-items:center;
    padding:10px 0; border-bottom:1px solid #21262d; font-size:0.82rem;
}}
.cfg-row-label {{color:{TEXT2};}}
.cfg-row-val {{font-weight:600; font-family:monospace;}}

/* scrollbar */
::-webkit-scrollbar {{width:6px;}}
::-webkit-scrollbar-track {{background:{BG};}}
::-webkit-scrollbar-thumb {{background:{BORDER}; border-radius:3px;}}

/* general overrides */
.stTabs [data-baseweb="tab-list"] {{background:transparent;}}
h1,h2,h3,h4 {{color:{TEXT} !important;}}

/* topbar import button - looks like text */
.topbar-import [data-testid="stBaseButton-secondary"] {{
    background: transparent !important;
    border: none !important;
    color: {TEXT2} !important;
    padding: 0 !important;
    font-size: 0.72rem !important;
    min-height: unset !important;
    height: auto !important;
    line-height: 1.4 !important;
    white-space: nowrap !important;
    box-shadow: none !important;
}}
.topbar-import [data-testid="stBaseButton-secondary"]:hover {{
    color: {TEXT} !important;
    background: transparent !important;
}}

/* compact KPI strip */
.kpi-strip {{display:flex;gap:12px;margin-bottom:8px;}}
.kpi-strip-card {{
    flex:1;background:{CARD};border:1px solid {BORDER};border-radius:8px;
    padding:14px 18px;display:flex;justify-content:space-between;align-items:center;
}}
.kpi-strip-label {{font-size:0.72rem;font-weight:600;text-transform:uppercase;letter-spacing:0.06em;color:{TEXT};}}
</style>
""", unsafe_allow_html=True)


# ==============================================================================
#  DATA LOADING (cached)
# ==============================================================================

def get_predictor():
    return RiskPredictor()

def get_historical_predictions():
    """Run predictions on historical data (stored in session_state)."""
    if 'hist_preds' in st.session_state and st.session_state.hist_preds is not None:
        return st.session_state.hist_preds
    df = load_data('data/historical_data.csv')
    predictor = get_predictor()
    preds = predictor.predict(df)
    # merge back useful original columns
    preds['Origin_Country']   = df['Origin_Country'].values
    preds['Destination_Port'] = df['Destination_Port'].values
    preds['Dwell_Time_Hours'] = df['Dwell_Time_Hours'].values
    preds['Declared_Value']   = df['Declared_Value'].values
    preds['HS_Code']          = df['HS_Code'].values
    preds['Trade_Regime']     = df['Trade_Regime (Import / Export / Transit)'].values
    preds['Declared_Weight']  = df['Declared_Weight'].values
    preds['Measured_Weight']  = df['Measured_Weight'].values
    st.session_state.hist_preds = preds
    return preds

@st.cache_data(ttl=60)
def get_feature_importances():
    """Return feature importance dict from the trained RF model."""
    rf = pickle.load(open('models/random_forest_model.pkl', 'rb'))
    fe = pickle.load(open('models/feature_engineer.pkl', 'rb'))
    # reconstruct feature names used during training
    sample = load_data('data/historical_data.csv').head(5)
    X = fe.transform(sample)
    cols_to_drop = ['Container_ID', 'Declaration_Date (YYYY-MM-DD)',
                    'Declaration_Time', 'Importer_ID', 'Exporter_ID', 'Clearance_Status']
    X = X.drop(columns=cols_to_drop, errors='ignore')
    names = list(X.columns)
    imps = rf.feature_importances_
    # friendly name mapping
    friendly = {
        'Weight_Diff_%': 'Weight Discrepancy %', 'Weight_Ratio': 'Weight Ratio (Meas/Decl)',
        'Dwell_Time_Hours': 'Dwell Time in Hours', 'Very_High_Dwell': 'Very High Dwell (>120h)',
        'High_Dwell': 'High Dwell Flag (>72h)', 'HS_Code': 'HS Code Category (first 2 digits)',
        'Declared_Value': 'Declared Shipment Value (USD)', 'Measured_Weight': 'Measured Weight (kg)',
        'Declaration_Hour': 'Declaration Hour', 'Value_Weight_Ratio': 'Value/Weight Ratio',
        'HS_Code_Risk_Score': 'HS Risk Score', 'Origin_Risk_Score': 'Origin Risk Score',
        'Port_Risk_Score': 'Port Risk Score', 'Origin_Country': 'Origin Country',
        'Destination_Port': 'Destination Port', 'Destination_Country': 'Dest. Country',
        'Trade_Regime (Import / Export / Transit)': 'Trade Regime',
        'Shipping_Line': 'Shipping Line', 'Trade_Risk_Score': 'Trade Risk Score',
        'Anomaly_Score': 'Anomaly Score',
    }
    out = {}
    for n, v in sorted(zip(names, imps), key=lambda x: -x[1]):
        out[friendly.get(n, n)] = round(v * 100, 2)
    return out

def get_active_predictions():
    """Return whatever predictions are active (uploaded or historical)."""
    if 'uploaded_preds' in st.session_state and st.session_state.uploaded_preds is not None:
        return st.session_state.uploaded_preds
    return get_historical_predictions()


# ==============================================================================
#  TOPBAR
# ==============================================================================

def render_topbar():
    now = datetime.now().strftime("%m/%d/%Y, %I:%M:%S %p").replace("AM", "am").replace("PM", "pm")

    col_title, col_rf, col_acc, col_import, col_time = st.columns([4, 2.2, 1.4, 1.4, 2])

    with col_title:
        st.markdown(f"""<div style="padding:2px 0 4px 0;">
            <div style="font-size:1.15rem;font-weight:700;font-style:italic;color:{TEXT};margin:0;">SmartContainer Risk Engine</div>
            <div style="font-size:0.68rem;color:{TEXT2};margin:0;">AI/ML Container Anomaly Detection &amp; Risk Classification &mdash; HackaMINEd-2026</div>
        </div>""", unsafe_allow_html=True)

    with col_rf:
        st.markdown(f"""<div style="text-align:right;padding-top:8px;">
            <span style="border:1px solid {GREEN};color:{GREEN};padding:5px 14px;border-radius:5px;font-size:0.72rem;font-weight:600;white-space:nowrap;">RF + Isolation Forest</span>
        </div>""", unsafe_allow_html=True)

    with col_acc:
        st.markdown(f"""<div style="text-align:center;padding-top:8px;">
            <span style="background:{GREEN};color:#000;padding:5px 12px;border-radius:5px;font-size:0.72rem;font-weight:700;white-space:nowrap;">Acc: 99.82%</span>
        </div>""", unsafe_allow_html=True)

    with col_import:
        st.markdown('<div class="topbar-import">', unsafe_allow_html=True)
        if st.button("\u2191 Import CSV", key="topbar_import_btn"):
            st.session_state._goto_import = True
            st.rerun()
        st.markdown('</div>', unsafe_allow_html=True)

    with col_time:
        st.markdown(f"""<div style="text-align:right;padding-top:10px;">
            <span style="font-size:0.72rem;color:{TEXT2};white-space:nowrap;">{now}</span>
        </div>""", unsafe_allow_html=True)

    st.markdown(f'<hr style="border:none;border-top:1px solid {BORDER};margin:0 0 14px 0;">', unsafe_allow_html=True)


# ==============================================================================
#  SIDEBAR
# ==============================================================================

def render_sidebar():
    with st.sidebar:
        st.markdown(f"""
        <div style="text-align:center; padding:16px 0 8px 0;">
            <div style="display:inline-flex;align-items:center;justify-content:center;
                        width:42px;height:42px;border-radius:10px;background:{ACCENT};
                        font-size:1.2rem;font-weight:800;color:#fff;margin-bottom:6px;">SC</div>
            <div style="font-size:1rem;font-weight:700;color:{TEXT} !important;">SmartContainer</div>
            <div style="font-size:0.7rem;color:{TEXT2} !important;">Risk Engine v2.0</div>
            <div style="margin-top:8px;">
                <span style="background:{ACCENT};color:#fff;padding:3px 12px;border-radius:4px;
                             font-size:0.68rem;font-weight:600;">HackaMINEd-2026</span>
            </div>
        </div>
        <hr style="border-color:{BORDER};margin:12px 0;">
        """, unsafe_allow_html=True)

        st.markdown(f"<p style='font-size:0.58rem;font-weight:700;letter-spacing:0.12em;color:{TEXT2} !important;margin-bottom:2px;'>OVERVIEW</p>", unsafe_allow_html=True)
        page = st.radio("nav", [
            "\U0001F4CA  Dashboard",
            "\U0001F50D  Predictions",
            "\U0001F4C8  Feature Analysis",
            "\u2A3E  Model Metrics",
            "\u2B06  Import CSV",
            "\u2699  Configuration",
        ], label_visibility="collapsed", key="nav")
        # strip icon prefix for page routing
        page = page.split("  ", 1)[-1] if page else "Dashboard"

        # Show status bar info
        preds = get_active_predictions()
        n = len(preds) if preds is not None else 0
        st.markdown(f"""
        <div style="position:absolute;bottom:0;left:0;width:100%;padding:10px 16px;
                    font-size:0.72rem;color:{TEXT2};">
            <span class="status-dot"></span> Engine online &mdash; {n:,} records
        </div>
        """, unsafe_allow_html=True)

    return page


# ==============================================================================
#  PAGE 1 - DASHBOARD
# ==============================================================================

def render_kpi_strip():
    """Compact KPI indicator strip at top of Dashboard page."""
    preds = get_active_predictions()
    if preds is None:
        return
    total    = len(preds)
    critical = int((preds['Risk_Level'] == 'Critical').sum())
    low_risk = int((preds['Risk_Level'] == 'Low Risk').sum())
    clear    = int((preds['Risk_Level'] == 'Clear').sum())
    st.markdown(f"""
    <div class="kpi-strip">
        <div class="kpi-strip-card">
            <div>
                <div class="kpi-strip-label">TOTAL</div>
                <div style="font-size:1.5rem;font-weight:700;color:{TEXT};margin-top:4px;">{total:,}</div>
            </div>
            <span style="font-size:1.2rem;color:{BLUE};">&#128230;</span>
        </div>
        <div class="kpi-strip-card">
            <div>
                <div class="kpi-strip-label">CRITICAL</div>
                <div style="font-size:1.5rem;font-weight:700;color:{TEXT};margin-top:4px;">{critical:,}</div>
            </div>
            <span style="font-size:1.1rem;color:{YELLOW};">&#9888;</span>
        </div>
        <div class="kpi-strip-card">
            <div>
                <div class="kpi-strip-label">LOW RISK</div>
                <div style="font-size:1.5rem;font-weight:700;color:{TEXT};margin-top:4px;">{low_risk:,}</div>
            </div>
            <span style="font-size:1.1rem;color:{YELLOW};">&#9678;</span>
        </div>
        <div class="kpi-strip-card">
            <div>
                <div class="kpi-strip-label">CLEARED</div>
                <div style="font-size:1.5rem;font-weight:700;color:{TEXT};margin-top:4px;">{clear:,}</div>
            </div>
            <span style="font-size:1.1rem;color:{GREEN};">&#10004;</span>
        </div>
    </div>
    """, unsafe_allow_html=True)


def page_dashboard():
    preds = get_active_predictions()
    if preds is None:
        st.error("No predictions available. Train the model first or import a CSV.")
        return

    total    = len(preds)
    critical = int((preds['Risk_Level'] == 'Critical').sum())
    low_risk = int((preds['Risk_Level'] == 'Low Risk').sum())
    clear    = int((preds['Risk_Level'] == 'Clear').sum())

    # -- Row 2 : Donut / Histogram / Critical Origins --
    c1, c2, c3 = st.columns([1, 1.3, 0.9])

    with c1:
        st.markdown(f'<div class="sc-card"><div class="sc-card-title">RISK DISTRIBUTION <span class="sc-badge">Donut</span></div>', unsafe_allow_html=True)
        fig = go.Figure(go.Pie(
            labels=['Critical', 'Low Risk', 'Clear'],
            values=[critical, low_risk, clear],
            hole=0.55,
            marker=dict(colors=[RED, YELLOW, GREEN]),
            textinfo='none',
            hovertemplate='%{label}: %{value:,}<extra></extra>',
        ))
        fig.add_annotation(text=f"<b>{total:,}</b><br><span style='font-size:10px;color:{TEXT2}'>CONTAINERS</span>",
                           showarrow=False, font=dict(size=18, color=TEXT))
        fig.update_layout(
            paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
            margin=dict(t=10, b=10, l=10, r=10), height=260,
            showlegend=False,
        )
        st.plotly_chart(fig, use_container_width=True, key="donut")
        # legend
        st.markdown(f"""
        <div style="display:flex;justify-content:center;gap:18px;font-size:0.75rem;margin-top:-10px;">
            <span><span style="color:{RED};">&#9632;</span> Critical &nbsp;<b>{critical:,}</b> <span style="color:{TEXT2};">{critical/total*100:.1f}%</span></span>
            <span><span style="color:{YELLOW};">&#9632;</span> Low Risk &nbsp;<b>{low_risk:,}</b> <span style="color:{TEXT2};">{low_risk/total*100:.1f}%</span></span>
            <span><span style="color:{GREEN};">&#9632;</span> Clear &nbsp;<b>{clear:,}</b> <span style="color:{TEXT2};">{clear/total*100:.1f}%</span></span>
        </div></div>""", unsafe_allow_html=True)

    with c2:
        st.markdown(f'<div class="sc-card"><div class="sc-card-title">SCORE HISTOGRAM <span class="sc-badge">All records</span></div>', unsafe_allow_html=True)
        bands = [(0,10), (10,20), (20,40), (40,60), (60,80), (80,100)]
        band_colors = [GREEN, '#2ea043', YELLOW, '#e3b341', RED, '#da3633']
        band_labels = ['0-10','10-20','20-40','40-60','60-80','80-100']
        counts = []
        for lo, hi in bands:
            if hi == 100:
                counts.append(int(((preds['Risk_Score'] >= lo) & (preds['Risk_Score'] <= hi)).sum()))
            else:
                counts.append(int(((preds['Risk_Score'] >= lo) & (preds['Risk_Score'] < hi)).sum()))

        fig = go.Figure(go.Bar(
            x=band_labels, y=counts,
            marker_color=band_colors,
            text=[f"{c/1000:.1f}k" if c >= 1000 else str(c) for c in counts],
            textposition='outside',
            textfont=dict(size=11, color=TEXT),
        ))
        fig.update_layout(
            paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
            margin=dict(t=30, b=40, l=40, r=10), height=250,
            xaxis=dict(color=TEXT2, gridcolor='rgba(0,0,0,0)'),
            yaxis=dict(color=TEXT2, gridcolor='#21262d', showgrid=True),
            font=dict(color=TEXT),
        )
        st.plotly_chart(fig, use_container_width=True, key="histo")
        st.markdown(f'<div style="text-align:center;font-size:0.7rem;color:{TEXT2};margin-top:-10px;">{total:,} containers across all score bands</div></div>', unsafe_allow_html=True)

    with c3:
        st.markdown(f'<div class="sc-card"><div class="sc-card-title">CRITICAL ORIGINS <span class="sc-badge">Top 6</span></div>', unsafe_allow_html=True)
        if 'Origin_Country' in preds.columns:
            crit_df = preds[preds['Risk_Level'] == 'Critical']
            origin_counts = crit_df['Origin_Country'].value_counts().head(6)
            max_val = origin_counts.max() if len(origin_counts) else 1
            rows_html = ""
            for country, cnt in origin_counts.items():
                pct = cnt / max_val * 100
                rows_html += f"""
                <div class="port-row">
                    <span class="port-label">{country}</span>
                    <div class="port-bar-track"><div class="port-bar-fill" style="width:{pct}%;background:{RED};"></div></div>
                    <span class="port-val">{cnt}</span>
                </div>"""
            st.markdown(rows_html + "</div>", unsafe_allow_html=True)
        else:
            st.markdown("</div>", unsafe_allow_html=True)

    # -- Row 3 : Feature Importance / High-Risk Ports / Model Perf --
    c1, c2, c3 = st.columns([1, 1, 1])

    with c1:
        st.markdown(f'<div class="sc-card"><div class="sc-card-title">FEATURE IMPORTANCE <span class="sc-badge sc-badge-blue">RF</span></div>', unsafe_allow_html=True)
        fi = get_feature_importances()
        top_features = dict(list(fi.items())[:8])
        # short names for dashboard compact view
        short_names = {
            'Weight Discrepancy %': 'Weight Diff %', 'Weight Ratio (Meas/Decl)': 'Weight Ratio',
            'Dwell Time in Hours': 'Dwell Time', 'Very High Dwell (>120h)': 'V.High Dwell',
            'High Dwell Flag (>72h)': 'High Dwell', 'HS Code Category (first 2 digits)': 'HS Category',
            'Declared Shipment Value (USD)': 'Decl. Value', 'Measured Weight (kg)': 'Meas. Weight',
        }
        max_imp = max(top_features.values()) if top_features else 1
        rows_html = ""
        for feat, imp in top_features.items():
            pct = imp / max_imp * 100
            label = short_names.get(feat, feat)
            rows_html += f"""
            <div class="bar-row">
                <span class="bar-label">{label}</span>
                <div class="bar-track"><div class="bar-fill" style="width:{pct}%;background:{BLUE};"></div></div>
                <span class="bar-val">{imp:.1f}%</span>
            </div>"""
        st.markdown(rows_html + "</div>", unsafe_allow_html=True)

    with c2:
        st.markdown(f'<div class="sc-card"><div class="sc-card-title">HIGH-RISK PORTS <span class="sc-badge">Avg score</span></div>', unsafe_allow_html=True)
        if 'Destination_Port' in preds.columns:
            port_avg = preds.groupby('Destination_Port')['Risk_Score'].mean().nlargest(8)
            max_s = port_avg.max() if len(port_avg) else 1
            rows_html = ""
            for port, avg in port_avg.items():
                pct = avg / max_s * 100
                rows_html += f"""
                <div class="port-row">
                    <span class="port-label">{port}</span>
                    <div class="port-bar-track"><div class="port-bar-fill" style="width:{pct}%;background:linear-gradient(90deg,{RED},{YELLOW});"></div></div>
                    <span class="port-val">{avg:.2f}</span>
                </div>"""
            st.markdown(rows_html + "</div>", unsafe_allow_html=True)
        else:
            st.markdown("</div>", unsafe_allow_html=True)

    with c3:
        st.markdown(f"""<div class="sc-card">
            <div class="sc-card-title">MODEL PERFORMANCE <span class="sc-badge">Metrics</span></div>
            <div class="perf-grid">
                <div class="perf-box"><div class="perf-num" style="color:{TEXT};">99.82%</div><div class="perf-lbl">Accuracy</div></div>
                <div class="perf-box"><div class="perf-num" style="color:{GREEN};">96.7%</div><div class="perf-lbl">F1 Critical</div></div>
                <div class="perf-box"><div class="perf-num" style="color:{YELLOW};">99.6%</div><div class="perf-lbl">F1 Low Risk</div></div>
                <div class="perf-box"><div class="perf-num" style="color:{BLUE};">{total:,}</div><div class="perf-lbl">Records</div></div>
            </div>
            <div style="font-size:0.72rem;color:{TEXT2};line-height:1.7;">
                <b>Classifier:</b> Random Forest (n=150)<br>
                <b>Anomaly:</b> Isolation Forest (3%)<br>
                <b>Features:</b> 15 engineered<br>
                <b>Explain:</b> Rule-based SHAP-style
            </div>
        </div>""", unsafe_allow_html=True)


# ==============================================================================
#  PAGE 2 - PREDICTIONS
# ==============================================================================

def page_predictions():
    preds = get_active_predictions()
    if preds is None:
        st.error("No predictions. Import data first.")
        return

    total    = len(preds)
    critical = int((preds['Risk_Level'] == 'Critical').sum())
    low_risk = int((preds['Risk_Level'] == 'Low Risk').sum())
    clear    = int((preds['Risk_Level'] == 'Clear').sum())

    st.markdown(f"""
    <h2 style="margin-bottom:2px;">Container Risk Predictions</h2>
    <p style="color:{TEXT2};font-size:0.8rem;margin-top:0;">{total:,} containers &mdash; click any row for full detail</p>
    """, unsafe_allow_html=True)

    # Filters
    col_search, col_btns, col_export = st.columns([2, 3, 1])
    with col_search:
        search = st.text_input("Search", placeholder="Search ID, origin, port, HS code...", label_visibility="collapsed")
    with col_btns:
        level_filter = st.radio("filter", ["ALL", "CRITICAL", "LOW RISK", "CLEAR"],
                                horizontal=True, label_visibility="collapsed", key="pred_filter")
    with col_export:
        csv_out = preds.to_csv(index=False).encode('utf-8')
        st.download_button("Export CSV", csv_out, "risk_predictions.csv", "text/csv",
                           use_container_width=True)

    # Apply filters
    df = preds.copy()
    if level_filter == "CRITICAL":
        df = df[df['Risk_Level'] == 'Critical']
    elif level_filter == "LOW RISK":
        df = df[df['Risk_Level'] == 'Low Risk']
    elif level_filter == "CLEAR":
        df = df[df['Risk_Level'] == 'Clear']

    if search:
        mask = df.apply(lambda r: r.astype(str).str.contains(search, case=False).any(), axis=1)
        df = df[mask]

    # Sort by risk score descending
    df = df.sort_values('Risk_Score', ascending=False)

    st.markdown(f"""<div class="sc-card" style="padding:12px 16px;">
        <span style="font-size:0.75rem;font-weight:600;">All Predictions</span><br>
        <span style="font-size:0.68rem;color:{TEXT2};">{total:,} containers &rarr; {critical:,} Critical, {low_risk:,} Low Risk, {clear:,} Clear</span>
    </div>""", unsafe_allow_html=True)

    # Pagination
    PAGE_SIZE = 50
    total_filtered = len(df)
    total_pages = max(1, (total_filtered + PAGE_SIZE - 1) // PAGE_SIZE)
    if 'pred_page' not in st.session_state:
        st.session_state.pred_page = 1
    current_page = st.session_state.pred_page
    if current_page > total_pages:
        current_page = 1
        st.session_state.pred_page = 1
    start_idx = (current_page - 1) * PAGE_SIZE
    end_idx = min(start_idx + PAGE_SIZE, total_filtered)

    display_df = df.iloc[start_idx:end_idx]
    rows_html = ""
    for _, row in display_df.iterrows():
        cid = row['Container_ID']
        score = row['Risk_Score']
        level = row['Risk_Level']
        origin = row.get('Origin_Country', '')
        dest   = row.get('Destination_Port', '')
        dwell  = row.get('Dwell_Time_Hours', '')
        value  = row.get('Declared_Value', 0)
        expl   = row.get('Explanation', '')

        # level badge
        if level == 'Critical':
            badge = f'<span class="level-badge level-critical">CRITICAL</span>'
        elif level == 'Low Risk':
            badge = f'<span class="level-badge level-low">LOW RISK</span>'
        else:
            badge = f'<span class="level-badge level-clear">CLEAR</span>'

        # score bar
        bar_w = min(score, 100)
        score_html = f"""<div class="score-bar-wrap">
            <span class="score-num" style="color:{RED if score>=50 else YELLOW if score>=20 else TEXT};">{score:.1f}</span>
            <div class="score-track"><div class="score-fill" style="width:{bar_w}%;background:{RED if score>=50 else YELLOW if score>=20 else GREEN};"></div></div>
        </div>"""

        # format value
        if isinstance(value, (int, float)):
            if value >= 1_000_000:
                val_str = f"${value/1_000_000:.1f}M"
            elif value >= 1_000:
                val_str = f"${value/1_000:.0f}K"
            else:
                val_str = f"${value:,.0f}"
        else:
            val_str = str(value)

        dwell_color = f' style="color:{RED};"' if (isinstance(dwell, (int, float)) and dwell > 120) else ''

        dwell_fmt = f"{dwell:.1f}" if isinstance(dwell, (int, float)) else str(dwell)

        rows_html += f"""<tr>
            <td>{cid}</td>
            <td>{score_html}</td>
            <td>{badge}</td>
            <td>{origin}</td>
            <td>{dest}</td>
            <td{dwell_color}>{dwell_fmt}</td>
            <td>{val_str}</td>
            <td style="color:{TEXT2};font-size:0.72rem;">{expl}</td>
        </tr>"""

    st.markdown(f"""
    <div style="border:1px solid {BORDER};border-radius:8px;">
    <table class="pred-table">
        <thead><tr>
            <th>Container ID</th><th>Risk Score</th><th>Level</th>
            <th>Origin</th><th>Destination</th><th>Dwell (H)</th>
            <th>Value</th><th>Explanation</th>
        </tr></thead>
        <tbody>{rows_html}</tbody>
    </table></div>
    """, unsafe_allow_html=True)

    # Pagination controls
    pag_left, pag_mid, pag_right = st.columns([4, 2, 4])
    with pag_left:
        st.markdown(f"<span style='font-size:0.72rem;color:{TEXT2};'>Showing {start_idx+1}&ndash;{end_idx} of {total_filtered:,} rows</span>", unsafe_allow_html=True)
    with pag_right:
        pcols = st.columns([1,1,1,1,1,1,1])
        with pcols[0]:
            if st.button("\u2190", key="pred_prev", disabled=(current_page <= 1)):
                st.session_state.pred_page = max(1, current_page - 1)
                st.rerun()
        with pcols[1]:
            if st.button("1", key="pred_p1", type="primary" if current_page==1 else "secondary"):
                st.session_state.pred_page = 1
                st.rerun()
        with pcols[2]:
            if st.button("2", key="pred_p2", type="primary" if current_page==2 else "secondary"):
                st.session_state.pred_page = 2
                st.rerun()
        with pcols[3]:
            if st.button("3", key="pred_p3", type="primary" if current_page==3 else "secondary"):
                st.session_state.pred_page = 3
                st.rerun()
        with pcols[4]:
            st.markdown(f"<span style='color:{TEXT2};font-size:0.8rem;'>...</span>", unsafe_allow_html=True)
        with pcols[5]:
            if st.button(str(total_pages), key="pred_plast", type="primary" if current_page==total_pages else "secondary"):
                st.session_state.pred_page = total_pages
                st.rerun()
        with pcols[6]:
            if st.button("\u2192", key="pred_next", disabled=(current_page >= total_pages)):
                st.session_state.pred_page = min(total_pages, current_page + 1)
                st.rerun()


# ==============================================================================
#  PAGE 3 - FEATURE ANALYSIS
# ==============================================================================

def page_feature_analysis():
    st.markdown(f"""
    <h2 style="margin-bottom:2px;">Feature Analysis</h2>
    <p style="color:{TEXT2};font-size:0.85rem;margin-top:0;">Engineered features ranked by Random Forest importance</p>
    """, unsafe_allow_html=True)

    fi = get_feature_importances()

    # Feature Importance Rankings
    st.markdown(f'<div class="sc-card"><div class="sc-card-title">FEATURE IMPORTANCE RANKINGS &mdash; RANDOM FOREST</div>', unsafe_allow_html=True)
    max_imp = max(fi.values()) if fi else 1
    rows_html = ""
    for i, (feat, imp) in enumerate(fi.items(), 1):
        pct = imp / max_imp * 100
        rows_html += f"""
        <div class="bar-row" style="margin:10px 0;">
            <span style="width:20px;color:{TEXT2};font-size:0.75rem;flex-shrink:0;">{i}</span>
            <span class="bar-label" style="width:260px;">{feat}</span>
            <div class="bar-track" style="height:10px;"><div class="bar-fill" style="width:{pct}%;background:{BLUE};height:100%;border-radius:5px;"></div></div>
            <span class="bar-val" style="width:60px;">{imp:.2f}%</span>
        </div>"""
    st.markdown(rows_html + "</div>", unsafe_allow_html=True)

    # Bottom row
    c1, c2 = st.columns(2)
    with c1:
        st.markdown(f'<div class="sc-card"><div class="sc-card-title">TOP CRITICAL HS CODES</div>', unsafe_allow_html=True)
        preds = get_active_predictions()
        if preds is not None and 'HS_Code' in preds.columns:
            crit = preds[preds['Risk_Level'] == 'Critical']
            hs_counts = crit['HS_Code'].value_counts().head(5)
            rows_html = ""
            for hs, cnt in hs_counts.items():
                rows_html += f'<div class="cfg-row"><span class="cfg-row-label">HS {hs}</span><span class="cfg-row-val">{cnt}</span></div>'
            st.markdown(rows_html + "</div>", unsafe_allow_html=True)
        else:
            st.markdown("</div>", unsafe_allow_html=True)

    with c2:
        st.markdown(f'<div class="sc-card"><div class="sc-card-title">RISK BY TRADE REGIME</div>', unsafe_allow_html=True)
        if preds is not None and 'Trade_Regime' in preds.columns:
            regime_avg = preds.groupby('Trade_Regime')['Risk_Score'].mean().sort_values(ascending=False)
            rows_html = ""
            for regime, avg in regime_avg.items():
                rows_html += f'<div class="cfg-row"><span class="cfg-row-label">{regime}</span><span class="cfg-row-val" style="color:{RED if avg>10 else YELLOW};">{avg:.2f}</span></div>'
            st.markdown(rows_html + "</div>", unsafe_allow_html=True)
        else:
            st.markdown("</div>", unsafe_allow_html=True)


# ==============================================================================
#  PAGE 4 - MODEL METRICS
# ==============================================================================

def page_model_metrics():
    preds = get_active_predictions()
    total = len(preds) if preds is not None else 54000

    st.markdown(f"""
    <h2 style="margin-bottom:2px;">Model Performance Metrics</h2>
    <p style="color:{TEXT2};font-size:0.85rem;margin-top:0;">Random Forest + Isolation Forest ensemble, {total/1000:.0f}K training records</p>
    """, unsafe_allow_html=True)

    c1, c2 = st.columns(2)

    with c1:
        st.markdown(f"""<div class="sc-card">
            <div class="sc-card-title">CLASSIFICATION PERFORMANCE</div>
            <div class="cfg-row"><span class="cfg-row-label">Overall Accuracy</span><span class="cfg-row-val" style="color:{BLUE};">99.82%</span></div>
            <div class="cfg-row"><span class="cfg-row-label">F1 &mdash; Critical</span><span class="cfg-row-val" style="color:{RED};">96.68%</span></div>
            <div class="cfg-row"><span class="cfg-row-label">F1 &mdash; Low Risk</span><span class="cfg-row-val" style="color:{YELLOW};">99.62%</span></div>
            <div class="cfg-row"><span class="cfg-row-label">Records</span><span class="cfg-row-val">{total:,}</span></div>
            <div class="cfg-row"><span class="cfg-row-label">Test Split</span><span class="cfg-row-val">80 / 20 %</span></div>
            <div class="cfg-row" style="border-bottom:none;"><span class="cfg-row-label">Class Weighting</span><span class="cfg-row-val" style="color:{GREEN};">Balanced</span></div>
        </div>""", unsafe_allow_html=True)

    with c2:
        st.markdown(f"""<div class="sc-card">
            <div class="sc-card-title">ALGORITHM STACK</div>
            <div class="cfg-row"><span class="cfg-row-label">Classifier</span><span class="cfg-row-val" style="font-family:monospace;color:{BLUE};">Random Forest</span></div>
            <div class="cfg-row"><span class="cfg-row-label">n_estimators</span><span class="cfg-row-val" style="font-family:monospace;">150</span></div>
            <div class="cfg-row"><span class="cfg-row-label">max_depth</span><span class="cfg-row-val" style="font-family:monospace;">20</span></div>
            <div class="cfg-row"><span class="cfg-row-label">Anomaly Detector</span><span class="cfg-row-val" style="font-family:monospace;color:{RED};">Isolation Forest</span></div>
            <div class="cfg-row"><span class="cfg-row-label">Contamination</span><span class="cfg-row-val" style="font-family:monospace;">3 %</span></div>
            <div class="cfg-row" style="border-bottom:none;"><span class="cfg-row-label">Score Formula</span><span class="cfg-row-val" style="font-family:monospace;color:{YELLOW};">50% RF + 30% LR + 20% IF</span></div>
        </div>""", unsafe_allow_html=True)

    # Score band distribution
    st.markdown(f'<div class="sc-card"><div class="sc-card-title">SCORE BAND DISTRIBUTION &mdash; {total:,} CONTAINERS</div>', unsafe_allow_html=True)
    if preds is not None:
        bands = [(0,10,GREEN), (10,20,'#2ea043'), (20,40,YELLOW), (40,60,'#e3b341'), (60,80,RED), (80,100,'#da3633')]
        band_labels = ['0-10','10-20','20-40','40-60','60-80','80-100']
        rows_html = ""
        for (lo, hi, color), label in zip(bands, band_labels):
            if hi == 100:
                cnt = int(((preds['Risk_Score'] >= lo) & (preds['Risk_Score'] <= hi)).sum())
            else:
                cnt = int(((preds['Risk_Score'] >= lo) & (preds['Risk_Score'] < hi)).sum())
            pct = cnt / total * 100
            bar_w = pct / 80 * 100  # scale so 80% fills full bar
            rows_html += f"""
            <div class="band-row">
                <div class="band-color" style="background:{color}22;color:{color};">{label}</div>
                <div class="band-count">{cnt:,} containers</div>
                <div class="band-track"><div class="band-fill" style="width:{min(bar_w,100)}%;background:{color};"></div></div>
                <div class="band-pct">{pct:.1f}%</div>
            </div>"""
        st.markdown(rows_html + "</div>", unsafe_allow_html=True)
    else:
        st.markdown("</div>", unsafe_allow_html=True)


# ==============================================================================
#  PAGE 5 - IMPORT CSV
# ==============================================================================

def page_import_csv():
    st.markdown(f"""
    <h2 style="margin-bottom:2px;">Import CSV</h2>
    <p style="color:{TEXT2};font-size:0.85rem;margin-top:0;">Upload a shipment file &mdash; the engine will clean the data, engineer features, run the risk model, and refresh every dashboard section</p>
    """, unsafe_allow_html=True)

    st.markdown(f"""<div class="import-zone">
        <div style="font-size:2.5rem;color:{TEXT2};">&#8679;</div>
        <h3>Drop CSV file here or click to browse</h3>
        <p>Accepts standard HackaMINEd container shipment CSV format</p>
        <p style="font-size:0.7rem;color:{TEXT2};">Required: Container_ID, Declared_Weight, Measured_Weight, Dwell_Time_Hours, Declared_Value, HS_Code, Trade_Regime, Origin_Country, Destination_Port</p>
    </div>""", unsafe_allow_html=True)

    uploaded = st.file_uploader("Choose File", type=['csv'], label_visibility="collapsed")

    if uploaded is not None:
        try:
            df = pd.read_csv(uploaded)
            st.success(f"Loaded {len(df):,} containers from **{uploaded.name}**")

            st.markdown(f"<h4>Data Preview</h4>", unsafe_allow_html=True)
            st.dataframe(df.head(10), use_container_width=True, hide_index=True)

            # Validate required columns
            required = ['Container_ID', 'Declared_Weight', 'Measured_Weight', 'Dwell_Time_Hours', 'Declared_Value']
            missing = [c for c in required if c not in df.columns]
            if missing:
                st.error(f"Missing required columns: {', '.join(missing)}")
                return

            # Auto-run predictions
            with st.spinner("Running risk model on uploaded data..."):
                predictor = get_predictor()
                result = predictor.predict(df)
                # merge original columns
                for col in ['Origin_Country', 'Destination_Port', 'Dwell_Time_Hours',
                            'Declared_Value', 'HS_Code', 'Declared_Weight', 'Measured_Weight']:
                    if col in df.columns:
                        result[col] = df[col].values
                if 'Trade_Regime (Import / Export / Transit)' in df.columns:
                    result['Trade_Regime'] = df['Trade_Regime (Import / Export / Transit)'].values

            st.session_state.uploaded_preds = result
            st.success(f"Predictions generated for {len(result):,} containers! Navigate to Dashboard or Predictions to view results.")

            # Quick summary
            crit = int((result['Risk_Level'] == 'Critical').sum())
            low  = int((result['Risk_Level'] == 'Low Risk').sum())
            clr  = int((result['Risk_Level'] == 'Clear').sum())

            sc1, sc2, sc3, sc4 = st.columns(4)
            with sc1:
                st.metric("Total", f"{len(result):,}")
            with sc2:
                st.metric("Critical", f"{crit:,}")
            with sc3:
                st.metric("Low Risk", f"{low:,}")
            with sc4:
                st.metric("Clear", f"{clr:,}")

            # Download
            csv_bytes = result.to_csv(index=False).encode('utf-8')
            st.download_button("Download Predictions CSV", csv_bytes,
                               "risk_predictions.csv", "text/csv",
                               use_container_width=True)

        except Exception as e:
            st.error(f"Error processing file: {e}")


# ==============================================================================
#  PAGE 6 - CONFIGURATION
# ==============================================================================

def page_configuration():
    st.markdown(f"""
    <h2 style="margin-bottom:2px;">System Configuration</h2>
    <p style="color:{TEXT2};font-size:0.85rem;margin-top:0;">Model parameters and risk threshold settings</p>
    """, unsafe_allow_html=True)

    # Top cards
    c1, c2, c3 = st.columns(3)
    with c1:
        st.markdown(f"""<div class="cfg-card">
            <div class="cfg-label">MODEL TYPE</div>
            <div class="cfg-value">Random Forest</div>
            <div class="cfg-sub">sklearn.ensemble</div>
        </div>""", unsafe_allow_html=True)
    with c2:
        st.markdown(f"""<div class="cfg-card">
            <div class="cfg-label">ANOMALY</div>
            <div class="cfg-value">Isolation Forest</div>
            <div class="cfg-sub">contamination=3%</div>
        </div>""", unsafe_allow_html=True)
    with c3:
        preds = get_active_predictions()
        n = len(preds) if preds is not None else 54000
        st.markdown(f"""<div class="cfg-card" style="border-left:3px solid {GREEN};">
            <div class="cfg-label">DATASET</div>
            <div class="cfg-value">{n:,} records</div>
            <div class="cfg-sub">Trained on historical data</div>
        </div>""", unsafe_allow_html=True)

    c1, c2, c3 = st.columns(3)
    with c1:
        st.markdown(f"""<div class="cfg-card">
            <div class="cfg-label">FEATURES</div>
            <div class="cfg-value">15 engineered</div>
            <div class="cfg-sub">Weight, dwell, HS, value</div>
        </div>""", unsafe_allow_html=True)
    with c2:
        st.markdown(f"""<div class="cfg-card">
            <div class="cfg-label">CLASSES</div>
            <div class="cfg-value">3 levels</div>
            <div class="cfg-sub">Critical / Low Risk / Clear</div>
        </div>""", unsafe_allow_html=True)
    with c3:
        st.markdown(f"""<div class="cfg-card">
            <div class="cfg-label">EXPLAINABILITY</div>
            <div class="cfg-value">Rule-based</div>
            <div class="cfg-sub">SHAP-style reasons</div>
        </div>""", unsafe_allow_html=True)

    # Thresholds & Output
    c1, c2 = st.columns(2)
    with c1:
        st.markdown(f"""<div class="sc-card">
            <div class="sc-card-title">RISK SCORE THRESHOLDS</div>
            <div class="cfg-row"><span class="cfg-row-label">Critical</span><span class="cfg-row-val" style="color:{RED};">&ge; 50</span></div>
            <div class="cfg-row"><span class="cfg-row-label">Low Risk</span><span class="cfg-row-val" style="color:{YELLOW};">20 &ndash; 49</span></div>
            <div class="cfg-row"><span class="cfg-row-label">Clear</span><span class="cfg-row-val" style="color:{GREEN};">&lt; 20</span></div>
            <div class="cfg-row"><span class="cfg-row-label">Weight Discrepancy Flag</span><span class="cfg-row-val">&gt; 20%</span></div>
            <div class="cfg-row"><span class="cfg-row-label">High Dwell</span><span class="cfg-row-val">&gt; 72 h</span></div>
            <div class="cfg-row" style="border-bottom:none;"><span class="cfg-row-label">Very High Dwell</span><span class="cfg-row-val">&gt; 120 h</span></div>
        </div>""", unsafe_allow_html=True)

    with c2:
        st.markdown(f"""<div class="sc-card">
            <div class="sc-card-title">OUTPUT FIELDS</div>
            <div class="cfg-row"><span class="cfg-row-label">Risk_Score</span><span class="cfg-row-val" style="font-family:monospace;">0 &ndash; 100</span></div>
            <div class="cfg-row"><span class="cfg-row-label">Risk_Level</span><span class="cfg-row-val" style="font-family:monospace;color:{RED};">Critical / Low / Clear</span></div>
            <div class="cfg-row"><span class="cfg-row-label">Explanation</span><span class="cfg-row-val" style="font-family:monospace;">Rule-based</span></div>
            <div class="cfg-row"><span class="cfg-row-label">Export</span><span class="cfg-row-val" style="font-family:monospace;">CSV download</span></div>
            <div class="cfg-row" style="border-bottom:none;"><span class="cfg-row-label">Status</span><span style="background:{GREEN}22;color:{GREEN};padding:3px 12px;border-radius:4px;font-size:0.75rem;font-weight:700;">Active</span></div>
        </div>""", unsafe_allow_html=True)


# ==============================================================================
#  MAIN
# ==============================================================================

def main():
    if 'uploaded_preds' not in st.session_state:
        st.session_state.uploaded_preds = None

    # Handle topbar Import CSV button click (must set default before widget renders)
    if st.session_state.get('_goto_import'):
        st.session_state._goto_import = False
        st.session_state.nav = "\u2B06  Import CSV"

    page = render_sidebar()

    # Dashboard: compact KPI strip above topbar
    if page == "Dashboard":
        render_kpi_strip()

    render_topbar()

    if page == "Dashboard":
        page_dashboard()
    elif page == "Predictions":
        page_predictions()
    elif page == "Feature Analysis":
        page_feature_analysis()
    elif page == "Model Metrics":
        page_model_metrics()
    elif page == "Import CSV":
        page_import_csv()
    elif page == "Configuration":
        page_configuration()


if __name__ == "__main__":
    main()
