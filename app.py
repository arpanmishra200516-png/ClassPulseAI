"""
ClassPulse AI - Streamlit Dashboard Application
Multimodal Attendance & Engagement Agent
Polished UI with Plotly Dark Theme Visualizations, Session Comparer,
Interactive Score Explainer, and Downloadable HTML/PDF Reports.
"""

import os
import sys
import time
from pathlib import Path
import tempfile
import json
import logging

# Ensure project root is in sys.path
BASE_DIR = Path(__file__).resolve().parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

# Prevent Matplotlib cache permission errors on Windows
os.environ["MPLCONFIGDIR"] = str(BASE_DIR / "data" / ".mpl_cache")

import streamlit as st
import numpy as np
import pandas as pd
import cv2
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from config import (
    APP_NAME,
    APP_SUBTITLE,
    APP_VERSION,
    VISION_CONFIG,
    AUDIO_CONFIG,
    AGENT_CONFIG,
    RECOMMENDATION_CONFIG,
    SAMPLE_MEDIA_DIR,
    SESSIONS_DIR,
)
from core_pipeline import ClassPulsePipeline
from analytics.reports import SessionReportGenerator, SessionResult, SessionAnalyticsSummary, SCHEMA_VERSION

logger = logging.getLogger("ClassPulseAI.app")

# Streamlit Page Config
st.set_page_config(
    page_title=f"{APP_NAME} — {APP_SUBTITLE}",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Clean Minimalist Light CSS Styling (Stripe & Linear inspired)
st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&family=JetBrains+Mono:wght@400;500;600&display=swap');

    /* 1. Global Canvas & Typography */
    html, body, [data-testid="stAppViewContainer"], .main {
        background-color: #F8FAFC !important;
        color: #0F172A !important;
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif !important;
    }

    [data-testid="stHeader"] {
        background: transparent !important;
    }

    /* 2. Pure White Sidebar (#FFFFFF) with Clean Right Divider */
    [data-testid="stSidebar"] {
        background-color: #FFFFFF !important;
        border-right: 1px solid #E2E8F0 !important;
    }
    [data-testid="stSidebar"] h1, [data-testid="stSidebar"] h2, [data-testid="stSidebar"] h3 {
        color: #0F172A !important;
        letter-spacing: -0.02em;
    }
    [data-testid="stSidebar"] .stRadio label {
        color: #475569 !important;
        font-size: 0.9rem !important;
        transition: color 0.15s ease;
    }
    [data-testid="stSidebar"] .stRadio label:hover {
        color: #2563EB !important;
    }

    /* 3. Header & Status Chips */
    .hero-header {
        display: flex;
        justify-content: space-between;
        align-items: center;
        padding: 1.25rem 1.75rem;
        margin-bottom: 1.25rem;
        background: #FFFFFF;
        border: 1px solid #E2E8F0;
        border-radius: 12px;
        box-shadow: 0 1px 3px 0 rgba(0, 0, 0, 0.05), 0 1px 2px -1px rgba(0, 0, 0, 0.05);
    }
    .hero-title-group {
        display: flex;
        flex-direction: column;
        gap: 0.25rem;
    }
    .hero-title {
        font-size: 2.1rem;
        font-weight: 800;
        letter-spacing: -0.03em;
        color: #0F172A;
        display: flex;
        align-items: center;
        gap: 0.6rem;
        margin: 0;
        line-height: 1.2;
    }
    .hero-badge {
        font-size: 0.72rem;
        font-weight: 600;
        padding: 2px 8px;
        border-radius: 9999px;
        background: #EFF6FF;
        border: 1px solid #BFDBFE;
        color: #2563EB;
        letter-spacing: 0.02em;
    }
    .hero-subtitle {
        font-size: 0.92rem;
        color: #64748B;
        margin: 0;
        font-weight: 450;
    }
    .hero-pills {
        display: flex;
        gap: 0.6rem;
        align-items: center;
        flex-wrap: wrap;
    }
    .status-pill {
        display: inline-flex;
        align-items: center;
        gap: 0.45rem;
        padding: 0.4rem 0.85rem;
        border-radius: 9999px;
        font-size: 0.78rem;
        font-weight: 600;
        letter-spacing: 0.02em;
        transition: all 0.2s cubic-bezier(0.4, 0, 0.2, 1);
    }
    .status-pill:hover {
        transform: translateY(-1px);
    }
    .status-pill-ready {
        background: #ECFDF5;
        border: 1px solid #A7F3D0;
        color: #059669;
    }
    .status-pill-secure {
        background: #EFF6FF;
        border: 1px solid #BFDBFE;
        color: #2563EB;
    }
    .status-pill-mode {
        background: #EEF2FF;
        border: 1px solid #C7D2FE;
        color: #4F46E5;
    }
    .status-dot {
        width: 7px;
        height: 7px;
        border-radius: 50%;
        background-color: #059669;
        box-shadow: 0 0 6px #059669;
        display: inline-block;
    }

    /* 4. Headings & Subheadings */
    .main-header {
        font-size: 2.1rem;
        font-weight: 800;
        letter-spacing: -0.03em;
        color: #0F172A;
        margin-bottom: 0.1rem;
    }
    .main-subtitle {
        font-size: 0.95rem;
        color: #64748B;
        margin-bottom: 1.2rem;
    }
    h1, h2, h3 {
        color: #0F172A !important;
        letter-spacing: -0.02em !important;
    }

    /* 5. Minimalist Light Metric Cards & KPI Strip */
    .kpi-container {
        display: flex;
        gap: 12px;
        flex-wrap: wrap;
        margin-bottom: 0.5rem;
    }
    .kpi-box {
        background-color: #FFFFFF;
        border: 1px solid #E2E8F0;
        border-radius: 10px;
        padding: 14px 18px;
        flex: 1;
        min-width: 140px;
        text-align: center;
        box-shadow: 0 1px 3px 0 rgba(0, 0, 0, 0.05), 0 1px 2px -1px rgba(0, 0, 0, 0.05);
        transition: transform 0.2s cubic-bezier(0.4, 0, 0.2, 1), border-color 0.2s ease, box-shadow 0.2s ease;
    }
    .kpi-box:hover {
        transform: translateY(-2px);
        border-color: #CBD5E1;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.08), 0 2px 4px -1px rgba(0, 0, 0, 0.04);
    }
    .kpi-val {
        font-size: 1.7rem;
        font-weight: 700;
        letter-spacing: -0.02em;
        color: #0F172A;
        margin-top: 2px;
    }
    .kpi-lbl {
        font-size: 0.72rem;
        color: #64748B !important;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        font-weight: 600;
    }

    /* Native st.metric card styling */
    [data-testid="stMetric"] {
        background-color: #FFFFFF !important;
        border: 1px solid #E2E8F0 !important;
        border-radius: 10px !important;
        padding: 14px 18px !important;
        box-shadow: 0 1px 3px 0 rgba(0, 0, 0, 0.05), 0 1px 2px -1px rgba(0, 0, 0, 0.05) !important;
        transition: transform 0.2s cubic-bezier(0.4, 0, 0.2, 1), box-shadow 0.2s ease !important;
    }
    [data-testid="stMetric"]:hover {
        transform: translateY(-2px) !important;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.08) !important;
    }
    [data-testid="stMetricLabel"] {
        font-size: 0.72rem !important;
        letter-spacing: 0.05em !important;
        font-weight: 600 !important;
        color: #64748B !important;
        text-transform: uppercase !important;
    }
    [data-testid="stMetricValue"] {
        color: #0F172A !important;
        font-weight: 700 !important;
        font-size: 1.7rem !important;
    }

    /* 6. Overview Workflow Cards */
    .workflow-card {
        background: #FFFFFF;
        border: 1px solid #E2E8F0;
        border-radius: 12px;
        padding: 20px 22px;
        box-shadow: 0 1px 3px 0 rgba(0, 0, 0, 0.05);
        transition: transform 0.2s cubic-bezier(0.4, 0, 0.2, 1), border-color 0.2s ease, box-shadow 0.2s ease;
        height: 100%;
    }
    .workflow-card:hover {
        transform: translateY(-2px);
        border-color: #CBD5E1;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.08);
    }
    .workflow-icon {
        font-size: 1.6rem;
        margin-bottom: 0.6rem;
    }
    .workflow-title {
        font-size: 1.05rem;
        font-weight: 700;
        color: #0F172A;
        margin-bottom: 0.4rem;
        letter-spacing: -0.01em;
    }
    .workflow-desc {
        color: #64748B;
        font-size: 0.88rem;
        line-height: 1.5;
        margin: 0;
    }

    /* 7. Recommendation and Ethical Cards */
    .rec-card-high {
        background: #FFFFFF;
        border: 1px solid #E2E8F0;
        border-left: 4px solid #EF4444;
        padding: 16px 18px;
        margin-bottom: 12px;
        border-radius: 8px;
        box-shadow: 0 1px 3px 0 rgba(0, 0, 0, 0.05);
        transition: transform 0.2s ease;
        color: #0F172A;
    }
    .rec-card-high:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.08);
    }
    .rec-card-med {
        background: #FFFFFF;
        border: 1px solid #E2E8F0;
        border-left: 4px solid #F59E0B;
        padding: 16px 18px;
        margin-bottom: 12px;
        border-radius: 8px;
        box-shadow: 0 1px 3px 0 rgba(0, 0, 0, 0.05);
        transition: transform 0.2s ease;
        color: #0F172A;
    }
    .rec-card-med:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.08);
    }
    .rec-card-info {
        background: #FFFFFF;
        border: 1px solid #E2E8F0;
        border-left: 4px solid #2563EB;
        padding: 16px 18px;
        margin-bottom: 12px;
        border-radius: 8px;
        box-shadow: 0 1px 3px 0 rgba(0, 0, 0, 0.05);
        transition: transform 0.2s ease;
        color: #0F172A;
    }
    .rec-card-info:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.08);
    }
    .ethical-card {
        background: #F8FAFC;
        border: 1px solid #E2E8F0;
        border-radius: 10px;
        padding: 16px 18px;
        color: #334155;
        font-size: 0.88rem;
        line-height: 1.55;
        box-shadow: 0 1px 3px 0 rgba(0, 0, 0, 0.05);
    }

    /* 8. Stripe & Linear-inspired Interactive Elements */
    .stButton > button {
        border-radius: 8px !important;
        font-weight: 550 !important;
        transition: all 0.2s cubic-bezier(0.4, 0, 0.2, 1) !important;
        border: 1px solid #E2E8F0 !important;
        background: #FFFFFF !important;
        color: #0F172A !important;
        box-shadow: 0 1px 2px 0 rgba(0, 0, 0, 0.05) !important;
    }
    .stButton > button:hover {
        transform: translateY(-1px) !important;
        background: #F8FAFC !important;
        border-color: #CBD5E1 !important;
        color: #0F172A !important;
    }
    .stButton > button[kind="primary"] {
        background: #2563EB !important;
        border: 1px solid #1D4ED8 !important;
        color: #FFFFFF !important;
        box-shadow: 0 1px 3px 0 rgba(37, 99, 235, 0.3) !important;
    }
    .stButton > button[kind="primary"]:hover {
        background: #1D4ED8 !important;
        box-shadow: 0 4px 6px -1px rgba(37, 99, 235, 0.4) !important;
    }

    /* 9. Streamlit Inputs, Selects & Expander styling */
    .stTextInput input, .stSelectbox [data-baseweb="select"], .stTextArea textarea {
        border-radius: 8px !important;
        background-color: #FFFFFF !important;
        border: 1px solid #E2E8F0 !important;
        color: #0F172A !important;
    }
    [data-testid="stExpander"] {
        background: #FFFFFF !important;
        border: 1px solid #E2E8F0 !important;
        border-radius: 10px !important;
        box-shadow: 0 1px 3px 0 rgba(0, 0, 0, 0.05) !important;
        margin-bottom: 12px;
    }
    [data-testid="stExpander"] summary {
        color: #0F172A !important;
        font-weight: 600 !important;
    }

    /* 10. Prevent slider value labels overlapping slider track (BUG 9) */
    .stSlider {
        padding-top: 4px;
        margin-bottom: 14px;
    }
    .stSlider [data-baseweb="slider"] {
        margin-top: 18px !important;
        margin-bottom: 8px !important;
    }
    .stSlider [data-testid="stTickBar"] {
        margin-top: 8px !important;
    }
    .stSlider div[data-testid="stMarkdownContainer"] {
        margin-bottom: 4px;
    }

    /* 11. Custom Minimalist Scrollbar */
    ::-webkit-scrollbar {
        width: 6px;
        height: 6px;
    }
    ::-webkit-scrollbar-track {
        background: #F8FAFC;
    }
    ::-webkit-scrollbar-thumb {
        background: #CBD5E1;
        border-radius: 4px;
    }
    ::-webkit-scrollbar-thumb:hover {
        background: #94A3B8;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


def load_most_recent_session() -> Optional[SessionResult]:
    """Loads the most recent saved session from SESSIONS_DIR as active session."""
    if not SESSIONS_DIR.exists():
        return None
    session_files = sorted(
        [f for f in SESSIONS_DIR.glob("*.json") if not f.stem.startswith("test_")],
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    for sf in session_files:
        try:
            data = SessionReportGenerator.load_session_from_disk(sf.stem)
            if data:
                return SessionResult.from_dict(data)
        except Exception as e:
            logger.warning("Could not load session from %s: %s", sf, e)
    return None


def init_state():
    """Initializes session state with default configuration values and active session."""
    if "min_presence_ratio" not in st.session_state:
        st.session_state.min_presence_ratio = VISION_CONFIG["default_min_presence_ratio"]
    if "use_absolute_override" not in st.session_state:
        st.session_state.use_absolute_override = VISION_CONFIG["use_absolute_override"]
    if "min_presence_seconds" not in st.session_state:
        st.session_state.min_presence_seconds = VISION_CONFIG["default_min_presence_seconds"]
    if "gap_tolerance" not in st.session_state:
        st.session_state.gap_tolerance = VISION_CONFIG["gap_tolerance_seconds"]
    if "expected_participants" not in st.session_state:
        st.session_state.expected_participants = 2

    # Weights
    if "weight_presence" not in st.session_state:
        st.session_state.weight_presence = AGENT_CONFIG["weights"]["visual_presence"]
    if "weight_movement" not in st.session_state:
        st.session_state.weight_movement = AGENT_CONFIG["weights"]["movement_dynamics"]
    if "weight_audio" not in st.session_state:
        st.session_state.weight_audio = AGENT_CONFIG["weights"]["audio_activity"]

    # Recommendation thresholds
    if "low_eng_threshold" not in st.session_state:
        st.session_state.low_eng_threshold = RECOMMENDATION_CONFIG["low_engagement_threshold"]
    if "sustained_low_duration" not in st.session_state:
        st.session_state.sustained_low_duration = RECOMMENDATION_CONFIG["sustained_low_engagement_duration"]

    default_session_name = f"{time.strftime('%Y-%m-%d_%H%M')}_classroom_session"
    if "session_label" not in st.session_state:
        st.session_state.session_label = default_session_name

    if "pipeline" not in st.session_state:
        st.session_state.pipeline = ClassPulsePipeline(
            session_id=st.session_state.session_label,
            min_presence_ratio=st.session_state.min_presence_ratio,
            min_presence_seconds=st.session_state.min_presence_seconds,
            use_absolute_override=st.session_state.use_absolute_override,
            gap_tolerance_seconds=st.session_state.gap_tolerance,
            expected_participants=st.session_state.expected_participants,
        )

    # ONE SOURCE OF TRUTH (BUG 2)
    if "active_session" not in st.session_state or st.session_state.active_session is None:
        st.session_state.active_session = load_most_recent_session()

    if "is_running" not in st.session_state:
        st.session_state.is_running = False


init_state()


def reset_defaults():
    """Resets all configuration values to project defaults."""
    st.session_state.min_presence_ratio = VISION_CONFIG["default_min_presence_ratio"]
    st.session_state.use_absolute_override = VISION_CONFIG["use_absolute_override"]
    st.session_state.min_presence_seconds = VISION_CONFIG["default_min_presence_seconds"]
    st.session_state.gap_tolerance = VISION_CONFIG["gap_tolerance_seconds"]
    st.session_state.expected_participants = 2
    st.session_state.weight_presence = AGENT_CONFIG["weights"]["visual_presence"]
    st.session_state.weight_movement = AGENT_CONFIG["weights"]["movement_dynamics"]
    st.session_state.weight_audio = AGENT_CONFIG["weights"]["audio_activity"]
    st.session_state.low_eng_threshold = RECOMMENDATION_CONFIG["low_engagement_threshold"]
    st.session_state.sustained_low_duration = RECOMMENDATION_CONFIG["sustained_low_engagement_duration"]


# ==========================================
# SIDEBAR CONTROLS & THRESHOLDS
# ==========================================
with st.sidebar:
    st.title(f"🎓 {APP_NAME}")
    st.caption(f"v{APP_VERSION} | Schema v{SCHEMA_VERSION} | 100% Offline")

    # Currently loaded session label (BUG 2)
    active_s = st.session_state.get("active_session")
    active_s_id = active_s.session_id if active_s else "None (No active session)"
    st.caption(f"**Currently loaded session:** `{active_s_id}`")
    if active_s and active_s.audio.get("status") == "NO_AUDIO_TRACK":
        st.caption("👁️ *Vision-only mode (No audio track)*")

    navigation = st.radio(
        "Navigation",
        [
            "📌 Overview & Mission",
            "🚀 Live & Video Processing",
            "📋 Attendance Verification",
            "🎙️ Acoustic Activity (VAD)",
            "📈 Multimodal Engagement",
            "💡 Teaching Suggestions",
            "📊 Session Archive & Compare",
            "📥 Reports & Downloads",
        ],
    )

    st.markdown("---")
    st.subheader("⚙️ Parameter Controls")

    # Required Presence slider (BUG 9: formatted as %d%%, stored as float fraction)
    current_pct = int(round(float(st.session_state.min_presence_ratio) * 100))
    slider_pct = st.slider(
        "Required Presence (%)",
        min_value=20,
        max_value=95,
        value=current_pct,
        step=5,
        format="%d%%",
        help="Percentage of session duration a participant must be observed to qualify as present (default: 60%).",
    )
    st.session_state.min_presence_ratio = slider_pct / 100.0

    st.session_state.use_absolute_override = st.checkbox(
        "Enable Absolute Seconds Override",
        value=st.session_state.use_absolute_override,
    )
    if st.session_state.use_absolute_override:
        st.session_state.min_presence_seconds = st.slider(
            "Minimum Required Presence (Seconds)",
            min_value=2.0,
            max_value=60.0,
            value=float(st.session_state.min_presence_seconds or 10.0),
            step=1.0,
        )
    else:
        st.session_state.min_presence_seconds = None

    st.session_state.gap_tolerance = st.slider(
        "Detection Gap Tolerance (s)",
        min_value=0.5,
        max_value=5.0,
        value=float(st.session_state.gap_tolerance),
        step=0.5,
        help="Tolerates temporary head-turns or brief occlusions without resetting presence accumulation.",
    )

    st.session_state.expected_participants = st.slider(
        "Expected Cohort Size",
        min_value=1,
        max_value=10,
        value=int(st.session_state.expected_participants),
        step=1,
        help="Used to normalize visual presence scores without penalizing individual movement.",
    )

    with st.expander("⚖️ Engagement Weights"):
        st.session_state.weight_presence = st.slider("Presence Weight", 0.1, 0.7, float(st.session_state.weight_presence), 0.05)
        st.session_state.weight_movement = st.slider("Movement Weight", 0.1, 0.5, float(st.session_state.weight_movement), 0.05)
        st.session_state.weight_audio = st.slider("Audio Weight", 0.1, 0.7, float(st.session_state.weight_audio), 0.05)

    if st.button("🔄 Reset to Defaults", use_container_width=True):
        reset_defaults()
        st.rerun()

    # Sync pipeline configuration
    pipeline = st.session_state.pipeline
    pipeline.min_presence_ratio = st.session_state.min_presence_ratio
    pipeline.min_presence_seconds = st.session_state.min_presence_seconds
    pipeline.use_absolute_override = st.session_state.use_absolute_override
    pipeline.gap_tolerance_seconds = st.session_state.gap_tolerance
    pipeline.expected_participants = st.session_state.expected_participants
    pipeline.attendance_mgr.min_presence_ratio = st.session_state.min_presence_ratio
    pipeline.attendance_mgr.min_presence_seconds = st.session_state.min_presence_seconds
    pipeline.attendance_mgr.use_absolute_override = st.session_state.use_absolute_override
    pipeline.attendance_mgr.gap_tolerance_seconds = st.session_state.gap_tolerance
    pipeline.engagement_estimator.expected_participants = st.session_state.expected_participants
    pipeline.engagement_estimator.weights = {
        "visual_presence": st.session_state.weight_presence,
        "movement_dynamics": st.session_state.weight_movement,
        "audio_activity": st.session_state.weight_audio,
    }


# ==========================================
# REUSABLE TOP KPI ROW & HERO HEADER
# ==========================================
def render_hero_header(active_session: Optional[SessionResult] = None):
    """Renders the sleek, modern Clean Minimalist Light hero header with live status badges."""
    modality_pill = ""
    if active_session:
        has_aud = active_session.audio.get("has_audio", False)
        mode_text = "⚡ Bimodal (Vision + Audio)" if has_aud else "👁️ Vision-Only Mode"
        modality_pill = f'<div class="status-pill status-pill-mode">{mode_text}</div>'

    html = f"""
    <div class="hero-header">
        <div class="hero-title-group">
            <h1 class="hero-title">
                🎓 {APP_NAME} <span class="hero-badge">v{APP_VERSION}</span>
            </h1>
            <p class="hero-subtitle">{APP_SUBTITLE}</p>
        </div>
        <div class="hero-pills">
            <div class="status-pill status-pill-ready">
                <span class="status-dot"></span> Engine Ready
            </div>
            <div class="status-pill status-pill-secure">
                🔒 100% On-Device / Local
            </div>
            {modality_pill}
        </div>
    </div>
    """
    st.markdown(html, unsafe_allow_html=True)


def render_kpi_row(active_session: Optional[SessionResult]):
    """
    Renders high-contrast top KPI cards computed strictly from active_session.
    If a modality is missing, shows 'N/A' (not 0 and not stale) with a tooltip.
    Includes caption: 'Showing: <session name>'.
    """
    if active_session is None:
        p_count = "0"
        pres_count = "0"
        att_rate = "0.0%"
        mean_eng = "N/A"
        speech_pct = "N/A"
        noise_fl = "N/A"
        sess_name = "None"
        eng_tooltip = "No active session loaded"
        speech_tooltip = "No active session loaded"
        noise_tooltip = "No active session loaded"
    else:
        sess_name = active_session.session_id
        att_summary = active_session.attendance_summary or {}
        p_count = str(att_summary.get("total_temporary_tracking_ids", len(active_session.attendance.get("records", []))))
        pres_count = str(att_summary.get("present_count", sum(1 for r in active_session.attendance.get("records", []) if r.get("attendance_status") == "PRESENT")))
        att_rate = f"{att_summary.get('attendance_rate_percentage', 0.0):.1f}%"

        # Mean Engagement (BUG 3: N/A if missing timeline)
        eng = active_session.engagement or {}
        eng_summary = active_session.engagement_summary or {}
        mean_val = eng.get("mean_score")
        if mean_val is None:
            mean_val = eng_summary.get("mean_score")

        if mean_val is not None and eng.get("has_timeline"):
            mean_eng = f"{float(mean_val):.1f}"
            eng_tooltip = f"Mean multimodal engagement: {mean_eng} / 100"
        else:
            mean_eng = "N/A"
            eng_tooltip = "No engagement timeline recorded"

        # Audio KPIs (BUG 3: N/A if missing audio track)
        aud = active_session.audio or {}
        has_audio = aud.get("has_audio", False)
        aud_status = aud.get("status", "UNAVAILABLE")

        if has_audio and aud.get("speech_percentage") is not None:
            speech_pct = f"{float(aud['speech_percentage']):.1f}%"
            speech_tooltip = f"Vocal speech activity ratio: {speech_pct}"
        else:
            speech_pct = "N/A"
            speech_tooltip = "No audio track" if aud_status == "NO_AUDIO_TRACK" else "Audio track unavailable"

        if has_audio and aud.get("ambient_noise_floor") is not None:
            noise_fl = f"{float(aud['ambient_noise_floor']):.4f}"
            noise_tooltip = f"Ambient noise floor RMS: {noise_fl}"
        else:
            noise_fl = "N/A"
            noise_tooltip = "No audio track" if aud_status == "NO_AUDIO_TRACK" else "Audio track unavailable"

    html = f"""
    <div class="kpi-container">
        <div class="kpi-box" title="Total consolidated learners tracked">
            <div class="kpi-lbl">Participants</div>
            <div class="kpi-val">{p_count}</div>
        </div>
        <div class="kpi-box" title="Confirmed present participants">
            <div class="kpi-lbl">Confirmed Present</div>
            <div class="kpi-val" style="color: #059669;">{pres_count}</div>
        </div>
        <div class="kpi-box" title="Percentage of cohort confirmed present">
            <div class="kpi-lbl">Attendance %</div>
            <div class="kpi-val" style="color: #2563EB;">{att_rate}</div>
        </div>
        <div class="kpi-box" title="{eng_tooltip}">
            <div class="kpi-lbl">Mean Engagement</div>
            <div class="kpi-val" style="color: #4F46E5;">{mean_eng}</div>
        </div>
        <div class="kpi-box" title="{speech_tooltip}">
            <div class="kpi-lbl">Speech Ratio</div>
            <div class="kpi-val" style="color: #D97706;">{speech_pct}</div>
        </div>
        <div class="kpi-box" title="{noise_tooltip}">
            <div class="kpi-lbl">Noise Floor (RMS)</div>
            <div class="kpi-val" style="color: #64748B;">{noise_fl}</div>
        </div>
    </div>
    """
    st.markdown(html, unsafe_allow_html=True)
    st.caption(f"Showing: {sess_name}")
    return {
        "participants": p_count,
        "confirmed_present": pres_count,
        "attendance_rate": att_rate,
        "mean_engagement": mean_eng,
        "speech_ratio": speech_pct,
        "noise_floor": noise_fl,
        "session_name": sess_name,
    }


def render_reports_view(active_sess: Optional[SessionResult] = None):
    """Renders the Reports and Exports page view (BUG 1)."""
    st.markdown("### 📥 Reports & Downloadable Exports")

    # ONE SOURCE OF TRUTH (BUG 2)
    if active_sess is None:
        active_sess = st.session_state.get("active_session")
    render_kpi_row(active_sess)

    if active_sess is None:
        st.info("ℹ️ No active session loaded. Please process a video in 'Live & Video Processing' or load a saved session from 'Session Archive & Compare'.")
        return

    st.subheader(f"Session Summary: {active_sess.session_id}")
    st.write(f"Recorded: **{active_sess.created_at}** | Duration: **{active_sess.session_duration:.1f}s** | Schema: **{SCHEMA_VERSION}**")

    st.markdown("---")
    st.subheader("Download Export Packages")

    c1, c2, c3, c4 = st.columns(4)

    # 1. Printable HTML/PDF Report
    summary_obj = active_sess.summary
    if summary_obj is None:
        # Reconstruct summary for export
        summary_obj = SessionAnalyticsSummary(
            session_id=active_sess.session_id,
            created_at=active_sess.created_at,
            session_duration_seconds=active_sess.session_duration,
            total_tracking_ids=active_sess.attendance_summary.get("total_temporary_tracking_ids", len(active_sess.attendance.get("records", []))),
            present_count=active_sess.attendance_summary.get("present_count", 0),
            absent_count=active_sess.attendance_summary.get("absent_count", 0),
            attendance_rate_percentage=active_sess.attendance_summary.get("attendance_rate_percentage", 0.0),
            attendance_records=active_sess.attendance.get("records", []),
            raw_tracks_count=active_sess.attendance.get("raw_tracks_count", 0),
            consolidated_count=active_sess.attendance.get("consolidated_count", 0),
            discarded_short_tracks_count=active_sess.attendance.get("discarded_count", 0),
            mean_engagement_score=active_sess.engagement.get("mean_score", 0.0) or 0.0,
            median_engagement_score=active_sess.engagement.get("median_score", 0.0) or 0.0,
            min_engagement_score=active_sess.engagement.get("min_score", 0.0) or 0.0,
            max_engagement_score=active_sess.engagement.get("max_score", 0.0) or 0.0,
            std_engagement_score=active_sess.engagement.get("std_score", 0.0) or 0.0,
            high_engagement_percentage=0.0,
            moderate_engagement_percentage=0.0,
            low_engagement_percentage=0.0,
            average_rms_energy=active_sess.audio.get("average_rms"),
            ambient_noise_floor=active_sess.audio.get("ambient_noise_floor"),
            speech_percentage=active_sess.audio.get("speech_percentage"),
            audio_status=active_sess.audio.get("status", "UNAVAILABLE"),
            has_audio_track=active_sess.audio.get("has_audio", False),
            recommendations=active_sess.suggestions,
        )

    html_report = SessionReportGenerator.generate_html_report(summary_obj)
    c1.download_button(
        "📄 Download HTML/PDF Report",
        data=html_report,
        file_name=f"{active_sess.session_id}_report.html",
        mime="text/html",
    )

    # 2. Attendance CSV
    att_records = active_sess.attendance.get("records", [])
    df_att = pd.DataFrame(att_records) if att_records else pd.DataFrame(columns=["participant_id", "attendance_status", "accumulated_presence_seconds"])
    c2.download_button(
        "📋 Attendance CSV",
        data=df_att.to_csv(index=False),
        file_name=f"{active_sess.session_id}_attendance.csv",
        mime="text/csv",
    )

    # 3. Engagement CSV
    eng_timeline = active_sess.engagement.get("timeline", [])
    df_eng = pd.DataFrame(eng_timeline) if eng_timeline else pd.DataFrame(columns=["bin_start", "bin_end", "score", "level"])
    c3.download_button(
        "📈 Engagement CSV",
        data=df_eng.to_csv(index=False),
        file_name=f"{active_sess.session_id}_engagement.csv",
        mime="text/csv",
    )

    # 4. Full JSON
    json_str = active_sess.to_json(indent=2)
    c4.download_button(
        f"📦 Full JSON (Schema v{SCHEMA_VERSION})",
        data=json_str,
        file_name=f"{active_sess.session_id}_report.json",
        mime="application/json",
    )

    with st.expander("🔍 View Structured JSON Data"):
        st.json(active_sess.to_dict())


render_reports_page = render_reports_view


# ==========================================
# PAGE ROUTING WRAPPER (BUG 1: Try/Except per page)
# ==========================================
try:
    # ==========================================
    # 1. OVERVIEW & MISSION
    # ==========================================
    if navigation == "📌 Overview & Mission":
        render_hero_header(st.session_state.active_session)
        render_kpi_row(st.session_state.active_session)

        # 3-Card Streamlined Workflow Quick-Launch Grid
        wf1, wf2, wf3 = st.columns(3)
        with wf1:
            st.markdown(
                """
                <div class="workflow-card">
                    <div class="workflow-icon">🎥</div>
                    <div class="workflow-title">1. Capture & Video</div>
                    <p class="workflow-desc">
                        Analyze high-definition recorded classroom videos (MP4/MKV) or connect live webcam streams with automated frame skipping and audio extraction.
                    </p>
                </div>
                """,
                unsafe_allow_html=True,
            )
        with wf2:
            st.markdown(
                """
                <div class="workflow-card">
                    <div class="workflow-icon">📋</div>
                    <div class="workflow-title">2. Attendance Verification</div>
                    <p class="workflow-desc">
                        Track accumulated student presence against relative duration thresholds. Consolidates fragmented tracks and weeds out transient false positives.
                    </p>
                </div>
                """,
                unsafe_allow_html=True,
            )
        with wf3:
            st.markdown(
                """
                <div class="workflow-card">
                    <div class="workflow-icon">💡</div>
                    <div class="workflow-title">3. Multimodal Analytics</div>
                    <p class="workflow-desc">
                        Fuse 1.0s video and acoustic observations into explainable engagement indicators and actionable pedagogical suggestions for reflective teaching.
                    </p>
                </div>
                """,
                unsafe_allow_html=True,
            )

        st.markdown("<div style='height: 10px;'></div>", unsafe_allow_html=True)

        # Unobtrusive Expandable Drawer for Detailed Mission, Architecture & Safeguards
        with st.expander("🛡️ System Architecture, Educational Mission & Privacy Safeguards", expanded=False):
            col1, col2 = st.columns([3, 2])
            with col1:
                st.markdown("#### 🎯 Educational Mission")
                st.write(
                    """
                    **ClassPulse AI** is an offline, privacy-first classroom assistant. It bridges the gap 
                    between disruptive manual roll-calls and intrusive biometric surveillance.
                    
                    By uniting **computer vision face tracking** and **acoustic speech processing**, ClassPulse AI:
                    - Accurately tracks accumulated classroom presence over the entire session duration.
                    - Prevents false-positives from brief camera passes or single-frame photos.
                    - Estimates an explainable, rule-based multimodal engagement score (0–100).
                    - Detects classroom noise and lecture monologues to surface context-aware teaching suggestions.
                    """
                )

                st.markdown("#### ⚙️ Modular Subsystems Architecture")
                st.info(
                    """
                    - **Vision Engine**: MediaPipe BlazeFace (Primary) with OpenCV Haar cascade fallback and CLAHE low-light boost.
                    - **Face Tracker**: Hybrid Centroid Euclidean & IoU tracker with 12s re-entry persistence and track consolidation.
                    - **Attendance**: Relative percentage threshold (default 60%) with explicit presence/absence evaluations.
                    - **Audio Module**: Resampled 16 kHz acoustic processor, rolling 10th-percentile noise floor, and flicker-free VAD.
                    - **Multimodal Fusion**: Unified 1.0-second time bins with explainable engagement reasoning.
                    - **Pedagogical Engine**: Contextual recommendations with exact metric values and caveats.
                    """
                )

            with col2:
                st.markdown("#### 🔒 Privacy & Ethical Safeguards")
                st.markdown(
                    """
                    <div class="ethical-card">
                        <b>100% Local & Ephemeral:</b> All video frames and audio samples are processed in memory on this machine. No cloud uploads.<br><br>
                        <b>Anonymous Tracking:</b> Learners are assigned session-only labels (<code>Participant_01</code>, <code>Participant_02</code>). No facial recognition or biometric templates are stored.<br><br>
                        <b>Heuristic Indicators:</b> Engagement scores represent observed physical movement and audio dynamics. They do <u>not</u> assess student comprehension, cognitive attention, or academic competence.
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

    # ==========================================
    # 2. LIVE & VIDEO PROCESSING VIEW
    # ==========================================
    elif navigation == "🚀 Live & Video Processing":
        st.markdown("### 🚀 Real-World Video & Live Processing")

        render_kpi_row(st.session_state.active_session)

        c_left, c_right = st.columns([1, 1])

        with c_left:
            session_id_input = st.text_input(
                "Session Label",
                value=st.session_state.session_label,
                help="Unique session name for archival and export isolation (Format: YYYY-MM-DD_HHMM_<label>).",
            )
            st.session_state.session_label = session_id_input
            st.session_state.pipeline.session_id = session_id_input

            input_source = st.radio(
                "Primary Input Source",
                [
                    "📁 Upload Classroom Video (MP4/AVI/MKV)",
                    "📹 Live Webcam Feed",
                    "🧪 Built-in Synthetic Demo (Offline Evaluation)",
                    "🎙️ Standalone Audio File (WAV/MP3)",
                ],
            )

        # -------------------------------------------------------------
        # BUG 5: Dynamic threshold calculation on REAL media duration
        # -------------------------------------------------------------
        with c_right:
            st.markdown("#### Input Details & Threshold Validation")

            if input_source == "📹 Live Webcam Feed":
                st.info(
                    f"Threshold configured: {st.session_state.min_presence_ratio:.0%} of session duration "
                    f"(computed after the session ends). Gap tolerance: {st.session_state.gap_tolerance}s."
                )
            elif input_source == "🧪 Built-in Synthetic Demo (Offline Evaluation)":
                sim_vid = SAMPLE_MEDIA_DIR / "sample_classroom_video.mp4"
                demo_duration = 8.0
                req_sec = st.session_state.pipeline.attendance_mgr.calculate_required_seconds(demo_duration)
                st.success(
                    f"Threshold configured: {st.session_state.min_presence_ratio:.0%} of {demo_duration:.1f}s = {req_sec:.1f}s. "
                    f"Gap tolerance: {st.session_state.gap_tolerance}s."
                )
            elif input_source == "🎙️ Standalone Audio File (WAV/MP3)":
                st.info(f"Threshold configured: {st.session_state.min_presence_ratio:.0%} of audio duration. Select an audio file below.")
            else:
                # Video upload option
                st.info(
                    f"Threshold configured: {st.session_state.min_presence_ratio:.0%} of session duration. "
                    f"Select a video file below to compute exact required seconds."
                )

        st.markdown("---")

        # 1. Upload Video
        if input_source == "📁 Upload Classroom Video (MP4/AVI/MKV)":
            st.subheader("Upload & Process Classroom Video")
            uploaded_video = st.file_uploader("Select video file", type=["mp4", "avi", "mkv", "mov"])
            frame_skip = st.slider("Frame Skip (Speed vs. Granularity)", min_value=1, max_value=6, value=2)

            if uploaded_video is not None:
                tfile = tempfile.NamedTemporaryFile(delete=False, suffix=Path(uploaded_video.name).suffix)
                tfile.write(uploaded_video.read())
                tfile.flush()

                # Read actual duration from uploaded video BEFORE processing (BUG 5)
                probe_cap = cv2.VideoCapture(tfile.name)
                total_frames = int(probe_cap.get(cv2.CAP_PROP_FRAME_COUNT))
                fps_val = probe_cap.get(cv2.CAP_PROP_FPS)
                probe_cap.release()
                if fps_val <= 0 or np.isnan(fps_val):
                    fps_val = 30.0
                real_duration = (total_frames / fps_val) if total_frames > 0 else 12.0
                req_seconds_calc = st.session_state.pipeline.attendance_mgr.calculate_required_seconds(real_duration)

                st.success(
                    f"🎥 Video verified: **{uploaded_video.name}** ({real_duration:.1f}s, {total_frames} frames @ {fps_val:.1f} fps). "
                    f"Required presence: **{st.session_state.min_presence_ratio:.0%} of {real_duration:.1f}s = {req_seconds_calc:.1f}s**."
                )

                if st.button("🚀 Process Video in Streaming Fashion", type="primary"):
                    p_bar = st.progress(0.0)
                    status_box = st.empty()

                    def on_progress(pct, msg):
                        p_bar.progress(pct)
                        status_box.text(msg)

                    st.session_state.pipeline.reset(new_session_id=session_id_input)
                    success = st.session_state.pipeline.process_video_file(
                        tfile.name,
                        frame_skip=frame_skip,
                        progress_callback=on_progress,
                    )
                    if success:
                        summary = st.session_state.pipeline.finalize_session()
                        SessionReportGenerator.save_session_to_disk(summary)
                        # ONE SOURCE OF TRUTH (BUG 2)
                        st.session_state.active_session = SessionResult.from_summary(summary)
                        p_bar.progress(1.0)
                        status_box.text("Complete!")

                        # Notice on Audio status (BUG 6)
                        if not st.session_state.active_session.audio.get("has_audio"):
                            if st.session_state.active_session.audio.get("status") == "NO_AUDIO_TRACK":
                                st.info("ℹ️ This video has no audio track — audio metrics skipped. (Vision-only mode)")
                            elif st.session_state.active_session.audio.get("status") == "EXTRACTION_FAILED":
                                st.error(f"⚠️ Audio extraction error: {st.session_state.active_session.audio.get('extraction_error')}")
                                st.info("💡 Windows Setup: Install FFmpeg via `winget install Gyan.FFmpeg` or `pip install imageio-ffmpeg`.")

                        st.success(f"Video '{uploaded_video.name}' analyzed successfully! Explore the tabs.")
                        st.rerun()
                    else:
                        st.error("Failed to read video file. Please check video format.")

        # 2. Live Webcam Feed
        elif input_source == "📹 Live Webcam Feed":
            st.subheader("Live Webcam Capture & Tracking")
            cam_col1, cam_col2 = st.columns([1, 1])
            start_btn = cam_col1.button("▶️ Start Camera Analysis", type="primary")
            stop_btn = cam_col2.button("⏹️ Stop & Finalize")

            if start_btn:
                st.session_state.is_running = True
                st.session_state.pipeline.reset(new_session_id=session_id_input)

            if stop_btn:
                st.session_state.is_running = False
                summary = st.session_state.pipeline.finalize_session()
                SessionReportGenerator.save_session_to_disk(summary)
                st.session_state.active_session = SessionResult.from_summary(summary)
                st.success("Webcam session completed and archived!")
                st.rerun()

            if st.session_state.is_running:
                frame_view = st.empty()
                info_view = st.empty()

                cap = cv2.VideoCapture(0)
                if not cap.isOpened():
                    st.error("Camera device 0 not accessible. Verify device permissions.")
                    st.session_state.is_running = False
                else:
                    t0 = time.time()
                    frames_seen = 0
                    while st.session_state.is_running and frames_seen < 120:
                        ret, frame = cap.read()
                        if not ret:
                            break
                        elapsed = time.time() - t0
                        annotated, tracks = st.session_state.pipeline.process_live_frame(frame, elapsed, frame_interval=0.06)
                        rgb = cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)
                        frame_view.image(rgb, channels="RGB", use_container_width=True)
                        info_view.caption(f"Elapsed: {elapsed:.1f}s | Active Tracks: {len(tracks)} | Frame: {frames_seen}")
                        frames_seen += 1
                        time.sleep(0.04)

                    cap.release()
                    if frames_seen >= 120:
                        summary = st.session_state.pipeline.finalize_session()
                        SessionReportGenerator.save_session_to_disk(summary)
                        st.session_state.active_session = SessionResult.from_summary(summary)
                        st.info("Reached live demo capture limit. Session finalized!")
                        st.rerun()

        # 3. Built-in Synthetic Demo
        elif input_source == "🧪 Built-in Synthetic Demo (Offline Evaluation)":
            st.subheader("Built-in Offline Synthetic Simulation")
            st.write("Demonstrates face tracking, attendance thresholding, speech VAD, and engagement scoring with 2 participants.")
            if st.button("🚀 Run Synthetic Session", type="primary"):
                sim_vid = SAMPLE_MEDIA_DIR / "sample_classroom_video.mp4"
                sim_aud = SAMPLE_MEDIA_DIR / "sample_classroom_audio.wav"
                if not sim_vid.exists() or not sim_aud.exists():
                    st.error("Sample media files missing. Run create_sample_media.py.")
                else:
                    p_bar = st.progress(0.0)
                    status_box = st.empty()

                    def on_progress(pct, msg):
                        p_bar.progress(pct)
                        status_box.text(msg)

                    st.session_state.pipeline.reset(new_session_id=session_id_input)
                    st.session_state.pipeline.process_video_file(str(sim_vid), progress_callback=on_progress)
                    st.session_state.pipeline.process_audio_file(str(sim_aud))
                    summary = st.session_state.pipeline.finalize_session()
                    SessionReportGenerator.save_session_to_disk(summary)
                    # ONE SOURCE OF TRUTH (BUG 2)
                    st.session_state.active_session = SessionResult.from_summary(summary)
                    p_bar.progress(1.0)
                    status_box.text("Complete!")
                    st.success("Synthetic demo processed successfully! Explore the tabs above.")
                    st.rerun()

        # 4. Standalone Audio File
        elif input_source == "🎙️ Standalone Audio File (WAV/MP3)":
            st.subheader("Standalone Audio Analysis")
            uploaded_audio = st.file_uploader("Select audio file", type=["wav", "mp3", "ogg"])
            if uploaded_audio is not None:
                tfile = tempfile.NamedTemporaryFile(delete=False, suffix=Path(uploaded_audio.name).suffix)
                tfile.write(uploaded_audio.read())
                tfile.flush()

                if st.button("🚀 Process Audio Only", type="primary"):
                    st.session_state.pipeline.reset(new_session_id=session_id_input)
                    st.session_state.pipeline.process_audio_file(tfile.name)
                    summary = st.session_state.pipeline.finalize_session()
                    SessionReportGenerator.save_session_to_disk(summary)
                    st.session_state.active_session = SessionResult.from_summary(summary)
                    st.success(f"Audio '{uploaded_audio.name}' analyzed in AUDIO_ONLY mode!")
                    st.rerun()

    # ==========================================
    # 3. ATTENDANCE VERIFICATION VIEW
    # ==========================================
    elif navigation == "📋 Attendance Verification":
        st.markdown("### 📋 Presence-Based Attendance Verification")

        # ONE SOURCE OF TRUTH (BUG 2)
        active_sess = st.session_state.get("active_session")
        render_kpi_row(active_sess)

        if active_sess is None or not active_sess.attendance.get("records"):
            st.info("No active session data. Process a video or run the simulation in 'Live & Video Processing'.")
        else:
            att_records = active_sess.attendance["records"]
            df = pd.DataFrame(att_records)

            raw_cnt = active_sess.attendance.get("raw_tracks_count", len(df))
            cons_cnt = active_sess.attendance.get("consolidated_count", len(df))
            disc_cnt = active_sess.attendance.get("discarded_count", 0)

            # Transparent track consolidation metrics (BUG 4)
            st.info(
                f"🔍 **Track Analysis:** Raw tracks: **{raw_cnt}** ➔ Consolidated participants: **{cons_cnt}** | "
                f"Discarded short tracks (<{VISION_CONFIG.get('min_track_seconds', 2.0):.1f}s): **{disc_cnt}**"
            )

            # Soft hint warning on cohort size mismatch (BUG 4)
            if cons_cnt != st.session_state.expected_participants:
                st.warning(
                    f"⚠️ Detected **{cons_cnt}** consolidated participant(s), which differs from expected cohort size "
                    f"({st.session_state.expected_participants}). Expected cohort size is used only as a soft normalization hint."
                )

            if df.empty:
                st.warning("No face tracks recorded during this session.")
            else:
                # Interactive Plotly horizontal presence chart (BUG 9: taller chart, no label overlap)
                st.subheader("Accumulated Presence Duration vs. Threshold")
                fig = go.Figure()

                colors = ["#059669" if s == "PRESENT" else "#EF4444" for s in df["attendance_status"]]
                req_seconds = df["required_presence_seconds"].iloc[0] if "required_presence_seconds" in df.columns else 6.0

                chart_height = max(380, len(df) * 65 + 130)

                fig.add_trace(
                    go.Bar(
                        y=df["participant_id"],
                        x=df["accumulated_presence_seconds"],
                        orientation="h",
                        marker=dict(color=colors),
                        text=[f"{s} ({d:.1f}s)" for s, d in zip(df["attendance_status"], df["accumulated_presence_seconds"])],
                        textposition="outside",
                        hovertext=[
                            f"Participant: {p}<br>Status: {s}<br>Observed: {d:.1f}s<br>Required: {req_seconds:.1f}s<br>Rationale: {r}"
                            for p, s, d, r in zip(df["participant_id"], df["attendance_status"], df["accumulated_presence_seconds"], df.get("absence_reason", df.get("evaluation_rationale", [""] * len(df))))
                        ],
                        hoverinfo="text",
                    )
                )

                # Threshold line labelled 'Required 46.1s' (BUG 9)
                fig.add_vline(
                    x=req_seconds,
                    line_dash="dash",
                    line_color="#DC2626",
                    annotation_text=f"Required {req_seconds:.1f}s",
                    annotation_position="top right",
                    annotation_font_color="#DC2626",
                )

                fig.update_layout(
                    template="plotly_white",
                    paper_bgcolor="#FFFFFF",
                    plot_bgcolor="#F8FAFC",
                    font=dict(color="#0F172A"),
                    xaxis_title="Accumulated Presence (Seconds)",
                    yaxis_title="Anonymous Tracking ID",
                    margin=dict(l=40, r=40, t=50, b=40),
                    height=chart_height,
                )
                st.plotly_chart(fig, use_container_width=True)

                # Detailed Participant Verification Table (BUG 4 & BUG 7)
                st.subheader("Detailed Participant Verification Table")
                # Ensure evaluation rationale is mapped to absence_reason or evaluation_rationale
                if "evaluation_rationale" not in df.columns and "absence_reason" in df.columns:
                    df["evaluation_rationale"] = df["absence_reason"]

                st.dataframe(
                    df,
                    column_config={
                        "participant_id": st.column_config.TextColumn("Tracking ID", width="medium"),
                        "attendance_status": st.column_config.TextColumn("Status", width="small"),
                        "accumulated_presence_seconds": st.column_config.NumberColumn("Observed (s)", format="%.1f s"),
                        "required_presence_seconds": st.column_config.NumberColumn("Required (s)", format="%.1f s"),
                        "presence_percentage": st.column_config.ProgressColumn("Session Presence", min_value=0.0, max_value=100.0, format="%.1f%%"),
                        "evaluation_rationale": st.column_config.TextColumn("Evaluation Rationale", width="large"),
                    },
                    use_container_width=True,
                    hide_index=True,
                )

    # ==========================================
    # 4. ACOUSTIC ACTIVITY VIEW (VAD)
    # ==========================================
    elif navigation == "🎙️ Acoustic Activity (VAD)":
        st.markdown("### 🎙️ Acoustic Activity & Flicker-Free VAD")

        # ONE SOURCE OF TRUTH (BUG 2)
        active_sess = st.session_state.get("active_session")
        render_kpi_row(active_sess)

        if active_sess is None:
            st.info("No active session loaded. Process a session or select one from 'Session Archive & Compare'.")
        elif not active_sess.audio.get("has_audio"):
            # Informative message for no-audio or extraction failure (BUG 6)
            aud_status = active_sess.audio.get("status", "UNAVAILABLE")
            if aud_status == "NO_AUDIO_TRACK":
                st.info("🎙️ **This video has no audio track — audio metrics skipped. (Vision-only mode)**")
            elif aud_status == "EXTRACTION_FAILED":
                st.error(f"⚠️ Audio stream detected in video, but audio extraction failed: {active_sess.audio.get('extraction_error')}")
                st.markdown(
                    """
                    **Windows Setup Instructions:**
                    1. Ensure `imageio-ffmpeg` is installed: `pip install imageio-ffmpeg`
                    2. Or install system FFmpeg via winget: `winget install Gyan.FFmpeg`
                    3. Restart your terminal or add the ffmpeg binary to your system PATH.
                    """
                )
            else:
                st.info("No audio metrics available for this session. Audio track was omitted or unavailable.")
        else:
            raw_metrics = active_sess.audio.get("metrics", [])
            if not raw_metrics:
                st.info("Audio was processed but no analysis windows were recorded.")
            else:
                df_audio = pd.DataFrame(raw_metrics)

                st.subheader("Signal Energy (RMS) vs. Adaptive Noise Floor with Speech Regions")
                fig = go.Figure()

                # RMS trace
                fig.add_trace(
                    go.Scatter(
                        x=df_audio["start_time"],
                        y=df_audio["rms_energy"],
                        name="Signal RMS Energy",
                        line=dict(color="#2563EB", width=2.2),
                        hovertemplate="Time: %{x:.1f}s<br>RMS Energy: %{y:.4f}<extra></extra>",
                    )
                )

                # Noise floor trace
                fig.add_trace(
                    go.Scatter(
                        x=df_audio["start_time"],
                        y=df_audio["noise_floor"],
                        name="Adaptive Noise Floor (10th %ile)",
                        line=dict(color="#D97706", width=1.8, dash="dot"),
                        hovertemplate="Time: %{x:.1f}s<br>Noise Floor: %{y:.4f}<extra></extra>",
                    )
                )

                # Highlight speech intervals
                speech_rows = df_audio[df_audio["is_speech"] == True]
                for _, sm in speech_rows.iterrows():
                    fig.add_vrect(
                        x0=sm["start_time"],
                        x1=sm["end_time"],
                        fillcolor="#059669",
                        opacity=0.15,
                        layer="below",
                        line_width=0,
                    )

                fig.update_layout(
                    template="plotly_white",
                    paper_bgcolor="#FFFFFF",
                    plot_bgcolor="#F8FAFC",
                    font=dict(color="#0F172A"),
                    xaxis_title="Session Timeline (Seconds)",
                    yaxis_title="RMS Amplitude",
                    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
                    margin=dict(l=40, r=40, t=50, b=40),
                    height=360,
                )
                st.plotly_chart(fig, use_container_width=True)

                st.caption("🟢 Green shaded bands indicate Voice Activity Detection (VAD) with hangover smoothing.")

                with st.expander("🔍 View Raw Audio Analysis Windows"):
                    st.dataframe(df_audio, use_container_width=True)

    # ==========================================
    # 5. ENGAGEMENT ANALYTICS VIEW
    # ==========================================
    elif navigation == "📈 Multimodal Engagement":
        st.markdown("### 📈 Multimodal Engagement Trajectory")

        # ONE SOURCE OF TRUTH (BUG 2)
        active_sess = st.session_state.get("active_session")
        render_kpi_row(active_sess)

        if active_sess is None or not active_sess.engagement.get("has_timeline"):
            st.info("No engagement timeline recorded yet. Process a session to view engagement analysis.")
        else:
            timeline_data = active_sess.engagement.get("timeline", [])
            df_eng = pd.DataFrame(timeline_data)

            if not active_sess.audio.get("has_audio"):
                st.info("👁️ **Vision-Only Mode:** Acoustic metrics skipped. Engagement score is derived from observed visual presence and micro-movements.")

            st.subheader("Color-Coded Engagement Index Over Time (0–100 Scale)")

            fig = go.Figure()

            # Background color zones: Green (High > 68), Yellow (Moderate 40-68), Red (Low < 40)
            fig.add_hrect(y0=68, y1=100, fillcolor="#059669", opacity=0.08, layer="below", line_width=0, annotation_text="High Engagement (68-100)", annotation_position="top left", annotation_font_color="#059669")
            fig.add_hrect(y0=40, y1=68, fillcolor="#D97706", opacity=0.08, layer="below", line_width=0, annotation_text="Moderate (40-68)", annotation_position="top left", annotation_font_color="#D97706")
            fig.add_hrect(y0=0, y1=40, fillcolor="#DC2626", opacity=0.08, layer="below", line_width=0, annotation_text="Low Activity (<40)", annotation_position="bottom left", annotation_font_color="#DC2626")

            fig.add_trace(
                go.Scatter(
                    x=df_eng["bin_start"],
                    y=df_eng["score"],
                    mode="lines+markers",
                    name="Engagement Score",
                    line=dict(color="#2563EB", width=2.5),
                    marker=dict(size=6, color="#1D4ED8"),
                    hovertemplate="Time: %{x:.1f}s<br>Score: %{y:.1f}<br>Active: %{text}<extra></extra>",
                    text=df_eng["active_participants"].apply(lambda n: f"{n} participant(s)"),
                )
            )

            fig.update_layout(
                template="plotly_white",
                paper_bgcolor="#FFFFFF",
                plot_bgcolor="#F8FAFC",
                font=dict(color="#0F172A"),
                xaxis_title="Session Timeline (Seconds)",
                yaxis_title="Engagement Index (0–100)",
                yaxis=dict(range=[-2, 105]),
                margin=dict(l=40, r=40, t=40, b=40),
                height=380,
            )
            st.plotly_chart(fig, use_container_width=True)

            # "Explain This Score" Interactive Expander
            st.markdown("---")
            with st.expander("🔍 Explain This Score (Interactive Inspection)", expanded=True):
                st.markdown("Select any second from the session timeline to inspect its mathematical derivation:")
                max_sec = int(df_eng["bin_start"].max()) if not df_eng.empty else 0
                selected_sec = st.slider("Select Timeline Second (s)", min_value=0, max_value=max(1, max_sec), value=min(2, max_sec))

                matching = [r for r in timeline_data if int(r.get("bin_start", 0)) == selected_sec]
                if matching:
                    res = matching[0]
                    e_col1, e_col2, e_col3 = st.columns(3)
                    e_col1.metric("Selected Interval", f"t = {res.get('bin_start', 0):.1f}s - {res.get('bin_end', 1):.1f}s")
                    e_col2.metric("Computed Score", f"{res.get('score', 0):.1f} / 100", f"{res.get('level', 'MODERATE')} Level")
                    e_col3.metric("Modality", res.get("active_modality", "BIMODAL"))

                    st.markdown(f"**Mathematical Formula:** `{res.get('formula_string', '')}`")

                    b_col1, b_col2, b_col3 = st.columns(3)
                    b_col1.info(f"👤 **Presence Score:** {res.get('presence_score', 0):.1f} (Active visible: {res.get('active_participants', 0)})")
                    b_col2.info(f"🔄 **Movement Score:** {res.get('movement_score', 0):.1f}")
                    b_col3.info(f"🎙️ **Audio Score:** {res.get('audio_score', 0):.1f}")

                    st.caption(f"**Explanation:** {res.get('explanation', '')}")
                else:
                    st.caption("No observation found for this exact second.")

    # ==========================================
    # 6. TEACHING SUGGESTIONS VIEW
    # ==========================================
    elif navigation == "💡 Teaching Suggestions":
        st.markdown("### 💡 Data-Driven Pedagogical Suggestions")

        # ONE SOURCE OF TRUTH (BUG 2)
        active_sess = st.session_state.get("active_session")
        render_kpi_row(active_sess)

        if active_sess is None or not active_sess.suggestions:
            st.info("No teaching suggestions generated yet. Process a session to view pedagogical insights.")
        else:
            suggestions = active_sess.suggestions
            st.write(f"The recommendation engine evaluated session trends and generated **{len(suggestions)} non-prescriptive suggestion(s)**:")

            for s in suggestions:
                card_class = (
                    "rec-card-high" if s.get("priority") == "HIGH"
                    else "rec-card-med" if s.get("priority") == "MEDIUM"
                    else "rec-card-info"
                )
                metrics_dict = s.get("exact_metrics", {})
                metric_tags = " | ".join(f"{k}: {v}" for k, v in metrics_dict.items()) if metrics_dict else ""
                t_start = s.get("interval_start", 0.0)
                t_end = s.get("interval_end", 0.0)
                conf = s.get("confidence", 0.85)

                st.markdown(
                    f"""
                    <div class="{card_class}">
                        <div style="font-weight: 700; font-size: 1.1rem; color: #0F172A;">
                            [{s.get('priority', 'INFO')}] {s.get('title', 'Teaching Suggestion')}
                            <span style="font-size: 0.85rem; font-weight: normal; color: #64748B;">
                                [t = {t_start:.1f}s - {t_end:.1f}s | Confidence: {conf:.0%}]
                            </span>
                        </div>
                        <div style="margin-top: 6px; color: #334155; font-size: 0.95rem;">
                            <b>Observation:</b> {s.get('observation', '')}
                        </div>
                        <div style="margin-top: 6px; color: #2563EB; font-size: 0.95rem; font-weight: 500;">
                            <b>Pedagogical Suggestion:</b> {s.get('suggestion', '')}
                        </div>
                        <div style="margin-top: 4px; color: #64748B; font-size: 0.85rem;">
                            <b>Rationale:</b> {s.get('pedagogical_rationale', '')}
                        </div>
                        <div style="margin-top: 4px; color: #475569; font-size: 0.8rem; font-family: monospace;">
                            📊 Supporting Metrics: {metric_tags}
                        </div>
                        <div style="margin-top: 4px; color: #D97706; font-size: 0.8rem; font-style: italic;">
                            ⚠️ Caveat: {s.get('limitation_note', '')}
                        </div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

    # ==========================================
    # 7. SESSION ARCHIVE & COMPARE VIEW (BUG 8)
    # ==========================================
    elif navigation == "📊 Session Archive & Compare":
        st.markdown("### 📊 Session Archive & Cross-Session Comparison")

        session_files = sorted(
            [f for f in SESSIONS_DIR.glob("*.json") if not f.stem.startswith("test_")],
            key=lambda p: p.stat().st_mtime,
            reverse=True,
        )

        if not session_files:
            st.info("No saved session records found in `data/sessions/`.")
        else:
            st.write(f"Found **{len(session_files)} saved session file(s)** on disk:")

            # Option to hide sessions with missing modalities (BUG 8)
            hide_missing = st.checkbox("Hide sessions with missing modalities from comparison", value=False)

            filtered_files = []
            for f in session_files:
                d = SessionReportGenerator.load_session_from_disk(f.stem)
                if d:
                    has_aud = d.get("audio_summary", {}).get("audio_status") == "ANALYZED"
                    has_eng = len(d.get("engagement_timeline", [])) > 0 or d.get("engagement_summary", {}).get("mean_score") is not None
                    if hide_missing and (not has_aud or not has_eng):
                        continue
                    filtered_files.append(f)

            if not filtered_files:
                filtered_files = session_files

            sess_names = [f.stem for f in filtered_files]

            cmp_col1, cmp_col2 = st.columns(2)
            with cmp_col1:
                sess_a = st.selectbox("Select Session A", sess_names, index=0)
                if st.button(f"📂 Set '{sess_a}' as Active Session", key="btn_act_a"):
                    st.session_state.active_session = SessionResult.from_dict(SessionReportGenerator.load_session_from_disk(sess_a))
                    st.success(f"Session '{sess_a}' is now active!")
                    st.rerun()

            with cmp_col2:
                sess_b = st.selectbox("Select Session B (to compare)", sess_names, index=min(1, len(sess_names) - 1))
                if st.button(f"📂 Set '{sess_b}' as Active Session", key="btn_act_b"):
                    st.session_state.active_session = SessionResult.from_dict(SessionReportGenerator.load_session_from_disk(sess_b))
                    st.success(f"Session '{sess_b}' is now active!")
                    st.rerun()

            data_a = SessionReportGenerator.load_session_from_disk(sess_a)
            data_b = SessionReportGenerator.load_session_from_disk(sess_b)

            if data_a and data_b:
                st.markdown("---")
                st.subheader("Side-by-Side Comparison")

                # Handle missing modalities with 'N/A' (BUG 8)
                def get_metric_val(data, metric_type):
                    aud = data.get("audio_summary", {})
                    eng = data.get("engagement_summary", {})
                    has_aud = aud.get("audio_status") == "ANALYZED"

                    if metric_type == "duration":
                        return f"{data.get('session_duration_seconds', 0.0):.1f}s"
                    elif metric_type == "tracked":
                        return data.get("attendance_summary", {}).get("total_temporary_tracking_ids", 0)
                    elif metric_type == "present":
                        return data.get("attendance_summary", {}).get("present_count", 0)
                    elif metric_type == "att_rate":
                        return f"{data.get('attendance_summary', {}).get('attendance_rate_percentage', 0.0):.1f}%"
                    elif metric_type == "engagement":
                        m = eng.get("mean_score")
                        return f"{m:.1f}" if m is not None else "N/A"
                    elif metric_type == "speech":
                        sp = aud.get("speech_percentage")
                        return f"{sp:.1f}%" if has_aud and sp is not None else "N/A"
                    elif metric_type == "noise":
                        nf = aud.get("ambient_noise_floor")
                        return f"{nf:.4f}" if has_aud and nf is not None else "N/A"
                    elif metric_type == "suggestions":
                        return len(data.get("recommendations", []))
                    return "N/A"

                metrics_data = {
                    "Metric": [
                        "Session Duration",
                        "Participants Tracked",
                        "Confirmed Present",
                        "Attendance Rate (%)",
                        "Mean Engagement (0-100)",
                        "Speech Activity Ratio (%)",
                        "Noise Floor (RMS)",
                        "Suggestions Generated",
                    ],
                    f"Session A ({sess_a})": [
                        get_metric_val(data_a, "duration"),
                        get_metric_val(data_a, "tracked"),
                        get_metric_val(data_a, "present"),
                        get_metric_val(data_a, "att_rate"),
                        get_metric_val(data_a, "engagement"),
                        get_metric_val(data_a, "speech"),
                        get_metric_val(data_a, "noise"),
                        get_metric_val(data_a, "suggestions"),
                    ],
                    f"Session B ({sess_b})": [
                        get_metric_val(data_b, "duration"),
                        get_metric_val(data_b, "tracked"),
                        get_metric_val(data_b, "present"),
                        get_metric_val(data_b, "att_rate"),
                        get_metric_val(data_b, "engagement"),
                        get_metric_val(data_b, "speech"),
                        get_metric_val(data_b, "noise"),
                        get_metric_val(data_b, "suggestions"),
                    ],
                }
                df_cmp = pd.DataFrame(metrics_data)
                st.dataframe(df_cmp, use_container_width=True, hide_index=True)

            # Delete Session section with confirmation (BUG 8)
            st.markdown("---")
            with st.expander("🗑️ Delete a Saved Session"):
                sess_to_delete = st.selectbox("Select Session to Delete", [f.stem for f in session_files], key="sel_del")
                confirm_del = st.checkbox(f"Confirm permanent deletion of '{sess_to_delete}'", key="chk_del")
                if st.button(f"🗑️ Permanently Delete '{sess_to_delete}'", type="secondary", disabled=not confirm_del):
                    del_path = SESSIONS_DIR / f"{sess_to_delete}.json"
                    if del_path.exists():
                        del_path.unlink()
                        st.success(f"Session '{sess_to_delete}' deleted successfully!")
                        if st.session_state.active_session and st.session_state.active_session.session_id == sess_to_delete:
                            st.session_state.active_session = load_most_recent_session()
                        st.rerun()

    # ==========================================
    # 8. REPORTS & DOWNLOADS VIEW (BUG 1)
    # ==========================================
    elif navigation == "📥 Reports & Downloads":
        active_sess = st.session_state.get("active_session")
        render_reports_view(active_sess)

except Exception as e:
    logger.exception("Error rendering page '%s': %s", navigation, e)
    st.error(f"⚠️ An error occurred while rendering the page '{navigation}'. The details have been logged.")
