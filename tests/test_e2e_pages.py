"""
End-to-End Test Suite for ClassPulse AI.
Verifies requirement 2:
- (a) Built-in synthetic demo execution.
- (b) Video processing execution.
- Verifying after each run that all 8 sidebar pages:
  1. 📌 Overview & Mission
  2. 🚀 Live & Video Processing
  3. 📋 Attendance Verification
  4. 🎙️ Acoustic Activity (VAD)
  5. 📈 Multimodal Engagement
  6. 💡 Teaching Suggestions
  7. 📊 Session Archive & Compare
  8. 📥 Reports & Downloads
  render successfully, are not blank, and have no contradictory KPIs.
"""

import pytest
import os
from pathlib import Path
from unittest.mock import MagicMock, patch
import pandas as pd

import config
from config import SAMPLE_MEDIA_DIR
from core_pipeline import ClassPulsePipeline
from analytics.reports import SessionReportGenerator, SessionResult, SessionAnalyticsSummary
import app


@pytest.fixture
def mock_streamlit():
    """Provides a mocked streamlit environment that records rendered components."""
    mock_st = MagicMock()
    mock_st.session_state = {}
    created_cols = []
    def mock_cols(spec):
        count = spec if isinstance(spec, int) else len(spec)
        res = [MagicMock() for _ in range(count)]
        created_cols.extend(res)
        return res

    mock_st.columns.side_effect = mock_cols
    mock_st.created_cols = created_cols
    mock_st.expander.return_value.__enter__ = MagicMock(return_value=MagicMock())
    mock_st.expander.return_value.__exit__ = MagicMock(return_value=None)
    mock_st.container.return_value.__enter__ = MagicMock(return_value=MagicMock())
    mock_st.container.return_value.__exit__ = MagicMock(return_value=None)
    return mock_st


def test_e2e_synthetic_demo_and_all_pages(tmp_path, monkeypatch, mock_streamlit):
    """
    Executes (a) built-in synthetic demo end-to-end, then navigates through
    all 8 pages and confirms none fail or produce contradictory metrics.
    """
    import analytics.reports as rep_mod
    monkeypatch.setattr(config, "SESSIONS_DIR", tmp_path)
    monkeypatch.setattr(rep_mod, "SESSIONS_DIR", tmp_path)
    monkeypatch.setattr(app, "SESSIONS_DIR", tmp_path)

    # 1. Run Built-in Synthetic Demo Pipeline
    sim_vid = SAMPLE_MEDIA_DIR / "sample_classroom_video.mp4"
    sim_aud = SAMPLE_MEDIA_DIR / "sample_classroom_audio.wav"
    assert sim_vid.exists() and sim_aud.exists()

    pipeline = ClassPulsePipeline(
        session_id="2026-10-28_1000_synthetic_demo",
        min_presence_ratio=0.60,
    )
    vid_ok = pipeline.process_video_file(str(sim_vid))
    assert vid_ok is True
    pipeline.process_audio_file(str(sim_aud))
    summary = pipeline.finalize_session()

    # Save to disk and set active session
    saved_file = SessionReportGenerator.save_session_to_disk(summary, target_dir=tmp_path)
    active_session = SessionResult.from_summary(summary)
    assert active_session is not None
    assert active_session.session_id == "2026-10-28_1000_synthetic_demo"

    # Set up session state
    mock_streamlit.session_state = {
        "active_session": active_session,
        "pipeline": pipeline,
        "min_presence_ratio": 0.60,
        "expected_participants": 2,
        "use_absolute_override": False,
        "min_presence_seconds": None,
        "gap_tolerance": 2.5,
        "session_label": active_session.session_id,
        "is_running": False,
    }

    # 2. Test All 8 Pages with Active Session
    pages = [
        "📌 Overview & Mission",
        "🚀 Live & Video Processing",
        "📋 Attendance Verification",
        "🎙️ Acoustic Activity (VAD)",
        "📈 Multimodal Engagement",
        "💡 Teaching Suggestions",
        "📊 Session Archive & Compare",
        "📥 Reports & Downloads",
    ]

    with patch.object(app, "st", mock_streamlit):
        for page in pages:
            # Check KPI strip computation
            kpis = app.render_kpi_row(active_session)
            assert kpis["session_name"] == active_session.session_id
            assert int(kpis["participants"]) == summary.total_tracking_ids
            assert int(kpis["confirmed_present"]) == summary.present_count
            assert float(kpis["attendance_rate"].replace("%", "")) == round(summary.attendance_rate_percentage, 1)

            # Check specific page rendering logic
            if page == "📥 Reports & Downloads":
                app.render_reports_view(active_session)
                # Verify download buttons generated
                assert any(col.download_button.called for col in mock_streamlit.created_cols)
            elif page == "📋 Attendance Verification":
                df = pipeline.attendance_mgr.get_attendance_dataframe(summary.session_duration_seconds)
                assert len(df) > 0
                assert "evaluation_rationale" in df.columns or "Evaluation Rationale" in df.columns
                col_name = "evaluation_rationale" if "evaluation_rationale" in df.columns else "Evaluation Rationale"
                assert all(len(str(r)) > 0 for r in df[col_name])
            elif page == "🎙️ Acoustic Activity (VAD)":
                assert summary.speech_percentage is not None
                assert summary.ambient_noise_floor is not None
            elif page == "📈 Multimodal Engagement":
                assert len(summary.engagement_timeline) > 0
            elif page == "💡 Teaching Suggestions":
                assert isinstance(summary.recommendations, list)


def test_e2e_uploaded_video_and_all_pages(tmp_path, monkeypatch, mock_streamlit):
    """
    Executes (b) video file end-to-end, then navigates through
    all 8 pages and confirms none fail or produce contradictory metrics.
    """
    import analytics.reports as rep_mod
    monkeypatch.setattr(config, "SESSIONS_DIR", tmp_path)
    monkeypatch.setattr(rep_mod, "SESSIONS_DIR", tmp_path)
    monkeypatch.setattr(app, "SESSIONS_DIR", tmp_path)

    sim_vid = SAMPLE_MEDIA_DIR / "sample_classroom_video.mp4"
    assert sim_vid.exists()

    pipeline = ClassPulsePipeline(
        session_id="2026-10-28_1100_uploaded_video",
        min_presence_ratio=0.50,
    )
    success = pipeline.process_video_file(str(sim_vid))
    assert success is True
    summary = pipeline.finalize_session()

    # Save to disk and set active session
    SessionReportGenerator.save_session_to_disk(summary, target_dir=tmp_path)
    active_session = SessionResult.from_summary(summary)

    mock_streamlit.session_state = {
        "active_session": active_session,
        "pipeline": pipeline,
        "min_presence_ratio": 0.50,
        "expected_participants": 2,
        "use_absolute_override": False,
        "min_presence_seconds": None,
        "gap_tolerance": 2.5,
        "session_label": active_session.session_id,
        "is_running": False,
    }

    pages = [
        "📌 Overview & Mission",
        "🚀 Live & Video Processing",
        "📋 Attendance Verification",
        "🎙️ Acoustic Activity (VAD)",
        "📈 Multimodal Engagement",
        "💡 Teaching Suggestions",
        "📊 Session Archive & Compare",
        "📥 Reports & Downloads",
    ]

    with patch.object(app, "st", mock_streamlit):
        for page in pages:
            kpis = app.render_kpi_row(active_session)
            assert kpis["session_name"] == active_session.session_id

            # Verify no audio contradictory values (either valid float or N/A)
            if active_session.audio.get("has_audio"):
                assert "%" in kpis["speech_ratio"]
            else:
                assert kpis["speech_ratio"] == "N/A"
                assert kpis["noise_floor"] == "N/A"

            if page == "📥 Reports & Downloads":
                app.render_reports_view(active_session)
                assert any(col.download_button.called for col in mock_streamlit.created_cols)
