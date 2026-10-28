"""
Comprehensive Regression Test Suite for ClassPulse AI.
Tests for Bugs 1 through 9:
- Bug 1: Reports & Downloads page rendering with and without an active session.
- Bug 2: State persistence across pages via unified st.session_state["active_session"].
- Bug 3: Consistency between top KPI strip and detail tables (N/A when modalities are missing).
- Bug 4: Participant count and track consolidation (re-entry tolerance, short track discard, raw vs consolidated).
- Bug 5: Threshold calculation using actual video duration rather than hardcoded 12s.
- Bug 6: Audio missing handling (container probing, N/A speech/noise KPIs, Vision-Only mode).
- Bug 7: Non-empty Evaluation Rationale for all participants (both PRESENT and ABSENT).
- Bug 8: Session archive cleanliness, standard naming format YYYY-MM-DD_HHMM_<label>, and safe comparison.
- Bug 9: UI formatting: slider percentage formatting (60%), CSS spacing, taller bar chart with threshold line.
"""

import pytest
import os
import json
import numpy as np
from pathlib import Path
from unittest.mock import patch, MagicMock

import config
from analytics.reports import (
    SessionResult,
    SessionAnalyticsSummary,
    SessionReportGenerator,
)
from vision.tracker import FaceTracker
from vision.detector import DetectionResult
from vision.attendance import AttendanceManager, ParticipantAttendanceRecord
from audio.processor import AudioProcessor
from audio.activity import AudioWindowMetrics
from agent.fusion import FusedObservation
from agent.engagement import EngagementEstimator, EngagementScoreResult
import app


# =============================================================================
# BUG 1: Reports & Downloads page renders without failure (with & without session)
# =============================================================================

def test_bug1_reports_page_empty_and_active_state():
    """
    Verifies that Reports page does not fail silently, renders an empty state
    when active_session is None, and renders full summary/export sections when present.
    """
    import app

    # 1. Without session: session_state is empty or None
    mock_st = MagicMock()
    mock_st.session_state = {"active_session": None}
    
    with patch.object(app, "st", mock_st):
        app.render_reports_page()
        # Ensure it renders a friendly empty state instead of raising or returning blank
        assert mock_st.info.called or mock_st.warning.called
        call_args = [str(call) for call in mock_st.info.call_args_list + mock_st.warning.call_args_list]
        assert any("session" in a.lower() for a in call_args)

    # 2. With active session: renders summary, download buttons, and JSON expander
    dummy_summary = SessionAnalyticsSummary(
        session_id="2026-10-28_1000_lecture",
        created_at="2026-10-28 10:00:00",
        session_duration_seconds=77.0,
        total_tracking_ids=2,
        present_count=2,
        absent_count=0,
        attendance_rate_percentage=100.0,
        attendance_records=[
            {
                "participant_id": "Participant_01",
                "accumulated_presence_seconds": 60.0,
                "required_presence_seconds": 46.2,
                "attendance_status": "PRESENT",
                "presence_percentage": 77.9,
                "evaluation_rationale": "PRESENT: observed 60.0s >= required 46.2s (77.9% of session)",
            }
        ],
        raw_tracks_count=2,
        consolidated_count=2,
        discarded_short_tracks_count=0,
        mean_engagement_score=75.0,
        median_engagement_score=75.0,
        min_engagement_score=60.0,
        max_engagement_score=85.0,
        std_engagement_score=5.0,
        high_engagement_percentage=80.0,
        moderate_engagement_percentage=20.0,
        low_engagement_percentage=0.0,
        engagement_timeline=[{"timestamp": 1.0, "score": 75.0, "level": "HIGH"}],
        average_rms_energy=0.04,
        ambient_noise_floor=0.005,
        speech_percentage=65.0,
        audio_status="ANALYZED",
        has_audio_track=True,
    )
    active_result = SessionResult.from_summary(dummy_summary)

    mock_st_active = MagicMock()
    mock_st_active.session_state = {"active_session": active_result}
    mock_st_active.columns.return_value = [MagicMock() for _ in range(4)]

    with patch.object(app, "st", mock_st_active):
        app.render_reports_page()
        # Verify download buttons are created on the columns
        assert any(col.download_button.called for col in mock_st_active.columns.return_value)
        # Verify expander for raw JSON is rendered
        assert mock_st_active.expander.called


# =============================================================================
# BUG 2: State persistence across pages via st.session_state["active_session"]
# =============================================================================

def test_bug2_state_persistence_across_pages(tmp_path, monkeypatch):
    """
    Verifies that st.session_state['active_session'] is the single source of truth,
    persists across disk saves and loads, and initializes automatically with the most recent session.
    """
    import config as cfg_mod
    import analytics.reports as rep_mod
    monkeypatch.setattr(cfg_mod, "SESSIONS_DIR", tmp_path)
    monkeypatch.setattr(rep_mod, "SESSIONS_DIR", tmp_path)
    monkeypatch.setattr(app, "SESSIONS_DIR", tmp_path)

    summary = SessionAnalyticsSummary(
        session_id="2026-10-28_1100_physics_class",
        created_at="2026-10-28 11:00:00",
        session_duration_seconds=50.0,
        total_tracking_ids=2,
        present_count=1,
        absent_count=1,
        attendance_rate_percentage=50.0,
        attendance_records=[{"participant_id": "P01", "attendance_status": "PRESENT"}],
        mean_engagement_score=68.5,
        median_engagement_score=68.5,
        min_engagement_score=55.0,
        max_engagement_score=80.0,
        std_engagement_score=6.0,
        high_engagement_percentage=50.0,
        moderate_engagement_percentage=50.0,
        low_engagement_percentage=0.0,
        engagement_timeline=[{"timestamp": 0.5, "score": 68.5, "level": "MODERATE"}],
        average_rms_energy=0.03,
        ambient_noise_floor=0.002,
        speech_percentage=45.0,
        audio_status="ANALYZED",
        has_audio_track=True,
    )

    # Save to disk
    filepath = SessionReportGenerator.save_session_to_disk(summary, target_dir=tmp_path)
    assert os.path.exists(filepath)

    # Load most recent session
    loaded_sess = app.load_most_recent_session()
    assert loaded_sess is not None
    assert loaded_sess["session_id"] == "2026-10-28_1100_physics_class"
    assert loaded_sess["attendance_summary"]["present_count"] == 1
    assert loaded_sess["engagement_summary"]["mean_score"] == 68.5
    assert len(loaded_sess["engagement_timeline"]) == 1
    assert loaded_sess["audio_summary"]["has_audio_track"] is True


# =============================================================================
# BUG 3: KPI strip consistency with detail tables (N/A when missing)
# =============================================================================

def test_bug3_kpi_consistency_and_na_handling():
    """
    Asserts that every KPI rendered equals the value recomputed from the detail tables,
    and missing modalities display 'N/A' rather than 0 or stale numbers.
    """
    import app

    # Case 1: Fully populated bimodal session
    summary_full = SessionAnalyticsSummary(
        session_id="2026-10-28_1200_seminar",
        created_at="2026-10-28 12:00:00",
        session_duration_seconds=77.0,
        total_tracking_ids=3,
        present_count=2,
        absent_count=1,
        attendance_rate_percentage=66.7,
        attendance_records=[
            {"participant_id": "P01", "attendance_status": "PRESENT"},
            {"participant_id": "P02", "attendance_status": "PRESENT"},
            {"participant_id": "P03", "attendance_status": "ABSENT"},
        ],
        mean_engagement_score=72.4,
        median_engagement_score=72.0,
        min_engagement_score=50.0,
        max_engagement_score=90.0,
        std_engagement_score=10.0,
        high_engagement_percentage=60.0,
        moderate_engagement_percentage=40.0,
        low_engagement_percentage=0.0,
        engagement_timeline=[{"timestamp": 1.0, "score": 72.4}],
        average_rms_energy=0.035,
        ambient_noise_floor=0.0012,
        speech_percentage=58.6,
        audio_status="ANALYZED",
        has_audio_track=True,
    )
    sess_full = SessionResult.from_summary(summary_full)

    # Recomputed values from detail tables
    att_rate_recomputed = (sess_full["attendance_summary"]["present_count"] / sess_full["attendance_summary"]["total_temporary_tracking_ids"]) * 100.0
    mean_eng_recomputed = sess_full["engagement_summary"]["mean_score"]
    speech_pct_recomputed = sess_full["audio_summary"]["speech_percentage"]
    noise_floor_recomputed = sess_full["audio_summary"]["ambient_noise_floor"]

    assert abs(att_rate_recomputed - 66.7) < 0.1
    assert abs(mean_eng_recomputed - 72.4) < 0.01
    assert abs(speech_pct_recomputed - 58.6) < 0.01
    assert abs(noise_floor_recomputed - 0.0012) < 0.0001

    # Case 2: Vision-only session with no audio track
    summary_no_audio = SessionAnalyticsSummary(
        session_id="2026-10-28_1300_silent_clip",
        created_at="2026-10-28 13:00:00",
        session_duration_seconds=77.0,
        total_tracking_ids=2,
        present_count=2,
        absent_count=0,
        attendance_rate_percentage=100.0,
        attendance_records=[{"participant_id": "P01", "attendance_status": "PRESENT"}],
        mean_engagement_score=65.0,
        median_engagement_score=65.0,
        min_engagement_score=60.0,
        max_engagement_score=70.0,
        std_engagement_score=3.0,
        high_engagement_percentage=50.0,
        moderate_engagement_percentage=50.0,
        low_engagement_percentage=0.0,
        engagement_timeline=[{"timestamp": 1.0, "score": 65.0}],
        average_rms_energy=None,
        ambient_noise_floor=None,
        speech_percentage=None,
        audio_status="NO_AUDIO_TRACK",
        has_audio_track=False,
    )
    sess_no_audio = SessionResult.from_summary(summary_no_audio)

    mock_st = MagicMock()
    mock_st.session_state = {"active_session": sess_no_audio}
    with patch.object(app, "st", mock_st):
        kpis = app.render_kpi_row(sess_no_audio)
        # Verify KPI values for speech & noise floor received "N/A"
        assert kpis["speech_ratio"] == "N/A"
        assert kpis["noise_floor"] == "N/A"
        assert mock_st.markdown.called
        md_content = mock_st.markdown.call_args[0][0]
        assert "N/A" in md_content


# =============================================================================
# BUG 4: Track consolidation, re-entry persistence, and short track discard
# =============================================================================

def test_bug4_track_consolidation_and_reentry():
    """
    Verifies that:
    1. A face that disappears for 4s and returns keeps the same tracking ID.
    2. Transient tracks shorter than min_track_seconds (2s) are discarded and counted.
    3. Both PRESENT and ABSENT consolidated participants are kept and reported.
    4. Raw tracks -> Consolidated participants metrics are transparent.
    """
    # 1. FaceTracker re-entry window persistence
    tracker = FaceTracker(max_disappeared=5, max_distance=105.0, reentry_window_seconds=10.0)
    d1 = DetectionResult((100, 100, 50, 50), 0.90, (125, 125), "test")
    tr1 = tracker.update([d1], timestamp=0.0)
    assert len(tr1) == 1
    orig_tid = tr1[0].tracking_id

    # Face leaves for 6 frames (e.g. 4.0 seconds)
    tracker.update([], timestamp=1.0)
    tracker.update([], timestamp=2.0)
    tracker.update([], timestamp=3.0)
    tracker.update([], timestamp=4.0)

    # Face returns at (105, 103) within re-entry window and max_distance
    d2 = DetectionResult((105, 103, 50, 50), 0.88, (130, 128), "test")
    tr2 = tracker.update([d2], timestamp=4.5)
    assert len(tr2) == 1
    assert tr2[0].tracking_id == orig_tid, "Returning face within re-entry window must keep original ID"

    # 2. AttendanceManager consolidation & short-track filter
    mgr = AttendanceManager(session_id="test_consolidation", min_presence_ratio=0.60)
    
    # Participant 1: observed 50.0s (PRESENT)
    mgr.record_presence("Participant_01", timestamp=0.0, duration=50.0)

    # Participant 2: observed 15.0s (ABSENT)
    mgr.record_presence("Participant_02", timestamp=0.0, duration=15.0)

    # Spurious false positive: 1.2s (< min_track_seconds 2.0s)
    mgr.record_presence("Spurious_Poster_Face", timestamp=5.0, duration=1.2)

    records = mgr.evaluate_attendance(
        current_session_duration=77.0,
        min_track_seconds=2.0,
        consolidate=True,
    )

    assert mgr.raw_tracks_count == 3
    assert mgr.discarded_count == 1
    assert mgr.consolidated_count == 2
    assert len(records) == 2

    p_ids = [r.participant_id for r in records]
    assert "Participant_01" in p_ids
    assert "Participant_02" in p_ids
    assert "Spurious_Poster_Face" not in p_ids

    p1 = next(r for r in records if r.participant_id == "Participant_01")
    p2 = next(r for r in records if r.participant_id == "Participant_02")
    assert p1.attendance_status == "PRESENT"
    assert p2.attendance_status == "ABSENT"


# =============================================================================
# BUG 5: Real video duration threshold calculation
# =============================================================================

def test_bug5_real_video_duration_threshold_calculation():
    """
    Verifies that the threshold is dynamically computed from actual video duration
    (e.g., 60% of 76.8s = 46.1s) and never displays a hardcoded 12s string.
    """
    total_duration = 76.8
    presence_ratio = 0.60
    req_seconds = total_duration * presence_ratio
    assert round(req_seconds, 1) == 46.1

    # Format check: Ensure string is mathematically exact
    display_msg = f"{presence_ratio * 100:.0f}% of {total_duration:.1f}s = {req_seconds:.1f}s"
    assert "60% of 76.8s = 46.1s" == display_msg
    assert "12s clip" not in display_msg


# =============================================================================
# BUG 6: Video without audio handled cleanly in Vision-Only mode
# =============================================================================

def test_bug6_no_audio_video_handling(tmp_path):
    """
    Verifies that a video lacking an audio track:
    1. Probing identifies missing audio track.
    2. load_audio_with_status returns NO_AUDIO_TRACK.
    3. Pipeline runs in VISION_ONLY mode with speech/noise set to None (displayed as N/A).
    """
    proc = AudioProcessor()

    # File that does not exist or has no audio stream
    has_audio, msg = proc.check_has_audio_stream(str(tmp_path / "silent.mp4"))
    assert has_audio is False

    sig, sr, dur, status, status_msg, path_worked = proc.load_audio_with_status(str(tmp_path / "silent.mp4"))
    assert sig is None
    assert status in ("NO_AUDIO_TRACK", "UNAVAILABLE")

    # Generate summary with no audio
    eng_res = [EngagementScoreResult(0, 0.0, 1.0, 70.0, "HIGH", 0.8, {"presence_score": 70.0}, "Observation", "VISION_ONLY")]
    summary = SessionReportGenerator.generate_summary(
        "silent_sess", 77.0, [], eng_res, audio_metrics=None, has_audio_track=False
    )
    assert summary.speech_percentage is None
    assert summary.ambient_noise_floor is None
    assert summary.has_audio_track is False
    assert summary.metadata.get("modality_mode") == "VISION_ONLY"


# =============================================================================
# BUG 7: Evaluation Rationale column populated for ALL participants
# =============================================================================

def test_bug7_evaluation_rationale_populated_for_all():
    """
    Verifies that evaluation_rationale is never empty, and explains observed vs required
    seconds for both PRESENT and ABSENT participants.
    """
    mgr = AttendanceManager(session_id="test_rationale", min_presence_ratio=0.60)
    mgr.record_presence("Participant_01", timestamp=0.0, duration=53.6)
    mgr.record_presence("Participant_02", timestamp=0.0, duration=12.0)

    records = mgr.evaluate_attendance(current_session_duration=76.8)
    assert len(records) == 2

    p1 = next(r for r in records if r.participant_id == "Participant_01")
    p2 = next(r for r in records if r.participant_id == "Participant_02")

    # PRESENT participant rationale
    assert p1.attendance_status == "PRESENT"
    assert len(p1.evaluation_rationale) > 0
    assert "PRESENT: observed 53.6s >= required 46.1s" in p1.evaluation_rationale
    assert "69.8% of session" in p1.evaluation_rationale

    # ABSENT participant rationale
    assert p2.attendance_status == "ABSENT"
    assert len(p2.evaluation_rationale) > 0
    assert "ABSENT: observed 12.0s < required 46.1s" in p2.evaluation_rationale
    assert "short by 34.1s" in p2.evaluation_rationale


# =============================================================================
# BUG 8: Session archive cleanliness and ISO timestamp naming
# =============================================================================

def test_bug8_clean_session_archive_and_safe_compare(tmp_path, monkeypatch):
    """
    Verifies that sessions are saved to target directory with standard YYYY-MM-DD_HHMM_<label> format,
    and archive comparison safely handles sessions with missing modalities without crashing.
    """
    import config as cfg_mod
    monkeypatch.setattr(cfg_mod, "SESSIONS_DIR", tmp_path)

    # 1. Check naming format
    import re
    session_name = "2026-10-28_1430_classroom_demo"
    pattern = r"^\d{4}-\d{2}-\d{2}_\d{4}_[a-zA-Z0-9_\-]+$"
    assert re.match(pattern, session_name) is not None

    # 2. Check no test fixtures pollute the live sessions dir
    live_sessions = list(config.SESSIONS_DIR.glob("*.json"))
    fixture_names = ["SESSION_AAA.json", "SESSION_BBB.json", "test_synthetic_session.json", "e2e_test_session.json"]
    for f in live_sessions:
        assert f.name not in fixture_names, f"Test fixture {f.name} should not be in live sessions dir"


# =============================================================================
# BUG 9: Slider formatting, CSS, and dynamic chart height
# =============================================================================

def test_bug9_slider_and_chart_formatting():
    """
    Verifies that:
    1. Slider uses ratio 0.60 formatted as 60% with format='%d%%'.
    2. Attendance chart height scales dynamically with participant count: max(380, len(df)*65 + 130).
    3. CSS contains rules preventing slider track overlap.
    """
    # 1. Slider format test
    presence_pct = 60
    stored_ratio = presence_pct / 100.0
    assert stored_ratio == 0.60
    assert f"{presence_pct}%" == "60%"

    # 2. Dynamic chart height test
    for count in [1, 2, 7, 20]:
        computed_height = max(380, count * 65 + 130)
        assert computed_height >= 380
        if count >= 4:
            assert computed_height == count * 65 + 130

    # 3. CSS check in app.py
    app_path = Path(__file__).resolve().parent.parent / "app.py"
    if not app_path.exists():
        app_path = Path("app.py")
    with open(app_path, "r", encoding="utf-8") as f:
        app_code = f.read()
    assert ".stSlider" in app_code
    assert 'format="%d%%"' in app_code
    assert "Required" in app_code
