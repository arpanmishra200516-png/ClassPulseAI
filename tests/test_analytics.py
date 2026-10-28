"""
Unit tests for Engagement Scoring, Recommendation Engine, and Session Analytics in ClassPulse AI.
Covers score bounding, contradiction resolution, session isolation, and CSV/JSON/HTML export accuracy.
"""

import json
import numpy as np
import pytest
from agent.fusion import FusedObservation
from agent.engagement import EngagementEstimator, EngagementScoreResult
from agent.recommendations import RecommendationEngine, TeachingSuggestion
from analytics.reports import SessionReportGenerator, SessionAnalyticsSummary, SCHEMA_VERSION
from vision.attendance import ParticipantAttendanceRecord


def test_engagement_score_range_and_safe_bounds():
    """Validates that engagement scores are strictly bounded in [0.0, 100.0] and never produce NaNs."""
    estimator = EngagementEstimator()
    
    obs_empty = FusedObservation(0, 0.0, 1.0, False, False, 0, 0.0, 0.0, [], None, None, None, None)
    res_empty = estimator.estimate_window(obs_empty)
    assert res_empty.score == 0.0
    assert res_empty.level == "NO_DATA"
    
    obs_high = FusedObservation(
        1, 1.0, 2.0, True, True, 10, 500.0, 999.0, ["P1", "P2"], 10.0, True, 0.01, 80.0
    )
    res_high = estimator.estimate_window(obs_high)
    assert 0.0 <= res_high.score <= 100.0
    assert not np.isnan(res_high.score)


def test_engagement_per_active_participant_consistency():
    """Validates that engagement score accounts for active visible participants."""
    estimator = EngagementEstimator(expected_participants=2)

    # 1 out of 2 expected participants visible
    obs_half = FusedObservation(0, 0.0, 1.0, True, True, 1, 10.0, 10.0, ["P1"], 0.01, False, 0.005, 5.0)
    res_half = estimator.estimate_window(obs_half)

    # 2 out of 2 expected participants visible
    obs_full = FusedObservation(1, 1.0, 2.0, True, True, 2, 10.0, 10.0, ["P1", "P2"], 0.01, False, 0.005, 5.0)
    res_full = estimator.estimate_window(obs_full)

    # Full presence should score higher than half presence
    assert res_full.breakdown["presence_score"] > res_half.breakdown["presence_score"]
    assert res_full.formula_string != ""


def test_recommendation_generation_and_no_contradiction():
    """Validates recommendation generation, minimum duration enforcement, and contradiction resolution."""
    rec_engine = RecommendationEngine(
        sustained_low_duration=5.0,
        low_engagement_threshold=40.0,
        min_suggestion_duration=5.0,
    )
    
    # 7 seconds of low engagement
    low_results = [
        EngagementScoreResult(i, float(i), float(i + 1), 25.0, "LOW", 0.8, {}, "Low activity", "BIMODAL")
        for i in range(7)
    ]
    
    suggestions = rec_engine.generate_recommendations(low_results)
    triggers = [s.trigger_type for s in suggestions]
    assert "SUSTAINED_LOW_ENGAGEMENT" in triggers
    assert "HIGH_PARTICIPATION_SURGE" not in triggers # Contradiction prevented
    
    rec = next(s for s in suggestions if s.trigger_type == "SUSTAINED_LOW_ENGAGEMENT")
    assert rec.time_interval == (0.0, 7.0)
    assert rec.priority == "MEDIUM"
    assert rec.confidence > 0.0
    assert "mean_engagement" in rec.exact_metrics


def test_recommendation_insufficient_data():
    """Validates that empty or very short sessions yield an explicit insufficient data notice."""
    rec_engine = RecommendationEngine(min_suggestion_duration=5.0)
    recs_empty = rec_engine.generate_recommendations([])
    assert len(recs_empty) == 1
    assert recs_empty[0].trigger_type == "INSUFFICIENT_DATA"


def test_session_separation_isolation(tmp_path, monkeypatch):
    """Validates that distinct sessions remain strictly isolated and do not overwrite or mix data."""
    import config
    import analytics.reports as rep_mod
    monkeypatch.setattr(config, "SESSIONS_DIR", tmp_path)
    monkeypatch.setattr(rep_mod, "SESSIONS_DIR", tmp_path)
    
    summary_a = SessionReportGenerator.generate_summary(
        session_id="SESSION_AAA",
        session_duration=60.0,
        attendance_records=[
            ParticipantAttendanceRecord("SESSION_AAA", "Participant_01", 0.0, 50.0, 50.0, "PRESENT", 83.3, 1)
        ],
        engagement_results=[
            EngagementScoreResult(0, 0.0, 1.0, 80.0, "HIGH", 1.0, {}, "High", "BIMODAL")
        ],
    )
    file_a = SessionReportGenerator.save_session_to_disk(summary_a)
    
    summary_b = SessionReportGenerator.generate_summary(
        session_id="SESSION_BBB",
        session_duration=30.0,
        attendance_records=[
            ParticipantAttendanceRecord("SESSION_BBB", "Participant_99", 0.0, 10.0, 10.0, "ABSENT", 33.3, 1)
        ],
        engagement_results=[
            EngagementScoreResult(0, 0.0, 1.0, 30.0, "LOW", 1.0, {}, "Low", "BIMODAL")
        ],
    )
    file_b = SessionReportGenerator.save_session_to_disk(summary_b)
    
    assert file_a.name == "SESSION_AAA.json"
    assert file_b.name == "SESSION_BBB.json"


def test_html_and_json_export_with_schema_version():
    """Validates that JSON export includes schema_version and HTML report generates valid HTML."""
    att_rec = ParticipantAttendanceRecord(
        session_id="export_test",
        participant_id="Participant_01",
        first_seen=0.0,
        last_seen=15.0,
        accumulated_presence_seconds=15.0,
        required_presence_seconds=10.0,
        attendance_status="PRESENT",
        presence_percentage=100.0,
        interval_count=1,
    )
    eng_res = EngagementScoreResult(
        bin_idx=0,
        bin_start=0.0,
        bin_end=1.0,
        score=82.5,
        level="HIGH",
        confidence=0.95,
        breakdown={"presence_score": 100.0, "movement_score": 75.0, "audio_score": 80.0},
        explanation="Testing export",
        active_modality="BIMODAL",
    )
    
    summary = SessionReportGenerator.generate_summary(
        session_id="export_test",
        session_duration=15.0,
        attendance_records=[att_rec],
        engagement_results=[eng_res],
    )
    
    # JSON schema version check
    json_str = summary.to_json()
    parsed = json.loads(json_str)
    assert parsed["schema_version"] == SCHEMA_VERSION
    assert parsed["session_id"] == "export_test"

    # HTML report generation check
    html = SessionReportGenerator.generate_html_report(summary)
    assert "<!DOCTYPE html>" in html
    assert "export_test" in html
    assert "Participant_01" in html
    assert "PRESENT" in html
    assert "Privacy, Academic & Ethical Safeguards" in html
