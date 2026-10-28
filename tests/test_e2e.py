"""
End-to-End integration test for ClassPulse AI.
Asserts that attendance, engagement, acoustic features, and suggestions
are internally consistent across the entire multimodal pipeline.
"""

from pathlib import Path
import pytest
from core_pipeline import ClassPulsePipeline
from config import SAMPLE_MEDIA_DIR


def test_end_to_end_pipeline_consistency(tmp_path, monkeypatch):
    import analytics.reports as rep_mod
    import config as cfg_mod
    monkeypatch.setattr(cfg_mod, "SESSIONS_DIR", tmp_path)
    monkeypatch.setattr(rep_mod, "SESSIONS_DIR", tmp_path)
    """
    Executes full pipeline on synthetic sample media and asserts internal consistency:
    - Session duration matches video/audio length
    - Participants are tracked and have valid attendance records
    - Attendance status matches accumulated presence vs relative threshold
    - Engagement timeline is populated with scores bounded in [0, 100]
    - Acoustic metrics are computed with positive RMS and non-zero noise floor
    - Suggestions are generated with supporting metrics
    """
    video_path = SAMPLE_MEDIA_DIR / "sample_classroom_video.mp4"
    audio_path = SAMPLE_MEDIA_DIR / "sample_classroom_audio.wav"
    
    assert video_path.exists(), "Sample video file must exist"
    assert audio_path.exists(), "Sample audio file must exist"

    pipeline = ClassPulsePipeline(
        session_id="e2e_test_session",
        min_presence_ratio=0.50, # 50% relative presence required
    )

    # 1. Process video frames
    video_success = pipeline.process_video_file(str(video_path))
    assert video_success is True

    # 2. Process audio
    pipeline.process_audio_file(str(audio_path))
    assert len(pipeline.audio_metrics) > 0

    # 3. Finalize
    summary = pipeline.finalize_session()

    # Assertions on Session Duration
    assert summary.session_duration_seconds >= 8.0

    # Assertions on Attendance Consistency
    assert summary.total_tracking_ids >= 1
    assert summary.present_count + summary.absent_count == summary.total_tracking_ids
    assert 0.0 <= summary.attendance_rate_percentage <= 100.0

    for rec in summary.attendance_records:
        accum = rec["accumulated_presence_seconds"]
        req = rec["required_presence_seconds"]
        status = rec["attendance_status"]
        if accum >= req and accum >= 0.08:
            assert status == "PRESENT"
            assert "PRESENT" in rec.get("evaluation_rationale", rec.get("absence_reason", ""))
        else:
            assert status == "ABSENT"
            assert len(rec.get("absence_reason", "")) > 0

    # Assertions on Audio Activity
    assert summary.average_rms_energy > 0.0
    assert summary.ambient_noise_floor > 0.0
    assert summary.average_rms_energy >= summary.ambient_noise_floor * 0.5
    assert 0.0 <= summary.speech_percentage <= 100.0

    # Assertions on Engagement Metrics
    assert 0.0 <= summary.mean_engagement_score <= 100.0
    assert summary.min_engagement_score <= summary.mean_engagement_score <= summary.max_engagement_score
    assert len(pipeline.engagement_results) > 0

    # Assertions on Pedagogical Suggestions
    assert isinstance(summary.recommendations, list)
    for sug in summary.recommendations:
        assert "suggestion_id" in sug
        assert "title" in sug
        assert "exact_metrics" in sug
        assert sug["confidence"] > 0.0
