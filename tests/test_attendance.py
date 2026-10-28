"""
Unit tests for presence-based attendance logic in ClassPulse AI.
Covers threshold behavior, accumulation, duplicate prevention, gap tolerance,
relative percentage thresholds, and absence explanations.
"""

import pytest
from vision.attendance import AttendanceManager, ParticipantAttendanceRecord


def test_presence_duration_accumulation():
    """Validates that presence durations accumulate correctly across distinct intervals."""
    mgr = AttendanceManager(session_id="test_accum", min_presence_ratio=0.5, gap_tolerance_seconds=1.0)
    
    # 3 seconds from t=0 to t=3
    mgr.record_presence("Participant_01", timestamp=0.0, duration=1.0)
    mgr.record_presence("Participant_01", timestamp=1.0, duration=1.0)
    mgr.record_presence("Participant_01", timestamp=2.0, duration=1.0)
    
    # Gap from 3 to 10 (> gap_tolerance_seconds)
    # Another 2 seconds from t=10 to t=12
    mgr.record_presence("Participant_01", timestamp=10.0, duration=1.0)
    mgr.record_presence("Participant_01", timestamp=11.0, duration=1.0)

    total_presence = mgr.compute_accumulated_presence("Participant_01")
    assert pytest.approx(total_presence, 0.05) == 5.0


def test_duplicate_interval_prevention():
    """Validates that overlapping or duplicate interval recordings are never double-counted."""
    mgr = AttendanceManager(session_id="test_dup", min_presence_ratio=0.5, gap_tolerance_seconds=0.5)
    
    # Record overlapping intervals
    mgr.record_presence("Participant_01", timestamp=2.0, duration=4.0) # [2.0, 6.0]
    mgr.record_presence("Participant_01", timestamp=2.5, duration=2.0) # [2.5, 4.5] (subset)
    mgr.record_presence("Participant_01", timestamp=3.0, duration=3.0) # [3.0, 6.0] (partially overlapping)

    total_presence = mgr.compute_accumulated_presence("Participant_01")
    # Net span is exactly [2.0, 6.0] = 4.0 seconds
    assert pytest.approx(total_presence, 0.05) == 4.0


def test_relative_percentage_threshold():
    """Validates relative percentage-based attendance threshold against session duration."""
    # 60% of 10s session = 6.0s required
    mgr = AttendanceManager(session_id="test_rel", min_presence_ratio=0.60)
    
    # Participant_01 observed for 7.0s (> 6.0s) -> PRESENT
    for t in range(7):
        mgr.record_presence("Participant_01", timestamp=float(t), duration=1.0)

    # Participant_02 observed for 4.0s (< 6.0s) -> ABSENT
    for t in range(4):
        mgr.record_presence("Participant_02", timestamp=float(t), duration=1.0)

    records = mgr.evaluate_attendance(current_session_duration=10.0)
    status_map = {r.participant_id: r.attendance_status for r in records}
    reason_map = {r.participant_id: r.absence_reason for r in records}

    assert status_map["Participant_01"] == "PRESENT"
    assert status_map["Participant_02"] == "ABSENT"
    assert "required" in reason_map["Participant_02"]


def test_impossible_threshold_validation():
    """Validates that impossible thresholds (required > session length) trigger an explicit warning."""
    # Session length is 8s, but absolute override demands 20s
    mgr = AttendanceManager(
        session_id="test_impossible",
        min_presence_seconds=20.0,
        use_absolute_override=True,
    )
    is_valid, warning_msg = mgr.validate_threshold(session_duration=8.0)
    assert is_valid is False
    assert "exceeds total session duration" in warning_msg


def test_single_instant_detection_not_marked_present():
    """Validates that a single detected frame/instant does not grant attendance."""
    mgr = AttendanceManager(session_id="test_instant", min_presence_ratio=0.50)
    mgr.record_presence("Participant_01", timestamp=1.0, duration=0.0)
    
    records = mgr.evaluate_attendance(current_session_duration=30.0)
    assert len(records) == 1
    assert records[0].attendance_status == "ABSENT"
    assert "Transient" in records[0].absence_reason or "required" in records[0].absence_reason


def test_gap_tolerance_merging():
    """Validates that gaps below the tolerance threshold are smoothly bridged."""
    mgr = AttendanceManager(session_id="test_gap", min_presence_ratio=0.5, gap_tolerance_seconds=2.0)
    
    mgr.record_presence("Participant_01", timestamp=1.0, duration=1.0)
    mgr.record_presence("Participant_01", timestamp=3.5, duration=1.0)

    intervals = mgr.participant_intervals["Participant_01"]
    assert len(intervals) == 1
    assert intervals[0][0] == 1.0
    assert pytest.approx(intervals[0][1], 0.05) == 4.5
