"""
Presence-Based Attendance Manager for ClassPulse AI.
Implements relative percentage-based attendance thresholds with absolute overrides,
gap tolerance merging, duplicate prevention, and detailed absence explanations.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Optional
import numpy as np
import pandas as pd
from config import VISION_CONFIG


@dataclass
class ParticipantAttendanceRecord:
    """Structured attendance record for an anonymous tracking identifier."""
    session_id: str
    participant_id: str
    first_seen: float
    last_seen: float
    accumulated_presence_seconds: float
    attendance_status: str              # 'PRESENT' or 'ABSENT'
    presence_percentage: float          # Percentage of session duration observed
    interval_count: int
    required_presence_seconds: float = 0.0
    absence_reason: str = ""            # Detailed evaluation rationale (PRESENT or ABSENT)
    evaluation_rationale: str = ""      # Alias for evaluation rationale
    intervals: List[Tuple[float, float]] = field(default_factory=list)
    last_evaluated_at: float = 0.0

    def __post_init__(self):
        if not self.evaluation_rationale and self.absence_reason:
            self.evaluation_rationale = self.absence_reason
        elif not self.absence_reason and self.evaluation_rationale:
            self.absence_reason = self.evaluation_rationale

    def to_dict(self) -> dict:
        return {
            "session_id": self.session_id,
            "participant_id": self.participant_id,
            "first_seen": round(self.first_seen, 2),
            "last_seen": round(self.last_seen, 2),
            "accumulated_presence_seconds": round(self.accumulated_presence_seconds, 2),
            "required_presence_seconds": round(self.required_presence_seconds, 2),
            "attendance_status": self.attendance_status,
            "presence_percentage": round(self.presence_percentage, 1),
            "absence_reason": self.absence_reason,
            "evaluation_rationale": self.evaluation_rationale or self.absence_reason,
            "interval_count": self.interval_count,
            "last_evaluated_at": round(self.last_evaluated_at, 2),
        }


class AttendanceManager:
    """
    Computes presence-based attendance by recording observation timestamps,
    merging contiguous intervals within a gap tolerance, and preventing double-counting.
    Supports relative thresholds (% of session duration) and absolute overrides.
    """

    def __init__(
        self,
        session_id: str = "default_session",
        min_presence_ratio: float = VISION_CONFIG["default_min_presence_ratio"],
        min_presence_seconds: Optional[float] = None,
        use_absolute_override: bool = VISION_CONFIG["use_absolute_override"],
        gap_tolerance_seconds: float = VISION_CONFIG["gap_tolerance_seconds"],
        min_track_seconds: float = 0.0,
    ):
        self.session_id = session_id
        self.min_presence_ratio = float(min_presence_ratio)
        self.min_presence_seconds = float(min_presence_seconds) if min_presence_seconds is not None else None
        self.use_absolute_override = use_absolute_override
        self.gap_tolerance_seconds = gap_tolerance_seconds
        self.min_track_seconds = float(min_track_seconds)

        # Store raw observed intervals per participant: id -> [(start, end), ...]
        self.participant_intervals: Dict[str, List[Tuple[float, float]]] = {}
        # Store centroids per participant: id -> [(timestamp, (cx, cy)), ...]
        self.participant_centroids: Dict[str, List[Tuple[float, Tuple[int, int]]]] = {}
        # Track first seen and last seen
        self.first_seen: Dict[str, float] = {}
        self.last_seen: Dict[str, float] = {}

        self.session_start_time: Optional[float] = None
        self.session_end_time: Optional[float] = None

        # Tracking consolidation metrics
        self.raw_tracks_count: int = 0
        self.consolidated_count: int = 0
        self.discarded_count: int = 0

    def reset(self, session_id: Optional[str] = None) -> None:
        """Reset state for a new session."""
        if session_id:
            self.session_id = session_id
        self.participant_intervals.clear()
        self.participant_centroids.clear()
        self.first_seen.clear()
        self.last_seen.clear()
        self.session_start_time = None
        self.session_end_time = None
        self.raw_tracks_count = 0
        self.consolidated_count = 0
        self.discarded_count = 0

    def record_presence(
        self,
        participant_id: str,
        timestamp: float,
        duration: float = 0.0,
        centroid: Optional[Tuple[int, int]] = None,
        bbox: Optional[Tuple[int, int, int, int]] = None,
    ) -> None:
        """
        Record a presence observation at a specific timestamp.
        duration is the time elapsed since the previous frame.
        """
        if timestamp < 0:
            return

        # Track session boundaries
        if self.session_start_time is None or timestamp < self.session_start_time:
            self.session_start_time = timestamp
        if self.session_end_time is None or timestamp > self.session_end_time:
            self.session_end_time = timestamp

        # First and last seen
        if participant_id not in self.first_seen:
            self.first_seen[participant_id] = timestamp
        self.last_seen[participant_id] = max(self.last_seen.get(participant_id, timestamp), timestamp + duration)

        if centroid is not None:
            if participant_id not in self.participant_centroids:
                self.participant_centroids[participant_id] = []
            self.participant_centroids[participant_id].append((timestamp, centroid))

        start = timestamp
        end = timestamp + max(duration, 0.0)

        if participant_id not in self.participant_intervals:
            self.participant_intervals[participant_id] = [(start, end)]
        else:
            intervals = self.participant_intervals[participant_id]
            last_start, last_end = intervals[-1]

            # If the new observation is within gap_tolerance of the last interval, extend it
            if start <= last_end + self.gap_tolerance_seconds:
                intervals[-1] = (last_start, max(last_end, end))
            else:
                intervals.append((start, end))

    @staticmethod
    def merge_intervals(
        intervals: List[Tuple[float, float]],
        gap_tolerance: float = 0.0,
    ) -> List[Tuple[float, float]]:
        """Merges overlapping or adjacent intervals to prevent double-counting."""
        if not intervals:
            return []

        sorted_intervals = sorted(intervals, key=lambda x: x[0])
        merged: List[Tuple[float, float]] = [sorted_intervals[0]]

        for current_start, current_end in sorted_intervals[1:]:
            last_start, last_end = merged[-1]
            if current_start <= last_end + gap_tolerance:
                merged[-1] = (last_start, max(last_end, current_end))
            else:
                merged.append((current_start, current_end))

        return merged

    def compute_accumulated_presence(self, participant_id: str) -> float:
        """Calculates total accumulated presence duration in seconds without double-counting."""
        raw_intervals = self.participant_intervals.get(participant_id, [])
        if not raw_intervals:
            return 0.0

        merged = self.merge_intervals(raw_intervals, gap_tolerance=self.gap_tolerance_seconds)
        return sum(max(end - start, 0.0) for start, end in merged)

    def calculate_required_seconds(self, session_duration: float) -> float:
        """
        Determines the required presence seconds based on current configuration:
        relative ratio (e.g. 60% of session) or absolute seconds override.
        """
        if self.use_absolute_override and self.min_presence_seconds is not None:
            return self.min_presence_seconds
        return max(0.5, session_duration * self.min_presence_ratio)

    def validate_threshold(self, session_duration: float) -> Tuple[bool, str]:
        """
        Validates if the required threshold is mathematically achievable within the session.
        Returns (is_valid, warning_message).
        """
        required = self.calculate_required_seconds(session_duration)
        if required > session_duration:
            msg = (
                f"Threshold warning: Required presence ({required:.1f}s) exceeds total session "
                f"duration ({session_duration:.1f}s). All participants would be marked absent. "
                f"Consider lowering the threshold."
            )
            return False, msg
        return True, ""

    def consolidate_participants(
        self,
        min_track_seconds: Optional[float] = None,
    ) -> Dict[str, List[Tuple[float, float]]]:
        """
        Consolidates raw tracks:
        1. Merges tracks that belong to the same participant across brief gaps or duplicate detections.
        2. Discards tracks shorter than min_track_seconds (transient false positives).
        Updates self.raw_tracks_count, self.consolidated_count, self.discarded_count.
        """
        self.raw_tracks_count = len(self.participant_intervals)
        if not self.participant_intervals:
            self.consolidated_count = 0
            self.discarded_count = 0
            return {}

        effective_min_sec = min_track_seconds if min_track_seconds is not None else self.min_track_seconds

        # Sort candidate participant IDs by their first seen timestamp
        pids = sorted(self.participant_intervals.keys(), key=lambda p: self.first_seen.get(p, 0.0))

        consolidated: Dict[str, List[Tuple[float, float]]] = {}
        pid_centroids: Dict[str, List[Tuple[float, Tuple[int, int]]]] = {}

        for pid in pids:
            raw_ints = self.participant_intervals[pid]
            p_first = self.first_seen.get(pid, 0.0)
            p_last = self.last_seen.get(pid, 0.0)
            p_cents = self.participant_centroids.get(pid, [])

            matched_target = None

            # Try to match with an existing consolidated track
            for c_pid in list(consolidated.keys()):
                c_first = self.first_seen.get(c_pid, 0.0)
                c_last = self.last_seen.get(c_pid, 0.0)
                c_cents = pid_centroids.get(c_pid, [])

                # Condition 1: Temporal continuity in space (re-entry after gap)
                gap = p_first - c_last
                if 0.0 <= gap <= max(self.gap_tolerance_seconds * 2, 8.0):
                    if p_cents and c_cents:
                        c_last_pt = c_cents[-1][1]
                        p_first_pt = p_cents[0][1]
                        dist = float(np.sqrt((c_last_pt[0] - p_first_pt[0]) ** 2 + (c_last_pt[1] - p_first_pt[1]) ** 2))
                        if dist <= 130.0:  # Same physical seating area in classroom
                            matched_target = c_pid
                            break
                    elif gap <= self.gap_tolerance_seconds:
                        matched_target = c_pid
                        break

                # Condition 2: Simultaneous overlapping duplicate detections
                if p_first <= c_last and p_last >= c_first:
                    if p_cents and c_cents:
                        c_pt = c_cents[-1][1]
                        p_pt = p_cents[0][1]
                        dist = float(np.sqrt((c_pt[0] - p_pt[0]) ** 2 + (c_pt[1] - p_pt[1]) ** 2))
                        if dist <= 60.0:
                            matched_target = c_pid
                            break

            if matched_target is not None:
                consolidated[matched_target].extend(raw_ints)
                self.last_seen[matched_target] = max(self.last_seen[matched_target], p_last)
                if p_cents:
                    if matched_target not in pid_centroids:
                        pid_centroids[matched_target] = []
                    pid_centroids[matched_target].extend(p_cents)
            else:
                consolidated[pid] = list(raw_ints)
                if p_cents:
                    pid_centroids[pid] = list(p_cents)

        # Merge intervals for each consolidated participant
        for c_pid in consolidated:
            consolidated[c_pid] = self.merge_intervals(consolidated[c_pid], gap_tolerance=self.gap_tolerance_seconds)

        # Filter out short tracks if effective_min_sec > 0
        final_consolidated: Dict[str, List[Tuple[float, float]]] = {}
        discarded_count = 0

        for c_pid, intervals in consolidated.items():
            accumulated = sum(max(end - start, 0.0) for start, end in intervals)
            if effective_min_sec > 0.0 and accumulated < effective_min_sec:
                discarded_count += 1
            else:
                final_consolidated[c_pid] = intervals

        self.discarded_count = discarded_count
        self.consolidated_count = len(final_consolidated)
        return final_consolidated

    def evaluate_attendance(
        self,
        current_session_duration: Optional[float] = None,
        min_presence_ratio: Optional[float] = None,
        min_presence_seconds: Optional[float] = None,
        min_track_seconds: Optional[float] = None,
        consolidate: bool = True,
    ) -> List[ParticipantAttendanceRecord]:
        """
        Evaluates attendance for all tracked participants.
        Compares accumulated presence against the required presence seconds.
        Provides a clear human-readable explanation if absent.
        """
        # Determine total session duration
        if current_session_duration is not None and current_session_duration > 0:
            total_duration = current_session_duration
        elif self.session_start_time is not None and self.session_end_time is not None:
            total_duration = max(self.session_end_time - self.session_start_time, 0.1)
        else:
            total_duration = 0.1

        # Use overrides if supplied in method arguments
        ratio = min_presence_ratio if min_presence_ratio is not None else self.min_presence_ratio
        if min_presence_seconds is not None:
            required_sec = min_presence_seconds
        elif self.use_absolute_override and self.min_presence_seconds is not None:
            required_sec = self.min_presence_seconds
        else:
            required_sec = max(0.5, total_duration * ratio)

        if consolidate:
            eval_intervals = self.consolidate_participants(min_track_seconds=min_track_seconds)
        else:
            eval_intervals = self.participant_intervals
            self.raw_tracks_count = len(self.participant_intervals)
            self.consolidated_count = len(self.participant_intervals)
            self.discarded_count = 0

        records: List[ParticipantAttendanceRecord] = []
        now = self.session_end_time if self.session_end_time is not None else 0.0

        for pid, raw_intervals in eval_intervals.items():
            merged = self.merge_intervals(raw_intervals, gap_tolerance=self.gap_tolerance_seconds)
            accumulated = sum(max(end - start, 0.0) for start, end in merged)

            presence_pct = min(100.0, (accumulated / total_duration) * 100.0) if total_duration > 0 else 0.0
            short_by = max(0.0, required_sec - accumulated)

            # Non-empty rationale for every participant
            if accumulated < 0.08:
                is_present = False
                absence_reason = f"ABSENT: Transient detection ({accumulated:.2f}s < 0.08s minimum, short by {short_by:.1f}s)"
            elif accumulated >= required_sec:
                is_present = True
                absence_reason = f"PRESENT: observed {accumulated:.1f}s >= required {required_sec:.1f}s ({presence_pct:.1f}% of session)"
            else:
                is_present = False
                absence_reason = (
                    f"ABSENT: observed {accumulated:.1f}s < required {required_sec:.1f}s (short by {short_by:.1f}s, {presence_pct:.1f}% of session)"
                )

            status = "PRESENT" if is_present else "ABSENT"

            records.append(
                ParticipantAttendanceRecord(
                    session_id=self.session_id,
                    participant_id=pid,
                    first_seen=self.first_seen.get(pid, 0.0),
                    last_seen=self.last_seen.get(pid, 0.0),
                    accumulated_presence_seconds=accumulated,
                    required_presence_seconds=required_sec,
                    attendance_status=status,
                    presence_percentage=presence_pct,
                    interval_count=len(merged),
                    absence_reason=absence_reason,
                    evaluation_rationale=absence_reason,
                    intervals=merged,
                    last_evaluated_at=now,
                )
            )

        records.sort(key=lambda r: r.participant_id)
        return records

    def get_attendance_dataframe(
        self,
        current_session_duration: Optional[float] = None,
        min_track_seconds: Optional[float] = None,
    ) -> pd.DataFrame:
        """Returns structured attendance records as a clean pandas DataFrame."""
        records = self.evaluate_attendance(current_session_duration, min_track_seconds=min_track_seconds)
        if not records:
            return pd.DataFrame(
                columns=[
                    "participant_id",
                    "attendance_status",
                    "accumulated_presence_seconds",
                    "required_presence_seconds",
                    "presence_percentage",
                    "absence_reason",
                    "evaluation_rationale",
                    "first_seen",
                    "last_seen",
                    "interval_count",
                ]
            )

        data = [r.to_dict() for r in records]
        df = pd.DataFrame(data)
        cols = [
            "participant_id",
            "attendance_status",
            "accumulated_presence_seconds",
            "required_presence_seconds",
            "presence_percentage",
            "absence_reason",
            "evaluation_rationale",
            "first_seen",
            "last_seen",
            "interval_count",
        ]
        return df[cols]
