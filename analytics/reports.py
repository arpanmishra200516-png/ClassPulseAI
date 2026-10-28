"""
Session Analytics & Report Generator for ClassPulse AI.
Aggregates attendance metrics, multimodal engagement trends, audio features,
and teaching suggestions into structured reports with CSV, JSON (with schema version),
and downloadable printable HTML/PDF reports.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional
from pathlib import Path
import json
import time
import pandas as pd
import numpy as np

from vision.attendance import ParticipantAttendanceRecord
from audio.activity import AudioWindowMetrics
from agent.engagement import EngagementScoreResult
from agent.recommendations import TeachingSuggestion
from agent.fusion import FusedObservation
from config import EXPORT_DIR, SESSIONS_DIR, APP_NAME, APP_VERSION

SCHEMA_VERSION = "1.1.0"


@dataclass
class SessionAnalyticsSummary:
    """Consolidated session analytics summary."""
    session_id: str
    created_at: str
    session_duration_seconds: float

    # Attendance statistics
    total_tracking_ids: int
    present_count: int
    absent_count: int
    attendance_rate_percentage: float
    attendance_records: List[Dict[str, any]] = field(default_factory=list)
    raw_tracks_count: int = 0
    consolidated_count: int = 0
    discarded_short_tracks_count: int = 0

    # Engagement statistics
    mean_engagement_score: float = 0.0
    median_engagement_score: float = 0.0
    min_engagement_score: float = 0.0
    max_engagement_score: float = 0.0
    std_engagement_score: float = 0.0
    high_engagement_percentage: float = 0.0
    moderate_engagement_percentage: float = 0.0
    low_engagement_percentage: float = 0.0
    engagement_timeline: List[Dict[str, any]] = field(default_factory=list)

    # Audio statistics
    average_rms_energy: Optional[float] = None
    ambient_noise_floor: Optional[float] = None
    speech_percentage: Optional[float] = None
    audio_status: str = "NO_AUDIO_TRACK"
    has_audio_track: bool = True
    audio_extraction_error: Optional[str] = None
    audio_metrics: List[Dict[str, any]] = field(default_factory=list)

    # Pedagogical recommendations
    recommendations_count: int = 0
    recommendations: List[Dict[str, any]] = field(default_factory=list)

    # System and privacy metadata
    metadata: Dict[str, any] = field(default_factory=dict)
    schema_version: str = SCHEMA_VERSION

    def to_dict(self) -> dict:
        return {
            "schema_version": self.schema_version,
            "session_id": self.session_id,
            "created_at": self.created_at,
            "session_duration_seconds": round(self.session_duration_seconds, 2),
            "attendance_summary": {
                "total_temporary_tracking_ids": self.total_tracking_ids,
                "present_count": self.present_count,
                "absent_count": self.absent_count,
                "attendance_rate_percentage": round(self.attendance_rate_percentage, 1),
                "raw_tracks_count": self.raw_tracks_count,
                "consolidated_count": self.consolidated_count,
                "discarded_short_tracks_count": self.discarded_short_tracks_count,
                "records": self.attendance_records,
            },
            "engagement_summary": {
                "mean_score": round(self.mean_engagement_score, 1) if self.mean_engagement_score is not None else None,
                "median_score": round(self.median_engagement_score, 1) if self.median_engagement_score is not None else None,
                "min_score": round(self.min_engagement_score, 1) if self.min_engagement_score is not None else None,
                "max_score": round(self.max_engagement_score, 1) if self.max_engagement_score is not None else None,
                "std_score": round(self.std_engagement_score, 2) if self.std_engagement_score is not None else None,
                "percentage_time_high": round(self.high_engagement_percentage, 1) if self.high_engagement_percentage is not None else None,
                "percentage_time_moderate": round(self.moderate_engagement_percentage, 1) if self.moderate_engagement_percentage is not None else None,
                "percentage_time_low": round(self.low_engagement_percentage, 1) if self.low_engagement_percentage is not None else None,
            },
            "audio_summary": {
                "average_rms": round(self.average_rms_energy, 5) if self.average_rms_energy is not None else None,
                "ambient_noise_floor": round(self.ambient_noise_floor, 5) if self.ambient_noise_floor is not None else None,
                "speech_percentage": round(self.speech_percentage, 1) if self.speech_percentage is not None else None,
                "audio_status": self.audio_status,
                "has_audio_track": self.has_audio_track,
                "audio_extraction_error": self.audio_extraction_error,
            },
            "engagement_timeline": self.engagement_timeline,
            "audio_metrics": self.audio_metrics,
            "recommendations": self.recommendations,
            "metadata": self.metadata,
        }

    def to_json(self, indent: int = 2) -> str:
        """Returns JSON representation."""
        return json.dumps(self.to_dict(), indent=indent)


class SessionResult:
    """
    Unified Single Source of Truth for ClassPulse AI session state.
    Holds attendance records, audio metrics, engagement timeline, suggestions, and metadata.
    Supports both dict and attribute access.
    """
    def __init__(
        self,
        session_id: str,
        session_duration: float,
        created_at: str,
        attendance_data: dict,
        audio_data: dict,
        engagement_data: dict,
        suggestions: list,
        metadata: dict,
        summary: Optional[SessionAnalyticsSummary] = None,
    ):
        self.session_id = session_id
        self.session_duration = session_duration
        self.session_duration_seconds = session_duration
        self.created_at = created_at
        self.attendance = attendance_data
        self.attendance_summary = attendance_data.get("summary", {})
        self.audio = audio_data
        self.audio_summary = audio_data.get("summary", {})
        self.audio_metrics = audio_data.get("metrics", [])
        self.engagement = engagement_data
        self.engagement_summary = engagement_data.get("summary", {})
        self.engagement_timeline = engagement_data.get("timeline", [])
        self.suggestions = suggestions
        self.recommendations = suggestions
        self.metadata = metadata
        self.summary = summary

    def __getitem__(self, key: str):
        if hasattr(self, key):
            return getattr(self, key)
        if key == "engagement_timeline":
            return self.engagement.get("timeline", [])
        if key == "audio_metrics":
            return self.audio.get("metrics", [])
        if key in self.metadata:
            return self.metadata[key]
        raise KeyError(key)

    def get(self, key: str, default=None):
        return getattr(self, key, default)

    def to_dict(self) -> dict:
        if self.summary:
            d = self.summary.to_dict()
        else:
            d = {
                "session_id": self.session_id,
                "session_duration_seconds": self.session_duration,
                "created_at": self.created_at,
                "attendance_summary": self.attendance_summary,
                "audio_summary": self.audio_summary,
                "engagement_summary": self.engagement_summary,
                "recommendations": self.suggestions,
                "metadata": self.metadata,
            }
        d["engagement_timeline"] = self.engagement.get("timeline", [])
        d["audio_metrics"] = self.audio.get("metrics", [])
        return d

    def to_json(self, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent)

    @classmethod
    def from_summary(cls, summary: SessionAnalyticsSummary, pipeline=None) -> "SessionResult":
        att_records = summary.attendance_records
        att_summary = {
            "total_temporary_tracking_ids": summary.total_tracking_ids,
            "present_count": summary.present_count,
            "absent_count": summary.absent_count,
            "attendance_rate_percentage": summary.attendance_rate_percentage,
            "raw_tracks_count": getattr(summary, "raw_tracks_count", summary.total_tracking_ids),
            "consolidated_count": getattr(summary, "consolidated_count", summary.total_tracking_ids),
            "discarded_short_tracks_count": getattr(summary, "discarded_short_tracks_count", 0),
        }
        attendance_data = {
            "records": att_records,
            "summary": att_summary,
            "raw_tracks_count": att_summary["raw_tracks_count"],
            "consolidated_count": att_summary["consolidated_count"],
            "discarded_count": att_summary["discarded_short_tracks_count"],
        }

        has_audio = summary.has_audio_track and summary.audio_status == "ANALYZED"
        audio_data = {
            "status": summary.audio_status,
            "has_audio": has_audio,
            "speech_percentage": summary.speech_percentage if has_audio else None,
            "ambient_noise_floor": summary.ambient_noise_floor if has_audio else None,
            "average_rms": summary.average_rms_energy if has_audio else None,
            "extraction_error": summary.audio_extraction_error,
            "metrics": summary.audio_metrics,
            "summary": {
                "speech_percentage": summary.speech_percentage if has_audio else None,
                "ambient_noise_floor": summary.ambient_noise_floor if has_audio else None,
                "average_rms": summary.average_rms_energy if has_audio else None,
                "audio_status": summary.audio_status,
            },
        }

        has_timeline = len(summary.engagement_timeline) > 0
        valid_scores = [r["score"] for r in summary.engagement_timeline if r.get("level") != "NO_DATA"] if has_timeline else []
        engagement_data = {
            "timeline": summary.engagement_timeline,
            "has_timeline": has_timeline,
            "mean_score": summary.mean_engagement_score if has_timeline and valid_scores else None,
            "median_score": summary.median_engagement_score if has_timeline and valid_scores else None,
            "min_score": summary.min_engagement_score if has_timeline and valid_scores else None,
            "max_score": summary.max_engagement_score if has_timeline and valid_scores else None,
            "std_score": summary.std_engagement_score if has_timeline and valid_scores else None,
            "active_modality": summary.metadata.get("modality_mode", "BIMODAL" if has_audio else "VISION_ONLY"),
            "summary": {
                "mean_score": summary.mean_engagement_score if has_timeline and valid_scores else None,
                "median_score": summary.median_engagement_score if has_timeline and valid_scores else None,
            },
        }

        return cls(
            session_id=summary.session_id,
            session_duration=summary.session_duration_seconds,
            created_at=summary.created_at,
            attendance_data=attendance_data,
            audio_data=audio_data,
            engagement_data=engagement_data,
            suggestions=summary.recommendations,
            metadata=summary.metadata,
            summary=summary,
        )

    @classmethod
    def from_dict(cls, data: dict) -> "SessionResult":
        session_id = data.get("session_id", "unknown_session")
        duration = float(data.get("session_duration_seconds", 0.0))
        created_at = data.get("created_at", "")
        att_summary = data.get("attendance_summary", {})
        att_records = att_summary.get("records", data.get("attendance_records", []))
        
        aud_summary = data.get("audio_summary", {})
        aud_status = aud_summary.get("audio_status", "UNAVAILABLE")
        has_audio = aud_status == "ANALYZED"
        sp_pct = aud_summary.get("speech_percentage") if has_audio else None
        noise_fl = aud_summary.get("ambient_noise_floor") if has_audio else None
        rms = aud_summary.get("average_rms") if has_audio else None

        eng_summary = data.get("engagement_summary", {})
        timeline = data.get("engagement_timeline", [])
        has_timeline = len(timeline) > 0
        mean_eng = eng_summary.get("mean_score") if has_timeline else None

        attendance_data = {
            "records": att_records,
            "summary": att_summary,
            "raw_tracks_count": att_summary.get("raw_tracks_count", len(att_records)),
            "consolidated_count": att_summary.get("consolidated_count", len(att_records)),
            "discarded_count": att_summary.get("discarded_short_tracks_count", 0),
        }

        audio_data = {
            "status": aud_status,
            "has_audio": has_audio,
            "speech_percentage": sp_pct,
            "ambient_noise_floor": noise_fl,
            "average_rms": rms,
            "extraction_error": aud_summary.get("audio_extraction_error") or data.get("audio_extraction_error"),
            "metrics": data.get("audio_metrics", []),
            "summary": aud_summary,
        }

        engagement_data = {
            "timeline": timeline,
            "has_timeline": has_timeline,
            "mean_score": mean_eng,
            "median_score": eng_summary.get("median_score"),
            "min_score": eng_summary.get("min_score"),
            "max_score": eng_summary.get("max_score"),
            "std_score": eng_summary.get("std_score"),
            "active_modality": data.get("metadata", {}).get("modality_mode", "BIMODAL" if has_audio else "VISION_ONLY"),
            "summary": eng_summary,
        }

        return cls(
            session_id=session_id,
            session_duration=duration,
            created_at=created_at,
            attendance_data=attendance_data,
            audio_data=audio_data,
            engagement_data=engagement_data,
            suggestions=data.get("recommendations", []),
            metadata=data.get("metadata", {}),
        )


class SessionReportGenerator:
    """
    Creates comprehensive summaries and exports in CSV, JSON, and printable HTML/PDF formats.
    """

    @classmethod
    def generate_summary(
        cls,
        session_id: str,
        session_duration: float,
        attendance_records: List[ParticipantAttendanceRecord],
        engagement_results: List[EngagementScoreResult],
        audio_metrics: Optional[List[AudioWindowMetrics]] = None,
        suggestions: Optional[List[TeachingSuggestion]] = None,
        raw_tracks_count: Optional[int] = None,
        consolidated_count: Optional[int] = None,
        discarded_short_tracks_count: int = 0,
        audio_status: Optional[str] = None,
        has_audio_track: bool = True,
        audio_extraction_error: Optional[str] = None,
    ) -> SessionAnalyticsSummary:
        """Generates a complete SessionAnalyticsSummary object from session metrics."""
        total_tracked = len(attendance_records)
        present = sum(1 for r in attendance_records if r.attendance_status == "PRESENT")
        absent = total_tracked - present
        att_rate = (present / total_tracked * 100.0) if total_tracked > 0 else 0.0
        att_data = [r.to_dict() for r in attendance_records]

        valid_scores = [r.score for r in engagement_results if r.level != "NO_DATA"]
        if valid_scores:
            mean_score = float(np.mean(valid_scores))
            med_score = float(np.median(valid_scores))
            min_score = float(np.min(valid_scores))
            max_score = float(np.max(valid_scores))
            std_score = float(np.std(valid_scores))

            total_valid = len(valid_scores)
            pct_high = (sum(1 for s in valid_scores if s >= 68.0) / total_valid) * 100.0
            pct_mod = (sum(1 for s in valid_scores if 40.0 <= s < 68.0) / total_valid) * 100.0
            pct_low = (sum(1 for s in valid_scores if s < 40.0) / total_valid) * 100.0
        else:
            mean_score = med_score = min_score = max_score = std_score = 0.0
            pct_high = pct_mod = pct_low = 0.0

        if audio_metrics and len(audio_metrics) > 0:
            avg_rms = float(np.mean([m.rms_energy for m in audio_metrics]))
            avg_noise = float(np.mean([m.noise_floor for m in audio_metrics]))
            speech_windows = sum(1 for m in audio_metrics if m.is_speech)
            speech_pct = (speech_windows / len(audio_metrics)) * 100.0
            final_audio_status = "ANALYZED"
        else:
            avg_rms = None
            avg_noise = None
            speech_pct = None
            final_audio_status = audio_status or ("NO_AUDIO_TRACK" if not has_audio_track else "UNAVAILABLE")

        rec_data = [s.to_dict() for s in (suggestions or [])]
        eng_timeline_data = [r.to_dict() for r in (engagement_results or [])]
        audio_metrics_data = [m.to_dict() for m in (audio_metrics or [])]

        modality_mode = "BIMODAL" if final_audio_status == "ANALYZED" else "VISION_ONLY"

        metadata = {
            "application": APP_NAME,
            "version": APP_VERSION,
            "schema_version": SCHEMA_VERSION,
            "modality_mode": modality_mode,
            "note": (
                "Tracking identifiers are anonymous session labels, not verified real-world identities. "
                "Engagement scores are heuristic approximations for teacher guidance and reflection."
            ),
        }

        return SessionAnalyticsSummary(
            session_id=session_id,
            created_at=time.strftime("%Y-%m-%d %H:%M:%S"),
            session_duration_seconds=session_duration,
            total_tracking_ids=total_tracked,
            present_count=present,
            absent_count=absent,
            attendance_rate_percentage=att_rate,
            attendance_records=att_data,
            raw_tracks_count=raw_tracks_count if raw_tracks_count is not None else total_tracked,
            consolidated_count=consolidated_count if consolidated_count is not None else total_tracked,
            discarded_short_tracks_count=discarded_short_tracks_count,
            mean_engagement_score=mean_score,
            median_engagement_score=med_score,
            min_engagement_score=min_score,
            max_engagement_score=max_score,
            std_engagement_score=std_score,
            high_engagement_percentage=pct_high,
            moderate_engagement_percentage=pct_mod,
            low_engagement_percentage=pct_low,
            engagement_timeline=eng_timeline_data,
            average_rms_energy=avg_rms,
            ambient_noise_floor=avg_noise,
            speech_percentage=speech_pct,
            audio_status=final_audio_status,
            has_audio_track=has_audio_track,
            audio_extraction_error=audio_extraction_error,
            audio_metrics=audio_metrics_data,
            recommendations_count=len(rec_data),
            recommendations=rec_data,
            metadata=metadata,
            schema_version=SCHEMA_VERSION,
        )

    @classmethod
    def export_attendance_csv(cls, attendance_records: List[ParticipantAttendanceRecord]) -> str:
        """Returns CSV string for attendance records."""
        if not attendance_records:
            df = pd.DataFrame(columns=["participant_id", "attendance_status", "accumulated_presence_seconds"])
        else:
            df = pd.DataFrame([r.to_dict() for r in attendance_records])
        return df.to_csv(index=False)

    @classmethod
    def export_engagement_timeline_csv(cls, engagement_results: List[EngagementScoreResult]) -> str:
        """Returns CSV string for engagement timeline."""
        if not engagement_results:
            df = pd.DataFrame(columns=["bin_start", "bin_end", "score", "level"])
        else:
            df = pd.DataFrame([r.to_dict() for r in engagement_results])
        return df.to_csv(index=False)

    @classmethod
    def export_recommendations_csv(cls, suggestions: List[TeachingSuggestion]) -> str:
        """Returns CSV string for pedagogical recommendations."""
        if not suggestions:
            df = pd.DataFrame(columns=["suggestion_id", "title", "observation", "suggestion"])
        else:
            df = pd.DataFrame([s.to_dict() for s in suggestions])
        return df.to_csv(index=False)

    @classmethod
    def generate_html_report(cls, summary: SessionAnalyticsSummary) -> str:
        """
        Generates a standalone, beautifully formatted, printable HTML report.
        Includes complete session KPIs, attendance records, audio stats, recommendations,
        methodology, and ethical privacy notes.
        """
        att_rows = ""
        for rec in summary.attendance_records:
            status_style = (
                "color: #166534; background-color: #dcfce7; padding: 3px 8px; border-radius: 4px; font-weight: bold;"
                if rec["attendance_status"] == "PRESENT"
                else "color: #991b1b; background-color: #fee2e2; padding: 3px 8px; border-radius: 4px; font-weight: bold;"
            )
            reason_text = rec.get("absence_reason", "") or "Met presence criteria"
            att_rows += f"""
            <tr>
                <td><b>{rec['participant_id']}</b></td>
                <td><span style="{status_style}">{rec['attendance_status']}</span></td>
                <td>{rec['accumulated_presence_seconds']:.1f}s</td>
                <td>{rec.get('required_presence_seconds', 0.0):.1f}s</td>
                <td>{rec['presence_percentage']:.1f}%</td>
                <td><small>{reason_text}</small></td>
            </tr>
            """

        rec_blocks = ""
        for r in summary.recommendations:
            rec_blocks += f"""
            <div style="border-left: 4px solid #3b82f6; background-color: #f8fafc; padding: 12px; margin-bottom: 12px; border-radius: 4px;">
                <div style="font-weight: bold; font-size: 1.05em; color: #1e3a8a;">
                    [{r['priority']}] {r['title']} 
                    <span style="font-size: 0.85em; color: #64748b; font-weight: normal;">(t = {r['interval_start']:.1f}s - {r['interval_end']:.1f}s | Confidence: {r.get('confidence', 0.85):.0%})</span>
                </div>
                <div style="margin-top: 4px;"><b>Observation:</b> {r['observation']}</div>
                <div style="margin-top: 4px;"><b>Pedagogical Suggestion:</b> {r['suggestion']}</div>
                <div style="margin-top: 4px; color: #475569; font-size: 0.9em;"><b>Rationale:</b> {r['pedagogical_rationale']}</div>
                <div style="margin-top: 2px; color: #64748b; font-size: 0.85em; font-style: italic;">⚠️ {r['limitation_note']}</div>
            </div>
            """

        if not rec_blocks:
            rec_blocks = "<p>No active pedagogical interventions triggered during this session.</p>"

        speech_text = f"{summary.speech_percentage:.1f}%" if summary.speech_percentage is not None else "N/A"
        noise_text = f"{summary.ambient_noise_floor:.4f}" if summary.ambient_noise_floor is not None else "N/A"

        html = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <title>ClassPulse AI Report — {summary.session_id}</title>
    <style>
        body {{
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Helvetica, Arial, sans-serif;
            color: #1e293b;
            line-height: 1.5;
            padding: 30px;
            max-width: 900px;
            margin: auto;
        }}
        h1, h2, h3 {{ color: #1e3a8a; }}
        .header {{ border-bottom: 2px solid #e2e8f0; padding-bottom: 15px; margin-bottom: 20px; }}
        .kpi-container {{ display: flex; gap: 15px; margin-bottom: 25px; flex-wrap: wrap; }}
        .kpi-card {{
            flex: 1; min-width: 140px; background: #f1f5f9; padding: 12px;
            border-radius: 6px; text-align: center; border: 1px solid #cbd5e1;
        }}
        .kpi-num {{ font-size: 1.5em; font-weight: bold; color: #0f172a; margin-top: 4px; }}
        .kpi-label {{ font-size: 0.8em; color: #64748b; text-transform: uppercase; }}
        table {{ width: 100%; border-collapse: collapse; margin-top: 10px; margin-bottom: 25px; }}
        th, td {{ border: 1px solid #e2e8f0; padding: 8px 12px; text-align: left; font-size: 0.9em; }}
        th {{ background-color: #f8fafc; color: #334155; }}
        .disclaimer {{
            background-color: #fefce8; border: 1px solid #fef08a; padding: 12px;
            border-radius: 6px; font-size: 0.85em; color: #713f12; margin-top: 30px;
        }}
        @media print {{
            body {{ padding: 0; }}
            .no-print {{ display: none; }}
        }}
    </style>
</head>
<body>
    <div class="header">
        <h1>🎓 {APP_NAME} — Session Analytics Report</h1>
        <div><strong>Session Identifier:</strong> {summary.session_id}</div>
        <div><strong>Generated At:</strong> {summary.created_at} | <strong>Duration:</strong> {summary.session_duration_seconds:.1f}s | <strong>Schema Version:</strong> {summary.schema_version}</div>
    </div>

    <div class="kpi-container">
        <div class="kpi-card">
            <div class="kpi-label">Tracked Participants</div>
            <div class="kpi-num">{summary.total_tracking_ids}</div>
        </div>
        <div class="kpi-card">
            <div class="kpi-label">Confirmed Present</div>
            <div class="kpi-num">{summary.present_count}</div>
        </div>
        <div class="kpi-card">
            <div class="kpi-label">Attendance Rate</div>
            <div class="kpi-num">{summary.attendance_rate_percentage:.1f}%</div>
        </div>
        <div class="kpi-card">
            <div class="kpi-label">Mean Engagement</div>
            <div class="kpi-num">{summary.mean_engagement_score:.1f}/100</div>
        </div>
        <div class="kpi-card">
            <div class="kpi-label">Speech Activity</div>
            <div class="kpi-num">{speech_text}</div>
        </div>
        <div class="kpi-card">
            <div class="kpi-label">Noise Floor (RMS)</div>
            <div class="kpi-num">{noise_text}</div>
        </div>
    </div>

    <h2>1. Attendance Records (Accumulated Presence)</h2>
    <table>
        <thead>
            <tr>
                <th>Participant ID</th>
                <th>Status</th>
                <th>Observed Presence</th>
                <th>Required Presence</th>
                <th>Session %</th>
                <th>Evaluation Context</th>
            </tr>
        </thead>
        <tbody>
            {att_rows}
        </tbody>
    </table>

    <h2>2. Pedagogical Suggestions</h2>
    {rec_blocks}

    <h2>3. Engagement Summary</h2>
    <p>
        <strong>Mean Engagement Index:</strong> {summary.mean_engagement_score:.1f} / 100 &nbsp;|&nbsp;
        <strong>Median:</strong> {summary.median_engagement_score:.1f} &nbsp;|&nbsp;
        <strong>Std Dev:</strong> {summary.std_engagement_score:.2f} &nbsp;|&nbsp;
        <strong>High (>68):</strong> {summary.high_engagement_percentage:.1f}% &nbsp;|&nbsp;
        <strong>Moderate (40-68):</strong> {summary.moderate_engagement_percentage:.1f}% &nbsp;|&nbsp;
        <strong>Low (<40):</strong> {summary.low_engagement_percentage:.1f}%
    </p>

    <div class="disclaimer">
        <strong>Privacy, Academic & Ethical Safeguards:</strong><br>
        All participants are tracked anonymously using transient session labels (e.g., <code>Participant_01</code>).
        No biometric identity recognition or persistent facial embeddings are computed or saved.
        Engagement scores represent an explainable heuristic fusion of visible presence, physical micro-movement,
        and vocal acoustic activity. They do <em>not</em> constitute evaluation of learning comprehension, intellectual ability,
        or academic grades. All media processing occurred 100% locally.
    </div>
</body>
</html>
        """
        return html

    @classmethod
    def save_session_to_disk(cls, summary: SessionAnalyticsSummary, target_dir: Optional[Path] = None) -> Path:
        """Saves session summary JSON file into target_dir or config.SESSIONS_DIR."""
        import config
        save_dir = target_dir or config.SESSIONS_DIR
        save_dir.mkdir(parents=True, exist_ok=True)
        filepath = save_dir / f"{summary.session_id}.json"
        with open(filepath, "w", encoding="utf-8") as f:
            f.write(summary.to_json())
        return filepath

    @classmethod
    def load_session_from_disk(cls, session_id: str, target_dir: Optional[Path] = None) -> Optional[dict]:
        """Loads a session summary JSON from disk."""
        import config
        search_dir = target_dir or config.SESSIONS_DIR
        filepath = search_dir / f"{session_id}.json"
        if not filepath.exists():
            return None
        with open(filepath, "r", encoding="utf-8") as f:
            return json.load(f)
