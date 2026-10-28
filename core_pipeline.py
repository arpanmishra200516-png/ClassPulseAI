"""
Core Pipeline for ClassPulse AI.
Orchestrates vision detection, face tracking, presence-based attendance,
audio extraction and analysis, multimodal fusion, engagement scoring,
and session analytics generation.
Processes video in streaming fashion with frame skipping and defensive error handling.
"""

from typing import Dict, List, Optional, Callable, Tuple
from pathlib import Path
import cv2
import numpy as np

from vision.detector import FaceDetector, DetectionResult
from vision.tracker import FaceTracker, TrackedParticipant
from vision.attendance import AttendanceManager, ParticipantAttendanceRecord
from audio.processor import AudioProcessor
from audio.activity import AudioActivityAnalyzer, AudioWindowMetrics
from agent.fusion import MultimodalFusion, FusedObservation
from agent.engagement import EngagementEstimator, EngagementScoreResult
from agent.recommendations import RecommendationEngine, TeachingSuggestion
from analytics.reports import SessionReportGenerator, SessionAnalyticsSummary
from config import VISION_CONFIG, AUDIO_CONFIG, AGENT_CONFIG, RECOMMENDATION_CONFIG


class ClassPulsePipeline:
    """
    Main multimodal pipeline coordinating all sensory and reasoning modules.
    Handles missing modalities, edge cases, and real-world inputs defensively.
    """

    def __init__(
        self,
        session_id: str = "session_001",
        min_presence_ratio: float = VISION_CONFIG["default_min_presence_ratio"],
        min_presence_seconds: Optional[float] = None,
        use_absolute_override: bool = VISION_CONFIG.get("use_absolute_override", False),
        gap_tolerance_seconds: float = VISION_CONFIG["gap_tolerance_seconds"],
        expected_participants: int = 1,
        min_track_seconds: float = VISION_CONFIG.get("min_track_seconds", 2.0),
    ):
        self.session_id = session_id
        self.min_presence_ratio = float(min_presence_ratio)
        self.min_presence_seconds = float(min_presence_seconds) if min_presence_seconds is not None else None
        self.use_absolute_override = use_absolute_override
        self.gap_tolerance_seconds = gap_tolerance_seconds
        self.expected_participants = max(1, expected_participants)
        self.min_track_seconds = float(min_track_seconds)

        # Initialize sub-modules
        self.detector = FaceDetector(min_confidence=VISION_CONFIG["min_detection_confidence"])
        self.tracker = FaceTracker(
            max_disappeared=VISION_CONFIG["tracker_max_disappeared"],
            max_distance=VISION_CONFIG["tracker_max_distance"],
            reentry_window_seconds=VISION_CONFIG.get("reentry_window_seconds", 12.0),
            min_track_seconds=self.min_track_seconds,
        )
        self.attendance_mgr = AttendanceManager(
            session_id=session_id,
            min_presence_ratio=self.min_presence_ratio,
            min_presence_seconds=self.min_presence_seconds,
            use_absolute_override=self.use_absolute_override,
            gap_tolerance_seconds=gap_tolerance_seconds,
            min_track_seconds=self.min_track_seconds,
        )
        self.audio_processor = AudioProcessor(
            target_sample_rate=AUDIO_CONFIG["target_sample_rate"],
            window_duration=AUDIO_CONFIG["window_duration_seconds"],
            hop_duration=AUDIO_CONFIG["hop_duration_seconds"],
        )
        self.audio_analyzer = AudioActivityAnalyzer(
            min_energy_threshold=AUDIO_CONFIG["min_energy_threshold"],
            speech_multiplier=AUDIO_CONFIG["speech_energy_multiplier"],
            noise_percentile=AUDIO_CONFIG["ambient_noise_percentile"],
            vad_hangover_windows=AUDIO_CONFIG.get("vad_hangover_windows", 2),
        )
        self.fusion = MultimodalFusion(bin_size_seconds=AGENT_CONFIG["time_bin_seconds"])
        self.engagement_estimator = EngagementEstimator(
            weights=AGENT_CONFIG["weights"],
            expected_participants=self.expected_participants,
        )
        self.rec_engine = RecommendationEngine(
            sustained_low_duration=RECOMMENDATION_CONFIG["sustained_low_engagement_duration"],
            low_engagement_threshold=RECOMMENDATION_CONFIG["low_engagement_threshold"],
            sustained_high_noise_duration=RECOMMENDATION_CONFIG["sustained_high_noise_duration"],
            high_noise_threshold=RECOMMENDATION_CONFIG["high_noise_threshold"],
            monologue_duration=RECOMMENDATION_CONFIG["monologue_duration"],
            engagement_drop_delta=RECOMMENDATION_CONFIG["engagement_drop_delta"],
            min_suggestion_duration=RECOMMENDATION_CONFIG.get("min_suggestion_duration", 5.0),
        )

        # Session state
        self.vision_timeline: List[Dict[str, any]] = []
        self.audio_metrics: List[AudioWindowMetrics] = []
        self.fused_observations: List[FusedObservation] = []
        self.engagement_results: List[EngagementScoreResult] = []
        self.suggestions: List[TeachingSuggestion] = []
        self.session_duration: float = 0.0
        self.audio_status: str = "UNAVAILABLE"
        self.has_audio_track: bool = False
        self.audio_extraction_error: Optional[str] = None
        self.audio_extraction_path: Optional[str] = None

    def reset(self, new_session_id: Optional[str] = None) -> None:
        """Resets the pipeline for a new session."""
        if new_session_id:
            self.session_id = new_session_id
        self.tracker.reset()
        self.attendance_mgr.reset(self.session_id)
        self.vision_timeline.clear()
        self.audio_metrics.clear()
        self.fused_observations.clear()
        self.engagement_results.clear()
        self.suggestions.clear()
        self.session_duration = 0.0
        self.audio_status = "UNAVAILABLE"
        self.has_audio_track = False
        self.audio_extraction_error = None
        self.audio_extraction_path = None
        self.vision_timeline.clear()
        self.audio_metrics.clear()
        self.fused_observations.clear()
        self.engagement_results.clear()
        self.suggestions.clear()
        self.session_duration = 0.0

    def process_live_frame(
        self,
        frame: np.ndarray,
        timestamp: float,
        frame_interval: float = 0.033,
    ) -> Tuple[np.ndarray, List[TrackedParticipant]]:
        """
        Processes a single video frame. Handles low light, missing faces, and multi-face tracks.
        """
        if frame is None or not isinstance(frame, np.ndarray) or frame.size == 0:
            return frame, []

        self.session_duration = max(self.session_duration, timestamp)

        # Detect faces with MediaPipe / Haar cascade
        detections = self.detector.detect(frame)

        # Update hybrid IoU/centroid tracker
        tracks = self.tracker.update(detections, timestamp)

        # Record presence for attendance
        for t in tracks:
            self.attendance_mgr.record_presence(
                participant_id=t.tracking_id,
                timestamp=timestamp,
                duration=frame_interval,
                centroid=t.centroid,
                bbox=t.bbox,
            )

        # Append to vision timeline
        self.vision_timeline.append({
            "timestamp": timestamp,
            "tracks": tracks,
            "detections_count": len(detections),
        })

        # Evaluate attendance status map
        att_eval = self.attendance_mgr.evaluate_attendance(
            self.session_duration,
            min_track_seconds=self.min_track_seconds,
            consolidate=True,
        )
        status_map = {r.participant_id: r.attendance_status for r in att_eval}

        # Draw annotations
        annotated = FaceTracker.draw_tracks(frame, tracks, status_map)
        return annotated, tracks

    def process_audio_file(self, audio_or_video_path: str) -> List[AudioWindowMetrics]:
        """Processes an audio file or extracts audio from a video file."""
        try:
            samples, sr, duration, status, msg, path_worked = self.audio_processor.load_audio_with_status(audio_or_video_path)
            self.session_duration = max(self.session_duration, duration)
            self.audio_status = status
            self.has_audio_track = (status == "ANALYZED")
            self.audio_extraction_error = msg if status == "EXTRACTION_FAILED" else None
            self.audio_extraction_path = path_worked

            if samples is not None and len(samples) > 0:
                segments = self.audio_processor.segment_audio(samples, sr)
                self.audio_metrics = self.audio_analyzer.analyze_segments(segments)
            else:
                self.audio_metrics = []
        except Exception as e:
            print(f"[ClassPulsePipeline] Audio processing note: {e}")
            self.audio_metrics = []
            self.audio_status = "EXTRACTION_FAILED"
            self.has_audio_track = False
            self.audio_extraction_error = str(e)

        return self.audio_metrics

    def process_video_file(
        self,
        video_path: str,
        frame_skip: int = 2,
        progress_callback: Optional[Callable[[float, str], None]] = None,
    ) -> bool:
        """
        Processes a video file in streaming fashion with frame skipping for real-time speed.
        Extracts embedded audio if available.
        """
        try:
            cap = cv2.VideoCapture(video_path)
            if not cap.isOpened():
                return False

            fps = cap.get(cv2.CAP_PROP_FPS)
            if fps <= 0 or np.isnan(fps):
                fps = 30.0

            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            if total_frames <= 0:
                total_frames = 100

            frame_interval = 1.0 / fps
            frame_idx = 0

            # Step 1: Process visual frames in streaming fashion
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret or frame is None:
                    break

                if frame_idx % frame_skip == 0:
                    timestamp = frame_idx * frame_interval
                    self.process_live_frame(frame, timestamp, frame_interval * frame_skip)

                    if progress_callback and frame_idx % 24 == 0:
                        pct = min(0.70, (frame_idx / total_frames) * 0.70)
                        progress_callback(pct, f"Analyzing video frames: {frame_idx}/{total_frames}...")

                frame_idx += 1

            cap.release()

            # Step 2: Extract and analyze audio track if present
            if progress_callback:
                progress_callback(0.75, "Extracting and analyzing audio track...")
            self.process_audio_file(video_path)

            # Step 3: Run fusion and engagement reasoning
            if progress_callback:
                progress_callback(0.90, "Fusing multimodal observations and computing engagement...")
            self.finalize_session()

            if progress_callback:
                progress_callback(1.0, "Session analysis complete!")

            return True
        except Exception as e:
            print(f"[ClassPulsePipeline] Video processing note: {e}")
            return False

    def finalize_session(self) -> SessionAnalyticsSummary:
        """
        Runs signal fusion, engagement estimation, recommendation generation,
        and aggregates session analytics.
        """
        # Multimodal Fusion
        self.fused_observations = self.fusion.fuse(
            vision_timeline=self.vision_timeline,
            audio_metrics=self.audio_metrics,
            total_duration=self.session_duration,
        )

        # Engagement Estimation
        self.engagement_results = self.engagement_estimator.estimate_session(self.fused_observations)

        # Pedagogical Recommendations
        self.suggestions = self.rec_engine.generate_recommendations(
            engagement_results=self.engagement_results,
            fused_observations=self.fused_observations,
        )

        # Attendance Records with track consolidation
        attendance_records = self.attendance_mgr.evaluate_attendance(
            self.session_duration,
            min_track_seconds=self.min_track_seconds,
            consolidate=True,
        )

        # Generate Complete Summary
        summary = SessionReportGenerator.generate_summary(
            session_id=self.session_id,
            session_duration=self.session_duration,
            attendance_records=attendance_records,
            engagement_results=self.engagement_results,
            audio_metrics=self.audio_metrics,
            suggestions=self.suggestions,
            raw_tracks_count=self.attendance_mgr.raw_tracks_count,
            consolidated_count=self.attendance_mgr.consolidated_count,
            discarded_short_tracks_count=self.attendance_mgr.discarded_count,
            audio_status=self.audio_status,
            has_audio_track=self.has_audio_track,
            audio_extraction_error=self.audio_extraction_error,
        )

        SessionReportGenerator.save_session_to_disk(summary)
        return summary
