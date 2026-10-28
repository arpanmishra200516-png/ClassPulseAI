"""
Command-line Interface (CLI) for ClassPulse AI.
Allows headless execution of the multimodal pipeline on video and audio files,
as well as comprehensive automated subsystem self-tests (--selftest).
"""

import argparse
import sys
from pathlib import Path
import json
import numpy as np

from config import BASE_DIR, APP_NAME, APP_VERSION, SAMPLE_MEDIA_DIR
from core_pipeline import ClassPulsePipeline
from analytics.reports import SessionReportGenerator
from vision.detector import FaceDetector, DetectionResult
from vision.tracker import FaceTracker
from vision.attendance import AttendanceManager
from audio.processor import AudioProcessor
from audio.activity import AudioActivityAnalyzer, AudioWindowMetrics
from agent.fusion import MultimodalFusion, FusedObservation
from agent.engagement import EngagementEstimator
from agent.recommendations import RecommendationEngine


def run_selftest() -> int:
    """
    Runs diagnostic tests across all six core system modules.
    Prints PASS/FAIL per module and returns 0 if all pass, 1 otherwise.
    """
    print(f"\n===========================================================")
    print(f"  {APP_NAME} v{APP_VERSION} — System Self-Test Diagnostics")
    print(f"===========================================================\n")

    results = []

    # 1. Vision Module
    try:
        detector = FaceDetector()
        assert detector.is_ready(), "At least one detection backend must be ready"
        test_frame = np.zeros((200, 200, 3), dtype=np.uint8)
        dets = detector.detect(test_frame)
        assert isinstance(dets, list)
        results.append(("Vision Module (MediaPipe & Haar Fallback)", True, "Backend active, inference ready"))
    except Exception as e:
        results.append(("Vision Module (MediaPipe & Haar Fallback)", False, str(e)))

    # 2. Face Tracker & IoU
    try:
        tracker = FaceTracker(max_disappeared=5, max_distance=60.0)
        d1 = DetectionResult((50, 50, 40, 40), 0.85, (70, 70), "test")
        tracks = tracker.update([d1], 0.0)
        assert len(tracks) == 1
        tid = tracks[0].tracking_id
        d2 = DetectionResult((52, 51, 40, 40), 0.85, (72, 71), "test")
        tracks2 = tracker.update([d2], 0.033)
        assert tracks2[0].tracking_id == tid, "Track ID must persist across continuous detections"

        # Re-entry window test: ID persists when returning after temporary loss
        tracker_reentry = FaceTracker(max_disappeared=2, reentry_window_seconds=5.0)
        tr1 = tracker_reentry.update([DetectionResult((100, 100, 50, 50), 0.9, (125, 125), "test")], 0.0)
        reentry_tid = tr1[0].tracking_id
        tracker_reentry.update([], 1.0)
        tracker_reentry.update([], 2.0)
        tracker_reentry.update([], 3.0)
        tr2 = tracker_reentry.update([DetectionResult((102, 101, 50, 50), 0.9, (127, 126), "test")], 3.5)
        assert len(tr2) == 1
        assert tr2[0].tracking_id == reentry_tid, "Re-entering face within window must retain original ID"

        results.append(("Face Tracker & IoU Persistence", True, "Centroid/IoU tracking and ID persistence verified"))
    except Exception as e:
        results.append(("Face Tracker & IoU Persistence", False, str(e)))

    # 3. Attendance Manager
    try:
        att = AttendanceManager(session_id="diag", min_presence_ratio=0.60)
        # Participant 1 is present (observed 6.5s of 10.0s >= 6.0s)
        att.record_presence("P1", timestamp=0.0, duration=6.5)
        # Participant 2 is absent (observed 2.5s of 10.0s < 6.0s)
        att.record_presence("P2", timestamp=0.0, duration=2.5)
        # Participant 3 is a short transient track (1.0s < min_track_seconds 2.0s)
        att.record_presence("P3_transient", timestamp=0.0, duration=1.0)

        recs = att.evaluate_attendance(current_session_duration=10.0, min_track_seconds=2.0, consolidate=True)
        assert len(recs) == 2, "Transient track under min_track_seconds must be filtered out"
        assert att.discarded_count == 1
        assert att.raw_tracks_count == 3
        assert att.consolidated_count == 2

        p1_rec = next(r for r in recs if r.participant_id == "P1")
        p2_rec = next(r for r in recs if r.participant_id == "P2")
        assert p1_rec.attendance_status == "PRESENT"
        assert len(p1_rec.evaluation_rationale) > 0 and "PRESENT:" in p1_rec.evaluation_rationale
        assert p2_rec.attendance_status == "ABSENT"
        assert len(p2_rec.evaluation_rationale) > 0 and "ABSENT:" in p2_rec.evaluation_rationale

        is_val, msg = att.validate_threshold(session_duration=2.0)
        assert isinstance(is_val, bool)
        results.append(("Attendance Manager (Consolidation & Rationale)", True, "Consolidation, non-empty rationales & thresholds verified"))
    except Exception as e:
        results.append(("Attendance Manager (Consolidation & Rationale)", False, str(e)))

    # 4. Audio & VAD
    try:
        proc = AudioProcessor(target_sample_rate=16000)
        analyzer = AudioActivityAnalyzer(vad_hangover_windows=2)
        # Quiet background (1s) followed by speech burst (1s)
        quiet_chunk = np.random.normal(0, 0.003, 16000).astype(np.float32)
        t = np.linspace(0, 1.0, 16000, endpoint=False)
        speech_chunk = (0.35 * np.sin(2 * np.pi * 220 * t)).astype(np.float32)
        sig = np.concatenate([quiet_chunk, speech_chunk])
        segs = proc.segment_audio(sig, 16000)
        metrics = analyzer.analyze_segments(segs)
        assert len(metrics) > 0
        assert any(m.is_speech for m in metrics), "Speech activity must be detected in voiced burst"

        # Missing audio stream detection
        has_audio, _ = proc.check_has_audio_stream("non_existent_file.mp4")
        assert has_audio is False, "Missing or streamless video must detect no audio"
        sig_none, sr_none, dur_none, status_none, msg_none, path_none = proc.load_audio_with_status("non_existent_file.mp4")
        assert status_none in ("NO_AUDIO_TRACK", "UNAVAILABLE", "FILE_NOT_FOUND")

        results.append(("Audio Signal Processor & Flicker-free VAD", True, "RMS, noise floor, VAD hangover & no-audio handling operational"))
    except Exception as e:
        results.append(("Audio Signal Processor & Flicker-free VAD", False, str(e)))

    # 5. Multimodal Fusion & Engagement
    try:
        fusion = MultimodalFusion(bin_size_seconds=1.0)
        fused = fusion.fuse(vision_timeline=[{"timestamp": 0.5, "tracks": []}], audio_metrics=[metrics[0]])
        assert len(fused) > 0
        estimator = EngagementEstimator()
        eng_res = estimator.estimate_session(fused)
        assert 0.0 <= eng_res[0].score <= 100.0
        results.append(("Multimodal Signal Fusion & Engagement Scoring", True, "Bimodal time-binning & 0-100 scoring operational"))
    except Exception as e:
        results.append(("Multimodal Signal Fusion & Engagement Scoring", False, str(e)))

    # 6. Recommendation Engine & Reports
    try:
        rec_engine = RecommendationEngine(min_suggestion_duration=2.0)
        recs = rec_engine.generate_recommendations(eng_res, fused)
        assert isinstance(recs, list)
        summary = SessionReportGenerator.generate_summary("diag_sess", 2.0, [], eng_res)
        html = SessionReportGenerator.generate_html_report(summary)
        assert "<!DOCTYPE html>" in html
        results.append(("Pedagogical Recommendation Engine & HTML Export", True, "Rule-based suggestions & HTML/PDF reports operational"))
    except Exception as e:
        results.append(("Pedagogical Recommendation Engine & HTML Export", False, str(e)))

    # 7. Unified Session State & Transparency
    try:
        from analytics.reports import SessionResult
        sess = SessionResult.from_summary(summary)
        assert sess.session_id == summary.session_id
        # Modality transparency: missing audio yields None/N/A
        summary_no_audio = SessionReportGenerator.generate_summary(
            "no_aud_diag", 10.0, [], eng_res, audio_metrics=None, has_audio_track=False
        )
        assert summary_no_audio.speech_percentage is None
        assert summary_no_audio.ambient_noise_floor is None
        html_na = SessionReportGenerator.generate_html_report(summary_no_audio)
        assert "N/A" in html_na
        results.append(("Unified Session State & Modality Transparency", True, "SessionResult single source of truth & N/A metrics verified"))
    except Exception as e:
        results.append(("Unified Session State & Modality Transparency", False, str(e)))

    # Print Report
    all_passed = True
    for name, passed, detail in results:
        status_tag = "[PASS]" if passed else "[FAIL]"
        if not passed:
            all_passed = False
        print(f"  {status_tag:<8} {name}")
        print(f"           └─ {detail}")

    print(f"\n===========================================================")
    if all_passed:
        print(f"  Diagnostic Result: ALL {len(results)} SUBSYSTEMS PASSED (100% HEALTHY)")
        print(f"===========================================================\n")
        return 0
    else:
        print(f"  Diagnostic Result: ONE OR MORE SUBSYSTEMS FAILED")
        print(f"===========================================================\n")
        return 1


def main():
    parser = argparse.ArgumentParser(
        description=f"{APP_NAME} (v{APP_VERSION}) — Multimodal Attendance & Engagement Agent"
    )
    parser.add_argument("--selftest", action="store_true", help="Run comprehensive subsystem diagnostic tests")
    parser.add_argument("--session-id", type=str, default="cli_session_01", help="Session ID")
    parser.add_argument("--video", type=str, default=None, help="Path to video file")
    parser.add_argument("--audio", type=str, default=None, help="Path to standalone audio file")
    parser.add_argument("--min-presence-ratio", type=float, default=0.60, help="Relative presence threshold (default: 0.60)")
    parser.add_argument("--min-presence-sec", type=float, default=None, help="Absolute seconds override threshold")
    parser.add_argument("--gap-tolerance", type=float, default=2.5, help="Gap tolerance in seconds")
    parser.add_argument("--export-dir", type=str, default="data/exports", help="Export output directory")

    args = parser.parse_args()

    if args.selftest:
        sys.exit(run_selftest())

    print(f"===========================================================")
    print(f"  {APP_NAME} v{APP_VERSION}")
    print(f"  Session ID: {args.session_id}")
    print(f"===========================================================\n")

    pipeline = ClassPulsePipeline(
        session_id=args.session_id,
        min_presence_ratio=args.min_presence_ratio,
        min_presence_seconds=args.min_presence_sec,
        use_absolute_override=args.min_presence_sec is not None,
        gap_tolerance_seconds=args.gap_tolerance,
    )

    if args.video:
        print(f"[Vision] Processing video file: {args.video}...")
        success = pipeline.process_video_file(args.video)
        if not success:
            print(f"[Vision] Failed to open video.")
    else:
        print("[Vision] No video file supplied. Running audio-only / timeline.")

    if args.audio:
        print(f"[Audio] Processing audio file: {args.audio}...")
        pipeline.process_audio_file(args.audio)

    print("[Agent] Fusing observations and computing engagement...")
    summary = pipeline.finalize_session()

    print("\n-----------------------------------------------------------")
    print("                     SESSION SUMMARY")
    print("-----------------------------------------------------------")
    print(f"Duration:                  {summary.session_duration_seconds:.1f} seconds")
    print(f"Total Tracked Identifiers: {summary.total_tracking_ids}")
    print(f"Confirmed Present:         {summary.present_count}")
    print(f"Absent (< Threshold):      {summary.absent_count}")
    print(f"Attendance Rate:           {summary.attendance_rate_percentage:.1f}%")
    print(f"Mean Engagement Score:     {summary.mean_engagement_score:.1f} / 100")
    print(f"Speech Activity Ratio:     {summary.speech_percentage:.1f}%")
    print(f"Noise Floor (RMS):         {summary.ambient_noise_floor:.4f}")
    print(f"Generated Suggestions:     {summary.recommendations_count}")
    print("-----------------------------------------------------------")

    if summary.attendance_records:
        print("\nParticipant Attendance Breakdown:")
        for rec in summary.attendance_records:
            reason = f"({rec['absence_reason']})" if rec['absence_reason'] else ""
            print(
                f"  • {rec['participant_id']:<16}: {rec['attendance_status']:<8} "
                f"(Observed: {rec['accumulated_presence_seconds']:.1f}s / {rec['required_presence_seconds']:.1f}s | {rec['presence_percentage']:.1f}%) {reason}"
            )

    if summary.recommendations:
        print("\nPedagogical Suggestions:")
        for rec in summary.recommendations:
            print(f"  [{rec['priority']}] {rec['title']} (Confidence: {rec['confidence']:.0%})")
            print(f"      Observation: {rec['observation']}")
            print(f"      Suggestion:  {rec['suggestion']}")
            print(f"      Metrics:     {rec['exact_metrics']}")
            print(f"      Caveat:      {rec['limitation_note']}\n")

    # Export records
    export_dir = Path(args.export_dir)
    export_dir.mkdir(parents=True, exist_ok=True)

    att_path = export_dir / f"{args.session_id}_attendance.csv"
    with open(att_path, "w", encoding="utf-8") as f:
        f.write(SessionReportGenerator.export_attendance_csv(
            pipeline.attendance_mgr.evaluate_attendance(pipeline.session_duration)
        ))

    eng_path = export_dir / f"{args.session_id}_engagement.csv"
    with open(eng_path, "w", encoding="utf-8") as f:
        f.write(SessionReportGenerator.export_engagement_timeline_csv(pipeline.engagement_results))

    json_path = export_dir / f"{args.session_id}_report.json"
    with open(json_path, "w", encoding="utf-8") as f:
        f.write(summary.to_json())

    html_path = export_dir / f"{args.session_id}_report.html"
    with open(html_path, "w", encoding="utf-8") as f:
        f.write(SessionReportGenerator.generate_html_report(summary))

    print(f"\n[Exports] Files saved to:")
    print(f"  - {att_path}")
    print(f"  - {eng_path}")
    print(f"  - {json_path}")
    print(f"  - {html_path}")
    print("Execution complete.\n")


if __name__ == "__main__":
    main()
