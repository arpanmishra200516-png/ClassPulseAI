"""
Unit tests for Vision Module in ClassPulse AI.
Covers face detection, tracking ID stability, occlusion handling, and invalid inputs.
"""

import numpy as np
import pytest
from vision.detector import FaceDetector, DetectionResult
from vision.tracker import FaceTracker


def test_empty_and_invalid_frame_inputs():
    """Validates that detector and tracker handle None, empty, or abnormal frames without crashing."""
    detector = FaceDetector()
    tracker = FaceTracker()
    
    # None input
    assert detector.detect(None) == []
    assert tracker.update([], 0.0) == []
    
    # Zero-size frame
    empty_frame = np.empty((0, 0, 3), dtype=np.uint8)
    assert detector.detect(empty_frame) == []
    
    # Non-array input
    assert detector.detect("not_an_image") == []


def test_tracking_id_stability():
    """Validates that a participant maintains the same tracking ID across consecutive frames."""
    tracker = FaceTracker(max_disappeared=5, max_distance=50.0)
    
    # Frame 1: Detection at (100, 100)
    det1 = DetectionResult(bbox=(80, 80, 40, 40), confidence=0.85, centroid=(100, 100), method="test")
    tracks_f1 = tracker.update([det1], timestamp=0.0)
    assert len(tracks_f1) == 1
    assigned_id = tracks_f1[0].tracking_id
    
    # Frame 2: Slight movement to (105, 102) (displacement ~ 5.4px < max_distance)
    det2 = DetectionResult(bbox=(85, 82, 40, 40), confidence=0.85, centroid=(105, 102), method="test")
    tracks_f2 = tracker.update([det2], timestamp=0.033)
    assert len(tracks_f2) == 1
    assert tracks_f2[0].tracking_id == assigned_id
    assert tracks_f2[0].recent_movement > 0.0
    
    # Frame 3: Another slight movement to (108, 104)
    det3 = DetectionResult(bbox=(88, 84, 40, 40), confidence=0.85, centroid=(108, 104), method="test")
    tracks_f3 = tracker.update([det3], timestamp=0.066)
    assert len(tracks_f3) == 1
    assert tracks_f3[0].tracking_id == assigned_id


def test_temporary_occlusion_tolerance():
    """Validates that brief occlusions (missing detections) do not immediately drop the track."""
    tracker = FaceTracker(max_disappeared=3, max_distance=50.0)
    
    # Initial detection
    det = DetectionResult(bbox=(100, 100, 40, 40), confidence=0.85, centroid=(120, 120), method="test")
    tracker.update([det], timestamp=0.0)
    assert "Participant_01" in tracker.tracks
    
    # Frame missing detection 1
    t1 = tracker.update([], timestamp=0.033)
    assert len(t1) == 1
    assert tracker.tracks["Participant_01"].disappeared_frames == 1
    
    # Frame missing detection 2
    t2 = tracker.update([], timestamp=0.066)
    assert len(t2) == 1
    assert tracker.tracks["Participant_01"].disappeared_frames == 2
    
    # Participant reappears at frame 3
    reappear_det = DetectionResult(bbox=(102, 101, 40, 40), confidence=0.85, centroid=(122, 121), method="test")
    t3 = tracker.update([reappear_det], timestamp=0.100)
    assert len(t3) == 1
    assert t3[0].tracking_id == "Participant_01"
    assert t3[0].disappeared_frames == 0


def test_deregistration_after_max_disappeared():
    """Validates that a lost track is deregistered after exceeding max_disappeared frames."""
    tracker = FaceTracker(max_disappeared=2, max_distance=50.0)
    
    det = DetectionResult(bbox=(100, 100, 40, 40), confidence=0.85, centroid=(120, 120), method="test")
    tracker.update([det], timestamp=0.0)
    
    tracker.update([], timestamp=0.033) # 1
    tracker.update([], timestamp=0.066) # 2
    active_tracks = tracker.update([], timestamp=0.100) # 3 > max_disappeared
    
    assert len(active_tracks) == 0
    assert len(tracker.tracks) == 0
