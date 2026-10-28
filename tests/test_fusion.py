"""
Unit tests for Multimodal Fusion in ClassPulse AI.
Covers bimodal synchronization, vision-only sessions, audio-only sessions, and missing data handling.
"""

import pytest
from agent.fusion import MultimodalFusion, FusedObservation
from audio.activity import AudioWindowMetrics
from vision.tracker import TrackedParticipant


def test_bimodal_fusion_alignment():
    """Validates alignment of both visual tracks and audio windows into synchronized bins."""
    fusion = MultimodalFusion(bin_size_seconds=1.0)
    
    # 2 seconds of vision
    track1 = TrackedParticipant(
        tracking_id="Participant_01",
        bbox=(50, 50, 40, 40),
        centroid=(70, 70),
        first_seen_timestamp=0.0,
        last_seen_timestamp=0.5,
        recent_movement=5.0,
    )
    vision_timeline = [
        {"timestamp": 0.2, "tracks": [track1]},
        {"timestamp": 1.2, "tracks": [track1]},
    ]
    
    # 2 seconds of audio
    audio_metrics = [
        AudioWindowMetrics(0, 0.0, 1.0, 0.02, 0.1, 0.01, 0.005, 12.0, True, 0.8, "ACTIVE_SPEECH"),
        AudioWindowMetrics(1, 1.0, 2.0, 0.004, 0.02, -0.016, 0.004, 0.0, False, 0.1, "AMBIENT_NOISE"),
    ]
    
    fused = fusion.fuse(vision_timeline=vision_timeline, audio_metrics=audio_metrics)
    assert len(fused) == 2
    
    # Bin 0: [0.0, 1.0)
    assert fused[0].has_vision is True
    assert fused[0].has_audio is True
    assert fused[0].active_faces_count == 1
    assert "Participant_01" in fused[0].participant_ids_present
    assert fused[0].audio_is_speech is True
    
    # Bin 1: [1.0, 2.0)
    assert fused[1].has_vision is True
    assert fused[1].has_audio is True
    assert fused[1].audio_is_speech is False


def test_vision_only_fusion():
    """Validates that a vision-only session has explicit has_audio=False and None audio fields."""
    fusion = MultimodalFusion(bin_size_seconds=1.0)
    
    track = TrackedParticipant("Participant_01", (10, 10, 20, 20), (20, 20), 0.0, 0.5, recent_movement=2.0)
    vision_timeline = [{"timestamp": 0.5, "tracks": [track]}]
    
    fused = fusion.fuse(vision_timeline=vision_timeline, audio_metrics=None)
    assert len(fused) == 1
    assert fused[0].has_vision is True
    assert fused[0].has_audio is False
    assert fused[0].audio_rms_energy is None
    assert fused[0].audio_is_speech is None


def test_audio_only_fusion():
    """Validates that an audio-only session has explicit has_vision=False and zero faces."""
    fusion = MultimodalFusion(bin_size_seconds=1.0)
    
    audio_metrics = [
        AudioWindowMetrics(0, 0.0, 1.0, 0.05, 0.2, 0.0, 0.01, 14.0, True, 0.9, "ACTIVE_SPEECH")
    ]
    
    fused = fusion.fuse(vision_timeline=None, audio_metrics=audio_metrics)
    assert len(fused) == 1
    assert fused[0].has_vision is False
    assert fused[0].has_audio is True
    assert fused[0].active_faces_count == 0
    assert fused[0].participant_ids_present == []
    assert fused[0].audio_is_speech is True


def test_neither_modality_returns_empty():
    """Validates that no data results in an empty list without fabricating records."""
    fusion = MultimodalFusion(bin_size_seconds=1.0)
    fused = fusion.fuse(vision_timeline=None, audio_metrics=None)
    assert fused == []
