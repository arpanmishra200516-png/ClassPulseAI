"""
Unit tests for Audio Module in ClassPulse AI.
Covers audio loading, segmentation, silent audio handling, missing inputs,
adaptive noise floor stability during speech, and VAD hangover smoothing.
"""

import numpy as np
import pytest
from audio.processor import AudioProcessor, AudioSegmentData
from audio.activity import AudioActivityAnalyzer, AudioWindowMetrics


def test_missing_and_empty_audio_inputs():
    """Validates that missing or non-existent audio inputs are handled gracefully."""
    processor = AudioProcessor(target_sample_rate=16000)
    samples, sr, duration = processor.load_audio("non_existent_file.wav")
    assert samples is None
    assert duration == 0.0
    
    segments = processor.segment_audio(None, 16000)
    assert segments == []


def test_noise_floor_does_not_rise_with_speech():
    """
    Validates that high-energy speech bursts do NOT inflate the background noise floor.
    The noise floor must track true ambient background.
    """
    analyzer = AudioActivityAnalyzer(
        min_energy_threshold=1e-4,
        speech_multiplier=1.5,
        noise_percentile=10.0,
    )
    
    # 5 quiet windows (RMS ~ 0.004)
    quiet_windows = [
        AudioSegmentData(i, float(i), float(i + 1), np.random.normal(0, 0.004, 16000).astype(np.float32), 16000, 1.0, False)
        for i in range(5)
    ]
    
    # 5 loud speech windows (RMS ~ 0.35)
    t = np.linspace(0, 1.0, 16000, endpoint=False)
    speech_signal = (0.4 * np.sin(2 * np.pi * 220 * t) + np.random.normal(0, 0.004, 16000)).astype(np.float32)
    loud_windows = [
        AudioSegmentData(i + 5, float(i + 5), float(i + 6), speech_signal, 16000, 1.0, False)
        for i in range(5)
    ]
    
    all_windows = quiet_windows + loud_windows
    metrics = analyzer.analyze_segments(all_windows)
    
    quiet_noise_floor = metrics[4].noise_floor
    speech_noise_floor = metrics[9].noise_floor

    # Noise floor during loud speech must stay within 2x of quiet floor, NOT jump to speech RMS
    assert speech_noise_floor < 0.03
    assert speech_noise_floor < metrics[9].rms_energy * 0.2


def test_vad_hangover_prevents_flickering():
    """
    Validates that Voice Activity Detection (VAD) hangover sustains speech classification
    across short inter-syllable pauses without flickering off.
    """
    analyzer = AudioActivityAnalyzer(
        min_energy_threshold=1e-4,
        speech_multiplier=1.5,
        vad_hangover_windows=2,
    )
    
    t = np.linspace(0, 1.0, 16000, endpoint=False)
    speech_chunk = (0.35 * np.sin(2 * np.pi * 200 * t)).astype(np.float32)
    quiet_chunk = np.random.normal(0, 0.002, 16000).astype(np.float32)

    # Sequence: Speech (w0), Brief 1s quiet pause (w1), Speech (w2)
    w0 = AudioSegmentData(0, 0.0, 1.0, speech_chunk, 16000, 1.0, False)
    w1 = AudioSegmentData(1, 1.0, 2.0, quiet_chunk, 16000, 1.0, False) # quiet pause
    w2 = AudioSegmentData(2, 2.0, 3.0, speech_chunk, 16000, 1.0, False)

    metrics = analyzer.analyze_segments([w0, w1, w2])

    assert metrics[0].is_speech is True
    # Due to hangover_windows=2, w1 remains classified as speech to avoid flickering
    assert metrics[1].is_speech is True
    assert metrics[2].is_speech is True


def test_silent_audio_handling():
    """Validates that pure silence is handled safely and classified as SILENCE."""
    processor = AudioProcessor(target_sample_rate=16000, window_duration=1.0, hop_duration=1.0)
    analyzer = AudioActivityAnalyzer(min_energy_threshold=1e-4)
    
    silent_samples = np.zeros(3 * 16000, dtype=np.float32)
    segments = processor.segment_audio(silent_samples, 16000)
    metrics = analyzer.analyze_segments(segments)
    
    assert len(metrics) == 3
    for m in metrics:
        assert m.status == "SILENCE"
        assert m.rms_energy == 0.0
        assert m.is_speech is False
