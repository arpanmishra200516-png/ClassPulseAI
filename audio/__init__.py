"""Audio module for ClassPulse AI."""

from audio.processor import AudioProcessor, AudioSegmentData
from audio.activity import AudioActivityAnalyzer, AudioWindowMetrics

__all__ = [
    "AudioProcessor",
    "AudioSegmentData",
    "AudioActivityAnalyzer",
    "AudioWindowMetrics",
]
