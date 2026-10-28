"""
ClassPulse AI - Configuration Module
Centralized settings for vision, audio, fusion, engagement scoring, and session management.
"""

import os
from pathlib import Path

# Base Paths
BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
SESSIONS_DIR = DATA_DIR / "sessions"
SAMPLE_MEDIA_DIR = DATA_DIR / "sample_media"
EXPORT_DIR = DATA_DIR / "exports"

# Ensure runtime directories exist
for directory in [DATA_DIR, SESSIONS_DIR, SAMPLE_MEDIA_DIR, EXPORT_DIR]:
    directory.mkdir(parents=True, exist_ok=True)

# Prevent Matplotlib cache permission warnings on Windows
os.environ["MPLCONFIGDIR"] = str(DATA_DIR / ".mpl_cache")
(DATA_DIR / ".mpl_cache").mkdir(parents=True, exist_ok=True)

# Application Identity
APP_NAME = "ClassPulse AI"
APP_SUBTITLE = "Multimodal Attendance & Engagement Agent"
APP_VERSION = "1.0.0"

# Vision Module Settings
VISION_CONFIG = {
    # Preferred detector: 'mediapipe', 'haar', or 'auto'
    "detector_type": "mediapipe",
    "min_detection_confidence": 0.65,   # Raised to reduce false positives (posters, reflections)
    # Frame processing scale for performance
    "process_scale": 0.75,
    # Tracker settings
    "tracker_max_disappeared": 25,      # Frames before moving lost face to dormant re-entry
    "tracker_max_distance": 105.0,      # Maximum pixel distance to associate detection
    "reentry_window_seconds": 12.0,     # Time window during which returning faces maintain ID
    "min_track_seconds": 2.0,           # Discard tracks shorter than 2s (transient/noise)
    # Presence and relative attendance defaults
    "default_min_presence_ratio": 0.60, # 60% of session elapsed time
    "use_absolute_override": False,     # If True, use absolute seconds instead of ratio
    "default_min_presence_seconds": 10.0, # Minimum absolute seconds override
    "gap_tolerance_seconds": 2.5,       # Bridge momentary detection losses
}

# Audio Module Settings
AUDIO_CONFIG = {
    "target_sample_rate": 16000,
    "window_duration_seconds": 1.0,
    "hop_duration_seconds": 0.5,
    "min_energy_threshold": 1e-4,       # Below this is considered pure silence
    "speech_energy_multiplier": 1.60,   # Energy ratio above ambient noise to infer speech
    "ambient_noise_percentile": 10.0,   # 10th percentile for stable background noise floor
    "vad_hangover_windows": 2,          # Hangover windows (1.0s) to prevent speech flickering
}

# Multimodal Agent Settings
AGENT_CONFIG = {
    "time_bin_seconds": 1.0,            # Common alignment interval
    "weights": {
        "visual_presence": 0.40,        # Weight for face presence of active participants
        "movement_dynamics": 0.25,      # Weight for head/body micro-movement
        "audio_activity": 0.35,         # Weight for vocal/audio participation
    },
    # Missing modality confidence adjustments
    "vision_only_confidence": 0.75,
    "audio_only_confidence": 0.70,
    "bimodal_confidence": 1.00,
}

# Recommendation Engine Settings
RECOMMENDATION_CONFIG = {
    "sustained_low_engagement_duration": 6.0,  # Minimum seconds below low threshold
    "low_engagement_threshold": 38.0,          # Score threshold
    "sustained_high_noise_duration": 6.0,      # Seconds with elevated noise
    "high_noise_threshold": 0.08,              # RMS energy threshold for background noise
    "monologue_duration": 12.0,                # Sustained speech without movement
    "engagement_drop_delta": 25.0,             # Sudden drop in engagement score
    "participation_surge_delta": 25.0,         # Sudden rise in engagement score
    "min_suggestion_duration": 5.0,            # Minimum duration required to trigger a suggestion
}
