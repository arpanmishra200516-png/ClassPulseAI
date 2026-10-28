"""Vision module for ClassPulse AI."""

from vision.detector import FaceDetector, DetectionResult
from vision.tracker import FaceTracker, TrackedParticipant
from vision.attendance import AttendanceManager, ParticipantAttendanceRecord

__all__ = [
    "FaceDetector",
    "DetectionResult",
    "FaceTracker",
    "TrackedParticipant",
    "AttendanceManager",
    "ParticipantAttendanceRecord",
]
