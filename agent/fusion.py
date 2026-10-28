"""
Multimodal Signal Fusion for ClassPulse AI.
Synchronizes visual tracking observations and acoustic activity metrics across time bins.
Handles missing modalities defensively without fabricating data.
"""

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
import numpy as np
import pandas as pd

from vision.tracker import TrackedParticipant
from audio.activity import AudioWindowMetrics
from config import AGENT_CONFIG


@dataclass
class FusedObservation:
    """A synchronized multimodal snapshot for a specific time interval."""
    bin_idx: int
    bin_start: float
    bin_end: float
    has_vision: bool
    has_audio: bool

    # Vision observations
    active_faces_count: int
    mean_movement_displacement: float
    max_movement_displacement: float
    participant_ids_present: List[str]

    # Audio observations
    audio_rms_energy: Optional[float]
    audio_is_speech: Optional[bool]
    audio_noise_floor: Optional[float]
    audio_snr_db: Optional[float]

    def to_dict(self) -> dict:
        return {
            "bin_idx": self.bin_idx,
            "bin_start": round(self.bin_start, 2),
            "bin_end": round(self.bin_end, 2),
            "has_vision": self.has_vision,
            "has_audio": self.has_audio,
            "active_faces_count": self.active_faces_count,
            "mean_movement_displacement": round(self.mean_movement_displacement, 2),
            "max_movement_displacement": round(self.max_movement_displacement, 2),
            "participant_ids_present": ",".join(self.participant_ids_present),
            "audio_rms_energy": round(self.audio_rms_energy, 5) if self.audio_rms_energy is not None else None,
            "audio_is_speech": self.audio_is_speech,
            "audio_noise_floor": round(self.audio_noise_floor, 5) if self.audio_noise_floor is not None else None,
            "audio_snr_db": round(self.audio_snr_db, 2) if self.audio_snr_db is not None else None,
        }


class MultimodalFusion:
    """
    Fuses vision tracking events and audio window metrics into a coherent time-series.
    """

    def __init__(self, bin_size_seconds: float = AGENT_CONFIG["time_bin_seconds"]):
        self.bin_size = max(0.2, bin_size_seconds)

    def fuse(
        self,
        vision_timeline: Optional[List[Dict[str, any]]] = None,
        audio_metrics: Optional[List[AudioWindowMetrics]] = None,
        total_duration: Optional[float] = None,
    ) -> List[FusedObservation]:
        """
        Aligns visual frame observations and audio windows by timestamp.
        vision_timeline: List of dicts, each e.g.: {"timestamp": float, "tracks": List[TrackedParticipant]}
        audio_metrics: List of AudioWindowMetrics.
        """
        has_vision = bool(vision_timeline and len(vision_timeline) > 0)
        has_audio = bool(audio_metrics and len(audio_metrics) > 0)

        if not has_vision and not has_audio:
            return []

        # Determine start and end timestamps of the session
        start_time = 0.0
        end_time = 0.0

        if has_vision:
            v_starts = [v["timestamp"] for v in vision_timeline]
            end_time = max(end_time, max(v_starts))

        if has_audio:
            a_ends = [m.end_time for m in audio_metrics]
            end_time = max(end_time, max(a_ends))

        if total_duration is not None and total_duration > 0:
            end_time = max(end_time, total_duration)

        # Build discrete bins
        num_bins = max(1, int(np.ceil(end_time / self.bin_size)))
        fused_records: List[FusedObservation] = []

        for bin_idx in range(num_bins):
            bin_start = bin_idx * self.bin_size
            bin_end = bin_start + self.bin_size

            # 1. Aggregate Visual Data for this bin
            bin_vision_tracks: List[TrackedParticipant] = []
            bin_has_vision = False

            if has_vision:
                # Find vision frames landing in [bin_start, bin_end)
                frames_in_bin = [
                    v for v in vision_timeline
                    if bin_start <= v["timestamp"] < bin_end
                ]
                if frames_in_bin:
                    bin_has_vision = True
                    # Collect all tracks observed in this bin
                    for f in frames_in_bin:
                        bin_vision_tracks.extend(f.get("tracks", []))

            if bin_has_vision and bin_vision_tracks:
                unique_pids = list(set(t.tracking_id for t in bin_vision_tracks))
                movements = [t.recent_movement for t in bin_vision_tracks]
                mean_mov = float(np.mean(movements)) if movements else 0.0
                max_mov = float(np.max(movements)) if movements else 0.0
                active_faces = len(unique_pids)
            elif bin_has_vision:
                unique_pids = []
                mean_mov = 0.0
                max_mov = 0.0
                active_faces = 0
            else:
                unique_pids = []
                mean_mov = 0.0
                max_mov = 0.0
                active_faces = 0

            # 2. Aggregate Audio Data for this bin
            bin_has_audio = False
            bin_audio_rms = None
            bin_audio_speech = None
            bin_audio_noise = None
            bin_audio_snr = None

            if has_audio:
                # Find audio metrics whose interval overlaps with this bin
                overlapping_audio = [
                    m for m in audio_metrics
                    if (m.start_time < bin_end and m.end_time > bin_start)
                ]
                if overlapping_audio:
                    bin_has_audio = True
                    bin_audio_rms = float(np.mean([m.rms_energy for m in overlapping_audio]))
                    bin_audio_noise = float(np.mean([m.noise_floor for m in overlapping_audio]))
                    bin_audio_snr = float(np.mean([m.snr_estimate_db for m in overlapping_audio]))
                    # Speech is true if any overlapping window had detected speech
                    bin_audio_speech = any(m.is_speech for m in overlapping_audio)

            fused_records.append(
                FusedObservation(
                    bin_idx=bin_idx,
                    bin_start=round(bin_start, 2),
                    bin_end=round(bin_end, 2),
                    has_vision=bin_has_vision,
                    has_audio=bin_has_audio,
                    active_faces_count=active_faces,
                    mean_movement_displacement=mean_mov,
                    max_movement_displacement=max_mov,
                    participant_ids_present=unique_pids,
                    audio_rms_energy=bin_audio_rms,
                    audio_is_speech=bin_audio_speech,
                    audio_noise_floor=bin_audio_noise,
                    audio_snr_db=bin_audio_snr,
                )
            )

        return fused_records

    @staticmethod
    def to_dataframe(fused: List[FusedObservation]) -> pd.DataFrame:
        """Converts fused observations into a pandas DataFrame."""
        if not fused:
            return pd.DataFrame()
        return pd.DataFrame([f.to_dict() for f in fused])
