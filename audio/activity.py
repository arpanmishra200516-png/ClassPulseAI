"""
Audio Activity and Noise Analyzer for ClassPulse AI.
Implements RMS energy calculation, rolling low-percentile background noise floor estimation
(unaffected by speech bursts), and Voice Activity Detection (VAD) with hangover logic.
"""

from dataclasses import dataclass
from typing import List, Optional
import numpy as np
import pandas as pd

from audio.processor import AudioSegmentData
from config import AUDIO_CONFIG


@dataclass
class AudioWindowMetrics:
    """Detailed acoustic metrics for a single analysis window."""
    window_idx: int
    start_time: float
    end_time: float
    rms_energy: float
    peak_amplitude: float
    energy_delta: float
    noise_floor: float
    snr_estimate_db: float
    is_speech: bool
    speech_confidence: float
    status: str  # 'ACTIVE_SPEECH', 'AMBIENT_NOISE', 'SILENCE', 'UNAVAILABLE'

    def to_dict(self) -> dict:
        return {
            "window_idx": self.window_idx,
            "start_time": round(self.start_time, 2),
            "end_time": round(self.end_time, 2),
            "rms_energy": round(self.rms_energy, 5),
            "peak_amplitude": round(self.peak_amplitude, 4),
            "energy_delta": round(self.energy_delta, 5),
            "noise_floor": round(self.noise_floor, 5),
            "snr_estimate_db": round(self.snr_estimate_db, 2),
            "is_speech": self.is_speech,
            "speech_confidence": round(self.speech_confidence, 2),
            "status": self.status,
        }


class AudioActivityAnalyzer:
    """
    Analyzes audio windows to produce explainable signal energy, robust ambient noise,
    and flicker-free VAD with hangover smoothing.
    """

    def __init__(
        self,
        min_energy_threshold: float = AUDIO_CONFIG["min_energy_threshold"],
        speech_multiplier: float = AUDIO_CONFIG["speech_energy_multiplier"],
        noise_percentile: float = AUDIO_CONFIG["ambient_noise_percentile"],
        vad_hangover_windows: int = AUDIO_CONFIG.get("vad_hangover_windows", 2),
    ):
        self.min_energy_threshold = min_energy_threshold
        self.speech_multiplier = speech_multiplier
        self.noise_percentile = noise_percentile
        self.vad_hangover_windows = vad_hangover_windows

    def analyze_segments(self, segments: List[AudioSegmentData]) -> List[AudioWindowMetrics]:
        """
        Analyzes a series of AudioSegmentData windows.
        Calculates RMS, rolling 10th-percentile ambient noise floor that resists speech inflation,
        and applies VAD hangover logic to eliminate speech flickering.
        """
        if not segments:
            return []

        # Step 1: Feature extraction for each raw window
        raw_rms_list = []
        raw_peaks = []
        raw_zcr_list = []

        for seg in segments:
            chunk = seg.samples
            if chunk is None or len(chunk) == 0:
                raw_rms_list.append(0.0)
                raw_peaks.append(0.0)
                raw_zcr_list.append(0.0)
                continue

            rms = float(np.sqrt(np.mean(chunk.astype(np.float64) ** 2)))
            peak = float(np.max(np.abs(chunk)))

            # Zero crossing rate (ZCR)
            zero_crossings = np.sum(np.abs(np.diff(np.signbit(chunk))))
            zcr = float(zero_crossings) / float(len(chunk))

            raw_rms_list.append(rms)
            raw_peaks.append(peak)
            raw_zcr_list.append(zcr)

        # Step 2: Establish robust global baseline noise floor (10th percentile of non-zero frames)
        non_zero_rms = [r for r in raw_rms_list if r > self.min_energy_threshold]
        if non_zero_rms:
            baseline_noise_floor = float(np.percentile(non_zero_rms, self.noise_percentile))
        else:
            baseline_noise_floor = self.min_energy_threshold

        # Step 3: Rolling background noise estimation that does NOT rise during speech
        # We maintain a buffer of confirmed non-speech / quiet windows
        effective_noise_floor = max(baseline_noise_floor, self.min_energy_threshold)

        metrics_list: List[AudioWindowMetrics] = []
        prev_rms = 0.0
        hangover_counter = 0

        # Initial pass to classify candidates and compute running noise
        recent_quiet_energies = [effective_noise_floor]

        for idx, seg in enumerate(segments):
            rms = raw_rms_list[idx]
            peak = raw_peaks[idx]
            zcr = raw_zcr_list[idx]
            energy_delta = rms - prev_rms
            prev_rms = rms

            # Update rolling noise floor using 10th percentile of recent quiet frames (window of 15)
            if len(recent_quiet_energies) > 20:
                recent_quiet_energies = recent_quiet_energies[-20:]

            current_noise_floor = float(np.percentile(recent_quiet_energies, self.noise_percentile))
            current_noise_floor = max(current_noise_floor, self.min_energy_threshold)

            # Heuristic SNR estimate relative to stable background noise
            if rms > current_noise_floor:
                snr_db = float(20.0 * np.log10(max(rms / current_noise_floor, 1.0)))
            else:
                snr_db = 0.0

            # Raw speech detection criteria:
            # 1. Energy exceeds noise floor by configured multiplier
            # 2. ZCR within characteristic human vocal frequency band (0.015 - 0.40)
            speech_energy_thresh = current_noise_floor * self.speech_multiplier
            raw_is_speech = (rms >= speech_energy_thresh) and (0.015 <= zcr <= 0.40) and (rms >= self.min_energy_threshold * 2)

            # If NOT speech, record into quiet background history so noise floor updates
            if not raw_is_speech:
                recent_quiet_energies.append(rms)

            # Apply VAD hangover time to prevent segment flickering
            if raw_is_speech:
                hangover_counter = self.vad_hangover_windows
                is_speech = True
                status = "ACTIVE_SPEECH"
                speech_confidence = float(min(1.0, max(0.5, snr_db / 15.0)))
            elif hangover_counter > 0:
                hangover_counter -= 1
                is_speech = True
                status = "ACTIVE_SPEECH"
                speech_confidence = 0.60  # Maintained by hangover
            elif rms < self.min_energy_threshold:
                is_speech = False
                status = "SILENCE"
                speech_confidence = 0.0
            else:
                is_speech = False
                status = "AMBIENT_NOISE"
                speech_confidence = 0.10

            metric = AudioWindowMetrics(
                window_idx=seg.window_idx,
                start_time=seg.start_time,
                end_time=seg.end_time,
                rms_energy=rms,
                peak_amplitude=peak,
                energy_delta=energy_delta,
                noise_floor=current_noise_floor,
                snr_estimate_db=snr_db,
                is_speech=is_speech,
                speech_confidence=speech_confidence,
                status=status,
            )
            metrics_list.append(metric)

        return metrics_list

    def summarize(self, metrics: List[AudioWindowMetrics]) -> dict:
        """Produces summary statistics for the audio session."""
        if not metrics:
            return {
                "total_windows": 0,
                "duration_seconds": 0.0,
                "average_rms": 0.0,
                "peak_amplitude": 0.0,
                "speech_window_count": 0,
                "speech_percentage": 0.0,
                "ambient_noise_floor": 0.0,
                "status": "NO_AUDIO_DATA",
            }

        total_windows = len(metrics)
        duration = metrics[-1].end_time - metrics[0].start_time
        speech_windows = sum(1 for m in metrics if m.is_speech)
        avg_rms = float(np.mean([m.rms_energy for m in metrics]))
        max_peak = float(np.max([m.peak_amplitude for m in metrics]))
        avg_noise = float(np.mean([m.noise_floor for m in metrics]))
        speech_pct = (speech_windows / total_windows) * 100.0 if total_windows > 0 else 0.0

        return {
            "total_windows": total_windows,
            "duration_seconds": round(duration, 2),
            "average_rms": round(avg_rms, 5),
            "peak_amplitude": round(max_peak, 4),
            "speech_window_count": speech_windows,
            "speech_percentage": round(speech_pct, 1),
            "ambient_noise_floor": round(avg_noise, 5),
            "status": "ANALYZED",
        }

    @staticmethod
    def to_dataframe(metrics: List[AudioWindowMetrics]) -> pd.DataFrame:
        """Converts window metrics to a pandas DataFrame for plotting and export."""
        if not metrics:
            return pd.DataFrame(
                columns=[
                    "start_time",
                    "end_time",
                    "rms_energy",
                    "peak_amplitude",
                    "noise_floor",
                    "snr_estimate_db",
                    "is_speech",
                    "status",
                ]
            )
        return pd.DataFrame([m.to_dict() for m in metrics])
