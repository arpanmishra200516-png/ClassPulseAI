"""
Engagement Estimator for ClassPulse AI.
Computes an explainable, rule-based heuristic engagement score (0–100)
from fused multimodal observations with transparent feature contributions.
Guarantees engagement reflects active visible participants without distortion from absent ones.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional
import numpy as np
import pandas as pd

from agent.fusion import FusedObservation
from config import AGENT_CONFIG


@dataclass
class EngagementScoreResult:
    """Interpretable engagement calculation for a single analysis bin."""
    bin_idx: int
    bin_start: float
    bin_end: float
    score: float                           # 0.0 to 100.0
    level: str                             # 'HIGH', 'MODERATE', 'LOW', 'NO_DATA'
    confidence: float                      # 0.0 to 1.0
    breakdown: Dict[str, float]            # Individual component scores
    explanation: str                       # Transparent explanation of the result
    active_modality: str                   # 'BIMODAL', 'VISION_ONLY', 'AUDIO_ONLY', 'NONE'
    active_participants: int = 0
    expected_participants: int = 1
    formula_string: str = ""
    academic_disclaimer: str = (
        "Heuristic engagement indicator based on observable visual presence, movement, "
        "and acoustic activity. It does not measure student comprehension, learning capacity, "
        "or cognitive attention."
    )

    def to_dict(self) -> dict:
        return {
            "bin_idx": self.bin_idx,
            "bin_start": round(self.bin_start, 2),
            "bin_end": round(self.bin_end, 2),
            "score": round(self.score, 1),
            "level": self.level,
            "confidence": round(self.confidence, 2),
            "active_modality": self.active_modality,
            "active_participants": self.active_participants,
            "presence_score": round(self.breakdown.get("presence_score", 0.0), 1),
            "movement_score": round(self.breakdown.get("movement_score", 0.0), 1),
            "audio_score": round(self.breakdown.get("audio_score", 0.0), 1),
            "explanation": self.explanation,
            "formula_string": self.formula_string,
        }


class EngagementEstimator:
    """
    Computes explainable heuristic engagement scores from FusedObservations.
    Supports bimodal, vision-only, and audio-only sessions gracefully.
    """

    def __init__(
        self,
        weights: Optional[Dict[str, float]] = None,
        expected_participants: int = 1,
    ):
        self.weights = weights or AGENT_CONFIG["weights"]
        self.expected_participants = max(1, expected_participants)

    def _score_visual_presence(self, obs: FusedObservation) -> float:
        """
        Computes score for visual presence based on active participants.
        Scales up to 100% relative to expected participant count.
        """
        if not obs.has_vision or obs.active_faces_count == 0:
            return 0.0
        ratio = min(1.0, obs.active_faces_count / self.expected_participants)
        return float(ratio * 100.0)

    def _score_movement_dynamics(self, obs: FusedObservation) -> float:
        """
        Computes movement score strictly across active visible participants.
        Natural classroom micro-movements (3-45 px displacement) yield optimal score (~75-100).
        Zero movement (prolonged freeze / empty view) yields low score (35).
        Excessive erratic motion (> 80 px) yields moderate score with a slight penalty (55).
        """
        if not obs.has_vision or obs.active_faces_count == 0:
            return 0.0

        disp = obs.mean_movement_displacement
        if disp < 1.0:
            return 35.0
        elif 1.0 <= disp < 4.0:
            return 70.0
        elif 4.0 <= disp <= 35.0:
            return 95.0
        elif 35.0 < disp <= 80.0:
            return 80.0
        else:
            return 55.0

    def _score_audio_activity(self, obs: FusedObservation) -> float:
        """
        Computes score for audio activity.
        Active speech with healthy SNR yields high score (~80-100).
        Quiet ambient listening yields moderate score (~60).
        Distracting noisy environment without speech yields lower score (~30).
        Pure silence yields low-neutral score (~40).
        """
        if not obs.has_audio or obs.audio_rms_energy is None:
            return 0.0

        if obs.audio_is_speech:
            snr = obs.audio_snr_db or 5.0
            return float(min(100.0, max(75.0, 75.0 + (snr / 15.0) * 25.0)))
        elif obs.audio_rms_energy < 0.005:
            return 60.0  # Quiet classroom listening
        elif obs.audio_rms_energy > 0.08:
            return 30.0  # High noise without speech
        else:
            return 50.0  # Moderate background

    def estimate_window(self, obs: FusedObservation) -> EngagementScoreResult:
        """Evaluates engagement for a single fused observation bin."""
        has_v = obs.has_vision
        has_a = obs.has_audio
        active_pts = obs.active_faces_count

        # Scenario 1: Neither modality available
        if not has_v and not has_a:
            return EngagementScoreResult(
                bin_idx=obs.bin_idx,
                bin_start=obs.bin_start,
                bin_end=obs.bin_end,
                score=0.0,
                level="NO_DATA",
                confidence=0.0,
                breakdown={"presence_score": 0.0, "movement_score": 0.0, "audio_score": 0.0},
                explanation="No sensory observations available in this time interval.",
                active_modality="NONE",
                active_participants=0,
                expected_participants=self.expected_participants,
                formula_string="0.0 (No Data)",
            )

        w_v = self.weights.get("visual_presence", 0.40)
        w_m = self.weights.get("movement_dynamics", 0.25)
        w_a = self.weights.get("audio_activity", 0.35)

        # Scenario 2: Bimodal (both available)
        if has_v and has_a:
            s_pres = self._score_visual_presence(obs)
            s_mov = self._score_movement_dynamics(obs)
            s_aud = self._score_audio_activity(obs)

            total_w = w_v + w_m + w_a
            score = (w_v * s_pres + w_m * s_mov + w_a * s_aud) / total_w
            confidence = AGENT_CONFIG["bimodal_confidence"]
            modality = "BIMODAL"

            formula_str = (
                f"({w_v:.2f} × {s_pres:.1f} [Presence]) + "
                f"({w_m:.2f} × {s_mov:.1f} [Movement]) + "
                f"({w_a:.2f} × {s_aud:.1f} [Audio]) = {score:.1f}"
            )

            exp_parts = [f"{active_pts}/{self.expected_participants} participant(s) visible"]
            if obs.audio_is_speech:
                exp_parts.append("active speech detected")
            else:
                exp_parts.append("ambient/quiet room")

            explanation = f"Bimodal: {', '.join(exp_parts)} (Displacement: {obs.mean_movement_displacement:.1f}px)."

        # Scenario 3: Vision-Only
        elif has_v and not has_a:
            s_pres = self._score_visual_presence(obs)
            s_mov = self._score_movement_dynamics(obs)
            s_aud = 0.0

            total_w = max(0.01, w_v + w_m)
            score = (w_v * s_pres + w_m * s_mov) / total_w
            confidence = AGENT_CONFIG["vision_only_confidence"]
            modality = "VISION_ONLY"

            formula_str = (
                f"(({w_v:.2f} × {s_pres:.1f} [Presence]) + "
                f"({w_m:.2f} × {s_mov:.1f} [Movement])) / {total_w:.2f} = {score:.1f}"
            )
            explanation = (
                f"Vision-only: {active_pts}/{self.expected_participants} face(s) visible, "
                f"{obs.mean_movement_displacement:.1f}px displacement. Audio track unavailable."
            )

        # Scenario 4: Audio-Only
        else:
            s_pres = 0.0
            s_mov = 0.0
            s_aud = self._score_audio_activity(obs)

            score = s_aud
            confidence = AGENT_CONFIG["audio_only_confidence"]
            modality = "AUDIO_ONLY"

            formula_str = f"1.0 × {s_aud:.1f} [Audio Activity] = {score:.1f}"
            speech_str = "speech active" if obs.audio_is_speech else "ambient background"
            explanation = f"Audio-only: {speech_str} (RMS: {obs.audio_rms_energy:.4f}). Camera feed unavailable."

        score = float(np.clip(score, 0.0, 100.0))

        if score >= 68.0:
            level = "HIGH"
        elif score >= 40.0:
            level = "MODERATE"
        else:
            level = "LOW"

        return EngagementScoreResult(
            bin_idx=obs.bin_idx,
            bin_start=obs.bin_start,
            bin_end=obs.bin_end,
            score=score,
            level=level,
            confidence=confidence,
            breakdown={
                "presence_score": s_pres,
                "movement_score": s_mov,
                "audio_score": s_aud,
            },
            explanation=explanation,
            active_modality=modality,
            active_participants=active_pts,
            expected_participants=self.expected_participants,
            formula_string=formula_str,
        )

    def estimate_session(self, fused_observations: List[FusedObservation]) -> List[EngagementScoreResult]:
        """Calculates engagement score timeline for the entire fused observation series."""
        return [self.estimate_window(obs) for obs in fused_observations]

    @staticmethod
    def to_dataframe(results: List[EngagementScoreResult]) -> pd.DataFrame:
        """Converts engagement score results into a pandas DataFrame."""
        if not results:
            return pd.DataFrame()
        return pd.DataFrame([r.to_dict() for r in results])
