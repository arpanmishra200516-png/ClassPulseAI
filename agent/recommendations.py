"""
Pedagogical Recommendation Engine for ClassPulse AI.
Analyzes computed engagement trends, movement patterns, and audio activity
to generate actionable, non-prescriptive, explainable teaching suggestions.
Enforces minimum segment durations, merges adjacent segments, resolves contradictions,
and includes exact supporting metric values with confidence ratings.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Optional
import numpy as np
import pandas as pd

from agent.engagement import EngagementScoreResult
from agent.fusion import FusedObservation
from config import RECOMMENDATION_CONFIG


@dataclass
class TeachingSuggestion:
    """Represents a data-driven pedagogical suggestion for an instructor."""
    suggestion_id: str
    trigger_type: str                  # e.g., 'SUSTAINED_LOW_ENGAGEMENT', 'HIGH_NOISE'
    time_interval: Tuple[float, float] # (start_seconds, end_seconds)
    title: str
    observation: str                   # Concrete data observation that triggered this rule
    suggestion: str                    # Actionable teaching suggestion
    pedagogical_rationale: str         # Why this suggestion is pedagogically beneficial
    limitation_note: str               # Important limitations or caveats of the heuristic
    priority: str                      # 'INFO', 'LOW', 'MEDIUM', 'HIGH'
    confidence: float = 0.85           # 0.0 to 1.0 confidence level
    exact_metrics: Dict[str, any] = field(default_factory=dict) # Exact values supporting trigger

    def to_dict(self) -> dict:
        return {
            "suggestion_id": self.suggestion_id,
            "trigger_type": self.trigger_type,
            "interval_start": round(self.time_interval[0], 2),
            "interval_end": round(self.time_interval[1], 2),
            "duration_seconds": round(self.time_interval[1] - self.time_interval[0], 2),
            "title": self.title,
            "observation": self.observation,
            "suggestion": self.suggestion,
            "pedagogical_rationale": self.pedagogical_rationale,
            "limitation_note": self.limitation_note,
            "priority": self.priority,
            "confidence": round(self.confidence, 2),
            "exact_metrics": self.exact_metrics,
        }


class RecommendationEngine:
    """
    Generates explainable, rule-based teaching recommendations from computed session analytics.
    Deduplicates overlapping suggestions, merges adjacent intervals, and requires minimum duration.
    """

    def __init__(
        self,
        sustained_low_duration: float = RECOMMENDATION_CONFIG["sustained_low_engagement_duration"],
        low_engagement_threshold: float = RECOMMENDATION_CONFIG["low_engagement_threshold"],
        sustained_high_noise_duration: float = RECOMMENDATION_CONFIG["sustained_high_noise_duration"],
        high_noise_threshold: float = RECOMMENDATION_CONFIG["high_noise_threshold"],
        monologue_duration: float = RECOMMENDATION_CONFIG["monologue_duration"],
        engagement_drop_delta: float = RECOMMENDATION_CONFIG["engagement_drop_delta"],
        min_suggestion_duration: float = RECOMMENDATION_CONFIG.get("min_suggestion_duration", 5.0),
    ):
        self.sustained_low_duration = sustained_low_duration
        self.low_threshold = low_engagement_threshold
        self.sustained_high_noise_duration = sustained_high_noise_duration
        self.high_noise_threshold = high_noise_threshold
        self.monologue_duration = monologue_duration
        self.drop_delta = engagement_drop_delta
        self.min_suggestion_duration = min_suggestion_duration

    def generate_recommendations(
        self,
        engagement_results: List[EngagementScoreResult],
        fused_observations: Optional[List[FusedObservation]] = None,
    ) -> List[TeachingSuggestion]:
        """
        Scans engagement results and fused observations to generate context-aware suggestions.
        Merges adjacent intervals and resolves contradictions.
        """
        raw_suggestions: List[TeachingSuggestion] = []
        rec_counter = 1

        if not engagement_results:
            return [
                TeachingSuggestion(
                    suggestion_id="REC_01",
                    trigger_type="INSUFFICIENT_DATA",
                    time_interval=(0.0, 0.0),
                    title="Insufficient Session Observations",
                    observation="No timeline observation data recorded for this session.",
                    suggestion="Ensure camera feed and microphone input are connected and active before starting analysis.",
                    pedagogical_rationale="Meaningful instructional insights require sufficient real-time observations.",
                    limitation_note="System cannot generate recommendations without recorded data.",
                    priority="INFO",
                    confidence=1.0,
                    exact_metrics={"recorded_bins": 0},
                )
            ]

        total_duration = engagement_results[-1].bin_end - engagement_results[0].bin_start
        if total_duration < self.min_suggestion_duration:
            return [
                TeachingSuggestion(
                    suggestion_id="REC_01",
                    trigger_type="INSUFFICIENT_DATA",
                    time_interval=(engagement_results[0].bin_start, engagement_results[-1].bin_end),
                    title="Session Too Short for Trend Analysis",
                    observation=f"Session duration is only {total_duration:.1f}s (minimum required is {self.min_suggestion_duration:.1f}s).",
                    suggestion="Allow the session to run for at least 30-60 seconds to observe meaningful interaction trends.",
                    pedagogical_rationale="Brief snapshots are susceptible to transient fluctuations.",
                    limitation_note="Pedagogical patterns develop across minutes rather than seconds.",
                    priority="INFO",
                    confidence=1.0,
                    exact_metrics={"session_duration": round(total_duration, 1), "min_required": self.min_suggestion_duration},
                )
            ]

        # Rule 1: Detect Sustained Low Engagement with Contiguity Merging
        low_streaks: List[List[EngagementScoreResult]] = []
        current_streak: List[EngagementScoreResult] = []

        for res in engagement_results:
            if res.score < self.low_threshold and res.level != "NO_DATA":
                current_streak.append(res)
            else:
                if current_streak:
                    duration = current_streak[-1].bin_end - current_streak[0].bin_start
                    if duration >= self.sustained_low_duration:
                        low_streaks.append(current_streak)
                    current_streak = []

        if current_streak:
            duration = current_streak[-1].bin_end - current_streak[0].bin_start
            if duration >= self.sustained_low_duration:
                low_streaks.append(current_streak)

        for streak in low_streaks:
            start_t = streak[0].bin_start
            end_t = streak[-1].bin_end
            dur = end_t - start_t
            mean_score = float(np.mean([r.score for r in streak]))
            raw_suggestions.append(
                TeachingSuggestion(
                    suggestion_id=f"REC_{rec_counter:02d}",
                    trigger_type="SUSTAINED_LOW_ENGAGEMENT",
                    time_interval=(start_t, end_t),
                    title="Consider Introducing an Interactive Activity",
                    observation=(
                        f"Estimated engagement remained below {self.low_threshold:.0f} "
                        f"(average {mean_score:.1f}) for {dur:.1f} consecutive seconds."
                    ),
                    suggestion=(
                        "Pause for a 60-second think-pair-share, launch a quick poll, "
                        "or pose an open-ended concept check question."
                    ),
                    pedagogical_rationale="Active retrieval practice and interactive breaks rekindle focus during passive lecture segments.",
                    limitation_note="Low observed visual movement or silence may also indicate deep independent reading or quiet note-taking.",
                    priority="MEDIUM",
                    confidence=0.85,
                    exact_metrics={
                        "mean_engagement": round(mean_score, 1),
                        "threshold": self.low_threshold,
                        "duration_seconds": round(dur, 1),
                    },
                )
            )
            rec_counter += 1

        # Rule 2: Detect Sustained High Acoustic Noise (if audio fused)
        if fused_observations:
            noise_streaks: List[List[FusedObservation]] = []
            curr_noise: List[FusedObservation] = []

            for f_obs in fused_observations:
                if (
                    f_obs.has_audio
                    and f_obs.audio_rms_energy is not None
                    and f_obs.audio_rms_energy >= self.high_noise_threshold
                    and not f_obs.audio_is_speech
                ):
                    curr_noise.append(f_obs)
                else:
                    if curr_noise:
                        duration = curr_noise[-1].bin_end - curr_noise[0].bin_start
                        if duration >= self.sustained_high_noise_duration:
                            noise_streaks.append(curr_noise)
                        curr_noise = []

            if curr_noise:
                duration = curr_noise[-1].bin_end - curr_noise[0].bin_start
                if duration >= self.sustained_high_noise_duration:
                    noise_streaks.append(curr_noise)

            for streak in noise_streaks:
                start_t = streak[0].bin_start
                end_t = streak[-1].bin_end
                dur = end_t - start_t
                avg_noise_rms = float(np.mean([s.audio_rms_energy for s in streak]))
                raw_suggestions.append(
                    TeachingSuggestion(
                        suggestion_id=f"REC_{rec_counter:02d}",
                        trigger_type="SUSTAINED_HIGH_NOISE",
                        time_interval=(start_t, end_t),
                        title="Elevated Acoustic Background Noise Detected",
                        observation=(
                            f"Acoustic noise energy averaged {avg_noise_rms:.3f} RMS "
                            f"(above {self.high_noise_threshold:.3f}) for {dur:.1f}s without detected vocal speech."
                        ),
                        suggestion=(
                            "Check room acoustics, ask attendees to mute secondary devices, "
                            "or inspect microphone placement for mechanical rumble."
                        ),
                        pedagogical_rationale="Elevated background noise increases cognitive load and degrades comprehension for learners.",
                        limitation_note="Microphone sensitivity and proximity to fans or doors may cause false noise spikes.",
                        priority="LOW",
                        confidence=0.80,
                        exact_metrics={
                            "average_noise_rms": round(avg_noise_rms, 4),
                            "threshold_rms": self.high_noise_threshold,
                            "duration_seconds": round(dur, 1),
                        },
                    )
                )
                rec_counter += 1

        # Rule 3: Detect Sudden Engagement Drop
        for i in range(1, len(engagement_results)):
            prev = engagement_results[i - 1]
            curr = engagement_results[i]
            delta = prev.score - curr.score
            if delta >= self.drop_delta and curr.level != "NO_DATA":
                raw_suggestions.append(
                    TeachingSuggestion(
                        suggestion_id=f"REC_{rec_counter:02d}",
                        trigger_type="SUDDEN_ENGAGEMENT_DROP",
                        time_interval=(curr.bin_start, curr.bin_end),
                        title="Noticeable Drop in Estimated Activity",
                        observation=(
                            f"Engagement score dropped by {delta:.1f} points "
                            f"(from {prev.score:.1f} to {curr.score:.1f}) at t={curr.bin_start:.1f}s."
                        ),
                        suggestion="Check if an abrupt topic transition occurred or if a visual slide was unclear.",
                        pedagogical_rationale="Rapid attention drops often correspond to sudden confusion or complex cognitive overload.",
                        limitation_note="Occurs naturally when transitioning between active discussion and quiet note-taking.",
                        priority="LOW",
                        confidence=0.75,
                        exact_metrics={
                            "score_drop": round(delta, 1),
                            "previous_score": round(prev.score, 1),
                            "current_score": round(curr.score, 1),
                        },
                    )
                )
                rec_counter += 1
                break  # Record principal drop event

        # Rule 4: Participation Surge (requires minimum duration >= min_suggestion_duration or 4.0s)
        surge_streaks: List[List[EngagementScoreResult]] = []
        curr_surge: List[EngagementScoreResult] = []

        for res in engagement_results:
            if res.score >= 80.0:
                curr_surge.append(res)
            else:
                if curr_surge:
                    dur = curr_surge[-1].bin_end - curr_surge[0].bin_start
                    if dur >= 4.0:
                        surge_streaks.append(curr_surge)
                    curr_surge = []

        if curr_surge:
            dur = curr_surge[-1].bin_end - curr_surge[0].bin_start
            if dur >= 4.0:
                surge_streaks.append(curr_surge)

        for surge in surge_streaks:
            start_t = surge[0].bin_start
            end_t = surge[-1].bin_end
            dur = end_t - start_t
            mean_high = float(np.mean([s.score for s in surge]))
            raw_suggestions.append(
                TeachingSuggestion(
                    suggestion_id=f"REC_{rec_counter:02d}",
                    trigger_type="HIGH_PARTICIPATION_SURGE",
                    time_interval=(start_t, end_t),
                    title="High Participant Interaction Segment",
                    observation=f"Sustained high engagement (average {mean_high:.1f} / 100) observed for {dur:.1f}s.",
                    suggestion="Note the pedagogical activity or prompt used during this segment as an effective practice to repeat.",
                    pedagogical_rationale="Reinforcing proven teaching strategies improves overall course engagement.",
                    limitation_note="High activity does not guarantee mastery, but indicates healthy classroom responsiveness.",
                    priority="INFO",
                    confidence=0.90,
                    exact_metrics={
                        "mean_engagement": round(mean_high, 1),
                        "duration_seconds": round(dur, 1),
                    },
                )
            )
            rec_counter += 1

        # Post-Processing: Remove Contradictions and Overlaps
        clean_suggestions = self._resolve_contradictions(raw_suggestions)
        return clean_suggestions

    def _resolve_contradictions(self, suggestions: List[TeachingSuggestion]) -> List[TeachingSuggestion]:
        """
        Removes contradictory suggestions spanning identical intervals
        (e.g., SUSTAINED_LOW_ENGAGEMENT vs HIGH_PARTICIPATION_SURGE).
        """
        if not suggestions:
            return []

        resolved: List[TeachingSuggestion] = []
        for s in suggestions:
            # Check for conflict with existing resolved suggestions
            is_contradiction = False
            for r in resolved:
                # Check interval overlap
                overlap = max(0.0, min(s.time_interval[1], r.time_interval[1]) - max(s.time_interval[0], r.time_interval[0]))
                if overlap > 1.0:
                    # Low engagement vs Surge contradiction
                    if {s.trigger_type, r.trigger_type} == {"SUSTAINED_LOW_ENGAGEMENT", "HIGH_PARTICIPATION_SURGE"}:
                        is_contradiction = True
                        break

            if not is_contradiction:
                resolved.append(s)

        # Re-number suggestions cleanly
        for idx, item in enumerate(resolved, 1):
            item.suggestion_id = f"REC_{idx:02d}"

        return resolved

    @staticmethod
    def to_dataframe(suggestions: List[TeachingSuggestion]) -> pd.DataFrame:
        """Converts teaching suggestions to a pandas DataFrame for reporting."""
        if not suggestions:
            return pd.DataFrame()
        return pd.DataFrame([s.to_dict() for s in suggestions])
