"""Multimodal Agent module for ClassPulse AI."""

from agent.fusion import MultimodalFusion, FusedObservation
from agent.engagement import EngagementEstimator, EngagementScoreResult
from agent.recommendations import RecommendationEngine, TeachingSuggestion

__all__ = [
    "MultimodalFusion",
    "FusedObservation",
    "EngagementEstimator",
    "EngagementScoreResult",
    "RecommendationEngine",
    "TeachingSuggestion",
]
