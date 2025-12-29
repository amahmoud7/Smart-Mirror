"""Smart Mirror Edge - CV pipeline for Jetson Orin Nano."""

from .cv.pipeline import CVPipeline, FrameResult
from .cv.base import (
    CVModule,
    DetectionResult,
    FaceResult,
    PoseResult,
    GestureResult,
    HealthSignalResult,
    SkinAnalysisResult
)
from .router.event_router import EventRouter, VLMRequest, EventType

__all__ = [
    "CVPipeline",
    "FrameResult",
    "CVModule",
    "DetectionResult",
    "FaceResult",
    "PoseResult",
    "GestureResult",
    "HealthSignalResult",
    "SkinAnalysisResult",
    "EventRouter",
    "VLMRequest",
    "EventType"
]
