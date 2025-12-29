"""CV modules package."""

from .base import CVModule, DetectionResult
from .pipeline import CVPipeline, FrameResult

__all__ = ["CVModule", "DetectionResult", "CVPipeline", "FrameResult"]
