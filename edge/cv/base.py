"""Base classes for CV pipeline modules."""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any
import numpy as np


@dataclass
class DetectionResult:
    """Base result for all detection modules."""
    timestamp: float
    confidence: float
    module_name: str
    raw_data: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "timestamp": self.timestamp,
            "confidence": self.confidence,
            "module": self.module_name,
            "data": self.raw_data
        }


@dataclass
class FaceResult(DetectionResult):
    """Face detection result."""
    bbox: tuple[int, int, int, int] = (0, 0, 0, 0)  # x, y, w, h
    landmarks: dict = field(default_factory=dict)
    face_id: int | None = None  # For tracking across frames

    def to_dict(self) -> dict:
        base = super().to_dict()
        base["data"].update({
            "bbox": self.bbox,
            "landmarks": self.landmarks,
            "face_id": self.face_id
        })
        return base


@dataclass
class PoseResult(DetectionResult):
    """Pose estimation result."""
    keypoints: dict = field(default_factory=dict)  # joint_name -> (x, y, confidence)

    def to_dict(self) -> dict:
        base = super().to_dict()
        base["data"]["keypoints"] = self.keypoints
        return base


@dataclass
class GestureResult(DetectionResult):
    """Gesture recognition result."""
    gesture_name: str = ""
    hand: str = "unknown"  # left, right, both

    def to_dict(self) -> dict:
        base = super().to_dict()
        base["data"].update({
            "gesture": self.gesture_name,
            "hand": self.hand
        })
        return base


@dataclass
class HealthSignalResult(DetectionResult):
    """Health signal extraction result."""
    signal_type: str = ""  # heart_rate, spo2, respiratory_rate, etc.
    value: float = 0.0
    unit: str = ""

    def to_dict(self) -> dict:
        base = super().to_dict()
        base["data"].update({
            "signal_type": self.signal_type,
            "value": self.value,
            "unit": self.unit
        })
        return base


@dataclass
class SkinAnalysisResult(DetectionResult):
    """Skin analysis result (for dermoscopic or camera-based)."""
    classification: str = ""  # benign, malignant, unknown
    lesion_features: dict = field(default_factory=dict)
    roi_bbox: tuple[int, int, int, int] = (0, 0, 0, 0)

    def to_dict(self) -> dict:
        base = super().to_dict()
        base["data"].update({
            "classification": self.classification,
            "lesion_features": self.lesion_features,
            "roi_bbox": self.roi_bbox
        })
        return base


class CVModule(ABC):
    """Abstract base class for all CV modules."""

    def __init__(self, config: dict | None = None):
        self.config = config or {}
        self._initialized = False

    @property
    @abstractmethod
    def name(self) -> str:
        """Module identifier."""
        pass

    @abstractmethod
    def initialize(self) -> None:
        """Load models and prepare for inference."""
        pass

    @abstractmethod
    def process(self, frame: np.ndarray, context: dict | None = None) -> DetectionResult | list[DetectionResult] | None:
        """
        Process a single frame.

        Args:
            frame: BGR image as numpy array
            context: Optional context from other modules (e.g., face ROI for health signals)

        Returns:
            Detection result(s) or None if nothing detected
        """
        pass

    def cleanup(self) -> None:
        """Release resources."""
        pass

    def __enter__(self):
        self.initialize()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.cleanup()
