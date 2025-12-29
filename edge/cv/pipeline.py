"""CV Pipeline coordinator - orchestrates all CV modules."""

import time
from dataclasses import dataclass, field
from typing import Any
import numpy as np

from .base import CVModule, DetectionResult
from .detectors.face import FaceDetector
from .estimators.pose import PoseEstimator
from .estimators.gesture import GestureRecognizer
from .health.rppg import HeartRateEstimator


@dataclass
class FrameResult:
    """Aggregated results from all CV modules for a single frame."""
    timestamp: float
    frame_id: int
    faces: list = field(default_factory=list)
    pose: dict | None = None
    gestures: list = field(default_factory=list)
    health: dict | None = None
    skin_analysis: dict | None = None

    def to_dict(self) -> dict:
        return {
            "timestamp": self.timestamp,
            "frame_id": self.frame_id,
            "faces": [f.to_dict() for f in self.faces],
            "pose": self.pose.to_dict() if self.pose else None,
            "gestures": [g.to_dict() for g in self.gestures],
            "health": self.health.to_dict() if self.health else None,
            "skin_analysis": self.skin_analysis.to_dict() if self.skin_analysis else None
        }

    def has_detections(self) -> bool:
        """Check if any meaningful detections occurred."""
        return bool(self.faces or self.pose or self.gestures or self.health)


class CVPipeline:
    """
    Coordinates all CV modules in a unified pipeline.

    Manages:
    - Module lifecycle (init, process, cleanup)
    - Data flow between modules (face ROI -> health signals)
    - Result aggregation
    """

    def __init__(self, config: dict | None = None):
        self.config = config or {}
        self._frame_count = 0

        # Initialize modules based on config
        self.modules: dict[str, CVModule] = {}

        if self.config.get("face_detection", {}).get("enabled", True):
            self.modules["face"] = FaceDetector(self.config.get("face_detection"))

        if self.config.get("pose_estimation", {}).get("enabled", True):
            self.modules["pose"] = PoseEstimator(self.config.get("pose_estimation"))

        if self.config.get("gesture_recognition", {}).get("enabled", True):
            self.modules["gesture"] = GestureRecognizer(self.config.get("gesture_recognition"))

        if self.config.get("health_signals", {}).get("enabled", True):
            self.modules["health"] = HeartRateEstimator(self.config.get("health_signals"))

    def initialize(self) -> None:
        """Initialize all modules."""
        for name, module in self.modules.items():
            try:
                module.initialize()
            except Exception as e:
                raise RuntimeError(f"Failed to initialize {name}: {e}")

    def process_frame(self, frame: np.ndarray) -> FrameResult:
        """
        Process a single frame through all modules.

        Args:
            frame: BGR image as numpy array

        Returns:
            FrameResult with all detections
        """
        timestamp = time.time()
        self._frame_count += 1

        result = FrameResult(
            timestamp=timestamp,
            frame_id=self._frame_count
        )

        context = {}

        # Face detection (run first, provides context for others)
        if "face" in self.modules:
            faces = self.modules["face"].process(frame, context)
            result.faces = faces if faces else []

            # Provide face ROI for health signals
            if faces:
                primary_face = max(faces, key=lambda f: f.bbox[2] * f.bbox[3])  # Largest face
                context["face_roi"] = primary_face.bbox
                context["face_landmarks"] = primary_face.landmarks

        # Pose estimation
        if "pose" in self.modules:
            pose = self.modules["pose"].process(frame, context)
            result.pose = pose

        # Gesture recognition
        if "gesture" in self.modules:
            gestures = self.modules["gesture"].process(frame, context)
            result.gestures = gestures if gestures else []

        # Health signals (requires face ROI)
        if "health" in self.modules and "face_roi" in context:
            health = self.modules["health"].process(frame, context)
            result.health = health

        return result

    def cleanup(self) -> None:
        """Cleanup all modules."""
        for module in self.modules.values():
            module.cleanup()

    def __enter__(self):
        self.initialize()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.cleanup()

    @property
    def enabled_modules(self) -> list[str]:
        """List of enabled module names."""
        return list(self.modules.keys())
