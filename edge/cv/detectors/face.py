"""Face detection and tracking module."""

import time
import numpy as np

from ..base import CVModule, FaceResult


class FaceDetector(CVModule):
    """
    Face detection using YOLOv8n-face or RetinaFace.

    Optimized for Jetson with TensorRT acceleration.
    """

    def __init__(self, config: dict | None = None):
        super().__init__(config)
        self.model = None
        self.tracker = None  # For face ID persistence across frames
        self._next_face_id = 0

    @property
    def name(self) -> str:
        return "face_detector"

    def initialize(self) -> None:
        """Load face detection model."""
        model_type = self.config.get("model", "yolov8n-face")
        use_tensorrt = self.config.get("tensorrt", True)

        # TODO: Load actual model
        # Options:
        # - YOLOv8n-face (ultralytics)
        # - RetinaFace (insightface)
        # - MTCNN (facenet-pytorch)

        self._initialized = True

    def process(self, frame: np.ndarray, context: dict | None = None) -> list[FaceResult]:
        """
        Detect faces in frame.

        Returns:
            List of FaceResult for each detected face
        """
        if not self._initialized:
            raise RuntimeError("Module not initialized. Call initialize() first.")

        timestamp = time.time()
        results = []

        # TODO: Actual inference
        # detections = self.model(frame)
        # for det in detections:
        #     result = FaceResult(
        #         timestamp=timestamp,
        #         confidence=det.conf,
        #         module_name=self.name,
        #         bbox=(det.x, det.y, det.w, det.h),
        #         landmarks=det.landmarks,
        #         face_id=self._assign_face_id(det)
        #     )
        #     results.append(result)

        return results

    def _assign_face_id(self, detection) -> int:
        """Assign persistent ID using tracker."""
        # TODO: Implement tracking (e.g., SORT, DeepSORT, ByteTrack)
        face_id = self._next_face_id
        self._next_face_id += 1
        return face_id

    def cleanup(self) -> None:
        """Release model resources."""
        self.model = None
        self.tracker = None
        self._initialized = False
