"""Event router - decides what gets sent to the VLM."""

import time
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Callable
import numpy as np

from ..cv.pipeline import FrameResult


class EventType(Enum):
    """Types of events that can trigger VLM requests."""
    PRESENCE_DETECTED = auto()      # Person appeared in view
    PRESENCE_LOST = auto()          # Person left view
    GESTURE_COMMAND = auto()        # Actionable gesture detected
    HEALTH_ANOMALY = auto()         # Unusual health reading
    PERIODIC_UPDATE = auto()        # Regular status update
    USER_INITIATED = auto()         # User explicitly requested interaction
    SKIN_CHECK_REQUEST = auto()     # User wants skin analysis


@dataclass
class VLMRequest:
    """Request to be sent to VLM backend."""
    event_type: EventType
    timestamp: float
    frame: np.ndarray | None = None             # Optional frame for visual analysis
    frame_base64: str | None = None             # Encoded frame for transmission
    cv_context: dict = field(default_factory=dict)  # Aggregated CV results
    user_query: str | None = None               # Optional user text/voice input
    priority: int = 1                           # 1=normal, 2=high, 3=urgent

    def to_dict(self) -> dict:
        """Serialize for transmission (excluding raw frame)."""
        return {
            "event_type": self.event_type.name,
            "timestamp": self.timestamp,
            "frame_base64": self.frame_base64,
            "cv_context": self.cv_context,
            "user_query": self.user_query,
            "priority": self.priority
        }


class EventRouter:
    """
    Routes CV pipeline outputs to VLM requests.

    Implements policies for:
    - When to send frames to VLM (not every frame!)
    - What context to include
    - Priority handling
    - Rate limiting
    """

    def __init__(self, config: dict | None = None):
        self.config = config or {}

        # Rate limiting
        self._last_vlm_request_time = 0.0
        self._min_request_interval = self.config.get("min_request_interval", 1.0)  # seconds

        # Presence tracking
        self._presence_detected = False
        self._presence_lost_time = 0.0
        self._presence_timeout = self.config.get("presence_timeout", 3.0)

        # Periodic updates
        self._last_periodic_update = 0.0
        self._periodic_interval = self.config.get("periodic_interval", 30.0)

        # Health thresholds
        self._hr_min = self.config.get("hr_min", 50)
        self._hr_max = self.config.get("hr_max", 100)

        # Actionable gestures
        self._command_gestures = {
            "open_palm": "attention",
            "thumbs_up": "confirm",
            "thumbs_down": "reject",
            "wave": "greeting"
        }

        # Frame history for context
        self._recent_results: list[FrameResult] = []
        self._max_history = 10

    def process(
        self,
        frame_result: FrameResult,
        frame: np.ndarray | None = None,
        user_query: str | None = None
    ) -> VLMRequest | None:
        """
        Evaluate frame result and decide if VLM request is needed.

        Args:
            frame_result: Output from CV pipeline
            frame: Raw frame (included only if VLM needs visual analysis)
            user_query: Optional user input triggering the request

        Returns:
            VLMRequest if conditions are met, None otherwise
        """
        self._update_history(frame_result)
        current_time = frame_result.timestamp

        # User-initiated always goes through
        if user_query:
            return self._create_request(
                EventType.USER_INITIATED,
                frame_result,
                frame,
                user_query,
                priority=2
            )

        # Check rate limit
        if current_time - self._last_vlm_request_time < self._min_request_interval:
            return None

        # Check for events
        request = None

        # Presence detection
        presence_event = self._check_presence(frame_result, current_time)
        if presence_event:
            request = self._create_request(presence_event, frame_result, frame)

        # Gesture commands (higher priority)
        gesture_event = self._check_gestures(frame_result)
        if gesture_event:
            request = self._create_request(
                EventType.GESTURE_COMMAND,
                frame_result,
                frame,
                priority=2
            )

        # Health anomalies (highest priority)
        health_event = self._check_health(frame_result)
        if health_event:
            request = self._create_request(
                EventType.HEALTH_ANOMALY,
                frame_result,
                frame,
                priority=3
            )

        # Periodic updates (low priority)
        if request is None and self._should_send_periodic(current_time):
            request = self._create_request(
                EventType.PERIODIC_UPDATE,
                frame_result,
                frame=None  # No frame for periodic updates
            )

        if request:
            self._last_vlm_request_time = current_time

        return request

    def _check_presence(self, result: FrameResult, current_time: float) -> EventType | None:
        """Detect presence changes."""
        has_face = bool(result.faces)

        if has_face and not self._presence_detected:
            self._presence_detected = True
            return EventType.PRESENCE_DETECTED

        if not has_face and self._presence_detected:
            if self._presence_lost_time == 0:
                self._presence_lost_time = current_time
            elif current_time - self._presence_lost_time > self._presence_timeout:
                self._presence_detected = False
                self._presence_lost_time = 0
                return EventType.PRESENCE_LOST

        if has_face:
            self._presence_lost_time = 0

        return None

    def _check_gestures(self, result: FrameResult) -> bool:
        """Check for actionable gestures."""
        for gesture in result.gestures:
            if gesture.gesture_name in self._command_gestures:
                if gesture.confidence > 0.8:
                    return True
        return False

    def _check_health(self, result: FrameResult) -> bool:
        """Check for health anomalies."""
        if result.health is None:
            return False

        if result.health.confidence < 0.5:
            return False

        hr = result.health.value
        if hr < self._hr_min or hr > self._hr_max:
            return True

        return False

    def _should_send_periodic(self, current_time: float) -> bool:
        """Check if periodic update is due."""
        if not self._presence_detected:
            return False

        if current_time - self._last_periodic_update > self._periodic_interval:
            self._last_periodic_update = current_time
            return True

        return False

    def _create_request(
        self,
        event_type: EventType,
        frame_result: FrameResult,
        frame: np.ndarray | None = None,
        user_query: str | None = None,
        priority: int = 1
    ) -> VLMRequest:
        """Create VLM request with context."""
        import base64
        import cv2

        frame_b64 = None
        if frame is not None:
            # Encode frame as JPEG base64
            _, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
            frame_b64 = base64.b64encode(buffer).decode('utf-8')

        # Build context from recent history
        cv_context = self._build_context(frame_result)

        return VLMRequest(
            event_type=event_type,
            timestamp=frame_result.timestamp,
            frame=frame,
            frame_base64=frame_b64,
            cv_context=cv_context,
            user_query=user_query,
            priority=priority
        )

    def _build_context(self, current: FrameResult) -> dict:
        """Build context from current and recent results."""
        context = {
            "current": current.to_dict(),
            "summary": {
                "faces_detected": len(current.faces),
                "pose_detected": current.pose is not None,
                "active_gestures": [g.gesture_name for g in current.gestures],
                "heart_rate": current.health.value if current.health else None,
                "presence_duration": self._calculate_presence_duration()
            }
        }
        return context

    def _calculate_presence_duration(self) -> float:
        """Calculate how long user has been present."""
        if not self._presence_detected or not self._recent_results:
            return 0.0
        return self._recent_results[-1].timestamp - self._recent_results[0].timestamp

    def _update_history(self, result: FrameResult) -> None:
        """Update result history."""
        self._recent_results.append(result)
        if len(self._recent_results) > self._max_history:
            self._recent_results.pop(0)

    def request_skin_check(self, frame: np.ndarray, frame_result: FrameResult) -> VLMRequest:
        """Explicitly request skin analysis."""
        return self._create_request(
            EventType.SKIN_CHECK_REQUEST,
            frame_result,
            frame,
            priority=2
        )
