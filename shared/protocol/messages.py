"""Communication protocol between Jetson (edge) and PC (VLM server)."""

from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any
import json
import time


class MessageType(Enum):
    """Types of messages in the protocol."""
    # Edge -> Server
    CV_EVENT = "cv_event"               # CV pipeline event with optional frame
    HEALTH_UPDATE = "health_update"     # Periodic health data
    USER_QUERY = "user_query"           # User voice/text input
    SKIN_ANALYSIS_REQUEST = "skin_analysis_request"

    # Server -> Edge
    VLM_RESPONSE = "vlm_response"       # VLM text response
    UI_COMMAND = "ui_command"           # Command to update UI
    TTS_REQUEST = "tts_request"         # Text-to-speech request
    SKIN_ANALYSIS_RESULT = "skin_analysis_result"

    # Bidirectional
    HEARTBEAT = "heartbeat"             # Connection health check
    ERROR = "error"                     # Error message
    ACK = "ack"                         # Acknowledgment


@dataclass
class ProtocolMessage:
    """Base message structure for all communications."""
    type: MessageType
    payload: dict = field(default_factory=dict)
    message_id: str = ""
    timestamp: float = field(default_factory=time.time)
    source: str = ""  # "edge" or "server"

    def __post_init__(self):
        if not self.message_id:
            import uuid
            self.message_id = str(uuid.uuid4())[:8]

    def to_json(self) -> str:
        """Serialize to JSON string."""
        return json.dumps({
            "type": self.type.value,
            "payload": self.payload,
            "message_id": self.message_id,
            "timestamp": self.timestamp,
            "source": self.source
        })

    @classmethod
    def from_json(cls, json_str: str) -> "ProtocolMessage":
        """Deserialize from JSON string."""
        data = json.loads(json_str)
        return cls(
            type=MessageType(data["type"]),
            payload=data.get("payload", {}),
            message_id=data.get("message_id", ""),
            timestamp=data.get("timestamp", time.time()),
            source=data.get("source", "")
        )


# ----- Edge -> Server Messages -----

@dataclass
class CVEventMessage(ProtocolMessage):
    """CV pipeline event with context and optional frame."""

    def __init__(
        self,
        event_type: str,
        cv_context: dict,
        frame_base64: str | None = None,
        priority: int = 1
    ):
        super().__init__(
            type=MessageType.CV_EVENT,
            payload={
                "event_type": event_type,
                "cv_context": cv_context,
                "frame_base64": frame_base64,
                "priority": priority
            },
            source="edge"
        )


@dataclass
class UserQueryMessage(ProtocolMessage):
    """User voice or text input."""

    def __init__(
        self,
        query: str,
        cv_context: dict | None = None,
        frame_base64: str | None = None
    ):
        super().__init__(
            type=MessageType.USER_QUERY,
            payload={
                "query": query,
                "cv_context": cv_context,
                "frame_base64": frame_base64
            },
            source="edge"
        )


@dataclass
class SkinAnalysisRequest(ProtocolMessage):
    """Request for skin lesion analysis."""

    def __init__(
        self,
        frame_base64: str,
        roi: tuple[int, int, int, int] | None = None,
        patient_history: dict | None = None
    ):
        super().__init__(
            type=MessageType.SKIN_ANALYSIS_REQUEST,
            payload={
                "frame_base64": frame_base64,
                "roi": roi,
                "patient_history": patient_history or {}
            },
            source="edge"
        )


# ----- Server -> Edge Messages -----

@dataclass
class VLMResponseMessage(ProtocolMessage):
    """VLM response to be displayed/spoken."""

    def __init__(
        self,
        response_text: str,
        response_type: str = "general",  # general, recommendation, alert, greeting
        confidence: float = 1.0,
        metadata: dict | None = None
    ):
        super().__init__(
            type=MessageType.VLM_RESPONSE,
            payload={
                "response_text": response_text,
                "response_type": response_type,
                "confidence": confidence,
                "metadata": metadata or {}
            },
            source="server"
        )


@dataclass
class UICommandMessage(ProtocolMessage):
    """Command to update mirror UI."""

    def __init__(
        self,
        command: str,  # show_widget, hide_widget, update_value, show_alert
        target: str,   # widget identifier
        data: dict | None = None
    ):
        super().__init__(
            type=MessageType.UI_COMMAND,
            payload={
                "command": command,
                "target": target,
                "data": data or {}
            },
            source="server"
        )


@dataclass
class SkinAnalysisResult(ProtocolMessage):
    """Result of skin lesion analysis."""

    def __init__(
        self,
        classification: str,  # benign, malignant, uncertain
        confidence: float,
        explanation: str,
        recommendations: list[str],
        features_detected: dict | None = None
    ):
        super().__init__(
            type=MessageType.SKIN_ANALYSIS_RESULT,
            payload={
                "classification": classification,
                "confidence": confidence,
                "explanation": explanation,
                "recommendations": recommendations,
                "features_detected": features_detected or {}
            },
            source="server"
        )


@dataclass
class TTSRequestMessage(ProtocolMessage):
    """Request edge device to speak text."""

    def __init__(self, text: str, voice: str = "default", priority: int = 1):
        super().__init__(
            type=MessageType.TTS_REQUEST,
            payload={
                "text": text,
                "voice": voice,
                "priority": priority
            },
            source="server"
        )


# ----- Utility Messages -----

def create_heartbeat(source: str) -> ProtocolMessage:
    """Create heartbeat message."""
    return ProtocolMessage(
        type=MessageType.HEARTBEAT,
        source=source,
        payload={"alive": True}
    )


def create_ack(message_id: str, source: str) -> ProtocolMessage:
    """Create acknowledgment for a message."""
    return ProtocolMessage(
        type=MessageType.ACK,
        source=source,
        payload={"ack_message_id": message_id}
    )


def create_error(error_msg: str, source: str, related_id: str | None = None) -> ProtocolMessage:
    """Create error message."""
    return ProtocolMessage(
        type=MessageType.ERROR,
        source=source,
        payload={
            "error": error_msg,
            "related_message_id": related_id
        }
    )
