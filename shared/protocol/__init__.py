"""Protocol definitions."""

from .messages import (
    MessageType,
    ProtocolMessage,
    CVEventMessage,
    UserQueryMessage,
    SkinAnalysisRequest,
    VLMResponseMessage,
    UICommandMessage,
    TTSRequestMessage,
    create_heartbeat,
    create_ack,
    create_error
)

__all__ = [
    "MessageType",
    "ProtocolMessage",
    "CVEventMessage",
    "UserQueryMessage",
    "SkinAnalysisRequest",
    "VLMResponseMessage",
    "UICommandMessage",
    "TTSRequestMessage",
    "create_heartbeat",
    "create_ack",
    "create_error"
]
