"""Shared modules between edge and server."""

from .protocol.messages import (
    MessageType,
    ProtocolMessage,
    CVEventMessage,
    UserQueryMessage,
    SkinAnalysisRequest,
    VLMResponseMessage,
    UICommandMessage,
    SkinAnalysisResult,
    TTSRequestMessage,
    create_heartbeat,
    create_ack,
    create_error
)

from .schemas.vlm_schemas import (
    VLMInferenceRequest,
    VLMInferenceResponse,
    SYSTEM_PROMPTS
)

__all__ = [
    "MessageType",
    "ProtocolMessage",
    "CVEventMessage",
    "UserQueryMessage",
    "SkinAnalysisRequest",
    "VLMResponseMessage",
    "UICommandMessage",
    "SkinAnalysisResult",
    "TTSRequestMessage",
    "create_heartbeat",
    "create_ack",
    "create_error",
    "VLMInferenceRequest",
    "VLMInferenceResponse",
    "SYSTEM_PROMPTS"
]
