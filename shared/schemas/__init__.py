"""Schema definitions."""

from .vlm_schemas import (
    VLMInferenceRequest,
    VLMInferenceResponse,
    SYSTEM_PROMPTS,
    create_presence_request,
    create_gesture_request,
    create_health_query_request,
    create_skin_analysis_request,
    create_general_query_request
)

__all__ = [
    "VLMInferenceRequest",
    "VLMInferenceResponse",
    "SYSTEM_PROMPTS",
    "create_presence_request",
    "create_gesture_request",
    "create_health_query_request",
    "create_skin_analysis_request",
    "create_general_query_request"
]
