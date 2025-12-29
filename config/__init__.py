"""Configuration management."""

from .settings import (
    SmartMirrorConfig,
    CameraConfig,
    FaceDetectionConfig,
    PoseEstimationConfig,
    GestureRecognitionConfig,
    HealthSignalsConfig,
    SkinAnalysisConfig,
    VLMConfig,
    EventRouterConfig,
    UIConfig,
    get_jetson_config,
    get_development_config
)

__all__ = [
    "SmartMirrorConfig",
    "CameraConfig",
    "FaceDetectionConfig",
    "PoseEstimationConfig",
    "GestureRecognitionConfig",
    "HealthSignalsConfig",
    "SkinAnalysisConfig",
    "VLMConfig",
    "EventRouterConfig",
    "UIConfig",
    "get_jetson_config",
    "get_development_config"
]
