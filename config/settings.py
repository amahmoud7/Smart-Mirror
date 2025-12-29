"""Configuration management for Smart Mirror."""

from dataclasses import dataclass, field
from pathlib import Path
import json
import os


@dataclass
class CameraConfig:
    """Camera settings."""
    device_id: int = 0
    width: int = 1280
    height: int = 720
    fps: int = 30
    flip_horizontal: bool = True  # Mirror effect


@dataclass
class FaceDetectionConfig:
    """Face detection module settings."""
    enabled: bool = True
    model: str = "yolov8n-face"  # yolov8n-face, retinaface, mtcnn
    tensorrt: bool = True
    confidence_threshold: float = 0.5
    tracking: bool = True


@dataclass
class PoseEstimationConfig:
    """Pose estimation module settings."""
    enabled: bool = True
    model: str = "movenet_thunder"  # movenet_thunder, movenet_lightning, mediapipe
    confidence_threshold: float = 0.3


@dataclass
class GestureRecognitionConfig:
    """Gesture recognition module settings."""
    enabled: bool = True
    max_hands: int = 2
    detection_confidence: float = 0.7
    tracking_confidence: float = 0.5
    gesture_smoothing_window: int = 3


@dataclass
class HealthSignalsConfig:
    """Health signal extraction settings."""
    enabled: bool = True
    method: str = "CHROM"  # GREEN, CHROM, POS
    buffer_seconds: int = 10
    hr_min_threshold: int = 50
    hr_max_threshold: int = 100


@dataclass
class SkinAnalysisConfig:
    """Skin analysis module settings."""
    enabled: bool = True
    model_path: str = ""  # Path to trained LSTM model
    use_vlm_fallback: bool = True  # Use VLM for camera-based analysis


@dataclass
class VLMConfig:
    """VLM server connection settings."""
    host: str = "localhost"
    port: int = 11434  # Ollama default
    model: str = "qwen2.5-vl:7b"
    timeout: int = 30
    max_retries: int = 3


@dataclass
class EventRouterConfig:
    """Event router settings."""
    min_request_interval: float = 1.0  # seconds between VLM requests
    presence_timeout: float = 3.0  # seconds before marking presence lost
    periodic_interval: float = 30.0  # seconds between periodic updates


@dataclass
class UIConfig:
    """Display/UI settings."""
    fullscreen: bool = True
    show_debug_overlay: bool = False
    theme: str = "dark"


@dataclass
class SmartMirrorConfig:
    """Master configuration for Smart Mirror."""
    camera: CameraConfig = field(default_factory=CameraConfig)
    face_detection: FaceDetectionConfig = field(default_factory=FaceDetectionConfig)
    pose_estimation: PoseEstimationConfig = field(default_factory=PoseEstimationConfig)
    gesture_recognition: GestureRecognitionConfig = field(default_factory=GestureRecognitionConfig)
    health_signals: HealthSignalsConfig = field(default_factory=HealthSignalsConfig)
    skin_analysis: SkinAnalysisConfig = field(default_factory=SkinAnalysisConfig)
    vlm: VLMConfig = field(default_factory=VLMConfig)
    event_router: EventRouterConfig = field(default_factory=EventRouterConfig)
    ui: UIConfig = field(default_factory=UIConfig)

    # Paths
    models_dir: str = "./models"
    logs_dir: str = "./logs"

    def to_dict(self) -> dict:
        """Convert to dictionary."""
        import dataclasses
        return dataclasses.asdict(self)

    def save(self, path: str | Path) -> None:
        """Save configuration to JSON file."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, 'w') as f:
            json.dump(self.to_dict(), f, indent=2)

    @classmethod
    def load(cls, path: str | Path) -> "SmartMirrorConfig":
        """Load configuration from JSON file."""
        with open(path) as f:
            data = json.load(f)
        return cls.from_dict(data)

    @classmethod
    def from_dict(cls, data: dict) -> "SmartMirrorConfig":
        """Create config from dictionary."""
        return cls(
            camera=CameraConfig(**data.get("camera", {})),
            face_detection=FaceDetectionConfig(**data.get("face_detection", {})),
            pose_estimation=PoseEstimationConfig(**data.get("pose_estimation", {})),
            gesture_recognition=GestureRecognitionConfig(**data.get("gesture_recognition", {})),
            health_signals=HealthSignalsConfig(**data.get("health_signals", {})),
            skin_analysis=SkinAnalysisConfig(**data.get("skin_analysis", {})),
            vlm=VLMConfig(**data.get("vlm", {})),
            event_router=EventRouterConfig(**data.get("event_router", {})),
            ui=UIConfig(**data.get("ui", {})),
            models_dir=data.get("models_dir", "./models"),
            logs_dir=data.get("logs_dir", "./logs")
        )

    @classmethod
    def load_or_create(cls, path: str | Path) -> "SmartMirrorConfig":
        """Load config if exists, otherwise create default."""
        path = Path(path)
        if path.exists():
            return cls.load(path)
        else:
            config = cls()
            config.save(path)
            return config


# Environment-specific configs
def get_jetson_config() -> SmartMirrorConfig:
    """Configuration optimized for Jetson Orin Nano."""
    config = SmartMirrorConfig()

    # Optimize for edge device
    config.camera.width = 640
    config.camera.height = 480
    config.camera.fps = 30

    # Use TensorRT-optimized models
    config.face_detection.tensorrt = True
    config.face_detection.model = "yolov8n-face"

    # Use lighter pose model
    config.pose_estimation.model = "movenet_lightning"

    return config


def get_development_config() -> SmartMirrorConfig:
    """Configuration for development/testing."""
    config = SmartMirrorConfig()

    # Higher resolution for debugging
    config.camera.width = 1280
    config.camera.height = 720

    # Show debug info
    config.ui.show_debug_overlay = True
    config.ui.fullscreen = False

    # More frequent updates for testing
    config.event_router.periodic_interval = 10.0

    return config
