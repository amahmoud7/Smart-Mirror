"""JSON schemas for VLM requests and responses."""

from dataclasses import dataclass, field
from typing import Literal


@dataclass
class VLMInferenceRequest:
    """
    Request structure for VLM inference.

    Designed for Ollama API compatibility with vision models.
    """
    model: str = "qwen2.5-vl:7b"
    prompt: str = ""
    images: list[str] = field(default_factory=list)  # Base64 encoded images
    system_prompt: str = ""
    context_data: dict = field(default_factory=dict)  # CV pipeline outputs
    stream: bool = False
    options: dict = field(default_factory=dict)  # Model parameters

    def to_ollama_format(self) -> dict:
        """Convert to Ollama API format."""
        messages = []

        if self.system_prompt:
            messages.append({
                "role": "system",
                "content": self.system_prompt
            })

        # Build user message with image(s)
        user_content = self.prompt
        if self.context_data:
            user_content += f"\n\nContext from sensors:\n```json\n{self.context_data}\n```"

        user_message = {"role": "user", "content": user_content}
        if self.images:
            user_message["images"] = self.images

        messages.append(user_message)

        return {
            "model": self.model,
            "messages": messages,
            "stream": self.stream,
            "options": self.options
        }


# System prompts for different scenarios
SYSTEM_PROMPTS = {
    "general": """You are an AI assistant integrated into a smart mirror. You can see the user through the mirror's camera and have access to real-time data from various sensors.

Your capabilities:
- See and describe what's in front of the mirror
- Analyze facial expressions and body language
- Monitor health signals (heart rate when available)
- Recognize gestures
- Provide health and wellness recommendations

Be conversational, helpful, and concise. The user is looking at themselves in the mirror while interacting with you.""",

    "health_focus": """You are a health-focused AI assistant in a smart mirror. You have access to the user's vital signs and visual appearance.

When analyzing health signals:
- Be informative but not alarmist
- Suggest consulting healthcare professionals for concerns
- Focus on wellness and lifestyle recommendations
- Never diagnose medical conditions

Current health data will be provided in the context.""",

    "skin_analysis": """You are assisting with skin health awareness through a smart mirror. You can see images of the user's skin.

IMPORTANT: You are NOT providing medical diagnosis. You are:
- Helping users track changes in their skin over time
- Identifying features that may warrant professional evaluation (asymmetry, irregular borders, color variation, diameter)
- Recommending dermatologist visits when appropriate
- Providing general skin health education

Always recommend professional medical evaluation for any concerning features. Use the ABCDE criteria as a framework for discussion, not diagnosis.""",

    "greeting": """You are a friendly AI in a smart mirror. The user has just appeared in front of the mirror.

Provide a brief, warm greeting. You can:
- Comment on something you observe (without being intrusive)
- Mention the time of day appropriately
- Ask if they need assistance

Keep greetings under 2 sentences.""",

    "gesture_response": """You are responding to a gesture command from the user. The gesture type will be provided in the context.

Respond appropriately to the gesture:
- open_palm: User wants your attention
- thumbs_up: User is confirming/agreeing
- thumbs_down: User is declining/disagreeing
- wave: User is greeting or saying goodbye

Keep responses brief and action-oriented."""
}


@dataclass
class VLMInferenceResponse:
    """Response from VLM inference."""
    text: str = ""
    model: str = ""
    done: bool = True
    context: list[int] | None = None  # For conversation continuity
    total_duration: int = 0  # nanoseconds
    eval_count: int = 0

    @classmethod
    def from_ollama_response(cls, response: dict) -> "VLMInferenceResponse":
        """Parse Ollama API response."""
        return cls(
            text=response.get("message", {}).get("content", ""),
            model=response.get("model", ""),
            done=response.get("done", True),
            context=response.get("context"),
            total_duration=response.get("total_duration", 0),
            eval_count=response.get("eval_count", 0)
        )


# Request templates for common scenarios
def create_presence_request(cv_context: dict, image_b64: str | None = None) -> VLMInferenceRequest:
    """Create request for when user appears."""
    return VLMInferenceRequest(
        system_prompt=SYSTEM_PROMPTS["greeting"],
        prompt="A person has appeared in front of the smart mirror. Greet them appropriately based on what you observe.",
        images=[image_b64] if image_b64 else [],
        context_data=cv_context
    )


def create_gesture_request(gesture: str, cv_context: dict) -> VLMInferenceRequest:
    """Create request for gesture response."""
    return VLMInferenceRequest(
        system_prompt=SYSTEM_PROMPTS["gesture_response"],
        prompt=f"The user made a '{gesture}' gesture. Respond appropriately.",
        context_data=cv_context
    )


def create_health_query_request(cv_context: dict, user_query: str | None = None) -> VLMInferenceRequest:
    """Create request for health-related query."""
    prompt = user_query or "Provide a brief health status update based on the available sensor data."
    return VLMInferenceRequest(
        system_prompt=SYSTEM_PROMPTS["health_focus"],
        prompt=prompt,
        context_data=cv_context
    )


def create_skin_analysis_request(image_b64: str, cv_context: dict | None = None) -> VLMInferenceRequest:
    """Create request for skin analysis."""
    return VLMInferenceRequest(
        system_prompt=SYSTEM_PROMPTS["skin_analysis"],
        prompt="""Analyze this skin image. Describe what you observe using the ABCDE framework:
- Asymmetry
- Border irregularity
- Color variation
- Diameter
- Evolution (if history provided)

Provide your observations and recommend whether professional evaluation is warranted.""",
        images=[image_b64],
        context_data=cv_context or {}
    )


def create_general_query_request(
    user_query: str,
    cv_context: dict,
    image_b64: str | None = None
) -> VLMInferenceRequest:
    """Create request for general user query."""
    return VLMInferenceRequest(
        system_prompt=SYSTEM_PROMPTS["general"],
        prompt=user_query,
        images=[image_b64] if image_b64 else [],
        context_data=cv_context
    )
