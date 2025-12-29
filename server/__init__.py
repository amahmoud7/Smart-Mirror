"""Smart Mirror Server - VLM reasoning backend."""

from .vlm.client import VLMClient, VLMClientSync, OllamaConfig

__all__ = ["VLMClient", "VLMClientSync", "OllamaConfig"]
