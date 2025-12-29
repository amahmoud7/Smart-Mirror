"""VLM client for Ollama API."""

import asyncio
import aiohttp
import json
from dataclasses import dataclass
from typing import AsyncGenerator

from ...shared.schemas.vlm_schemas import (
    VLMInferenceRequest,
    VLMInferenceResponse,
    SYSTEM_PROMPTS,
    create_presence_request,
    create_gesture_request,
    create_health_query_request,
    create_skin_analysis_request,
    create_general_query_request
)


@dataclass
class OllamaConfig:
    """Ollama server configuration."""
    host: str = "localhost"
    port: int = 11434
    model: str = "qwen2.5-vl:7b"
    timeout: int = 60
    max_retries: int = 3

    @property
    def base_url(self) -> str:
        return f"http://{self.host}:{self.port}"


class VLMClient:
    """
    Async client for Ollama VLM API.

    Supports:
    - Chat completions with vision
    - Streaming responses
    - Automatic retries
    - Connection health checks
    """

    def __init__(self, config: OllamaConfig | None = None):
        self.config = config or OllamaConfig()
        self._session: aiohttp.ClientSession | None = None
        self._conversation_context: list[int] | None = None

    async def __aenter__(self):
        await self.connect()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self.disconnect()

    async def connect(self) -> None:
        """Initialize HTTP session."""
        timeout = aiohttp.ClientTimeout(total=self.config.timeout)
        self._session = aiohttp.ClientSession(timeout=timeout)

    async def disconnect(self) -> None:
        """Close HTTP session."""
        if self._session:
            await self._session.close()
            self._session = None

    async def health_check(self) -> bool:
        """Check if Ollama server is available."""
        if not self._session:
            return False

        try:
            async with self._session.get(f"{self.config.base_url}/api/tags") as resp:
                return resp.status == 200
        except Exception:
            return False

    async def list_models(self) -> list[str]:
        """List available models on Ollama server."""
        if not self._session:
            raise RuntimeError("Not connected. Call connect() first.")

        async with self._session.get(f"{self.config.base_url}/api/tags") as resp:
            data = await resp.json()
            return [model["name"] for model in data.get("models", [])]

    async def infer(self, request: VLMInferenceRequest) -> VLMInferenceResponse:
        """
        Send inference request to VLM.

        Args:
            request: VLM inference request with prompt and optional images

        Returns:
            VLM response with generated text
        """
        if not self._session:
            raise RuntimeError("Not connected. Call connect() first.")

        # Override model if not set
        if not request.model:
            request.model = self.config.model

        payload = request.to_ollama_format()

        last_error = None
        for attempt in range(self.config.max_retries):
            try:
                async with self._session.post(
                    f"{self.config.base_url}/api/chat",
                    json=payload
                ) as resp:
                    if resp.status != 200:
                        error_text = await resp.text()
                        raise RuntimeError(f"Ollama error {resp.status}: {error_text}")

                    data = await resp.json()
                    response = VLMInferenceResponse.from_ollama_response(data)

                    # Store context for conversation continuity
                    if response.context:
                        self._conversation_context = response.context

                    return response

            except asyncio.TimeoutError:
                last_error = "Request timed out"
            except aiohttp.ClientError as e:
                last_error = str(e)

            # Wait before retry
            if attempt < self.config.max_retries - 1:
                await asyncio.sleep(1 * (attempt + 1))

        raise RuntimeError(f"Failed after {self.config.max_retries} attempts: {last_error}")

    async def infer_stream(
        self,
        request: VLMInferenceRequest
    ) -> AsyncGenerator[str, None]:
        """
        Stream inference response token by token.

        Yields:
            Response text chunks as they are generated
        """
        if not self._session:
            raise RuntimeError("Not connected. Call connect() first.")

        if not request.model:
            request.model = self.config.model

        payload = request.to_ollama_format()
        payload["stream"] = True

        async with self._session.post(
            f"{self.config.base_url}/api/chat",
            json=payload
        ) as resp:
            if resp.status != 200:
                error_text = await resp.text()
                raise RuntimeError(f"Ollama error {resp.status}: {error_text}")

            async for line in resp.content:
                if line:
                    try:
                        data = json.loads(line)
                        if "message" in data and "content" in data["message"]:
                            yield data["message"]["content"]
                        if data.get("done"):
                            break
                    except json.JSONDecodeError:
                        continue

    def clear_conversation(self) -> None:
        """Clear conversation context for new session."""
        self._conversation_context = None

    # ----- Convenience Methods -----

    async def greet_user(self, cv_context: dict, image_b64: str | None = None) -> str:
        """Generate greeting for newly detected user."""
        request = create_presence_request(cv_context, image_b64)
        response = await self.infer(request)
        return response.text

    async def respond_to_gesture(self, gesture: str, cv_context: dict) -> str:
        """Generate response to gesture command."""
        request = create_gesture_request(gesture, cv_context)
        response = await self.infer(request)
        return response.text

    async def analyze_health(self, cv_context: dict, user_query: str | None = None) -> str:
        """Provide health-related insights."""
        request = create_health_query_request(cv_context, user_query)
        response = await self.infer(request)
        return response.text

    async def analyze_skin(self, image_b64: str, cv_context: dict | None = None) -> str:
        """Analyze skin image for concerning features."""
        request = create_skin_analysis_request(image_b64, cv_context)
        response = await self.infer(request)
        return response.text

    async def respond_to_query(
        self,
        query: str,
        cv_context: dict,
        image_b64: str | None = None
    ) -> str:
        """Respond to general user query."""
        request = create_general_query_request(query, cv_context, image_b64)
        response = await self.infer(request)
        return response.text


# Synchronous wrapper for non-async code
class VLMClientSync:
    """Synchronous wrapper for VLMClient."""

    def __init__(self, config: OllamaConfig | None = None):
        self.config = config or OllamaConfig()
        self._client = VLMClient(config)
        self._loop: asyncio.AbstractEventLoop | None = None

    def _get_loop(self) -> asyncio.AbstractEventLoop:
        if self._loop is None or self._loop.is_closed():
            self._loop = asyncio.new_event_loop()
        return self._loop

    def connect(self) -> None:
        self._get_loop().run_until_complete(self._client.connect())

    def disconnect(self) -> None:
        if self._loop:
            self._loop.run_until_complete(self._client.disconnect())
            self._loop.close()
            self._loop = None

    def health_check(self) -> bool:
        return self._get_loop().run_until_complete(self._client.health_check())

    def infer(self, request: VLMInferenceRequest) -> VLMInferenceResponse:
        return self._get_loop().run_until_complete(self._client.infer(request))

    def greet_user(self, cv_context: dict, image_b64: str | None = None) -> str:
        return self._get_loop().run_until_complete(
            self._client.greet_user(cv_context, image_b64)
        )

    def respond_to_gesture(self, gesture: str, cv_context: dict) -> str:
        return self._get_loop().run_until_complete(
            self._client.respond_to_gesture(gesture, cv_context)
        )

    def analyze_health(self, cv_context: dict, user_query: str | None = None) -> str:
        return self._get_loop().run_until_complete(
            self._client.analyze_health(cv_context, user_query)
        )

    def analyze_skin(self, image_b64: str, cv_context: dict | None = None) -> str:
        return self._get_loop().run_until_complete(
            self._client.analyze_skin(image_b64, cv_context)
        )

    def respond_to_query(
        self,
        query: str,
        cv_context: dict,
        image_b64: str | None = None
    ) -> str:
        return self._get_loop().run_until_complete(
            self._client.respond_to_query(query, cv_context, image_b64)
        )

    def __enter__(self):
        self.connect()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.disconnect()
