"""Smart Mirror Server - VLM reasoning backend."""

import asyncio
import json
import sys
from pathlib import Path

# Add parent to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from config import SmartMirrorConfig
from server.vlm.client import VLMClient, OllamaConfig
from shared.protocol.messages import (
    MessageType,
    ProtocolMessage,
    VLMResponseMessage,
    SkinAnalysisResult as SkinAnalysisResultMsg
)
from shared.schemas.vlm_schemas import (
    create_presence_request,
    create_gesture_request,
    create_health_query_request,
    create_skin_analysis_request,
    create_general_query_request
)


class SmartMirrorServer:
    """VLM reasoning server for Smart Mirror."""

    def __init__(self, config: SmartMirrorConfig):
        self.config = config
        self.running = False

        # VLM client
        ollama_config = OllamaConfig(
            host=config.vlm.host,
            port=config.vlm.port,
            model=config.vlm.model,
            timeout=config.vlm.timeout
        )
        self.vlm_client = VLMClient(ollama_config)

    async def start(self) -> None:
        """Start the server."""
        print("Starting Smart Mirror Server...")

        # Connect to Ollama
        await self.vlm_client.connect()

        # Check health
        if await self.vlm_client.health_check():
            print("Connected to Ollama")
            models = await self.vlm_client.list_models()
            print(f"Available models: {models}")
        else:
            print("WARNING: Could not connect to Ollama. Make sure it's running.")
            print("  Install: https://ollama.ai")
            print(f"  Pull model: ollama pull {self.config.vlm.model}")

        self.running = True
        print(f"Server running. VLM model: {self.config.vlm.model}")

        # In production, this would be a FastAPI server
        # For now, demonstrate VLM capability
        await self._demo_vlm()

    async def _demo_vlm(self) -> None:
        """Demonstrate VLM capabilities."""
        print("\n--- VLM Demo ---")

        # Check if VLM is available
        if not await self.vlm_client.health_check():
            print("VLM not available. Skipping demo.")
            return

        # Demo: Greeting
        print("\n1. Greeting a user:")
        cv_context = {
            "summary": {
                "faces_detected": 1,
                "pose_detected": True,
                "heart_rate": 72,
                "presence_duration": 2.5
            }
        }

        try:
            greeting = await self.vlm_client.greet_user(cv_context)
            print(f"   VLM: {greeting}")
        except Exception as e:
            print(f"   Error: {e}")

        # Demo: Health query
        print("\n2. Health status:")
        cv_context["summary"]["heart_rate"] = 85

        try:
            health_response = await self.vlm_client.analyze_health(cv_context)
            print(f"   VLM: {health_response}")
        except Exception as e:
            print(f"   Error: {e}")

        # Demo: Gesture response
        print("\n3. Gesture response (thumbs up):")
        try:
            gesture_response = await self.vlm_client.respond_to_gesture("thumbs_up", cv_context)
            print(f"   VLM: {gesture_response}")
        except Exception as e:
            print(f"   Error: {e}")

        print("\n--- Demo Complete ---")

    async def handle_message(self, message: ProtocolMessage) -> ProtocolMessage | None:
        """Handle incoming message from edge device."""
        if message.type == MessageType.CV_EVENT:
            return await self._handle_cv_event(message)
        elif message.type == MessageType.USER_QUERY:
            return await self._handle_user_query(message)
        elif message.type == MessageType.SKIN_ANALYSIS_REQUEST:
            return await self._handle_skin_analysis(message)
        elif message.type == MessageType.HEARTBEAT:
            from shared.protocol.messages import create_ack
            return create_ack(message.message_id, "server")

        return None

    async def _handle_cv_event(self, message: ProtocolMessage) -> VLMResponseMessage:
        """Handle CV event from edge."""
        payload = message.payload
        event_type = payload.get("event_type", "")
        cv_context = payload.get("cv_context", {})
        frame_b64 = payload.get("frame_base64")

        if event_type == "PRESENCE_DETECTED":
            response = await self.vlm_client.greet_user(cv_context, frame_b64)
            return VLMResponseMessage(response, "greeting")

        elif event_type == "GESTURE_COMMAND":
            gestures = cv_context.get("summary", {}).get("active_gestures", [])
            gesture = gestures[0] if gestures else "unknown"
            response = await self.vlm_client.respond_to_gesture(gesture, cv_context)
            return VLMResponseMessage(response, "gesture_response")

        elif event_type == "HEALTH_ANOMALY":
            response = await self.vlm_client.analyze_health(cv_context)
            return VLMResponseMessage(response, "alert", confidence=0.8)

        else:
            # General periodic update
            response = await self.vlm_client.analyze_health(cv_context)
            return VLMResponseMessage(response, "general")

    async def _handle_user_query(self, message: ProtocolMessage) -> VLMResponseMessage:
        """Handle user text/voice query."""
        payload = message.payload
        query = payload.get("query", "")
        cv_context = payload.get("cv_context", {})
        frame_b64 = payload.get("frame_base64")

        response = await self.vlm_client.respond_to_query(query, cv_context, frame_b64)
        return VLMResponseMessage(response, "general")

    async def _handle_skin_analysis(self, message: ProtocolMessage) -> SkinAnalysisResultMsg:
        """Handle skin analysis request."""
        payload = message.payload
        frame_b64 = payload.get("frame_base64", "")
        cv_context = payload.get("patient_history", {})

        response = await self.vlm_client.analyze_skin(frame_b64, cv_context)

        # Parse VLM response into structured result
        # In production, would use more structured prompting
        return SkinAnalysisResultMsg(
            classification="see_dermatologist",
            confidence=0.7,
            explanation=response,
            recommendations=["Consult a dermatologist for professional evaluation"]
        )

    async def stop(self) -> None:
        """Stop the server."""
        print("Stopping Smart Mirror Server...")
        self.running = False
        await self.vlm_client.disconnect()
        print("Server stopped.")


def main():
    """Entry point."""
    # Load configuration
    config_path = Path(__file__).parent.parent / "config" / "config.json"

    if config_path.exists():
        config = SmartMirrorConfig.load(config_path)
    else:
        config = SmartMirrorConfig()

    # Run server
    server = SmartMirrorServer(config)

    async def run():
        await server.start()
        # Keep running
        try:
            while server.running:
                await asyncio.sleep(1)
        except KeyboardInterrupt:
            pass
        finally:
            await server.stop()

    asyncio.run(run())


if __name__ == "__main__":
    main()
