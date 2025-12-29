"""Smart Mirror Edge - Main entry point for Jetson."""

import asyncio
import signal
import sys
from pathlib import Path

import cv2
import numpy as np

# Add parent to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from config import SmartMirrorConfig, get_jetson_config
from edge.cv.pipeline import CVPipeline
from edge.router.event_router import EventRouter, EventType


class SmartMirrorEdge:
    """Main application for Smart Mirror edge device."""

    def __init__(self, config: SmartMirrorConfig):
        self.config = config
        self.running = False

        # Components
        self.pipeline: CVPipeline | None = None
        self.router: EventRouter | None = None
        self.cap: cv2.VideoCapture | None = None

        # VLM connection (WebSocket to server)
        self._ws_client = None

    async def start(self) -> None:
        """Initialize and start the edge pipeline."""
        print("Starting Smart Mirror Edge...")

        # Initialize camera
        self._init_camera()

        # Initialize CV pipeline
        self.pipeline = CVPipeline(self.config.to_dict())
        self.pipeline.initialize()
        print(f"CV Pipeline initialized with modules: {self.pipeline.enabled_modules}")

        # Initialize event router
        self.router = EventRouter(self.config.to_dict().get("event_router", {}))

        # Connect to VLM server
        await self._connect_to_server()

        self.running = True
        print("Smart Mirror Edge running. Press Ctrl+C to stop.")

        # Main loop
        await self._run_loop()

    def _init_camera(self) -> None:
        """Initialize camera capture."""
        cam_config = self.config.camera

        self.cap = cv2.VideoCapture(cam_config.device_id)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, cam_config.width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, cam_config.height)
        self.cap.set(cv2.CAP_PROP_FPS, cam_config.fps)

        if not self.cap.isOpened():
            raise RuntimeError(f"Failed to open camera {cam_config.device_id}")

        print(f"Camera initialized: {cam_config.width}x{cam_config.height}@{cam_config.fps}fps")

    async def _connect_to_server(self) -> None:
        """Connect to VLM server via WebSocket."""
        vlm_config = self.config.vlm
        server_url = f"ws://{vlm_config.host}:{vlm_config.port + 1}/ws"

        # TODO: Implement WebSocket connection
        # For now, we'll use HTTP polling
        print(f"VLM server configured at: {vlm_config.host}:{vlm_config.port}")

    async def _run_loop(self) -> None:
        """Main processing loop."""
        frame_count = 0

        while self.running:
            # Capture frame
            ret, frame = self.cap.read()
            if not ret:
                print("Failed to capture frame")
                await asyncio.sleep(0.01)
                continue

            # Mirror effect
            if self.config.camera.flip_horizontal:
                frame = cv2.flip(frame, 1)

            # Process through CV pipeline
            result = self.pipeline.process_frame(frame)
            frame_count += 1

            # Route to VLM if needed
            vlm_request = self.router.process(result, frame)

            if vlm_request:
                await self._send_to_vlm(vlm_request)

            # Show debug overlay if enabled
            if self.config.ui.show_debug_overlay:
                self._draw_debug_overlay(frame, result)
                cv2.imshow("Smart Mirror Debug", frame)

                if cv2.waitKey(1) & 0xFF == ord('q'):
                    self.running = False

            # Maintain frame rate
            await asyncio.sleep(1 / self.config.camera.fps)

        await self.stop()

    async def _send_to_vlm(self, request) -> None:
        """Send request to VLM server."""
        # TODO: Implement actual sending
        print(f"VLM Request: {request.event_type.name}")

    def _draw_debug_overlay(self, frame: np.ndarray, result) -> None:
        """Draw debug information on frame."""
        # FPS and frame info
        cv2.putText(
            frame,
            f"Frame: {result.frame_id}",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 0),
            2
        )

        # Face count
        cv2.putText(
            frame,
            f"Faces: {len(result.faces)}",
            (10, 60),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 0),
            2
        )

        # Draw face boxes
        for face in result.faces:
            x, y, w, h = face.bbox
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

        # Health info
        if result.health:
            hr = result.health.value
            conf = result.health.confidence
            cv2.putText(
                frame,
                f"HR: {hr:.0f} bpm ({conf:.0%})",
                (10, 90),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 255),
                2
            )

        # Gesture info
        for i, gesture in enumerate(result.gestures):
            cv2.putText(
                frame,
                f"Gesture: {gesture.gesture_name}",
                (10, 120 + i * 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (255, 0, 255),
                2
            )

    async def stop(self) -> None:
        """Stop the edge application."""
        print("Stopping Smart Mirror Edge...")
        self.running = False

        if self.cap:
            self.cap.release()

        if self.pipeline:
            self.pipeline.cleanup()

        cv2.destroyAllWindows()
        print("Smart Mirror Edge stopped.")


def main():
    """Entry point."""
    # Load or create configuration
    config_path = Path(__file__).parent.parent / "config" / "config.json"

    if config_path.exists():
        config = SmartMirrorConfig.load(config_path)
    else:
        # Use Jetson-optimized config by default
        config = get_jetson_config()
        config.save(config_path)

    # Handle Ctrl+C
    app = SmartMirrorEdge(config)

    def signal_handler(sig, frame):
        print("\nInterrupted by user")
        asyncio.get_event_loop().create_task(app.stop())

    signal.signal(signal.SIGINT, signal_handler)

    # Run
    asyncio.run(app.start())


if __name__ == "__main__":
    main()
