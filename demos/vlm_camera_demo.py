"""
VLM Camera Demo - Test VLM integration with local camera.

Prerequisites:
1. Install Ollama: https://ollama.ai
2. Pull vision model: ollama pull qwen3-vl:8b
3. Install dependencies: pip install opencv-python aiohttp

Usage:
    python vlm_camera_demo.py

Controls:
    SPACE - Send current frame to VLM for analysis
    G     - Greet (presence detection simulation)
    H     - Health query
    S     - Skin analysis mode
    C     - Chat mode (type your own question - feed continues!)
    Q     - Quit
"""

import asyncio
import base64
import sys
import time
import threading
from pathlib import Path

import cv2
import aiohttp

# Add parent to path
sys.path.insert(0, str(Path(__file__).parent.parent))


class VLMCameraDemo:
    """Interactive demo for testing VLM with camera."""

    def __init__(
        self,
        camera_id: int = 0,
        ollama_host: str = "localhost",
        ollama_port: int = 11434,
        model: str = "qwen2.5-vl:7b"
    ):
        self.camera_id = camera_id
        self.ollama_url = f"http://{ollama_host}:{ollama_port}/api/chat"
        self.model = model

        self.cap = None
        self.session = None
        self.running = False

        # State
        self.last_response = ""
        self.processing = False
        self.mode = "general"  # general, health, skin, chat

        # Chat mode state
        self.chat_mode_active = False
        self.pending_question = None
        self.current_frame = None  # Always holds the latest frame
        self.frame_lock = threading.Lock()

    async def start(self):
        """Initialize camera and HTTP session."""
        print("=" * 60)
        print("Smart Mirror VLM Camera Demo")
        print("=" * 60)

        # Check Ollama
        print(f"\nConnecting to Ollama at {self.ollama_url}...")
        self.session = aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=60))

        try:
            async with self.session.get(f"http://localhost:11434/api/tags") as resp:
                if resp.status == 200:
                    data = await resp.json()
                    models = [m["name"] for m in data.get("models", [])]
                    print(f"Available models: {models}")
                    if not any(self.model in m for m in models):
                        print(f"\nWARNING: Model '{self.model}' not found!")
                        print(f"Run: ollama pull {self.model}")
                else:
                    print("WARNING: Could not connect to Ollama")
        except Exception as e:
            print(f"ERROR: Cannot connect to Ollama: {e}")
            print("\nMake sure Ollama is running:")
            print("  1. Install from https://ollama.ai")
            print(f"  2. Run: ollama pull {self.model}")
            print("  3. Ollama should start automatically, or run: ollama serve")
            await self.session.close()
            return

        # Initialize camera
        print(f"\nOpening camera {self.camera_id}...")
        self.cap = cv2.VideoCapture(self.camera_id)

        if not self.cap.isOpened():
            print(f"ERROR: Could not open camera {self.camera_id}")
            print("\nTry different camera IDs:")
            for i in range(5):
                test_cap = cv2.VideoCapture(i)
                if test_cap.isOpened():
                    print(f"  Camera {i}: Available")
                    test_cap.release()
                else:
                    print(f"  Camera {i}: Not available")
            await self.session.close()
            return

        # Set camera properties
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

        actual_w = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        actual_h = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        print(f"Camera opened: {actual_w}x{actual_h}")

        print("\n" + "=" * 60)
        print("Controls:")
        print("  SPACE - Analyze current frame (general query)")
        print("  G     - Greet user (simulate presence detection)")
        print("  H     - Health status query")
        print("  S     - Skin analysis mode")
        print("  C     - Chat (type question - camera keeps running!)")
        print("  Q     - Quit")
        print("=" * 60 + "\n")

        self.running = True
        await self._run_loop()

    async def _run_loop(self):
        """Main display and input loop."""
        while self.running:
            ret, frame = self.cap.read()
            if not ret:
                print("Failed to capture frame")
                await asyncio.sleep(0.1)
                continue

            # Mirror effect
            frame = cv2.flip(frame, 1)

            # Store current frame (thread-safe)
            with self.frame_lock:
                self.current_frame = frame.copy()

            # Check if chat input is ready
            if self.pending_question is not None:
                question = self.pending_question
                self.pending_question = None
                # Capture frame NOW (at moment of inference) and send to VLM
                asyncio.create_task(self._process_chat(question, frame.copy()))

            # Draw UI overlay
            display_frame = self._draw_overlay(frame.copy())

            # Show frame
            cv2.imshow("Smart Mirror VLM Demo", display_frame)

            # Handle keyboard input
            key = cv2.waitKey(1) & 0xFF

            if key == ord('q'):
                self.running = False
            elif key == ord(' ') and not self.processing:
                asyncio.create_task(self._analyze_frame(frame.copy(), "general"))
            elif key == ord('g') and not self.processing:
                asyncio.create_task(self._analyze_frame(frame.copy(), "greeting"))
            elif key == ord('h') and not self.processing:
                asyncio.create_task(self._analyze_frame(frame.copy(), "health"))
            elif key == ord('s') and not self.processing:
                asyncio.create_task(self._analyze_frame(frame.copy(), "skin"))
            elif key == ord('c') and not self.processing and not self.chat_mode_active:
                # Start chat mode in a separate thread (non-blocking)
                self.chat_mode_active = True
                self.mode = "chat (typing...)"
                threading.Thread(target=self._get_chat_input, daemon=True).start()

            # Small delay to prevent CPU spinning
            await asyncio.sleep(0.01)

        await self.stop()

    def _get_chat_input(self):
        """Get user input in a separate thread (doesn't block camera)."""
        print("\n" + "=" * 60)
        print("CHAT MODE - Camera continues running!")
        print("Type your question and press Enter.")
        print("Frame will be captured when you submit.")
        print("=" * 60)

        try:
            question = input("\nYour question: ").strip()
        except EOFError:
            question = ""

        if question:
            print(f"Question received: {question}")
            print("Capturing frame and sending to VLM...")
            self.pending_question = question
        else:
            print("No question entered. Cancelled.")
            self.chat_mode_active = False
            self.mode = "general"

    async def _process_chat(self, question: str, frame):
        """Process chat question with captured frame."""
        self.processing = True
        self.mode = "chat"

        print(f"\n[CHAT] Processing: {question}")
        start_time = time.time()

        try:
            # Encode the frame captured at inference time
            _, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
            image_b64 = base64.b64encode(buffer).decode('utf-8')

            system_prompt = """You are an AI assistant in a smart mirror. You can see the person in front of the mirror through the camera.
Answer their question based on what you observe in the image. Be helpful, conversational, and concise."""

            payload = {
                "model": self.model,
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": question, "images": [image_b64]}
                ],
                "stream": False
            }

            async with self.session.post(self.ollama_url, json=payload) as resp:
                if resp.status == 200:
                    data = await resp.json()
                    response_text = data.get("message", {}).get("content", "No response")

                    elapsed = time.time() - start_time
                    print(f"\n[{elapsed:.1f}s] VLM Response:")
                    print("-" * 40)
                    print(response_text)
                    print("-" * 40)

                    self.last_response = response_text
                else:
                    error = await resp.text()
                    print(f"Error: {resp.status} - {error}")
                    self.last_response = f"Error: {resp.status}"

        except Exception as e:
            print(f"Error: {e}")
            self.last_response = f"Error: {e}"

        self.processing = False
        self.chat_mode_active = False
        print("\nPress C to chat again, or use other commands.")

    def _draw_overlay(self, frame):
        """Draw status overlay on frame."""
        h, w = frame.shape[:2]

        # Semi-transparent background for text
        overlay = frame.copy()
        cv2.rectangle(overlay, (10, 10), (w - 10, 120), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.5, frame, 0.5, 0, frame)

        # Status text
        if self.chat_mode_active and not self.processing:
            status = "CHAT MODE - Type in terminal, press Enter"
            color = (255, 255, 0)  # Yellow
        elif self.processing:
            status = "Processing with VLM..."
            color = (0, 255, 255)  # Cyan
        else:
            status = "Ready - Press SPACE to analyze, C to chat"
            color = (0, 255, 0)  # Green

        cv2.putText(frame, status, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

        # Mode
        cv2.putText(frame, f"Mode: {self.mode}", (20, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)

        # Controls hint
        cv2.putText(frame, "SPACE=Analyze G=Greet H=Health S=Skin C=Chat Q=Quit", (20, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 200, 200), 1)

        # Response area at bottom
        if self.last_response:
            # Wrap text
            lines = self._wrap_text(self.last_response, 90)
            y_offset = h - 30 - (len(lines) * 22)

            # Background
            overlay2 = frame.copy()
            cv2.rectangle(overlay2, (10, y_offset - 10), (w - 10, h - 10), (0, 0, 0), -1)
            cv2.addWeighted(overlay2, 0.7, frame, 0.3, 0, frame)

            for i, line in enumerate(lines[-10:]):  # Show last 10 lines
                cv2.putText(
                    frame, line, (20, y_offset + i * 22),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1
                )

        return frame

    def _wrap_text(self, text: str, max_chars: int) -> list[str]:
        """Wrap text to multiple lines."""
        words = text.replace('\n', ' ').split()
        lines = []
        current_line = ""

        for word in words:
            if len(current_line) + len(word) + 1 <= max_chars:
                current_line += (" " if current_line else "") + word
            else:
                if current_line:
                    lines.append(current_line)
                current_line = word

        if current_line:
            lines.append(current_line)

        return lines

    async def _analyze_frame(self, frame, mode: str):
        """Send frame to VLM for analysis."""
        self.processing = True
        self.mode = mode

        print(f"\n[{mode.upper()}] Sending frame to VLM...")
        start_time = time.time()

        try:
            # Encode frame as base64
            _, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
            image_b64 = base64.b64encode(buffer).decode('utf-8')

            # Build prompt based on mode
            system_prompt, user_prompt = self._get_prompts(mode)

            # Build request
            payload = {
                "model": self.model,
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt, "images": [image_b64]}
                ],
                "stream": False
            }

            # Send request
            async with self.session.post(self.ollama_url, json=payload) as resp:
                if resp.status == 200:
                    data = await resp.json()
                    response_text = data.get("message", {}).get("content", "No response")

                    elapsed = time.time() - start_time
                    print(f"[{elapsed:.1f}s] VLM Response:")
                    print("-" * 40)
                    print(response_text)
                    print("-" * 40)

                    self.last_response = response_text
                else:
                    error = await resp.text()
                    print(f"Error: {resp.status} - {error}")
                    self.last_response = f"Error: {resp.status}"

        except Exception as e:
            print(f"Error: {e}")
            self.last_response = f"Error: {e}"

        self.processing = False

    def _get_prompts(self, mode: str) -> tuple[str, str]:
        """Get system and user prompts for the mode."""
        if mode == "greeting":
            system = """You are an AI assistant in a smart mirror. A person has appeared in front of you.
Give a brief, warm greeting. You can comment on something you observe (time of day, their appearance, etc).
Keep it to 1-2 sentences."""
            user = "Greet the person you see in this image."

        elif mode == "health":
            system = """You are a health-aware AI in a smart mirror. Observe the person and provide a brief wellness observation.
You can comment on: apparent energy level, posture, any visible signs of fatigue or stress.
Be supportive, not diagnostic. Keep it brief (2-3 sentences)."""
            user = "Look at this person and provide a brief wellness observation. What do you notice about their apparent state?"

        elif mode == "skin":
            system = """You are assisting with skin health awareness. Analyze any visible skin in this image.
Use the ABCDE framework if you see any spots or lesions:
- Asymmetry, Border, Color, Diameter, Evolution
IMPORTANT: You are NOT diagnosing. Only describe what you observe and recommend professional evaluation if anything looks concerning.
Be concise (3-4 sentences max)."""
            user = "Examine the visible skin in this image. Describe any notable features and whether professional evaluation might be warranted."

        else:  # general
            system = """You are an AI assistant in a smart mirror. You can see the person in front of the mirror.
Describe what you see and offer helpful observations or suggestions.
Be conversational and concise (2-3 sentences)."""
            user = "What do you see in this image? Provide a brief, helpful observation."

        return system, user

    async def stop(self):
        """Cleanup resources."""
        print("\nShutting down...")
        if self.cap:
            self.cap.release()
        if self.session:
            await self.session.close()
        cv2.destroyAllWindows()
        print("Demo stopped.")


async def main():
    """Entry point."""
    import argparse

    parser = argparse.ArgumentParser(description="Smart Mirror VLM Camera Demo")
    parser.add_argument("--camera", type=int, default=0, help="Camera device ID (default: 0)")
    parser.add_argument("--host", type=str, default="localhost", help="Ollama host")
    parser.add_argument("--port", type=int, default=11434, help="Ollama port")
    parser.add_argument("--model", type=str, default="qwen2.5-vl:7b", help="VLM model name")
    args = parser.parse_args()

    demo = VLMCameraDemo(
        camera_id=args.camera,
        ollama_host=args.host,
        ollama_port=args.port,
        model=args.model
    )
    await demo.start()


if __name__ == "__main__":
    asyncio.run(main())
