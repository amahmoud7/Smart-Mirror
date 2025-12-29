"""
VLM Voice + Camera Demo - Smart Mirror with voice interaction.

Prerequisites:
1. Ollama with vision model: ollama pull qwen3-vl:8b
2. Install dependencies:
   pip install opencv-python aiohttp speechrecognition pyaudio elevenlabs

Usage:
    python vlm_voice_demo.py --elevenlabs-key YOUR_API_KEY

Controls:
    SPACE - Analyze frame (general)
    V     - Voice input (hold to record, release to send)
    C     - Text chat (type in terminal)
    G/H/S - Greeting/Health/Skin modes
    M     - Toggle mute (TTS on/off)
    Q     - Quit
"""

import asyncio
import base64
import sys
import time
import threading
import tempfile
import os
from pathlib import Path
from queue import Queue

import cv2
import aiohttp

# Add parent to path
sys.path.insert(0, str(Path(__file__).parent.parent))


class VoiceHandler:
    """Handles speech-to-text and text-to-speech."""

    def __init__(self, elevenlabs_api_key: str | None = None, voice_id: str = "Rachel"):
        self.elevenlabs_api_key = elevenlabs_api_key
        self.voice_id = voice_id
        self.tts_enabled = True
        self.stt_enabled = True

        # Speech recognition
        self.recognizer = None
        self.microphone = None

        # ElevenLabs
        self.elevenlabs_client = None

        self._init_stt()
        self._init_tts()

    def _init_stt(self):
        """Initialize speech-to-text."""
        try:
            import speech_recognition as sr
            self.recognizer = sr.Recognizer()
            self.microphone = sr.Microphone()

            # Adjust for ambient noise on startup
            print("Calibrating microphone for ambient noise...")
            with self.microphone as source:
                self.recognizer.adjust_for_ambient_noise(source, duration=1)

            # Increase pause threshold - wait longer before assuming speech is done
            self.recognizer.pause_threshold = 1.5  # seconds of silence before phrase is complete
            self.recognizer.phrase_threshold = 0.3  # minimum seconds of speech to consider
            self.recognizer.non_speaking_duration = 1.0  # seconds of silence to keep before/after
            print("Microphone ready (pause threshold: 1.5s)")
            self.stt_enabled = True
        except ImportError:
            print("WARNING: speech_recognition not installed. Voice input disabled.")
            print("  Install with: pip install speechrecognition pyaudio")
            self.stt_enabled = False
        except Exception as e:
            print(f"WARNING: Could not initialize microphone: {e}")
            print("  Make sure you have a microphone connected.")
            self.stt_enabled = False

    def _init_tts(self):
        """Initialize text-to-speech with ElevenLabs."""
        if not self.elevenlabs_api_key:
            print("WARNING: No ElevenLabs API key provided. TTS disabled.")
            print("  Run with: --elevenlabs-key YOUR_KEY")
            self.tts_enabled = False
            return

        try:
            from elevenlabs.client import ElevenLabs
            self.elevenlabs_client = ElevenLabs(api_key=self.elevenlabs_api_key)
            print(f"ElevenLabs TTS initialized (voice: {self.voice_id})")
            self.tts_enabled = True
        except ImportError:
            print("WARNING: elevenlabs not installed. TTS disabled.")
            print("  Install with: pip install elevenlabs")
            self.tts_enabled = False
        except Exception as e:
            print(f"WARNING: Could not initialize ElevenLabs: {e}")
            self.tts_enabled = False

    def listen(self, timeout: float = 10.0, phrase_time_limit: float = 30.0) -> str | None:
        """
        Listen for speech and convert to text.

        Returns:
            Transcribed text or None if failed/nothing detected
        """
        if not self.stt_enabled:
            return None

        import speech_recognition as sr

        try:
            with self.microphone as source:
                print("Listening... (speak now)")
                audio = self.recognizer.listen(
                    source,
                    timeout=timeout,
                    phrase_time_limit=phrase_time_limit
                )

            print("Processing speech...")

            # Use Google's free speech recognition
            # For production, consider Whisper or other options
            text = self.recognizer.recognize_google(audio)
            return text

        except sr.WaitTimeoutError:
            print("No speech detected (timeout)")
            return None
        except sr.UnknownValueError:
            print("Could not understand audio")
            return None
        except sr.RequestError as e:
            print(f"Speech recognition error: {e}")
            return None
        except Exception as e:
            print(f"Error during listening: {e}")
            return None

    def speak(self, text: str) -> None:
        """
        Convert text to speech using ElevenLabs and play it.

        Args:
            text: Text to speak
        """
        if not self.tts_enabled or not self.elevenlabs_client:
            print(f"[TTS disabled] Would say: {text}")
            return

        try:
            # Generate audio
            audio_generator = self.elevenlabs_client.text_to_speech.convert(
                voice_id=self.voice_id,
                text=text,
                model_id="eleven_turbo_v2_5",  # Fast, good quality
                output_format="mp3_44100_128"
            )

            # Save to temp file and play
            with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as f:
                for chunk in audio_generator:
                    f.write(chunk)
                temp_path = f.name

            # Play audio
            self._play_audio(temp_path)

            # Cleanup
            os.unlink(temp_path)

        except Exception as e:
            print(f"TTS Error: {e}")

    def _play_audio(self, filepath: str) -> None:
        """Play audio file."""
        try:
            # Try pygame first (most reliable cross-platform)
            import pygame
            pygame.mixer.init()
            pygame.mixer.music.load(filepath)
            pygame.mixer.music.play()
            while pygame.mixer.music.get_busy():
                time.sleep(0.1)
            pygame.mixer.quit()
        except ImportError:
            try:
                # Fallback to playsound
                from playsound import playsound
                playsound(filepath)
            except ImportError:
                # Last resort: system command
                import platform
                if platform.system() == "Windows":
                    os.system(f'start /min wmplayer "{filepath}"')
                elif platform.system() == "Darwin":
                    os.system(f'afplay "{filepath}"')
                else:
                    os.system(f'mpg123 -q "{filepath}"')


class VLMVoiceDemo:
    """Smart Mirror demo with voice interaction."""

    def __init__(
        self,
        camera_id: int = 0,
        ollama_host: str = "localhost",
        ollama_port: int = 11434,
        model: str = "qwen3-vl:8b",
        elevenlabs_key: str | None = None,
        voice_id: str = "Rachel"
    ):
        self.camera_id = camera_id
        self.ollama_url = f"http://{ollama_host}:{ollama_port}/api/chat"
        self.model = model

        self.cap = None
        self.session = None
        self.running = False

        # Voice handler
        self.voice = VoiceHandler(elevenlabs_key, voice_id)

        # State
        self.last_response = ""
        self.processing = False
        self.mode = "ready"
        self.tts_muted = False

        # Chat mode state
        self.chat_mode_active = False
        self.pending_question = None
        self.current_frame = None
        self.frame_lock = threading.Lock()

        # Voice mode state
        self.voice_mode_active = False

    async def start(self):
        """Initialize and start the demo."""
        print("=" * 60)
        print("Smart Mirror VLM + Voice Demo")
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
                else:
                    print("WARNING: Could not connect to Ollama")
        except Exception as e:
            print(f"ERROR: Cannot connect to Ollama: {e}")
            await self.session.close()
            return

        # Initialize camera
        print(f"\nOpening camera {self.camera_id}...")
        self.cap = cv2.VideoCapture(self.camera_id)

        if not self.cap.isOpened():
            print(f"ERROR: Could not open camera {self.camera_id}")
            await self.session.close()
            return

        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

        actual_w = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        actual_h = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        print(f"Camera opened: {actual_w}x{actual_h}")

        print("\n" + "=" * 60)
        print("Controls:")
        print("  V     - Voice input (press, then speak)")
        print("  C     - Text chat (type in terminal)")
        print("  SPACE - Quick analyze")
        print("  G     - Greeting mode")
        print("  H     - Health observation")
        print("  S     - Skin analysis")
        print("  M     - Toggle TTS mute")
        print("  Q     - Quit")
        print("=" * 60 + "\n")

        self.running = True
        await self._run_loop()

    async def _run_loop(self):
        """Main loop."""
        while self.running:
            ret, frame = self.cap.read()
            if not ret:
                await asyncio.sleep(0.1)
                continue

            frame = cv2.flip(frame, 1)

            with self.frame_lock:
                self.current_frame = frame.copy()

            # Check for pending chat question
            if self.pending_question is not None:
                question = self.pending_question
                self.pending_question = None
                asyncio.create_task(self._process_query(question, frame.copy(), speak=True))

            # Draw overlay
            display_frame = self._draw_overlay(frame.copy())
            cv2.imshow("Smart Mirror Voice Demo", display_frame)

            # Handle input
            key = cv2.waitKey(1) & 0xFF

            if key == ord('q'):
                self.running = False
            elif key == ord('v') and not self.processing and not self.voice_mode_active:
                # Voice input
                self.voice_mode_active = True
                self.mode = "listening..."
                threading.Thread(target=self._voice_input, args=(frame.copy(),), daemon=True).start()
            elif key == ord('c') and not self.processing and not self.chat_mode_active:
                # Text chat
                self.chat_mode_active = True
                self.mode = "chat (typing...)"
                threading.Thread(target=self._get_chat_input, daemon=True).start()
            elif key == ord(' ') and not self.processing:
                asyncio.create_task(self._process_query("What do you see?", frame.copy(), speak=True))
            elif key == ord('g') and not self.processing:
                asyncio.create_task(self._process_query(
                    "Greet me warmly based on what you observe.",
                    frame.copy(), speak=True
                ))
            elif key == ord('h') and not self.processing:
                asyncio.create_task(self._process_query(
                    "Give me a brief wellness observation based on my appearance.",
                    frame.copy(), speak=True
                ))
            elif key == ord('s') and not self.processing:
                asyncio.create_task(self._process_query(
                    "Analyze any visible skin using the ABCDE framework. Be concise.",
                    frame.copy(), speak=True
                ))
            elif key == ord('m'):
                self.tts_muted = not self.tts_muted
                status = "muted" if self.tts_muted else "unmuted"
                print(f"\nTTS {status}")

            await asyncio.sleep(0.01)

        await self.stop()

    def _voice_input(self, frame):
        """Handle voice input in separate thread."""
        print("\n" + "=" * 60)
        print("VOICE MODE - Speak your question now...")
        print("=" * 60)

        text = self.voice.listen(timeout=10.0, phrase_time_limit=30.0)

        if text:
            print(f"You said: {text}")
            self.pending_question = text
        else:
            print("No speech detected or could not understand.")
            self.mode = "ready"

        self.voice_mode_active = False

    def _get_chat_input(self):
        """Get text input in separate thread."""
        print("\n" + "=" * 60)
        print("CHAT MODE - Type your question")
        print("=" * 60)

        try:
            question = input("\nYour question: ").strip()
        except EOFError:
            question = ""

        if question:
            self.pending_question = question
        else:
            print("No question entered.")
            self.mode = "ready"

        self.chat_mode_active = False

    async def _process_query(self, question: str, frame, speak: bool = True):
        """Process query with VLM and optionally speak response."""
        self.processing = True
        self.mode = "thinking..."

        print(f"\n[QUERY] {question}")
        start_time = time.time()

        try:
            _, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
            image_b64 = base64.b64encode(buffer).decode('utf-8')

            system_prompt = """You are an AI assistant in a smart mirror. You can see the user through the camera.
Be conversational, helpful, and concise. Keep responses to 2-3 sentences unless more detail is needed.
You're speaking out loud, so be natural and friendly."""

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
                    print(f"\n[{elapsed:.1f}s] Response:")
                    print("-" * 40)
                    print(response_text)
                    print("-" * 40)

                    self.last_response = response_text

                    # Speak the response
                    if speak and not self.tts_muted:
                        self.mode = "speaking..."
                        # Run TTS in thread to not block
                        threading.Thread(
                            target=self.voice.speak,
                            args=(response_text,),
                            daemon=True
                        ).start()
                else:
                    error = await resp.text()
                    print(f"Error: {resp.status} - {error}")
                    self.last_response = f"Error: {resp.status}"

        except Exception as e:
            print(f"Error: {e}")
            self.last_response = f"Error: {e}"

        self.processing = False
        self.mode = "ready"

    def _draw_overlay(self, frame):
        """Draw UI overlay."""
        h, w = frame.shape[:2]

        overlay = frame.copy()
        cv2.rectangle(overlay, (10, 10), (w - 10, 130), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.5, frame, 0.5, 0, frame)

        # Status
        if self.voice_mode_active:
            status = "LISTENING - Speak now..."
            color = (0, 165, 255)  # Orange
        elif self.chat_mode_active:
            status = "CHAT MODE - Type in terminal"
            color = (255, 255, 0)  # Yellow
        elif self.processing:
            status = f"Processing... ({self.mode})"
            color = (0, 255, 255)  # Cyan
        else:
            status = "Ready - Press V to speak, C to chat"
            color = (0, 255, 0)  # Green

        cv2.putText(frame, status, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

        # Mode and TTS status
        tts_status = "TTS: OFF" if self.tts_muted else "TTS: ON"
        cv2.putText(frame, f"Mode: {self.mode} | {tts_status}", (20, 70),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

        # Voice status
        voice_status = "Voice: " + ("Ready" if self.voice.stt_enabled else "Disabled")
        cv2.putText(frame, voice_status, (20, 95), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)

        # Controls
        cv2.putText(frame, "V=Voice C=Chat SPACE=Analyze G=Greet H=Health S=Skin M=Mute Q=Quit",
                    (20, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (180, 180, 180), 1)

        # Response at bottom
        if self.last_response:
            lines = self._wrap_text(self.last_response, 95)
            y_offset = h - 25 - (len(lines) * 20)

            overlay2 = frame.copy()
            cv2.rectangle(overlay2, (10, y_offset - 10), (w - 10, h - 10), (0, 0, 0), -1)
            cv2.addWeighted(overlay2, 0.7, frame, 0.3, 0, frame)

            for i, line in enumerate(lines[-12:]):
                cv2.putText(frame, line, (20, y_offset + i * 20),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)

        return frame

    def _wrap_text(self, text: str, max_chars: int) -> list[str]:
        """Wrap text to lines."""
        words = text.replace('\n', ' ').split()
        lines = []
        current = ""
        for word in words:
            if len(current) + len(word) + 1 <= max_chars:
                current += (" " if current else "") + word
            else:
                if current:
                    lines.append(current)
                current = word
        if current:
            lines.append(current)
        return lines

    async def stop(self):
        """Cleanup."""
        print("\nShutting down...")
        if self.cap:
            self.cap.release()
        if self.session:
            await self.session.close()
        cv2.destroyAllWindows()
        print("Demo stopped.")


async def main():
    import argparse

    parser = argparse.ArgumentParser(description="Smart Mirror VLM + Voice Demo")
    parser.add_argument("--camera", type=int, default=0, help="Camera ID")
    parser.add_argument("--host", type=str, default="localhost", help="Ollama host")
    parser.add_argument("--port", type=int, default=11434, help="Ollama port")
    parser.add_argument("--model", type=str, default="qwen3-vl:8b", help="VLM model")
    parser.add_argument("--elevenlabs-key", type=str, default=None, help="ElevenLabs API key")
    parser.add_argument("--voice-id", type=str, default="Rachel", help="ElevenLabs voice ID")
    args = parser.parse_args()

    # Check for API key in environment if not provided
    elevenlabs_key = args.elevenlabs_key or os.environ.get("ELEVENLABS_API_KEY")

    demo = VLMVoiceDemo(
        camera_id=args.camera,
        ollama_host=args.host,
        ollama_port=args.port,
        model=args.model,
        elevenlabs_key=elevenlabs_key,
        voice_id=args.voice_id
    )
    await demo.start()


if __name__ == "__main__":
    asyncio.run(main())
