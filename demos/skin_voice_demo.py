"""
Skin Analysis Demo with Voice Interaction

Speak naturally to request skin analysis:
- "Check my skin"
- "Analyze this mole"
- "Do I have any concerning spots?"
- "Scan for lesions"

Usage:
    python skin_voice_demo.py --camera 0 --vlm-model qwen3-vl:8b --elevenlabs-key YOUR_KEY

Controls:
    V     - Push to talk (manual trigger)
    SPACE - Analyze without voice
    L     - Toggle continuous listening
    M     - Toggle TTS mute
    Q     - Quit
"""

import asyncio
import base64
import sys
import time
import os
import threading
import re
from pathlib import Path
from dataclasses import dataclass, field
from datetime import datetime

import cv2
import numpy as np
import aiohttp

sys.path.insert(0, str(Path(__file__).parent.parent))


# Voice command patterns
SKIN_ANALYSIS_TRIGGERS = [
    r"check.*(skin|mole|spot|lesion)",
    r"analyze.*(skin|mole|spot|lesion|this)",
    r"scan.*(skin|mole|spot|lesion)",
    r"look at.*(skin|mole|spot|this)",
    r"inspect.*(skin|mole|spot)",
    r"do i have.*(cancer|melanoma|concern)",
    r"is this.*(cancer|dangerous|bad|concerning|malignant|benign)",
    r"what do you (see|think)",
    r"examine",
]


class VoiceHandler:
    """Speech-to-text and text-to-speech handler."""

    def __init__(self, elevenlabs_key: str | None = None, voice_id: str = "21m00Tcm4TlvDq8ikWAM"):
        self.elevenlabs_key = elevenlabs_key
        self.voice_id = voice_id
        self.recognizer = None
        self.microphone = None
        self.stt_enabled = False
        self.tts_enabled = False

        self._init_stt()
        self._init_tts()

    def _init_stt(self):
        """Initialize speech recognition."""
        try:
            import speech_recognition as sr
            self.recognizer = sr.Recognizer()
            self.microphone = sr.Microphone()

            print("Calibrating microphone...")
            with self.microphone as source:
                self.recognizer.adjust_for_ambient_noise(source, duration=1)

            self.recognizer.pause_threshold = 1.5
            self.recognizer.phrase_threshold = 0.3
            self.stt_enabled = True
            print("Voice input ready.")
        except ImportError:
            print("Speech recognition not available. Install: pip install speechrecognition pyaudio")
        except Exception as e:
            print(f"Microphone error: {e}")

    def _init_tts(self):
        """Initialize text-to-speech."""
        if not self.elevenlabs_key:
            print("No ElevenLabs key - TTS disabled")
            return

        try:
            from elevenlabs.client import ElevenLabs
            self.elevenlabs_client = ElevenLabs(api_key=self.elevenlabs_key)
            self.tts_enabled = True
            print("Voice output ready.")
        except ImportError:
            print("ElevenLabs not available. Install: pip install elevenlabs")
        except Exception as e:
            print(f"TTS error: {e}")

    def listen(self, timeout: float = 10.0, phrase_limit: float = 30.0) -> str | None:
        """Listen for speech and return text."""
        if not self.stt_enabled:
            return None

        import speech_recognition as sr

        try:
            with self.microphone as source:
                audio = self.recognizer.listen(source, timeout=timeout, phrase_time_limit=phrase_limit)
            return self.recognizer.recognize_google(audio)
        except sr.WaitTimeoutError:
            return None
        except sr.UnknownValueError:
            return None
        except Exception as e:
            print(f"Listen error: {e}")
            return None

    def speak(self, text: str):
        """Speak text using ElevenLabs."""
        if not self.tts_enabled:
            print(f"[Would say]: {text}")
            return

        try:
            import tempfile

            audio = self.elevenlabs_client.text_to_speech.convert(
                voice_id=self.voice_id,
                text=text,
                model_id="eleven_turbo_v2_5",
                output_format="mp3_44100_128"
            )

            with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as f:
                for chunk in audio:
                    f.write(chunk)
                path = f.name

            try:
                import pygame
                pygame.mixer.init()
                pygame.mixer.music.load(path)
                pygame.mixer.music.play()
                while pygame.mixer.music.get_busy():
                    time.sleep(0.1)
                pygame.mixer.quit()
            except:
                pass

            os.unlink(path)
        except Exception as e:
            print(f"TTS error: {e}")

    def is_skin_analysis_request(self, text: str) -> bool:
        """Check if text is a skin analysis request."""
        if not text:
            return False
        text_lower = text.lower()
        for pattern in SKIN_ANALYSIS_TRIGGERS:
            if re.search(pattern, text_lower):
                return True
        return False


class SkinCancerModel:
    """CV model for skin lesion classification."""

    def __init__(self, model_path: str | None = None):
        self.model_path = model_path
        self.feature_extractor = None
        self.classifier = None
        self.is_loaded = False

    def load(self) -> bool:
        if not self.model_path or not os.path.exists(self.model_path):
            print("No trained CV model found. Using VLM-only mode.")
            return False

        try:
            import tensorflow as tf
            from tensorflow.keras.applications import MobileNet
            from tensorflow.keras.models import Model, load_model

            print(f"Loading CV model: {self.model_path}")
            base = MobileNet(weights='imagenet', include_top=False, pooling='avg', input_shape=(224, 224, 3))
            self.feature_extractor = Model(inputs=base.input, outputs=base.output)
            self.classifier = load_model(self.model_path)
            self.is_loaded = True
            print("CV model loaded.")
            return True
        except Exception as e:
            print(f"Failed to load model: {e}")
            return False

    def predict(self, image: np.ndarray) -> tuple[str | None, float, dict]:
        if not self.is_loaded:
            return None, 0.0, {"error": "model_not_loaded"}

        try:
            from tensorflow.keras.applications.mobilenet import preprocess_input

            # Validate input
            is_valid, info = self._validate_input(image)
            if not is_valid:
                return None, 0.0, {"error": "invalid_input", "details": info}

            # Process
            img_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            img_resized = cv2.resize(img_rgb, (224, 224))
            img_array = np.expand_dims(img_resized, axis=0).astype(np.float32)
            img_preprocessed = preprocess_input(img_array)

            features = self.feature_extractor.predict(img_preprocessed, verbose=0)
            sequence = np.zeros((1, 2, 1024), dtype=np.float32)
            sequence[0, 1, :] = features[0]

            prob = self.classifier.predict(sequence, verbose=0)[0][0]

            if prob > 0.5:
                return "malignant", float(prob), {"mode": "single_image", "validation": info}
            else:
                return "benign", float(1 - prob), {"mode": "single_image", "validation": info}
        except Exception as e:
            return None, 0.0, {"error": str(e)}

    def _validate_input(self, image: np.ndarray) -> tuple[bool, dict]:
        """Validate that image contains lesion-like features, not just skin/face."""
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        h, w = gray.shape

        # Basic stats
        variance = np.var(gray)
        mean_intensity = np.mean(gray)

        # Look for dark concentrated regions (lesion-like)
        dark_threshold = mean_intensity * 0.6
        dark_mask = gray < dark_threshold
        dark_ratio = np.sum(dark_mask) / gray.size

        # Check for blob-like structures (lesions are usually compact dark spots)
        # Use morphological operations to find concentrated dark regions
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
        dark_binary = (dark_mask * 255).astype(np.uint8)
        closed = cv2.morphologyEx(dark_binary, cv2.MORPH_CLOSE, kernel)
        contours, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # Look for significant contours (potential lesions)
        min_lesion_area = (h * w) * 0.02  # At least 2% of image
        max_lesion_area = (h * w) * 0.6   # Not more than 60%
        significant_contours = [c for c in contours if min_lesion_area < cv2.contourArea(c) < max_lesion_area]

        # Analyze color in center region
        center = image[h//4:3*h//4, w//4:3*w//4]
        hsv = cv2.cvtColor(center, cv2.COLOR_BGR2HSV)
        hue_std = np.std(hsv[:,:,0])
        sat_mean = np.mean(hsv[:,:,1])

        # Check for skin tones (face detection proxy)
        # Skin typically has low saturation and specific hue range
        skin_hue_mask = (hsv[:,:,0] < 25) | (hsv[:,:,0] > 160)  # Red-ish hues
        skin_sat_mask = hsv[:,:,1] < 100  # Low saturation
        skin_ratio = np.sum(skin_hue_mask & skin_sat_mask) / (center.shape[0] * center.shape[1])

        info = {
            "variance": float(variance),
            "dark_ratio": float(dark_ratio),
            "hue_std": float(hue_std),
            "sat_mean": float(sat_mean),
            "skin_ratio": float(skin_ratio),
            "blob_count": len(significant_contours),
            "max_blob_area": float(max([cv2.contourArea(c) for c in significant_contours]) / (h*w)) if significant_contours else 0.0
        }
        reasons = []

        # Stricter validation
        if variance < 800:
            reasons.append("Image too uniform")
        if len(significant_contours) == 0:
            reasons.append("No lesion-like regions detected")
        if dark_ratio < 0.03 or dark_ratio > 0.7:
            reasons.append("Dark region ratio inconsistent with lesion")
        if skin_ratio > 0.5:
            reasons.append("Appears to be mostly normal skin/face")
        if hue_std < 8:
            reasons.append("Low color variation - no distinct lesion")

        info["reasons"] = reasons
        info["is_valid"] = len(reasons) == 0

        # Calculate quality score (0-1) for confidence adjustment
        quality = 1.0
        if len(significant_contours) == 0:
            quality *= 0.3
        if skin_ratio > 0.3:
            quality *= 0.5
        if dark_ratio < 0.05:
            quality *= 0.4
        info["quality_score"] = quality

        return len(reasons) == 0, info


class SkinVoiceDemo:
    """Skin analysis with voice interaction."""

    def __init__(
        self,
        camera_id: int = 0,
        model_path: str | None = None,
        ollama_host: str = "localhost",
        ollama_port: int = 11434,
        vlm_model: str = "qwen3-vl:8b",
        elevenlabs_key: str | None = None,
        voice_id: str = "21m00Tcm4TlvDq8ikWAM"
    ):
        self.camera_id = camera_id
        self.ollama_url = f"http://{ollama_host}:{ollama_port}/api/chat"
        self.vlm_model = vlm_model

        self.cap = None
        self.session = None
        self.running = False

        self.cv_model = SkinCancerModel(model_path)
        self.voice = VoiceHandler(elevenlabs_key, voice_id)

        self.processing = False
        self.listening = False
        self.continuous_listen = False
        self.tts_muted = False
        self.mode = "ready"
        self.last_response = ""
        self.user_query = ""
        self.loop = None  # Main event loop reference

    async def start(self):
        self.loop = asyncio.get_running_loop()  # Store loop reference
        print("=" * 60)
        print("Skin Analysis with Voice Interaction")
        print("=" * 60)

        self.cv_model.load()

        self.session = aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=120))

        try:
            async with self.session.get("http://localhost:11434/api/tags") as resp:
                if resp.status == 200:
                    print("VLM connected.")
        except:
            print("WARNING: Ollama not available")

        self.cap = cv2.VideoCapture(self.camera_id)
        if not self.cap.isOpened():
            print("ERROR: Cannot open camera")
            return

        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

        print("\n" + "=" * 60)
        print("Voice Commands:")
        print('  "Check my skin"')
        print('  "Analyze this mole"')
        print('  "Is this concerning?"')
        print('  "Do I have any lesions?"')
        print("\nKeyboard:")
        print("  V - Push to talk")
        print("  L - Toggle continuous listening")
        print("  SPACE - Analyze (no voice)")
        print("  M - Mute TTS")
        print("  Q - Quit")
        print("=" * 60 + "\n")

        if self.voice.stt_enabled:
            greeting = "Hello! I'm your skin analysis assistant. Say 'check my skin' or position a mole in the green box and ask me to analyze it."
            if not self.tts_muted:
                threading.Thread(target=self.voice.speak, args=(greeting,), daemon=True).start()

        self.running = True
        await self._run_loop()

    async def _run_loop(self):
        while self.running:
            ret, frame = self.cap.read()
            if not ret:
                await asyncio.sleep(0.1)
                continue

            frame = cv2.flip(frame, 1)
            display = self._draw_overlay(frame.copy())
            cv2.imshow("Skin Analysis - Voice Enabled", display)

            key = cv2.waitKey(1) & 0xFF

            if key == ord('q'):
                self.running = False
            elif key == ord('v') and not self.processing and not self.listening:
                threading.Thread(target=self._voice_triggered_analysis, args=(frame.copy(),), daemon=True).start()
            elif key == ord(' ') and not self.processing:
                asyncio.create_task(self._analyze(frame.copy(), "Analyze the skin in this image."))
            elif key == ord('l'):
                self.continuous_listen = not self.continuous_listen
                status = "ON" if self.continuous_listen else "OFF"
                print(f"\nContinuous listening: {status}")
                if self.continuous_listen and not self.listening:
                    threading.Thread(target=self._continuous_listen_loop, daemon=True).start()
            elif key == ord('m'):
                self.tts_muted = not self.tts_muted
                print(f"\nTTS: {'muted' if self.tts_muted else 'unmuted'}")

            await asyncio.sleep(0.01)

        await self.stop()

    def _voice_triggered_analysis(self, frame):
        """Handle push-to-talk voice input."""
        self.listening = True
        self.mode = "listening..."
        print("\n[LISTENING] Speak now...")

        text = self.voice.listen(timeout=10.0, phrase_limit=30.0)

        if text:
            print(f"You said: {text}")
            self.user_query = text

            if self.voice.is_skin_analysis_request(text):
                # Acknowledge and analyze
                if not self.tts_muted:
                    threading.Thread(target=self.voice.speak, args=("Let me analyze that for you.",), daemon=True).start()
                # Schedule coroutine on main loop
                future = asyncio.run_coroutine_threadsafe(self._analyze(frame, text), self.loop)
                future.result()  # Wait for completion
            else:
                # General question - still send to VLM
                future = asyncio.run_coroutine_threadsafe(self._analyze(frame, text), self.loop)
                future.result()
        else:
            print("No speech detected.")
            self.mode = "ready"

        self.listening = False

    def _continuous_listen_loop(self):
        """Background thread for continuous listening."""
        while self.continuous_listen and self.running:
            if self.processing or self.listening:
                time.sleep(0.5)
                continue

            self.listening = True
            self.mode = "listening..."

            text = self.voice.listen(timeout=5.0, phrase_limit=15.0)

            if text and self.voice.is_skin_analysis_request(text):
                print(f"\n[VOICE TRIGGER] {text}")
                self.user_query = text

                # Capture current frame
                ret, frame = self.cap.read()
                if ret:
                    frame = cv2.flip(frame, 1)
                    if not self.tts_muted:
                        self.voice.speak("Analyzing now.")
                    # Schedule coroutine on main loop
                    future = asyncio.run_coroutine_threadsafe(self._analyze(frame, text), self.loop)
                    future.result()

            self.listening = False
            time.sleep(0.5)  # Brief pause between listens

    def _calculate_no_lesion_confidence(self, details: dict) -> float:
        """Calculate confidence that no lesion is present based on validation metrics."""
        confidence = 0.5  # Start at 50%

        # Skin ratio: higher = more likely normal skin
        skin_ratio = details.get("skin_ratio", 0)
        if skin_ratio > 0.6:
            confidence += 0.25
        elif skin_ratio > 0.4:
            confidence += 0.15
        elif skin_ratio > 0.2:
            confidence += 0.05

        # Blob count: 0 blobs = more confident no lesion
        blob_count = details.get("blob_count", 0)
        if blob_count == 0:
            confidence += 0.15
        elif blob_count == 1:
            confidence -= 0.1  # Might be a lesion

        # Dark ratio: very low or very high = likely no distinct lesion
        dark_ratio = details.get("dark_ratio", 0)
        if dark_ratio < 0.03:
            confidence += 0.1  # Very uniform, no dark spots
        elif dark_ratio > 0.5:
            confidence += 0.05  # Too dark overall, not a localized lesion

        # Hue variation: low = uniform skin tone, no lesion
        hue_std = details.get("hue_std", 0)
        if hue_std < 5:
            confidence += 0.1
        elif hue_std < 10:
            confidence += 0.05

        # Number of validation failures: more failures = more confident no lesion
        reasons = details.get("reasons", [])
        confidence += len(reasons) * 0.03

        # Clamp to reasonable range
        return max(0.5, min(0.95, confidence))

    async def _analyze(self, frame: np.ndarray, query: str):
        """Run full analysis pipeline."""
        self.processing = True
        self.mode = "analyzing..."

        print("\n" + "=" * 60)
        print(f"ANALYSIS - Query: {query}")
        print("=" * 60)

        cv_result = None
        cv_confidence = 0.0

        # Step 1: CV Model
        cv_skipped = False
        no_lesion_confidence = 0.0
        if self.cv_model.is_loaded:
            print("\n[CV Model]")
            h, w = frame.shape[:2]
            cx, cy = w // 2, h // 2
            roi = frame[cy-200:cy+200, cx-200:cx+200]

            prediction, confidence, meta = self.cv_model.predict(roi)

            if prediction:
                cv_result = prediction
                cv_confidence = confidence
                print(f"  Result: {confidence:.1%} confidence - {prediction.upper()}")
            elif "error" in meta and meta["error"] == "invalid_input":
                cv_skipped = True
                details = meta.get("details", {})

                # Calculate "no lesion" confidence from validation metrics
                reasons = details.get("reasons", [])
                no_lesion_confidence = self._calculate_no_lesion_confidence(details)

                print(f"  Result: {no_lesion_confidence:.1%} confidence - NO LESION DETECTED")
                print(f"  Validation factors:")
                for reason in reasons:
                    print(f"    - {reason}")
            else:
                print("  No prediction")

        # Step 2: VLM Analysis
        print("\n[VLM Analysis]")
        vlm_response = await self._vlm_analyze(frame, query, cv_result, cv_confidence, cv_skipped, no_lesion_confidence)

        print("-" * 60)
        print(vlm_response)
        print("-" * 60)

        self.last_response = vlm_response

        # Step 3: Speak response
        if not self.tts_muted:
            # Create spoken summary
            summary = self._create_spoken_summary(vlm_response, cv_result, cv_confidence, cv_skipped, no_lesion_confidence)
            threading.Thread(target=self.voice.speak, args=(summary,), daemon=True).start()

        self.processing = False
        self.mode = "ready"

    async def _vlm_analyze(self, frame: np.ndarray, query: str, cv_result: str | None, cv_confidence: float, cv_skipped: bool = False, no_lesion_confidence: float = 0.0) -> str:
        _, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 90])
        image_b64 = base64.b64encode(buffer).decode('utf-8')

        cv_context = ""
        if cv_result:
            cv_context = f"\nCV model assessment: {cv_confidence:.1%} confidence - {cv_result.upper()}. Consider this but form your own assessment.\n"
        elif cv_skipped:
            cv_context = f"\nCV model assessment: {no_lesion_confidence:.1%} confidence - NO LESION DETECTED. The image appears to show normal skin without distinct lesions. Provide general skin health observations.\n"

        system = """You are a skin health assistant in a smart mirror. You help users understand their skin health.

IMPORTANT:
- You are NOT providing medical diagnosis
- Always recommend professional dermatologist evaluation for concerns
- Use the ABCDE framework: Asymmetry, Border, Color, Diameter, Evolution
- Be concise but thorough
- Speak naturally as your response will be read aloud"""

        user_prompt = f"""{query}
{cv_context}
Analyze the skin visible in this image. If you see any lesions or moles, assess them using ABCDE criteria.
Keep your response conversational and under 100 words unless more detail is needed."""

        payload = {
            "model": self.vlm_model,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user_prompt, "images": [image_b64]}
            ],
            "stream": False
        }

        try:
            async with self.session.post(self.ollama_url, json=payload) as resp:
                if resp.status == 200:
                    data = await resp.json()
                    return data.get("message", {}).get("content", "Analysis unavailable.")
                return f"Error: {resp.status}"
        except Exception as e:
            return f"Error: {e}"

    def _create_spoken_summary(self, vlm_response: str, cv_result: str | None, cv_confidence: float, cv_skipped: bool = False, no_lesion_confidence: float = 0.0) -> str:
        """Create a concise spoken summary with confidence level."""
        # Start with confidence assessment
        if cv_skipped:
            intro = f"With {no_lesion_confidence:.0%} confidence, I don't detect any lesions in the image. "
        elif cv_result == "malignant":
            intro = f"With {cv_confidence:.0%} confidence, the analysis suggests this could be concerning. "
        elif cv_result == "benign":
            intro = f"With {cv_confidence:.0%} confidence, this appears to be benign. "
        else:
            intro = ""

        # Extract key points for speech (keep it brief)
        lines = vlm_response.split('\n')
        summary_lines = []

        for line in lines[:5]:  # First few lines
            clean = line.strip()
            if clean and not clean.startswith('#') and not clean.startswith('*'):
                # Remove markdown formatting
                clean = re.sub(r'\*\*([^*]+)\*\*', r'\1', clean)
                clean = re.sub(r'\*([^*]+)\*', r'\1', clean)
                if len(clean) > 10:
                    summary_lines.append(clean)

        summary = intro + ' '.join(summary_lines[:2])

        # Truncate if too long
        if len(summary) > 350:
            summary = summary[:347] + "..."

        # Add recommendation if concerning
        if cv_result == "malignant" or "consult" in vlm_response.lower() or "dermatologist" in vlm_response.lower():
            if "dermatologist" not in summary.lower():
                summary += " I recommend consulting a dermatologist."

        return summary if summary else "Analysis complete. Please see the screen for details."

    def _draw_overlay(self, frame):
        h, w = frame.shape[:2]
        cx, cy = w // 2, h // 2

        # Center box
        box_color = (0, 165, 255) if self.listening else (0, 255, 255) if self.processing else (0, 255, 0)
        cv2.rectangle(frame, (cx-200, cy-200), (cx+200, cy+200), box_color, 2)

        # Top bar
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, 0), (w, 100), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)

        # Status
        if self.listening:
            status = "LISTENING - Speak now..."
            color = (0, 165, 255)
        elif self.processing:
            status = "ANALYZING..."
            color = (0, 255, 255)
        else:
            status = 'Ready - Say "check my skin" or press V'
            color = (0, 255, 0)

        cv2.putText(frame, status, (20, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

        # Mode info
        listen_status = "Continuous: ON" if self.continuous_listen else "Continuous: OFF"
        tts_status = "TTS: OFF" if self.tts_muted else "TTS: ON"
        cv2.putText(frame, f"{listen_status} | {tts_status}", (20, 65), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)

        # Controls
        cv2.putText(frame, "V=Talk L=Listen M=Mute Q=Quit", (20, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (180, 180, 180), 1)

        # Instructions
        cv2.putText(frame, "Position lesion/mole in box", (cx-120, cy+220), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1)

        # Last response
        if self.last_response and not self.processing:
            # Bottom panel
            lines = [self.last_response[i:i+100] for i in range(0, min(len(self.last_response), 300), 100)]
            panel_h = 30 + len(lines) * 22
            overlay2 = frame.copy()
            cv2.rectangle(overlay2, (0, h - panel_h), (w, h), (0, 0, 0), -1)
            cv2.addWeighted(overlay2, 0.7, frame, 0.3, 0, frame)

            for i, line in enumerate(lines):
                cv2.putText(frame, line, (20, h - panel_h + 25 + i * 22), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)

        return frame

    async def stop(self):
        print("\nShutting down...")
        if self.cap:
            self.cap.release()
        if self.session:
            await self.session.close()
        cv2.destroyAllWindows()


async def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--camera", type=int, default=0)
    parser.add_argument("--model-path", type=str, default=None)
    parser.add_argument("--vlm-model", type=str, default="qwen3-vl:8b")
    parser.add_argument("--elevenlabs-key", type=str, default=None)
    parser.add_argument("--voice-id", type=str, default="21m00Tcm4TlvDq8ikWAM")
    args = parser.parse_args()

    model_path = args.model_path
    if not model_path:
        default = Path(__file__).parent.parent / "models" / "skin_cancer" / "skin_cancer_lstm.keras"
        if default.exists():
            model_path = str(default)

    demo = SkinVoiceDemo(
        camera_id=args.camera,
        model_path=model_path,
        vlm_model=args.vlm_model,
        elevenlabs_key=args.elevenlabs_key or os.environ.get("ELEVENLABS_API_KEY"),
        voice_id=args.voice_id
    )
    await demo.start()


if __name__ == "__main__":
    asyncio.run(main())
