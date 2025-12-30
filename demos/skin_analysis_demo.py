"""
Skin Analysis Demo - Combines trained CV model with VLM explanations.

This demo:
1. Captures skin images from camera
2. Runs through trained MobileNet+LSTM model (if available)
3. Sends to VLM for detailed ABCDE analysis and explanation
4. Provides spoken feedback via ElevenLabs

Usage:
    python skin_analysis_demo.py --camera 0 --model qwen3-vl:8b

With trained model:
    python skin_analysis_demo.py --model-path ../models/skin_cancer/skin_cancer_lstm.keras

Controls:
    SPACE - Capture and analyze current frame
    H     - View analysis history
    V     - Voice input for questions
    M     - Toggle TTS
    Q     - Quit
"""

import asyncio
import base64
import sys
import time
import os
import threading
import json
from pathlib import Path
from dataclasses import dataclass, field
from datetime import datetime

import cv2
import numpy as np
import aiohttp

sys.path.insert(0, str(Path(__file__).parent.parent))


@dataclass
class SkinAnalysisResult:
    """Result from skin analysis."""
    timestamp: datetime
    image_path: str | None = None
    cv_prediction: str | None = None  # "benign", "malignant", or None
    cv_confidence: float = 0.0
    vlm_analysis: str = ""
    abcde_scores: dict = field(default_factory=dict)
    recommendation: str = ""


class SkinCancerModel:
    """Wrapper for the trained MobileNet + LSTM model."""

    def __init__(self, model_path: str | None = None):
        self.model_path = model_path
        self.feature_extractor = None
        self.classifier = None
        self.is_loaded = False

    def load(self) -> bool:
        """Load the trained model."""
        if not self.model_path or not os.path.exists(self.model_path):
            print("No trained model found. Using VLM-only analysis.")
            return False

        try:
            import tensorflow as tf
            from tensorflow.keras.applications import MobileNet
            from tensorflow.keras.models import Model, load_model

            print(f"Loading model from: {self.model_path}")

            # Load feature extractor
            base = MobileNet(weights='imagenet', include_top=False, pooling='avg', input_shape=(224, 224, 3))
            self.feature_extractor = Model(inputs=base.input, outputs=base.output)

            # Load classifier
            self.classifier = load_model(self.model_path)

            self.is_loaded = True
            print("Model loaded successfully!")
            return True

        except Exception as e:
            print(f"Failed to load model: {e}")
            return False

    def predict(self, image: np.ndarray) -> tuple[str, float, dict]:
        """
        Predict if lesion is benign or malignant.

        Args:
            image: BGR image of the skin lesion (should be cropped to lesion area)

        Returns:
            (prediction, confidence, metadata) - e.g., ("malignant", 0.85, {...})
        """
        if not self.is_loaded:
            return None, 0.0, {"error": "model_not_loaded"}

        try:
            from tensorflow.keras.applications.mobilenet import preprocess_input

            # Validate input - check if image looks like a dermoscopic lesion image
            is_valid, validation_info = self._validate_input(image)

            if not is_valid:
                return None, 0.0, {"error": "invalid_input", "details": validation_info}

            # Preprocess
            img_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            img_resized = cv2.resize(img_rgb, (224, 224))
            img_array = np.expand_dims(img_resized, axis=0).astype(np.float32)
            img_preprocessed = preprocess_input(img_array)

            # Extract features
            features = self.feature_extractor.predict(img_preprocessed, verbose=0)

            # For single image, we create a sequence of [zeros, current]
            # WARNING: This is not ideal - model was trained on 2-image sequences
            # Results may be less accurate than with proper temporal data
            sequence = np.zeros((1, 2, 1024), dtype=np.float32)
            sequence[0, 1, :] = features[0]

            # Predict
            prob = self.classifier.predict(sequence, verbose=0)[0][0]

            metadata = {
                "mode": "single_image",
                "warning": "Model trained on dermoscopic image sequences. Single image results may be less reliable.",
                "validation": validation_info
            }

            if prob > 0.5:
                return "malignant", float(prob), metadata
            else:
                return "benign", float(1 - prob), metadata

        except Exception as e:
            print(f"Prediction error: {e}")
            return None, 0.0, {"error": str(e)}

    def _validate_input(self, image: np.ndarray) -> tuple[bool, dict]:
        """
        Validate if the input image is suitable for skin lesion analysis.

        Checks:
        - Image has sufficient contrast (likely contains a lesion)
        - Image is not too uniform (just skin with no lesion)
        - Image appears to be focused on a small area (not a wide face shot)
        """
        info = {}

        # Convert to grayscale for analysis
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # Check image variance (low variance = uniform image = likely no lesion)
        variance = np.var(gray)
        info["variance"] = float(variance)

        # Check for presence of darker regions (potential lesion)
        mean_intensity = np.mean(gray)
        dark_pixels = np.sum(gray < mean_intensity * 0.7)
        dark_ratio = dark_pixels / gray.size
        info["dark_region_ratio"] = float(dark_ratio)

        # Check color variation in center (lesions have color variation)
        h, w = image.shape[:2]
        center_crop = image[h//4:3*h//4, w//4:3*w//4]
        hsv = cv2.cvtColor(center_crop, cv2.COLOR_BGR2HSV)
        hue_std = np.std(hsv[:,:,0])
        info["hue_variation"] = float(hue_std)

        # Decision logic
        # These thresholds may need tuning based on real dermoscopic images
        is_valid = True
        reasons = []

        if variance < 500:
            reasons.append("Image too uniform - may not contain a distinct lesion")
            is_valid = False

        if dark_ratio < 0.05:
            reasons.append("No distinct darker region detected - ensure lesion is in frame")
            is_valid = False

        if hue_std < 5:
            reasons.append("Low color variation - dermoscopic images typically show color variation in lesions")
            is_valid = False

        info["is_valid"] = is_valid
        info["reasons"] = reasons

        return is_valid, info

    def predict_sequence(self, images: list[np.ndarray]) -> tuple[str, float]:
        """
        Predict using a sequence of images (temporal analysis).

        Args:
            images: List of BGR images in temporal order

        Returns:
            (prediction, confidence)
        """
        if not self.is_loaded or len(images) < 2:
            return self.predict(images[-1]) if images else (None, 0.0)

        try:
            from tensorflow.keras.applications.mobilenet import preprocess_input

            features_list = []

            for img in images[-2:]:  # Use last 2 images
                img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                img_resized = cv2.resize(img_rgb, (224, 224))
                img_array = np.expand_dims(img_resized, axis=0).astype(np.float32)
                img_preprocessed = preprocess_input(img_array)

                features = self.feature_extractor.predict(img_preprocessed, verbose=0)
                features_list.append(features[0])

            sequence = np.array([features_list], dtype=np.float32)
            prob = self.classifier.predict(sequence, verbose=0)[0][0]

            if prob > 0.5:
                return "malignant", float(prob)
            else:
                return "benign", float(1 - prob)

        except Exception as e:
            print(f"Sequence prediction error: {e}")
            return None, 0.0


class SkinAnalysisDemo:
    """Demo combining CV model with VLM analysis."""

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

        # Skin cancer model
        self.cv_model = SkinCancerModel(model_path)

        # Voice
        self.elevenlabs_key = elevenlabs_key
        self.voice_id = voice_id
        self.tts_muted = False

        # State
        self.processing = False
        self.mode = "ready"
        self.last_result: SkinAnalysisResult | None = None
        self.analysis_history: list[SkinAnalysisResult] = []

    async def start(self):
        """Initialize and run the demo."""
        print("=" * 60)
        print("Skin Analysis Demo - CV Model + VLM")
        print("=" * 60)

        # Load CV model
        self.cv_model.load()

        # Connect to Ollama
        print(f"\nConnecting to Ollama ({self.vlm_model})...")
        self.session = aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=120))

        try:
            async with self.session.get("http://localhost:11434/api/tags") as resp:
                if resp.status == 200:
                    print("Ollama connected.")
                else:
                    print("WARNING: Ollama not responding")
        except Exception as e:
            print(f"ERROR: Cannot connect to Ollama: {e}")
            return

        # Open camera
        print(f"\nOpening camera {self.camera_id}...")
        self.cap = cv2.VideoCapture(self.camera_id)

        if not self.cap.isOpened():
            print(f"ERROR: Cannot open camera {self.camera_id}")
            return

        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

        print("\n" + "=" * 60)
        print("Controls:")
        print("  SPACE - Capture and analyze skin")
        print("  R     - Draw ROI (region of interest)")
        print("  H     - Show history")
        print("  M     - Toggle TTS")
        print("  Q     - Quit")
        print("=" * 60)
        print("\nPosition the skin area in the center of the frame.")
        print("For best results, ensure good lighting and focus.\n")

        self.running = True
        await self._run_loop()

    async def _run_loop(self):
        """Main processing loop."""
        while self.running:
            ret, frame = self.cap.read()
            if not ret:
                await asyncio.sleep(0.1)
                continue

            frame = cv2.flip(frame, 1)

            # Draw overlay
            display = self._draw_overlay(frame.copy())
            cv2.imshow("Skin Analysis Demo", display)

            key = cv2.waitKey(1) & 0xFF

            if key == ord('q'):
                self.running = False
            elif key == ord(' ') and not self.processing:
                asyncio.create_task(self._analyze_skin(frame.copy()))
            elif key == ord('h'):
                self._show_history()
            elif key == ord('m'):
                self.tts_muted = not self.tts_muted
                print(f"TTS {'muted' if self.tts_muted else 'unmuted'}")

            await asyncio.sleep(0.01)

        await self.stop()

    async def _analyze_skin(self, frame: np.ndarray):
        """Run full analysis pipeline."""
        self.processing = True
        self.mode = "analyzing..."
        start_time = time.time()

        print("\n" + "=" * 60)
        print("SKIN ANALYSIS")
        print("=" * 60)

        result = SkinAnalysisResult(timestamp=datetime.now())

        # Step 1: CV Model prediction (if available)
        if self.cv_model.is_loaded:
            print("\n[1/2] Running CV model...")

            # Get the center region (where user should position lesion)
            h, w = frame.shape[:2]
            center_size = 200
            cx, cy = w // 2, h // 2
            roi = frame[cy-center_size:cy+center_size, cx-center_size:cx+center_size]

            prediction, confidence, metadata = self.cv_model.predict(roi)

            if prediction:
                result.cv_prediction = prediction
                result.cv_confidence = confidence
                print(f"CV Model: {prediction.upper()} (confidence: {confidence:.1%})")
                if "warning" in metadata:
                    print(f"Warning: {metadata['warning']}")
            elif "error" in metadata:
                if metadata["error"] == "invalid_input":
                    print("CV Model: SKIPPED - Input validation failed")
                    for reason in metadata.get("details", {}).get("reasons", []):
                        print(f"  - {reason}")
                    print("  Tip: Position a close-up of a skin lesion/mole in the green box")
                else:
                    print(f"CV Model: Error - {metadata['error']}")
            else:
                print("CV Model: No prediction")

        # Step 2: VLM Analysis
        print("\n[2/2] Running VLM analysis...")
        vlm_response = await self._vlm_analyze(frame, result.cv_prediction, result.cv_confidence)

        result.vlm_analysis = vlm_response

        # Parse VLM response for structured data
        result.recommendation = self._extract_recommendation(vlm_response)

        # Store result
        self.last_result = result
        self.analysis_history.append(result)

        elapsed = time.time() - start_time
        print(f"\n[{elapsed:.1f}s] Analysis complete")
        print("-" * 60)
        print(vlm_response)
        print("-" * 60)

        # Speak summary
        if not self.tts_muted and self.elevenlabs_key:
            summary = self._create_spoken_summary(result)
            threading.Thread(target=self._speak, args=(summary,), daemon=True).start()

        self.processing = False
        self.mode = "ready"

    async def _vlm_analyze(
        self,
        frame: np.ndarray,
        cv_prediction: str | None,
        cv_confidence: float
    ) -> str:
        """Send to VLM for detailed analysis."""

        # Encode image
        _, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 90])
        image_b64 = base64.b64encode(buffer).decode('utf-8')

        # Build context from CV model
        cv_context = ""
        if cv_prediction:
            cv_context = f"""
A computer vision model trained on dermoscopic images has analyzed this image.
Model prediction: {cv_prediction.upper()}
Model confidence: {cv_confidence:.1%}

Please consider this prediction in your analysis, but form your own independent assessment.
"""

        system_prompt = """You are a dermatology education assistant helping with skin health awareness through a smart mirror.

IMPORTANT DISCLAIMERS:
- You are NOT providing medical diagnosis
- You are helping with skin health education and awareness
- Always recommend professional dermatologist evaluation for any concerns
- Your analysis is for educational purposes only

When analyzing skin images, use the ABCDE framework:
- A (Asymmetry): Is one half unlike the other?
- B (Border): Is the border irregular, ragged, or blurred?
- C (Color): Is there color variation (tan, brown, black, red, white, blue)?
- D (Diameter): Is it larger than 6mm (pencil eraser size)?
- E (Evolution): Has it changed over time? (If history provided)

Provide your analysis in this format:
1. ABCDE Assessment (brief for each criterion)
2. Overall Impression
3. Recommendation (always include "consult a dermatologist" for any concerns)

Be concise but thorough. Focus on objective observations."""

        user_prompt = f"""Please analyze this skin image for educational purposes.
{cv_context}
Provide your ABCDE assessment and recommendations."""

        payload = {
            "model": self.vlm_model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt, "images": [image_b64]}
            ],
            "stream": False
        }

        try:
            async with self.session.post(self.ollama_url, json=payload) as resp:
                if resp.status == 200:
                    data = await resp.json()
                    return data.get("message", {}).get("content", "No analysis available")
                else:
                    return f"VLM Error: {resp.status}"
        except Exception as e:
            return f"VLM Error: {e}"

    def _extract_recommendation(self, vlm_response: str) -> str:
        """Extract recommendation from VLM response."""
        lower = vlm_response.lower()

        if "consult" in lower or "see a dermatologist" in lower or "professional evaluation" in lower:
            return "Recommend professional evaluation"
        elif "monitor" in lower or "watch" in lower:
            return "Monitor for changes"
        else:
            return "See full analysis"

    def _create_spoken_summary(self, result: SkinAnalysisResult) -> str:
        """Create a brief spoken summary."""
        parts = []

        if result.cv_prediction:
            parts.append(f"The analysis model classified this as {result.cv_prediction} with {result.cv_confidence:.0%} confidence.")

        if result.recommendation:
            parts.append(result.recommendation + ".")

        parts.append("Please see the full analysis on screen for details.")

        return " ".join(parts)

    def _speak(self, text: str):
        """Speak text using ElevenLabs."""
        if not self.elevenlabs_key:
            return

        try:
            from elevenlabs.client import ElevenLabs
            import tempfile

            client = ElevenLabs(api_key=self.elevenlabs_key)
            audio = client.text_to_speech.convert(
                voice_id=self.voice_id,
                text=text,
                model_id="eleven_turbo_v2_5",
                output_format="mp3_44100_128"
            )

            with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as f:
                for chunk in audio:
                    f.write(chunk)
                path = f.name

            # Play
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

    def _show_history(self):
        """Show analysis history."""
        print("\n" + "=" * 60)
        print("ANALYSIS HISTORY")
        print("=" * 60)

        if not self.analysis_history:
            print("No analyses yet.")
        else:
            for i, result in enumerate(self.analysis_history[-5:], 1):
                print(f"\n[{i}] {result.timestamp.strftime('%H:%M:%S')}")
                if result.cv_prediction:
                    print(f"    CV: {result.cv_prediction} ({result.cv_confidence:.1%})")
                print(f"    Recommendation: {result.recommendation}")

        print("=" * 60 + "\n")

    def _draw_overlay(self, frame: np.ndarray) -> np.ndarray:
        """Draw UI overlay."""
        h, w = frame.shape[:2]

        # Center guide
        center_size = 200
        cx, cy = w // 2, h // 2
        cv2.rectangle(
            frame,
            (cx - center_size, cy - center_size),
            (cx + center_size, cy + center_size),
            (0, 255, 0) if not self.processing else (0, 255, 255),
            2
        )

        # Top bar
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, 0), (w, 80), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)

        # Status
        status = "ANALYZING..." if self.processing else "Ready - Press SPACE to analyze"
        color = (0, 255, 255) if self.processing else (0, 255, 0)
        cv2.putText(frame, status, (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

        # Model status
        model_status = "CV Model: Loaded" if self.cv_model.is_loaded else "CV Model: VLM-only mode"
        cv2.putText(frame, model_status, (20, 55), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)

        # Instructions
        cv2.putText(frame, "Position LESION/MOLE close-up in green box", (cx - 180, cy + center_size + 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1)
        cv2.putText(frame, "(CV model trained on dermoscopic images)", (cx - 150, cy + center_size + 55),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (180, 180, 180), 1)

        # Last result
        if self.last_result and not self.processing:
            # Bottom panel
            panel_h = 150
            cv2.rectangle(overlay, (0, h - panel_h), (w, h), (0, 0, 0), -1)
            cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)

            y = h - panel_h + 25
            cv2.putText(frame, "Last Analysis:", (20, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)

            if self.last_result.cv_prediction:
                pred_color = (0, 0, 255) if self.last_result.cv_prediction == "malignant" else (0, 255, 0)
                cv2.putText(frame, f"CV: {self.last_result.cv_prediction.upper()} ({self.last_result.cv_confidence:.0%})",
                            (20, y + 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, pred_color, 1)

            cv2.putText(frame, f"Rec: {self.last_result.recommendation}",
                        (20, y + 55), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

            # Wrap VLM summary
            summary = self.last_result.vlm_analysis[:200] + "..." if len(self.last_result.vlm_analysis) > 200 else self.last_result.vlm_analysis
            cv2.putText(frame, summary[:100], (20, y + 80), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1)

        return frame

    async def stop(self):
        """Cleanup."""
        print("\nShutting down...")
        if self.cap:
            self.cap.release()
        if self.session:
            await self.session.close()
        cv2.destroyAllWindows()


async def main():
    import argparse

    parser = argparse.ArgumentParser(description="Skin Analysis Demo")
    parser.add_argument("--camera", type=int, default=0)
    parser.add_argument("--model-path", type=str, default=None, help="Path to trained .keras model")
    parser.add_argument("--vlm-model", type=str, default="qwen3-vl:8b")
    parser.add_argument("--elevenlabs-key", type=str, default=None)
    parser.add_argument("--voice-id", type=str, default="21m00Tcm4TlvDq8ikWAM")
    args = parser.parse_args()

    # Check for model in default location
    model_path = args.model_path
    if not model_path:
        default_path = Path(__file__).parent.parent / "models" / "skin_cancer" / "skin_cancer_lstm.keras"
        if default_path.exists():
            model_path = str(default_path)

    demo = SkinAnalysisDemo(
        camera_id=args.camera,
        model_path=model_path,
        vlm_model=args.vlm_model,
        elevenlabs_key=args.elevenlabs_key or os.environ.get("ELEVENLABS_API_KEY"),
        voice_id=args.voice_id
    )
    await demo.start()


if __name__ == "__main__":
    asyncio.run(main())
