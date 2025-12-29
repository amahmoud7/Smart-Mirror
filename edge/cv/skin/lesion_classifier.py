"""Skin lesion classification module - ported from original prototype."""

import time
from pathlib import Path
import numpy as np

from ..base import CVModule, SkinAnalysisResult


class LesionClassifier(CVModule):
    """
    Skin lesion classifier using MobileNet + LSTM.

    Ported from the original Smart Mirror prototype.
    Designed for dermoscopic images but can work with camera input (lower accuracy).

    Architecture:
    - MobileNet: Feature extraction (1024-dim vectors)
    - LSTM: Temporal sequence analysis for progression tracking

    For single-image analysis (no history), uses direct MobileNet classification.
    For time-series analysis, uses LSTM to detect benign->malignant progression.
    """

    def __init__(self, config: dict | None = None):
        super().__init__(config)
        self.feature_extractor = None
        self.sequence_classifier = None
        self.single_image_classifier = None

        # Model paths
        self.models_dir = Path(config.get("models_dir", "./models/skin_cancer")) if config else Path("./models/skin_cancer")

        # Image preprocessing
        self.input_size = (224, 224)

        # History buffer for time-series
        self._feature_history: dict[str, list] = {}  # patient_id -> list of features
        self._max_history = 2  # Match original model (2-image sequences)

    @property
    def name(self) -> str:
        return "lesion_classifier"

    def initialize(self) -> None:
        """Load classification models."""
        try:
            self._load_feature_extractor()
            self._load_sequence_classifier()
            self._initialized = True
        except Exception as e:
            raise RuntimeError(f"Failed to load skin lesion models: {e}")

    def _load_feature_extractor(self) -> None:
        """Load MobileNet feature extractor."""
        # TODO: Load actual model
        # from tensorflow.keras.applications import MobileNet
        # from tensorflow.keras.models import Model
        #
        # base_model = MobileNet(
        #     weights='imagenet',
        #     include_top=False,
        #     pooling='avg',
        #     input_shape=(224, 224, 3)
        # )
        # self.feature_extractor = base_model
        pass

    def _load_sequence_classifier(self) -> None:
        """Load LSTM sequence classifier."""
        model_path = self.models_dir / "lstm_model.h5"

        # TODO: Load trained LSTM model
        # if model_path.exists():
        #     from tensorflow.keras.models import load_model
        #     self.sequence_classifier = load_model(model_path)
        # else:
        #     # Fallback: create model architecture for inference
        #     from tensorflow.keras.models import Sequential
        #     from tensorflow.keras.layers import LSTM, Dense, Masking
        #
        #     self.sequence_classifier = Sequential([
        #         Masking(mask_value=0, input_shape=(2, 1024)),
        #         LSTM(64, return_sequences=False),
        #         Dense(1, activation='sigmoid')
        #     ])
        pass

    def process(
        self,
        frame: np.ndarray,
        context: dict | None = None
    ) -> SkinAnalysisResult | None:
        """
        Analyze skin lesion in frame.

        Args:
            frame: BGR image (full frame or cropped ROI)
            context: Optional context with:
                - 'roi': (x, y, w, h) to crop lesion region
                - 'patient_id': for time-series tracking
                - 'anatomical_site': body location

        Returns:
            SkinAnalysisResult with classification
        """
        if not self._initialized:
            raise RuntimeError("Module not initialized. Call initialize() first.")

        timestamp = time.time()
        roi_bbox = (0, 0, 0, 0)

        # Extract ROI if provided
        if context and "roi" in context:
            x, y, w, h = context["roi"]
            roi_bbox = (x, y, w, h)
            frame = frame[y:y+h, x:x+w]

        if frame.size == 0:
            return None

        # Preprocess image
        processed = self._preprocess_image(frame)

        # Extract features
        features = self._extract_features(processed)

        # Determine classification mode
        patient_id = context.get("patient_id") if context else None

        if patient_id and self.sequence_classifier:
            # Time-series mode: use LSTM
            classification, confidence, features_dict = self._classify_sequence(
                features, patient_id
            )
        else:
            # Single-image mode: direct classification
            classification, confidence, features_dict = self._classify_single(features)

        return SkinAnalysisResult(
            timestamp=timestamp,
            confidence=confidence,
            module_name=self.name,
            classification=classification,
            lesion_features=features_dict,
            roi_bbox=roi_bbox
        )

    def _preprocess_image(self, image: np.ndarray) -> np.ndarray:
        """Preprocess image for model input."""
        import cv2

        # Resize
        resized = cv2.resize(image, self.input_size)

        # Convert BGR to RGB
        rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)

        # Normalize for MobileNet
        # from tensorflow.keras.applications.mobilenet import preprocess_input
        # processed = preprocess_input(rgb.astype(np.float32))

        # Placeholder normalization
        processed = rgb.astype(np.float32) / 127.5 - 1.0

        return processed

    def _extract_features(self, image: np.ndarray) -> np.ndarray:
        """Extract feature vector from preprocessed image."""
        if self.feature_extractor is None:
            # Return dummy features for now
            return np.zeros(1024, dtype=np.float32)

        # features = self.feature_extractor.predict(np.expand_dims(image, 0))[0]
        # return features

        return np.zeros(1024, dtype=np.float32)

    def _classify_sequence(
        self,
        current_features: np.ndarray,
        patient_id: str
    ) -> tuple[str, float, dict]:
        """
        Classify using time-series LSTM.

        Tracks feature history per patient and uses sequence for classification.
        """
        # Update history
        if patient_id not in self._feature_history:
            self._feature_history[patient_id] = []

        self._feature_history[patient_id].append(current_features)

        # Keep only last N features
        if len(self._feature_history[patient_id]) > self._max_history:
            self._feature_history[patient_id].pop(0)

        history = self._feature_history[patient_id]

        if len(history) < 2:
            # Not enough history for sequence classification
            return "insufficient_data", 0.0, {"status": "collecting_history"}

        # Pad sequence
        sequence = np.array(history)
        # padded = pad_sequences([sequence], maxlen=2, padding='pre', dtype='float32')

        # Predict
        # prob = self.sequence_classifier.predict(padded)[0][0]
        prob = 0.5  # Placeholder

        classification = "malignant" if prob > 0.5 else "benign"
        confidence = float(prob if prob > 0.5 else 1 - prob)

        return classification, confidence, {
            "progression_score": float(prob),
            "history_length": len(history)
        }

    def _classify_single(self, features: np.ndarray) -> tuple[str, float, dict]:
        """Classify single image without temporal context."""
        # For single images, we'd need a different classifier head
        # or use heuristics on the feature vector

        # Placeholder: random classification
        # In production, would use a trained single-image classifier
        return "uncertain", 0.3, {"note": "single_image_mode"}

    def clear_patient_history(self, patient_id: str) -> None:
        """Clear stored feature history for a patient."""
        if patient_id in self._feature_history:
            del self._feature_history[patient_id]

    def cleanup(self) -> None:
        """Release model resources."""
        self.feature_extractor = None
        self.sequence_classifier = None
        self._feature_history.clear()
        self._initialized = False


# ABCDE feature extraction (for VLM context)
def extract_abcde_features(image: np.ndarray) -> dict:
    """
    Extract ABCDE dermatology features from lesion image.

    This provides structured data for the VLM to reason about.

    A - Asymmetry
    B - Border
    C - Color
    D - Diameter (relative to image)
    E - Evolution (requires history)
    """
    import cv2

    features = {}

    # Convert to grayscale for analysis
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Threshold to find lesion boundary
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    # Find contours
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if not contours:
        return {"error": "no_lesion_detected"}

    # Get largest contour (assumed to be the lesion)
    lesion_contour = max(contours, key=cv2.contourArea)

    # A - Asymmetry (compare halves)
    moments = cv2.moments(lesion_contour)
    if moments["m00"] > 0:
        cx = int(moments["m10"] / moments["m00"])
        cy = int(moments["m01"] / moments["m00"])

        # Simple asymmetry: compare left/right areas
        left_mask = binary.copy()
        left_mask[:, cx:] = 0
        right_mask = binary.copy()
        right_mask[:, :cx] = 0

        left_area = cv2.countNonZero(left_mask)
        right_area = cv2.countNonZero(right_mask)

        asymmetry_score = abs(left_area - right_area) / max(left_area + right_area, 1)
        features["asymmetry"] = {
            "score": float(asymmetry_score),
            "interpretation": "high" if asymmetry_score > 0.3 else "low"
        }

    # B - Border irregularity (perimeter vs area ratio)
    perimeter = cv2.arcLength(lesion_contour, True)
    area = cv2.contourArea(lesion_contour)

    if area > 0:
        circularity = 4 * np.pi * area / (perimeter ** 2)
        border_irregularity = 1 - circularity
        features["border"] = {
            "irregularity_score": float(border_irregularity),
            "interpretation": "irregular" if border_irregularity > 0.3 else "regular"
        }

    # C - Color variation
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    lesion_hsv = hsv[binary > 0]

    if len(lesion_hsv) > 0:
        color_std = np.std(lesion_hsv, axis=0)
        features["color"] = {
            "hue_variation": float(color_std[0]),
            "saturation_variation": float(color_std[1]),
            "value_variation": float(color_std[2]),
            "interpretation": "varied" if np.mean(color_std) > 30 else "uniform"
        }

    # D - Diameter (relative to image)
    x, y, w, h = cv2.boundingRect(lesion_contour)
    diameter = max(w, h)
    relative_diameter = diameter / max(image.shape[:2])
    features["diameter"] = {
        "pixels": int(diameter),
        "relative_to_image": float(relative_diameter),
        "interpretation": "large" if relative_diameter > 0.5 else "small"
    }

    return features
