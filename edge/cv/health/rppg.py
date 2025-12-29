"""Remote Photoplethysmography (rPPG) for heart rate estimation."""

import time
from collections import deque
import numpy as np

from ..base import CVModule, HealthSignalResult


class HeartRateEstimator(CVModule):
    """
    Non-contact heart rate estimation using rPPG.

    Extracts blood volume pulse from facial skin color variations.
    Requires stable face ROI from face detector.

    Methods supported:
    - GREEN: Simple green channel analysis
    - CHROM: Chrominance-based (De Haan & Jeanne, 2013)
    - POS: Plane-orthogonal-to-skin (Wang et al., 2017)
    """

    def __init__(self, config: dict | None = None):
        super().__init__(config)
        self.method = config.get("method", "CHROM") if config else "CHROM"
        self.buffer_size = config.get("buffer_seconds", 10) if config else 10
        self.fps = config.get("fps", 30) if config else 30

        # Signal buffers
        self._rgb_buffer: deque = deque(maxlen=self.buffer_size * self.fps)
        self._timestamp_buffer: deque = deque(maxlen=self.buffer_size * self.fps)
        self._last_hr = 0.0

    @property
    def name(self) -> str:
        return "heart_rate_estimator"

    def initialize(self) -> None:
        """Initialize signal processing components."""
        self._rgb_buffer.clear()
        self._timestamp_buffer.clear()
        self._initialized = True

    def process(self, frame: np.ndarray, context: dict | None = None) -> HealthSignalResult | None:
        """
        Process frame for heart rate estimation.

        Args:
            frame: BGR image
            context: Must contain 'face_roi' with (x, y, w, h) of forehead/cheek region

        Returns:
            HealthSignalResult with heart rate, or None if insufficient data
        """
        if not self._initialized:
            raise RuntimeError("Module not initialized. Call initialize() first.")

        timestamp = time.time()

        # Require face ROI from face detector
        if context is None or "face_roi" not in context:
            return None

        roi = context["face_roi"]
        x, y, w, h = roi

        # Extract forehead region (upper 1/3 of face)
        forehead_y = y
        forehead_h = h // 3
        forehead_roi = frame[forehead_y:forehead_y + forehead_h, x:x + w]

        if forehead_roi.size == 0:
            return None

        # Compute spatial average of RGB
        rgb_mean = np.mean(forehead_roi, axis=(0, 1))  # BGR -> still works
        self._rgb_buffer.append(rgb_mean)
        self._timestamp_buffer.append(timestamp)

        # Need at least 3 seconds of data
        min_samples = 3 * self.fps
        if len(self._rgb_buffer) < min_samples:
            return HealthSignalResult(
                timestamp=timestamp,
                confidence=0.0,
                module_name=self.name,
                signal_type="heart_rate",
                value=0.0,
                unit="bpm",
                raw_data={"status": "buffering", "samples": len(self._rgb_buffer)}
            )

        # Estimate heart rate
        hr, confidence = self._estimate_heart_rate()
        self._last_hr = hr

        return HealthSignalResult(
            timestamp=timestamp,
            confidence=confidence,
            module_name=self.name,
            signal_type="heart_rate",
            value=hr,
            unit="bpm"
        )

    def _estimate_heart_rate(self) -> tuple[float, float]:
        """
        Estimate heart rate from RGB buffer.

        Returns:
            (heart_rate_bpm, confidence)
        """
        rgb_signal = np.array(self._rgb_buffer)
        timestamps = np.array(self._timestamp_buffer)

        # Compute actual FPS from timestamps
        actual_fps = len(timestamps) / (timestamps[-1] - timestamps[0])

        if self.method == "GREEN":
            pulse_signal = self._green_method(rgb_signal)
        elif self.method == "CHROM":
            pulse_signal = self._chrom_method(rgb_signal)
        elif self.method == "POS":
            pulse_signal = self._pos_method(rgb_signal)
        else:
            pulse_signal = self._green_method(rgb_signal)

        # Bandpass filter (0.7 - 3.5 Hz = 42-210 bpm)
        pulse_signal = self._bandpass_filter(pulse_signal, actual_fps, 0.7, 3.5)

        # FFT to find dominant frequency
        hr, confidence = self._fft_peak_detection(pulse_signal, actual_fps)

        return hr, confidence

    def _green_method(self, rgb: np.ndarray) -> np.ndarray:
        """Simple green channel extraction."""
        return rgb[:, 1]  # Green channel (BGR format)

    def _chrom_method(self, rgb: np.ndarray) -> np.ndarray:
        """CHROM method: chrominance-based rPPG."""
        # Normalize
        rgb_norm = rgb / np.mean(rgb, axis=0)

        # Chrominance signals
        Xs = 3 * rgb_norm[:, 2] - 2 * rgb_norm[:, 1]  # 3R - 2G
        Ys = 1.5 * rgb_norm[:, 2] + rgb_norm[:, 1] - 1.5 * rgb_norm[:, 0]  # 1.5R + G - 1.5B

        # Combine
        alpha = np.std(Xs) / (np.std(Ys) + 1e-8)
        pulse = Xs - alpha * Ys

        return pulse

    def _pos_method(self, rgb: np.ndarray) -> np.ndarray:
        """POS method: plane-orthogonal-to-skin."""
        # Normalize
        rgb_norm = rgb / np.mean(rgb, axis=0)

        # Project onto plane orthogonal to skin tone
        Xs = rgb_norm[:, 1] - rgb_norm[:, 0]  # G - B
        Ys = rgb_norm[:, 1] + rgb_norm[:, 0] - 2 * rgb_norm[:, 2]  # G + B - 2R

        # Combine
        alpha = np.std(Xs) / (np.std(Ys) + 1e-8)
        pulse = Xs + alpha * Ys

        return pulse

    def _bandpass_filter(self, signal: np.ndarray, fps: float, low: float, high: float) -> np.ndarray:
        """Apply bandpass filter to signal."""
        from scipy import signal as scipy_signal

        nyquist = fps / 2
        low_norm = low / nyquist
        high_norm = high / nyquist

        # Clamp to valid range
        low_norm = max(0.01, min(low_norm, 0.99))
        high_norm = max(low_norm + 0.01, min(high_norm, 0.99))

        b, a = scipy_signal.butter(2, [low_norm, high_norm], btype='band')
        filtered = scipy_signal.filtfilt(b, a, signal)

        return filtered

    def _fft_peak_detection(self, signal: np.ndarray, fps: float) -> tuple[float, float]:
        """Find dominant frequency using FFT."""
        # Window the signal
        window = np.hanning(len(signal))
        windowed = signal * window

        # FFT
        fft = np.fft.rfft(windowed)
        freqs = np.fft.rfftfreq(len(windowed), 1 / fps)
        magnitude = np.abs(fft)

        # Find peak in valid HR range (42-210 bpm = 0.7-3.5 Hz)
        valid_mask = (freqs >= 0.7) & (freqs <= 3.5)
        valid_freqs = freqs[valid_mask]
        valid_mag = magnitude[valid_mask]

        if len(valid_mag) == 0:
            return 0.0, 0.0

        peak_idx = np.argmax(valid_mag)
        peak_freq = valid_freqs[peak_idx]
        hr = peak_freq * 60  # Convert Hz to BPM

        # Confidence based on peak prominence
        confidence = valid_mag[peak_idx] / (np.mean(valid_mag) + 1e-8)
        confidence = min(1.0, confidence / 5)  # Normalize

        return float(hr), float(confidence)

    def cleanup(self) -> None:
        self._rgb_buffer.clear()
        self._timestamp_buffer.clear()
        self._initialized = False
