# Copyright (c) 2026 Alessandro Orrù
# Licensed under MIT

from pathlib import Path
from andromeda.config import AudioConfig, StopWordConfig, WakeWordConfig
import logging
import threading
import numpy as np

logger = logging.getLogger(__name__)


# Listens for the wake word during TTS playback to allow barge-in interruption
# Uses an energy gate to filter out TTS echo picked up by the microphone
class StopWordListener:

    def __init__(self, audio_cfg: AudioConfig, wake_cfg: WakeWordConfig, stop_cfg: StopWordConfig) -> None:
        self._audio_cfg = audio_cfg
        self._wake_cfg = wake_cfg
        self._stop_cfg = stop_cfg
        self._model = None
        self._detected = threading.Event()
        self._is_active = False
        self._energy_threshold: float = 0.0
        self._lock = threading.Lock()


    # Load wake word model (same as WakeWordDetector but independent instance)
    def initialize(self) -> None:
        try:
            import openwakeword
            from openwakeword.model import Model

            openwakeword.utils.download_models()
            model_path = Path(self._wake_cfg.model_path)

            if model_path.exists():
                self._model = Model(wakeword_models=[str(model_path)], inference_framework="onnx")
                logger.info("StopWordListener loaded model: %s", model_path)
            else:
                self._model = Model(wakeword_models=["hey_jarvis"], inference_framework="onnx")
                logger.warning("Custom model not found, StopWordListener using 'hey_jarvis'")

        except Exception:
            logger.exception("Failed to initialize StopWordListener")
            self._model = None


    # Start listening for stop word
    def start(self, energy_threshold: float = 0.0) -> None:
        self._energy_threshold = energy_threshold
        self._detected.clear()
        self._is_active = True

        if self._model:
            with self._lock:
                self._model.reset()

        logger.debug("StopWordListener started (energy_threshold=%.1f)", energy_threshold)


    # Stop listening
    def stop(self) -> None:
        self._is_active = False
        self._detected.set()  # unblock any waiter


    # Process a single audio frame. Called from audio callback thread
    def process_frame(self, frame_bytes: bytes) -> None:
        if not self._is_active or self._model is None:
            return

        # Energy gate: filter out TTS echo and background noise
        audio = np.frombuffer(frame_bytes, dtype=np.int16)
        if self._energy_threshold > 0.0:
            rms = float(np.sqrt(np.mean(audio.astype(np.float32) ** 2)))
            if rms < self._energy_threshold:
                return  # Below threshold — likely TTS echo

        # Run wake word detection
        with self._lock:
            prediction = self._model.predict(audio)

        for model_name, score in prediction.items():
            if score > self._wake_cfg.threshold:
                logger.info("Stop word detected: %s (score=%.3f)", model_name, score)
                self._detected.set()
                with self._lock:
                    self._model.reset()
                return


    # Block until stop word is detected
    def wait_for_detection(self, timeout: float | None = None) -> bool:
        return self._detected.wait(timeout=timeout)
