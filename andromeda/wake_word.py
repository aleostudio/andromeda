# Copyright (c) 2026 Alessandro Orrù
# Licensed under MIT

from pathlib import Path
from andromeda.config import AudioConfig, WakeWordConfig
import logging
import threading
import numpy as np

logger = logging.getLogger(__name__)


# Detects custom wake word from audio frames using OpenWakeWord
# Custom word how-to: https://github.com/dscripka/openwakeword?tab=readme-ov-file#training-new-models
# Built-in word: "hey jarvis"
class WakeWordDetector:

    def __init__(self, audio_cfg: AudioConfig, wake_cfg: WakeWordConfig) -> None:
        self._audio_cfg = audio_cfg
        self._wake_cfg = wake_cfg
        self._model = None
        self._detected = threading.Event()
        self._shutdown = False
        self._lock = threading.Lock()


    # Download and load model
    def initialize(self) -> None:
        try:
            import openwakeword
            from openwakeword.model import Model

            openwakeword.utils.download_models()
            model_path = Path(self._wake_cfg.model_path)

            if model_path.exists():
                # Defined model
                self._model = Model(wakeword_models=[str(model_path)], inference_framework="onnx")
                logger.info("Loaded custom wake word model: %s", model_path)
            else:
                # Fallback: built-in model "hey_jarvis"
                self._model = Model(wakeword_models=["hey_jarvis"], inference_framework="onnx")
                logger.warning("Custom model not found at %s, using built-in 'hey_jarvis'", model_path)

        except Exception:
            logger.exception("Failed to initialize wake word detector")
            raise


    # Process a single audio frame. Called from audio callback thread
    def process_frame(self, frame_bytes: bytes) -> None:
        if self._model is None:
            return

        audio = np.frombuffer(frame_bytes, dtype=np.int16)

        with self._lock:
            prediction = self._model.predict(audio)

        # Check all model predictions against threshold
        for model_name, score in prediction.items():
            if score > self._wake_cfg.threshold:
                logger.info("Wake word detected: %s (score=%.3f)", model_name, score)
                self._detected.set()
                # Reset model state to avoid repeated triggers
                with self._lock:
                    self._model.reset()

                return


    # Block until wake word is detected. Returns True if detected, False on shutdown/timeout
    def wait_for_detection(self, timeout: float | None = None) -> bool:
        self._detected.wait(timeout=timeout)
        if self._shutdown:
            return False
        detected = self._detected.is_set()
        self._detected.clear()
        return detected


    # Reset detection state
    def reset(self) -> None:
        self._detected.clear()
        if self._model:
            with self._lock:
                self._model.reset()


    # Unblock any waiting thread and prevent further detections
    def shutdown(self) -> None:
        self._shutdown = True
        self._detected.set()
