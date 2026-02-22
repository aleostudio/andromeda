# Copyright (c) 2026 Alessandro Orrù
# Licensed under MIT

import logging
import threading
import time
import numpy as np
import webrtcvad
from andromeda.config import AudioConfig, VADConfig

logger = logging.getLogger("[ VAD ]")


# Detect speech start/end using WebRTC VAD with adaptive energy filtering
# Track speech state and signals when the user has stopped talking based on a configurable silence timeout
class VoiceActivityDetector:

    def __init__(self, audio_cfg: AudioConfig, vad_cfg: VADConfig) -> None:
        self._audio_cfg = audio_cfg
        self._vad_cfg = vad_cfg
        self._vad = webrtcvad.Vad(vad_cfg.aggressiveness)

        # State tracking
        self._is_active = False
        self._speech_detected = False
        self._last_speech_time: float = 0.0
        self._start_time: float = 0.0

        # Adaptive energy gate
        self._energy_threshold: float = 0.0
        self._last_decay_time: float = 0.0

        # Events
        self._speech_ended = threading.Event()
        self._lock = threading.Lock()


    # Set energy threshold for filtering background noise
    def set_energy_threshold(self, threshold: float) -> None:
        with self._lock:
            self._energy_threshold = threshold
            self._last_decay_time = time.monotonic()
        if threshold > 0:
            logger.debug("VAD energy threshold set to %.1f", threshold)


    # Start monitoring for speech activity
    def start(self) -> None:
        with self._lock:
            self._is_active = True
            self._speech_detected = False
            self._last_speech_time = time.monotonic()
            self._start_time = time.monotonic()
            self._last_decay_time = time.monotonic()
            self._speech_ended.clear()

        logger.debug("VAD monitoring started")


    # Stop monitoring
    def stop(self) -> None:
        with self._lock:
            self._is_active = False
            self._speech_ended.set()


    # Process audio frame for speech detection. Called from audio thread
    def process_frame(self, frame_bytes: bytes, frame_array: np.ndarray) -> None:
        if not self._is_active:
            return

        try:
            is_speech = self._vad.is_speech(frame_bytes, sample_rate=self._audio_cfg.sample_rate)
        except Exception:
            return

        # Energy gate: reject speech frames below threshold
        if is_speech and self._energy_threshold > 0.0:
            try:
                rms = float(np.sqrt(np.mean(frame_array.astype(np.float32) ** 2)))
                if rms < self._energy_threshold:
                    is_speech = False
            except Exception:
                pass  # Never crash the audio callback

        now = time.monotonic()

        with self._lock:
            # Apply energy decay over time so threshold adapts downward
            if self._energy_threshold > 0.0:
                elapsed_since_decay = now - self._last_decay_time
                if elapsed_since_decay >= 1.0:
                    decay_steps = int(elapsed_since_decay)
                    self._energy_threshold *= self._vad_cfg.energy_decay_rate ** decay_steps
                    self._last_decay_time = now

            if is_speech:
                self._speech_detected = True
                self._last_speech_time = now

            # Check conditions for ending
            elapsed = now - self._start_time
            silence_duration = now - self._last_speech_time

            # Max recording time exceeded
            if elapsed > self._vad_cfg.max_recording_sec:
                logger.info("Max recording time reached (%.1fs)", elapsed)
                self._speech_ended.set()
                self._is_active = False
                return

            # Silence timeout after speech was detected
            if self._speech_detected and silence_duration > self._vad_cfg.silence_timeout_sec:
                logger.info("Silence timeout after %.1fs of speech (silence=%.1fs)", elapsed, silence_duration)
                self._speech_ended.set()
                self._is_active = False


    # Block until speech ends (silence timeout or max time)
    def wait_for_speech_end(self, timeout: float | None = None) -> bool:
        return self._speech_ended.wait(timeout=timeout)


    # Whether any speech was detected during this monitoring session
    @property
    def had_speech(self) -> bool:
        with self._lock:
            return self._speech_detected


    # Duration of current/last monitoring session
    @property
    def duration(self) -> float:
        with self._lock:
            if self._start_time == 0:
                return 0.0

            return time.monotonic() - self._start_time
