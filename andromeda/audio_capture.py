# Copyright (c) 2026 Alessandro Orrù
# Licensed under MIT

from enum import Enum, auto
from typing import Callable
from andromeda.config import AudioConfig, NoiseConfig
import collections
import logging
import threading
import numpy as np
import sounddevice as sd

logger = logging.getLogger(__name__)


class AudioRouteMode(Enum):
    NORMAL = auto()     # frames go to all listeners (wake word + VAD)
    MUTED = auto()      # frames dropped entirely
    STOP_ONLY = auto()  # frames go only to stop-word listener


# Continuous audio capture from microphone with ring buffer
class AudioCapture:

    def __init__(self, audio_cfg: AudioConfig, noise_cfg: NoiseConfig, ring_buffer_sec: float = 3.0) -> None:
        self._cfg = audio_cfg
        self._noise_cfg = noise_cfg
        self._stream: sd.InputStream | None = None

        # Ring buffer for wake word (fixed size, last N seconds)
        ring_size = int(ring_buffer_sec * audio_cfg.sample_rate / audio_cfg.chunk_samples)
        self._ring_buffer: collections.deque[bytes] = collections.deque(maxlen=ring_size)

        # Recording buffer (variable size, grows during LISTENING)
        self._recording_buffer: list[bytes] = []
        self._is_recording = False

        # Callbacks
        self._on_audio_frame: list[Callable[[bytes], None]] = []
        self._on_stop_word_frame: list[Callable[[bytes], None]] = []

        self._lock = threading.Lock()
        self._route_mode = AudioRouteMode.NORMAL


    # Register callback for each audio frame (for wake word / VAD)
    def on_audio_frame(self, callback: Callable[[bytes], None]) -> None:
        self._on_audio_frame.append(callback)


    # Register callback for stop-word frames (active only in STOP_ONLY mode)
    def on_stop_word_frame(self, callback: Callable[[bytes], None]) -> None:
        self._on_stop_word_frame.append(callback)


    # Start audio capture stream
    def start(self) -> None:
        self._stream = sd.InputStream(
            samplerate=self._cfg.sample_rate,
            channels=self._cfg.channels,
            dtype=self._cfg.dtype,
            blocksize=self._cfg.chunk_samples,
            callback=self._audio_callback,
        )
        self._stream.start()
        logger.info("Audio capture started: %dHz, chunk=%dms", self._cfg.sample_rate, self._cfg.chunk_ms)


    # Stop audio capture
    def stop(self) -> None:
        if self._stream:
            self._stream.stop()
            self._stream.close()
            self._stream = None
            logger.info("Audio capture stopped")


    # Mute input (during TTS playback to avoid echo)
    def mute(self) -> None:
        self._route_mode = AudioRouteMode.MUTED


    # Unmute input (normal mode)
    def unmute(self) -> None:
        self._route_mode = AudioRouteMode.NORMAL


    # Route audio only to stop-word listeners
    def set_route_mode_stop_only(self) -> None:
        self._route_mode = AudioRouteMode.STOP_ONLY


    # Start collecting audio into recording buffer
    def start_recording(self) -> None:
        with self._lock:
            self._recording_buffer.clear()
            self._is_recording = True
        logger.debug("Recording started")


    # Stop recording and return collected audio as numpy array
    def stop_recording(self) -> np.ndarray:
        with self._lock:
            self._is_recording = False
            if not self._recording_buffer:
                return np.array([], dtype=np.int16)
            audio_bytes = b"".join(self._recording_buffer)
            self._recording_buffer.clear()

        audio = np.frombuffer(audio_bytes, dtype=np.int16).astype(np.float32) / 32768.0
        logger.debug("Recording stopped: %.2f sec", len(audio) / self._cfg.sample_rate)

        if self._noise_cfg.enabled:
            audio = self._reduce_noise(audio)

        return audio


    # Get current ring buffer contents as bytes
    def get_ring_buffer_audio(self) -> bytes:
        with self._lock:
            return b"".join(self._ring_buffer)


    # Analyze ring buffer to estimate current speech RMS energy
    def calibrate_speech_energy(self, vad, sample_rate: int) -> float:
        with self._lock:
            frames = list(self._ring_buffer)

        if not frames:
            return 0.0

        speech_energies = []
        for frame_bytes in frames:
            try:
                if vad.is_speech(frame_bytes, sample_rate=sample_rate):
                    audio = np.frombuffer(frame_bytes, dtype=np.int16)
                    rms = float(np.sqrt(np.mean(audio.astype(np.float32) ** 2)))
                    speech_energies.append(rms)
            except Exception:
                continue

        if not speech_energies:
            return 0.0

        median_energy = float(np.median(speech_energies))
        logger.debug("Calibration: %d speech frames, median RMS=%.1f", len(speech_energies), median_energy)

        return median_energy


    # Sounddevice callback - runs in separate thread
    def _audio_callback(self, indata: np.ndarray, frames: int, time_info: object, status: sd.CallbackFlags) -> None:
        if status:
            logger.warning("Audio status: %s", status)

        mode = self._route_mode

        if mode == AudioRouteMode.MUTED:
            return

        frame_bytes = indata.tobytes()

        with self._lock:
            self._ring_buffer.append(frame_bytes)
            if self._is_recording:
                self._recording_buffer.append(frame_bytes)

        if mode == AudioRouteMode.STOP_ONLY:
            # Only dispatch to stop-word listeners
            for cb in self._on_stop_word_frame:
                try:
                    cb(frame_bytes)
                except Exception:
                    logger.exception("Stop-word frame callback error")
            return

        # Dispatch to all listeners (wake word detector, VAD)
        for cb in self._on_audio_frame:
            try:
                cb(frame_bytes)
            except Exception:
                logger.exception("Audio frame callback error")


    # Apply noise reduction to recorded audio
    def _reduce_noise(self, audio: np.ndarray) -> np.ndarray:
        try:
            import noisereduce as nr

            return nr.reduce_noise(
                y=audio,
                sr=self._cfg.sample_rate,
                stationary=self._noise_cfg.stationary,
                prop_decrease=self._noise_cfg.prop_decrease,
            )

        except Exception:
            logger.warning("Noise reduction failed, using raw audio")
            return audio
