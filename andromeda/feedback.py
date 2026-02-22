# Copyright (c) 2026 Alessandro Orrù
# Licensed under MIT

import logging
import threading
import numpy as np
import sounddevice as sd
from collections.abc import Callable
from pathlib import Path
from andromeda.config import AudioConfig, FeedbackConfig

logger = logging.getLogger("[ FEEDBACK ]")

# Fade-out duration (ms) when thinking tone is stopped
_THINKING_FADE_MS = 200


# Play short audio cues for state transitions
# Use synthesized tones as fallback if WAV files are not found
class AudioFeedback:

    def __init__(self, audio_cfg: AudioConfig, feedback_cfg: FeedbackConfig) -> None:
        self._audio_cfg = audio_cfg
        self._cfg = feedback_cfg
        self._sounds: dict[str, np.ndarray] = {}
        self._thinking_stop = threading.Event()
        self._thinking_thread: threading.Thread | None = None


    # Load or generate feedback sounds
    def initialize(self) -> None:
        self._sounds["wake"] = self._load_or_generate(self._cfg.wake_sound, self._gen_wake_tone)
        self._sounds["done"] = self._load_or_generate(self._cfg.done_sound, self._gen_done_tone)
        self._sounds["error"] = self._load_or_generate(self._cfg.error_sound, self._gen_error_tone)
        self._sounds["thinking"] = self._load_or_generate(self._cfg.thinking_sound, self._gen_thinking_tone)
        logger.info("Audio feedback initialized")


    # Play a named feedback sound (non-blocking)
    def play(self, sound_name: str) -> None:
        if sound_name == "thinking":
            self._start_thinking()
            return

        audio = self._sounds.get(sound_name)
        if audio is not None:
            try:
                sd.play(audio, samplerate=self._audio_cfg.sample_rate)
            except Exception:
                logger.warning("Failed to play feedback: %s", sound_name)


    # Signal the thinking tone to fade out smoothly (non-blocking)
    # The daemon thread handles the 200ms fade-out and stream cleanup
    def stop(self) -> None:
        self._thinking_stop.set()


    # Play a named feedback sound (blocking)
    def play_blocking(self, sound_name: str) -> None:
        audio = self._sounds.get(sound_name)
        if audio is not None:
            try:
                sd.play(audio, samplerate=self._audio_cfg.sample_rate, blocking=True)
            except Exception:
                logger.warning("Failed to play feedback: %s", sound_name)


    # Start thinking tone on a dedicated OutputStream (allows smooth fade-out)
    def _start_thinking(self) -> None:
        # Stop any previous thinking tone
        self._thinking_stop.set()
        if self._thinking_thread and self._thinking_thread.is_alive():
            self._thinking_thread.join(timeout=0.5)

        audio = self._sounds.get("thinking")
        if audio is None:
            return

        self._thinking_stop.clear()
        self._thinking_thread = threading.Thread(target=self._run_thinking, args=(audio.copy(),), daemon=True)
        self._thinking_thread.start()


    # Background thread: play thinking tone chunk-by-chunk, fade out on stop signal
    def _run_thinking(self, audio: np.ndarray) -> None:
        sr = self._audio_cfg.sample_rate
        chunk_size = sr // 10  # 100ms chunks
        fade_samples = int(sr * _THINKING_FADE_MS / 1000)

        try:
            stream = sd.OutputStream(samplerate=sr, channels=1, dtype="float32")
            stream.start()
        except Exception:
            logger.debug("Failed to open thinking tone stream", exc_info=True)
            return

        try:
            i = 0
            while i < len(audio):
                if self._thinking_stop.is_set():
                    # Fade out smoothly instead of cutting abruptly
                    remaining = audio[i : i + fade_samples].copy()
                    if len(remaining) > 0:
                        fade = np.linspace(1.0, 0.0, len(remaining), dtype=np.float32)
                        remaining *= fade
                        stream.write(remaining.reshape(-1, 1))
                    break

                end = min(i + chunk_size, len(audio))
                stream.write(audio[i:end].reshape(-1, 1))
                i = end
        except Exception:
            logger.debug("Thinking tone playback error", exc_info=True)
        finally:
            try:
                stream.stop()
                stream.close()
            except Exception:
                pass


    # Load WAV file or generate synthetic tone
    def _load_or_generate(self, wav_path: str, generator: Callable) -> np.ndarray:
        path = Path(wav_path)
        if path.exists():
            try:
                import wave

                with wave.open(str(path), "rb") as wf:
                    audio_bytes = wf.readframes(wf.getnframes())
                    return np.frombuffer(audio_bytes, dtype=np.int16).astype(np.float32) / 32768.0
            except Exception:
                logger.warning("Failed to load %s, using synthetic", path)

        return generator()


    # Rising two-tone chime (wake word detected)
    def _gen_wake_tone(self) -> np.ndarray:
        sr = self._audio_cfg.sample_rate
        t1 = np.linspace(0, 0.1, int(sr * 0.1), dtype=np.float32)
        t2 = np.linspace(0, 0.1, int(sr * 0.1), dtype=np.float32)
        tone1 = 0.3 * np.sin(2 * np.pi * 800 * t1)
        tone2 = 0.3 * np.sin(2 * np.pi * 1200 * t2)
        # Apply fade
        fade = np.linspace(1.0, 0.0, len(t2), dtype=np.float32)
        tone2 *= fade
        silence = np.zeros(int(sr * 0.03), dtype=np.float32)

        return np.concatenate([tone1, silence, tone2])


    # Single soft tone (processing done)
    def _gen_done_tone(self) -> np.ndarray:
        sr = self._audio_cfg.sample_rate
        t = np.linspace(0, 0.15, int(sr * 0.15), dtype=np.float32)
        tone = 0.2 * np.sin(2 * np.pi * 600 * t)
        fade = np.linspace(1.0, 0.0, len(t), dtype=np.float32)

        return tone * fade


    # Descending two-tone (error)
    def _gen_error_tone(self) -> np.ndarray:
        sr = self._audio_cfg.sample_rate
        t1 = np.linspace(0, 0.15, int(sr * 0.15), dtype=np.float32)
        t2 = np.linspace(0, 0.15, int(sr * 0.15), dtype=np.float32)
        tone1 = 0.3 * np.sin(2 * np.pi * 500 * t1)
        tone2 = 0.3 * np.sin(2 * np.pi * 300 * t2)
        fade = np.linspace(1.0, 0.0, len(t2), dtype=np.float32)
        tone2 *= fade
        silence = np.zeros(int(sr * 0.05), dtype=np.float32)

        return np.concatenate([tone1, silence, tone2])


    # Warm ambient pad with gentle pulse (thinking/processing indicator)
    # Open fifth chord (C4 + G4 + C5) with slow breathing modulation
    # Volume controlled by FeedbackConfig.thinking_volume (0.0-1.0)
    def _gen_thinking_tone(self) -> np.ndarray:
        sr = self._audio_cfg.sample_rate
        vol = self._cfg.thinking_volume
        duration = 15.0
        t = np.linspace(0, duration, int(sr * duration), dtype=np.float32)

        # Harmonically rich pad: root + fifth + octave
        tone = (
            0.12 * np.sin(2 * np.pi * 261.6 * t)     # C4  (root)
            + 0.08 * np.sin(2 * np.pi * 392.0 * t)    # G4  (perfect fifth)
            + 0.05 * np.sin(2 * np.pi * 523.3 * t)    # C5  (octave)
        )

        # Apply volume (0.0-1.0)
        tone *= vol

        # Gentle breathing pulse at ~1.5 Hz
        pulse = 0.65 + 0.35 * np.sin(2 * np.pi * 1.5 * t)
        tone *= pulse

        # Smooth fade-in (1s)
        fade_in = int(sr * 1.0)
        tone[:fade_in] *= np.linspace(0, 1, fade_in, dtype=np.float32)

        # Natural fade-out (2s) at the end in case LLM takes very long
        fade_out = int(sr * 2.0)
        tone[-fade_out:] *= np.linspace(1, 0, fade_out, dtype=np.float32)

        return tone
