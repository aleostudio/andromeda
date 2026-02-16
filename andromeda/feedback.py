from pathlib import Path
from andromeda.config import AudioConfig, FeedbackConfig
import logging
import numpy as np
import sounddevice as sd

logger = logging.getLogger(__name__)


# Play short audio cues for state transitions
# Use synthesized tones as fallback if WAV files are not found
class AudioFeedback:

    def __init__(self, audio_cfg: AudioConfig, feedback_cfg: FeedbackConfig) -> None:
        self._audio_cfg = audio_cfg
        self._cfg = feedback_cfg
        self._sounds: dict[str, np.ndarray] = {}


    # Load or generate feedback sounds
    def initialize(self) -> None:
        self._sounds["wake"] = self._load_or_generate(self._cfg.wake_sound, self._gen_wake_tone)
        self._sounds["done"] = self._load_or_generate(self._cfg.done_sound, self._gen_done_tone)
        self._sounds["error"] = self._load_or_generate(self._cfg.error_sound, self._gen_error_tone)
        logger.info("Audio feedback initialized")


    # Play a named feedback sound (non-blocking)
    def play(self, sound_name: str) -> None:
        audio = self._sounds.get(sound_name)
        if audio is not None:
            try:
                sd.play(audio, samplerate=self._audio_cfg.sample_rate)
            except Exception:
                logger.warning("Failed to play feedback: %s", sound_name)


    # Play a named feedback sound (blocking)
    def play_blocking(self, sound_name: str) -> None:
        audio = self._sounds.get(sound_name)
        if audio is not None:
            try:
                sd.play(audio, samplerate=self._audio_cfg.sample_rate, blocking=True)
            except Exception:
                logger.warning("Failed to play feedback: %s", sound_name)


    # Load WAV file or generate synthetic tone
    def _load_or_generate(self, wav_path: str, generator: callable) -> np.ndarray:
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
