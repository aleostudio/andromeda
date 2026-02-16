# Copyright (c) 2026 Alessandro Orrù
# Licensed under MIT

from andromeda.config import AudioConfig, TTSConfig
import asyncio
import io
import logging
import threading
import wave
import numpy as np
import sounddevice as sd

logger = logging.getLogger(__name__)


# Local text-to-speech using Piper
# Models: https://github.com/rhasspy/piper/blob/master/VOICES.md
class TextToSpeech:

    def __init__(self, audio_cfg: AudioConfig, tts_cfg: TTSConfig) -> None:
        self._audio_cfg = audio_cfg
        self._tts_cfg = tts_cfg
        self._voice = None
        self._is_speaking = False
        self._stop_event = threading.Event()
        self._playback_stream: sd.OutputStream | None = None
        self._fallback_proc: asyncio.subprocess.Process | None = None


    # Load Piper voice model
    def initialize(self) -> None:
        try:
            from piper import PiperVoice

            self._voice = PiperVoice.load(self._tts_cfg.model_path, config_path=self._tts_cfg.model_config)
            logger.info("Piper TTS loaded: %s", self._tts_cfg.model_path)

        except FileNotFoundError:
            logger.warning("Piper model not found at %s. Falling back to macOS 'say' command.", self._tts_cfg.model_path)
            self._voice = None
        except Exception:
            logger.exception("Failed to load Piper TTS")
            self._voice = None


    # Synthesize and play text (interruptible)
    async def speak(self, text: str) -> None:
        if not text.strip():
            return

        self._is_speaking = True
        self._stop_event.clear()
        logger.info("TTS speaking: %s", text[:80])

        try:
            if self._voice:
                await self._speak_piper(text)
            else:
                await self._speak_macos_fallback(text)
        except Exception:
            logger.exception("TTS playback failed")
        finally:
            self._is_speaking = False


    # Interrupt current TTS playback immediately
    def stop_playback(self) -> None:
        logger.info("TTS playback interrupted")
        self._stop_event.set()

        # Stop Piper audio stream
        if self._playback_stream and self._playback_stream.active:
            try:
                self._playback_stream.stop()
            except Exception:
                pass

        # Terminate macOS 'say' subprocess
        if self._fallback_proc and self._fallback_proc.returncode is None:
            try:
                self._fallback_proc.terminate()
            except Exception:
                pass


    # Synthesize with Piper and play through sounddevice (chunk-by-chunk, interruptible)
    async def _speak_piper(self, text: str) -> None:
        # Synthesize to WAV in memory
        wav_buffer = io.BytesIO()
        with wave.open(wav_buffer, "wb") as wav_file:
            self._voice.synthesize_wav(text, wav_file)

        # Read WAV data
        wav_buffer.seek(0)
        with wave.open(wav_buffer, "rb") as wav_file:
            sample_rate = wav_file.getframerate()
            audio_bytes = wav_file.readframes(wav_file.getnframes())

        audio = np.frombuffer(audio_bytes, dtype=np.int16).astype(np.float32) / 32768.0

        # Play audio in chunks (interruptible via stop_event)
        loop = asyncio.get_event_loop()
        await loop.run_in_executor(None, lambda: self._play_chunked(audio, sample_rate))


    # Play audio chunk-by-chunk, checking stop_event between chunks
    def _play_chunked(self, audio: np.ndarray, sample_rate: int) -> None:
        chunk_size = sample_rate // 10  # 100ms chunks

        self._playback_stream = sd.OutputStream(
            samplerate=sample_rate,
            channels=1,
            dtype="float32",
        )
        self._playback_stream.start()

        try:
            for i in range(0, len(audio), chunk_size):
                if self._stop_event.is_set():
                    logger.debug("TTS chunk playback stopped at %.1fs", i / sample_rate)
                    break
                chunk = audio[i : i + chunk_size]
                self._playback_stream.write(chunk.reshape(-1, 1))
        finally:
            self._playback_stream.stop()
            self._playback_stream.close()
            self._playback_stream = None


    # Fallback to macOS built-in 'say' command (interruptible)
    async def _speak_macos_fallback(self, text: str) -> None:
        # Use Italian voice if available
        self._fallback_proc = await asyncio.create_subprocess_exec("say", "-v", "Alice", "-r", "180", text)
        await self._fallback_proc.wait()
        self._fallback_proc = None


    # Check if is currently speaking
    @property
    def is_speaking(self) -> bool:
        return self._is_speaking
