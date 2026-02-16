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

# Fade duration in samples applied to the end of each sentence to prevent audio pops
_FADE_SAMPLES = 64


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


    # Consume sentences from a queue and speak them on a single audio stream (streaming mode)
    async def speak_streamed(self, sentence_queue: asyncio.Queue) -> None:
        self._is_speaking = True
        self._stop_event.clear()

        try:
            if self._voice:
                await self._speak_streamed_piper(sentence_queue)
            else:
                await self._speak_streamed_fallback(sentence_queue)
        except Exception:
            logger.exception("TTS streamed playback failed")
        finally:
            self._is_speaking = False


    # Streaming Piper: single audio stream, synthesize and play sentence by sentence
    async def _speak_streamed_piper(self, sentence_queue: asyncio.Queue) -> None:
        loop = asyncio.get_event_loop()
        stream_opened = False

        try:
            while True:
                if self._stop_event.is_set():
                    break

                sentence = await sentence_queue.get()

                if sentence is None:
                    break

                sentence = sentence.strip()
                if not sentence:
                    continue

                logger.info("TTS streaming sentence: %s", sentence[:80])

                # Synthesize in executor (blocking)
                audio, sample_rate = await loop.run_in_executor(None, self._synthesize_piper, sentence)

                # Open stream on first sentence (we now know the sample rate)
                if not stream_opened:
                    self._playback_stream = sd.OutputStream(
                        samplerate=sample_rate,
                        channels=1,
                        dtype="float32",
                    )
                    self._playback_stream.start()
                    stream_opened = True

                # Apply fade-out to prevent pop between sentences
                _apply_fade_out(audio)

                # Write chunks to the shared stream
                await loop.run_in_executor(
                    None, lambda a=audio, sr=sample_rate: self._write_chunks(a, sr),
                )

                if self._stop_event.is_set():
                    break

        finally:
            if stream_opened and self._playback_stream:
                self._playback_stream.stop()
                self._playback_stream.close()
                self._playback_stream = None


    # Streaming fallback: macOS 'say' sentence by sentence
    async def _speak_streamed_fallback(self, sentence_queue: asyncio.Queue) -> None:
        while True:
            if self._stop_event.is_set():
                break

            sentence = await sentence_queue.get()

            if sentence is None:
                break

            sentence = sentence.strip()
            if not sentence:
                continue

            logger.info("TTS streaming sentence: %s", sentence[:80])
            await self._speak_macos_fallback(sentence)

            if self._stop_event.is_set():
                break


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
        loop = asyncio.get_event_loop()
        audio, sample_rate = await loop.run_in_executor(None, self._synthesize_piper, text)

        _apply_fade_out(audio)
        await loop.run_in_executor(None, lambda: self._play_chunked(audio, sample_rate))


    # Synthesize text to numpy array (blocking, runs in executor)
    def _synthesize_piper(self, text: str) -> tuple[np.ndarray, int]:
        wav_buffer = io.BytesIO()
        with wave.open(wav_buffer, "wb") as wav_file:
            self._voice.synthesize_wav(text, wav_file)

        wav_buffer.seek(0)
        with wave.open(wav_buffer, "rb") as wav_file:
            sample_rate = wav_file.getframerate()
            audio_bytes = wav_file.readframes(wav_file.getnframes())

        audio = np.frombuffer(audio_bytes, dtype=np.int16).astype(np.float32) / 32768.0
        return audio, sample_rate


    # Write audio chunks to an already-open playback stream, checking stop_event
    def _write_chunks(self, audio: np.ndarray, sample_rate: int) -> None:
        chunk_size = sample_rate // 10  # 100ms chunks
        for i in range(0, len(audio), chunk_size):
            if self._stop_event.is_set():
                logger.debug("TTS chunk playback stopped at %.1fs", i / sample_rate)
                break
            chunk = audio[i : i + chunk_size]
            self._playback_stream.write(chunk.reshape(-1, 1))


    # Play audio chunk-by-chunk with its own stream (used by non-streaming speak)
    def _play_chunked(self, audio: np.ndarray, sample_rate: int) -> None:
        self._playback_stream = sd.OutputStream(
            samplerate=sample_rate,
            channels=1,
            dtype="float32",
        )
        self._playback_stream.start()

        try:
            self._write_chunks(audio, sample_rate)
        finally:
            self._playback_stream.stop()
            self._playback_stream.close()
            self._playback_stream = None


    # Fallback to macOS built-in 'say' command (interruptible)
    async def _speak_macos_fallback(self, text: str) -> None:
        self._fallback_proc = await asyncio.create_subprocess_exec("say", "-v", "Alice", "-r", "180", text)
        await self._fallback_proc.wait()
        self._fallback_proc = None


    # Check if is currently speaking
    @property
    def is_speaking(self) -> bool:
        return self._is_speaking


# Apply a short linear fade-out at the end of audio to prevent pops
def _apply_fade_out(audio: np.ndarray) -> None:
    if len(audio) < _FADE_SAMPLES:
        return
    fade = np.linspace(1.0, 0.0, _FADE_SAMPLES, dtype=np.float32)
    audio[-_FADE_SAMPLES:] *= fade
