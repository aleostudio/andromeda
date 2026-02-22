# Copyright (c) 2026 Alessandro Orrù
# Licensed under MIT

import asyncio
import collections
import io
import logging
import threading
import wave
import numpy as np
import sounddevice as sd
from andromeda.config import AudioConfig, TTSConfig

logger = logging.getLogger("[ TTS ]")

# Fade duration in samples applied to the end of each sentence to prevent audio pops
_FADE_SAMPLES = 64

# Max entries in the TTS audio cache
_TTS_CACHE_MAX = 64


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

        self._syn_config = None  # Piper SynthesisConfig, built on initialize()

        # TTS audio cache: maps normalized text -> (audio_array, sample_rate)
        self._cache: dict[str, tuple[np.ndarray, int]] = {}
        self._cache_order: collections.deque[str] = collections.deque()  # LRU tracking


    # Load Piper voice model
    def initialize(self) -> None:
        try:
            from piper import PiperVoice
            from piper.config import SynthesisConfig

            self._voice = PiperVoice.load(self._tts_cfg.model_path, config_path=self._tts_cfg.model_config)
            self._syn_config = SynthesisConfig(
                speaker_id=self._tts_cfg.speaker_id or None,
                length_scale=self._tts_cfg.length_scale or None,
            )
            logger.info("Piper TTS loaded: %s (length_scale=%.2f)", self._tts_cfg.model_path, self._tts_cfg.length_scale)

        except FileNotFoundError:
            logger.warning("Piper model not found at %s. Falling back to macOS 'say' command.", self._tts_cfg.model_path)
            self._voice = None
        except Exception:
            logger.exception("Failed to load Piper TTS")
            self._voice = None


    # Pre-warm the TTS cache with commonly used phrases
    def prewarm_cache(self, phrases: list[str] | None = None) -> None:
        if not self._voice:
            return

        default_phrases = [
            "Non ho sentito nulla. Riprova.",
            "Non ho capito. Puoi ripetere?",
            "Si è verificato un errore. Riprova.",
        ]
        for text in (phrases or default_phrases):
            key = self._cache_key(text)
            if key not in self._cache:
                try:
                    audio, sr = self._synthesize_piper(text)
                    self._cache_put(key, audio, sr)
                    logger.debug("TTS cache pre-warmed: %s", text[:40])
                except Exception:
                    logger.warning("Failed to pre-warm TTS cache for: %s", text[:40])

        logger.info("TTS cache pre-warmed with %d phrases", len(self._cache))


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
    # Uses prefetch: synthesizes next sentence while current one is playing
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


    # Streaming Piper with prefetch: synthesize next sentence while playing current
    async def _speak_streamed_piper(self, sentence_queue: asyncio.Queue) -> None:
        loop = asyncio.get_running_loop()
        stream_opened = False
        prefetch_task: asyncio.Task | None = None
        prefetch_result: tuple[str, np.ndarray, int] | None = None

        try:
            while not self._stop_event.is_set():
                # If we have a prefetched sentence, use it directly instead of reading from queue
                if prefetch_result is not None:
                    sentence, audio, sample_rate = prefetch_result
                    prefetch_result = None
                else:
                    sentence = await self._next_sentence(sentence_queue)
                    if sentence is None:
                        break
                    audio, sample_rate = await self._synthesize_cached(loop, sentence)

                logger.info("TTS streaming sentence: %s", " ".join(sentence[:200].split()))

                # Start prefetching the next sentence immediately
                prefetch_task = asyncio.create_task(
                    self._prefetch_next(sentence_queue, loop),
                )

                # Open stream on first sentence (we now know the sample rate)
                if not stream_opened:
                    stream_opened = self._open_playback_stream(sample_rate)
                    if not stream_opened:
                        break

                # Apply fade-out and write chunks to the shared stream
                _apply_fade_out(audio)
                await loop.run_in_executor(None, lambda a=audio, sr=sample_rate: self._write_chunks(a, sr))

                # Inter-sentence silence: gives natural pacing and creates a clean
                # audio window for wake word detection (mic hears only the user)
                if self._tts_cfg.sentence_silence > 0 and not self._stop_event.is_set():
                    silence_len = int(sample_rate * self._tts_cfg.sentence_silence)
                    silence = np.zeros(silence_len, dtype=np.float32)
                    await loop.run_in_executor(None, lambda s=silence, sr=sample_rate: self._write_chunks(s, sr))

                # Collect prefetched result for next iteration
                prefetch_result = await self._collect_prefetch(prefetch_task)
                prefetch_task = None

        finally:
            self._cleanup_stream(prefetch_task, stream_opened)


    # Consume next valid sentence from the queue; returns None on stop/end-of-stream
    async def _next_sentence(self, queue: asyncio.Queue) -> str | None:
        while not self._stop_event.is_set():
            try:
                sentence = await asyncio.wait_for(queue.get(), timeout=0.2)
            except asyncio.TimeoutError:
                continue

            if sentence is None:
                return None

            stripped = sentence.strip()
            if stripped:
                return stripped

        return None


    # Open the shared playback stream; returns True on success
    def _open_playback_stream(self, sample_rate: int) -> bool:
        try:
            self._playback_stream = sd.OutputStream(
                samplerate=sample_rate,
                channels=1,
                dtype="float32",
            )
            self._playback_stream.start()
            return True
        except Exception:
            logger.exception("Failed to open streaming audio output")
            return False


    # Await a prefetch task and return its result (or None on failure)
    @staticmethod
    async def _collect_prefetch(task: asyncio.Task | None) -> tuple[str, np.ndarray, int] | None:
        if task is None:
            return None
        try:
            return await task
        except Exception:
            logger.warning("Prefetch task failed")
            return None


    # Cancel pending prefetch and close the playback stream
    def _cleanup_stream(self, prefetch_task: asyncio.Task | None, stream_opened: bool) -> None:
        if prefetch_task is not None:
            prefetch_task.cancel()
        if stream_opened and self._playback_stream:
            try:
                self._playback_stream.stop()
                self._playback_stream.close()
            except Exception:
                logger.warning("Error closing streaming audio output")
            self._playback_stream = None


    # Prefetch: consume the next sentence from the queue and synthesize it ahead of time
    # Returns (sentence, audio, sample_rate) so the main loop can use both directly
    async def _prefetch_next(self, sentence_queue: asyncio.Queue, loop: asyncio.AbstractEventLoop) -> tuple[str, np.ndarray, int] | None:
        try:
            # Non-blocking — if queue is empty, just return None
            sentence = sentence_queue.get_nowait()
        except asyncio.QueueEmpty:
            return None

        if sentence is None:
            # Put the sentinel back so the main loop sees it
            await sentence_queue.put(None)
            return None

        sentence = sentence.strip()
        if not sentence:
            return None

        audio, sr = await self._synthesize_cached(loop, sentence)
        return sentence, audio, sr


    # Synthesize with cache lookup
    async def _synthesize_cached(self, loop: asyncio.AbstractEventLoop, text: str) -> tuple[np.ndarray, int]:
        key = self._cache_key(text)

        # Check cache first
        cached = self._cache.get(key)
        if cached is not None:
            logger.debug("TTS cache hit: %s", text[:40])
            return cached[0].copy(), cached[1]

        # Synthesize and cache
        audio, sr = await loop.run_in_executor(None, self._synthesize_piper, text)
        self._cache_put(key, audio, sr)

        return audio, sr


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

            logger.info("TTS streaming sentence: %s", " ".join(sentence[:200].split()))
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
        loop = asyncio.get_running_loop()
        audio, sample_rate = await self._synthesize_cached(loop, text)
        _apply_fade_out(audio)
        await loop.run_in_executor(None, lambda: self._play_chunked(audio, sample_rate))


    # Synthesize text to numpy array (blocking, runs in executor)
    def _synthesize_piper(self, text: str) -> tuple[np.ndarray, int]:
        try:
            wav_buffer = io.BytesIO()
            with wave.open(wav_buffer, "wb") as wav_file:
                self._voice.synthesize_wav(text, wav_file, syn_config=self._syn_config)

            wav_buffer.seek(0)
            with wave.open(wav_buffer, "rb") as wav_file:
                sample_rate = wav_file.getframerate()
                audio_bytes = wav_file.readframes(wav_file.getnframes())

            audio = np.frombuffer(audio_bytes, dtype=np.int16).astype(np.float32) / 32768.0

            return audio, sample_rate

        except Exception:
            logger.exception("Piper synthesis failed for: %s", text[:40])
            # Return silence so caller doesn't crash
            sr = self._audio_cfg.sample_rate

            return np.zeros(sr // 2, dtype=np.float32), sr


    # Write audio chunks to an already-open playback stream, checking stop_event
    def _write_chunks(self, audio: np.ndarray, sample_rate: int) -> None:
        chunk_size = sample_rate // 10  # 100ms chunks
        for i in range(0, len(audio), chunk_size):
            if self._stop_event.is_set():
                logger.debug("TTS chunk playback stopped at %.1fs", i / sample_rate)
                break
            try:
                chunk = audio[i : i + chunk_size]
                self._playback_stream.write(chunk.reshape(-1, 1))
            except Exception:
                if not self._stop_event.is_set():
                    logger.warning("Audio stream write error at chunk %d", i)
                break


    # Play audio chunk-by-chunk with its own stream (used by non-streaming speak)
    def _play_chunked(self, audio: np.ndarray, sample_rate: int) -> None:
        try:
            self._playback_stream = sd.OutputStream(samplerate=sample_rate, channels=1, dtype="float32")
            self._playback_stream.start()
        except Exception:
            logger.exception("Failed to open audio output stream")
            self._playback_stream = None
            return

        try:
            self._write_chunks(audio, sample_rate)
        finally:
            try:
                self._playback_stream.stop()
                self._playback_stream.close()
            except Exception:
                logger.warning("Error closing audio output stream")

            self._playback_stream = None


    # Fallback to macOS built-in 'say' command (interruptible)
    async def _speak_macos_fallback(self, text: str) -> None:
        try:
            self._fallback_proc = await asyncio.create_subprocess_exec("say", "-v", "Alice", "-r", "180", text)
            await self._fallback_proc.wait()
        except Exception:
            logger.warning("macOS 'say' fallback failed")
        finally:
            self._fallback_proc = None


    # Check if is currently speaking
    @property
    def is_speaking(self) -> bool:
        return self._is_speaking


    # Cache helpers
    @staticmethod
    def _cache_key(text: str) -> str:
        return text.strip().lower()


    def _cache_put(self, key: str, audio: np.ndarray, sample_rate: int) -> None:
        if key in self._cache:
            return

        if len(self._cache) >= _TTS_CACHE_MAX:
            oldest = self._cache_order.popleft()
            self._cache.pop(oldest, None)
        self._cache[key] = (audio.copy(), sample_rate)
        self._cache_order.append(key)


# Apply a short linear fade-out at the end of audio to prevent pops
def _apply_fade_out(audio: np.ndarray) -> None:
    if len(audio) < _FADE_SAMPLES:
        return

    fade = np.linspace(1.0, 0.0, _FADE_SAMPLES, dtype=np.float32)
    audio[-_FADE_SAMPLES:] *= fade
