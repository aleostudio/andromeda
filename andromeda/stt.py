# Copyright (c) 2026 Alessandro Orrù
# Licensed under MIT

import asyncio
import logging
import numpy as np
from andromeda.config import STTConfig

logger = logging.getLogger("[ STT ]")


# Local speech-to-text using faster-whisper (CTranslate2)
class SpeechRecognizer:

    def __init__(self, stt_cfg: STTConfig, speech_pad_ms: int = 300) -> None:
        self._cfg = stt_cfg
        self._speech_pad_ms = speech_pad_ms
        self._model = None


    # Load Whisper model. This downloads the model on first run
    def initialize(self) -> None:
        from faster_whisper import WhisperModel

        logger.info("Loading Whisper model: %s (device=%s, compute=%s)", self._cfg.model_size, self._cfg.device, self._cfg.compute_type)
        self._model = WhisperModel(self._cfg.model_size, device=self._cfg.device, compute_type=self._cfg.compute_type)
        logger.info("Whisper model loaded successfully")


    # Transcribe audio array to text (runs blocking Whisper in executor)
    async def transcribe(self, audio: np.ndarray) -> str:
        if self._model is None:
            raise RuntimeError("Model not initialized. Call initialize() first.")

        if len(audio) == 0:
            return ""

        if audio.dtype != np.float32:
            audio = audio.astype(np.float32)

        loop = asyncio.get_running_loop()

        return await loop.run_in_executor(None, self._transcribe_sync, audio)


    # Synchronous transcription (called from executor)
    def _transcribe_sync(self, audio: np.ndarray) -> str:
        try:
            segments, info = self._model.transcribe(
                audio,
                language=self._cfg.language,
                beam_size=self._cfg.beam_size,
                vad_filter=self._cfg.vad_filter,
                vad_parameters={"min_silence_duration_ms": 500, "speech_pad_ms": self._speech_pad_ms},
            )

            # Collect all segment texts
            texts = []
            for segment in segments:
                text = segment.text.strip()
                if text:
                    texts.append(text)
                    logger.debug("Segment [%.1fs -> %.1fs]: %s", segment.start, segment.end, text)

            result = " ".join(texts)

            if result:
                logger.info("Transcription (lang=%s, prob=%.2f): %s", info.language, info.language_probability, result)
            else:
                logger.info("No speech detected in audio")

            return result

        except Exception:
            logger.exception("Transcription failed")
            return ""
