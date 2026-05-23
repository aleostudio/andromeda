# Copyright (c) 2026 Alessandro Orrù
# Licensed under MIT

import asyncio
import logging
import re
import numpy as np
from andromeda.config import STTConfig

logger = logging.getLogger("[ STT ]")

_HALLUCINATION_PATTERNS = (
    re.compile(r"\bsottotitoli\b.*\brevisione\b", re.IGNORECASE),
    re.compile(r"\ba\s+cura\s+di\b", re.IGNORECASE),
    re.compile(r"\bsubtitles?\s+by\b", re.IGNORECASE),
)


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

        if self._is_too_quiet(audio):
            return ""

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
                    if self._is_low_confidence_segment(segment):
                        logger.info(
                            "Discarded low-confidence STT segment: text=%s no_speech=%.2f avg_logprob=%.2f",
                            text,
                            getattr(segment, "no_speech_prob", 0.0),
                            getattr(segment, "avg_logprob", 0.0),
                        )
                        continue
                    texts.append(text)
                    logger.debug("Segment [%.1fs -> %.1fs]: %s", segment.start, segment.end, text)

            result = " ".join(texts)
            if self._is_hallucinated_text(result):
                logger.info("Discarded known STT hallucination: %s", result)
                return ""

            if result:
                logger.info("Transcription (lang=%s, prob=%.2f): %s", info.language, info.language_probability, result)
            else:
                logger.info("No speech detected in audio")

            return result

        except Exception:
            logger.exception("Transcription failed")
            return ""


    def _is_too_quiet(self, audio: np.ndarray) -> bool:
        rms = float(np.sqrt(np.mean(audio ** 2))) if len(audio) else 0.0
        if rms < self._cfg.min_audio_rms:
            logger.info("Audio too quiet for STT: rms=%.5f threshold=%.5f", rms, self._cfg.min_audio_rms)
            return True

        return False


    def _is_low_confidence_segment(self, segment) -> bool:
        no_speech_prob = float(getattr(segment, "no_speech_prob", 0.0) or 0.0)
        avg_logprob = float(getattr(segment, "avg_logprob", 0.0) or 0.0)

        return (
            no_speech_prob > self._cfg.max_no_speech_prob
            or avg_logprob < self._cfg.min_avg_logprob
        )


    @staticmethod
    def _is_hallucinated_text(text: str) -> bool:
        if not text.strip():
            return False

        return any(pattern.search(text) for pattern in _HALLUCINATION_PATTERNS)
