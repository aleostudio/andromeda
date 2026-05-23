# Copyright (c) 2026 Alessandro Orrù
# Licensed under MIT

from types import SimpleNamespace
import numpy as np
from andromeda.config import STTConfig
from andromeda.stt import SpeechRecognizer


class TestSTTQualityFilters:
    def test_too_quiet_audio_is_rejected(self):
        recognizer = SpeechRecognizer(STTConfig(min_audio_rms=0.01))
        audio = np.zeros(16000, dtype=np.float32)

        assert recognizer._is_too_quiet(audio) is True

    def test_loud_audio_is_accepted(self):
        recognizer = SpeechRecognizer(STTConfig(min_audio_rms=0.01))
        audio = np.full(16000, 0.02, dtype=np.float32)

        assert recognizer._is_too_quiet(audio) is False

    def test_known_subtitle_hallucination_is_rejected(self):
        assert SpeechRecognizer._is_hallucinated_text(
            "Sottotitoli e revisione a cura di QTSS",
        ) is True

    def test_normal_text_is_not_hallucination(self):
        assert SpeechRecognizer._is_hallucinated_text("Che tempo fa oggi?") is False

    def test_low_confidence_segment_is_rejected(self):
        recognizer = SpeechRecognizer(
            STTConfig(max_no_speech_prob=0.6, min_avg_logprob=-1.0),
        )
        segment = SimpleNamespace(no_speech_prob=0.8, avg_logprob=-0.2)

        assert recognizer._is_low_confidence_segment(segment) is True

    def test_high_confidence_segment_is_accepted(self):
        recognizer = SpeechRecognizer(
            STTConfig(max_no_speech_prob=0.6, min_avg_logprob=-1.0),
        )
        segment = SimpleNamespace(no_speech_prob=0.1, avg_logprob=-0.2)

        assert recognizer._is_low_confidence_segment(segment) is False
