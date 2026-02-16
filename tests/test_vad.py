# Copyright (c) 2026 Alessandro Orrù
# Licensed under MIT

import struct
import time
from unittest.mock import patch
import numpy as np
import pytest
from andromeda.config import AudioConfig, VADConfig
from andromeda.vad import VoiceActivityDetector


# Create a fake audio frame of int16 samples
def _make_frame(value: int = 0, num_samples: int = 480) -> bytes:
    return struct.pack(f"<{num_samples}h", *([value] * num_samples))

# Create a loud audio frame (high energy)
def _make_loud_frame(amplitude: int = 5000, num_samples: int = 480) -> bytes:
    return struct.pack(f"<{num_samples}h", *([amplitude] * num_samples))


class TestVADInit:
    def test_initial_state(self):
        vad = VoiceActivityDetector(AudioConfig(), VADConfig())
        assert vad.had_speech is False
        assert vad.duration == 0


class TestEnergyThreshold:
    def test_set_threshold(self):
        vad = VoiceActivityDetector(AudioConfig(), VADConfig())
        vad.set_energy_threshold(100.0)
        assert vad._energy_threshold == 100

    def test_reset_threshold(self):
        vad = VoiceActivityDetector(AudioConfig(), VADConfig())
        vad.set_energy_threshold(100.0)
        vad.set_energy_threshold(0.0)
        assert vad._energy_threshold == 0


class TestVADStartStop:
    def test_start_resets_state(self):
        vad = VoiceActivityDetector(AudioConfig(), VADConfig())
        vad.start()
        assert vad.had_speech is False
        assert vad.duration >= 0.0

    def test_stop_sets_event(self):
        vad = VoiceActivityDetector(AudioConfig(), VADConfig())
        vad.start()
        vad.stop()
        assert vad._speech_ended.is_set()

    def test_stop_deactivates(self):
        vad = VoiceActivityDetector(AudioConfig(), VADConfig())
        vad.start()
        vad.stop()
        assert not vad._is_active


class TestProcessFrame:
    def test_inactive_ignores_frame(self):
        vad = VoiceActivityDetector(AudioConfig(), VADConfig())
        # Not started, should do nothing
        frame = _make_frame(0, 480)
        vad.process_frame(frame)
        assert vad.had_speech is False

    def test_max_recording_timeout(self):
        cfg = VADConfig(max_recording_sec=0.01)
        vad = VoiceActivityDetector(AudioConfig(), cfg)
        vad.start()
        # Wait a tiny bit to exceed max
        time.sleep(0.02)
        frame = _make_frame(0, 480)
        vad.process_frame(frame)
        assert vad._speech_ended.is_set()


class TestWaitForSpeechEnd:
    def test_returns_true_when_set(self):
        vad = VoiceActivityDetector(AudioConfig(), VADConfig())
        vad.start()
        vad._speech_ended.set()
        assert vad.wait_for_speech_end(timeout=0.1) is True

    def test_timeout(self):
        vad = VoiceActivityDetector(AudioConfig(), VADConfig())
        vad.start()
        result = vad.wait_for_speech_end(timeout=0.05)
        assert result is False


class TestHadSpeech:
    def test_initially_false(self):
        vad = VoiceActivityDetector(AudioConfig(), VADConfig())
        assert vad.had_speech is False

    def test_false_after_start(self):
        vad = VoiceActivityDetector(AudioConfig(), VADConfig())
        vad.start()
        assert vad.had_speech is False


class TestDuration:
    def test_zero_before_start(self):
        vad = VoiceActivityDetector(AudioConfig(), VADConfig())
        assert vad.duration == 0

    def test_positive_after_start(self):
        vad = VoiceActivityDetector(AudioConfig(), VADConfig())
        vad.start()
        time.sleep(0.01)
        assert vad.duration > 0.0
