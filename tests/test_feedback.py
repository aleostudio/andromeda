# Copyright (c) 2026 Alessandro Orrù
# Licensed under MIT

from unittest.mock import patch
import numpy as np
import pytest
from andromeda.config import AudioConfig, FeedbackConfig
from andromeda.feedback import AudioFeedback


class TestAudioFeedbackInit:
    def test_init(self):
        fb = AudioFeedback(AudioConfig(), FeedbackConfig())
        assert len(fb._sounds) == 0


class TestToneGeneration:
    @pytest.fixture
    def feedback(self):
        fb = AudioFeedback(AudioConfig(), FeedbackConfig())
        return fb

    def test_wake_tone(self, feedback):
        tone = feedback._gen_wake_tone()
        assert isinstance(tone, np.ndarray)
        assert tone.dtype == np.float32
        assert len(tone) > 0

    def test_done_tone(self, feedback):
        tone = feedback._gen_done_tone()
        assert isinstance(tone, np.ndarray)
        assert tone.dtype == np.float32
        assert len(tone) > 0

    def test_error_tone(self, feedback):
        tone = feedback._gen_error_tone()
        assert isinstance(tone, np.ndarray)
        assert tone.dtype == np.float32
        assert len(tone) > 0

    def test_tones_different(self, feedback):
        wake = feedback._gen_wake_tone()
        done = feedback._gen_done_tone()
        error = feedback._gen_error_tone()
        # Different lengths or content
        assert len(wake) != len(done) or not np.array_equal(wake[:len(done)], done)
        assert len(done) != len(error) or not np.array_equal(done[:len(error)], error)

    def test_tone_amplitude_bounded(self, feedback):
        for gen in [feedback._gen_wake_tone, feedback._gen_done_tone, feedback._gen_error_tone]:
            tone = gen()
            assert np.max(np.abs(tone)) <= 1.0


class TestInitialize:
    def test_initialize_with_missing_files(self):
        """Should generate synthetic tones when WAV files don't exist."""
        cfg = FeedbackConfig(
            wake_sound="nonexistent/wake.wav",
            done_sound="nonexistent/done.wav",
            error_sound="nonexistent/error.wav",
        )
        fb = AudioFeedback(AudioConfig(), cfg)
        fb.initialize()

        assert "wake" in fb._sounds
        assert "done" in fb._sounds
        assert "error" in fb._sounds
        assert len(fb._sounds["wake"]) > 0
        assert len(fb._sounds["done"]) > 0
        assert len(fb._sounds["error"]) > 0


class TestPlay:
    def test_play_unknown_sound(self):
        """Playing unknown sound name should not crash."""
        fb = AudioFeedback(AudioConfig(), FeedbackConfig())
        fb.initialize()
        fb.play("nonexistent")  # Should not raise

    @patch("sounddevice.play")
    def test_play_calls_sounddevice(self, mock_sd_play):
        fb = AudioFeedback(AudioConfig(), FeedbackConfig())
        fb.initialize()
        fb.play("wake")
        mock_sd_play.assert_called_once()

    @patch("sounddevice.play", side_effect=Exception("device error"))
    def test_play_handles_error(self, mock_sd_play):
        fb = AudioFeedback(AudioConfig(), FeedbackConfig())
        fb.initialize()
        fb.play("wake")  # Should not raise despite error

    @patch("sounddevice.play")
    def test_play_blocking(self, mock_sd_play):
        fb = AudioFeedback(AudioConfig(), FeedbackConfig())
        fb.initialize()
        fb.play_blocking("done")
        mock_sd_play.assert_called_once()
        # Check blocking=True was passed
        _, kwargs = mock_sd_play.call_args
        assert kwargs.get("blocking") is True
