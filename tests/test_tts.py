# Copyright (c) 2026 Alessandro Orrù
# Licensed under MIT

import asyncio
import numpy as np
import pytest
from andromeda.config import AudioConfig, TTSConfig
from andromeda.tts import TextToSpeech, _FADE_SAMPLES, _apply_fade_out


class TestApplyFadeOut:
    def test_fade_applied(self):
        audio = np.ones(200, dtype=np.float32)
        _apply_fade_out(audio)
        # Last sample should be near zero
        assert audio[-1] == pytest.approx(0.0, abs=0.02)
        # First sample (outside fade region) should be 1.0
        assert audio[0] == 1
        # Sample just before fade region should be 1.0
        assert audio[200 - _FADE_SAMPLES - 1] == 1

    def test_fade_gradual(self):
        audio = np.ones(200, dtype=np.float32)
        _apply_fade_out(audio)
        # Values should decrease monotonically in the fade region
        fade_region = audio[-_FADE_SAMPLES:]
        for i in range(len(fade_region) - 1):
            assert fade_region[i] >= fade_region[i + 1]

    def test_too_short_audio_no_change(self):
        audio = np.ones(_FADE_SAMPLES - 1, dtype=np.float32)
        _apply_fade_out(audio)
        # Should be unchanged (too short for fade)
        np.testing.assert_array_equal(audio, np.ones(_FADE_SAMPLES - 1, dtype=np.float32))

    def test_exact_fade_length(self):
        audio = np.ones(_FADE_SAMPLES, dtype=np.float32)
        _apply_fade_out(audio)
        assert audio[0] == pytest.approx(1.0, abs=0.02)
        assert audio[-1] == pytest.approx(0.0, abs=0.02)

    def test_zero_audio_stays_zero(self):
        audio = np.zeros(200, dtype=np.float32)
        _apply_fade_out(audio)
        np.testing.assert_array_equal(audio, np.zeros(200, dtype=np.float32))

    def test_negative_values_fade(self):
        audio = -np.ones(200, dtype=np.float32)
        _apply_fade_out(audio)
        assert audio[-1] == pytest.approx(0.0, abs=0.02)
        assert audio[0] == -1.0


class TestFadeConstants:
    def test_fade_samples_positive(self):
        assert _FADE_SAMPLES > 0

    def test_fade_samples_reasonable(self):
        # At 22050Hz, 64 samples = ~3ms of fade — reasonable
        assert _FADE_SAMPLES <= 256


class TestTTSPlaybackLock:
    @pytest.mark.asyncio
    async def test_speak_waits_for_streaming_playback(self, monkeypatch):
        tts = TextToSpeech(AudioConfig(), TTSConfig())
        events: list[str] = []

        async def fake_stream(_queue):
            events.append("stream_start")
            await asyncio.sleep(0.05)
            events.append("stream_end")

        async def fake_speak(text):
            events.append(f"speak_start:{text}")
            await asyncio.sleep(0.01)
            events.append(f"speak_end:{text}")

        monkeypatch.setattr(tts, "_speak_streamed_engine", fake_stream)
        monkeypatch.setattr(tts, "_speak", fake_speak)

        stream_task = asyncio.create_task(tts.speak_streamed(asyncio.Queue()))
        await asyncio.sleep(0)
        speak_task = asyncio.create_task(tts.speak("timer completato"))

        await asyncio.gather(stream_task, speak_task)

        assert events == [
            "stream_start",
            "stream_end",
            "speak_start:timer completato",
            "speak_end:timer completato",
        ]

    @pytest.mark.asyncio
    async def test_notification_cue_waits_for_streaming_playback(self, monkeypatch):
        tts = TextToSpeech(AudioConfig(), TTSConfig())
        events: list[str] = []

        async def fake_stream(_queue):
            events.append("stream_start")
            await asyncio.sleep(0.05)
            events.append("stream_end")

        async def fake_speak(text):
            events.append(f"speak:{text}")

        monkeypatch.setattr(tts, "_speak_streamed_engine", fake_stream)
        monkeypatch.setattr(tts, "_speak", fake_speak)

        stream_task = asyncio.create_task(tts.speak_streamed(asyncio.Queue()))
        await asyncio.sleep(0)
        notification_task = asyncio.create_task(
            tts.speak_notification("timer completato", cue=lambda: events.append("cue")),
        )

        await asyncio.gather(stream_task, notification_task)

        assert events == [
            "stream_start",
            "stream_end",
            "cue",
            "speak:timer completato",
        ]
