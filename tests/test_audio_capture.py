# Copyright (c) 2026 Alessandro Orrù
# Licensed under MIT

import struct
from unittest.mock import MagicMock, patch
import numpy as np
import pytest
from andromeda.audio_capture import AudioCapture, AudioRouteMode
from andromeda.config import AudioConfig, NoiseConfig


class TestAudioRouteMode:
    def test_modes_exist(self):
        assert AudioRouteMode.NORMAL is not None
        assert AudioRouteMode.MUTED is not None


class TestAudioCaptureInit:
    def test_initial_state(self):
        cap = AudioCapture(AudioConfig(), NoiseConfig())
        assert cap._route_mode == AudioRouteMode.NORMAL
        assert cap._is_recording is False
        assert len(cap._recording_buffer) == 0


class TestMuteUnmute:
    def test_mute(self):
        cap = AudioCapture(AudioConfig(), NoiseConfig())
        cap.mute()
        assert cap._route_mode == AudioRouteMode.MUTED

    def test_unmute(self):
        cap = AudioCapture(AudioConfig(), NoiseConfig())
        cap.mute()
        cap.unmute()
        assert cap._route_mode == AudioRouteMode.NORMAL


class TestRecording:
    def test_start_recording(self):
        cap = AudioCapture(AudioConfig(), NoiseConfig(enabled=False))
        cap.start_recording()
        assert cap._is_recording is True

    def test_stop_recording_empty(self):
        cap = AudioCapture(AudioConfig(), NoiseConfig(enabled=False))
        cap.start_recording()
        audio = cap.stop_recording()
        assert isinstance(audio, np.ndarray)
        assert len(audio) == 0

    def test_stop_recording_with_data(self):
        cap = AudioCapture(AudioConfig(), NoiseConfig(enabled=False))
        cap.start_recording()

        # Simulate audio frames
        frame = struct.pack("<480h", *([1000] * 480))
        cap._recording_buffer.append(frame)
        cap._recording_buffer.append(frame)

        audio = cap.stop_recording()
        assert isinstance(audio, np.ndarray)
        assert len(audio) == 960  # 480 * 2
        assert audio.dtype == np.float32

    def test_stop_recording_clears_buffer(self):
        cap = AudioCapture(AudioConfig(), NoiseConfig(enabled=False))
        cap.start_recording()
        cap._recording_buffer.append(b"\x00" * 960)
        cap.stop_recording()
        assert len(cap._recording_buffer) == 0
        assert cap._is_recording is False


class TestAudioCallback:
    def test_normal_mode_dispatches(self):
        cap = AudioCapture(AudioConfig(), NoiseConfig(enabled=False))
        received = []
        cap.on_audio_frame(lambda fb, fa: received.append(fb))

        # Simulate callback
        indata = np.zeros((480, 1), dtype=np.int16)
        cap._audio_callback(indata, 480, None, None)

        assert len(received) == 1

    def test_muted_drops_frames(self):
        cap = AudioCapture(AudioConfig(), NoiseConfig(enabled=False))
        received = []
        cap.on_audio_frame(lambda fb, fa: received.append(fb))
        cap.mute()

        indata = np.zeros((480, 1), dtype=np.int16)
        cap._audio_callback(indata, 480, None, None)

        assert len(received) == 0

    def test_recording_stores_frames(self):
        cap = AudioCapture(AudioConfig(), NoiseConfig(enabled=False))
        cap.start_recording()

        indata = np.zeros((480, 1), dtype=np.int16)
        cap._audio_callback(indata, 480, None, None)

        assert len(cap._recording_buffer) == 1

    def test_ring_buffer_stores_frames(self):
        cap = AudioCapture(AudioConfig(), NoiseConfig(enabled=False))
        indata = np.zeros((480, 1), dtype=np.int16)
        cap._audio_callback(indata, 480, None, None)

        assert len(cap._ring_buffer) == 1

    def test_callback_error_handled(self):
        cap = AudioCapture(AudioConfig(), NoiseConfig(enabled=False))
        cap.on_audio_frame(lambda fb, fa: 1 / 0)  # Will raise

        indata = np.zeros((480, 1), dtype=np.int16)
        # Should not raise
        cap._audio_callback(indata, 480, None, None)


class TestRingBuffer:
    def test_get_ring_buffer_empty(self):
        cap = AudioCapture(AudioConfig(), NoiseConfig(enabled=False))
        assert cap.get_ring_buffer_audio() == b""

    def test_ring_buffer_max_size(self):
        cap = AudioCapture(AudioConfig(), NoiseConfig(enabled=False), ring_buffer_sec=0.1)
        indata = np.zeros((480, 1), dtype=np.int16)

        # Push many frames
        for _ in range(100):
            cap._audio_callback(indata, 480, None, None)

        # Ring buffer should be capped
        assert len(cap._ring_buffer) <= cap._ring_buffer.maxlen


class TestCalibrateSpeechEnergy:
    def test_empty_buffer(self):
        cap = AudioCapture(AudioConfig(), NoiseConfig(enabled=False))
        mock_vad = MagicMock()
        energy = cap.calibrate_speech_energy(mock_vad, 16000)
        assert energy == 0

    def test_no_speech_frames(self):
        cap = AudioCapture(AudioConfig(), NoiseConfig(enabled=False))
        mock_vad = MagicMock()
        mock_vad.is_speech.return_value = False

        # Add some frames to ring buffer
        frame = struct.pack("<480h", *([100] * 480))
        cap._ring_buffer.append(frame)

        energy = cap.calibrate_speech_energy(mock_vad, 16000)
        assert energy == 0

    def test_with_speech_frames(self):
        cap = AudioCapture(AudioConfig(), NoiseConfig(enabled=False))
        mock_vad = MagicMock()
        mock_vad.is_speech.return_value = True

        # Add loud frames
        frame = struct.pack("<480h", *([5000] * 480))
        cap._ring_buffer.append(frame)
        cap._ring_buffer.append(frame)

        energy = cap.calibrate_speech_energy(mock_vad, 16000)
        assert energy > 0.0
