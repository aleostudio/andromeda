# Copyright (c) 2026 Alessandro Orrù
# Licensed under MIT

import tempfile
from pathlib import Path
import pytest
import yaml
from andromeda.config import (
    AgentConfig,
    AppConfig,
    AudioConfig,
    ConversationConfig,
    FeedbackConfig,
    HealthCheckConfig,
    LoggingConfig,
    NoiseConfig,
    STTConfig,
    ToolsConfig,
    TTSConfig,
    VADConfig,
    WakeWordConfig,
)


class TestAudioConfig:
    def test_defaults(self):
        cfg = AudioConfig()
        assert cfg.sample_rate == 16000
        assert cfg.channels == 1
        assert cfg.chunk_ms == 30
        assert cfg.dtype == "int16"

    def test_chunk_samples(self):
        cfg = AudioConfig(sample_rate=16000, chunk_ms=30)
        assert cfg.chunk_samples == 480  # 16000 * 30 / 1000

    def test_chunk_samples_20ms(self):
        cfg = AudioConfig(sample_rate=16000, chunk_ms=20)
        assert cfg.chunk_samples == 320

    def test_chunk_bytes(self):
        cfg = AudioConfig(sample_rate=16000, chunk_ms=30)
        assert cfg.chunk_bytes == 960  # 480 * 2

    def test_frozen(self):
        cfg = AudioConfig()
        with pytest.raises(AttributeError):
            cfg.sample_rate = 44100

    def test_invalid_sample_rate(self):
        with pytest.raises(ValueError, match="sample_rate"):
            AudioConfig(sample_rate=0)

    def test_invalid_channels(self):
        with pytest.raises(ValueError, match="channels"):
            AudioConfig(channels=0)

    def test_invalid_chunk_ms(self):
        with pytest.raises(ValueError, match="chunk_ms"):
            AudioConfig(chunk_ms=15)


class TestVADConfig:
    def test_defaults(self):
        cfg = VADConfig()
        assert cfg.aggressiveness == 3
        assert cfg.silence_timeout_sec == pytest.approx(1.5)
        assert cfg.speech_pad_ms == 300
        assert cfg.max_recording_sec == pytest.approx(30.0)
        assert cfg.min_recording_sec == pytest.approx(0.5)
        assert cfg.energy_threshold_factor == pytest.approx(0.6)
        assert cfg.energy_decay_rate == pytest.approx(0.98)

    def test_custom_values(self):
        cfg = VADConfig(
            aggressiveness=2,
            silence_timeout_sec=1.5,
            energy_threshold_factor=0.6,
        )
        assert cfg.aggressiveness == 2
        assert cfg.silence_timeout_sec == pytest.approx(1.5)
        assert cfg.energy_threshold_factor == pytest.approx(0.6)

    def test_invalid_aggressiveness(self):
        with pytest.raises(ValueError, match="aggressiveness"):
            VADConfig(aggressiveness=5)

    def test_invalid_energy_decay_rate(self):
        with pytest.raises(ValueError, match="energy_decay_rate"):
            VADConfig(energy_decay_rate=0.0)

    def test_invalid_max_recording(self):
        with pytest.raises(ValueError, match="max_recording_sec"):
            VADConfig(max_recording_sec=-1)


class TestSTTConfig:
    def test_defaults(self):
        cfg = STTConfig()
        assert cfg.model_size == "medium"
        assert cfg.beam_size == 1
        assert cfg.vad_filter is False

    def test_invalid_beam_size(self):
        with pytest.raises(ValueError, match="beam_size"):
            STTConfig(beam_size=0)


class TestAgentConfig:
    def test_defaults(self):
        cfg = AgentConfig()
        assert cfg.provider == "ollama"
        assert cfg.base_url == "http://localhost:11434"
        assert cfg.model == "llama3.1:8b"
        assert cfg.max_tokens == 500
        assert cfg.streaming is True
        assert cfg.streaming_clause_split is True

    def test_streaming_flag(self):
        cfg = AgentConfig(streaming=False)
        assert cfg.streaming is False

    def test_streaming_clause_split_flag(self):
        cfg = AgentConfig(streaming_clause_split=False)
        assert cfg.streaming_clause_split is False

    def test_system_prompt_present(self):
        cfg = AgentConfig()
        assert "Andromeda" in cfg.system_prompt

    def test_invalid_max_tokens(self):
        with pytest.raises(ValueError, match="max_tokens"):
            AgentConfig(max_tokens=0)


class TestConversationConfig:
    def test_defaults(self):
        cfg = ConversationConfig()
        assert cfg.follow_up_timeout_sec == pytest.approx(5.0)
        assert cfg.history_timeout_sec == pytest.approx(300.0)

    def test_custom_values(self):
        cfg = ConversationConfig(
            follow_up_timeout_sec=10.0,
            history_timeout_sec=0.0,
        )
        assert cfg.follow_up_timeout_sec == pytest.approx(10.0)
        assert cfg.history_timeout_sec == pytest.approx(0.0)


class TestTTSConfig:
    def test_defaults(self):
        cfg = TTSConfig()
        assert cfg.engine == "piper"
        assert cfg.speaker_id == 0
        assert cfg.length_scale == pytest.approx(1.0)
        assert cfg.sentence_silence == pytest.approx(0.3)


class TestWakeWordConfig:
    def test_defaults(self):
        cfg = WakeWordConfig()
        assert cfg.threshold == pytest.approx(0.5)

    def test_invalid_threshold(self):
        with pytest.raises(ValueError, match="threshold"):
            WakeWordConfig(threshold=1.5)


class TestNoiseConfig:
    def test_defaults(self):
        cfg = NoiseConfig()
        assert cfg.enabled is False
        assert cfg.prop_decrease == pytest.approx(0.75)

    def test_invalid_prop_decrease(self):
        with pytest.raises(ValueError, match="prop_decrease"):
            NoiseConfig(prop_decrease=2.0)


class TestHealthCheckConfig:
    def test_defaults(self):
        cfg = HealthCheckConfig()
        assert cfg.port == 8080

    def test_invalid_port(self):
        with pytest.raises(ValueError, match="port"):
            HealthCheckConfig(port=0)


class TestLoggingConfig:
    def test_defaults(self):
        cfg = LoggingConfig()
        assert cfg.level == "INFO"

    def test_invalid_level(self):
        with pytest.raises(ValueError, match="level"):
            LoggingConfig(level="VERBOSE")


class TestToolsConfig:
    def test_defaults(self):
        cfg = ToolsConfig()
        assert cfg.knowledge_base_path == "data/knowledge.json"
        assert cfg.allow_sensitive_memory is False
        assert cfg.weather_timeout_sec == pytest.approx(10.0)
        assert cfg.news_timeout_sec == pytest.approx(10.0)
        assert cfg.timer_max_sec == 3600
        assert cfg.allow_system_control is True


class TestAppConfig:
    def test_defaults(self):
        cfg = AppConfig()
        assert isinstance(cfg.audio, AudioConfig)
        assert isinstance(cfg.vad, VADConfig)
        assert isinstance(cfg.agent, AgentConfig)
        assert isinstance(cfg.conversation, ConversationConfig)
        assert isinstance(cfg.tts, TTSConfig)
        assert isinstance(cfg.tools, ToolsConfig)
        assert isinstance(cfg.feedback, FeedbackConfig)
        assert isinstance(cfg.logging, LoggingConfig)

    def test_from_yaml_missing_file(self):
        cfg = AppConfig.from_yaml("/nonexistent/config.yaml")
        assert cfg == AppConfig()

    def test_from_yaml_empty_file(self):
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".yaml", delete=False,
        ) as f:
            f.write("")
            path = f.name

        cfg = AppConfig.from_yaml(path)
        assert isinstance(cfg, AppConfig)
        Path(path).unlink()

    def test_from_yaml_partial(self):
        data = {
            "agent": {"model": "mistral:7b", "streaming": True, "streaming_clause_split": False},
            "vad": {"aggressiveness": 2},
        }
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".yaml", delete=False,
        ) as f:
            yaml.dump(data, f)
            path = f.name

        cfg = AppConfig.from_yaml(path)
        assert cfg.agent.model == "mistral:7b"
        assert cfg.agent.streaming is True
        assert cfg.agent.streaming_clause_split is False
        assert cfg.vad.aggressiveness == 2
        # Defaults preserved
        assert cfg.audio.sample_rate == 16000
        assert cfg.tts.engine == "piper"
        Path(path).unlink()

    def test_from_yaml_conversation_section(self):
        data = {
            "conversation": {
                "follow_up_timeout_sec": 8.0,
                "history_timeout_sec": 600.0,
            },
        }
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".yaml", delete=False,
        ) as f:
            yaml.dump(data, f)
            path = f.name

        cfg = AppConfig.from_yaml(path)
        assert cfg.conversation.follow_up_timeout_sec == pytest.approx(8.0)
        assert cfg.conversation.history_timeout_sec == pytest.approx(600.0)
        Path(path).unlink()

    def test_from_yaml_full(self):
        data = {
            "audio": {
                "sample_rate": 44100,
                "channels": 2,
                "chunk_ms": 20,
                "dtype": "float32",
            },
            "agent": {"model": "custom:3b", "max_tokens": 200},
            "logging": {"level": "DEBUG"},
        }
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".yaml", delete=False,
        ) as f:
            yaml.dump(data, f)
            path = f.name

        cfg = AppConfig.from_yaml(path)
        assert cfg.audio.sample_rate == 44100
        assert cfg.audio.channels == 2
        assert cfg.agent.model == "custom:3b"
        assert cfg.logging.level == "DEBUG"
        Path(path).unlink()

    def test_from_yaml_invalid_config_raises(self):
        data = {
            "vad": {"aggressiveness": 99},
        }
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".yaml", delete=False,
        ) as f:
            yaml.dump(data, f)
            path = f.name

        with pytest.raises(ValueError, match="aggressiveness"):
            AppConfig.from_yaml(path)
        Path(path).unlink()
