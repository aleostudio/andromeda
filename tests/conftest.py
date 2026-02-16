# Copyright (c) 2026 Alessandro Orrù
# Licensed under MIT

import pytest
from andromeda.config import (
    AgentConfig,
    AppConfig,
    AudioConfig,
    ConversationConfig,
    FeedbackConfig,
    LoggingConfig,
    NoiseConfig,
    STTConfig,
    ToolsConfig,
    TTSConfig,
    VADConfig,
    WakeWordConfig,
)


@pytest.fixture
def audio_cfg() -> AudioConfig:
    return AudioConfig()


@pytest.fixture
def vad_cfg() -> VADConfig:
    return VADConfig()


@pytest.fixture
def agent_cfg() -> AgentConfig:
    return AgentConfig()


@pytest.fixture
def conversation_cfg() -> ConversationConfig:
    return ConversationConfig()


@pytest.fixture
def tts_cfg() -> TTSConfig:
    return TTSConfig()


@pytest.fixture
def feedback_cfg() -> FeedbackConfig:
    return FeedbackConfig()


@pytest.fixture
def tools_cfg() -> ToolsConfig:
    return ToolsConfig()


@pytest.fixture
def app_cfg() -> AppConfig:
    return AppConfig()
