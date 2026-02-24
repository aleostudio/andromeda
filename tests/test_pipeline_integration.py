# Copyright (c) 2026 Alessandro Orrù
# Licensed under MIT

from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch
import numpy as np
import pytest
from andromeda.main import VoiceAssistant
from andromeda.messages import GENERIC_ERROR_RETRY
from andromeda.metrics import PerformanceMetrics
from andromeda.state_machine import AssistantState


def _build_assistant_for_processing(streaming: bool = False) -> VoiceAssistant:
    assistant = VoiceAssistant.__new__(VoiceAssistant)
    assistant._cfg = SimpleNamespace(agent=SimpleNamespace(streaming=streaming))
    assistant._metrics = PerformanceMetrics()
    assistant._recorded_audio = np.array([0.1, 0.2], dtype=np.float32)
    assistant._stt = SimpleNamespace(transcribe=AsyncMock(return_value="test input"))
    assistant._audio = SimpleNamespace(mute=MagicMock(), unmute=MagicMock())
    assistant._tts = SimpleNamespace(speak=AsyncMock())
    assistant._feedback = SimpleNamespace(play=MagicMock(), stop=MagicMock())
    assistant._response_text = ""
    assistant._tts_interrupted = False
    assistant._process_standard = AsyncMock()
    assistant._process_streaming = AsyncMock()
    assistant._speak_error = AsyncMock()

    return assistant


class TestPipelineIntegration:
    @pytest.mark.asyncio
    async def test_processing_fast_intent_path(self):
        assistant = _build_assistant_for_processing(streaming=False)
        assistant._stt.transcribe = AsyncMock(return_value="che ora è")

        with patch("andromeda.main.match_and_execute", AsyncMock(return_value="Sono le dieci")):
            next_state = await VoiceAssistant._handle_processing(assistant, AssistantState.PROCESSING)

        assert next_state == AssistantState.SPEAKING
        assert assistant._response_text == "Sono le dieci"
        assert assistant._tts.speak.await_count == 1
        assert assistant._audio.mute.call_count == 1
        assert assistant._audio.unmute.call_count == 1

    @pytest.mark.asyncio
    async def test_processing_llm_failure_fallback(self):
        assistant = _build_assistant_for_processing(streaming=False)
        assistant._process_standard = AsyncMock(side_effect=RuntimeError("llm failure"))

        with patch("andromeda.main.match_and_execute", AsyncMock(return_value=None)):
            next_state = await VoiceAssistant._handle_processing(assistant, AssistantState.PROCESSING)

        assert next_state == AssistantState.SPEAKING
        assistant._speak_error.assert_awaited_once_with(GENERIC_ERROR_RETRY)
        assert assistant._response_text == GENERIC_ERROR_RETRY
