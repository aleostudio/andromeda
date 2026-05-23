# Copyright (c) 2026 Alessandro Orrù
# Licensed under MIT

from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch
import numpy as np
import pytest
from andromeda.main import VoiceAssistant
from andromeda.messages import msg
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
        assistant._speak_error.assert_awaited_once_with(msg("core.generic_error_retry"))
        assert assistant._response_text == msg("core.generic_error_retry")

    @pytest.mark.asyncio
    async def test_standard_processing_does_not_monitor_interrupt_when_barge_in_disabled(self):
        assistant = VoiceAssistant.__new__(VoiceAssistant)
        assistant._cfg = SimpleNamespace(conversation=SimpleNamespace(barge_in_enabled=False))
        assistant._metrics = PerformanceMetrics()
        assistant._agent = SimpleNamespace(process=AsyncMock(return_value="risposta"))
        assistant._tts = SimpleNamespace(speak=AsyncMock())
        assistant._audio = SimpleNamespace(mute=MagicMock(), monitor_only=MagicMock())
        assistant._wake_word = SimpleNamespace(reset=MagicMock())
        assistant._monitor_interrupt = AsyncMock()
        assistant._response_text = ""

        await VoiceAssistant._process_standard(assistant, "test")

        assistant._audio.mute.assert_called_once()
        assistant._audio.monitor_only.assert_not_called()
        assistant._wake_word.reset.assert_not_called()
        assistant._monitor_interrupt.assert_not_called()

    def test_request_shutdown_unblocks_runtime_components(self):
        assistant = VoiceAssistant.__new__(VoiceAssistant)
        assistant._shutdown_requested = False
        assistant._wake_word = SimpleNamespace(shutdown=MagicMock())
        assistant._vad = SimpleNamespace(stop=MagicMock())
        assistant._tts = SimpleNamespace(stop_playback=MagicMock())
        assistant._feedback = SimpleNamespace(stop=MagicMock())

        VoiceAssistant.request_shutdown(assistant)

        assert assistant._shutdown_requested is True
        assistant._wake_word.shutdown.assert_called_once()
        assistant._vad.stop.assert_called_once()
        assistant._tts.stop_playback.assert_called_once()
        assistant._feedback.stop.assert_called_once()

    def test_listening_to_idle_transition_plays_idle_cue(self):
        assistant = VoiceAssistant.__new__(VoiceAssistant)
        assistant._shutdown_requested = False
        assistant._feedback = SimpleNamespace(play=MagicMock())

        VoiceAssistant._handle_transition(
            assistant,
            AssistantState.LISTENING,
            AssistantState.IDLE,
        )

        assistant._feedback.play.assert_called_once_with("idle")

    def test_shutdown_suppresses_idle_transition_cue(self):
        assistant = VoiceAssistant.__new__(VoiceAssistant)
        assistant._shutdown_requested = True
        assistant._feedback = SimpleNamespace(play=MagicMock())

        VoiceAssistant._handle_transition(
            assistant,
            AssistantState.LISTENING,
            AssistantState.IDLE,
        )

        assistant._feedback.play.assert_not_called()
