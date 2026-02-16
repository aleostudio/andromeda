# Copyright (c) 2026 Alessandro Orrù
# Licensed under MIT

import asyncio
import pytest
from andromeda.state_machine import TRANSITIONS, AssistantState, StateMachine


class TestAssistantState:
    def test_all_states_exist(self):
        assert AssistantState.IDLE is not None
        assert AssistantState.LISTENING is not None
        assert AssistantState.PROCESSING is not None
        assert AssistantState.SPEAKING is not None
        assert AssistantState.ERROR is not None

    def test_str(self):
        assert str(AssistantState.IDLE) == "IDLE"
        assert str(AssistantState.LISTENING) == "LISTENING"
        assert str(AssistantState.SPEAKING) == "SPEAKING"


class TestTransitions:
    def test_idle_can_go_to_listening(self):
        assert AssistantState.LISTENING in TRANSITIONS[AssistantState.IDLE]

    def test_idle_cannot_go_to_processing(self):
        assert AssistantState.PROCESSING not in TRANSITIONS[AssistantState.IDLE]

    def test_listening_can_go_to_processing(self):
        assert AssistantState.PROCESSING in TRANSITIONS[AssistantState.LISTENING]

    def test_listening_can_go_to_idle(self):
        assert AssistantState.IDLE in TRANSITIONS[AssistantState.LISTENING]

    def test_listening_can_go_to_error(self):
        assert AssistantState.ERROR in TRANSITIONS[AssistantState.LISTENING]

    def test_processing_can_go_to_speaking(self):
        assert AssistantState.SPEAKING in TRANSITIONS[AssistantState.PROCESSING]

    def test_processing_can_go_to_idle(self):
        assert AssistantState.IDLE in TRANSITIONS[AssistantState.PROCESSING]

    def test_speaking_can_go_to_idle(self):
        assert AssistantState.IDLE in TRANSITIONS[AssistantState.SPEAKING]

    def test_speaking_can_go_to_listening(self):
        """Follow-up conversation: SPEAKING -> LISTENING."""
        assert AssistantState.LISTENING in TRANSITIONS[AssistantState.SPEAKING]

    def test_error_can_go_to_idle(self):
        assert AssistantState.IDLE in TRANSITIONS[AssistantState.ERROR]

    def test_error_cannot_go_elsewhere(self):
        assert TRANSITIONS[AssistantState.ERROR] == {AssistantState.IDLE}


class TestStateMachine:
    def test_initial_state(self):
        sm = StateMachine()
        assert sm.state == AssistantState.IDLE

    @pytest.mark.asyncio
    async def test_valid_transition(self):
        sm = StateMachine()
        await sm.transition_to(AssistantState.LISTENING)
        assert sm.state == AssistantState.LISTENING

    @pytest.mark.asyncio
    async def test_invalid_transition_stays(self):
        sm = StateMachine()
        # IDLE -> SPEAKING is not valid
        await sm.transition_to(AssistantState.SPEAKING)
        assert sm.state == AssistantState.IDLE

    @pytest.mark.asyncio
    async def test_transition_callback(self):
        sm = StateMachine()
        transitions = []
        sm.on_transition(lambda old, new: transitions.append((old, new)))

        await sm.transition_to(AssistantState.LISTENING)
        assert len(transitions) == 1
        assert transitions[0] == (AssistantState.IDLE, AssistantState.LISTENING)

    @pytest.mark.asyncio
    async def test_transition_callback_not_called_on_invalid(self):
        sm = StateMachine()
        transitions = []
        sm.on_transition(lambda old, new: transitions.append((old, new)))

        await sm.transition_to(AssistantState.SPEAKING)  # Invalid
        assert len(transitions) == 0

    @pytest.mark.asyncio
    async def test_register_handler(self):
        sm = StateMachine()
        handler_called = False

        async def handler(state):
            await asyncio.sleep(0)
            nonlocal handler_called
            handler_called = True
            raise asyncio.CancelledError()  # Stop after handler runs

        sm.register_handler(AssistantState.IDLE, handler)

        with pytest.raises(asyncio.CancelledError):
            await sm.run()

        assert handler_called

    @pytest.mark.asyncio
    async def test_run_transitions_through_states(self):
        sm = StateMachine()
        visited = []

        async def idle_handler(state):
            await asyncio.sleep(0)
            visited.append("IDLE")
            return AssistantState.LISTENING

        async def listening_handler(state):
            await asyncio.sleep(0)
            visited.append("LISTENING")
            return AssistantState.PROCESSING

        async def processing_handler(state):
            await asyncio.sleep(0)
            visited.append("PROCESSING")
            return AssistantState.SPEAKING

        async def speaking_handler(state):
            await asyncio.sleep(0)
            visited.append("SPEAKING")
            # Simulate follow-up disabled -> back to IDLE
            raise asyncio.CancelledError()  # Stop after one full cycle

        sm.register_handler(AssistantState.IDLE, idle_handler)
        sm.register_handler(AssistantState.LISTENING, listening_handler)
        sm.register_handler(AssistantState.PROCESSING, processing_handler)
        sm.register_handler(AssistantState.SPEAKING, speaking_handler)

        with pytest.raises(asyncio.CancelledError):
            await sm.run()

        assert visited == ["IDLE", "LISTENING", "PROCESSING", "SPEAKING"]

    @pytest.mark.asyncio
    async def test_handler_exception_goes_to_error(self):
        """Exception in LISTENING handler -> transitions to ERROR."""
        sm = StateMachine()
        reached_error = False

        async def idle_handler(state):
            await asyncio.sleep(0)
            return AssistantState.LISTENING

        async def bad_listening_handler(state):
            raise ValueError("Something broke")

        async def error_handler(state):
            await asyncio.sleep(0)
            nonlocal reached_error
            reached_error = True
            raise asyncio.CancelledError()  # Stop

        sm.register_handler(AssistantState.IDLE, idle_handler)
        sm.register_handler(AssistantState.LISTENING, bad_listening_handler)
        sm.register_handler(AssistantState.ERROR, error_handler)

        with pytest.raises(asyncio.CancelledError):
            await sm.run()

        assert reached_error

    @pytest.mark.asyncio
    async def test_speaking_to_listening_follow_up(self):
        """Verify SPEAKING -> LISTENING transition for multi-turn conversation."""
        sm = StateMachine()

        await sm.transition_to(AssistantState.LISTENING)
        await sm.transition_to(AssistantState.PROCESSING)
        await sm.transition_to(AssistantState.SPEAKING)
        await sm.transition_to(AssistantState.LISTENING)

        assert sm.state == AssistantState.LISTENING
