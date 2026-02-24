# Copyright (c) 2026 Alessandro Orrù
# Licensed under MIT

import asyncio
import pytest
from andromeda.intent import _intents, clear_intents, match_and_execute, register_intent


@pytest.fixture(autouse=True)
def clear_registered_intents():
    """Clear registered intents before each test."""
    clear_intents()
    yield
    clear_intents()


class TestRegisterIntent:
    def test_register_single_pattern(self):
        register_intent(patterns=[r"\bciao\b"], tool_handler=lambda args: "hello")
        assert len(_intents) == 1
        assert len(_intents[0]["patterns"]) == 1

    def test_register_multiple_patterns(self):
        register_intent(
            patterns=[r"\bora\b", r"\bore\b", r"\borario\b"],
            tool_handler=lambda args: "time",
        )
        assert len(_intents) == 1
        assert len(_intents[0]["patterns"]) == 3

    def test_register_with_args(self):
        register_intent(
            patterns=[r"\bvolume\b"],
            tool_handler=lambda args: "vol",
            args={"action": "volume_up"},
        )
        assert _intents[0]["args"] == {"action": "volume_up"}

    def test_register_without_args(self):
        register_intent(patterns=[r"\btest\b"], tool_handler=lambda args: "ok")
        assert _intents[0]["args"] == {}

    def test_register_multiple_intents(self):
        register_intent(patterns=[r"\bciao\b"], tool_handler=lambda args: "hi")
        register_intent(patterns=[r"\baddio\b"], tool_handler=lambda args: "bye")
        assert len(_intents) == 2


class TestMatchAndExecute:
    @pytest.mark.asyncio
    async def test_match_simple_pattern(self):
        register_intent(patterns=[r"\bciao\b"], tool_handler=lambda args: "hello!")
        result = await match_and_execute("ciao come stai")
        assert result == "hello!"

    @pytest.mark.asyncio
    async def test_no_match_returns_none(self):
        register_intent(patterns=[r"\bciao\b"], tool_handler=lambda args: "hello!")
        result = await match_and_execute("buongiorno a tutti")
        assert result is None

    @pytest.mark.asyncio
    async def test_case_insensitive(self):
        register_intent(patterns=[r"\bciao\b"], tool_handler=lambda args: "hello!")
        result = await match_and_execute("CIAO mondo")
        assert result == "hello!"

    @pytest.mark.asyncio
    async def test_first_intent_wins(self):
        register_intent(patterns=[r"\btest\b"], tool_handler=lambda args: "first")
        register_intent(patterns=[r"\btest\b"], tool_handler=lambda args: "second")
        result = await match_and_execute("test")
        assert result == "first"

    @pytest.mark.asyncio
    async def test_multiple_patterns_any_matches(self):
        register_intent(
            patterns=[r"\bche\s+ora\b", r"\bore\s+sono\b"],
            tool_handler=lambda args: "sono le 10",
        )
        assert await match_and_execute("che ora è?") == "sono le 10"
        assert await match_and_execute("che ore sono?") == "sono le 10"

    @pytest.mark.asyncio
    async def test_args_passed_to_handler(self):
        def handler(args):
            return f"action={args.get('action')}"

        register_intent(
            patterns=[r"\balza\b"],
            tool_handler=handler,
            args={"action": "volume_up"},
        )
        result = await match_and_execute("alza il volume")
        assert result == "action=volume_up"

    @pytest.mark.asyncio
    async def test_async_handler(self):
        async def async_handler(args):
            await asyncio.sleep(0)  # Simulate async work
            return "async_result"

        register_intent(patterns=[r"\basync\b"], tool_handler=async_handler)
        result = await match_and_execute("test async")
        assert result == "async_result"

    @pytest.mark.asyncio
    async def test_empty_text(self):
        register_intent(patterns=[r"\bciao\b"], tool_handler=lambda args: "hello!")
        result = await match_and_execute("")
        assert result is None

    @pytest.mark.asyncio
    async def test_whitespace_text(self):
        register_intent(patterns=[r"\bciao\b"], tool_handler=lambda args: "hello!")
        result = await match_and_execute("   ")
        assert result is None

    @pytest.mark.asyncio
    async def test_real_datetime_pattern(self):
        """Test patterns similar to actual datetime intents."""
        register_intent(
            patterns=[r"\b(che\s+)?or[ae]\b", r"\bche\s+ore\s+sono\b"],
            tool_handler=lambda args: "Sono le 15:30",
        )
        assert await match_and_execute("che ora è") == "Sono le 15:30"
        assert await match_and_execute("che ore sono") == "Sono le 15:30"
        assert await match_and_execute("dimmi l'ora") == "Sono le 15:30"

    @pytest.mark.asyncio
    async def test_real_volume_patterns(self):
        """Test patterns similar to actual volume intents."""
        register_intent(
            patterns=[r"\balza.*volume\b", r"\bvolume.*alto\b", r"\bpiù\s+forte\b"],
            tool_handler=lambda args: "Volume alzato.",
            args={"action": "volume_up"},
        )
        assert await match_and_execute("alza il volume") == "Volume alzato."
        assert await match_and_execute("più forte") == "Volume alzato."
        assert await match_and_execute("abbassa il volume") is None
