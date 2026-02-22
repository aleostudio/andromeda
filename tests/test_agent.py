# Copyright (c) 2026 Alessandro Orrù
# Licensed under MIT

import asyncio
import json
import time
import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from andromeda.agent import AIAgent
from andromeda.config import AgentConfig, ConversationConfig


class TestHistoryTimeout:
    def test_no_clear_when_no_config(self):
        agent = AIAgent(AgentConfig())
        agent._conversation = [{"role": "user", "content": "test"}]
        agent._last_interaction = time.monotonic() - 9999
        agent._check_history_timeout()
        assert len(agent._conversation) == 1

    def test_no_clear_when_timeout_zero(self):
        conv_cfg = ConversationConfig(history_timeout_sec=0)
        agent = AIAgent(AgentConfig(), conv_cfg)
        agent._conversation = [{"role": "user", "content": "test"}]
        agent._last_interaction = time.monotonic() - 9999
        agent._check_history_timeout()
        assert len(agent._conversation) == 1

    def test_no_clear_when_first_interaction(self):
        conv_cfg = ConversationConfig(history_timeout_sec=60)
        agent = AIAgent(AgentConfig(), conv_cfg)
        agent._conversation = [{"role": "user", "content": "test"}]
        agent._last_interaction = 0
        agent._check_history_timeout()
        assert len(agent._conversation) == 1

    def test_clear_when_timeout_exceeded(self):
        conv_cfg = ConversationConfig(history_timeout_sec=60)
        agent = AIAgent(AgentConfig(), conv_cfg)
        agent._conversation = [
            {"role": "user", "content": "ciao"},
            {"role": "assistant", "content": "ciao!"},
        ]
        agent._last_interaction = time.monotonic() - 120  # 120s > 60s timeout
        agent._check_history_timeout()
        assert len(agent._conversation) == 0

    def test_no_clear_when_recent(self):
        conv_cfg = ConversationConfig(history_timeout_sec=300)
        agent = AIAgent(AgentConfig(), conv_cfg)
        agent._conversation = [{"role": "user", "content": "test"}]
        agent._last_interaction = time.monotonic() - 10  # Only 10s ago
        agent._check_history_timeout()
        assert len(agent._conversation) == 1


class TestParseToolArgs:
    def test_dict_passthrough(self):
        assert AIAgent._parse_tool_args({"key": "value"}) == {"key": "value"}

    def test_string_json(self):
        assert AIAgent._parse_tool_args('{"action": "save"}') == {"action": "save"}

    def test_invalid_string(self):
        assert AIAgent._parse_tool_args("not json") == {}

    def test_empty_dict(self):
        assert AIAgent._parse_tool_args({}) == {}

    def test_empty_string(self):
        assert AIAgent._parse_tool_args("") == {}

    def test_nested_json_string(self):
        result = AIAgent._parse_tool_args('{"city": "Roma", "country": "Italia"}')
        assert result == {"city": "Roma", "country": "Italia"}


class TestFlushClauses:
    @pytest.mark.asyncio
    async def test_no_split_single_fragment(self):
        queue = asyncio.Queue()
        remainder = await AIAgent._flush_clauses("hello world", queue)
        assert remainder == "hello world"
        assert queue.empty()

    @pytest.mark.asyncio
    async def test_split_two_sentences(self):
        queue = asyncio.Queue()
        remainder = await AIAgent._flush_clauses("Prima frase. Seconda frase", queue)
        assert remainder == "Seconda frase"
        assert await queue.get() == "Prima frase."

    @pytest.mark.asyncio
    async def test_split_exclamation(self):
        queue = asyncio.Queue()
        remainder = await AIAgent._flush_clauses("Ciao! Come stai", queue)
        assert remainder == "Come stai"
        assert await queue.get() == "Ciao!"

    @pytest.mark.asyncio
    async def test_split_question_mark(self):
        queue = asyncio.Queue()
        remainder = await AIAgent._flush_clauses("Come va? Bene", queue)
        assert remainder == "Bene"
        assert await queue.get() == "Come va?"

    @pytest.mark.asyncio
    async def test_multiple_sentences(self):
        queue = asyncio.Queue()
        text = "Prima. Seconda. Terza. Quarta"
        remainder = await AIAgent._flush_clauses(text, queue)
        assert remainder == "Quarta"
        sentences = []
        while not queue.empty():
            sentences.append(await queue.get())
        assert len(sentences) == 3

    @pytest.mark.asyncio
    async def test_empty_string(self):
        queue = asyncio.Queue()
        remainder = await AIAgent._flush_clauses("", queue)
        assert remainder == ""
        assert queue.empty()


class TestFlushRemainder:
    @pytest.mark.asyncio
    async def test_pushes_text(self):
        queue = asyncio.Queue()
        await AIAgent._flush_remainder("final text", queue)
        assert await queue.get() == "final text"

    @pytest.mark.asyncio
    async def test_strips_whitespace(self):
        queue = asyncio.Queue()
        await AIAgent._flush_remainder("  spaced  ", queue)
        assert await queue.get() == "spaced"

    @pytest.mark.asyncio
    async def test_empty_not_pushed(self):
        queue = asyncio.Queue()
        await AIAgent._flush_remainder("", queue)
        assert queue.empty()

    @pytest.mark.asyncio
    async def test_whitespace_only_not_pushed(self):
        queue = asyncio.Queue()
        await AIAgent._flush_remainder("   ", queue)
        assert queue.empty()


class TestToolRegistration:
    def test_register_tool(self):
        agent = AIAgent(AgentConfig())
        definition = {
            "function": {"name": "test_tool"},
        }
        handler = lambda args: "result"

        agent.register_tool(definition, handler)
        assert len(agent._tools) == 1
        assert "test_tool" in agent._tool_handlers

    def test_register_multiple_tools(self):
        agent = AIAgent(AgentConfig())
        for i in range(3):
            agent.register_tool(
                {"function": {"name": f"tool_{i}"}},
                lambda args, n=i: f"result_{n}",
            )
        assert len(agent._tools) == 3
        assert len(agent._tool_handlers) == 3


class TestExecuteToolCall:
    @pytest.mark.asyncio
    async def test_sync_handler(self):
        agent = AIAgent(AgentConfig())
        agent._tool_handlers["greet"] = lambda args: f"Hello {args.get('name')}"

        result = await agent._execute_tool_call({
            "function": {"name": "greet", "arguments": {"name": "World"}}
        })
        assert result == "Hello World"

    @pytest.mark.asyncio
    async def test_async_handler(self):
        agent = AIAgent(AgentConfig())

        async def async_handler(args):
            await asyncio.sleep(0)
            return f"Async {args.get('val')}"

        agent._tool_handlers["async_tool"] = async_handler

        result = await agent._execute_tool_call({
            "function": {"name": "async_tool", "arguments": {"val": "test"}}
        })
        assert result == "Async test"

    @pytest.mark.asyncio
    async def test_unknown_tool(self):
        agent = AIAgent(AgentConfig())
        result = await agent._execute_tool_call({
            "function": {"name": "unknown", "arguments": {}}
        })
        assert "not available" in result

    @pytest.mark.asyncio
    async def test_handler_exception(self):
        agent = AIAgent(AgentConfig())
        agent._tool_handlers["bad_tool"] = lambda args: 1 / 0

        result = await agent._execute_tool_call({
            "function": {"name": "bad_tool", "arguments": {}}
        })
        assert "Error" in result


class TestBuildPayload:
    def test_basic_payload(self):
        agent = AIAgent(AgentConfig(model="test-model", max_tokens=100))
        messages = [{"role": "user", "content": "ciao"}]
        payload = agent._build_payload(messages)

        assert payload["model"] == "test-model"
        assert payload["messages"] == messages
        assert payload["stream"] is False
        assert payload["options"]["num_predict"] == 100

    def test_streaming_payload(self):
        agent = AIAgent(AgentConfig())
        messages = [{"role": "user", "content": "ciao"}]
        payload = agent._build_payload(messages, stream=True)

        assert payload["stream"] is True

    def test_payload_with_tools(self):
        agent = AIAgent(AgentConfig())
        agent._tools = [{"function": {"name": "test"}}]
        payload = agent._build_payload([])

        assert "tools" in payload
        assert len(payload["tools"]) == 1

    def test_payload_without_tools(self):
        agent = AIAgent(AgentConfig())
        payload = agent._build_payload([])

        assert "tools" not in payload


class TestHistoryTrimming:
    @pytest.mark.asyncio
    async def test_history_trimmed(self):
        agent = AIAgent(AgentConfig())
        agent.initialize()
        agent._max_history = 4

        # Mock the completion to avoid real HTTP calls
        agent._complete_with_tools = AsyncMock(return_value="response")

        for i in range(10):
            await agent.process(f"message {i}")

        # Conversation should be trimmed: last 4 entries + 1 new user + 1 new assistant = trimmed to 4 before adding
        assert len(agent._conversation) <= agent._max_history + 2

        await agent.close()


class TestClearHistory:
    def test_clear(self):
        agent = AIAgent(AgentConfig())
        agent._conversation = [
            {"role": "user", "content": "ciao"},
            {"role": "assistant", "content": "ciao!"},
        ]
        agent.clear_history()
        assert len(agent._conversation) == 0


class TestInitializeClose:
    def test_initialize(self):
        agent = AIAgent(AgentConfig())
        agent.initialize()
        assert agent._client is not None

    @pytest.mark.asyncio
    async def test_close(self):
        agent = AIAgent(AgentConfig())
        agent.initialize()
        await agent.close()
        assert agent._client is None

    @pytest.mark.asyncio
    async def test_process_without_init_raises(self):
        agent = AIAgent(AgentConfig())
        with pytest.raises(RuntimeError, match="not initialized"):
            await agent.process("test")

    @pytest.mark.asyncio
    async def test_process_streaming_without_init_raises(self):
        agent = AIAgent(AgentConfig())
        queue = asyncio.Queue()
        with pytest.raises(RuntimeError, match="not initialized"):
            await agent.process_streaming("test", queue)
