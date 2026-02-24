# Copyright (c) 2026 Alessandro Orrù
# Licensed under MIT

import asyncio
import contextlib
import inspect
import json
import logging
import re
import time
import httpx
from collections.abc import Callable
from andromeda.config import AgentConfig, ConversationConfig
from andromeda.messages import (GENERIC_ERROR_RETRY, OLLAMA_TIMEOUT,
                                OLLAMA_UNREACHABLE, REQUEST_TOO_COMPLEX)

logger = logging.getLogger("[ AGENT ]")

# Common Italian abbreviations that should NOT trigger sentence splitting
_ABBREV = r'(?<![Dd]ott)(?<![Ss]ig)(?<![Pp]rof)(?<![Ii]ng)(?<![Aa]vv)(?<![Gg]en)(?<![Ee]cc)(?<![Ee]s)'

# Regex to split on sentence endings AND mid-sentence pauses (commas, semicolons, colons)
# Uses (?<=\D\.) to avoid splitting on numbered list items like "2. Taglia..."
_SENTENCE_RE = re.compile(rf'{_ABBREV}(?<=\D\.)\s+|(?<=[!?])\s+')
_CLAUSE_RE = re.compile(rf'{_ABBREV}(?<=\D\.)\s+|(?<=[!?;:,])\s+')

# Minimum clause length to push to TTS (avoid very short fragments)
_MIN_CLAUSE_LEN = 20

# Strip markdown: inline formatting (bold/italic/code/strikethrough)
_MARKDOWN_INLINE_RE = re.compile(r'[*_`~]+')

# Strip markdown: headers and list prefixes (line-start patterns)
_MARKDOWN_BLOCK_RE = re.compile(r'^\s*(?:#{1,6}\s+|[-*+]\s+|\d+\.\s+)', re.MULTILINE)

# Constants to avoid repeats
_MODEL_CHAT_PATH = "/api/chat"
_AGENT_NOT_INITIALIZED_ERR = "Agent not initialized. Call initialize() first."
_QUEUE_PUT_TIMEOUT_SEC = 1.0

# Conversational AI agent
class AIAgent:

    def __init__(self, agent_cfg: AgentConfig, conversation_cfg: ConversationConfig | None = None) -> None:
        self._cfg = agent_cfg
        self._conversation_cfg = conversation_cfg
        self._client: httpx.AsyncClient | None = None
        self._conversation: list[dict] = []
        self._tools: list[dict] = []
        self._tool_handlers: dict[str, Callable] = {}
        self._max_history: int = conversation_cfg.max_history if conversation_cfg else 20
        self._last_interaction: float = 0.0  # monotonic timestamp of last interaction
        self._compaction_threshold: int = conversation_cfg.compaction_threshold if conversation_cfg else 16


    def initialize(self) -> None:
        self._client = httpx.AsyncClient(base_url=self._cfg.base_url, timeout=httpx.Timeout(self._cfg.timeout_sec, connect=10.0))
        logger.info("AI Agent initialized: model=%s, url=%s", self._cfg.model, self._cfg.base_url)


    # Pre-warm the Ollama model by sending a minimal request to load it into memory
    async def prewarm_model(self) -> None:
        if self._client is None:
            raise RuntimeError(_AGENT_NOT_INITIALIZED_ERR)

        try:
            logger.info("Pre-warming Ollama model: %s...", self._cfg.model)
            response = await self._client.post(_MODEL_CHAT_PATH, json={
                "model": self._cfg.model,
                "messages": [{"role": "user", "content": "ciao"}],
                "stream": False,
                "options": {"num_predict": 1},
            })
            response.raise_for_status()
            logger.info("Ollama model pre-warmed successfully")
        except httpx.ConnectError:
            logger.warning("Cannot pre-warm model: Ollama not reachable at %s", self._cfg.base_url)
        except Exception:
            logger.warning("Failed to pre-warm Ollama model", exc_info=True)


    def register_tool(self, tool_definition: dict, handler: Callable) -> None:
        self._tools.append(tool_definition)
        func_name = tool_definition["function"]["name"]
        self._tool_handlers[func_name] = handler
        logger.info("Registered tool: %s", func_name)


    # Clear stale history if user has been inactive too long
    def _check_history_timeout(self) -> None:
        if not self._conversation_cfg or self._conversation_cfg.history_timeout_sec <= 0:
            return
        if self._last_interaction == 0:
            return

        elapsed = time.monotonic() - self._last_interaction
        if elapsed > self._conversation_cfg.history_timeout_sec:
            logger.info("History timeout (%.0fs idle), clearing conversation", elapsed)
            self._conversation.clear()


    # Compact old conversation history into a summary to preserve context without overflowing
    async def _compact_history_if_needed(self) -> None:
        if len(self._conversation) < self._compaction_threshold:
            return
        if self._client is None:
            return

        # Keep the most recent 4 turns intact, summarize the rest
        turns_to_compact = self._conversation[:-4]
        recent_turns = self._conversation[-4:]

        if len(turns_to_compact) < 4:
            return  # Not enough to compact

        try:
            compact_text = "\n".join(
                f"{m['role']}: {m['content'][:200]}" for m in turns_to_compact if m.get("content")
            )
            summary_response = await self._client.post(_MODEL_CHAT_PATH, json={
                "model": self._cfg.model,
                "messages": [
                    {
                        "role": "system",
                        "content": "Riassumi brevemente questa conversazione in 2-3 frasi in italiano. Cattura solo i punti chiave e le informazioni importanti."
                    },
                    {
                        "role": "user",
                        "content": compact_text
                    },
                ],
                "stream": False,
                "options": {"num_predict": 150},
            })
            summary_response.raise_for_status()
            summary = summary_response.json().get("message", {}).get("content", "").strip()

            if summary:
                self._conversation = [
                    {
                        "role": "system",
                        "content": f"[Riepilogo conversazione precedente: {summary}]"
                    },
                    *recent_turns,
                ]
                logger.info("History compacted: %d turns -> summary + %d recent", len(turns_to_compact), len(recent_turns))
            else:
                # Fallback: just trim
                self._conversation = recent_turns

        except Exception:
            logger.warning("History compaction failed, trimming instead", exc_info=True)
            self._conversation = recent_turns


    # Send user text to Ollama and return response (non-streaming, used as fallback)
    async def process(self, user_text: str) -> str:
        if self._client is None:
            raise RuntimeError(_AGENT_NOT_INITIALIZED_ERR)

        self._check_history_timeout()
        self._last_interaction = time.monotonic()

        self._conversation.append({"role": "user", "content": user_text})

        # Compact history if it's getting long
        await self._compact_history_if_needed()

        # Trim history to prevent context overflow
        if len(self._conversation) > self._max_history:
            self._conversation = self._conversation[-self._max_history:]

        try:
            response_text = await self._complete_with_tools()
            self._conversation.append({"role": "assistant", "content": response_text})
            logger.info("Agent response: %s", " ".join(response_text[:200].split()))
            return response_text

        except httpx.ConnectError:
            logger.error("Cannot connect to Ollama at %s. Is it running?", self._cfg.base_url)
            return OLLAMA_UNREACHABLE
        except httpx.TimeoutException:
            logger.error("Ollama request timed out")
            return OLLAMA_TIMEOUT
        except Exception:
            logger.exception("Agent processing failed")
            return GENERIC_ERROR_RETRY


    # Streaming version: push sentences to queue as they are generated
    async def process_streaming(self, user_text: str, sentence_queue: asyncio.Queue) -> str:
        if self._client is None:
            raise RuntimeError(_AGENT_NOT_INITIALIZED_ERR)

        self._check_history_timeout()
        self._last_interaction = time.monotonic()

        self._conversation.append({"role": "user", "content": user_text})

        # Compact history if it's getting long
        await self._compact_history_if_needed()

        if len(self._conversation) > self._max_history:
            self._conversation = self._conversation[-self._max_history:]

        try:
            # First, handle tool calls (non-streaming, tools need full response)
            full_text = await self._complete_with_tools_streaming(sentence_queue)
            self._conversation.append({"role": "assistant", "content": full_text})
            logger.info("Agent response (streamed): %s", " ".join(full_text[:200].split()))
            return full_text

        except httpx.ConnectError:
            logger.error("Cannot connect to Ollama at %s. Is it running?", self._cfg.base_url)
            msg = OLLAMA_UNREACHABLE
            await self._queue_put(sentence_queue, msg)
            return msg
        except httpx.TimeoutException:
            logger.error("Ollama request timed out")
            msg = OLLAMA_TIMEOUT
            await self._queue_put(sentence_queue, msg)
            return msg
        except Exception:
            logger.exception("Agent processing failed")
            msg = GENERIC_ERROR_RETRY
            await self._queue_put(sentence_queue, msg)
            return msg
        finally:
            # Signal end of stream
            try:
                sentence_queue.put_nowait(None)
            except asyncio.QueueFull:
                with contextlib.suppress(asyncio.QueueEmpty):
                    sentence_queue.get_nowait()
                with contextlib.suppress(asyncio.QueueFull):
                    sentence_queue.put_nowait(None)


    # Handle tool calling loop, then stream the final text response
    async def _complete_with_tools_streaming(self, sentence_queue: asyncio.Queue) -> str:
        messages = [
            {"role": "system", "content": self._cfg.system_prompt},
            *self._conversation,
        ]

        # Tool calling loop (non-streaming — tools need the full response to parse)
        used_tools = False
        for _ in range(5):
            message = await self._chat_request(messages)
            tool_calls = message.get("tool_calls", [])

            if not tool_calls:
                break

            used_tools = True
            messages.append(message)
            for tool_call in tool_calls:
                result = await self._execute_tool_call(tool_call)
                messages.append({"role": "tool", "content": result})

        if used_tools:
            # Tools were called — stream a fresh response that incorporates tool results
            return await self._stream_response(messages, sentence_queue)

        # No tools — use the text we already received (avoids a duplicate HTTP call)
        text = message.get("content", "").strip()
        await self._enqueue_text(text, sentence_queue)
        return text


    # Split already-complete text into sentences and push to queue (no HTTP call)
    @staticmethod
    async def _enqueue_text(text: str, queue: asyncio.Queue) -> None:
        if not text:
            return
        sentences = _SENTENCE_RE.split(text)
        for sentence in sentences:
            stripped = AIAgent._strip_markdown(sentence)
            if stripped:
                await AIAgent._queue_put(queue, stripped)


    # Stream response from Ollama, splitting into clauses and pushing to queue
    async def _stream_response(self, messages: list[dict], sentence_queue: asyncio.Queue) -> str:
        payload = self._build_payload(messages, stream=True)
        full_text = ""
        buffer = ""

        try:
            async with self._client.stream("POST", _MODEL_CHAT_PATH, json=payload) as response:
                response.raise_for_status()
                async for token in self._iter_tokens(response):
                    full_text += token
                    buffer += token
                    buffer = await self._flush_clauses(buffer, sentence_queue)
        except Exception:
            logger.exception("Error during response streaming")
            if not full_text:
                raise  # Re-raise so caller sends error to queue

        await self._flush_remainder(buffer, sentence_queue)

        return full_text


    # Build Ollama API payload
    def _build_payload(self, messages: list[dict], stream: bool = False) -> dict:
        payload: dict = {
            "model": self._cfg.model,
            "messages": messages,
            "stream": stream,
            "options": {"num_predict": self._cfg.max_tokens},
        }
        if self._tools:
            payload["tools"] = self._tools

        return payload


    # Yield content tokens from a streaming response, skipping empty/invalid chunks
    @staticmethod
    async def _iter_tokens(response: httpx.Response):
        async for line in response.aiter_lines():
            if not line.strip():
                continue
            try:
                chunk = json.loads(line)
            except json.JSONDecodeError:
                continue
            token = chunk.get("message", {}).get("content", "")
            if token:
                yield token


    # Push complete clauses from buffer to queue for lower latency TTS
    # Falls back to sentence-level splitting for short text
    @staticmethod
    async def _flush_clauses(buffer: str, queue: asyncio.Queue) -> str:
        # First try sentence boundaries (highest quality split)
        sentences = _SENTENCE_RE.split(buffer)
        if len(sentences) > 1:
            await AIAgent._enqueue_parts(sentences[:-1], queue, "sentence")
            return sentences[-1]

        # If buffer is short, keep accumulating
        if len(buffer) <= _MIN_CLAUSE_LEN * 2:
            return buffer

        # Try clause-level split for lower latency on long buffers
        return await AIAgent._try_clause_split(buffer, queue)


    # Strip markdown formatting from text for clean TTS output
    @staticmethod
    def _strip_markdown(text: str) -> str:
        text = _MARKDOWN_BLOCK_RE.sub('', text)
        return _MARKDOWN_INLINE_RE.sub('', text).strip()


    # Enqueue non-empty stripped parts to the TTS queue
    @staticmethod
    async def _enqueue_parts(parts: list[str], queue: asyncio.Queue, label: str) -> None:
        for part in parts:
            stripped = AIAgent._strip_markdown(part)
            if stripped:
                logger.debug("Streaming %s: %s", label, stripped)
                await AIAgent._queue_put(queue, stripped)


    # Try to split buffer at clause boundaries; return remainder or original buffer
    @staticmethod
    async def _try_clause_split(buffer: str, queue: asyncio.Queue) -> str:
        clauses = _CLAUSE_RE.split(buffer)
        first_clause = AIAgent._strip_markdown(clauses[0]) if len(clauses) > 1 else ""

        if not first_clause or len(first_clause) < _MIN_CLAUSE_LEN:
            return buffer

        logger.debug("Streaming clause: %s", first_clause)
        await AIAgent._queue_put(queue, first_clause)

        remainder = _CLAUSE_RE.split(buffer, maxsplit=1)

        return remainder[1] if len(remainder) > 1 else buffer


    # Push any remaining text in buffer to queue
    @staticmethod
    async def _flush_remainder(buffer: str, queue: asyncio.Queue) -> None:
        remainder = AIAgent._strip_markdown(buffer)
        if remainder:
            logger.debug("Streaming final: %s", remainder)
            await AIAgent._queue_put(queue, remainder)


    @staticmethod
    async def _queue_put(queue: asyncio.Queue, value: str) -> None:
        try:
            await asyncio.wait_for(queue.put(value), timeout=_QUEUE_PUT_TIMEOUT_SEC)
        except asyncio.TimeoutError:
            logger.warning("Streaming queue is full, dropping chunk")


    # Run completion loop, handling tool calls until we get a text response
    async def _complete_with_tools(self) -> str:
        messages = [
            {"role": "system", "content": self._cfg.system_prompt},
            *self._conversation,
        ]

        for _ in range(5):  # Max tool call iterations
            message = await self._chat_request(messages)
            tool_calls = message.get("tool_calls", [])

            if not tool_calls:
                return self._strip_markdown(message.get("content", ""))

            messages.append(message)
            for tool_call in tool_calls:
                result = await self._execute_tool_call(tool_call)
                messages.append({"role": "tool", "content": result})

        return REQUEST_TOO_COMPLEX


    # Send a single chat request to Ollama (non-streaming)
    async def _chat_request(self, messages: list[dict]) -> dict:
        payload = self._build_payload(messages, stream=False)
        response = await self._client.post(_MODEL_CHAT_PATH, json=payload)
        response.raise_for_status()
        try:
            return response.json().get("message", {})
        except (ValueError, AttributeError):
            logger.error("Invalid JSON response from Ollama")
            return {}


    # Parse and execute a single tool call, returning the result string
    async def _execute_tool_call(self, tool_call: dict) -> str:
        func = tool_call.get("function", {})
        func_name = func.get("name", "")
        func_args = self._parse_tool_args(func.get("arguments", {}))

        logger.info("[ TOOL ] called: %s | args: %s", func_name, json.dumps(func_args, ensure_ascii=False))

        handler = self._tool_handlers.get(func_name)
        if not handler:
            logger.warning("[ TOOL ] '%s' not registered", func_name)
            return f"[ TOOL ] '{func_name}' not available"

        try:
            if inspect.iscoroutinefunction(handler):
                result = await handler(func_args)
            else:
                result = handler(func_args)
            logger.info("[ TOOL ] %s result: %s", func_name, str(result)[:200])
            return str(result)
        except Exception as e:
            logger.exception("[ TOOL ] %s failed: %s", func_name, e)
            return f"Error: {e}"


    # Parse tool arguments (may come as string or dict from Ollama)
    @staticmethod
    def _parse_tool_args(args: dict | str) -> dict:
        if isinstance(args, str):
            try:
                return json.loads(args)
            except json.JSONDecodeError:
                return {}

        return args


    # Clear conversation history
    def clear_history(self) -> None:
        self._conversation.clear()
        logger.info("Conversation history cleared")


    # Close HTTP client
    async def close(self) -> None:
        if self._client:
            await self._client.aclose()
            self._client = None
