# Copyright (c) 2026 Alessandro Orrù
# Licensed under MIT

from andromeda.config import AgentConfig, ConversationConfig
import asyncio
import inspect
import json
import logging
import re
import time
import httpx

logger = logging.getLogger(__name__)


# Conversational AI agent
class AIAgent:

    def __init__(self, agent_cfg: AgentConfig, conversation_cfg: ConversationConfig | None = None) -> None:
        self._cfg = agent_cfg
        self._conversation_cfg = conversation_cfg
        self._client: httpx.AsyncClient | None = None
        self._conversation: list[dict] = []
        self._tools: list[dict] = []
        self._tool_handlers: dict[str, callable] = {}
        self._max_history: int = 20  # Keep last N turns
        self._last_interaction: float = 0.0  # monotonic timestamp of last interaction


    def initialize(self) -> None:
        self._client = httpx.AsyncClient(base_url=self._cfg.base_url, timeout=httpx.Timeout(120.0, connect=10.0))
        logger.info("AI Agent initialized: model=%s, url=%s", self._cfg.model, self._cfg.base_url)


    def register_tool(self, tool_definition: dict, handler: callable) -> None:
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


    # Send user text to Ollama and return response (non-streaming, used as fallback)
    async def process(self, user_text: str) -> str:
        if self._client is None:
            raise RuntimeError("Agent not initialized. Call initialize() first.")

        self._check_history_timeout()
        self._last_interaction = time.monotonic()

        self._conversation.append({"role": "user", "content": user_text})

        # Trim history to prevent context overflow
        if len(self._conversation) > self._max_history:
            self._conversation = self._conversation[-self._max_history:]

        try:
            response_text = await self._complete_with_tools()
            self._conversation.append({"role": "assistant", "content": response_text})
            logger.info("Agent response: %s", response_text[:100])
            return response_text

        except httpx.ConnectError:
            logger.error("Cannot connect to Ollama at %s. Is it running?", self._cfg.base_url)
            return "Non riesco a connettermi al modello. Verifica che Ollama sia in esecuzione."
        except httpx.TimeoutException:
            logger.error("Ollama request timed out")
            return "La richiesta ha impiegato troppo tempo. Riprova."
        except Exception:
            logger.exception("Agent processing failed")
            return "Si è verificato un errore. Riprova."


    # Streaming version: push sentences to queue as they are generated
    async def process_streaming(self, user_text: str, sentence_queue: asyncio.Queue) -> str:
        if self._client is None:
            raise RuntimeError("Agent not initialized. Call initialize() first.")

        self._check_history_timeout()
        self._last_interaction = time.monotonic()

        self._conversation.append({"role": "user", "content": user_text})

        if len(self._conversation) > self._max_history:
            self._conversation = self._conversation[-self._max_history:]

        try:
            # First, handle tool calls (non-streaming, tools need full response)
            full_text = await self._complete_with_tools_streaming(sentence_queue)
            self._conversation.append({"role": "assistant", "content": full_text})
            logger.info("Agent response (streamed): %s", full_text[:100])
            return full_text

        except httpx.ConnectError:
            logger.error("Cannot connect to Ollama at %s. Is it running?", self._cfg.base_url)
            msg = "Non riesco a connettermi al modello. Verifica che Ollama sia in esecuzione."
            await sentence_queue.put(msg)
            return msg
        except httpx.TimeoutException:
            logger.error("Ollama request timed out")
            msg = "La richiesta ha impiegato troppo tempo. Riprova."
            await sentence_queue.put(msg)
            return msg
        except Exception:
            logger.exception("Agent processing failed")
            msg = "Si è verificato un errore. Riprova."
            await sentence_queue.put(msg)
            return msg
        finally:
            # Signal end of stream
            await sentence_queue.put(None)


    # Handle tool calling loop, then stream the final text response
    async def _complete_with_tools_streaming(self, sentence_queue: asyncio.Queue) -> str:
        messages = [
            {"role": "system", "content": self._cfg.system_prompt},
            *self._conversation,
        ]

        # Tool calling loop (non-streaming — tools need the full response to parse)
        for _ in range(5):
            message = await self._chat_request(messages)
            tool_calls = message.get("tool_calls", [])

            if not tool_calls:
                break

            messages.append(message)
            for tool_call in tool_calls:
                result = await self._execute_tool_call(tool_call)
                messages.append({"role": "tool", "content": result})

        # Now stream the final text response sentence by sentence
        return await self._stream_response(messages, sentence_queue)


    # Stream response from Ollama, splitting into sentences and pushing to queue
    async def _stream_response(self, messages: list[dict], sentence_queue: asyncio.Queue) -> str:
        payload = self._build_payload(messages, stream=True)
        full_text = ""
        buffer = ""

        async with self._client.stream("POST", "/api/chat", json=payload) as response:
            response.raise_for_status()
            async for token in self._iter_tokens(response):
                full_text += token
                buffer += token
                buffer = await self._flush_sentences(buffer, sentence_queue)

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


    # Push complete sentences from buffer to queue, return remaining text
    @staticmethod
    async def _flush_sentences(buffer: str, queue: asyncio.Queue) -> str:
        # Regex to split text on sentence boundaries (keeps the delimiter attached)
        _SENTENCE_RE = re.compile(r'(?<=[.!?])\s+')
        sentences = _SENTENCE_RE.split(buffer)

        if len(sentences) <= 1:
            return buffer
        for sentence in sentences[:-1]:
            stripped = sentence.strip()
            if stripped:
                logger.debug("Streaming sentence: %s", stripped)
                await queue.put(stripped)

        return sentences[-1]


    # Push any remaining text in buffer to queue
    @staticmethod
    async def _flush_remainder(buffer: str, queue: asyncio.Queue) -> None:
        remainder = buffer.strip()
        if remainder:
            logger.debug("Streaming final: %s", remainder)
            await queue.put(remainder)


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
                return message.get("content", "").strip()

            messages.append(message)
            for tool_call in tool_calls:
                result = await self._execute_tool_call(tool_call)
                messages.append({"role": "tool", "content": result})

        return "Mi dispiace, la richiesta è troppo complessa. Puoi riformulare?"


    # Send a single chat request to Ollama (non-streaming)
    async def _chat_request(self, messages: list[dict]) -> dict:
        payload = self._build_payload(messages, stream=False)
        response = await self._client.post("/api/chat", json=payload)
        response.raise_for_status()
        return response.json().get("message", {})


    # Parse and execute a single tool call, returning the result string
    async def _execute_tool_call(self, tool_call: dict) -> str:
        func = tool_call.get("function", {})
        func_name = func.get("name", "")
        func_args = self._parse_tool_args(func.get("arguments", {}))

        logger.info("Tool called: %s | args: %s", func_name, json.dumps(func_args, ensure_ascii=False))

        handler = self._tool_handlers.get(func_name)
        if not handler:
            logger.warning("Tool '%s' not registered", func_name)
            return f"Tool '{func_name}' not available"

        try:
            if inspect.iscoroutinefunction(handler):
                result = await handler(func_args)
            else:
                result = handler(func_args)
            logger.info("Tool %s result: %s", func_name, str(result)[:200])
            return str(result)
        except Exception as e:
            logger.exception("Tool %s failed: %s", func_name, e)
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
