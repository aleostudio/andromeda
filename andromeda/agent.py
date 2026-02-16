from andromeda.config import AgentConfig
import asyncio
import inspect
import json
import logging
import httpx

logger = logging.getLogger(__name__)


# Conversational AI agent
class AIAgent:

    def __init__(self, agent_cfg: AgentConfig) -> None:
        self._cfg = agent_cfg
        self._client: httpx.AsyncClient | None = None
        self._conversation: list[dict] = []
        self._tools: list[dict] = []
        self._tool_handlers: dict[str, callable] = {}
        self._max_history: int = 20  # Keep last N turns


    def initialize(self) -> None:
        self._client = httpx.AsyncClient(base_url=self._cfg.base_url, timeout=httpx.Timeout(120.0, connect=10.0))
        logger.info("AI Agent initialized: model=%s, url=%s", self._cfg.model, self._cfg.base_url)


    def register_tool(self, tool_definition: dict, handler: callable) -> None:
        """Register a tool for the agent to use.

        Args:
            tool_definition: Ollama/OpenAI-style tool schema:
                {
                    "type": "function",
                    "function": {
                        "name": "get_weather",
                        "description": "Get weather for a city",
                        "parameters": { ... JSON Schema ... }
                    }
                }
            handler: Async callable that receives tool arguments dict and returns result string
        """
        self._tools.append(tool_definition)
        func_name = tool_definition["function"]["name"]
        self._tool_handlers[func_name] = handler
        logger.info("Registered tool: %s", func_name)


    # Send user text to Ollama and return response
    async def process(self, user_text: str) -> str:
        if self._client is None:
            raise RuntimeError("Agent not initialized. Call initialize() first.")

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


    # Send a single chat request to Ollama
    async def _chat_request(self, messages: list[dict]) -> dict:
        payload: dict = {
            "model": self._cfg.model,
            "messages": messages,
            "stream": False,
            "options": {"num_predict": self._cfg.max_tokens},
        }
        if self._tools:
            payload["tools"] = self._tools

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


    # Close HTTP client
    async def close(self) -> None:
        if self._client:
            await self._client.aclose()
            self._client = None
