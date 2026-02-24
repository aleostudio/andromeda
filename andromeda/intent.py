# Copyright (c) 2026 Alessandro Orrù
# Licensed under MIT

import inspect
import logging
import re
import threading
from collections.abc import Callable
from andromeda.messages import GENERIC_ERROR_RETRY

logger = logging.getLogger("[ INTENT ]")


# Fast intent matcher — intercepts simple requests before they reach the LLM.
# Each intent has keyword patterns and a tool handler to call directly.
# Returns None if no intent matched (falls through to LLM).

_intents: list[dict] = []
_lock = threading.Lock()


def register_intent(patterns: list[str], tool_handler: Callable, args: dict | None = None) -> None:
    compiled = [re.compile(p, re.IGNORECASE) for p in patterns]
    with _lock:
        _intents.append({"patterns": compiled, "handler": tool_handler, "args": args or {}})


def clear_intents() -> None:
    with _lock:
        _intents.clear()


async def match_and_execute(text: str) -> str | None:
    text_lower = text.lower().strip()
    with _lock:
        intents = list(_intents)

    for intent in intents:
        for pattern in intent["patterns"]:
            try:
                if pattern.search(text_lower):
                    logger.info("Fast intent matched: %s", pattern.pattern)
                    return await _run_handler(intent["handler"], intent["args"])
            except Exception:
                logger.exception("Fast intent error: %s", pattern.pattern)
                return None

    return None


async def _run_handler(handler: Callable, args: dict) -> str:
    try:
        if inspect.iscoroutinefunction(handler):
            return str(await handler(args))
        return str(handler(args))
    except Exception:
        logger.exception("Fast intent handler failed")

        return GENERIC_ERROR_RETRY
