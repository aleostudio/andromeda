# Copyright (c) 2026 Alessandro Orrù
# Licensed under MIT

import inspect
import logging
import re

logger = logging.getLogger(__name__)


# Fast intent matcher — intercepts simple requests before they reach the LLM.
# Each intent has keyword patterns and a tool handler to call directly.
# Returns None if no intent matched (falls through to LLM).

_intents: list[dict] = []


def register_intent(patterns: list[str], tool_handler: callable, args: dict | None = None) -> None:
    compiled = [re.compile(p, re.IGNORECASE) for p in patterns]
    _intents.append({"patterns": compiled, "handler": tool_handler, "args": args or {}})


async def match_and_execute(text: str) -> str | None:
    text_lower = text.lower().strip()

    for intent in _intents:
        for pattern in intent["patterns"]:
            try:
                if pattern.search(text_lower):
                    logger.info("Fast intent matched: %s", pattern.pattern)
                    return await _run_handler(intent["handler"], intent["args"])
            except Exception:
                logger.exception("Fast intent error: %s", pattern.pattern)
                return None

    return None


async def _run_handler(handler: callable, args: dict) -> str:
    try:
        if inspect.iscoroutinefunction(handler):
            return str(await handler(args))
        return str(handler(args))
    except Exception:
        logger.exception("Fast intent handler failed")

        return "Si è verificato un errore. Riprova."
