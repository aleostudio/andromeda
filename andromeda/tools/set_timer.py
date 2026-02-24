# Copyright (c) 2026 Alessandro Orrù
# Licensed under MIT

import asyncio
import logging
import time
from dataclasses import dataclass, field

logger = logging.getLogger("[ TOOL SET TIMER ]")
audit_logger = logging.getLogger("[ TOOL AUDIT ]")


@dataclass
class _TimerState:
    feedback: object | None = None
    max_sec: int = 3600
    active_timers: dict[str, asyncio.Task] = field(default_factory=dict)


_state = _TimerState()


DEFINITION = {
    "type": "function",
    "function": {
        "name": "set_timer",
        "description": (
            "Imposta un timer o un conto alla rovescia. "
            "Usa questo strumento quando l'utente chiede di impostare un timer, "
            "un'allarme, un promemoria a tempo, o un conto alla rovescia."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "seconds": {
                    "type": "integer",
                    "description": "Durata del timer in secondi (es. 300 per 5 minuti)",
                },
                "label": {
                    "type": "string",
                    "description": "Etichetta opzionale per il timer (es. 'pasta', 'bucato')",
                },
            },
            "required": ["seconds"],
        },
    },
}


def configure(feedback, max_sec: int) -> None:
    for task in _state.active_timers.values():
        task.cancel()
    _state.active_timers = {}
    _state.feedback = feedback
    _state.max_sec = max_sec


async def _timer_task(timer_id: str, seconds: int, label: str) -> None:
    try:
        await asyncio.sleep(seconds)
        logger.info("Timer '%s' (%s) completed", timer_id, label)
        audit_logger.info("tool=set_timer event=completed timer_id=%s label=%s seconds=%d", timer_id, label, seconds)
        if _state.feedback:
            loop = asyncio.get_running_loop()
            # Play wake sound 3 times as alarm (in executor to avoid blocking event loop)
            for _ in range(3):
                await loop.run_in_executor(None, _state.feedback.play_blocking, "wake")
                await asyncio.sleep(0.3)
    except asyncio.CancelledError:
        logger.info("Timer '%s' cancelled", timer_id)
        audit_logger.info("tool=set_timer event=cancelled timer_id=%s label=%s", timer_id, label)
        raise
    finally:
        _state.active_timers.pop(timer_id, None)


def handler(args: dict) -> str:
    try:
        seconds = int(args.get("seconds", 0))
    except (TypeError, ValueError):
        return "Errore: la durata deve essere un numero intero di secondi."
    label = str(args.get("label", "timer"))

    if seconds <= 0:
        return "Errore: la durata deve essere maggiore di zero."

    if seconds > _state.max_sec:
        return f"Errore: la durata massima è {_state.max_sec} secondi ({_state.max_sec // 60} minuti)."

    timer_id = f"{label}_{time.monotonic_ns()}"
    task = asyncio.get_running_loop().create_task(_timer_task(timer_id, seconds, label))
    _state.active_timers[timer_id] = task
    audit_logger.info("tool=set_timer event=created timer_id=%s label=%s seconds=%d", timer_id, label, seconds)

    if seconds >= 60:
        minutes = seconds // 60
        remaining = seconds % 60
        if remaining:
            duration_str = f"{minutes} minuti e {remaining} secondi"
        else:
            duration_str = f"{minutes} minuti"
    else:
        duration_str = f"{seconds} secondi"

    return f"Timer '{label}' impostato per {duration_str}."
