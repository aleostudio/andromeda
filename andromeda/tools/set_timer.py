# Copyright (c) 2026 Alessandro Orrù
# Licensed under MIT

import asyncio
import logging
import time
from dataclasses import dataclass, field
from andromeda.messages import msg

logger = logging.getLogger("[ TOOL SET TIMER ]")
audit_logger = logging.getLogger("[ TOOL AUDIT ]")


@dataclass
class TimerEntry:
    timer_id: str
    label: str
    seconds: int
    end_at: float
    task: asyncio.Task

    def cancel(self) -> None:
        self.task.cancel()


@dataclass
class _TimerState:
    feedback: object | None = None
    tts: object | None = None
    max_sec: int = 3600
    active_timers: dict[str, TimerEntry] = field(default_factory=dict)


_state = _TimerState()


DEFINITION = {
    "type": "function",
    "function": {
        "name": "set_timer",
        "description": (
            "Imposta un timer, un conto alla rovescia, oppure controlla quanto manca ai timer attivi. "
            "Usa questo strumento quando l'utente chiede di impostare un timer, "
            "un'allarme, un promemoria a tempo, un conto alla rovescia, "
            "oppure quando chiede quanto manca al timer."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "action": {
                    "type": "string",
                    "enum": ["set", "status"],
                    "description": "set: crea un timer; status: dice quanto manca ai timer attivi",
                    "default": "set",
                },
                "seconds": {
                    "type": "integer",
                    "description": "Durata del timer in secondi (es. 300 per 5 minuti). Richiesto per action=set.",
                },
                "label": {
                    "type": "string",
                    "description": (
                        "Etichetta del timer (es. 'pasta', 'bucato'). "
                        "Richiesta per action=set; se manca, chiedi all'utente: 'Un timer per cosa?'"
                    ),
                },
            },
            "required": [],
        },
    },
}


def configure(feedback, max_sec: int, tts=None) -> None:
    for timer in _state.active_timers.values():
        timer.cancel()

    _state.active_timers = {}
    _state.feedback = feedback
    _state.tts = tts
    _state.max_sec = max_sec


def _format_duration(seconds: int) -> str:
    if seconds >= 60:
        minutes = seconds // 60
        remaining = seconds % 60
        if remaining:
            return msg("timer.duration_minutes_seconds", minutes=minutes, seconds=remaining)

        return msg("timer.duration_minutes", minutes=minutes)

    return msg("timer.duration_seconds", seconds=seconds)


async def _timer_task(timer_id: str, seconds: int, label: str) -> None:
    try:
        await asyncio.sleep(seconds)
        logger.info("Timer '%s' (%s) completed", timer_id, label)
        audit_logger.info("tool=set_timer event=completed timer_id=%s label=%s seconds=%d", timer_id, label, seconds)
        completion_message = msg("timer.completed", label=label)
        if _state.feedback:
            loop = asyncio.get_running_loop()

            # Alternate a cue with a spoken completion message.
            for _ in range(3):
                await loop.run_in_executor(None, _state.feedback.play_blocking, "done")
                if _state.tts:
                    await _state.tts.speak(completion_message)
                await asyncio.sleep(0.2)
        elif _state.tts:
            await _state.tts.speak(completion_message)

    except asyncio.CancelledError:
        logger.info("Timer '%s' cancelled", timer_id)
        audit_logger.info("tool=set_timer event=cancelled timer_id=%s label=%s", timer_id, label)
        raise
    finally:
        _state.active_timers.pop(timer_id, None)


def _status() -> str:
    now = time.monotonic()
    active = [timer for timer in _state.active_timers.values() if not timer.task.done()]
    if not active:
        return msg("timer.none_active")

    active.sort(key=lambda timer: timer.end_at)
    parts = []
    for timer in active:
        remaining = max(1, int(round(timer.end_at - now)))
        parts.append(
            msg("timer.status_item", label=timer.label, duration=_format_duration(remaining)),
        )

    return msg("timer.status_header", timers="; ".join(parts))


def _set(args: dict) -> str:
    label = str(args.get("label", "")).strip()
    if not label:
        return msg("timer.missing_label")

    try:
        seconds = int(args.get("seconds", 0))
    except (TypeError, ValueError):
        return msg("timer.invalid_seconds")

    if seconds <= 0:
        return msg("timer.non_positive")

    if seconds > _state.max_sec:
        return msg("timer.max_exceeded", max_sec=_state.max_sec, max_min=_state.max_sec // 60)

    timer_id = f"{label}_{time.monotonic_ns()}"
    end_at = time.monotonic() + seconds
    task = asyncio.get_running_loop().create_task(_timer_task(timer_id, seconds, label))
    _state.active_timers[timer_id] = TimerEntry(
        timer_id=timer_id,
        label=label,
        seconds=seconds,
        end_at=end_at,
        task=task,
    )
    audit_logger.info("tool=set_timer event=created timer_id=%s label=%s seconds=%d", timer_id, label, seconds)

    return msg("timer.set", label=label, duration=_format_duration(seconds))


def handler(args: dict) -> str:
    action = str(args.get("action", "set")).strip() or "set"
    if action == "status":
        return _status()

    return _set(args)
