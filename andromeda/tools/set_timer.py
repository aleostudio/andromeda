# Copyright (c) 2026 Alessandro Orrù
# Licensed under MIT

import asyncio
import inspect
import logging
import time
from dataclasses import dataclass, field
from andromeda.messages import msg
from andromeda.storage import SQLiteStore, TimerRecord

logger = logging.getLogger("[ TOOL SET TIMER ]")
audit_logger = logging.getLogger("[ TOOL AUDIT ]")


@dataclass
class TimerEntry:
    timer_id: str
    label: str
    seconds: int
    due_at: float
    task: asyncio.Task

    def cancel(self) -> None:
        self.task.cancel()


@dataclass
class _TimerState:
    feedback: object | None = None
    tts: object | None = None
    store: SQLiteStore | None = None
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


def configure(feedback, max_sec: int, tts=None, store: SQLiteStore | None = None) -> None:
    for timer in _state.active_timers.values():
        timer.cancel()

    _state.active_timers = {}
    _state.feedback = feedback
    _state.tts = tts
    _state.store = store or SQLiteStore(":memory:")
    _state.store.connect()
    _state.max_sec = max_sec


def _format_duration(seconds: int) -> str:
    if seconds >= 60:
        minutes = seconds // 60
        remaining = seconds % 60
        if remaining:
            return msg("timer.duration_minutes_seconds", minutes=minutes, seconds=remaining)

        return msg("timer.duration_minutes", minutes=minutes)

    return msg("timer.duration_seconds", seconds=seconds)


def resume_persisted_timers() -> None:
    _restore_active_timers()


def _store() -> SQLiteStore:
    if _state.store is None:
        _state.store = SQLiteStore(":memory:")
        _state.store.connect()

    return _state.store


def _restore_active_timers() -> None:
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return

    for record in _store().list_active_timers():
        if record.id in _state.active_timers:
            continue
        _schedule_timer(record)


def _schedule_timer(record: TimerRecord) -> None:
    remaining = max(0.0, record.due_at - time.time())
    task = asyncio.get_running_loop().create_task(
        _timer_task(record.id, remaining, record.label, record.duration_sec),
    )
    _state.active_timers[record.id] = TimerEntry(
        timer_id=record.id,
        label=record.label,
        seconds=record.duration_sec,
        due_at=record.due_at,
        task=task,
    )


async def _timer_task(timer_id: str, sleep_seconds: float, label: str, duration_sec: int) -> None:
    try:
        await asyncio.sleep(sleep_seconds)
        logger.info("Timer '%s' (%s) completed", timer_id, label)
        _store().complete_timer(timer_id, time.time())
        audit_logger.info(
            "tool=set_timer event=completed timer_id=%s label=%s seconds=%d",
            timer_id,
            label,
            duration_sec,
        )
        completion_message = msg("timer.completed", label=label)
        await _play_completion_notification(completion_message)

    except asyncio.CancelledError:
        logger.info("Timer '%s' cancelled", timer_id)
        audit_logger.info("tool=set_timer event=cancelled timer_id=%s label=%s", timer_id, label)
        raise
    finally:
        _state.active_timers.pop(timer_id, None)


async def _play_completion_notification(message: str) -> None:
    cue = None
    if _state.feedback:
        cue = lambda: _state.feedback.play_blocking("done")

    speak_notification = getattr(_state.tts, "speak_notification", None)
    if callable(speak_notification):
        result = speak_notification(message, cue=cue, repeats=3, pause_sec=0.2)
        if inspect.isawaitable(result):
            await result
            return

    if cue:
        loop = asyncio.get_running_loop()
        for index in range(3):
            await loop.run_in_executor(None, cue)
            if _state.tts:
                await _state.tts.speak(message)
            if index < 2:
                await asyncio.sleep(0.2)
    elif _state.tts:
        await _state.tts.speak(message)


def _status() -> str:
    _restore_active_timers()
    now = time.time()
    active = [timer for timer in _state.active_timers.values() if not timer.task.done()]
    if not active:
        return msg("timer.none_active")

    active.sort(key=lambda timer: timer.due_at)
    parts = []
    for timer in active:
        remaining = max(1, int(round(timer.due_at - now)))
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

    _restore_active_timers()
    started_at = time.time()
    due_at = started_at + seconds
    timer_id = f"{label}_{time.time_ns()}"
    _store().create_timer(timer_id, label, seconds, started_at, due_at)
    _schedule_timer(
        TimerRecord(
            id=timer_id,
            label=label,
            duration_sec=seconds,
            started_at=started_at,
            due_at=due_at,
            status="active",
            completed_at=None,
        ),
    )
    audit_logger.info("tool=set_timer event=created timer_id=%s label=%s seconds=%d", timer_id, label, seconds)

    return msg("timer.set", label=label, duration=_format_duration(seconds))


def handler(args: dict) -> str:
    action = str(args.get("action", "set")).strip() or "set"
    if action == "status":
        return _status()

    return _set(args)
