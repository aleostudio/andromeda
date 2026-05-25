# Copyright (c) 2026 Alessandro Orrù
# Licensed under MIT

import asyncio
import inspect
import json
import logging
import time
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from andromeda.messages import msg
from andromeda.storage import ScheduledEventRecord, SQLiteStore

logger = logging.getLogger("[ TOOL SCHEDULE EVENT ]")
audit_logger = logging.getLogger("[ TOOL AUDIT ]")


@dataclass
class ScheduledEventTask:
    event_id: str
    title: str
    due_at_epoch: float
    task: asyncio.Task

    def cancel(self) -> None:
        self.task.cancel()


@dataclass
class _ScheduledEventState:
    feedback: object | None = None
    tts: object | None = None
    store: SQLiteStore | None = None
    active_events: dict[str, ScheduledEventTask] = field(default_factory=dict)


_state = _ScheduledEventState()


DEFINITION = {
    "type": "function",
    "function": {
        "name": "schedule_event",
        "description": (
            "Crea, elenca o cancella promemoria ed eventi programmati persistenti. "
            "Usa questo strumento quando l'utente chiede di ricordargli qualcosa a una data/ora, "
            "di creare un promemoria, o di elencare i promemoria attivi."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "action": {
                    "type": "string",
                    "enum": ["set", "list", "delete"],
                    "description": "set: crea un promemoria; list: elenca quelli attivi; delete: cancella per id",
                    "default": "set",
                },
                "title": {
                    "type": "string",
                    "description": "Titolo del promemoria, ad esempio 'chiamare Marco'. Richiesto per action=set.",
                },
                "due_at": {
                    "type": "string",
                    "description": (
                        "Data/ora ISO 8601 del promemoria. Esempio: "
                        "'2026-05-24T18:30:00+02:00'. Richiesto se non viene passato seconds."
                    ),
                },
                "seconds": {
                    "type": "integer",
                    "description": "Secondi da ora per promemoria relativi. Alternativa a due_at.",
                },
                "id": {
                    "type": "string",
                    "description": "Id del promemoria da cancellare. Richiesto per action=delete.",
                },
                "payload": {
                    "type": "object",
                    "description": "Dati opzionali associati al promemoria.",
                },
            },
            "required": [],
        },
    },
}


def configure(feedback, tts=None, store: SQLiteStore | None = None) -> None:
    for event in _state.active_events.values():
        event.cancel()

    _state.active_events = {}
    _state.feedback = feedback
    _state.tts = tts
    _state.store = store or SQLiteStore(":memory:")
    _state.store.connect()


def resume_persisted_events() -> None:
    _restore_active_events()


def _store() -> SQLiteStore:
    if _state.store is None:
        _state.store = SQLiteStore(":memory:")
        _state.store.connect()

    return _state.store


def _parse_due_at(args: dict) -> datetime | None:
    seconds = args.get("seconds")
    if seconds is not None:
        try:
            seconds_int = int(seconds)
        except (TypeError, ValueError):
            return None
        if seconds_int <= 0:
            return None
        return datetime.now(UTC) + timedelta(seconds=seconds_int)

    due_at = str(args.get("due_at", "")).strip()
    if not due_at:
        return None

    try:
        parsed = datetime.fromisoformat(due_at.replace("Z", "+00:00"))
    except ValueError:
        return None

    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=datetime.now().astimezone().tzinfo)

    return parsed.astimezone(UTC)


def _format_due_at(due_at: datetime) -> str:
    local = due_at.astimezone()
    return local.strftime("%d/%m/%Y %H:%M")


def _record_due_at(record: ScheduledEventRecord) -> datetime:
    return datetime.fromisoformat(record.due_at.replace("Z", "+00:00")).astimezone(UTC)


def _restore_active_events() -> None:
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return

    _delete_expired_events()
    for record in _store().list_active_scheduled_events():
        if record.id in _state.active_events:
            continue
        _schedule_event(record)


def _delete_expired_events() -> None:
    deleted = _store().delete_expired_scheduled_events(
        datetime.now(UTC).isoformat(),
        exclude_ids=set(_state.active_events),
    )
    if deleted:
        logger.info("Deleted %d expired scheduled events", deleted)
        audit_logger.info("tool=schedule_event event=expired_deleted count=%d", deleted)


def _schedule_event(record: ScheduledEventRecord) -> None:
    due_at = _record_due_at(record)
    due_at_epoch = due_at.timestamp()
    delay = max(0.0, due_at_epoch - time.time())
    task = asyncio.get_running_loop().create_task(_event_task(record.id, delay, record.title))
    _state.active_events[record.id] = ScheduledEventTask(
        event_id=record.id,
        title=record.title,
        due_at_epoch=due_at_epoch,
        task=task,
    )


async def _event_task(event_id: str, delay: float, title: str) -> None:
    fired = False
    try:
        await asyncio.sleep(delay)
        fired = True
        audit_logger.info("tool=schedule_event event=fired event_id=%s title=%s", event_id, title)
        completion_message = msg("event.completed", title=title)
        await _play_event_notification(completion_message)
    except asyncio.CancelledError:
        logger.info("Scheduled event '%s' cancelled", event_id)
        audit_logger.info("tool=schedule_event event=task_cancelled event_id=%s title=%s", event_id, title)
        raise
    except Exception:
        logger.exception("Scheduled event '%s' failed while firing", event_id)
    finally:
        if fired:
            _store().complete_scheduled_event(event_id)
            audit_logger.info("tool=schedule_event event=record_deleted event_id=%s title=%s", event_id, title)
        _state.active_events.pop(event_id, None)


async def _play_event_notification(message: str) -> None:
    cue = None
    if _state.feedback:
        cue = lambda: _state.feedback.play_blocking("done")

    speak_notification = getattr(_state.tts, "speak_notification", None)
    if callable(speak_notification):
        result = speak_notification(message, cue=cue)
        if inspect.isawaitable(result):
            await result
            return

    if cue:
        loop = asyncio.get_running_loop()
        await loop.run_in_executor(None, cue)
    if _state.tts:
        await _state.tts.speak(message)


def _set(args: dict) -> str:
    title = str(args.get("title", "")).strip()
    if not title:
        return msg("event.missing_title")

    due_at = _parse_due_at(args)
    if due_at is None:
        if not str(args.get("due_at", "")).strip() and args.get("seconds") is None:
            return msg("event.missing_due_at")
        return msg("event.invalid_due_at")

    if due_at <= datetime.now(UTC):
        return msg("event.past_due_at")

    payload = args.get("payload")
    payload_json = json.dumps(payload, ensure_ascii=False) if payload is not None else None
    event_id = f"event_{time.time_ns()}"
    due_at_text = due_at.isoformat()
    _store().create_scheduled_event(event_id, title, due_at_text, payload_json=payload_json)
    _restore_active_events()
    audit_logger.info("tool=schedule_event event=created event_id=%s title=%s due_at=%s", event_id, title, due_at_text)

    return msg("event.set", title=title, due_at=_format_due_at(due_at))


def _list() -> str:
    _restore_active_events()
    _delete_expired_events()
    records = _store().list_active_scheduled_events()
    if not records:
        return msg("event.none_active")

    parts = [
        msg("event.list_item", id=record.id, title=record.title, due_at=_format_due_at(_record_due_at(record)))
        for record in records
    ]
    return msg("event.list_header", events="; ".join(parts))


def _delete(args: dict) -> str:
    event_id = str(args.get("id", "")).strip()
    if not event_id:
        return msg("event.delete_missing_id")

    task = _state.active_events.pop(event_id, None)
    if task is not None:
        task.cancel()

    if not _store().cancel_scheduled_event(event_id):
        return msg("event.delete_not_found")

    audit_logger.info("tool=schedule_event event=cancelled event_id=%s", event_id)
    return msg("event.deleted")


def handler(args: dict) -> str:
    action = str(args.get("action", "set")).strip() or "set"
    if action == "set":
        return _set(args)
    if action == "list":
        return _list()
    if action == "delete":
        return _delete(args)

    return msg("event.invalid_action", action=action)
