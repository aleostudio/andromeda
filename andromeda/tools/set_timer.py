import asyncio
import logging
import time

logger = logging.getLogger(__name__)

_feedback = None
_max_sec: int = 3600
_active_timers: dict[str, asyncio.Task] = {}


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
    global _feedback, _max_sec
    _feedback = feedback
    _max_sec = max_sec


async def _timer_task(timer_id: str, seconds: int, label: str) -> None:
    try:
        await asyncio.sleep(seconds)
        logger.info("Timer '%s' (%s) completed", timer_id, label)
        if _feedback:
            # Play wake sound 3 times as alarm
            for _ in range(3):
                _feedback.play_blocking("wake")
                await asyncio.sleep(0.3)
    except asyncio.CancelledError:
        logger.info("Timer '%s' cancelled", timer_id)
        raise
    finally:
        _active_timers.pop(timer_id, None)


def handler(args: dict) -> str:
    try:
        seconds = int(args.get("seconds", 0))
    except (TypeError, ValueError):
        return "Errore: la durata deve essere un numero intero di secondi."
    label = str(args.get("label", "timer"))

    if seconds <= 0:
        return "Errore: la durata deve essere maggiore di zero."

    if seconds > _max_sec:
        return f"Errore: la durata massima è {_max_sec} secondi ({_max_sec // 60} minuti)."

    timer_id = f"{label}_{int(time.time())}"
    task = asyncio.create_task(_timer_task(timer_id, seconds, label))
    _active_timers[timer_id] = task

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
