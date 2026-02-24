# Copyright (c) 2026 Alessandro Orrù
# Licensed under MIT

import logging
from datetime import datetime
from andromeda.messages import get_localized_datetime, msg

logger = logging.getLogger("[ TOOL GET DATETIME ]")


DEFINITION = {
    "type": "function",
    "function": {
        "name": "get_datetime",
        "description": (
            "Ottieni la data e l'ora corrente. "
            "Usa questo strumento quando l'utente chiede che giorno è, che ora è, "
            "o qualsiasi informazione su data e orario."
        ),
        "parameters": {
            "type": "object",
            "properties": {},
            "required": [],
        },
    },
}


def handler(_args: dict) -> str:
    now = datetime.now()
    date_text, time_text = get_localized_datetime(now)

    return msg("datetime.output", date=date_text, time=time_text)
