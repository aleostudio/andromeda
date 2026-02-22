# Copyright (c) 2026 Alessandro Orrù
# Licensed under MIT

import locale
import logging
from datetime import datetime

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
    try:
        locale.setlocale(locale.LC_TIME, "it_IT.UTF-8")
    except locale.Error:
        pass

    now = datetime.now()

    return f"Data: {now.strftime('%A %d %B %Y')}, Ora: {now.strftime('%H:%M')}"
