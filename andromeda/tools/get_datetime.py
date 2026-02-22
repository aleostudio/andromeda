# Copyright (c) 2026 Alessandro Orrù
# Licensed under MIT

import logging
from datetime import datetime

logger = logging.getLogger("[ TOOL GET DATETIME ]")

# Italian day/month names to avoid thread-unsafe locale.setlocale()
_DAYS_IT = ["lunedì", "martedì", "mercoledì", "giovedì", "venerdì", "sabato", "domenica"]
_MONTHS_IT = ["", "gennaio", "febbraio", "marzo", "aprile", "maggio", "giugno", "luglio", "agosto", "settembre", "ottobre", "novembre", "dicembre"]


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
    day_name = _DAYS_IT[now.weekday()]
    month_name = _MONTHS_IT[now.month]

    return f"Data: {day_name} {now.day} {month_name} {now.year}, Ora: {now.strftime('%H:%M')}"
