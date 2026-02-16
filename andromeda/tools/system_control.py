# Copyright (c) 2026 Alessandro Orrù
# Licensed under MIT

import asyncio
import logging

logger = logging.getLogger(__name__)


# AppleScript commands for macOS system control
_ACTIONS = {
    "volume_up": {
        "script": "set volume output volume ((output volume of (get volume settings)) + 10)",
        "response": "Volume alzato.",
    },
    "volume_down": {
        "script": "set volume output volume ((output volume of (get volume settings)) - 10)",
        "response": "Volume abbassato.",
    },
    "volume_mute": {
        "script": "set volume output muted not (output muted of (get volume settings))",
        "response": "Audio mutato o smutato.",
    },
    "volume_get": {
        "script": "output volume of (get volume settings)",
        "response": None,  # dynamic response from script output
    },
    "brightness_up": {
        "script": "tell application \"System Events\" to key code 144",
        "response": "Luminosità aumentata.",
    },
    "brightness_down": {
        "script": "tell application \"System Events\" to key code 145",
        "response": "Luminosità diminuita.",
    },
}

DEFINITION = {
    "type": "function",
    "function": {
        "name": "system_control",
        "description": (
            "Controlla le impostazioni di sistema del computer: volume e luminosità. "
            "Usa questo strumento quando l'utente chiede di alzare, abbassare o mutare il volume, "
            "o di regolare la luminosità dello schermo."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "action": {
                    "type": "string",
                    "enum": ["volume_up", "volume_down", "volume_mute", "volume_get",
                             "brightness_up", "brightness_down"],
                    "description": (
                        "Azione da eseguire: "
                        "volume_up (alza volume), volume_down (abbassa volume), "
                        "volume_mute (muta/smuta), volume_get (ottieni livello volume), "
                        "brightness_up (aumenta luminosità), brightness_down (diminuisci luminosità)"
                    ),
                },
            },
            "required": ["action"],
        },
    },
}


async def handler(args: dict) -> str:
    action = args.get("action", "")

    action_info = _ACTIONS.get(action)
    if not action_info:
        return f"Azione '{action}' non riconosciuta. Azioni disponibili: {', '.join(_ACTIONS.keys())}"

    try:
        proc = await asyncio.create_subprocess_exec(
            "osascript", "-e", action_info["script"],
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        stdout, stderr = await proc.communicate()

        if proc.returncode != 0:
            error = stderr.decode().strip()
            logger.error("osascript failed: %s", error)
            return f"Errore nell'esecuzione del comando: {error}"

        # For volume_get, return the actual value
        if action_info["response"] is None:
            value = stdout.decode().strip()
            return f"Il volume attuale è al {value} percento."

        return action_info["response"]

    except FileNotFoundError:
        return "Comando osascript non disponibile. Questa funzione richiede macOS."
    except Exception:
        logger.exception("System control failed for action: %s", action)
        return "Errore nel controllo di sistema."
