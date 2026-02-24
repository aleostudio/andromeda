# Copyright (c) 2026 Alessandro Orrù
# Licensed under MIT

import asyncio
import logging
import platform

logger = logging.getLogger("[ TOOL SYSTEM CONTROL ]")
audit_logger = logging.getLogger("[ TOOL AUDIT ]")


# Constants
_PLATFORM = platform.system()
_VOLUME_INCREASED = "Volume alzato"
_VOLUME_DECREASED = "Volume abbassato"
_VOLUME_TOGGLE = "Audio mutato o smutato"
_BRIGHTNESS_INCREASED = "Luminosità aumentata"
_BRIGHTNESS_DECREASED = "Luminosità diminuita"
_DEFAULT_SINK = "@DEFAULT_SINK@"


# macOS
def _macos_actions() -> dict:
    return {
        "volume_up": {
            "cmd": ["osascript", "-e", "set volume output volume ((output volume of (get volume settings)) + 10)"],
            "response": _VOLUME_INCREASED,
        },
        "volume_down": {
            "cmd": ["osascript", "-e", "set volume output volume ((output volume of (get volume settings)) - 10)"],
            "response": _VOLUME_DECREASED,
        },
        "volume_mute": {
            "cmd": ["osascript", "-e", "set volume output muted not (output muted of (get volume settings))"],
            "response": _VOLUME_TOGGLE,
        },
        "volume_get": {
            "cmd": ["osascript", "-e", "output volume of (get volume settings)"],
            "response": None,
        },
        "brightness_up": {
            "cmd": ["osascript", "-e", 'tell application "System Events" to key code 144'],
            "response": _BRIGHTNESS_INCREASED,
        },
        "brightness_down": {
            "cmd": ["osascript", "-e", 'tell application "System Events" to key code 145'],
            "response": _BRIGHTNESS_DECREASED,
        },
    }


# Linux
def _linux_actions() -> dict:
    return {
        "volume_up": {
            "cmd": ["pactl", "set-sink-volume", _DEFAULT_SINK, "+10%"],
            "response": _VOLUME_INCREASED,
        },
        "volume_down": {
            "cmd": ["pactl", "set-sink-volume", _DEFAULT_SINK, "-10%"],
            "response": _VOLUME_DECREASED,
        },
        "volume_mute": {
            "cmd": ["pactl", "set-sink-mute", _DEFAULT_SINK, "toggle"],
            "response": _VOLUME_TOGGLE,
        },
        "volume_get": {
            "cmd": ["pactl", "get-sink-volume", _DEFAULT_SINK],
            "response": None,
        },
        "brightness_up": {
            "cmd": ["brightnessctl", "set", "+10%"],
            "response": _BRIGHTNESS_INCREASED,
        },
        "brightness_down": {
            "cmd": ["brightnessctl", "set", "10%-"],
            "response": _BRIGHTNESS_DECREASED,
        },
    }


# Windows
def _windows_actions() -> dict:
    # Lightweight CLI tool for Windows: https://www.nirsoft.net/utils/nircmd.html
    return {
        "volume_up": {
            "cmd": ["nircmd", "changesysvolume", "6553"],
            "response": _VOLUME_INCREASED,
        },
        "volume_down": {
            "cmd": ["nircmd", "changesysvolume", "-6553"],
            "response": _VOLUME_DECREASED,
        },
        "volume_mute": {
            "cmd": ["nircmd", "mutesysvolume", "2"],
            "response": _VOLUME_TOGGLE,
        },
        "volume_get": {
            # nircmd doesn't support reading volume; fallback message
            "cmd": None,
            "response": None,
        },
        "brightness_up": {
            "cmd": ["powershell", "-Command",
                    "(Get-WmiObject -Namespace root/WMI -Class WmiMonitorBrightnessMethods).WmiSetBrightness(1, "
                    "([Math]::Min(100, (Get-WmiObject -Namespace root/WMI -Class WmiMonitorBrightness)"
                    ".CurrentBrightness + 10)))"],
            "response": _BRIGHTNESS_INCREASED,
        },
        "brightness_down": {
            "cmd": ["powershell", "-Command",
                    "(Get-WmiObject -Namespace root/WMI -Class WmiMonitorBrightnessMethods).WmiSetBrightness(1, "
                    "([Math]::Max(0, (Get-WmiObject -Namespace root/WMI -Class WmiMonitorBrightness)"
                    ".CurrentBrightness - 10)))"],
            "response": _BRIGHTNESS_DECREASED,
        },
    }


def _get_actions() -> dict:
    if _PLATFORM == "Darwin":
        return _macos_actions()
    if _PLATFORM == "Linux":
        return _linux_actions()
    if _PLATFORM == "Windows":
        return _windows_actions()

    return {}


_ACTIONS = _get_actions()


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
                    "enum": ["volume_up", "volume_down", "volume_mute", "volume_get", "brightness_up", "brightness_down"],
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


async def _run_cmd(cmd: list[str], _action: str) -> tuple[int, str, str]:
    proc = await asyncio.create_subprocess_exec(*cmd, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE)
    stdout, stderr = await proc.communicate()

    return proc.returncode or 0, stdout.decode().strip(), stderr.decode().strip()


def _extract_volume_percent(raw: str) -> str:
    if _PLATFORM != "Linux":
        return raw

    for part in raw.split("/"):
        part = part.strip()
        if part.endswith("%"):
            return part[:-1].strip()

    return raw


def _missing_tool_hint() -> str:
    tool_hint = {
        "Darwin": "osascript (incluso in macOS)",
        "Linux": "pactl (PulseAudio) e brightnessctl",
        "Windows": "nircmd (nirsoft.net)",
    }

    return tool_hint.get(_PLATFORM, "gli strumenti di sistema")


async def handler(args: dict) -> str:
    action = args.get("action", "")

    if not _ACTIONS:
        return f"Controllo di sistema non supportato su questa piattaforma ({_PLATFORM})."

    action_info = _ACTIONS.get(action)
    if not action_info:
        available = ", ".join(_ACTIONS.keys())
        return f"Azione '{action}' non riconosciuta. Azioni disponibili: {available}"

    cmd = action_info.get("cmd")
    if cmd is None:
        return "Questa azione non è disponibile sulla piattaforma corrente."

    try:
        code, out, err = await _run_cmd(cmd, action)

        if code != 0:
            logger.error("System control command failed: %s", err)
            audit_logger.info("tool=system_control action=%s result=error", action)
            return f"Errore nell'esecuzione del comando: {err}"

        response = action_info.get("response")
        if response is not None:
            audit_logger.info("tool=system_control action=%s result=ok", action)
            return response

        value = _extract_volume_percent(out)
        audit_logger.info("tool=system_control action=%s result=ok", action)

        return f"Il volume attuale è al {value} percento."

    except FileNotFoundError:
        audit_logger.info("tool=system_control action=%s result=missing_command", action)
        return f"Comando non trovato. Assicurati di avere installato {_missing_tool_hint()}."
    except Exception:
        logger.exception("System control failed for action: %s", action)
        audit_logger.info("tool=system_control action=%s result=error", action)
        return "Errore nel controllo di sistema."
