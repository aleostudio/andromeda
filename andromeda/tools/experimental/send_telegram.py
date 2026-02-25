# Copyright (c) 2026 Alessandro Orrù
# Licensed under MIT

import os
import json
import logging
from typing import Any, Dict, Optional

import requests

logger = logging.getLogger("[ TOOL TELEGRAM SEND MESSAGE ]")

# HOW TO CREATE A TELEGRAM BOT THAT WILL WRITE TO YOU THROUGH ANDROMEDA
# ==============================================================================
# 1.  Open Telegram
# 2.  Search for "BotFather"
# 3.  Init chat with him and click or type "/start"
# 4.  Type "/newbot"
# 5.  Choose a name (e.g. "Andromeda")
# 6.  Choose a unique username (e.g. "andromeda_something_bot" -> must finish with "_bot")
# 7.  BotFather will return a token like "123456789:AAAbbbCCCdddEEEfffGGG": this is your BOT_TOKEN
# 8.  Inside BotFather chat, write something (e.g. "hello")
# 9.  Now open in your browser this url: https://api.telegram.org/bot<TOKEN>/getUpdates
# 10. Change <TOKEN> with your bot token (pay attention to not delete /bot)
# 11. Write something else on your phone and then refresh the page on the browser
# 12. You will see a message like: {"ok":true,"result":[{"message":{"message_id":6,"from":{"id":12345678,"is_bot":false, ...
# 13. Look at "from":{"id": 12345678": this number is your user id (CHAT_ID)
# 14. Launch Andromeda and tell her: send a telegram message with "your text"
# ==============================================================================

# Just as example to test the tool: move them in env vars (TELEGRAM_BOT_TOKEN, _TELEGRAM_DEFAULT_CHAT_ID)
_TELEGRAM_BOT_TOKEN = "123456789:AAAbbbCCCdddEEEfffGGG"
_TELEGRAM_DEFAULT_CHAT_ID = "12345678"


DEFINITION = {
    "type": "function",
    "function": {
        "name": "telegram_send_message",
        "description": (
            "Invia un messaggio Telegram tramite Bot API (sendMessage). "
            "Usa questo tool quando devi notificare l'utente o inviare un aggiornamento su Telegram."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "chat_id": {
                    "type": "string",
                    "description": (
                        "ID della chat (numero come stringa) oppure username canale tipo '@channelusername'. "
                        "Se omesso, usa TELEGRAM_CHAT_ID dall'env."
                    ),
                },
                "text": {
                    "type": "string",
                    "description": "Testo del messaggio da inviare (1-4096 caratteri).",
                },
                "parse_mode": {
                    "type": "string",
                    "enum": ["", "MarkdownV2", "HTML", "Markdown"],
                    "description": "Modalità di formattazione. Consigliato: MarkdownV2 o HTML. Vuoto = nessuna.",
                    "default": "",
                },
                "disable_notification": {
                    "type": "boolean",
                    "description": "Se true, invia il messaggio in modalità silenziosa.",
                    "default": False,
                },
                "protect_content": {
                    "type": "boolean",
                    "description": "Se true, protegge il contenuto da inoltro/salvataggio (dove supportato).",
                    "default": False,
                },
                "message_thread_id": {
                    "type": "integer",
                    "description": "ID del topic/thread (per forum supergroup o chat con topics).",
                },
                "reply_to_message_id": {
                    "type": "integer",
                    "description": "Se presente, prova a rispondere a quel messaggio (semplice).",
                },
            },
            "required": ["text"],
            "additionalProperties": False,
        },
    },
}


def _ok(payload: dict) -> str:
    return json.dumps({"ok": True, **payload}, ensure_ascii=False)


def _err(message: str, details: Optional[dict] = None) -> str:
    body = {"ok": False, "error": message}
    if details:
        body["details"] = details

    return json.dumps(body, ensure_ascii=False)


def _get_token() -> str:
    token = os.getenv("TELEGRAM_BOT_TOKEN", _TELEGRAM_BOT_TOKEN).strip()
    if not token:
        raise RuntimeError("Missing env var TELEGRAM_BOT_TOKEN")
    return token


def handler(_args: dict) -> str:
    try:
        token = _get_token()
        chat_id = (_args.get("chat_id") or os.getenv("TELEGRAM_CHAT_ID", _TELEGRAM_DEFAULT_CHAT_ID)).strip()
        text = (_args.get("text") or "").strip()

        if not chat_id:
            return _err("Missing chat_id (provide it or set env TELEGRAM_CHAT_ID)")
        if not text:
            return _err("Missing text")

        url = f"https://api.telegram.org/bot{token}/sendMessage"

        payload: Dict[str, Any] = {
            "chat_id": chat_id,
            "text": text,
            "disable_notification": bool(_args.get("disable_notification", False)),
            "protect_content": bool(_args.get("protect_content", False)),
        }

        parse_mode = (_args.get("parse_mode") or "").strip()
        if parse_mode:
            payload["parse_mode"] = parse_mode

        if _args.get("message_thread_id") is not None:
            payload["message_thread_id"] = int(_args["message_thread_id"])

        # Semplificazione: reply_to_message_id -> usa reply_parameters
        # (compatibile con Bot API moderna, ma qui teniamo un payload leggero)
        if _args.get("reply_to_message_id") is not None:
            payload["reply_parameters"] = {"message_id": int(_args["reply_to_message_id"])}

        r = requests.post(url, json=payload, timeout=8)
        data = r.json() if r.headers.get("content-type", "").startswith("application/json") else None

        if r.status_code != 200:
            return _err(f"Telegram HTTP error {r.status_code}", {"response": data})

        if not data or not data.get("ok"):
            return _err("Telegram API returned ok=false", {"response": data})

        # data["result"] è il Message inviato
        result = data.get("result", {})
        return _ok({
            "message_id": result.get("message_id"),
            "chat_id": (result.get("chat") or {}).get("id"),
            "date": result.get("date"),
        })

    except (requests.RequestException, ValueError, TypeError, RuntimeError) as e:
        logger.exception("Telegram sendMessage failed")
        return _err(str(e))
