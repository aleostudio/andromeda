# Copyright (c) 2026 Alessandro Orrù
# Licensed under MIT

import json
import logging
import os
import tempfile
from pathlib import Path

logger = logging.getLogger("[ TOOL KNOWLEDGE BASE ]")

_store_path: str = "data/knowledge.json"

# In-memory cache — avoids reading from disk on every access
_cache: dict | None = None


DEFINITION = {
    "type": "function",
    "function": {
        "name": "knowledge_base",
        "description": (
            "Salva o recupera informazioni dalla memoria persistente. "
            "Usa questo strumento quando l'utente chiede di ricordare qualcosa, "
            "memorizzare un'informazione, o recuperare qualcosa che ha detto in precedenza. "
            "Esempio: 'ricorda che la password del wifi è ABC123' oppure 'qual è la password del wifi?'"
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "action": {
                    "type": "string",
                    "enum": ["save", "recall", "list", "delete"],
                    "description": (
                        "save: salva un'informazione, "
                        "recall: recupera per chiave, "
                        "list: elenca tutte le chiavi salvate, "
                        "delete: elimina una chiave"
                    ),
                },
                "key": {
                    "type": "string",
                    "description": "Chiave identificativa (es. 'password_wifi', 'compleanno_mamma')",
                },
                "value": {
                    "type": "string",
                    "description": "Valore da salvare (solo per action=save)",
                },
            },
            "required": ["action"],
        },
    },
}


def configure(store_path: str) -> None:
    global _store_path, _cache
    _store_path = store_path
    _cache = None


def _load_store() -> dict:
    global _cache
    if _cache is not None:
        return _cache

    path = Path(_store_path)
    if not path.exists():
        _cache = {}
        return _cache
    try:
        _cache = json.loads(path.read_text(encoding="utf-8"))
        logger.debug("Knowledge base loaded from disk: %d entries", len(_cache))
        return _cache
    except (json.JSONDecodeError, OSError):
        logger.warning("Failed to load knowledge base from %s", path)
        _cache = {}
        return _cache


def _save_store(data: dict) -> None:
    global _cache
    path = Path(_store_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    # Atomic write: write to temp file then rename to prevent corruption on crash
    fd, tmp_path = tempfile.mkstemp(dir=path.parent, suffix=".tmp")

    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        os.replace(tmp_path, path)
    except Exception:
        os.unlink(tmp_path)
        raise

    # Update in-memory cache after successful write
    _cache = data


def _action_save(store: dict, key: str, value: str) -> str:
    if not key or not value:
        return "Errore: serve sia una chiave che un valore per salvare."
    store[key] = value
    _save_store(store)
    logger.info("Knowledge base: saved '%s'", key)

    return f"Ho memorizzato '{key}': {value}"


def _action_recall(store: dict, key: str) -> str:
    if not key:
        return "Errore: specifica quale informazione vuoi recuperare."
    result = store.get(key)
    if result is not None:
        return f"{key}: {result}"
    # Fuzzy search: check if key is substring of any stored key
    matches = {k: v for k, v in store.items() if key.lower() in k.lower()}
    if not matches:
        return f"Non ho trovato nulla per '{key}'."
    parts = [f"- {k}: {v}" for k, v in matches.items()]

    return "Ho trovato queste corrispondenze: " + ", ".join(parts)


def _action_list(store: dict) -> str:
    if not store:
        return "La memoria è vuota, non ho ancora salvato nulla."
    keys = ", ".join(store.keys())

    return f"Informazioni memorizzate: {keys}"


def _action_delete(store: dict, key: str) -> str:
    if not key:
        return "Errore: specifica quale informazione vuoi eliminare."
    if key not in store:
        return f"'{key}' non è presente in memoria."
    del store[key]
    _save_store(store)

    return f"Ho eliminato '{key}' dalla memoria."


_ACTION_MAP = {
    "save": lambda store, key, value: _action_save(store, key, value),
    "recall": lambda store, key, _value: _action_recall(store, key),
    "list": lambda store, _key, _value: _action_list(store),
    "delete": lambda store, key, _value: _action_delete(store, key),
}


def handler(args: dict) -> str:
    action = args.get("action", "")
    key = args.get("key", "").strip()
    value = args.get("value", "").strip()

    action_fn = _ACTION_MAP.get(action)
    if action_fn is None:
        return f"Azione '{action}' non riconosciuta. Usa: save, recall, list, delete."

    return action_fn(_load_store(), key, value)
