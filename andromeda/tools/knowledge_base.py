# Copyright (c) 2026 Alessandro Orrù
# Licensed under MIT

import json
import logging
import os
import re
import tempfile
import threading
from dataclasses import dataclass, field
from pathlib import Path
from andromeda.messages import msg

logger = logging.getLogger("[ TOOL KNOWLEDGE BASE ]")
audit_logger = logging.getLogger("[ TOOL AUDIT ]")

_SENSITIVE_PATTERNS = (
    re.compile(r"password", re.IGNORECASE),
    re.compile(r"passwd", re.IGNORECASE),
    re.compile(r"token", re.IGNORECASE),
    re.compile(r"secret", re.IGNORECASE),
    re.compile(r"api[_\s-]?key", re.IGNORECASE),
    re.compile(r"private[_\s-]?key", re.IGNORECASE),
    re.compile(r"\bssn\b", re.IGNORECASE),
    re.compile(r"\bcredit[_\s-]?card\b", re.IGNORECASE),
)


@dataclass
class _KnowledgeBaseState:
    store_path: str = "data/knowledge.json"
    cache: dict | None = None
    allow_sensitive_memory: bool = False
    lock: threading.RLock = field(default_factory=threading.RLock)


_state = _KnowledgeBaseState()


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
                "allow_sensitive": {
                    "type": "boolean",
                    "description": (
                        "Conferma esplicita per salvare dati sensibili. "
                        "Usa true solo se l'utente ha confermato chiaramente."
                    ),
                    "default": False,
                },
            },
            "required": ["action"],
        },
    },
}


def configure(store_path: str, allow_sensitive_memory: bool = False) -> None:
    with _state.lock:
        _state.store_path = store_path
        _state.cache = None
        _state.allow_sensitive_memory = allow_sensitive_memory


def _is_sensitive_text(text: str) -> bool:
    for pattern in _SENSITIVE_PATTERNS:
        if pattern.search(text):
            return True

    return False


def _load_store() -> dict:
    with _state.lock:
        if _state.cache is not None:
            return _state.cache

        path = Path(_state.store_path)
        if not path.exists():
            _state.cache = {}
            return _state.cache
        try:
            _state.cache = json.loads(path.read_text(encoding="utf-8"))
            logger.debug("Knowledge base loaded from disk: %d entries", len(_state.cache))
            return _state.cache
        except (json.JSONDecodeError, OSError):
            logger.warning("Failed to load knowledge base from %s", path)
            _state.cache = {}
            return _state.cache


def _save_store(data: dict) -> None:
    with _state.lock:
        path = Path(_state.store_path)
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
        _state.cache = data


def _action_save(store: dict, key: str, value: str, allow_sensitive: bool) -> str:
    if not key or not value:
        return msg("kb.save_missing_fields")

    key_value_text = f"{key} {value}"
    is_sensitive = _is_sensitive_text(key_value_text)
    if is_sensitive and not _state.allow_sensitive_memory and not allow_sensitive:
        audit_logger.info("tool=knowledge_base action=save_blocked_sensitive key=%s", key)
        return msg("kb.sensitive_blocked")

    store[key] = value
    _save_store(store)
    logger.info("Knowledge base: saved '%s'", key)
    audit_logger.info("tool=knowledge_base action=save key=%s", key)

    return msg("kb.saved", key=key, value=value)


def _action_recall(store: dict, key: str) -> str:
    if not key:
        return msg("kb.recall_missing_key")

    result = store.get(key)
    if result is not None:
        return f"{key}: {result}"

    # Fuzzy search: check if key is substring of any stored key
    matches = {k: v for k, v in store.items() if key.lower() in k.lower()}
    if not matches:
        return msg("kb.recall_not_found", key=key)

    parts = [f"- {k}: {v}" for k, v in matches.items()]

    return msg("kb.recall_matches", matches=", ".join(parts))


def _action_list(store: dict) -> str:
    if not store:
        return msg("kb.empty")
    keys = ", ".join(store.keys())

    return msg("kb.list", keys=keys)


def _action_delete(store: dict, key: str) -> str:
    if not key:
        return msg("kb.delete_missing_key")
    if key not in store:
        return msg("kb.delete_not_found", key=key)
    del store[key]
    _save_store(store)
    audit_logger.info("tool=knowledge_base action=delete key=%s", key)

    return msg("kb.deleted", key=key)


_ACTION_MAP = {
    "save": lambda store, key, value, allow_sensitive: _action_save(store, key, value, allow_sensitive),
    "recall": lambda store, key, _value: _action_recall(store, key),
    "list": lambda store, _key, _value: _action_list(store),
    "delete": lambda store, key, _value: _action_delete(store, key),
}


def handler(args: dict) -> str:
    action = args.get("action", "")
    key = args.get("key", "").strip()
    value = args.get("value", "").strip()
    allow_sensitive = bool(args.get("allow_sensitive", False))

    action_fn = _ACTION_MAP.get(action)
    if action_fn is None:
        return msg("kb.invalid_action", action=action)

    with _state.lock:
        if action == "save":
            return action_fn(_load_store(), key, value, allow_sensitive)
        return action_fn(_load_store(), key, value)
