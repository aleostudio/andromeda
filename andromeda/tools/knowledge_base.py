# Copyright (c) 2026 Alessandro Orrù
# Licensed under MIT

import json
import logging
import re
import threading
from dataclasses import dataclass, field
from pathlib import Path
from andromeda.messages import msg
from andromeda.storage import SQLiteStore

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
    store: SQLiteStore | None = None
    legacy_json_path: str = "data/knowledge.json"
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


def configure(
    store: SQLiteStore | str = "data/andromeda.sqlite3",
    allow_sensitive_memory: bool = False,
    *,
    legacy_json_path: str = "data/knowledge.json",
) -> None:
    with _state.lock:
        if isinstance(store, SQLiteStore):
            _state.store = store
        else:
            path = Path(store)
            if path.suffix.lower() == ".json":
                path = path.with_suffix(".sqlite3")
                legacy_json_path = str(store)
            _state.store = SQLiteStore(path)
        _state.store.connect()
        _state.legacy_json_path = legacy_json_path
        _state.allow_sensitive_memory = allow_sensitive_memory
        _import_legacy_json_locked()


def _is_sensitive_text(text: str) -> bool:
    for pattern in _SENSITIVE_PATTERNS:
        if pattern.search(text):
            return True

    return False


def _store() -> SQLiteStore:
    if _state.store is None:
        _state.store = SQLiteStore("data/andromeda.sqlite3")
        _state.store.connect()
        _import_legacy_json_locked()

    return _state.store


def _import_legacy_json_locked() -> None:
    path = Path(_state.legacy_json_path)
    if not path.exists():
        return

    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        logger.warning("Failed to import legacy knowledge JSON from %s", path)
        return

    if not isinstance(raw, dict):
        logger.warning("Legacy knowledge JSON is not an object: %s", path)
        return

    imported = _store().import_memories({str(k): str(v) for k, v in raw.items()})
    if imported:
        logger.info("Imported %d legacy knowledge entries into SQLite", imported)


def _action_save(key: str, value: str, allow_sensitive: bool) -> str:
    if not key or not value:
        return msg("kb.save_missing_fields")

    key_value_text = f"{key} {value}"
    is_sensitive = _is_sensitive_text(key_value_text)
    if is_sensitive and not _state.allow_sensitive_memory and not allow_sensitive:
        audit_logger.info("tool=knowledge_base action=save_blocked_sensitive key=%s", key)
        return msg("kb.sensitive_blocked")

    _store().save_memory(key, value, sensitive=is_sensitive)
    logger.info("Knowledge base: saved '%s'", key)
    audit_logger.info("tool=knowledge_base action=save key=%s", key)

    return msg("kb.saved", key=key, value=value)


def _action_recall(key: str) -> str:
    if not key:
        return msg("kb.recall_missing_key")

    result = _store().get_memory(key)
    if result is not None:
        return f"{key}: {result.value}"

    matches = _store().search_memories(key)
    if not matches:
        return msg("kb.recall_not_found", key=key)

    parts = [f"- {record.key}: {record.value}" for record in matches]

    return msg("kb.recall_matches", matches=", ".join(parts))


def _action_list() -> str:
    records = _store().list_memories()
    if not records:
        return msg("kb.empty")
    keys = ", ".join(record.key for record in records)

    return msg("kb.list", keys=keys)


def _action_delete(key: str) -> str:
    if not key:
        return msg("kb.delete_missing_key")
    if not _store().delete_memory(key):
        return msg("kb.delete_not_found", key=key)
    audit_logger.info("tool=knowledge_base action=delete key=%s", key)

    return msg("kb.deleted", key=key)


_ACTION_MAP = {
    "save": lambda key, value, allow_sensitive: _action_save(key, value, allow_sensitive),
    "recall": lambda key, _value: _action_recall(key),
    "list": lambda _key, _value: _action_list(),
    "delete": lambda key, _value: _action_delete(key),
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
            return action_fn(key, value, allow_sensitive)
        return action_fn(key, value)
