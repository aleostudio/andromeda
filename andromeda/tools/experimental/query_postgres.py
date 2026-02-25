# Copyright (c) 2026 Alessandro Orrù
# Licensed under MIT

import os
import json
import logging
from datetime import datetime, date, time
from decimal import Decimal
from typing import Any, Dict, List, Optional
from uuid import UUID

import psycopg
from psycopg.rows import dict_row
from psycopg.errors import Error as PsycopgError

logger = logging.getLogger("[ TOOL POSTGRES QUERY ]")

# Just as example to test the tool: move it in env var (PG_DSN)
_PG_DSN = "postgresql://your_user:your_password@localhost:5432/your_database?sslmode=disable"


DEFINITION = {
    "type": "function",
    "function": {
        "name": "postgres_query",
        "description": (
            "Esegui query su PostgreSQL in modo sicuro (parametrizzato). "
            "Usa questo strumento per leggere record (select_one/select_many), "
            "ottenere un valore singolo (scalar) o eseguire scritture (execute)."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "operation": {
                    "type": "string",
                    "enum": ["select_one", "select_many", "scalar", "execute"],
                    "description": "Tipo di operazione da eseguire.",
                },
                "sql": {
                    "type": "string",
                    "description": (
                        "Query SQL parametrizzata. Usa placeholder %s. "
                        "Esempio: 'SELECT * FROM users WHERE id = %s'."
                    ),
                },
                "params": {
                    "type": "array",
                    "description": (
                        "Lista parametri per i placeholder %s, in ordine. "
                        "Esempio: [123, 'active']."
                    ),
                    "items": {},
                    "default": [],
                },
                "limit": {
                    "type": "integer",
                    "minimum": 1,
                    "maximum": 500,
                    "description": "Limite massimo righe (solo select_many, opzionale).",
                    "default": 50,
                },
            },
            "required": ["operation", "sql"],
            "additionalProperties": False,
        },
    },
}


def _json_default(o: Any):
    if isinstance(o, (datetime, date, time)):
        return o.isoformat()
    if isinstance(o, Decimal):
        return str(o)
    if isinstance(o, UUID):
        return str(o)
    if isinstance(o, (bytes, bytearray, memoryview)):
        return f"<bytes:{len(o)}>"
    return str(o)


def _ok(payload: dict) -> str:
    return json.dumps({"ok": True, **payload}, default=_json_default, ensure_ascii=False)


def _err(message: str) -> str:
    return json.dumps({"ok": False, "error": message}, ensure_ascii=False)


def _get_conn_kwargs() -> dict:
    dsn = os.getenv("PG_DSN", _PG_DSN).strip()
    if not dsn:
        raise RuntimeError("Missing Postgres env var PG_DSN")

    return {"conninfo": dsn}


def _parse_args(_args: dict) -> dict:
    op = _args.get("operation")
    sql = _args.get("sql")
    if not op:
        raise ValueError("Missing required parameter: operation")
    if not sql:
        raise ValueError("Missing required parameter: sql")

    params = _args.get("params") or []
    if not isinstance(params, list):
        raise TypeError("params must be an array")

    limit = int(_args.get("limit", 50) or 50)
    limit = min(limit, 500)

    return {"operation": op, "sql": sql, "params": params, "limit": limit}


def _op_select_one(cur, a: dict) -> str:
    cur.execute(a["sql"], a["params"])
    row = cur.fetchone()
    return _ok({"item": row})


def _op_select_many(cur, a: dict) -> str:
    sql = a["sql"].rstrip().rstrip(";")
    capped_sql = f"{sql} LIMIT {a['limit']}"
    cur.execute(capped_sql, a["params"])
    rows = cur.fetchall()
    return _ok({"count": len(rows), "items": rows})


def _op_scalar(cur, a: dict) -> str:
    cur.execute(a["sql"], a["params"])
    row = cur.fetchone()
    if not row:
        value = None
    elif isinstance(row, dict):
        value = next(iter(row.values()))
    else:
        value = row[0]

    return _ok({"value": value})


def _op_execute(cur, a: dict) -> str:
    cur.execute(a["sql"], a["params"])
    return _ok({"rowcount": cur.rowcount})


_OPS = {
    "select_one": _op_select_one,
    "select_many": _op_select_many,
    "scalar": _op_scalar,
    "execute": _op_execute,
}


def handler(_args: dict) -> str:
    try:
        a = _parse_args(_args)

        op_fn = _OPS.get(a["operation"])
        if not op_fn:
            return _err(f"Unsupported operation: {a['operation']}")

        conn_kwargs = _get_conn_kwargs()

        if "conninfo" in conn_kwargs:
            with psycopg.connect(conn_kwargs["conninfo"], row_factory=dict_row) as conn:
                with conn.cursor() as cur:
                    out = op_fn(cur, a)
                if a["operation"] == "execute":
                    conn.commit()
                return out
        else:
            with psycopg.connect(**conn_kwargs, row_factory=dict_row) as conn:
                with conn.cursor() as cur:
                    out = op_fn(cur, a)
                if a["operation"] == "execute":
                    conn.commit()
                return out

    except (PsycopgError, RuntimeError, ValueError, TypeError) as e:
        logger.exception("Postgres query failed")
        return _err(str(e))
