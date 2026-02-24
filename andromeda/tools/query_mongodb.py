# Copyright (c) 2026 Alessandro Orrù
# Licensed under MIT

import os
import json
import logging
from datetime import datetime
from typing import Any
from bson import ObjectId
from pymongo import MongoClient
from pymongo.errors import PyMongoError

logger = logging.getLogger("[ TOOL MONGO QUERY ]")

# Just as example to test the tool: move them in env vars (MONGO_URI, MONGO_DB)
_MONGO_URI="mongodb://your_user:your_password@localhost:27017/?retryWrites=true&w=majority"
_MONGO_DB="your_database"


DEFINITION = {
    "type": "function",
    "function": {
        "name": "mongo_query",
        "description": (
            "Esegui query su MongoDB. "
            "Usa questo strumento quando devi leggere dati da Mongo (es: recuperare record, "
            "cercare per criteri, contare documenti, o fare piccole aggregazioni). "
            "Operazioni supportate: find, find_one, count, aggregate."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "collection": {
                    "type": "string",
                    "description": "Nome della collection su cui operare (es: 'users', 'events').",
                },
                "operation": {
                    "type": "string",
                    "enum": ["find", "find_one", "count", "aggregate"],
                    "description": "Tipo di operazione da eseguire.",
                },
                "filter": {
                    "type": "object",
                    "description": "Filtro MongoDB (es: {'userId': '...'}). Usato per find/find_one/count.",
                    "default": {},
                },
                "projection": {
                    "type": "object",
                    "description": "Projection MongoDB (es: {'_id': 0, 'name': 1}). Usato per find/find_one.",
                },
                "sort": {
                    "type": "array",
                    "description": (
                        "Ordinamento come lista di coppie [campo, direzione]. "
                        "Esempio: [['createdAt', -1], ['name', 1]]."
                    ),
                    "items": {
                        "type": "array",
                        "minItems": 2,
                        "maxItems": 2,
                        "items": [{"type": "string"}, {"type": "integer", "enum": [-1, 1]}],
                    },
                },
                "limit": {
                    "type": "integer",
                    "minimum": 1,
                    "maximum": 500,
                    "description": "Numero massimo di documenti da restituire (solo find).",
                    "default": 50,
                },
                "skip": {
                    "type": "integer",
                    "minimum": 0,
                    "maximum": 100000,
                    "description": "Numero di documenti da saltare (solo find).",
                    "default": 0,
                },
                "pipeline": {
                    "type": "array",
                    "description": "Pipeline di aggregazione (solo aggregate).",
                    "items": {"type": "object"},
                },
            },
            "required": ["collection", "operation"],
            "additionalProperties": False,
        },
    },
}


def _json_default(o: Any):
    if isinstance(o, ObjectId):
        return str(o)

    if isinstance(o, datetime):
        return o.isoformat()

    return str(o)


def _get_client() -> MongoClient:
    mongo_uri = os.getenv("MONGO_URI", _MONGO_URI)
    if not mongo_uri:
        raise RuntimeError("Missing env var MONGO_URI")

    return MongoClient(mongo_uri, serverSelectionTimeoutMS=4000)


def _ok(payload: dict) -> str:
    return json.dumps({"ok": True, **payload}, default=_json_default, ensure_ascii=False)


def _err(message: str) -> str:
    return json.dumps({"ok": False, "error": message}, ensure_ascii=False)


def _parse_args(_args: dict) -> dict:
    collection = _args.get("collection")
    if not collection:
        raise ValueError("Missing required parameter: collection")
    
    operation = _args.get("operation")
    if not operation:
        raise ValueError("Missing required parameter: operation")

    limit = int(_args.get("limit", 50) or 50)
    limit = min(limit, 500)
    skip = int(_args.get("skip", 0) or 0)

    return {
        "collection": collection,
        "operation": operation,
        "filter": _args.get("filter") or {},
        "projection": _args.get("projection"),
        "sort": _args.get("sort"),
        "limit": limit,
        "skip": skip,
        "pipeline": _args.get("pipeline"),
    }


def _op_find(col, a: dict) -> str:
    filt, proj = a["filter"], a["projection"]
    cursor = col.find(filt, proj) if proj else col.find(filt)
    cursor = cursor.skip(a["skip"]).limit(a["limit"])

    if a["sort"]:
        cursor = cursor.sort([(f, d) for f, d in a["sort"]])

    docs = list(cursor)
    return _ok({"count": len(docs), "items": docs})


def _op_find_one(col, a: dict) -> str:
    filt, proj = a["filter"], a["projection"]
    doc = col.find_one(filt, proj) if proj else col.find_one(filt)
    return _ok({"item": doc})


def _op_count(col, a: dict) -> str:
    c = col.count_documents(a["filter"])
    return _ok({"count": c})


def _op_aggregate(col, a: dict) -> str:
    pipeline = a["pipeline"]
    if not isinstance(pipeline, list) or not pipeline:
        return _err("aggregate requires non-empty 'pipeline' array")

    result = list(col.aggregate(pipeline, allowDiskUse=True))
    return _ok({"count": len(result), "items": result})


_OPS = {
    "find": _op_find,
    "find_one": _op_find_one,
    "count": _op_count,
    "aggregate": _op_aggregate,
}


def handler(_args: dict) -> str:
    db_name = os.getenv("MONGO_DB", _MONGO_DB)
    if not db_name:
        return _err("Missing env var MONGO_DB")

    try:
        a = _parse_args(_args)

        op_fn = _OPS.get(a["operation"])
        if not op_fn:
            return _err(f"Unsupported operation: {a['operation']}")

        client = _get_client()
        try:
            col = client[db_name][a["collection"]]
            return op_fn(col, a)
        finally:
            client.close()

    except (PyMongoError, RuntimeError, ValueError, TypeError) as e:
        logger.exception("Mongo query failed")
        return _err(str(e))
