# Copyright (c) 2026 Alessandro Orrù
# Licensed under MIT

import sqlite3
import threading
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path


_SCHEMA_VERSION = 1


@dataclass(frozen=True)
class MemoryRecord:
    key: str
    value: str
    sensitive: bool
    created_at: str
    updated_at: str


class SQLiteStore:
    def __init__(self, path: str | Path) -> None:
        self._path = Path(path)
        self._lock = threading.RLock()
        self._conn: sqlite3.Connection | None = None


    def connect(self) -> None:
        with self._lock:
            if self._conn is not None:
                return

            self._path.parent.mkdir(parents=True, exist_ok=True)
            self._conn = sqlite3.connect(str(self._path), check_same_thread=False)
            self._conn.row_factory = sqlite3.Row
            self._conn.execute("PRAGMA journal_mode=WAL")
            self._conn.execute("PRAGMA foreign_keys=ON")
            self._migrate()


    def close(self) -> None:
        with self._lock:
            if self._conn is not None:
                self._conn.close()
                self._conn = None


    def _connection(self) -> sqlite3.Connection:
        self.connect()
        if self._conn is None:
            raise RuntimeError("SQLite store is not connected")

        return self._conn


    def _migrate(self) -> None:
        conn = self._connection()
        version = int(conn.execute("PRAGMA user_version").fetchone()[0])
        if version < 1:
            self._create_schema_v1(conn)
            conn.execute(f"PRAGMA user_version={_SCHEMA_VERSION}")
            conn.commit()


    @staticmethod
    def _create_schema_v1(conn: sqlite3.Connection) -> None:
        conn.executescript(
            """
            CREATE TABLE IF NOT EXISTS memories (
                key TEXT PRIMARY KEY,
                value TEXT NOT NULL,
                sensitive INTEGER NOT NULL DEFAULT 0,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL
            );

            CREATE TABLE IF NOT EXISTS timers (
                id TEXT PRIMARY KEY,
                label TEXT NOT NULL,
                duration_sec INTEGER NOT NULL,
                started_at REAL NOT NULL,
                due_at REAL NOT NULL,
                status TEXT NOT NULL,
                completed_at REAL
            );

            CREATE INDEX IF NOT EXISTS idx_timers_status_due_at
                ON timers(status, due_at);

            CREATE TABLE IF NOT EXISTS scheduled_events (
                id TEXT PRIMARY KEY,
                title TEXT NOT NULL,
                due_at TEXT NOT NULL,
                recurrence TEXT,
                payload_json TEXT,
                status TEXT NOT NULL,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL
            );

            CREATE INDEX IF NOT EXISTS idx_scheduled_events_status_due_at
                ON scheduled_events(status, due_at);
            """
        )


    @staticmethod
    def _now_text() -> str:
        return datetime.now(UTC).isoformat()


    def save_memory(self, key: str, value: str, *, sensitive: bool = False) -> None:
        now = self._now_text()
        with self._lock:
            conn = self._connection()
            conn.execute(
                """
                INSERT INTO memories (key, value, sensitive, created_at, updated_at)
                VALUES (?, ?, ?, ?, ?)
                ON CONFLICT(key) DO UPDATE SET
                    value = excluded.value,
                    sensitive = excluded.sensitive,
                    updated_at = excluded.updated_at
                """,
                (key, value, int(sensitive), now, now),
            )
            conn.commit()


    def get_memory(self, key: str) -> MemoryRecord | None:
        with self._lock:
            row = self._connection().execute(
                "SELECT key, value, sensitive, created_at, updated_at FROM memories WHERE key = ?",
                (key,),
            ).fetchone()

        return self._row_to_memory(row) if row else None


    def search_memories(self, query: str) -> list[MemoryRecord]:
        like = f"%{query.lower()}%"
        with self._lock:
            rows = self._connection().execute(
                """
                SELECT key, value, sensitive, created_at, updated_at
                FROM memories
                WHERE lower(key) LIKE ?
                ORDER BY key
                """,
                (like,),
            ).fetchall()

        return [self._row_to_memory(row) for row in rows]


    def list_memories(self) -> list[MemoryRecord]:
        with self._lock:
            rows = self._connection().execute(
                """
                SELECT key, value, sensitive, created_at, updated_at
                FROM memories
                ORDER BY key
                """
            ).fetchall()

        return [self._row_to_memory(row) for row in rows]


    def import_memories(self, memories: dict[str, str]) -> int:
        if not memories:
            return 0

        now = self._now_text()
        rows = [
            (str(key), str(value), 0, now, now)
            for key, value in memories.items()
            if str(key).strip() and str(value).strip()
        ]
        if not rows:
            return 0

        with self._lock:
            conn = self._connection()
            before = conn.total_changes
            conn.executemany(
                """
                INSERT OR IGNORE INTO memories (key, value, sensitive, created_at, updated_at)
                VALUES (?, ?, ?, ?, ?)
                """,
                rows,
            )
            conn.commit()

            return conn.total_changes - before


    def count_memories(self) -> int:
        with self._lock:
            row = self._connection().execute("SELECT COUNT(*) FROM memories").fetchone()

        return int(row[0])


    def delete_memory(self, key: str) -> bool:
        with self._lock:
            conn = self._connection()
            cursor = conn.execute("DELETE FROM memories WHERE key = ?", (key,))
            conn.commit()

        return cursor.rowcount > 0


    @staticmethod
    def _row_to_memory(row: sqlite3.Row) -> MemoryRecord:
        return MemoryRecord(
            key=str(row["key"]),
            value=str(row["value"]),
            sensitive=bool(row["sensitive"]),
            created_at=str(row["created_at"]),
            updated_at=str(row["updated_at"]),
        )
