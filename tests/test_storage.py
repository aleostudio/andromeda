# Copyright (c) 2026 Alessandro Orrù
# Licensed under MIT

import sqlite3
from andromeda.storage import SQLiteStore


def test_sqlite_store_initializes_schema(tmp_path):
    store = SQLiteStore(tmp_path / "andromeda.sqlite3")
    store.connect()

    assert store.count_memories() == 0

    store.close()


def test_sqlite_store_repairs_missing_schema_with_existing_user_version(tmp_path):
    db_path = tmp_path / "andromeda.sqlite3"
    with sqlite3.connect(db_path) as conn:
        conn.execute("PRAGMA user_version=1")

    store = SQLiteStore(db_path)
    store.connect()

    assert store.count_memories() == 0
    assert store.list_active_timers() == []
    assert store.list_active_scheduled_events() == []

    store.close()


def test_memory_crud(tmp_path):
    store = SQLiteStore(tmp_path / "andromeda.sqlite3")
    store.save_memory("wifi", "ABC123")

    record = store.get_memory("wifi")
    assert record is not None
    assert record.key == "wifi"
    assert record.value == "ABC123"
    assert record.sensitive is False

    store.save_memory("wifi", "XYZ789", sensitive=True)
    updated = store.get_memory("wifi")
    assert updated is not None
    assert updated.value == "XYZ789"
    assert updated.sensitive is True

    matches = store.search_memories("wi")
    assert [record.key for record in matches] == ["wifi"]

    assert store.delete_memory("wifi") is True
    assert store.get_memory("wifi") is None

    store.close()


def test_import_memories_is_idempotent(tmp_path):
    store = SQLiteStore(tmp_path / "andromeda.sqlite3")

    assert store.import_memories({"wifi": "ABC123"}) == 1
    assert store.import_memories({"wifi": "ABC123"}) == 0
    assert store.count_memories() == 1

    store.close()


def test_timer_lifecycle(tmp_path):
    store = SQLiteStore(tmp_path / "andromeda.sqlite3")
    store.create_timer("timer-1", "pasta", 60, 1000.0, 1060.0)

    active = store.list_active_timers()
    assert len(active) == 1
    assert active[0].id == "timer-1"
    assert active[0].label == "pasta"
    assert active[0].duration_sec == 60
    assert active[0].due_at == 1060.0

    assert store.complete_timer("timer-1", 1060.0) is True
    assert store.list_active_timers() == []
    assert store.complete_timer("timer-1", 1060.0) is False

    store.close()


def test_scheduled_event_lifecycle(tmp_path):
    store = SQLiteStore(tmp_path / "andromeda.sqlite3")
    store.create_scheduled_event("event-1", "chiamare Marco", "2026-05-24T18:00:00+00:00")

    active = store.list_active_scheduled_events()
    assert len(active) == 1
    assert active[0].id == "event-1"
    assert active[0].title == "chiamare Marco"
    assert active[0].due_at == "2026-05-24T18:00:00+00:00"

    assert store.complete_scheduled_event("event-1") is True
    assert store.list_active_scheduled_events() == []
    assert store.complete_scheduled_event("event-1") is False

    store.close()


def test_scheduled_event_cancel(tmp_path):
    store = SQLiteStore(tmp_path / "andromeda.sqlite3")
    store.create_scheduled_event("event-1", "chiamare Marco", "2026-05-24T18:00:00+00:00")

    assert store.cancel_scheduled_event("event-1") is True
    assert store.list_active_scheduled_events() == []
    assert store.cancel_scheduled_event("event-1") is False

    store.close()


def test_delete_expired_scheduled_events(tmp_path):
    store = SQLiteStore(tmp_path / "andromeda.sqlite3")
    store.create_scheduled_event("expired", "scaduto", "2026-05-24T18:00:00+00:00")
    store.create_scheduled_event("future", "futuro", "2026-05-24T19:00:00+00:00")

    assert store.delete_expired_scheduled_events("2026-05-24T18:30:00+00:00") == 1

    active = store.list_active_scheduled_events()
    assert len(active) == 1
    assert active[0].id == "future"

    store.close()


def test_delete_expired_scheduled_events_excludes_running_ids(tmp_path):
    store = SQLiteStore(tmp_path / "andromeda.sqlite3")
    store.create_scheduled_event("expired", "scaduto", "2026-05-24T18:00:00+00:00")

    assert store.delete_expired_scheduled_events(
        "2026-05-24T18:30:00+00:00",
        exclude_ids={"expired"},
    ) == 0
    assert len(store.list_active_scheduled_events()) == 1

    store.close()
