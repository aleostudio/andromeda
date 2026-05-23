# Copyright (c) 2026 Alessandro Orrù
# Licensed under MIT

from andromeda.storage import SQLiteStore


def test_sqlite_store_initializes_schema(tmp_path):
    store = SQLiteStore(tmp_path / "andromeda.sqlite3")
    store.connect()

    assert store.count_memories() == 0

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
