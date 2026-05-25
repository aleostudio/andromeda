# Copyright (c) 2026 Alessandro Orrù
# Licensed under MIT

import asyncio
import json
import sqlite3
from datetime import UTC, datetime, timedelta
from unittest.mock import AsyncMock, MagicMock, patch
import pytest
from andromeda.config import ToolsConfig
from andromeda.intent import clear_intents, match_and_execute
from andromeda.storage import SQLiteStore
from andromeda.tools import get_datetime, knowledge_base, register_all_tools, schedule_event, set_timer, system_control


class TestGetDatetime:
    def test_handler_returns_string(self):
        result = get_datetime.handler({})
        assert isinstance(result, str)

    def test_handler_contains_data(self):
        result = get_datetime.handler({})
        assert "Data:" in result
        assert "Ora:" in result

    def test_handler_current_year(self):
        result = get_datetime.handler({})
        assert str(datetime.now().year) in result

    def test_definition_structure(self):
        assert get_datetime.DEFINITION["type"] == "function"
        assert get_datetime.DEFINITION["function"]["name"] == "get_datetime"
        assert "parameters" in get_datetime.DEFINITION["function"]


class TestKnowledgeBase:
    @pytest.fixture(autouse=True)
    def temp_store(self, tmp_path):
        """Configure knowledge base with temporary store."""
        sqlite_path = tmp_path / "andromeda.sqlite3"
        legacy_path = tmp_path / "knowledge.json"
        knowledge_base.configure(str(sqlite_path), legacy_json_path=str(legacy_path))
        yield sqlite_path

    def test_save(self):
        result = knowledge_base.handler({"action": "save", "key": "wifi", "value": "ABC123"})
        assert "memorizzato" in result
        assert "wifi" in result

    def test_recall_existing(self):
        knowledge_base.handler({"action": "save", "key": "wifi", "value": "ABC123"})
        result = knowledge_base.handler({"action": "recall", "key": "wifi"})
        assert "ABC123" in result

    def test_recall_missing(self):
        result = knowledge_base.handler({"action": "recall", "key": "nonexistent"})
        assert "Non ho trovato" in result

    def test_recall_fuzzy(self):
        knowledge_base.handler({"action": "save", "key": "wifi_network_name", "value": "mywifi"})
        result = knowledge_base.handler({"action": "recall", "key": "wifi"})
        assert "mywifi" in result

    def test_list_empty(self):
        result = knowledge_base.handler({"action": "list"})
        assert "vuota" in result

    def test_list_with_data(self):
        knowledge_base.handler({"action": "save", "key": "k1", "value": "v1"})
        knowledge_base.handler({"action": "save", "key": "k2", "value": "v2"})
        result = knowledge_base.handler({"action": "list"})
        assert "k1" in result
        assert "k2" in result

    def test_delete_existing(self):
        knowledge_base.handler({"action": "save", "key": "temp", "value": "data"})
        result = knowledge_base.handler({"action": "delete", "key": "temp"})
        assert "eliminato" in result
        # Verify it's gone
        result = knowledge_base.handler({"action": "recall", "key": "temp"})
        assert "Non ho trovato" in result

    def test_delete_missing(self):
        result = knowledge_base.handler({"action": "delete", "key": "nonexistent"})
        assert "non è presente" in result

    def test_invalid_action(self):
        result = knowledge_base.handler({"action": "invalid"})
        assert "non riconosciuta" in result

    def test_save_no_key(self):
        result = knowledge_base.handler({"action": "save", "key": "", "value": "val"})
        assert "Errore" in result

    def test_save_no_value(self):
        result = knowledge_base.handler({"action": "save", "key": "k", "value": ""})
        assert "Errore" in result

    def test_recall_no_key(self):
        result = knowledge_base.handler({"action": "recall", "key": ""})
        assert "Errore" in result

    def test_delete_no_key(self):
        result = knowledge_base.handler({"action": "delete", "key": ""})
        assert "Errore" in result

    def test_persistence(self, temp_store):
        """Data survives reload and is persisted in SQLite."""
        knowledge_base.handler({"action": "save", "key": "persist", "value": "data123"})

        with sqlite3.connect(temp_store) as conn:
            row = conn.execute("SELECT value FROM memories WHERE key = ?", ("persist",)).fetchone()

        assert row == ("data123",)

    def test_imports_legacy_json(self, tmp_path):
        legacy_path = tmp_path / "knowledge.json"
        legacy_path.write_text(json.dumps({"legacy_key": "legacy_value"}), encoding="utf-8")

        knowledge_base.configure(
            str(tmp_path / "andromeda.sqlite3"),
            legacy_json_path=str(legacy_path),
        )
        result = knowledge_base.handler({"action": "recall", "key": "legacy_key"})

        assert "legacy_value" in result

    def test_definition_structure(self):
        assert knowledge_base.DEFINITION["function"]["name"] == "knowledge_base"
        params = knowledge_base.DEFINITION["function"]["parameters"]
        assert "action" in params["properties"]
        assert "key" in params["properties"]
        assert "value" in params["properties"]

    def test_blocks_sensitive_without_opt_in(self):
        result = knowledge_base.handler({"action": "save", "key": "wifi_password", "value": "ABC123"})
        assert "Dato sensibile rilevato" in result

    def test_sensitive_save_with_opt_in(self):
        result = knowledge_base.handler({
            "action": "save",
            "key": "wifi_password",
            "value": "ABC123",
            "allow_sensitive": True,
        })
        assert "memorizzato" in result


class TestSetTimer:
    @pytest.fixture(autouse=True)
    def setup_timer(self, tmp_path):
        mock_feedback = MagicMock()
        mock_tts = MagicMock()
        mock_tts.speak = AsyncMock()
        store = SQLiteStore(tmp_path / "andromeda.sqlite3")
        set_timer.configure(mock_feedback, max_sec=3600, tts=mock_tts, store=store)
        set_timer._state.active_timers.clear()
        yield store
        # Cancel any remaining timers
        for timer in set_timer._state.active_timers.values():
            timer.cancel()
        set_timer._state.active_timers.clear()
        store.close()

    def test_invalid_seconds_string(self):
        result = set_timer.handler({"seconds": "abc", "label": "pasta"})
        assert "Errore" in result

    def test_zero_seconds(self):
        result = set_timer.handler({"seconds": 0, "label": "pasta"})
        assert "Errore" in result

    def test_negative_seconds(self):
        result = set_timer.handler({"seconds": -5, "label": "pasta"})
        assert "Errore" in result

    def test_exceeds_max(self):
        result = set_timer.handler({"seconds": 99999, "label": "pasta"})
        assert "Errore" in result
        assert "massima" in result

    def test_missing_label_asks_question(self):
        result = set_timer.handler({"seconds": 30})
        assert result == "Un timer cosa?"

    @pytest.mark.asyncio
    async def test_valid_seconds(self):
        result = set_timer.handler({"seconds": 30, "label": "pasta"})
        assert "timer" in result.lower()
        assert "30 secondi" in result

    @pytest.mark.asyncio
    async def test_timer_is_persisted(self, setup_timer):
        set_timer.handler({"seconds": 30, "label": "pasta"})

        active = setup_timer.list_active_timers()

        assert len(active) == 1
        assert active[0].label == "pasta"
        assert active[0].duration_sec == 30

    @pytest.mark.asyncio
    async def test_valid_minutes(self):
        result = set_timer.handler({"seconds": 300, "label": "pasta"})
        assert "pasta" in result
        assert "5 minuti" in result

    @pytest.mark.asyncio
    async def test_minutes_and_seconds(self):
        result = set_timer.handler({"seconds": 90, "label": "pasta"})
        assert "1 minuti" in result
        assert "30 secondi" in result

    @pytest.mark.asyncio
    async def test_custom_label(self):
        result = set_timer.handler({"seconds": 60, "label": "bucato"})
        assert "bucato" in result

    @pytest.mark.asyncio
    async def test_status_no_active_timers(self):
        result = set_timer.handler({"action": "status"})
        assert "Non ci sono timer attivi" in result

    @pytest.mark.asyncio
    async def test_status_lists_active_timers(self):
        set_timer.handler({"seconds": 300, "label": "pasta"})
        set_timer.handler({"seconds": 60, "label": "bucato"})

        result = set_timer.handler({"action": "status"})

        assert "Timer attivi" in result
        assert "pasta" in result
        assert "bucato" in result
        assert "mancano" in result

    @pytest.mark.asyncio
    async def test_status_restores_active_timers_from_sqlite(self, setup_timer):
        set_timer.handler({"seconds": 300, "label": "pasta"})

        for timer in set_timer._state.active_timers.values():
            timer.cancel()
        set_timer._state.active_timers.clear()

        result = set_timer.handler({"action": "status"})

        assert "Timer attivi" in result
        assert "pasta" in result

    def test_definition_structure(self):
        assert set_timer.DEFINITION["function"]["name"] == "set_timer"
        params = set_timer.DEFINITION["function"]["parameters"]
        assert "action" in params["properties"]
        assert "seconds" in params["properties"]
        assert "label" in params["properties"]


class TestScheduleEvent:
    @pytest.fixture(autouse=True)
    def setup_events(self, tmp_path):
        mock_feedback = MagicMock()
        mock_tts = MagicMock()
        mock_tts.speak = AsyncMock()
        store = SQLiteStore(tmp_path / "andromeda.sqlite3")
        schedule_event.configure(mock_feedback, tts=mock_tts, store=store)
        schedule_event._state.active_events.clear()
        yield store
        for event in schedule_event._state.active_events.values():
            event.cancel()
        schedule_event._state.active_events.clear()
        store.close()

    def test_missing_title(self):
        result = schedule_event.handler({"action": "set", "seconds": 60})
        assert "titolo" in result

    def test_missing_due_at(self):
        result = schedule_event.handler({"action": "set", "title": "chiamare Marco"})
        assert "quando" in result

    def test_invalid_due_at(self):
        result = schedule_event.handler({"action": "set", "title": "chiamare Marco", "due_at": "not-a-date"})
        assert "non valida" in result

    @pytest.mark.asyncio
    async def test_set_relative_event(self, setup_events):
        result = schedule_event.handler({"action": "set", "title": "chiamare Marco", "seconds": 60})

        active = setup_events.list_active_scheduled_events()

        assert "Promemoria" in result
        assert len(active) == 1
        assert active[0].title == "chiamare Marco"

    @pytest.mark.asyncio
    async def test_list_events(self):
        schedule_event.handler({"action": "set", "title": "chiamare Marco", "seconds": 60})

        result = schedule_event.handler({"action": "list"})

        assert "Promemoria attivi" in result
        assert "chiamare Marco" in result

    @pytest.mark.asyncio
    async def test_delete_event(self, setup_events):
        schedule_event.handler({"action": "set", "title": "chiamare Marco", "seconds": 60})
        event_id = setup_events.list_active_scheduled_events()[0].id

        result = schedule_event.handler({"action": "delete", "id": event_id})

        assert "eliminato" in result
        assert setup_events.list_active_scheduled_events() == []

    @pytest.mark.asyncio
    async def test_restores_active_events_from_sqlite(self, setup_events):
        schedule_event.handler({"action": "set", "title": "chiamare Marco", "seconds": 60})

        for event in schedule_event._state.active_events.values():
            event.cancel()
        schedule_event._state.active_events.clear()

        result = schedule_event.handler({"action": "list"})

        assert "Promemoria attivi" in result
        assert "chiamare Marco" in result

    @pytest.mark.asyncio
    async def test_deletes_expired_events_without_firing_on_restore(self, setup_events):
        setup_events.create_scheduled_event(
            "expired-event",
            "chiamare Marco",
            (datetime.now(UTC) - timedelta(seconds=60)).isoformat(),
        )

        result = schedule_event.handler({"action": "list"})

        assert "Non ci sono promemoria attivi" in result
        assert setup_events.list_active_scheduled_events() == []

    @pytest.mark.asyncio
    async def test_fired_event_deletes_record(self, setup_events):
        due_at = datetime.now(UTC) + timedelta(milliseconds=20)
        setup_events.create_scheduled_event(
            "runtime-event",
            "chiamare Marco",
            due_at.isoformat(),
        )

        schedule_event.resume_persisted_events()
        await asyncio.sleep(0.1)

        assert setup_events.list_active_scheduled_events() == []

    def test_definition_structure(self):
        assert schedule_event.DEFINITION["function"]["name"] == "schedule_event"
        params = schedule_event.DEFINITION["function"]["parameters"]
        assert "action" in params["properties"]
        assert "title" in params["properties"]
        assert "due_at" in params["properties"]
        assert "seconds" in params["properties"]


class TestSystemControl:
    @pytest.mark.asyncio
    async def test_unknown_action(self):
        result = await system_control.handler({"action": "fly_to_moon"})
        assert "non riconosciuta" in result

    @pytest.mark.asyncio
    async def test_valid_action_keys(self):
        """Verify all defined actions exist."""
        for action in ["volume_up", "volume_down", "volume_mute",
                       "volume_get", "brightness_up", "brightness_down"]:
            assert action in system_control._ACTIONS

    @pytest.mark.asyncio
    @patch("asyncio.create_subprocess_exec")
    async def test_volume_up(self, mock_exec):
        mock_proc = AsyncMock()
        mock_proc.communicate.return_value = (b"", b"")
        mock_proc.returncode = 0
        mock_exec.return_value = mock_proc

        result = await system_control.handler({"action": "volume_up"})
        assert "Volume alzato" in result

    @pytest.mark.asyncio
    @patch("asyncio.create_subprocess_exec")
    async def test_volume_get_returns_value(self, mock_exec):
        mock_proc = AsyncMock()
        mock_proc.communicate.return_value = (b"75", b"")
        mock_proc.returncode = 0
        mock_exec.return_value = mock_proc

        result = await system_control.handler({"action": "volume_get"})
        assert "75" in result
        assert "percento" in result

    @pytest.mark.asyncio
    @patch("asyncio.create_subprocess_exec")
    async def test_osascript_failure(self, mock_exec):
        mock_proc = AsyncMock()
        mock_proc.communicate.return_value = (b"", b"some error")
        mock_proc.returncode = 1
        mock_exec.return_value = mock_proc

        result = await system_control.handler({"action": "volume_up"})
        assert "Errore" in result

    @pytest.mark.asyncio
    @patch("asyncio.create_subprocess_exec", side_effect=FileNotFoundError)
    async def test_command_not_found(self, mock_exec):
        result = await system_control.handler({"action": "volume_up"})
        assert "Comando non trovato" in result

    def test_definition_structure(self):
        assert system_control.DEFINITION["function"]["name"] == "system_control"
        params = system_control.DEFINITION["function"]["parameters"]
        assert "action" in params["properties"]
        assert "enum" in params["properties"]["action"]


class DummyAgent:
    def __init__(self):
        self.registered: list[str] = []

    def register_tool(self, definition, _handler):
        self.registered.append(definition["function"]["name"])


class TestToolRegistration:
    @pytest.fixture(autouse=True)
    def clear_fast_intents(self):
        clear_intents()
        yield
        clear_intents()

    @pytest.mark.asyncio
    async def test_system_control_disabled(self):
        agent = DummyAgent()
        feedback = MagicMock()
        cfg = ToolsConfig(allow_system_control=False)
        register_all_tools(agent, cfg, feedback, store=SQLiteStore(":memory:"))

        assert "system_control" not in agent.registered
        assert await match_and_execute("alza il volume") is None

    @pytest.mark.asyncio
    async def test_system_control_enabled(self):
        agent = DummyAgent()
        feedback = MagicMock()
        cfg = ToolsConfig(allow_system_control=True)
        register_all_tools(agent, cfg, feedback, store=SQLiteStore(":memory:"))

        assert "system_control" in agent.registered

    @pytest.mark.asyncio
    async def test_timer_status_fast_intent(self):
        agent = DummyAgent()
        feedback = MagicMock()
        cfg = ToolsConfig()
        register_all_tools(agent, cfg, feedback, store=SQLiteStore(":memory:"))

        result = await match_and_execute("quanto manca al timer?")

        assert "Non ci sono timer attivi" in result
