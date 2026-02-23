# Copyright (c) 2026 Alessandro Orrù
# Licensed under MIT

import asyncio
import json
import tempfile
from datetime import datetime
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch
import pytest
from andromeda.tools import get_datetime, knowledge_base, set_timer, system_control


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
        store_path = str(tmp_path / "knowledge.json")
        knowledge_base.configure(store_path)
        yield store_path

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
        knowledge_base.handler({"action": "save", "key": "password_wifi", "value": "secret"})
        result = knowledge_base.handler({"action": "recall", "key": "wifi"})
        assert "secret" in result

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
        """Data survives reload (new handler calls load from file)."""
        knowledge_base.handler({"action": "save", "key": "persist", "value": "data123"})

        # Verify file exists and contains data
        store = json.loads(Path(temp_store).read_text())
        assert store["persist"] == "data123"

    def test_definition_structure(self):
        assert knowledge_base.DEFINITION["function"]["name"] == "knowledge_base"
        params = knowledge_base.DEFINITION["function"]["parameters"]
        assert "action" in params["properties"]
        assert "key" in params["properties"]
        assert "value" in params["properties"]


class TestSetTimer:
    @pytest.fixture(autouse=True)
    def setup_timer(self):
        mock_feedback = MagicMock()
        set_timer.configure(mock_feedback, max_sec=3600)
        set_timer._active_timers.clear()
        yield
        # Cancel any remaining timers
        for task in set_timer._active_timers.values():
            task.cancel()
        set_timer._active_timers.clear()

    def test_invalid_seconds_string(self):
        result = set_timer.handler({"seconds": "abc"})
        assert "Errore" in result

    def test_zero_seconds(self):
        result = set_timer.handler({"seconds": 0})
        assert "Errore" in result

    def test_negative_seconds(self):
        result = set_timer.handler({"seconds": -5})
        assert "Errore" in result

    def test_exceeds_max(self):
        result = set_timer.handler({"seconds": 99999})
        assert "Errore" in result
        assert "massima" in result

    @pytest.mark.asyncio
    async def test_valid_seconds(self):
        result = set_timer.handler({"seconds": 30})
        assert "timer" in result.lower()
        assert "30 secondi" in result

    @pytest.mark.asyncio
    async def test_valid_minutes(self):
        result = set_timer.handler({"seconds": 300, "label": "pasta"})
        assert "pasta" in result
        assert "5 minuti" in result

    @pytest.mark.asyncio
    async def test_minutes_and_seconds(self):
        result = set_timer.handler({"seconds": 90})
        assert "1 minuti" in result
        assert "30 secondi" in result

    @pytest.mark.asyncio
    async def test_custom_label(self):
        result = set_timer.handler({"seconds": 60, "label": "bucato"})
        assert "bucato" in result

    def test_definition_structure(self):
        assert set_timer.DEFINITION["function"]["name"] == "set_timer"
        params = set_timer.DEFINITION["function"]["parameters"]
        assert "seconds" in params["properties"]
        assert "label" in params["properties"]


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
