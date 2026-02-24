# Copyright (c) 2026 Alessandro Orrù
# Licensed under MIT

from datetime import datetime
import pytest
from andromeda import messages


class TestMessages:
    @pytest.fixture(autouse=True)
    def restore_locale(self):
        messages.set_locale("it")
        yield
        messages.set_locale("it")

    def test_default_locale_is_it(self):
        messages.set_locale(None)
        assert messages.get_locale() == "it"
        assert "Non ho sentito nulla" in messages.msg("core.no_speech_retry")

    def test_english_locale(self):
        messages.set_locale("en")
        assert messages.get_locale() == "en"
        assert "could not hear anything" in messages.msg("core.no_speech_retry")

    def test_datetime_localization(self):
        now = datetime(2026, 2, 24, 13, 45)
        messages.set_locale("en")
        date_text_en, time_text_en = messages.get_localized_datetime(now)
        assert "february" in date_text_en
        assert time_text_en == "13:45"

        messages.set_locale("it")
        date_text_it, time_text_it = messages.get_localized_datetime(now)
        assert "febbraio" in date_text_it
        assert time_text_it == "13:45"
