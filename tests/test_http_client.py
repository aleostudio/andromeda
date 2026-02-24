# Copyright (c) 2026 Alessandro Orrù
# Licensed under MIT

import asyncio
import httpx
import pytest
from andromeda.tools import http_client


class OkClient:
    async def request(self, method, url, params=None, _timeout=None):
        await asyncio.sleep(0)
        return httpx.Response(
            200,
            request=httpx.Request(method, url, params=params),
            text="ok",
        )


class FlakyClient:
    def __init__(self):
        self.calls = 0

    async def request(self, method, url, params=None, _timeout=None):
        await asyncio.sleep(0)
        self.calls += 1
        if self.calls == 1:
            raise httpx.TimeoutException("timeout")
        return httpx.Response(
            200,
            request=httpx.Request(method, url, params=params),
            text="ok",
        )


class FailingClient:
    async def request(self, method, url, params=None, _timeout=None):
        await asyncio.sleep(0)
        raise httpx.ConnectError("connect", request=httpx.Request(method, url))


class TestHttpClientResilience:
    @pytest.fixture(autouse=True)
    def clean_state(self):
        http_client._circuit_state.clear()
        yield
        http_client._circuit_state.clear()

    @pytest.mark.asyncio
    async def test_request_success(self, monkeypatch):
        monkeypatch.setattr(http_client, "get_client", lambda: OkClient())

        response = await http_client.request_with_retry(
            "GET",
            "https://example.com",
        )

        assert response.status_code == 200

    @pytest.mark.asyncio
    async def test_request_retry_then_success(self, monkeypatch):
        client = FlakyClient()
        monkeypatch.setattr(http_client, "get_client", lambda: client)

        response = await http_client.request_with_retry(
            "GET",
            "https://example.com",
            retries=2,
        )

        assert response.status_code == 200
        assert client.calls == 2

    @pytest.mark.asyncio
    async def test_circuit_breaker_opens(self, monkeypatch):
        monkeypatch.setattr(http_client, "get_client", lambda: FailingClient())

        with pytest.raises(httpx.ConnectError):
            await http_client.request_with_retry("GET", "https://example.com", retries=0)
        with pytest.raises(httpx.ConnectError):
            await http_client.request_with_retry("GET", "https://example.com", retries=0)
        with pytest.raises(httpx.ConnectError):
            await http_client.request_with_retry("GET", "https://example.com", retries=0)
        with pytest.raises(RuntimeError, match="Circuit open"):
            await http_client.request_with_retry("GET", "https://example.com", retries=0)
