# Copyright (c) 2026 Alessandro Orrù
# Licensed under MIT

import asyncio
import json
import pytest
from andromeda.config import HealthCheckConfig
from andromeda.health import HealthCheckServer


def _run(coro):
    """Helper to run async code in tests without pytest-asyncio."""
    return asyncio.get_event_loop().run_until_complete(coro)


class TestHealthCheckInit:
    def test_init_disabled(self):
        server = HealthCheckServer(HealthCheckConfig(enabled=False))
        assert server._server is None

    def test_init_enabled(self):
        server = HealthCheckServer(HealthCheckConfig(enabled=True))
        assert server._server is None  # Not started yet


class TestHealthCheckStartStop:
    def test_start_disabled_noop(self):
        server = HealthCheckServer(HealthCheckConfig(enabled=False))
        _run(server.start())
        assert server._server is None

    def test_start_stop_enabled(self):
        server = HealthCheckServer(
            HealthCheckConfig(enabled=True, port=18080),
        )
        _run(server.start())
        assert server._server is not None
        _run(server.stop())
        assert server._server is None

    def test_stop_without_start(self):
        server = HealthCheckServer(HealthCheckConfig(enabled=False))
        _run(server.stop())  # Should not crash


class TestHealthCheckProviders:
    def test_set_state_provider(self):
        server = HealthCheckServer(HealthCheckConfig())
        server.set_state_provider(lambda: "IDLE")
        assert server._state_provider is not None

    def test_set_metrics_provider(self):
        server = HealthCheckServer(HealthCheckConfig())
        server.set_metrics_provider(lambda: {"stt": {}})
        assert server._metrics_provider is not None


class TestBuildResponse:
    def test_basic_response(self):
        server = HealthCheckServer(HealthCheckConfig())
        body = server._build_response()
        data = json.loads(body)
        assert data["status"] == "ok"
        assert "uptime_sec" in data

    def test_with_state_provider(self):
        server = HealthCheckServer(HealthCheckConfig())
        server.set_state_provider(lambda: "LISTENING")
        body = server._build_response()
        data = json.loads(body)
        assert data["state"] == "LISTENING"

    def test_with_metrics_provider(self):
        server = HealthCheckServer(HealthCheckConfig())
        server.set_metrics_provider(
            lambda: {"stt": {"avg_ms": 100}},
        )
        body = server._build_response()
        data = json.loads(body)
        assert "metrics" in data
        assert data["metrics"]["stt"]["avg_ms"] == 100

    def test_state_provider_error(self):
        server = HealthCheckServer(HealthCheckConfig())
        server.set_state_provider(lambda: 1 / 0)
        body = server._build_response()
        data = json.loads(body)
        assert data["state"] == "unknown"

    def test_metrics_provider_error(self):
        server = HealthCheckServer(HealthCheckConfig())
        server.set_metrics_provider(lambda: 1 / 0)
        body = server._build_response()
        data = json.loads(body)
        assert "metrics" not in data


class TestHealthCheckHTTP:
    def test_http_response(self):
        async def _test():
            server = HealthCheckServer(
                HealthCheckConfig(enabled=True, port=18081),
            )
            server.set_state_provider(lambda: "IDLE")
            await server.start()

            try:
                reader, writer = await asyncio.open_connection(
                    "127.0.0.1", 18081,
                )
                writer.write(
                    b"GET / HTTP/1.1\r\nHost: localhost\r\n\r\n",
                )
                await writer.drain()

                response = await asyncio.wait_for(
                    reader.read(4096), timeout=5.0,
                )
                response_str = response.decode("utf-8")

                assert "HTTP/1.1 200 OK" in response_str
                assert "application/json" in response_str

                # Extract JSON body
                body = response_str.split("\r\n\r\n", 1)[1]
                data = json.loads(body)
                assert data["status"] == "ok"
                assert data["state"] == "IDLE"

                writer.close()
            finally:
                await server.stop()

        _run(_test())
