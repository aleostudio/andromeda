# Copyright (c) 2026 Alessandro Orrù
# Licensed under MIT

import asyncio
import json
import logging
from andromeda.config import HealthCheckConfig

logger = logging.getLogger("[ HEALTH ]")


# Lightweight async HTTP health check endpoint using raw asyncio (no extra deps)
class HealthCheckServer:

    def __init__(self, config: HealthCheckConfig) -> None:
        self._config = config
        self._server: asyncio.Server | None = None
        self._state_provider = None
        self._metrics_provider = None
        self._uptime_start: float = 0.0


    # Set a callable that returns the current assistant state string
    def set_state_provider(self, provider) -> None:
        self._state_provider = provider


    # Set a callable that returns metrics summary dict
    def set_metrics_provider(self, provider) -> None:
        self._metrics_provider = provider


    async def start(self) -> None:
        if not self._config.enabled:
            return

        import time
        self._uptime_start = time.monotonic()

        self._server = await asyncio.start_server(
            self._handle_connection,
            self._config.host,
            self._config.port,
        )
        logger.info("Health check server started on %s:%d", self._config.host, self._config.port)


    async def stop(self) -> None:
        if self._server:
            self._server.close()
            await self._server.wait_closed()
            self._server = None
            logger.info("Health check server stopped")


    async def _handle_connection(self, reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
        try:
            # Read the HTTP request (we only care about the first line)
            request_line = await asyncio.wait_for(reader.readline(), timeout=5.0)

            # Drain remaining headers
            while True:
                line = await asyncio.wait_for(reader.readline(), timeout=5.0)
                if line == b"\r\n" or line == b"\n" or not line:
                    break

            request_str = request_line.decode("utf-8", errors="replace").strip()
            logger.debug("Health check request: %s", request_str)

            body = self._build_response()
            response = (
                "HTTP/1.1 200 OK\r\n"
                "Content-Type: application/json\r\n"
                f"Content-Length: {len(body)}\r\n"
                "Connection: close\r\n"
                "\r\n"
                f"{body}"
            )

            writer.write(response.encode("utf-8"))
            await writer.drain()

        except Exception:
            logger.debug("Health check connection error", exc_info=True)
        finally:
            writer.close()


    def _build_response(self) -> str:
        import time

        data = {"status": "ok", "uptime_sec": round(time.monotonic() - self._uptime_start, 1) if self._uptime_start else 0}

        if self._state_provider:
            try:
                data["state"] = str(self._state_provider())
            except Exception:
                data["state"] = "unknown"

        if self._metrics_provider:
            try:
                data["metrics"] = self._metrics_provider()
            except Exception:
                pass

        return json.dumps(data)
