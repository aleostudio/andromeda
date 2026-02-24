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
        self._server = await asyncio.start_server(self._handle_connection, self._config.host, self._config.port)
        logger.info("Health check server started on %s:%d", self._config.host, self._config.port)


    async def stop(self) -> None:
        if self._server:
            self._server.close()
            await self._server.wait_closed()
            self._server = None
            logger.info("Health check server stopped")


    async def _handle_connection(self, reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
        try:
            # Read the HTTP request (we only care about the first line, limit to 8KB)
            raw_request = await asyncio.wait_for(reader.read(8192), timeout=5.0)
            request = raw_request.decode("utf-8", errors="replace")
            request_line = request.splitlines()[0] if request else ""
            logger.debug("Health check request: %s", request_line)

            method, path, is_valid = self._parse_request_line(request_line)
            if not is_valid:
                writer.write(self._http_response(400, {"status": "bad_request"}))
                await writer.drain()
                return

            if method != "GET":
                writer.write(self._http_response(405, {"status": "method_not_allowed"}))
                await writer.drain()
                return

            if path != "/":
                writer.write(self._http_response(404, {"status": "not_found"}))
                await writer.drain()
                return

            writer.write(self._http_response(200, self._build_response_data()))
            await writer.drain()

        except Exception:
            logger.debug("Health check connection error", exc_info=True)
        finally:
            writer.close()
            try:
                await writer.wait_closed()
            except Exception:
                pass


    @staticmethod
    def _parse_request_line(request_line: str) -> tuple[str, str, bool]:
        parts = request_line.split()
        if len(parts) < 3:
            return "", "", False

        method, path, http_version = parts[0], parts[1], parts[2]
        if not http_version.startswith("HTTP/"):
            return "", "", False

        return method.upper(), path, True


    @staticmethod
    def _http_response(status_code: int, payload: dict) -> bytes:
        reason_map = {
            200: "OK",
            400: "Bad Request",
            404: "Not Found",
            405: "Method Not Allowed",
        }
        body = json.dumps(payload).encode("utf-8")
        reason = reason_map.get(status_code, "OK")
        headers = (
            f"HTTP/1.1 {status_code} {reason}\r\n"
            "Content-Type: application/json\r\n"
            f"Content-Length: {len(body)}\r\n"
            "Connection: close\r\n"
            "\r\n"
        ).encode("utf-8")

        return headers + body


    def _build_response_data(self) -> dict:
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

        return data


    # Backward-compatible helper used by tests
    def _build_response(self) -> str:
        return json.dumps(self._build_response_data())
