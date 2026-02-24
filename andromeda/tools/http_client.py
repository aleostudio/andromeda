# Copyright (c) 2026 Alessandro Orrù
# Licensed under MIT

import logging
import asyncio
import time
from dataclasses import dataclass
from urllib.parse import urlparse
import httpx

logger = logging.getLogger("[ TOOL HTTP CLIENT ]")

_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/131.0.0.0 Safari/537.36"
    ),
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    "Accept-Language": "it-IT,it;q=0.9,en;q=0.8",
}

_client: httpx.AsyncClient | None = None
_circuit_state: dict[str, dict[str, float]] = {}
_CIRCUIT_FAIL_THRESHOLD = 3
_CIRCUIT_OPEN_SEC = 20.0


# Return the shared HTTP client, creating it on first use
def get_client() -> httpx.AsyncClient:
    global _client
    if _client is None or _client.is_closed:
        _client = httpx.AsyncClient(
            headers=_HEADERS,
            timeout=httpx.Timeout(15.0, connect=5.0),
            follow_redirects=True,
        )
        logger.debug("Shared HTTP client created")

    return _client


def _circuit_key(url: str) -> str:
    parsed = urlparse(url)

    return parsed.netloc or parsed.path


def _is_circuit_open(key: str) -> bool:
    state = _circuit_state.get(key)
    if not state:
        return False

    return state.get("open_until", 0.0) > time.monotonic()


def _mark_success(key: str) -> None:
    if key in _circuit_state:
        _circuit_state[key] = {"fails": 0.0, "open_until": 0.0}


def _mark_failure(key: str) -> None:
    state = _circuit_state.get(key, {"fails": 0.0, "open_until": 0.0})
    fails = state["fails"] + 1.0
    open_until = state["open_until"]
    if fails >= _CIRCUIT_FAIL_THRESHOLD:
        open_until = time.monotonic() + _CIRCUIT_OPEN_SEC
        logger.warning("Circuit opened for %s", key)
    _circuit_state[key] = {"fails": fails, "open_until": open_until}


def _is_retryable_status(status_code: int) -> bool:
    return status_code == 429 or status_code >= 500


def _build_http_error(response: httpx.Response) -> httpx.HTTPStatusError:
    return httpx.HTTPStatusError(
        f"HTTP {response.status_code}",
        request=response.request,
        response=response,
    )


async def _request_once(
    client: httpx.AsyncClient,
    method: str,
    url: str,
    *,
    params: dict | None = None,
    timeout_sec: float | None = None,
) -> httpx.Response:
    try:
        if timeout_sec is None:
            return await client.request(method, url, params=params)
        async with asyncio.timeout(timeout_sec):
            return await client.request(method, url, params=params)
    except TimeoutError as e:
        raise httpx.TimeoutException("Request timeout") from e


def _classify_http_error(error: httpx.HTTPStatusError) -> tuple[bool, bool]:
    retryable = _is_retryable_status(error.response.status_code)

    return retryable, not retryable


@dataclass
class _AttemptResult:
    response: httpx.Response | None = None
    error: Exception | None = None
    should_retry: bool = False
    should_mark_failure: bool = False


async def _attempt_request(
    client: httpx.AsyncClient,
    method: str,
    url: str,
    *,
    params: dict | None = None,
    timeout_sec: float | None = None,
) -> _AttemptResult:
    try:
        response = await _request_once(
            client,
            method,
            url,
            params=params,
            timeout_sec=timeout_sec,
        )
        if response.status_code >= 400:
            raise _build_http_error(response)

        return _AttemptResult(response=response)

    except (httpx.ConnectError, httpx.TimeoutException, httpx.NetworkError) as e:
        return _AttemptResult(error=e, should_retry=True, should_mark_failure=True)
    except httpx.HTTPStatusError as e:
        _should_retry, should_raise = _classify_http_error(e)
        if should_raise:
            return _AttemptResult(error=e)

        return _AttemptResult(error=e, should_retry=True, should_mark_failure=True)


async def request_with_retry(
    method: str,
    url: str,
    *,
    params: dict | None = None,
    timeout_sec: float | None = None,
    retries: int = 2,
    backoff_sec: float = 0.25,
) -> httpx.Response:
    key = _circuit_key(url)
    if _is_circuit_open(key):
        raise RuntimeError("Circuit open")

    client = get_client()
    attempts = retries + 1

    for attempt in range(attempts):
        result = await _attempt_request(
            client,
            method,
            url,
            params=params,
            timeout_sec=timeout_sec,
        )
        if result.response is not None:
            _mark_success(key)
            return result.response

        if result.should_mark_failure:
            _mark_failure(key)

        if not result.should_retry:
            raise result.error or RuntimeError("HTTP request failed")

        if attempt < attempts - 1:
            await asyncio.sleep(backoff_sec * (2 ** attempt))

    raise result.error or RuntimeError("HTTP request failed")


# Close the shared HTTP client (call on shutdown)
async def close_client() -> None:
    global _client, _circuit_state
    if _client is not None and not _client.is_closed:
        await _client.aclose()
        _client = None
        logger.debug("Shared HTTP client closed")
    _circuit_state = {}
