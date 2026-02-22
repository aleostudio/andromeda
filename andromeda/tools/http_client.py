# Copyright (c) 2026 Alessandro Orrù
# Licensed under MIT

import logging
import httpx

logger = logging.getLogger(__name__)

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


# Close the shared HTTP client (call on shutdown)
async def close_client() -> None:
    global _client
    if _client is not None and not _client.is_closed:
        await _client.aclose()
        _client = None
        logger.debug("Shared HTTP client closed")
