# Copyright (c) 2026 Alessandro Orrù
# Licensed under MIT

import logging
import re
import ipaddress
import time
import httpx
from dataclasses import dataclass, field
from urllib.parse import parse_qs, unquote, urlparse
from bs4 import BeautifulSoup
from andromeda.tools.http_client import request_with_retry

logger = logging.getLogger("[ TOOL WEB SEARCH ]")


_CACHE_TTL_SEC: float = 300.0  # 5 minutes
_CACHE_MAX_SIZE: int = 100
_CONNECTIVITY_TTL_SEC: float = 30.0  # re-check every 30s
_CONNECTIVITY_TIMEOUT_SEC: float = 3.0
_CONNECTIVITY_PROBES: tuple[tuple[str, str], ...] = (
    ("HEAD", "https://html.duckduckgo.com/html/"),
    ("GET", "https://www.gstatic.com/generate_204"),
)
_OFFLINE_MSG = "Non conosco la risposta e al momento non posso cercare online. Riprova quando sarà disponibile una connessione."
_INJECTION_PATTERNS = (
    re.compile(r"ignore\s+(all|any|the)\s+(previous|prior)\s+instructions?", re.IGNORECASE),
    re.compile(r"system\s+prompt", re.IGNORECASE),
    re.compile(r"developer\s+message", re.IGNORECASE),
    re.compile(r"jailbreak", re.IGNORECASE),
    re.compile(r"act\s+as\s+", re.IGNORECASE),
)


@dataclass
class _WebSearchState:
    timeout_sec: float = 10.0
    max_results: int = 3
    max_content_chars: int = 2000
    fetch_page_content: bool = False
    cache: dict[str, tuple[str, float]] = field(default_factory=dict)
    connectivity_cache: tuple[bool, float] = (False, 0.0)


_state = _WebSearchState()


DEFINITION = {
    "type": "function",
    "function": {
        "name": "web_search",
        "description": (
            "Cerca informazioni sul web o visita una pagina specifica. "
            "Usa questo strumento SOLO come ultima risorsa, quando non conosci "
            "la risposta e hai bisogno di cercare informazioni aggiornate su internet. "
            "Modalità 1: passa 'query' per cercare su DuckDuckGo. "
            "Modalità 2: passa 'url' per visitare una pagina web ed estrarne il contenuto. "
            "Puoi combinare le due modalità: prima cerca con 'query' per trovare il sito giusto, "
            "poi usa 'url' per leggere la pagina."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "Query di ricerca per DuckDuckGo (es. 'ultima versione kubernetes', 'sito ufficiale pytorch')"
                },
                "url": {
                    "type": "string",
                    "description": (
                        "URL di una pagina web da visitare e leggere (es. 'https://kubernetes.io/releases/'). "
                        "Usa questo parametro per accedere direttamente a una pagina specifica ed estrarne il contenuto."
                    ),
                },
            },
            "required": [],
        },
    },
}


def configure(timeout_sec: float, max_results: int, max_content_chars: int, fetch_page_content: bool) -> None:
    _state.timeout_sec = timeout_sec
    _state.max_results = max_results
    _state.max_content_chars = max_content_chars
    _state.fetch_page_content = fetch_page_content
    _state.cache = {}
    _state.connectivity_cache = (False, 0.0)


def _is_allowed_url(url: str) -> bool:
    parsed = urlparse(url)
    if parsed.scheme not in ("http", "https"):
        return False

    hostname = (parsed.hostname or "").lower()
    if not hostname:
        return False
    if hostname in ("localhost",) or hostname.endswith(".local"):
        return False

    try:
        ip = ipaddress.ip_address(hostname)
        if ip.is_private or ip.is_loopback or ip.is_link_local or ip.is_reserved:
            return False
    except ValueError:
        pass

    return True


# Extract real URL from DuckDuckGo redirect wrapper
def _extract_url(raw_href: str) -> str:
    parsed = urlparse(raw_href)
    uddg = parse_qs(parsed.query).get("uddg", [""])[0]

    return unquote(uddg) if uddg else raw_href


# Parse DuckDuckGo HTML lite results
def _parse_search_results(html: str, limit: int) -> list[dict]:
    soup = BeautifulSoup(html, "html.parser")
    results: list[dict] = []

    for div in soup.select(".result.results_links"):
        if len(results) >= limit:
            break

        link = div.select_one("a.result__a")
        snippet_el = div.select_one(".result__snippet")
        if not link:
            continue

        title = link.get_text(strip=True)
        raw_href = link.get("href", "")
        url = _extract_url(raw_href)
        snippet = snippet_el.get_text(strip=True) if snippet_el else ""

        if title and snippet:
            results.append({"title": title, "url": url, "snippet": snippet})

    return results


# Fetch a page and extract its main text content
async def _fetch_and_extract(url: str) -> str:
    if not _is_allowed_url(url):
        logger.warning("Blocked URL fetch for unsafe host: %s", url)
        return ""

    try:
        resp = await request_with_retry("GET", url, timeout=_state.timeout_sec)
    except Exception:
        logger.debug("Failed to fetch page content from %s", url)
        return ""

    soup = BeautifulSoup(resp.text, "html.parser")

    # Remove non-content elements
    for tag in soup.select("script, style, nav, header, footer, aside, iframe, noscript"):
        tag.decompose()

    # Try to find main content area
    main = (
        soup.select_one("article")
        or soup.select_one("main")
        or soup.select_one("body")
    )
    if not main:
        return ""

    return _sanitize_content(main.get_text(separator=" ", strip=True))


def _sanitize_content(text: str) -> str:
    if not text:
        return ""
    cleaned = text
    for pattern in _INJECTION_PATTERNS:
        cleaned = pattern.sub(" ", cleaned)

    return " ".join(cleaned.split())


# Quick connectivity probe with caching
async def _check_connectivity() -> bool:
    is_online, checked_at = _state.connectivity_cache
    if (time.monotonic() - checked_at) < _CONNECTIVITY_TTL_SEC:
        return is_online

    online = False
    for method, url in _CONNECTIVITY_PROBES:
        try:
            resp = await request_with_retry(method, url, timeout=_CONNECTIVITY_TIMEOUT_SEC, retries=0)
            if resp.status_code < 500:
                online = True
                break
        except Exception:
            continue

    _state.connectivity_cache = (online, time.monotonic())
    logger.debug("Connectivity check: %s", "online" if online else "offline")

    return online


# Search DuckDuckGo and return formatted results
async def _handle_search(query: str) -> str:
    try:
        search_resp = await request_with_retry(
            "GET",
            "https://html.duckduckgo.com/html/",
            params={"q": query},
            timeout=_state.timeout_sec,
        )
        results = _parse_search_results(search_resp.text, _state.max_results)
        if not results:
            return ("Non ho trovato risultati utili per questa ricerca. Non so rispondere.")

        # Optionally fetch full page content of top result
        if _state.fetch_page_content:
            page_text = await _fetch_and_extract(results[0]["url"])
            if page_text:
                results[0]["content"] = page_text[:_state.max_content_chars]

        # Format output with URLs so LLM can follow up
        output = f"Risultati della ricerca web per '{query}': "

        for i, r in enumerate(results, 1):
            output += f"{i}. {r['title']} ({r['url']}): {r['snippet']} "
            if r.get("content"):
                output += f"Contenuto pagina: {r['content']} "

        return output

    except httpx.ConnectError:
        logger.error("Cannot connect to DuckDuckGo")
        return _OFFLINE_MSG
    except httpx.TimeoutException:
        logger.error("DuckDuckGo request timed out")
        return "La ricerca web ha impiegato troppo tempo."
    except RuntimeError:
        logger.error("Web search circuit breaker open")
        return "La ricerca web è temporaneamente non disponibile. Riprova tra poco."
    except Exception:
        logger.exception("Web search failed")
        return "Errore nella ricerca web. Non so rispondere."


# Fetch a specific URL and return its text content
async def _handle_fetch(url: str) -> str:
    if not _is_allowed_url(url):
        return "L'URL richiesto non è consentito per ragioni di sicurezza."

    try:
        page_text = await _fetch_and_extract(url)

        if not page_text:
            return f"Non sono riuscito a estrarre contenuto dalla pagina {url}."

        truncated = page_text[:_state.max_content_chars]
        return f"Contenuto della pagina {url}: {truncated}"

    except httpx.ConnectError:
        logger.error("Cannot connect to %s", url)
        return _OFFLINE_MSG
    except httpx.TimeoutException:
        logger.error("Request to %s timed out", url)
        return f"La richiesta alla pagina {url} ha impiegato troppo tempo."
    except RuntimeError:
        logger.error("Page fetch circuit breaker open for %s", url)
        return "Il recupero pagina è temporaneamente non disponibile. Riprova tra poco."
    except Exception:
        logger.exception("Page fetch failed for %s", url)
        return f"Errore nel recupero della pagina {url}. Non so rispondere."


async def handler(args: dict) -> str:
    query = args.get("query", "").strip()
    url = args.get("url", "").strip()

    if not query and not url:
        return "Errore: specifica 'query' o 'url'."
    if url and not _is_allowed_url(url):
        return "L'URL richiesto non è consentito per ragioni di sicurezza."

    # Build cache key from both params
    cache_key = f"q:{query.lower()}|u:{url.lower()}"
    cached = _state.cache.get(cache_key)
    if cached is not None:
        result, ts = cached
        if (time.monotonic() - ts) < _CACHE_TTL_SEC:
            logger.debug("Web search cache hit for '%s'", cache_key)
            return result

    # Verify internet connectivity before attempting
    if not await _check_connectivity():
        logger.info("Web search skipped: device is offline")
        return _OFFLINE_MSG

    # Dispatch based on parameters
    if query and url:
        search_result = await _handle_search(query)
        page_result = await _handle_fetch(url)
        output = f"{search_result} {page_result}"
    elif url:
        output = await _handle_fetch(url)
    else:
        output = await _handle_search(query)

    # Cache the result (evict oldest if full)
    if len(_state.cache) >= _CACHE_MAX_SIZE:
        oldest_key = min(_state.cache, key=lambda k: _state.cache[k][1])
        del _state.cache[oldest_key]
    _state.cache[cache_key] = (output, time.monotonic())

    return output
