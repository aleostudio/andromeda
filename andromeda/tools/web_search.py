# Copyright (c) 2026 Alessandro Orrù
# Licensed under MIT

import logging
import time
import httpx
from urllib.parse import parse_qs, unquote, urlparse
from bs4 import BeautifulSoup
from andromeda.tools.http_client import get_client

logger = logging.getLogger("[ TOOL WEB SEARCH ]")


_timeout_sec: float = 10.0
_max_results: int = 3
_max_content_chars: int = 2000
_fetch_page_content: bool = False

# Result cache: maps cache_key -> (result_str, timestamp)
_cache: dict[str, tuple[str, float]] = {}
_CACHE_TTL_SEC: float = 300.0  # 5 minutes

# Connectivity check cache to avoid repeated probes
_connectivity_cache: tuple[bool, float] = (False, 0.0)
_CONNECTIVITY_TTL_SEC: float = 30.0  # re-check every 30s
_CONNECTIVITY_TIMEOUT_SEC: float = 3.0
_OFFLINE_MSG = "Non conosco la risposta e al momento non posso cercare online. Riprova quando sarà disponibile una connessione."


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
    global _timeout_sec, _max_results
    global _max_content_chars, _fetch_page_content
    _timeout_sec = timeout_sec
    _max_results = max_results
    _max_content_chars = max_content_chars
    _fetch_page_content = fetch_page_content


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
    try:
        client = get_client()
        resp = await client.get(url, timeout=_timeout_sec)
        resp.raise_for_status()
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

    return main.get_text(separator=" ", strip=True)


# Quick connectivity probe with caching
async def _check_connectivity() -> bool:
    global _connectivity_cache
    is_online, checked_at = _connectivity_cache
    if (time.monotonic() - checked_at) < _CONNECTIVITY_TTL_SEC:
        return is_online

    try:
        client = get_client()
        resp = await client.head("https://1.1.1.1", timeout=_CONNECTIVITY_TIMEOUT_SEC)
        online = resp.status_code < 500
    except Exception:
        online = False

    _connectivity_cache = (online, time.monotonic())
    logger.debug("Connectivity check: %s", "online" if online else "offline")

    return online


# Search DuckDuckGo and return formatted results
async def _handle_search(query: str) -> str:
    try:
        client = get_client()
        search_resp = await client.get("https://html.duckduckgo.com/html/", params={"q": query}, timeout=_timeout_sec)
        search_resp.raise_for_status()
        results = _parse_search_results(search_resp.text, _max_results)
        if not results:
            return ("Non ho trovato risultati utili per questa ricerca. Non so rispondere.")

        # Optionally fetch full page content of top result
        if _fetch_page_content:
            page_text = await _fetch_and_extract(results[0]["url"])
            if page_text:
                results[0]["content"] = page_text[:_max_content_chars]

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
    except Exception:
        logger.exception("Web search failed")
        return "Errore nella ricerca web. Non so rispondere."


# Fetch a specific URL and return its text content
async def _handle_fetch(url: str) -> str:
    try:
        page_text = await _fetch_and_extract(url)

        if not page_text:
            return f"Non sono riuscito a estrarre contenuto dalla pagina {url}."

        truncated = page_text[:_max_content_chars]
        return f"Contenuto della pagina {url}: {truncated}"

    except httpx.ConnectError:
        logger.error("Cannot connect to %s", url)
        return _OFFLINE_MSG
    except httpx.TimeoutException:
        logger.error("Request to %s timed out", url)
        return f"La richiesta alla pagina {url} ha impiegato troppo tempo."
    except Exception:
        logger.exception("Page fetch failed for %s", url)
        return f"Errore nel recupero della pagina {url}. Non so rispondere."


async def handler(args: dict) -> str:
    query = args.get("query", "").strip()
    url = args.get("url", "").strip()

    if not query and not url:
        return "Errore: specifica 'query' o 'url'."

    # Build cache key from both params
    cache_key = f"q:{query.lower()}|u:{url.lower()}"
    cached = _cache.get(cache_key)
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

    # Cache the result
    _cache[cache_key] = (output, time.monotonic())

    return output
