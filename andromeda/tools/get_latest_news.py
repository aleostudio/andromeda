# Copyright (c) 2026 Alessandro Orrù
# Licensed under MIT

import logging
import re
import time
import httpx
from dataclasses import dataclass, field
from datetime import datetime
from bs4 import BeautifulSoup
from andromeda.messages import msg
from andromeda.tools.http_client import request_with_retry

logger = logging.getLogger("[ TOOL GET LATEST NEWS ]")


_CACHE_TTL_SEC: float = 600.0
_CACHE_MAX_SIZE: int = 50


@dataclass
class _NewsState:
    timeout_sec: float = 10.0
    cache: dict[str, tuple[str, float]] = field(default_factory=dict)


_state = _NewsState()


DEFINITION = {
    "type": "function",
    "function": {
        "name": "get_latest_news",
        "description": (
            "Recupera le ultime notizie dal sito Il Post (ilpost.it). "
            "Usa questo strumento quando l'utente chiede le ultime notizie, "
            "cosa sta succedendo nel mondo, le news del giorno, o informazioni di attualità."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "category": {
                    "type": "string",
                    "description": (
                        "Categoria di notizie da cercare. Valori possibili: 'homepage' (default, notizie principali), "
                        "'italia', 'mondo', 'politica', 'economia', 'sport', 'cultura', 'tecnologia', 'scienza', 'internet'."
                    ),
                    "enum": ["homepage", "italia", "mondo", "politica", "economia", "sport", "cultura", "tecnologia", "scienza", "internet"],
                    "default": "homepage",
                },
                "limit": {
                    "type": "integer",
                    "description": "Numero massimo di notizie da restituire (default: 5, max: 15).",
                    "default": 5,
                },
            },
            "required": [],
        },
    },
}


_CATEGORY_URLS = {
    "homepage": "https://www.ilpost.it",
    "italia": "https://www.ilpost.it/italia/",
    "mondo": "https://www.ilpost.it/mondo/",
    "politica": "https://www.ilpost.it/politica/",
    "economia": "https://www.ilpost.it/economia/",
    "sport": "https://www.ilpost.it/sport/",
    "cultura": "https://www.ilpost.it/cultura/",
    "tecnologia": "https://www.ilpost.it/tecnologia/",
    "scienza": "https://www.ilpost.it/scienza/",
    "internet": "https://www.ilpost.it/internet/",
}

_ARTICLE_URL_RE = re.compile(
    r"^https://www\.ilpost\.it/\d{4}/\d{2}/\d{2}/[\w-]+/?$",
)


def _normalize_href(href: str) -> str:
    return href.rstrip("/") + "/"


def _extract_summary(link_element, title: str) -> str:
    for sibling in link_element.parent.find_all(["p", "span"], recursive=True):
        text = sibling.get_text(strip=True)
        if text != title and len(text) > 20:
            return text

    return ""


def _parse_article(link) -> dict | None:
    href = _normalize_href(link.get("href", ""))
    if not _ARTICLE_URL_RE.match(href):
        return None

    heading = link.select_one("h1, h2, h3, h4")
    title = heading.get_text(strip=True) if heading else ""
    if not title:
        return None

    return {
        "title": title, 
        "summary": _extract_summary(link, title), 
        "url": href
    }


def _parse_articles(html: str, limit: int) -> list[dict]:
    soup = BeautifulSoup(html, "html.parser")
    seen_urls: set[str] = set()
    articles: list[dict] = []

    for link in soup.select("a[href]"):
        article = _parse_article(link)
        if not article or article["url"] in seen_urls:
            continue

        seen_urls.add(article["url"])
        articles.append(article)

        if len(articles) >= limit:
            break

    return articles


async def _fetch_page(url: str) -> str:
    response = await request_with_retry("GET", url, timeout_sec=_state.timeout_sec)

    return response.text


def configure(timeout_sec: float) -> None:
    _state.timeout_sec = timeout_sec
    _state.cache = {}


async def handler(args: dict) -> str:
    category = args.get("category", "homepage")
    limit = min(max(args.get("limit", 5), 1), 15)
    url = _CATEGORY_URLS.get(category, _CATEGORY_URLS["homepage"])

    # Check cache first
    cache_key = f"{category}:{limit}"
    cached = _state.cache.get(cache_key)
    if cached is not None:
        result, ts = cached
        if (time.monotonic() - ts) < _CACHE_TTL_SEC:
            logger.debug("News cache hit for '%s'", cache_key)
            return result

    try:
        html = await _fetch_page(url)
        articles = _parse_articles(html, limit)

        if not articles:
            return msg("news.none_found", category=category)

        now = datetime.now().strftime("%d/%m/%Y %H:%M")

        news = msg("news.output_header", category=category.upper(), now=now)
        for i, art in enumerate(articles, 1):
            news = news + str(i) + ": " + art['title'] + ". "

        # Cache the result (evict oldest if full)
        if len(_state.cache) >= _CACHE_MAX_SIZE:
            oldest_key = min(_state.cache, key=lambda k: _state.cache[k][1])
            del _state.cache[oldest_key]
        _state.cache[cache_key] = (news, time.monotonic())

        return news

    except httpx.HTTPStatusError as e:
        logger.error("Il Post HTTP error: %s", e)
        return msg("news.http_error", status=e.response.status_code)
    except httpx.RequestError as e:
        logger.error("Il Post request error: %s", e)
        return msg("news.connection_error", error=e)
    except RuntimeError:
        logger.error("Il Post circuit breaker open")
        return msg("news.unavailable")
    except Exception as e:
        logger.error("Il Post unexpected error: %s", e)
        return msg("news.generic_error", error=e)
