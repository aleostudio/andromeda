# Copyright (c) 2026 Alessandro Orrù
# Licensed under MIT

import logging
import time
import httpx
from dataclasses import dataclass, field
from andromeda.tools.http_client import request_with_retry

logger = logging.getLogger("[ TOOL GET WEATHER ]")


_CACHE_TTL_SEC: float = 300.0
_CACHE_MAX_SIZE: int = 50


@dataclass
class _WeatherState:
    timeout_sec: float = 10.0
    cache: dict[str, tuple[str, float]] = field(default_factory=dict)


_state = _WeatherState()

# WMO weather codes to Italian descriptions
_WMO_CODES = {
    0: "cielo sereno",
    1: "prevalentemente sereno",
    2: "parzialmente nuvoloso",
    3: "coperto",
    45: "nebbia",
    48: "nebbia con brina",
    51: "pioggerella leggera",
    53: "pioggerella moderata",
    55: "pioggerella intensa",
    61: "pioggia leggera",
    63: "pioggia moderata",
    65: "pioggia forte",
    71: "neve leggera",
    73: "neve moderata",
    75: "neve forte",
    80: "rovesci leggeri",
    81: "rovesci moderati",
    82: "rovesci violenti",
    95: "temporale",
    96: "temporale con grandine leggera",
    99: "temporale con grandine forte",
}


DEFINITION = {
    "type": "function",
    "function": {
        "name": "get_weather",
        "description": (
            "Ottieni il meteo corrente per una città. "
            "Usa questo strumento quando l'utente chiede che tempo fa, la temperatura, o le previsioni meteo."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "city": {
                    "type": "string",
                    "description": "Nome della città (es. 'Roma', 'Milano', 'Napoli')",
                },
            },
            "required": ["city"],
        },
    },
}


def configure(timeout_sec: float) -> None:
    _state.timeout_sec = timeout_sec
    _state.cache = {}


async def handler(args: dict) -> str:
    city = args.get("city", "").strip()
    if not city:
        return "Errore: nessuna città specificata."

    # Check cache first
    cache_key = city.lower()
    cached = _state.cache.get(cache_key)
    if cached is not None:
        result, ts = cached
        if (time.monotonic() - ts) < _CACHE_TTL_SEC:
            logger.debug("Weather cache hit for '%s'", city)
            return result

    try:
        # Geocode city name to coordinates
        geo_resp = await request_with_retry(
            "GET",
            "https://geocoding-api.open-meteo.com/v1/search",
            params={"name": city, "count": 1, "language": "it"},
            timeout=_state.timeout_sec,
        )
        geo_data = geo_resp.json()

        results = geo_data.get("results", [])
        if not results:
            return f"Non ho trovato la città '{city}'."

        loc = results[0]
        lat, lon = loc["latitude"], loc["longitude"]
        city_name = loc.get("name", city)

        # Fetch current weather
        weather_resp = await request_with_retry(
            "GET",
            "https://api.open-meteo.com/v1/forecast",
            params={
                "latitude": lat,
                "longitude": lon,
                "current": "temperature_2m,relative_humidity_2m,weather_code,wind_speed_10m",
                "timezone": "auto",
            },
            timeout=_state.timeout_sec,
        )
        weather_data = weather_resp.json()
        current = weather_data.get("current", {})
        temp = current.get("temperature_2m", "N/D")
        humidity = current.get("relative_humidity_2m", "N/D")
        wind = current.get("wind_speed_10m", "N/D")
        code = current.get("weather_code", -1)
        condition = _WMO_CODES.get(code, "condizioni sconosciute")

        result = (
            f"Meteo a {city_name}: {condition}, "
            f"temperatura {temp}°C, "
            f"umidità {humidity}%, "
            f"vento {wind} km/h"
        )

        # Cache the result (evict oldest if full)
        if len(_state.cache) >= _CACHE_MAX_SIZE:
            oldest_key = min(_state.cache, key=lambda k: _state.cache[k][1])
            del _state.cache[oldest_key]
        _state.cache[cache_key] = (result, time.monotonic())

        return result

    except httpx.ConnectError:
        logger.error("Cannot connect to Open-Meteo API")
        return "Non riesco a connettermi al servizio meteo. Verifica la connessione internet."
    except httpx.TimeoutException:
        logger.error("Open-Meteo request timed out")
        return "La richiesta meteo ha impiegato troppo tempo."
    except RuntimeError:
        logger.error("Open-Meteo circuit breaker open")
        return "Il servizio meteo è temporaneamente non disponibile. Riprova tra poco."
    except Exception:
        logger.exception("Weather tool failed")
        return "Errore nel recupero dei dati meteo."
