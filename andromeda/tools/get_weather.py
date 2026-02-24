# Copyright (c) 2026 Alessandro Orrù
# Licensed under MIT

import logging
import time
import httpx
from dataclasses import dataclass, field
from andromeda.messages import msg, weather_condition
from andromeda.tools.http_client import request_with_retry

logger = logging.getLogger("[ TOOL GET WEATHER ]")


_CACHE_TTL_SEC: float = 300.0
_CACHE_MAX_SIZE: int = 50


@dataclass
class _WeatherState:
    timeout_sec: float = 10.0
    cache: dict[str, tuple[str, float]] = field(default_factory=dict)


_state = _WeatherState()

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
        return msg("weather.missing_city")

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
            timeout_sec=_state.timeout_sec,
        )
        geo_data = geo_resp.json()

        results = geo_data.get("results", [])
        if not results:
            return msg("weather.city_not_found", city=city)

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
            timeout_sec=_state.timeout_sec,
        )
        weather_data = weather_resp.json()
        current = weather_data.get("current", {})
        temp = current.get("temperature_2m", "N/D")
        humidity = current.get("relative_humidity_2m", "N/D")
        wind = current.get("wind_speed_10m", "N/D")
        code = current.get("weather_code", -1)
        condition = weather_condition(code)
        result = msg(
            "weather.output",
            city=city_name,
            condition=condition,
            temp=temp,
            humidity=humidity,
            wind=wind,
        )

        # Cache the result (evict oldest if full)
        if len(_state.cache) >= _CACHE_MAX_SIZE:
            oldest_key = min(_state.cache, key=lambda k: _state.cache[k][1])
            del _state.cache[oldest_key]
        _state.cache[cache_key] = (result, time.monotonic())

        return result

    except httpx.ConnectError:
        logger.error("Cannot connect to Open-Meteo API")
        return msg("weather.connection_error")
    except httpx.TimeoutException:
        logger.error("Open-Meteo request timed out")
        return msg("weather.timeout")
    except RuntimeError:
        logger.error("Open-Meteo circuit breaker open")
        return msg("weather.unavailable")
    except Exception:
        logger.exception("Weather tool failed")
        return msg("weather.generic_error")
