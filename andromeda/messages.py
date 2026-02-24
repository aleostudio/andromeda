# Copyright (c) 2026 Alessandro Orrù
# Licensed under MIT

from datetime import datetime

_MESSAGES = {
    "it": {
        "core.no_speech_retry": "Non ho sentito nulla. Riprova.",
        "core.not_understood_retry": "Non ho capito. Puoi ripetere?",
        "core.generic_error_retry": "Si è verificato un errore. Riprova.",
        "core.processing_error_retry": "Ho riscontrato un errore. Ripeti la domanda.",
        "agent.ollama_unreachable": "Non riesco a connettermi al modello. Verifica che Ollama sia in esecuzione.",
        "agent.ollama_timeout": "La richiesta ha impiegato troppo tempo. Riprova.",
        "agent.request_too_complex": "Mi dispiace, la richiesta è troppo complessa. Puoi riformulare?",
        "system.volume_up": "Volume alzato",
        "system.volume_down": "Volume abbassato",
        "system.volume_toggle": "Audio mutato o smutato",
        "system.brightness_up": "Luminosità aumentata",
        "system.brightness_down": "Luminosità diminuita",
        "system.unsupported_platform": "Controllo di sistema non supportato su questa piattaforma ({platform}).",
        "system.unknown_action": "Azione '{action}' non riconosciuta. Azioni disponibili: {available}",
        "system.action_unavailable": "Questa azione non è disponibile sulla piattaforma corrente.",
        "system.command_error": "Errore nell'esecuzione del comando: {error}",
        "system.current_volume": "Il volume attuale è al {value} percento.",
        "system.command_not_found": "Comando non trovato. Assicurati di avere installato {hint}.",
        "system.generic_error": "Errore nel controllo di sistema.",
        "system.tool_hint_default": "gli strumenti di sistema",
        "system.tool_hint_darwin": "osascript (incluso in macOS)",
        "system.tool_hint_linux": "pactl (PulseAudio) e brightnessctl",
        "system.tool_hint_windows": "nircmd (nirsoft.net)",
        "datetime.output": "Data: {date}, Ora: {time}",
        "timer.invalid_seconds": "Errore: la durata deve essere un numero intero di secondi.",
        "timer.non_positive": "Errore: la durata deve essere maggiore di zero.",
        "timer.max_exceeded": "Errore: la durata massima è {max_sec} secondi ({max_min} minuti).",
        "timer.set": "Timer '{label}' impostato per {duration}.",
        "timer.duration_minutes_seconds": "{minutes} minuti e {seconds} secondi",
        "timer.duration_minutes": "{minutes} minuti",
        "timer.duration_seconds": "{seconds} secondi",
        "kb.invalid_action": "Azione '{action}' non riconosciuta. Usa: save, recall, list, delete.",
        "kb.save_missing_fields": "Errore: serve sia una chiave che un valore per salvare.",
        "kb.sensitive_blocked": "Dato sensibile rilevato. Per salvare questa informazione imposta allow_sensitive a true dopo conferma esplicita.",
        "kb.saved": "Ho memorizzato '{key}': {value}",
        "kb.recall_missing_key": "Errore: specifica quale informazione vuoi recuperare.",
        "kb.recall_not_found": "Non ho trovato nulla per '{key}'.",
        "kb.recall_matches": "Ho trovato queste corrispondenze: {matches}",
        "kb.empty": "La memoria è vuota, non ho ancora salvato nulla.",
        "kb.list": "Informazioni memorizzate: {keys}",
        "kb.delete_missing_key": "Errore: specifica quale informazione vuoi eliminare.",
        "kb.delete_not_found": "'{key}' non è presente in memoria.",
        "kb.deleted": "Ho eliminato '{key}' dalla memoria.",
        "weather.missing_city": "Errore: nessuna città specificata.",
        "weather.city_not_found": "Non ho trovato la città '{city}'.",
        "weather.output": "Meteo a {city}: {condition}, temperatura {temp}°C, umidità {humidity}%, vento {wind} km/h",
        "weather.connection_error": "Non riesco a connettermi al servizio meteo. Verifica la connessione internet.",
        "weather.timeout": "La richiesta meteo ha impiegato troppo tempo.",
        "weather.unavailable": "Il servizio meteo è temporaneamente non disponibile. Riprova tra poco.",
        "weather.generic_error": "Errore nel recupero dei dati meteo.",
        "weather.unknown_condition": "condizioni sconosciute",
        "news.none_found": "Nessuna notizia trovata per la categoria '{category}' su Il Post.",
        "news.output_header": "Ultime notizie Il Post - {category} (aggiornate al {now}): ",
        "news.http_error": "Errore nel recupero delle notizie: HTTP {status}",
        "news.connection_error": "Errore di connessione a Il Post: {error}",
        "news.unavailable": "Il servizio notizie è temporaneamente non disponibile. Riprova tra poco.",
        "news.generic_error": "Errore imprevisto nel recupero delle notizie: {error}",
        "web.offline": "Non conosco la risposta e al momento non posso cercare online. Riprova quando sarà disponibile una connessione.",
        "web.no_results": "Non ho trovato risultati utili per questa ricerca. Non so rispondere.",
        "web.search_output": "Risultati della ricerca web per '{query}': ",
        "web.page_content": "Contenuto pagina: {content}",
        "web.timeout": "La ricerca web ha impiegato troppo tempo.",
        "web.unavailable": "La ricerca web è temporaneamente non disponibile. Riprova tra poco.",
        "web.generic_error": "Errore nella ricerca web. Non so rispondere.",
        "web.url_not_allowed": "L'URL richiesto non è consentito per ragioni di sicurezza.",
        "web.page_extract_failed": "Non sono riuscito a estrarre contenuto dalla pagina {url}.",
        "web.page_content_output": "Contenuto della pagina {url}: {content}",
        "web.page_timeout": "La richiesta alla pagina {url} ha impiegato troppo tempo.",
        "web.page_unavailable": "Il recupero pagina è temporaneamente non disponibile. Riprova tra poco.",
        "web.page_generic_error": "Errore nel recupero della pagina {url}. Non so rispondere.",
        "web.missing_query_or_url": "Errore: specifica 'query' o 'url'.",
    },
    "en": {
        "core.no_speech_retry": "I could not hear anything. Please try again.",
        "core.not_understood_retry": "I did not understand. Could you repeat?",
        "core.generic_error_retry": "An error occurred. Please try again.",
        "core.processing_error_retry": "I encountered an error. Please repeat your request.",
        "agent.ollama_unreachable": "I cannot connect to the model. Please verify that Ollama is running.",
        "agent.ollama_timeout": "The request took too long. Please try again.",
        "agent.request_too_complex": "I am sorry, this request is too complex. Could you rephrase it?",
        "system.volume_up": "Volume increased",
        "system.volume_down": "Volume decreased",
        "system.volume_toggle": "Audio muted or unmuted",
        "system.brightness_up": "Brightness increased",
        "system.brightness_down": "Brightness decreased",
        "system.unsupported_platform": "System control is not supported on this platform ({platform}).",
        "system.unknown_action": "Action '{action}' is not recognized. Available actions: {available}",
        "system.action_unavailable": "This action is not available on the current platform.",
        "system.command_error": "Error while running the command: {error}",
        "system.current_volume": "The current volume is {value} percent.",
        "system.command_not_found": "Command not found. Make sure you have installed {hint}.",
        "system.generic_error": "System control error.",
        "system.tool_hint_default": "system tools",
        "system.tool_hint_darwin": "osascript (included in macOS)",
        "system.tool_hint_linux": "pactl (PulseAudio) and brightnessctl",
        "system.tool_hint_windows": "nircmd (nirsoft.net)",
        "datetime.output": "Date: {date}, Time: {time}",
        "timer.invalid_seconds": "Error: duration must be an integer number of seconds.",
        "timer.non_positive": "Error: duration must be greater than zero.",
        "timer.max_exceeded": "Error: maximum duration is {max_sec} seconds ({max_min} minutes).",
        "timer.set": "Timer '{label}' set for {duration}.",
        "timer.duration_minutes_seconds": "{minutes} minutes and {seconds} seconds",
        "timer.duration_minutes": "{minutes} minutes",
        "timer.duration_seconds": "{seconds} seconds",
        "kb.invalid_action": "Action '{action}' is not recognized. Use: save, recall, list, delete.",
        "kb.save_missing_fields": "Error: both key and value are required to save.",
        "kb.sensitive_blocked": "Sensitive data detected. To save this information, set allow_sensitive to true after explicit confirmation.",
        "kb.saved": "I saved '{key}': {value}",
        "kb.recall_missing_key": "Error: specify which information you want to retrieve.",
        "kb.recall_not_found": "I could not find anything for '{key}'.",
        "kb.recall_matches": "I found these matches: {matches}",
        "kb.empty": "Memory is empty, I have not saved anything yet.",
        "kb.list": "Saved information: {keys}",
        "kb.delete_missing_key": "Error: specify which information you want to delete.",
        "kb.delete_not_found": "'{key}' is not present in memory.",
        "kb.deleted": "I removed '{key}' from memory.",
        "weather.missing_city": "Error: no city specified.",
        "weather.city_not_found": "I could not find the city '{city}'.",
        "weather.output": "Weather in {city}: {condition}, temperature {temp}°C, humidity {humidity}%, wind {wind} km/h",
        "weather.connection_error": "I cannot connect to the weather service. Please check your internet connection.",
        "weather.timeout": "The weather request took too long.",
        "weather.unavailable": "The weather service is temporarily unavailable. Please try again soon.",
        "weather.generic_error": "Error while retrieving weather data.",
        "weather.unknown_condition": "unknown conditions",
        "news.none_found": "No news found for category '{category}' on Il Post.",
        "news.output_header": "Latest Il Post news - {category} (updated at {now}): ",
        "news.http_error": "Error retrieving news: HTTP {status}",
        "news.connection_error": "Connection error to Il Post: {error}",
        "news.unavailable": "The news service is temporarily unavailable. Please try again soon.",
        "news.generic_error": "Unexpected error while retrieving news: {error}",
        "web.offline": "I do not know the answer and I cannot search online right now. Please try again when a connection is available.",
        "web.no_results": "I could not find useful results for this search. I cannot answer.",
        "web.search_output": "Web search results for '{query}': ",
        "web.page_content": "Page content: {content}",
        "web.timeout": "The web search took too long.",
        "web.unavailable": "Web search is temporarily unavailable. Please try again soon.",
        "web.generic_error": "Web search error. I cannot answer.",
        "web.url_not_allowed": "The requested URL is not allowed for security reasons.",
        "web.page_extract_failed": "I could not extract content from page {url}.",
        "web.page_content_output": "Content of page {url}: {content}",
        "web.page_timeout": "The request to page {url} took too long.",
        "web.page_unavailable": "Page retrieval is temporarily unavailable. Please try again soon.",
        "web.page_generic_error": "Error retrieving page {url}. I cannot answer.",
        "web.missing_query_or_url": "Error: specify 'query' or 'url'.",
    },
}

_DAYS = {
    "it": ["lunedì", "martedì", "mercoledì", "giovedì", "venerdì", "sabato", "domenica"],
    "en": ["monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday"],
}

_MONTHS = {
    "it": ["", "gennaio", "febbraio", "marzo", "aprile", "maggio", "giugno", "luglio", "agosto", "settembre", "ottobre", "novembre", "dicembre"],
    "en": ["", "january", "february", "march", "april", "may", "june", "july", "august", "september", "october", "november", "december"],
}

_WEATHER_CODES = {
    "it": {
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
    },
    "en": {
        0: "clear sky",
        1: "mostly clear",
        2: "partly cloudy",
        3: "overcast",
        45: "fog",
        48: "rime fog",
        51: "light drizzle",
        53: "moderate drizzle",
        55: "heavy drizzle",
        61: "light rain",
        63: "moderate rain",
        65: "heavy rain",
        71: "light snow",
        73: "moderate snow",
        75: "heavy snow",
        80: "light showers",
        81: "moderate showers",
        82: "violent showers",
        95: "thunderstorm",
        96: "thunderstorm with light hail",
        99: "thunderstorm with heavy hail",
    },
}

_locale = "it"


def _normalize_locale(locale: str | None) -> str:
    if not locale:
        return "it"
    lowered = locale.lower()
    if lowered.startswith("en"):
        return "en"

    return "it"


def set_locale(locale: str | None) -> None:
    global _locale
    _locale = _normalize_locale(locale)


def get_locale() -> str:
    return _locale


def msg(message_key: str, **kwargs) -> str:
    template = _MESSAGES.get(_locale, _MESSAGES["it"]).get(message_key)
    if template is None:
        template = _MESSAGES["it"].get(message_key, message_key)
    if kwargs:
        return template.format(**kwargs)

    return template


def get_localized_datetime(now: datetime) -> tuple[str, str]:
    locale = get_locale()
    day_name = _DAYS[locale][now.weekday()]
    month_name = _MONTHS[locale][now.month]
    date_text = f"{day_name} {now.day} {month_name} {now.year}"

    return date_text, now.strftime("%H:%M")


def weather_condition(code: int) -> str:
    locale = get_locale()

    return _WEATHER_CODES[locale].get(code, msg("weather.unknown_condition"))
