# Copyright (c) 2026 Alessandro Orrù
# Licensed under MIT

import logging
from andromeda.agent import AIAgent
from andromeda.config import ToolsConfig
from andromeda.feedback import AudioFeedback
from andromeda.intent import clear_intents, register_intent
from andromeda.tools import (
    get_datetime, 
    get_latest_news, 
    get_weather, 
    knowledge_base, 
    set_timer, 
    system_control, 
    web_search, 
    # query_mongodb,
    # query_postgres,
)

logger = logging.getLogger("[ TOOLS ]")


# Tools list
_TOOLS = [
    get_datetime, 
    get_latest_news, 
    get_weather, 
    set_timer,
    knowledge_base, 
    web_search,

    # Disabled - just for test
    # query_mongodb,
    # query_postgres,
]


# Register all available tools with the AI agent
def register_all_tools(agent: AIAgent, tools_cfg: ToolsConfig, feedback: AudioFeedback) -> None:
    clear_intents()
    set_timer.configure(feedback, tools_cfg.timer_max_sec)
    knowledge_base.configure(tools_cfg.knowledge_base_path, tools_cfg.allow_sensitive_memory)
    get_weather.configure(tools_cfg.weather_timeout_sec)
    get_latest_news.configure(tools_cfg.news_timeout_sec)
    web_search.configure(
        tools_cfg.web_search_timeout_sec,
        tools_cfg.web_search_max_results,
        tools_cfg.web_search_max_content_chars,
        tools_cfg.web_search_fetch_page_content,
    )

    for tool_module in _TOOLS:
        agent.register_tool(tool_module.DEFINITION, tool_module.handler)

    if tools_cfg.allow_system_control:
        agent.register_tool(system_control.DEFINITION, system_control.handler)

    # Fast intents — bypass LLM for simple, deterministic requests
    register_intent(patterns=[r"\b(che\s+)?or[ae]\b", r"\bche\s+ore\s+sono\b"], tool_handler=get_datetime.handler)
    register_intent(patterns=[r"\b(che\s+)?giorno\b", r"\b(che\s+)?data\b"], tool_handler=get_datetime.handler)
    intents_count = 2

    if tools_cfg.allow_system_control:
        register_intent(patterns=[r"\balza.*volume\b", r"\bvolume.*alto\b", r"\bpiù\s+forte\b"], tool_handler=system_control.handler, args={"action": "volume_up"})
        register_intent(patterns=[r"\babbassa.*volume\b", r"\bvolume.*basso\b", r"\bpiù\s+piano\b"], tool_handler=system_control.handler, args={"action": "volume_down"})
        register_intent(patterns=[r"\bmut[ao]\b.*\b(volume|audio)\b", r"\b(volume|audio)\b.*\bmut[ao]\b", r"\bsilenzi[oa]\b"], tool_handler=system_control.handler, args={"action": "volume_mute"})
        intents_count += 3

    tools_count = len(_TOOLS) + (1 if tools_cfg.allow_system_control else 0)
    logger.info("Registered %d tools, %d fast intents", tools_count, intents_count)
