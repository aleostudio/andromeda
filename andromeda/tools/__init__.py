# Copyright (c) 2026 Alessandro Orrù
# Licensed under MIT

from andromeda.agent import AIAgent
from andromeda.config import ToolsConfig
from andromeda.feedback import AudioFeedback
from andromeda.intent import register_intent
from andromeda.tools import (get_datetime, get_latest_news, get_weather, knowledge_base, set_timer, system_control)
import logging

logger = logging.getLogger(__name__)


_TOOLS = [get_datetime, get_latest_news, get_weather, set_timer, system_control, knowledge_base]


def register_all_tools(agent: AIAgent, tools_cfg: ToolsConfig, feedback: AudioFeedback) -> None:
    """Register all available tools with the AI agent."""

    set_timer.configure(feedback, tools_cfg.timer_max_sec)
    knowledge_base.configure(tools_cfg.knowledge_base_path)
    get_weather.configure(tools_cfg.weather_timeout_sec)

    for tool_module in _TOOLS:
        agent.register_tool(tool_module.DEFINITION, tool_module.handler)

    # Fast intents — bypass LLM for simple, deterministic requests
    register_intent(patterns=[r"\b(che\s+)?or[ae]\b", r"\bche\s+ore\s+sono\b"], tool_handler=get_datetime.handler)
    register_intent(patterns=[r"\b(che\s+)?giorno\b", r"\b(che\s+)?data\b"], tool_handler=get_datetime.handler)
    register_intent(patterns=[r"\balza.*volume\b", r"\bvolume.*alto\b", r"\bpiù\s+forte\b"], tool_handler=system_control.handler, args={"action": "volume_up"})
    register_intent(patterns=[r"\babbassa.*volume\b", r"\bvolume.*basso\b", r"\bpiù\s+piano\b"], tool_handler=system_control.handler, args={"action": "volume_down"})
    register_intent(patterns=[r"\bmut[ao]\b.*\b(volume|audio)\b", r"\b(volume|audio)\b.*\bmut[ao]\b", r"\bsilenzi[oa]\b"], tool_handler=system_control.handler, args={"action": "volume_mute"})

    logger.info("Registered %d tools, %d fast intents", len(_TOOLS), 5)
