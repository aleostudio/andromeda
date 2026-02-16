from andromeda.agent import AIAgent
from andromeda.config import ToolsConfig
from andromeda.feedback import AudioFeedback
from andromeda.tools import (get_datetime, get_weather, knowledge_base, set_timer, system_control)
import logging

logger = logging.getLogger(__name__)


_TOOLS = [get_datetime, get_weather, set_timer, system_control, knowledge_base]


def register_all_tools(agent: AIAgent, tools_cfg: ToolsConfig, feedback: AudioFeedback) -> None:
    """Register all available tools with the AI agent."""

    set_timer.configure(feedback, tools_cfg.timer_max_sec)
    knowledge_base.configure(tools_cfg.knowledge_base_path)
    get_weather.configure(tools_cfg.weather_timeout_sec)

    for tool_module in _TOOLS:
        agent.register_tool(tool_module.DEFINITION, tool_module.handler)

    logger.info("Registered %d tools", len(_TOOLS))
