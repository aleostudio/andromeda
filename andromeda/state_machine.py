from typing import Any, Callable, Coroutine, TypeAlias
from enum import Enum, auto
import asyncio
import logging


logger = logging.getLogger(__name__)

StateHandler: TypeAlias = Callable[["AssistantState"], Coroutine[Any, Any, "AssistantState"]]


class AssistantState(Enum):
    IDLE = auto()          # Listening for wake word only
    LISTENING = auto()     # Recording user speech
    PROCESSING = auto()    # STT + Agent inference
    SPEAKING = auto()      # TTS playback
    ERROR = auto()         # Recoverable error, transition back to IDLE

    def __str__(self) -> str:
        return self.name


# Valid state transitions
TRANSITIONS: dict[AssistantState, set[AssistantState]] = {
    AssistantState.IDLE: {AssistantState.LISTENING},
    AssistantState.LISTENING: {AssistantState.PROCESSING, AssistantState.IDLE, AssistantState.ERROR},
    AssistantState.PROCESSING: {AssistantState.SPEAKING, AssistantState.IDLE, AssistantState.ERROR},
    AssistantState.SPEAKING: {AssistantState.IDLE, AssistantState.LISTENING, AssistantState.ERROR},
    AssistantState.ERROR: {AssistantState.IDLE},
}


# Manages assistant state transitions with validation and event dispatching
class StateMachine:

    def __init__(self) -> None:
        self._state = AssistantState.IDLE
        self._handlers: dict[AssistantState, StateHandler] = {}
        self._on_transition: list[Callable[[AssistantState, AssistantState], None]] = []
        self._lock = asyncio.Lock()


    @property
    def state(self) -> AssistantState:
        return self._state


    # Register async handler for a state. Handler returns next state
    def register_handler(self, state: AssistantState, handler: StateHandler) -> None:
        self._handlers[state] = handler


    # Register callback for any state transition
    def on_transition(self, callback: Callable[[AssistantState, AssistantState], None]) -> None:
        self._on_transition.append(callback)


    # Validate and perform state transition
    async def transition_to(self, new_state: AssistantState) -> None:
        async with self._lock:
            if new_state not in TRANSITIONS.get(self._state, set()):
                logger.warning("Invalid transition: %s -> %s (allowed: %s)", self._state, new_state, TRANSITIONS.get(self._state, set()))
                return

            old_state = self._state
            self._state = new_state
            logger.info("State: %s -> %s", old_state, new_state)

            for cb in self._on_transition:
                try:
                    cb(old_state, new_state)
                except Exception:
                    logger.exception("Transition callback error")


    # Main loop: execute handler for current state, transition to returned state
    async def run(self) -> None:
        logger.info("State machine started in %s", self._state)
        while True:
            handler = self._handlers.get(self._state)
            if handler is None:
                logger.error("No handler for state %s, returning to IDLE", self._state)
                await self.transition_to(AssistantState.IDLE)
                await asyncio.sleep(0.1)
                continue

            try:
                next_state = await handler(self._state)
                await self.transition_to(next_state)
            except asyncio.CancelledError:
                logger.info("State machine cancelled")
                raise
            except Exception:
                logger.exception("Error in state %s", self._state)
                await self.transition_to(AssistantState.ERROR)
