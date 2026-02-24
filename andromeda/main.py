# Copyright (c) 2026 Alessandro Orrù
# Licensed under MIT

import warnings
warnings.filterwarnings("ignore", message="pkg_resources is deprecated")

import asyncio
import contextlib
import logging
import signal
import sys
import webrtcvad
from pathlib import Path
from andromeda.agent import AIAgent
from andromeda.audio_capture import AudioCapture
from andromeda.config import AppConfig
from andromeda.feedback import AudioFeedback
from andromeda.health import HealthCheckServer
from andromeda.intent import match_and_execute
from andromeda.messages import msg, set_locale
from andromeda.metrics import PerformanceMetrics
from andromeda.state_machine import AssistantState, StateMachine
from andromeda.stt import SpeechRecognizer
from andromeda.tools import register_all_tools
from andromeda.tools.http_client import close_client
from andromeda.tts import TextToSpeech
from andromeda.vad import VoiceActivityDetector
from andromeda.wake_word import WakeWordDetector

logger = logging.getLogger("[ MAIN ]")


# Main application class that wires all components together
class VoiceAssistant:

    def __init__(self, config: AppConfig) -> None:
        self._cfg = config

        # Components
        self._audio = AudioCapture(config.audio, config.noise)
        self._wake_word = WakeWordDetector(config.audio, config.wake_word)
        self._vad = VoiceActivityDetector(config.audio, config.vad)
        self._stt = SpeechRecognizer(config.stt, speech_pad_ms=config.vad.speech_pad_ms)
        self._agent = AIAgent(config.agent, config.conversation)
        self._tts = TextToSpeech(config.audio, config.tts)
        self._feedback = AudioFeedback(config.audio, config.feedback)
        self._metrics = PerformanceMetrics()
        self._health = HealthCheckServer(config.health_check)

        # State machine
        self._sm = StateMachine()
        self._sm.register_handler(AssistantState.IDLE, self._handle_idle)
        self._sm.register_handler(AssistantState.LISTENING, self._handle_listening)
        self._sm.register_handler(AssistantState.PROCESSING, self._handle_processing)
        self._sm.register_handler(AssistantState.SPEAKING, self._handle_speaking)
        self._sm.register_handler(AssistantState.ERROR, self._handle_error)

        # Shared state between handlers
        self._recorded_audio = None
        self._response_text = ""
        self._speech_energy: float = 0.0
        self._is_follow_up: bool = False  # True when listening for follow-up (no wake word needed)
        self._tts_interrupted: bool = False  # True when TTS was interrupted by wake word
        self._calibration_vad = webrtcvad.Vad(config.vad.aggressiveness)  # Reuse for calibration


    # Initialize all components. Call before run().
    def initialize(self) -> None:
        set_locale(self._cfg.stt.language)
        logger.info(".-------------------------------------------------------------.")
        logger.info("|                     _                              _        |")
        logger.info("|     /\\             | |                            | |       |")
        logger.info("|    /  \\   _ __   __| |_ __ ___  _ __ ___   ___  __| | __ _  |")
        logger.info("|   / /\\ \\ | '_ \\ / _` | '__/ _ \\| '_ ` _ \\ / _ \\/ _` |/ _` | |")
        logger.info("|  / ____ \\| | | | (_| | | | (_) | | | | | |  __/ (_| | (_| | |")
        logger.info("| /_/    \\_\\_| |_|\\__,_|_|  \\___/|_| |_| |_|\\___|\\__,_|\\__,_| |")
        logger.info("|                                                             |")
        logger.info("|          Smart home assistant completely offline            |")
        logger.info("|            alessandro.orru <at> aleostudio.com              |")
        logger.info("'-------------------------------------------------------------'")
        logger.info("")
        logger.info("Initializing home assistant...")

        # Raise every blocker exception
        try:
            self._feedback.initialize()
        except Exception:
            logger.exception("Failed to initialize audio feedback")

        try:
            self._wake_word.initialize()
        except Exception:
            logger.exception("Failed to initialize wake word detector")
            raise

        try:
            self._stt.initialize()
        except Exception:
            logger.exception("Failed to initialize STT engine")
            raise

        try:
            self._agent.initialize()
        except Exception:
            logger.exception("Failed to initialize AI agent")
            raise

        try:
            self._tts.initialize()
        except Exception:
            logger.exception("Failed to initialize TTS engine")
            raise

        if self._cfg.tts.prewarm_cache:
            try:
                self._tts.prewarm_cache()
            except Exception:
                logger.warning("Failed to pre-warm TTS cache")

        # Register tools
        try:
            register_all_tools(self._agent, self._cfg.tools, self._feedback)
        except Exception:
            logger.exception("Failed to register tools")

        # Wire audio callbacks
        self._audio.on_audio_frame(self._wake_word.process_frame)
        self._audio.on_audio_frame(self._vad.process_frame)

        # Fade out thinking tone when TTS starts playing
        self._tts.set_on_first_audio(self._feedback.stop)

        # Wire health check providers
        self._health.set_state_provider(lambda: self._sm.state)
        self._health.set_metrics_provider(self._metrics.get_summary)

        logger.info("All components initialized")


    # Run the assistant main loop
    async def run(self) -> None:
        try:
            await self._health.start()
        except Exception:
            logger.warning("Failed to start health check server")

        # Pre-warm Ollama model (load into GPU/RAM)
        if self._cfg.agent.prewarm:
            await self._agent.prewarm_model()

        try:
            self._audio.start()
        except Exception:
            logger.exception("Failed to start audio capture")
            raise

        logger.info("Voice assistant is running. Listening for wake word...")
        try:
            self._feedback.play("done")
        except Exception:
            logger.warning("Failed to play startup ready cue")

        try:
            await self._sm.run()
        finally:
            try:
                self._wake_word.shutdown()
            except Exception:
                logger.warning("Error shutting down wake word detector")
            try:
                self._audio.stop()
            except Exception:
                logger.warning("Error stopping audio capture")
            try:
                await self._health.stop()
            except Exception:
                logger.warning("Error stopping health check")
            try:
                self._metrics.log_summary()
            except Exception:
                logger.warning("Error logging metrics summary")


    # IDLE: Listen for wake word in background
    async def _handle_idle(self, _state: AssistantState) -> AssistantState:
        self._wake_word.reset()
        self._audio.unmute()
        self._is_follow_up = False

        # Wait for wake word detection (runs in thread via event)
        loop = asyncio.get_running_loop()
        detected = await loop.run_in_executor(None, lambda: self._wake_word.wait_for_detection(timeout=None))

        if detected:
            self._metrics.start_pipeline()

            # Calibrate speech energy from the ring buffer audio
            self._speech_energy = self._audio.calibrate_speech_energy(self._calibration_vad, self._cfg.audio.sample_rate)
            energy_threshold = self._speech_energy * self._cfg.vad.energy_threshold_factor
            self._vad.set_energy_threshold(energy_threshold)
            logger.info("Calibrated speech energy: %.1f, VAD energy threshold: %.1f", self._speech_energy, energy_threshold)
            self._feedback.play("wake")

            return AssistantState.LISTENING

        return AssistantState.IDLE


    # LISTENING: Record user speech until silence timeout
    async def _handle_listening(self, _state: AssistantState) -> AssistantState:
        if self._is_follow_up:
            return await self._handle_follow_up_listening()

        self._audio.start_recording()
        self._vad.start()

        # Wait for speech to end
        loop = asyncio.get_running_loop()
        await loop.run_in_executor(None, lambda: self._vad.wait_for_speech_end(timeout=self._cfg.vad.max_recording_sec + 1))

        self._vad.stop()
        self._recorded_audio = self._audio.stop_recording()

        # Reset energy threshold for next session
        self._vad.set_energy_threshold(0.0)

        # Check minimum recording
        min_samples = int(self._cfg.vad.min_recording_sec * self._cfg.audio.sample_rate)
        if len(self._recorded_audio) < min_samples or not self._vad.had_speech:
            logger.info("Recording too short or no speech detected, asking user to retry")
            await self._speak_error(msg("core.no_speech_retry"))
            return AssistantState.IDLE

        self._feedback.play("done")

        return AssistantState.PROCESSING


    # FOLLOW-UP LISTENING: Wait briefly for user to speak without wake word
    async def _handle_follow_up_listening(self) -> AssistantState:
        follow_up_timeout = self._cfg.conversation.follow_up_timeout_sec
        logger.info("Follow-up listening for %.1fs...", follow_up_timeout)

        # Reuse last calibrated energy for VAD threshold
        if self._speech_energy > 0:
            energy_threshold = self._speech_energy * self._cfg.vad.energy_threshold_factor
            self._vad.set_energy_threshold(energy_threshold)

        self._audio.start_recording()
        self._vad.start()

        # Wait for speech end OR follow-up timeout (whichever comes first)
        loop = asyncio.get_running_loop()
        await loop.run_in_executor(None, lambda: self._vad.wait_for_speech_end(timeout=follow_up_timeout))

        self._vad.stop()
        self._recorded_audio = self._audio.stop_recording()
        self._vad.set_energy_threshold(0.0)

        # Check if user actually spoke
        min_samples = int(self._cfg.vad.min_recording_sec * self._cfg.audio.sample_rate)
        if len(self._recorded_audio) < min_samples or not self._vad.had_speech:
            logger.info("No follow-up detected, back to IDLE")
            self._is_follow_up = False

            return AssistantState.IDLE

        logger.info("Follow-up speech detected")
        self._metrics.start_pipeline()
        self._feedback.play("done")

        return AssistantState.PROCESSING


    # PROCESSING: STT transcription + AI response + TTS
    async def _handle_processing(self, _state: AssistantState) -> AssistantState:
        # Transcribe
        with self._metrics.measure("stt"):
            text = await self._stt.transcribe(self._recorded_audio)

        if not text.strip():
            logger.info("Empty transcription, asking user to repeat")
            await self._speak_error(msg("core.not_understood_retry"))
            return AssistantState.SPEAKING

        logger.info("User said: %s", text)

        # Try fast intent match first (no LLM needed)
        with self._metrics.measure("intent_match"):
            fast_response = await match_and_execute(text)

        if fast_response:
            logger.info("Fast intent response: %s", fast_response[:80])
            self._response_text = fast_response
            self._audio.mute()
            try:
                with self._metrics.measure("tts"):
                    await self._tts.speak(fast_response)
            finally:
                self._audio.unmute()
            self._metrics.end_pipeline()

            return AssistantState.SPEAKING

        # No fast match — use LLM
        self._tts_interrupted = False
        self._feedback.play("thinking")
        try:
            if self._cfg.agent.streaming:
                with self._metrics.measure("llm_streaming"):
                    await self._process_streaming(text)
            else:
                with self._metrics.measure("llm"):
                    await self._process_standard(text)
        except Exception:
            logger.exception("LLM processing failed")
            await self._speak_error(msg("core.generic_error_retry"))
            self._response_text = msg("core.generic_error_retry")
        finally:
            self._feedback.stop()
            self._audio.unmute()

        self._metrics.end_pipeline()

        if self._tts_interrupted:
            logger.info("TTS was interrupted by voice command, returning to IDLE")
            return AssistantState.IDLE

        return AssistantState.SPEAKING


    # Standard mode: wait for full response, then speak
    async def _process_standard(self, text: str) -> None:
        self._audio.mute()
        self._response_text = await self._agent.process(text)

        # Enable wake word detection during TTS for voice interruption
        self._wake_word.reset()
        self._audio.monitor_only()

        monitor_task = asyncio.create_task(self._monitor_interrupt())
        with self._metrics.measure("tts"):
            await self._tts.speak(self._response_text)
        monitor_task.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await monitor_task


    # Streaming mode: speak sentence-by-sentence as LLM generates
    async def _process_streaming(self, text: str) -> None:
        sentence_queue: asyncio.Queue[str | None] = asyncio.Queue(maxsize=32)

        # Enable wake word detection during TTS for voice interruption
        self._wake_word.reset()
        self._audio.monitor_only()

        agent_task = asyncio.create_task(self._agent.process_streaming(text, sentence_queue))
        tts_task = asyncio.create_task(self._tts.speak_streamed(sentence_queue))
        monitor_task = asyncio.create_task(self._monitor_interrupt(tasks_to_cancel=[agent_task]))

        self._response_text = ""
        try:
            with contextlib.suppress(asyncio.CancelledError):
                self._response_text = await agent_task
        except Exception:
            logger.exception("Agent streaming failed")
            # Ensure TTS gets the sentinel so it doesn't hang forever
            with contextlib.suppress(Exception):
                await sentence_queue.put(None)

        await tts_task
        monitor_task.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await monitor_task


    # Monitor wake word during TTS playback to allow voice interruption
    async def _monitor_interrupt(self, tasks_to_cancel: list[asyncio.Task] | None = None) -> None:
        loop = asyncio.get_running_loop()
        logger.debug("Interrupt monitoring active")
        poll_count = 0

        while True:
            detected = await loop.run_in_executor(
                None, lambda: self._wake_word.wait_for_detection(timeout=0.5),
            )
            if detected:
                logger.info("Interrupt: wake word detected during speech")
                self._tts.stop_playback()
                self._tts_interrupted = True
                if tasks_to_cancel:
                    for task in tasks_to_cancel:
                        task.cancel()
                return

            # Periodically reset wake word model to prevent state accumulation
            # from TTS audio echo picked up by the microphone
            poll_count += 1
            if poll_count % 10 == 0:
                self._wake_word.reset_model_only()
                logger.debug("Interrupt monitor: model reset (poll %d)", poll_count)


    # SPEAKING: After TTS is done, decide whether to listen for follow-up
    async def _handle_speaking(self, _state: AssistantState) -> AssistantState:
        # yield to event loop (handler must be async for state machine)
        await asyncio.sleep(0)

        # After speaking, listen for follow-up instead of going back to IDLE
        if self._cfg.conversation.follow_up_timeout_sec > 0:
            self._is_follow_up = True
            return AssistantState.LISTENING

        return AssistantState.IDLE


    # ERROR: Speak error message and return to IDLE
    async def _handle_error(self, _state: AssistantState) -> AssistantState:
        try:
            await self._speak_error(msg("core.processing_error_retry"))
        except Exception:
            logger.exception("Failed to speak error message")

        return AssistantState.IDLE


    # Speak a short error message to the user
    async def _speak_error(self, message: str) -> None:
        logger.warning("Spoken error: %s", message)
        try:
            self._audio.mute()
        except Exception:
            logger.warning("Failed to mute audio for error message")
        try:
            await self._tts.speak(message)
        except Exception:
            logger.exception("TTS failed to speak error: %s", message)
        finally:
            try:
                self._audio.unmute()
            except Exception:
                logger.warning("Failed to unmute audio after error")


    # Release all resources. Unblocks threads waiting on events
    async def shutdown(self) -> None:
        try:
            self._wake_word.shutdown()
        except Exception:
            logger.warning("Error shutting down wake word detector")
        try:
            await self._agent.close()
        except Exception:
            logger.warning("Error closing AI agent")
        try:
            await self._health.stop()
        except Exception:
            logger.warning("Error stopping health check server")
        try:
            await close_client()
        except Exception:
            logger.warning("Error closing shared HTTP client")


# Logging setup
def setup_logging(config: AppConfig) -> None:
    logging.basicConfig(
        level=getattr(logging, config.logging.level),
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        handlers=[logging.StreamHandler(sys.stdout)],
    )


# Entrypoint
def main() -> None:
    config_path = Path("config.yaml")
    if len(sys.argv) > 1:
        config_path = Path(sys.argv[1])

    config = AppConfig.from_yaml(config_path)
    setup_logging(config)

    assistant = VoiceAssistant(config)
    assistant.initialize()

    # Graceful shutdown
    loop = asyncio.new_event_loop()

    def shutdown_handler() -> None:
        try:
            logger.info("Shutting down...")
            for task in asyncio.all_tasks(loop):
                task.cancel()
        except Exception:
            pass  # Best-effort during shutdown

    try:
        loop.add_signal_handler(signal.SIGINT, shutdown_handler)
        loop.add_signal_handler(signal.SIGTERM, shutdown_handler)
    except NotImplementedError:
        logger.warning("Signal handlers are not supported on this platform")

    try:
        with contextlib.suppress(asyncio.CancelledError):
            loop.run_until_complete(assistant.run())
    finally:
        loop.run_until_complete(assistant.shutdown())
        loop.close()
        logger.info("Voice assistant stopped")


# Explicit instance
if __name__ == "__main__":
    main()
