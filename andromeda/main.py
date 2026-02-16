# Copyright (c) 2026 Alessandro Orrù
# Licensed under MIT

import warnings
warnings.filterwarnings("ignore", message="pkg_resources is deprecated")

import asyncio
import logging
import signal
import sys
import webrtcvad
from pathlib import Path
from andromeda.agent import AIAgent
from andromeda.audio_capture import AudioCapture
from andromeda.config import AppConfig
from andromeda.feedback import AudioFeedback
from andromeda.state_machine import AssistantState, StateMachine
from andromeda.stt import SpeechRecognizer
from andromeda.tools import register_all_tools
from andromeda.tts import TextToSpeech
from andromeda.vad import VoiceActivityDetector
from andromeda.wake_word import WakeWordDetector

logger = logging.getLogger("andromeda.main")


# Main application class that wires all components together
class VoiceAssistant:

    def __init__(self, config: AppConfig) -> None:
        self._cfg = config

        # Components
        self._audio = AudioCapture(config.audio, config.noise)
        self._wake_word = WakeWordDetector(config.audio, config.wake_word)
        self._vad = VoiceActivityDetector(config.audio, config.vad)
        self._stt = SpeechRecognizer(config.stt)
        self._agent = AIAgent(config.agent)
        self._tts = TextToSpeech(config.audio, config.tts)
        self._feedback = AudioFeedback(config.audio, config.feedback)

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
        self._speech_energy: float = 0.0  # calibrated speech energy level


    # Initialize all components. Call before run().
    def initialize(self) -> None:
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

        self._feedback.initialize()
        self._wake_word.initialize()
        self._stt.initialize()
        self._agent.initialize()
        self._tts.initialize()

        # Register tools
        register_all_tools(self._agent, self._cfg.tools, self._feedback)

        # Wire audio callbacks
        self._audio.on_audio_frame(self._wake_word.process_frame)
        self._audio.on_audio_frame(self._vad.process_frame)

        logger.info("All components initialized")


    # Run the assistant main loop
    async def run(self) -> None:
        self._audio.start()
        logger.info("Voice assistant is running. Listening for wake word...")

        try:
            await self._sm.run()
        finally:
            self._audio.stop()


    # IDLE: Listen for wake word in background
    async def _handle_idle(self, _state: AssistantState) -> AssistantState:
        self._wake_word.reset()
        self._audio.unmute()

        # Wait for wake word detection (runs in thread via event)
        loop = asyncio.get_event_loop()
        detected = await loop.run_in_executor(
            None,
            lambda: self._wake_word.wait_for_detection(timeout=None),
        )

        if detected:
            # Calibrate speech energy from the ring buffer audio
            calibration_vad = webrtcvad.Vad(self._cfg.vad.aggressiveness)
            self._speech_energy = self._audio.calibrate_speech_energy(
                calibration_vad, self._cfg.audio.sample_rate
            )
            energy_threshold = self._speech_energy * self._cfg.vad.energy_threshold_factor
            self._vad.set_energy_threshold(energy_threshold)
            logger.info(
                "Calibrated speech energy: %.1f, VAD energy threshold: %.1f",
                self._speech_energy, energy_threshold,
            )

            self._feedback.play("wake")
            return AssistantState.LISTENING

        return AssistantState.IDLE


    # LISTENING: Record user speech until silence timeout
    async def _handle_listening(self, _state: AssistantState) -> AssistantState:
        self._audio.start_recording()
        self._vad.start()

        # Wait for speech to end
        loop = asyncio.get_event_loop()
        await loop.run_in_executor(None, lambda: self._vad.wait_for_speech_end(timeout=self._cfg.vad.max_recording_sec + 1))

        self._vad.stop()
        self._recorded_audio = self._audio.stop_recording()

        # Reset energy threshold for next session
        self._vad.set_energy_threshold(0.0)

        # Check minimum recording
        min_samples = int(self._cfg.vad.min_recording_sec * self._cfg.audio.sample_rate)
        if len(self._recorded_audio) < min_samples or not self._vad.had_speech:
            logger.info("Recording too short or no speech detected, back to IDLE")
            return AssistantState.IDLE

        self._feedback.play("done")
        return AssistantState.PROCESSING


    # PROCESSING: STT transcription + AI agent call
    async def _handle_processing(self, _state: AssistantState) -> AssistantState:
        # Transcribe
        text = await self._stt.transcribe(self._recorded_audio)

        if not text.strip():
            logger.info("Empty transcription, back to IDLE")
            return AssistantState.IDLE

        logger.info("User said: %s", text)

        # AI response
        self._response_text = await self._agent.process(text)

        return AssistantState.SPEAKING


    # SPEAKING: TTS playback, mic muted to avoid echo
    async def _handle_speaking(self, _state: AssistantState) -> AssistantState:
        self._audio.mute()
        try:
            await self._tts.speak(self._response_text)
        finally:
            self._audio.unmute()
        return AssistantState.IDLE


    # ERROR: Play error sound and return to IDLE
    async def _handle_error(self, _state: AssistantState) -> AssistantState:
        self._feedback.play_blocking("error")
        await asyncio.sleep(0.5)

        return AssistantState.IDLE


    # Release all resources. Unblocks threads waiting on events
    async def shutdown(self) -> None:
        self._wake_word.shutdown()
        await self._agent.close()


# Logging setup
def setup_logging(config: AppConfig) -> None:
    logging.basicConfig(
        level=getattr(logging, config.logging.level),
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        handlers=[
            logging.StreamHandler(sys.stdout),
        ],
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
        logger.info("Shutting down...")
        for task in asyncio.all_tasks(loop):
            task.cancel()

    loop.add_signal_handler(signal.SIGINT, shutdown_handler)
    loop.add_signal_handler(signal.SIGTERM, shutdown_handler)

    try:
        loop.run_until_complete(assistant.run())
    except asyncio.CancelledError:
        pass
    finally:
        loop.run_until_complete(assistant.shutdown())
        loop.close()
        logger.info("Voice assistant stopped")


# Explicit instance
if __name__ == "__main__":
    main()
