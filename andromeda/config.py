# Copyright (c) 2026 Alessandro Orrù
# Licensed under MIT

from dataclasses import dataclass, field
from pathlib import Path
from typing import Self
import yaml

@dataclass(frozen=True)
class AudioConfig:
    sample_rate: int = 16000
    channels: int = 1
    chunk_ms: int = 30
    dtype: str = "int16"

    @property
    def chunk_samples(self) -> int:
        return int(self.sample_rate * self.chunk_ms / 1000)

    @property
    def chunk_bytes(self) -> int:
        return self.chunk_samples * 2


@dataclass(frozen=True)
class WakeWordConfig:
    engine: str = "openwakeword"
    model_path: str = "models/openwakeword/andromeda.onnx"
    threshold: float = 0.5


@dataclass(frozen=True)
class VADConfig:
    aggressiveness: int = 2
    silence_timeout_sec: float = 2.0
    speech_pad_ms: int = 300
    max_recording_sec: float = 30.0
    min_recording_sec: float = 0.5
    energy_threshold_factor: float = 0.3
    energy_decay_rate: float = 0.95


@dataclass(frozen=True)
class NoiseConfig:
    enabled: bool = True
    stationary: bool = True
    prop_decrease: float = 0.75


@dataclass(frozen=True)
class STTConfig:
    engine: str = "faster-whisper"
    model_size: str = "large-v3"
    device: str = "auto"
    compute_type: str = "int8"
    language: str = "it"
    beam_size: int = 5
    vad_filter: bool = True


@dataclass(frozen=True)
class AgentConfig:
    provider: str = "ollama"
    base_url: str = "http://localhost:11434"
    model: str = "llama3.1:8b"
    max_tokens: int = 500
    streaming: bool = False  # stream response sentence-by-sentence for lower latency
    system_prompt: str = "Sei un assistente vocale domestico intelligente. Ti chiami Andromeda. Rispondi in italiano, in modo conciso e naturale. Le tue risposte verranno lette ad alta voce, quindi usa frasi brevi e chiare, evita formattazione markdown, elenchi puntati, simboli speciali, non usare abbreviazioni ambigue e quando dai numeri, scrivi la forma parlata (es. duemila e non 2000)"


@dataclass(frozen=True)
class TTSConfig:
    engine: str = "piper"
    model_path: str = "models/piper/it_IT-paola-medium.onnx"
    model_config: str = "models/piper/it_IT-paola-medium.onnx.json"
    speaker_id: int = 0
    length_scale: float = 1.0
    sentence_silence: float = 0.3


@dataclass(frozen=True)
class FeedbackConfig:
    wake_sound: str = "sounds/wake.wav"
    done_sound: str = "sounds/done.wav"
    error_sound: str = "sounds/error.wav"


@dataclass(frozen=True)
class ConversationConfig:
    follow_up_timeout_sec: float = 5.0  # seconds to wait for follow-up after TTS ends
    history_timeout_sec: float = 300.0  # clear conversation history after N seconds of inactivity (0 = never)


@dataclass(frozen=True)
class ToolsConfig:
    knowledge_base_path: str = "data/knowledge.json"
    weather_timeout_sec: float = 10.0
    timer_max_sec: int = 3600



@dataclass(frozen=True)
class LoggingConfig:
    level: str = "INFO"


@dataclass(frozen=True)
class AppConfig:
    audio: AudioConfig = field(default_factory=AudioConfig)
    wake_word: WakeWordConfig = field(default_factory=WakeWordConfig)
    vad: VADConfig = field(default_factory=VADConfig)
    noise: NoiseConfig = field(default_factory=NoiseConfig)
    stt: STTConfig = field(default_factory=STTConfig)
    agent: AgentConfig = field(default_factory=AgentConfig)
    tts: TTSConfig = field(default_factory=TTSConfig)
    conversation: ConversationConfig = field(default_factory=ConversationConfig)
    tools: ToolsConfig = field(default_factory=ToolsConfig)
    feedback: FeedbackConfig = field(default_factory=FeedbackConfig)
    logging: LoggingConfig = field(default_factory=LoggingConfig)


    # Load configuration from YAML file, merging with defaults
    @classmethod
    def from_yaml(cls, path: str | Path) -> Self:
        config_path = Path(path)
        if not config_path.exists():
            return cls()

        with config_path.open() as f:
            raw = yaml.safe_load(f) or {}

        return cls(
            audio=AudioConfig(**raw.get("audio", {})),
            wake_word=WakeWordConfig(**raw.get("wake_word", {})),
            vad=VADConfig(**raw.get("vad", {})),
            noise=NoiseConfig(**raw.get("noise", {})),
            stt=STTConfig(**raw.get("stt", {})),
            agent=AgentConfig(**raw.get("agent", {})),
            tts=TTSConfig(**raw.get("tts", {})),
            conversation=ConversationConfig(**raw.get("conversation", {})),
            tools=ToolsConfig(**raw.get("tools", {})),
            feedback=FeedbackConfig(**raw.get("feedback", {})),
            logging=LoggingConfig(**raw.get("logging", {})),
        )
