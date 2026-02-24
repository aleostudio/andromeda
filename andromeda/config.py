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


    def __post_init__(self) -> None:
        if self.sample_rate <= 0:
            raise ValueError(f"sample_rate must be > 0, got {self.sample_rate}")
        if self.channels < 1:
            raise ValueError(f"channels must be >= 1, got {self.channels}")
        if self.chunk_ms not in (10, 20, 30):
            raise ValueError(f"chunk_ms must be 10, 20 or 30, got {self.chunk_ms}")


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

    def __post_init__(self) -> None:
        if not 0.0 <= self.threshold <= 1.0:
            raise ValueError(f"threshold must be 0.0-1.0, got {self.threshold}")


@dataclass(frozen=True)
class VADConfig:
    aggressiveness: int = 3
    silence_timeout_sec: float = 1.5
    speech_pad_ms: int = 300
    max_recording_sec: float = 30.0
    min_recording_sec: float = 0.5
    energy_threshold_factor: float = 0.6
    energy_decay_rate: float = 0.98

    def __post_init__(self) -> None:
        if self.aggressiveness not in (0, 1, 2, 3):
            raise ValueError(f"aggressiveness must be 0-3, got {self.aggressiveness}")
        if self.silence_timeout_sec <= 0:
            raise ValueError(f"silence_timeout_sec must be > 0, got {self.silence_timeout_sec}")
        if self.max_recording_sec <= 0:
            raise ValueError(f"max_recording_sec must be > 0, got {self.max_recording_sec}")
        if self.energy_threshold_factor < 0:
            raise ValueError(f"energy_threshold_factor must be >= 0, got {self.energy_threshold_factor}")
        if not 0.0 < self.energy_decay_rate <= 1.0:
            raise ValueError(f"energy_decay_rate must be (0, 1], got {self.energy_decay_rate}")


@dataclass(frozen=True)
class NoiseConfig:
    enabled: bool = False
    stationary: bool = True
    prop_decrease: float = 0.75

    def __post_init__(self) -> None:
        if not 0.0 <= self.prop_decrease <= 1.0:
            raise ValueError(f"prop_decrease must be 0.0-1.0, got {self.prop_decrease}")


@dataclass(frozen=True)
class STTConfig:
    engine: str = "faster-whisper"
    model_size: str = "medium"
    device: str = "auto"
    compute_type: str = "int8"
    language: str = "it"
    beam_size: int = 1
    vad_filter: bool = False

    def __post_init__(self) -> None:
        if self.beam_size < 1:
            raise ValueError(f"beam_size must be >= 1, got {self.beam_size}")


@dataclass(frozen=True)
class AgentConfig:
    provider: str = "ollama"
    base_url: str = "http://localhost:11434"
    model: str = "llama3.1:8b"
    max_tokens: int = 500
    timeout_sec: float = 60.0
    streaming: bool = True
    prewarm: bool = True
    system_prompt: str = (
        "Sei un assistente vocale domestico intelligente. Ti chiami Andromeda. "
        "Rispondi in italiano, in modo conciso e naturale. "
        "Le tue risposte verranno lette ad alta voce, quindi "
        "usa frasi brevi e chiare, evita formattazione "
        "markdown, elenchi puntati, simboli speciali, "
        "non usare abbreviazioni ambigue e quando dai numeri, "
        "scrivi la forma parlata (es. duemila e non 2000)"
    )

    def __post_init__(self) -> None:
        if self.max_tokens < 1:
            raise ValueError(f"max_tokens must be >= 1, got {self.max_tokens}")
        if self.timeout_sec <= 0:
            raise ValueError(f"timeout_sec must be > 0, got {self.timeout_sec}")


@dataclass(frozen=True)
class TTSConfig:
    engine: str = "piper"
    piper_model_path: str = "models/piper/it_IT-paola-medium.onnx"
    piper_model_config: str = "models/piper/it_IT-paola-medium.onnx.json"
    kokoro_repo_id: str = "hexgrad/Kokoro-82M"
    kokoro_lang_code: str = "i"
    kokoro_voice: str = "if_sara"
    kokoro_speed: float = 1.0
    speaker_id: int = 0
    length_scale: float = 1.0
    sentence_silence: float = 0.3
    prewarm_cache: bool = True


@dataclass(frozen=True)
class FeedbackConfig:
    wake_sound: str = "sounds/wake.wav"
    done_sound: str = "sounds/done.wav"
    error_sound: str = "sounds/error.wav"
    thinking_sound: str = "sounds/thinking.wav"
    thinking_volume: float = 0.5

    def __post_init__(self) -> None:
        if not 0.0 <= self.thinking_volume <= 1.0:
            raise ValueError(f"thinking_volume must be 0.0-1.0, got {self.thinking_volume}")


@dataclass(frozen=True)
class ConversationConfig:
    follow_up_timeout_sec: float = 5.0
    history_timeout_sec: float = 300.0
    max_history: int = 20
    compaction_threshold: int = 16


@dataclass(frozen=True)
class ToolsConfig:
    knowledge_base_path: str = "data/knowledge.json"
    allow_sensitive_memory: bool = False
    weather_timeout_sec: float = 10.0
    news_timeout_sec: float = 10.0
    timer_max_sec: int = 3600
    allow_system_control: bool = True
    web_search_timeout_sec: float = 10.0
    web_search_max_results: int = 3
    web_search_max_content_chars: int = 2000
    web_search_fetch_page_content: bool = False


@dataclass(frozen=True)
class HealthCheckConfig:
    enabled: bool = False
    host: str = "127.0.0.1"
    port: int = 8080

    def __post_init__(self) -> None:
        if not 1 <= self.port <= 65535:
            raise ValueError(f"port must be 1-65535, got {self.port}")


@dataclass(frozen=True)
class LoggingConfig:
    level: str = "INFO"

    def __post_init__(self) -> None:
        valid = ("DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL")
        if self.level.upper() not in valid:
            raise ValueError(f"level must be one of {valid}, got '{self.level}'")


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
    health_check: HealthCheckConfig = field(default_factory=HealthCheckConfig)
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
            health_check=HealthCheckConfig(**raw.get("health_check", {})),
            logging=LoggingConfig(**raw.get("logging", {})),
        )
