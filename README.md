# Andromeda

Smart home assistant completely **offline** using **openWakeWord** for wake word, **Whisper** for STT,
**Piper** for TTS and **Ollama** for LLM model inference with tool calling.

## Index

- [Architecture](#architecture)
- [Features](#features)
- [Prerequisites](#prerequisites)
- [Configuration](#configuration)
- [Run Andromeda](#run-andromeda)
- [Tools](#tools)
- [Write new tool](#write-new-tool)
- [Testing](#testing)
- [Debug in VSCode](#debug-in-vscode)
- [License](#license)

## Architecture

```txt
Mic ──▶ Buffer ──▶ Wake Word (OpenWakeWord)
                      │
                 [TRIGGERED]
                      ▼
            VAD + Recording Buffer
           (webrtcvad + energy gate)
                      │
              [SILENCE TIMEOUT]
                      ▼
                STT (Whisper)
                      │
              ┌───────┴───────┐
              │               │
       Fast Intents     AI Agent (Ollama)
       (regex match)     ──▶ Tool calling
              │               │
              └───────┬───────┘
                      ▼
               TTS (Piper) ──▶ Speaker
                      │
              [FOLLOW-UP WAIT]
                      ▼
              Listen again or IDLE
```

[↑ index](#index)

---

## Features

### Multi-turn conversation

After Andromeda responds, the assistant keeps listening for a **follow-up question** without requiring the wake word again. A configurable timeout (`follow_up_timeout_sec`) determines how long it waits before returning to idle. This enables natural, multi-turn conversations.

### Streaming TTS

When enabled (`streaming: true` in config), Andromeda speaks **sentence-by-sentence** as the LLM generates text, significantly reducing perceived latency. A single audio stream is maintained across sentences with fade-out applied to prevent audio pops between fragments.

### Fast intents

Simple deterministic requests (time, date, volume control) are matched with **regex patterns** and executed instantly without involving the LLM. This makes common queries near-instant. Fast intents are configured in `andromeda/tools/__init__.py`.

### Spoken errors

All error conditions (no speech detected, empty transcription, processing failures) are communicated to the user through **spoken TTS messages** rather than failing silently.

### Conversation history timeout

Conversation history is automatically cleared after a configurable period of inactivity (`history_timeout_sec`), preventing stale context from affecting new interactions.

### Adaptive energy gating

The VAD uses an **adaptive energy threshold** calibrated from the wake word utterance. This filters out background noise while adapting to the speaker's volume. The threshold decays over time to handle changing conditions.

[↑ index](#index)

---

## Prerequisites

- Python >= 3.11
- [uv](https://docs.astral.sh/uv/getting-started/installation) and [pip](https://pip.pypa.io/en/stable/installation) installed
- [Ollama](https://ollama.com) with desired model

[↑ index](#index)

---

## Configuration

Init **virtualenv** over **3.11** and install dependencies with:

```bash
uv venv --python 3.11
source .venv/bin/activate
uv sync
```

Then, download models with:

```bash
uv run python setup.py
```

Anyway, we suggest you to use:

```bash
make setup
```

It will create folders, force deps sync and download models for **OpenWakeWord**, **Whisper** and **Piper**.

At this point, ensure you have **Ollama model** installed, with:

```bash
ollama pull llama3.1:8b
```

If you are running Andromeda on a **Mac Silicon** and you have at least **24gb** of ram, through `ollama pull`, you can also safely use models like:

- `mistral-nemo:12b`
- `glm-4.7-flash:latest`
- `gpt-oss:20b`
- ``

Once setup is finished, customize your `config.yaml` file and update your model if changed.

### Configuration reference

| Section | Key | Default | Description |
| :--- | :--- | :--- | :--- |
| `audio` | `sample_rate` | `16000` | Audio sample rate in Hz |
| `audio` | `channels` | `1` | Number of audio channels |
| `audio` | `chunk_ms` | `30` | Frame size for VAD (10, 20, or 30 ms) |
| `wake_word` | `model_path` | `models/openwakeword/andromeda.onnx` | Custom wake word model path |
| `wake_word` | `threshold` | `0.5` | Detection confidence (0.0 - 1.0) |
| `vad` | `aggressiveness` | `2` | WebRTC VAD aggressiveness (0-3) |
| `vad` | `silence_timeout_sec` | `2.0` | Seconds of silence to end recording |
| `vad` | `max_recording_sec` | `30.0` | Maximum recording duration |
| `vad` | `energy_threshold_factor` | `0.3` | Energy gate threshold multiplier |
| `vad` | `energy_decay_rate` | `0.95` | Per-second energy threshold decay |
| `noise` | `enabled` | `true` | Enable noise reduction on recordings |
| `stt` | `model_size` | `large-v3` | Whisper model (tiny, base, small, medium, large-v3) |
| `stt` | `device` | `auto` | Compute device (auto, cpu, cuda) |
| `stt` | `language` | `it` | Transcription language |
| `stt` | `beam_size` | `5` | Beam search width (1 = faster, 5 = more accurate) |
| `agent` | `model` | `llama3.1:8b` | Ollama model name |
| `agent` | `max_tokens` | `500` | Maximum response tokens |
| `agent` | `streaming` | `false` | Stream TTS sentence-by-sentence |
| `tts` | `model_path` | `models/piper/it_IT-paola-medium.onnx` | Piper voice model |
| `conversation` | `follow_up_timeout_sec` | `5.0` | Seconds to wait for follow-up (0 = disabled) |
| `conversation` | `history_timeout_sec` | `300.0` | Clear history after inactivity (0 = never) |
| `tools` | `knowledge_base_path` | `data/knowledge.json` | Persistent memory storage path |
| `tools` | `timer_max_sec` | `3600` | Maximum timer duration in seconds |
| `logging` | `level` | `INFO` | Log level (DEBUG, INFO, WARNING, ERROR) |

[↑ index](#index)

---

## Run Andromeda

Start **Andromeda** with:

```bash
uv run python -m andromeda.main
```

or simply:

```bash
make dev
```

At this point, you can start to talk to **Andromeda**
simply saying `Andromeda` on your mic. You will hear a **beep** that determines the recognition of the wake word and then you can do your question.

This is the **state machine** for every interaction:

```txt
         ┌────────────┐                   ┌──────────┐
IDLE ──▶ │ Wake word  │ ──▶ LISTENING ──▶ │   Your   │
▲  ▲     │ recognized │                   │ sentence │
│  │     └────────────┘                   └─────┬────┘
│  │                                            │
│  │                  ┌───────────────┐         │
│  │                  │ Patterns/LLM  │         ▼
│  │                  └───────┬───────┘   ┌────────────┐
│  └───── SPEAKING ◀──── PROCESSING ◀──── │  Silence   │
│                             │           │ recognized │
│                             ▼           └────────────┘
│                     ┌──────────────┐
└──────────────────── │ cancel/error │
                      └──────────────┘
```

[↑ index](#index)

---

## Tools

Andromeda comes with the following built-in example tools that the LLM can invoke:

| Tool | Description |
| :--- | :--- |
| `get_datetime` | Returns current date and time in Italian |
| `get_weather` | Fetches current weather via Open-Meteo API |
| `get_latest_news` | Scrapes latest news from Il Post by category |
| `knowledge_base` | Persistent key-value memory (save, recall, list, delete) |
| `set_timer` | Countdown timer with audio alarm |
| `system_control` | macOS volume and brightness control via AppleScript |

### Fast intents (no LLM)

These requests are handled instantly via pattern matching:

| Pattern | Action |
| :--- | :--- |
| "che ora/ore", "che ore sono" | Returns current date and time |
| "che giorno/data" | Returns current date |
| "alza volume", "piu forte" | Volume up |
| "abbassa volume", "piu piano" | Volume down |
| "muta audio", "silenzio" | Toggle mute |

[↑ index](#index)

---

## Write new tool

Adding a new tool on Andromeda is super easy. First create your `function_name.py`
into `/andromeda/tools` folder (e.g. `/andromeda/tools/get_news.py`).

The file must be similar to this following structure. It must have:

- A `DEFINITION` constant with:
  - A clear short name (get_news)
  - A clear description: this text will be used by the LLM to understand the capability
  - A list of parameters (optional): what LLM should pass to the tool
- A `handler` method with the tool logic

The handler **must return a readable text**.

```python
# Copyright (c) 2026 Alessandro Orrù
# Licensed under MIT

from datetime import datetime
import locale
import logging

logger = logging.getLogger(__name__)


DEFINITION = {
    "type": "function",
    "function": {
        "name": "get_news",
        "description": (
            "Recupera le ultime notizie dal sito XXX. "
            "Usa questo strumento quando l'utente chiede le ultime notizie, "
            "cosa sta succedendo nel mondo o le news del giorno."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "category": {
                    "type": "string",
                    "description": (
                        "Categoria di notizie da cercare. "
                        "Valori possibili: "
                        "'italia', 'mondo', 'politica', 'economia'."
                    ),
                    "enum": [
                        "homepage", "italia", "mondo", "politica", "economia",
                    ],
                    "default": "homepage",
                }
            },
            "required": [],
        },
    },
}


def handler(_args: dict) -> str:

    # YOUR LOGIC HERE

    return f"Your tool result"
```

Save your file and add the reference in `/andromeda/tools/__init__.py`.

Restart your assistant and enjoy!

[↑ index](#index)

---

## Testing

Run the test suite with:

```bash
uv run pytest tests/ -v
```

Tests cover all core components:

| Test file | Coverage |
| :--- | :--- |
| `test_config.py` | Config defaults, YAML loading, frozen dataclasses |
| `test_state_machine.py` | State transitions, validation, handler execution, error recovery |
| `test_agent.py` | History timeout, tool parsing, sentence splitting, payload building |
| `test_intent.py` | Pattern matching, async handlers, real intent patterns |
| `test_tools.py` | All tools (datetime, knowledge base, timer, system control) |
| `test_audio_capture.py` | Recording, ring buffer, mute/unmute, callbacks |
| `test_vad.py` | Speech detection, energy threshold, timeouts |
| `test_tts.py` | Audio fade-out function |
| `test_feedback.py` | Tone generation, playback |

Install dev dependencies before running tests:

```bash
uv sync --extra dev
```

[↑ index](#index)

---

## Debug in VSCode

To debug your Python microservice you need to:

- Install **VSCode**
- Ensure you have **Python extension** installed
- Ensure you have selected the **right interpreter with virtualenv** on VSCode
- Click on **Run and Debug** menu and **create a launch.json file**
- From dropdown, select **Python debugger** and **FastAPI**
- Change the `.vscode/launch.json` created in the project root with this (customizing host and port if changed):

```json
{
    "version": "0.2.0",
    "configurations": [

        {
            "name": "Andromeda debug",
            "type": "debugpy",
            "request": "launch",
            "module": "andromeda.main",
            "args": [
                "--reload"
            ],
        }
    ]
}
```

- Put some breakpoint in the code, then press the **green play button**
- Call the API to debug

[↑ index](#index)

---

## License

This project is licensed under the MIT License.

[↑ index](#index)

---

Made with ♥️ by Alessandro Orru
