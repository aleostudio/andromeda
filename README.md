# Andromeda

Smart home assistant completely **offline** using **openWakeWord** for wake word, **Whisper** for STT,
**Piper** for TTS and **Ollama** for LLM model inference with tool calling.

## Index

- [Architecture](#architecture)
- [Prerequisites](#prerequisites)
- [Configuration](#configuration)
- [Run Andromeda](#run-andromeda)
- [Write new tool](#write-new-tool)
- [Debug in VSCode](#debug-in-vscode)
- [License](#license)

## Architecture

```txt
Mic ──▶ Buffer ──▶ Wake Word (OpenWakeWord)
                      │
                 [TRIGGERED]
                      ▼
            VAD + Recording Buffer
           (webrtcvad + noisereduce)
                      │
              [SILENCE TIMEOUT]
                      ▼
                STT (Whisper)
                      ▼
              AI agent (Ollama) ──▶ Tool calling
                      ▼
                 TTS (Piper) ──▶ Speaker

```

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

If you are running Andromeda on a **Mac Silicon** and you have at least **24gb** of ram, you can also safely use `gpt-oss:20b`.

Once setup is finished, customize your `config.yaml` file.

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
│  │                  ┌──────────────┐          │
│  │                  │ Patterns/LLM │          ▼
│  │                  └───────┬──────┘    ┌────────────┐
│  └───── SPEAKING ◀──── PROCESSING ◀──── │  Silence   │
│                             │           │ recognized │
│                             ▼           └────────────┘
│                     ┌──────────────┐
└──────────────────── │ cancel/error │
                      └──────────────┘
```

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

## Debug in VSCode

To debug your Python microservice you need to:

- Install **VSCode**
- Ensure you have **Python extension** installed
- Ensure you have selected the **right interpreter with virtualenv** on VSCode
- Click on **Run and Debug** menu and **create a launch.json file**
- From dropdown, select **Python debugger** and **FastAPI**
- Change the ```.vscode/launch.json``` created in the project root with this (customizing host and port if changed):

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

Made with ♥️ by Alessandro Orrù
