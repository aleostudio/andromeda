# Andromeda

Smart home assistant completely **offline** using **openWakeWord** for wake word, **Whisper** for STT,
**Piper** for TTS and **Ollama** for LLM model inference with tool calling.

## Index

- [Architecture](#architecture)
- [Prerequisites](#prerequisites)
- [Configuration](#configuration)
- [Run Andromeda](#run-andromeda)
- [Debug in VSCode](#debug-in-vscode)

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
                      │
                      ▼
                STT (Whisper)
                      │
                      ▼
              AI agent (Ollama) ──▶ Tool calling
                      │
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
IDLE ──▶ │ wake word  │ ──▶ LISTENING ──▶ │   your   │
▲  ▲     │ recognized │                   │ sentence │
│  │     └────────────┘                   └─────┬────┘
│  │                                            │
│  │                                            ▼
│  │                                      ┌────────────┐
│  └───── SPEAKING ◀──── PROCESSING ◀──── │  silence   │
│                             │           │ recognized │
│                             ▼           └────────────┘
│                     ┌──────────────┐
└──────────────────── │ cancel/error │
                      └──────────────┘
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

Made with ♥️ by Alessandro Orrù
