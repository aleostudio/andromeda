#!/usr/bin/env python3

import subprocess
from pathlib import Path
from huggingface_hub import snapshot_download

ROOT_DIR = Path(__file__).parent
VENV_DIR = ROOT_DIR / ".venv"
UV = "uv"


# Subprocess runner
def run(cmd: str, check: bool = True) -> subprocess.CompletedProcess:
    print(f"    {cmd}")
    return subprocess.run(cmd, shell=True, check=check, capture_output=True, text=True)


# Return path to python inside the venv
def python_bin() -> str:
    return str(VENV_DIR / "bin" / "python")


# Main script
def main() -> None:
    _create_folders()
    _create_venv()
    _install_deps()
    _download_wake_word_model()
    _download_piper_model()
    _download_kokoro_model()
    _download_whisper_model()
    _print_instructions()


def _create_folders() -> None:
    print("")
    print("[*] Creating directories...")
    (ROOT_DIR / "models").mkdir(exist_ok=True)
    (ROOT_DIR / "models" / "openwakeword").mkdir(exist_ok=True)
    (ROOT_DIR / "models" / "piper").mkdir(exist_ok=True)
    (ROOT_DIR / "models" / "kokoro").mkdir(exist_ok=True)
    (ROOT_DIR / "sounds").mkdir(exist_ok=True)
    print("    Directories created")


def _create_venv() -> None:
    print("")
    print("[*] Creating virtual environment (python 3.11) with uv...")
    if not VENV_DIR.exists():
        run(f"{UV} venv --python 3.11")
        print("    .venv created")
    else:
        print("    .venv already exists")


def _install_deps() -> None:
    print("")
    print("[*] Installing dependencies with uv sync...")
    run(f"{UV} sync --all-extras")
    print("    Dependencies installed")


def _download_wake_word_model() -> None:
    print("")
    print("[*] Downloading OpenWakeWord base models...")
    result = run(
        f'{python_bin()} -c "import openwakeword; openwakeword.utils.download_models()"',
        check=False,
    )
    if result.returncode == 0:
        print("    OpenWakeWord base models ready")
    else:
        print("    OpenWakeWord base models will download on first run")

    print("")
    print("[*] Downloading wake word 'Andromeda'...")
    models_dir = ROOT_DIR / "models" / "openwakeword"
    andromeda_openwakeword_baseurl = "https://github.com/fwartner/home-assistant-wakewords-collection/raw/refs/heads/main/en/andromeda"
    andromeda_openwakeword_model = models_dir / "andromeda.onnx"
    if not andromeda_openwakeword_model.exists():
        run(f"curl -L -o {andromeda_openwakeword_model} {andromeda_openwakeword_baseurl}/andromeda.onnx")
        print("    OpenWakeWord model 'Andromeda' downloaded")
    else:
        print("    OpenWakeWord model 'Andromeda' already present")


def _download_piper_model() -> None:
    print("")
    print("[*] Downloading Piper Italian voice models (TTS)...")
    models_dir = ROOT_DIR / "models" / "piper"

    riccardo_piper_baseurl = "https://huggingface.co/rhasspy/piper-voices/resolve/main/it/it_IT/riccardo/x_low"
    riccardo_piper_model = models_dir / "it_IT-riccardo-x_low.onnx"
    riccardo_piper_config = models_dir / "it_IT-riccardo-x_low.onnx.json"
    if not riccardo_piper_model.exists():
        run(f"curl -L -o {riccardo_piper_model} {riccardo_piper_baseurl}/it_IT-riccardo-x_low.onnx")
        run(f"curl -L -o {riccardo_piper_config} {riccardo_piper_baseurl}/it_IT-riccardo-x_low.onnx.json")
        print("    Piper voice 'Riccardo' downloaded")
    else:
        print("    Piper voice 'Riccardo' already present")

    paola_piper_baseurl = "https://huggingface.co/rhasspy/piper-voices/resolve/main/it/it_IT/paola/medium"
    paola_piper_model = models_dir / "it_IT-paola-medium.onnx"
    paola_piper_config = models_dir / "it_IT-paola-medium.onnx.json"
    if not paola_piper_model.exists():
        run(f"curl -L -o {paola_piper_model} {paola_piper_baseurl}/it_IT-paola-medium.onnx")
        run(f"curl -L -o {paola_piper_config} {paola_piper_baseurl}/it_IT-paola-medium.onnx.json")
        print("    Piper voice 'Paola' downloaded")
    else:
        print("    Piper voice 'Paola' already present")


def _download_kokoro_model() -> None:
    # Kokoro TTS model download (HF cache in project)
    print("")
    print("[*] Downloading Kokoro TTS model (project HF cache)...")
    models_dir = ROOT_DIR / "models" / "kokoro"
    models_dir.mkdir(exist_ok=True)
    result = run(
        f'{python_bin()} -c "import os; '
        f"os.environ['HF_HOME']='{models_dir}'; "
        f"from huggingface_hub import snapshot_download; "
        f"snapshot_download(repo_id='hexgrad/Kokoro-82M')\"",
        check=False,
    )

    if result.returncode == 0:
        print("    Kokoro model ready (cached)")
    else:
        print("    Kokoro model will download on first run")
        print(result.stderr[:200])


def _download_whisper_model() -> None:
    print("")
    print("[*] Downloading Whisper model medium and large-v3 (STT)...")
    print("    This will download ~3GB on first run")
    print("    Models will be cached in ~/.cache/huggingface/")
    result = run(
        f'{python_bin()} -c "from faster_whisper import WhisperModel; '
        f"WhisperModel('medium', device='auto', compute_type='int8')\"",
        check=False,
    )
    if result.returncode == 0:
        print("    Whisper model medium ready")
    else:
        print("    Whisper model medium will download on first run")
        print(f"{result.stderr[:200]}")

    result = run(
        f'{python_bin()} -c "from faster_whisper import WhisperModel; '
        f"WhisperModel('large-v3', device='auto', compute_type='int8')\"",
        check=False,
    )
    if result.returncode == 0:
        print("    Whisper model large-v3 ready")
    else:
        print("    Whisper model large-v3 will download on first run")
        print(f"{result.stderr[:200]}")


def _print_instructions() -> None:
    print("")
    print("    To create a custom wake word (e.g. 'ehi computer'):")
    print("    Option A - Use OpenWakeWord's training notebook:")
    print("    1. Go to: https://github.com/dscripka/openwakeword")
    print("    2. Follow 'Training Custom Models' guide")
    print("    3. Place the .onnx file in models/ehi_computer.onnx")
    print("")
    print("    Current available wake words: 'andromeda', 'jarvis' (built-in)")
    print("")
    print("[*] Setup complete!")
    print("")
    print("To run the assistant:")
    print("uv run python -m andromeda.main")
    print("")
    print("Run with custom config:")
    print("uv run python -m andromeda.main /path/to/config.yaml")
    print("")


if __name__ == "__main__":
    main()