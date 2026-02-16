#!/usr/bin/env python3

import subprocess
from pathlib import Path

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
    # Folders creation
    print("")
    print("[*] Creating directories...")
    (ROOT_DIR / "models").mkdir(exist_ok=True)
    (ROOT_DIR / "models" / "openwakeword").mkdir(exist_ok=True)
    (ROOT_DIR / "models" / "piper").mkdir(exist_ok=True)
    (ROOT_DIR / "sounds").mkdir(exist_ok=True)
    print("    Directories created")

    # Virtual env creation
    print("")
    print("[*] Creating virtual environment (python 3.11) with uv...")
    if not VENV_DIR.exists():
        run(f"{UV} venv --python 3.11")
        print("    .venv created")
    else:
        print("    .venv already exists")

    # Dependencies
    print("")
    print("[*] Installing dependencies with uv sync...")
    run(f"{UV} sync --all-extras")
    print("    Dependencies installed")

    # Wake word model (openWakeWord)
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

    # Italian voice download (Piper)
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

    # Whisper model download
    print("")
    print("[*] Downloading Whisper model large-v3 (STT)...")
    print("    This will download ~3GB on first run")
    print("    The model will be cached in ~/.cache/huggingface/")
    result = run(
        f'{python_bin()} -c "from faster_whisper import WhisperModel; '
        f"WhisperModel('large-v3', device='auto', compute_type='int8')\"",
        check=False,
    )
    if result.returncode == 0:
        print("    Whisper model ready")
    else:
        print("    Whisper model will download on first run")
        print(f"{result.stderr[:200]}")

    # OpenWakeWord model
    print("")
    print("[*] Downloading OpenWakeWord base models...")
    result = run(
        f'{python_bin()} -c "import openwakeword; openwakeword.utils.download_models()"',
        check=False,
    )
    if result.returncode == 0:
        print("    OpenWakeWord models ready")
    else:
        print("    OpenWakeWord models will download on first run")

    print("")
    print("    To create a custom wake word (e.g. 'ehi computer'):")
    print("    Option A - Use OpenWakeWord's training notebook:")
    print("    1. Go to: https://github.com/dscripka/openwakeword")
    print("    2. Follow 'Training Custom Models' guide")
    print("    3. Place the .onnx file in models/ehi_computer.onnx")
    print("")
    print("    Option B - Use Picovoice Porcupine (alternative):")
    print("    1. Create account at https://console.picovoice.ai")
    print("    2. Train custom keyword")
    print("    3. Download .ppn file and update config.yaml")
    print("")
    print("    For now, the assistant will use the built-in 'hey_jarvis' keyword.")
    print("")
    print("[*] Setup complete!")
    print("")

    # Instructions
    print("To run the assistant:")
    print("uv run python -m andromeda.main")
    print("")
    print("Run with custom config:")
    print("uv run python -m andromeda.main /path/to/config.yaml")
    print("")


if __name__ == "__main__":
    main()