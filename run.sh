#!/usr/bin/env bash
# Local Audio-to-English Studio Launcher (FastAPI + Neo-Brutalist Architecture)

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

if [ ! -d "venv" ]; then
    echo "Creating virtual environment with uv..."
    if command -v uv &> /dev/null; then
        UV_BIN="uv"
    elif [ -f "$HOME/.local/bin/uv" ]; then
        UV_BIN="$HOME/.local/bin/uv"
    else
        echo "Installing uv package manager..."
        curl -LsSf https://astral.sh/uv/install.sh | sh
        UV_BIN="$HOME/.local/bin/uv"
    fi

    $UV_BIN venv --python 3.11 venv || python3 -m venv venv
    if [ -f "$UV_BIN" ] || command -v uv &> /dev/null; then
        $UV_BIN pip install --python venv/bin/python -r requirements.txt
    else
        ./venv/bin/pip install -r requirements.txt
    fi
fi

# Export NVIDIA dynamic libraries for CTranslate2 CUDA acceleration
NVIDIA_LIBS=$(find "$SCRIPT_DIR/venv/lib/python3.11/site-packages/nvidia" -type d -name "lib" 2>/dev/null | tr '\n' ':')
if [ -n "$NVIDIA_LIBS" ]; then
    export LD_LIBRARY_PATH="$NVIDIA_LIBS:$LD_LIBRARY_PATH"
fi

echo "=========================================================="
echo " Starting SonicScribe Industrial Studio (Neo-Brutalist)..."
echo " Web UI running at: http://127.0.0.1:7860"
echo " Engine: faster-whisper (CUDA int8_float16) + Ollama Llama 3.2 (3B)"
echo " Press Ctrl+C to stop the server"
echo "=========================================================="

./venv/bin/python -m uvicorn server:app --host 0.0.0.0 --port 7860
