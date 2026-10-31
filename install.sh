#!/usr/bin/env bash
# Sonicity one-line installer
set -e

REPO_URL="https://github.com/Sumeet-basfore/Sonicity.git"
TARGET_DIR="Sonicity"

echo "=================================================="
echo "  Sonicity installer"
echo "=================================================="

# Check for required system tools
command -v git >/dev/null 2>&1 || {
    echo "Error: git is not installed. Please install git first."
    exit 1
}

command -v python3 >/dev/null 2>&1 || {
    echo "Error: python3 is not installed. Python 3.10+ is required."
    exit 1
}

if ! command -v ffmpeg >/dev/null 2>&1; then
    echo "Notice: ffmpeg is not installed on your system."
    echo "Audio format conversions and stem separation require ffmpeg."
    echo "Install it via:"
    echo "  Ubuntu/Debian: sudo apt update && sudo apt install -y ffmpeg"
    echo "  Arch Linux:    sudo pacman -S ffmpeg"
    echo "  macOS:         brew install ffmpeg"
    echo ""
fi

# Detect whether we are already inside the cloned repository
if [ -f "server.py" ] && [ -f "transcriber.py" ]; then
    INSTALL_DIR="$(pwd)"
    echo "Detected existing Sonicity directory: $INSTALL_DIR"
else
    if [ -d "$TARGET_DIR" ]; then
        echo "Directory '$TARGET_DIR' already exists. Navigating into it..."
        cd "$TARGET_DIR"
        INSTALL_DIR="$(pwd)"
    else
        echo "Cloning Sonicity into ./$TARGET_DIR..."
        git clone "$REPO_URL" "$TARGET_DIR"
        cd "$TARGET_DIR"
        INSTALL_DIR="$(pwd)"
    fi
fi

# Check for uv package manager for fast wheel resolution
if command -v uv >/dev/null 2>&1; then
    UV_BIN="uv"
elif [ -f "$HOME/.local/bin/uv" ]; then
    UV_BIN="$HOME/.local/bin/uv"
elif [ -f "$HOME/.cargo/bin/uv" ]; then
    UV_BIN="$HOME/.cargo/bin/uv"
else
    echo "Installing uv (fast python package installer)..."
    curl -LsSf https://astral.sh/uv/install.sh | sh
    UV_BIN="$HOME/.local/bin/uv"
    if [ ! -f "$UV_BIN" ]; then
        UV_BIN="uv"
    fi
fi

# Set up virtual environment
echo "Setting up Python environment..."
if [ ! -d "venv" ]; then
    if [ -x "$UV_BIN" ] || command -v uv >/dev/null 2>&1; then
        $UV_BIN venv --python 3.11 venv || $UV_BIN venv venv || python3 -m venv venv
    else
        python3 -m venv venv
    fi
fi

# Install dependencies
echo "Installing project dependencies..."
if [ -x "$UV_BIN" ] || command -v uv >/dev/null 2>&1; then
    $UV_BIN pip install --python venv/bin/python -r requirements.txt
else
    ./venv/bin/pip install --upgrade pip
    ./venv/bin/pip install -r requirements.txt
fi

# Ensure run.sh is executable
chmod +x run.sh

echo ""
echo "=================================================="
echo "  Installation complete!"
echo "=================================================="
echo "To start Sonicity, run:"
echo ""
echo "  cd $INSTALL_DIR"
echo "  ./run.sh"
echo ""
echo "Then open http://127.0.0.1:7860 in your browser."
echo "=================================================="
