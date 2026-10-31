# ⚡ SONICITY // SPEC-09
### Industrial Audio Compiler, AI Vocal Separator & Apple Music Synced Lyrics Studio

[![CUDA 12](https://img.shields.io/badge/CUDA-12.x-76B900?logo=nvidia&logoColor=white)](https://developer.nvidia.com/cuda-toolkit)
[![faster-whisper](https://img.shields.io/badge/faster--whisper-large--v3--turbo-00E575)](https://github.com/SYSTRAN/faster-whisper)
[![Demucs](https://img.shields.io/badge/Demucs-htdemucs-FF2A85)](https://github.com/facebookresearch/demucs)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

**Sonicity** is a 100% offline, GPU-accelerated audio compilation suite designed with a tactile **Neo-Brutalist Swiss Industrial interface**. It combines hardware-accelerated speech-to-text, real-time WebSocket streaming dictation, AI vocal & instrumental stem separation (Demucs), and Apple Music-style per-letter animated synced lyrics with a 3D moving gradient aurora mesh.

---

## ⚡ Core Architecture & Features

### 1. 🎙️ High-Performance Audio Compilation
- **Engine**: `faster-whisper` (`large-v3-turbo`) with CTranslate2 CUDA `int8_float16` execution (~8x-23x faster than real-time on RTX 3050).
- **Dual-Stream Teleprinter**: Transcribes raw vernacular speech (Hindi/Hinglish/99+ languages) while concurrently routing through Ollama / Groq into clean agent-ready English.
- **Audio Trimmer**: Built-in non-destructive range selector for transcribing audio sections.

### 2. 🎸 AI Stem Separation & Karaoke Mode (Demucs)
- **Model**: Facebook Research's `htdemucs` 2-stem model running locally on CUDA.
- **Instant Playhead Stem Swapping**:
  - `[ 🎵 MASTER ]`: Original multi-track recording.
  - `[ 🎤 ACAPELLA ]`: 100% isolated dry vocals.
  - `[ 🎸 KARAOKE ]`: Vocal-free instrumental backing track.
- Seamlessly swaps audio streams in-memory during playback without losing track time or breaking lyrics synchronization.

### 3. 🎵 Apple Music-Inspired Synced Lyrics & 3D Aurora Mesh
- **True Vocal-Locked Synchronization**: Canonical human lyrics (via LRCLIB or manual paste) are physically locked to Whisper's acoustic word timestamps. Each word is matched to the millisecond the singer vocalized it.
- **Phonetic Vowel Elongation**: Characters within words are duration-weighted phonetically (vowels receive 2.2x weighting) for fluid singing sweeps.
- **3D Moving Gradient Aurora Mesh**: Hardware-accelerated 60fps ambient background with 4 morphing fluid orbs and frosted glass blur.
- **Tactile Word & Line Seeking**: Click any line or individual word to jump playback directly to that syllable.

### 4. 🔴 Low-Latency Live Streaming Dictation
- **WebSocket Pipeline**: Streams 16kHz float32 PCM audio chunks directly to faster-whisper in VRAM (~30ms inference latency).
- **Live Interim Teleprinter**: Displays streaming interim text with pulsing cursor (`Listening... ▌`) and automatically finalizes sentence boundaries on speech pauses (>0.8s).

### 5. 📦 Universal Export Suite
- **Artifacts**: Plain Text (`.txt`), Markdown (`.md`), Subtitles (`.srt`, `.vtt`), Telemetry JSON (`.json`), Standard LRC (`.lrc`), Enhanced LRC (`.elrc`), and Apple Music Timed Text (`.ttml`).

---

## 🚀 Quick Start

### Prerequisites
- Linux / Windows with Python 3.11+
- NVIDIA GPU with CUDA 12+ (e.g. RTX 3050 Laptop / Desktop or higher)
- `ffmpeg` installed on your system

### Installation & Launch

```bash
# Clone the repository
git clone https://github.com/Sumeet-basfore/Sonicity.git
cd Sonicity

# Launch the studio (auto-provisions venv & downloads dependencies)
./run.sh
```

Open your browser to:
```
http://127.0.0.1:7860
```

---

## 🗂️ Project Layout

```
Sonicity/
├── server.py              # FastAPI server, WebSockets & REST endpoints
├── transcriber.py         # faster-whisper CUDA transcription manager
├── separator.py           # Demucs htdemucs AI vocal & karaoke separator
├── lyrics_resolver.py     # LRCLIB resolver & acoustic word alignment engine
├── refiner.py             # Ollama / Groq LLM agent intelligence & translations
├── downloader.py          # yt-dlp media stream extractor
├── run.sh                 # Environment setup and launcher
├── prototypes/
│   └── prototype-neobrutal-2.html  # Neo-Brutalist Swiss Industrial Studio UI
└── tests/
    └── test_ws_dictate.py # Live WebSocket dictation test suite
```

---

## 📄 License
MIT License. Free and open source for all developers and creators.
