# Sonicity

Local audio transcription, Demucs vocal separation, and synchronized lyrics player.

Runs entirely on your machine. No telemetry, no external audio uploads, and no cloud dependencies required.

```bash
curl -fsSL https://raw.githubusercontent.com/Sumeet-basfore/Sonicity/main/install.sh | bash
```

---

## What it does

- **Local speech to text**: Transcribes audio and video files using faster-whisper (CTranslate2). Runs on CUDA with int8_float16 precision, averaging 8x to 20x faster than real time on an RTX 3050 (6GB VRAM), or on CPU with int8.
- **Vocal and instrumental separation**: Extracts dry vocals and backing tracks using Facebook Research's Demucs (`htdemucs`). Lets you switch between the full mix, acapella vocals, and karaoke backing tracks mid-playback without losing your place in the track.
- **Vocal-locked synced lyrics**: Fetches human-verified lyrics from LRCLIB and locks every word to Whisper's acoustic timestamps. Vowels receive extra phonetic weight so character sweeps follow sustained singing notes instead of moving at a flat mechanical speed.
- **Live microphone dictation**: Streams 16kHz audio over WebSockets directly into the Whisper decoder. Interim partial text renders in about 30ms, and pauses over 0.8 seconds finalize sentences automatically.
- **Translation and clean-up**: Automatically translates spoken foreign languages into English, with optional post-processing through local Ollama models (`llama3.2:3b`, etc.) or cloud providers (Groq, OpenAI).
- **Export formats**: Saves transcripts and subtitles as TXT, Markdown, SRT, VTT, JSON, standard LRC, Enhanced LRC, and Apple Music TTML.

---

## Requirements

- **Linux** (tested on Ubuntu 22.04+ / Arch) or **macOS / Windows WSL2**
- **Python 3.10+**
- **ffmpeg** installed on system PATH
- **NVIDIA GPU** with 4GB+ VRAM recommended for CUDA acceleration (works on CPU with slower inference)

---

## Quick start

### 1. Automatic installation

Run the installer script:

```bash
curl -fsSL https://raw.githubusercontent.com/Sumeet-basfore/Sonicity/main/install.sh | bash
```

Or clone the repository and run the install script directly:

```bash
git clone https://github.com/Sumeet-basfore/Sonicity.git
cd Sonicity
./install.sh
```

### 2. Launching the studio

Run the launcher script:

```bash
./run.sh
```

The script sets up NVIDIA library paths and starts the server. Open your browser to:

```
http://127.0.0.1:7860
```

---

## How it works

### Acoustic lyrics alignment

Standard LRC files only mark the start of each line, which causes traditional players to divide line duration evenly across words. When a singer holds a word for three seconds and rushes the next four, flat division drifts out of sync.

Sonicity solves this by fusing two data sources:
1. Canonical lyrics from LRCLIB provide correct spelling, capitalization, and line breaks.
2. Whisper's acoustic model runs on the isolated vocal stem to detect word boundaries down to the millisecond.
3. Monotonic token matching maps canonical words to acoustic timestamps. Vowels (`a, e, i, o, u`) receive 2.2x duration weighting during character interpolation so the highlight sweep tracks the singer's vocal cords.

### In-memory stem switching

When you click Acapella or Karaoke, the browser swaps `audioEl.src` to the isolated Demucs stem while preserving `currentTime`. Playback continues without buffering, and the lyrics sweep maintains uninterrupted sync.

---

## Project layout

```
Sonicity/
├── server.py              # FastAPI server, REST routes, and dictation WebSocket
├── transcriber.py         # faster-whisper CTranslate2 loader and batch pipeline
├── separator.py           # Demucs htdemucs stem separation runner
├── lyrics_resolver.py     # LRCLIB lookup and acoustic word alignment
├── refiner.py             # LLM translation and structured JSON extractors
├── downloader.py          # yt-dlp media downloader
├── install.sh             # Automatic environment installer
├── run.sh                 # Studio launch script with CUDA library auto-detection
├── requirements.txt       # Python dependencies
└── prototypes/
    └── prototype-neobrutal-2.html  # Studio user interface
```

---

## Keyboard shortcuts

- `Space`: Play or pause audio
- `Ctrl + Enter`: Start audio compilation
- `J`: Jump backward 5 seconds
- `K`: Jump forward 5 seconds

---

## License

MIT
