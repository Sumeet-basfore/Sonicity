# Live Streaming Dictation & AI Vocal Separation (Karaoke Studio) Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Transform SonicScribe Studio into a full real-time speech and music workstation by adding:
1. **Live Streaming Dictation**: Real-time microphone audio streaming over WebSockets with instant word-by-word teleprinter updates (<200ms latency).
2. **AI Vocal Separation & Karaoke Mode**: 2-stem acapella/instrumental isolation (Demucs) with real-time audio stem switching synchronized with Apple Music per-letter lyrics sweeps.

**Architecture:**
- **Live Dictation**: Fast HTML5 Web Audio API (16kHz PCM chunks) streaming over FastAPI WebSockets (`/api/ws/dictate`). Backend maintains an acoustic ring buffer, runs Silero VAD speech framing, and executes greedy faster-whisper inference to stream `interim` and `final` subtitle cards in real time.
- **Stem Separation**: Dedicated separation engine (`separator.py`) powered by Demucs `htdemucs` (2-stem model: `vocals` vs `no_vocals`). Generates isolated audio stems served via FastAPI. Frontend Tape Deck gains an interactive 3-way stem switch (`MASTER` / `ACAPELLA` / `KARAOKE`) that swaps audio tracks seamlessly at the exact millisecond while Apple Music lyrics continue sweeping.

**Tech Stack:**
- Python 3.11, FastAPI WebSockets, faster-whisper (CUDA int8_float16), Demucs (`htdemucs`), PyTorch CUDA
- HTML5 Web Audio API, WebSockets, Canvas Waveform Visualizer, CSS gradient per-letter sweeper

---

## Task Breakdown

### Phase 1: Live Streaming Microphone Dictation

#### Task 1.1: Backend WebSocket Streaming Endpoint
- **File:** `server.py`
- **Objective:** Implement `@app.websocket("/api/ws/dictate")` to receive binary PCM audio buffers, accumulate speech frames, and emit streaming partial transcripts.
- **Implementation Steps:**
  1. Add WebSocket route in `server.py` handling client connections and binary chunk payloads.
  2. Implement an in-memory audio queue that aggregates 16kHz 16-bit PCM chunks.
  3. Run faster-whisper greedy decoding (`beam_size=1`, `temperature=0.0`, `without_timestamps=True`) on speech segments identified by VAD.
  4. Stream JSON packets back to client:
     - `{"type": "interim", "text": "partial text...", "is_final": false}`
     - `{"type": "final", "segment": {"start": 0.0, "end": 2.4, "text": "locked sentence."}}`
  5. Test connection with a Python test client script (`tests/test_ws_dictate.py`).

#### Task 1.2: Frontend Real-Time AudioWorklet & WebSocket Client
- **File:** `prototypes/prototype-neobrutal-2.html`
- **Objective:** Capture browser microphone at 16kHz and send binary buffers over WebSocket with live visual feedback.
- **Implementation Steps:**
  1. Add an on-air modal / banner: `[🎙️ LIVE DICTATION // ON-AIR]` with audio VU peak meters.
  2. Implement Web Audio API `AudioContext` and `ScriptProcessorNode` / `AudioWorklet` to downsample microphone audio to 16,000Hz mono PCM.
  3. Open WebSocket connection to `ws://<host>:<port>/api/ws/dictate` on mic trigger.
  4. Display incoming interim text dynamically in the teleprinter lane with pulsating cursor.
  5. Automatically lock finalized cards and route to translation / agent intelligence on stop.

---

### Phase 2: AI Vocal Separation & Karaoke Mode

#### Task 2.1: Python Stem Separation Engine (`separator.py`)
- **File:** Create `separator.py`
- **Objective:** Wrap Demucs `htdemucs` 2-stem model to cleanly separate any track into `vocals.wav` and `no_vocals.wav` (instrumental backing).
- **Implementation Steps:**
  1. Install `demucs` and `torchaudio` into the virtual environment using `uv`.
  2. Write `separate_stems(audio_filepath: str, output_dir: str)` in `separator.py`.
  3. Ensure GPU execution (`device="cuda"`) with fallback to CPU if VRAM is constrained.
  4. Verify separation on sample FLAC file (`The Long Faces - Jane - 01 - Jane.flac`).
  5. Confirm generated files: `vocals.wav` and `no_vocals.wav`.

#### Task 2.2: Backend Stem API Endpoints
- **File:** `server.py`
- **Objective:** Expose stem separation progress and audio delivery endpoints.
- **Implementation Steps:**
  1. Add `POST /api/stems/separate`: Accepts `{ filepath: str }`, triggers separation, returns stem filepaths.
  2. Add `GET /api/stems/status/{track_id}`: Telemetry on separation progress.
  3. Serve separated stem audio tracks via `/api/audio/...` endpoint.

#### Task 2.3: Frontend Tape Deck Stem Switcher & Karaoke Mode
- **File:** `prototypes/prototype-neobrutal-2.html`
- **Objective:** Add interactive stem controls to the tape deck and integrate with Apple Music Synced Lyrics.
- **Implementation Steps:**
  1. Add `[⚡ SEPARATE VOCALS & STEMS]` action button on the Tape Deck card.
  2. Add 3-way segmented stem selector:
     - `[ 🎵 MASTER MIX ]` (Original full audio)
     - `[ 🎤 ACAPELLA ]` (Isolated vocals only)
     - `[ 🎸 KARAOKE INSTRUMENTAL ]` (Vocals removed, backing track only)
  3. Implement seamless audio swapping:
     ```javascript
     function switchAudioStem(stemType) {
       const curTime = audioEl.currentTime;
       const wasPlaying = !audioEl.paused;
       audioEl.src = stemUrls[stemType];
       audioEl.currentTime = curTime;
       if (wasPlaying) audioEl.play();
     }
     ```
  4. In Tab 5 (`05 // SYNCED LYRICS`), add a `🎤 KARAOKE MODE` banner that highlights the instrumental backing track while lyrics illuminate per-letter in real time!

---

### Phase 3: Integration & End-to-End Verification

#### Task 3.1: Live Dictation End-to-End Test
- Verify continuous speech into microphone produces instant partial and final teleprinter cards without dropping words or crashing the WebSocket server.

#### Task 3.2: Karaoke End-to-End Test
- Separate `The Long Faces - Jane - 01 - Jane.flac`.
- Switch to `KARAOKE INSTRUMENTAL` at 00:30.
- Verify vocals are gone from the audio output while the Apple Music lyrics illuminate letter-by-letter with 0 millisecond desync.
