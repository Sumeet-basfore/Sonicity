# SPEC-09 // HI-RES AUDIOPHILE MUSIC STUDIO & DIRECT-TO-DAC ENGINE
## Comprehensive Architecture, Library Scanner, Background TTML Compiler & Bit-Perfect Implementation Plan

**Target Application**: SonicScribe / Sonicity  
**File Location**: `/home/sumeet/audio-to-text/AUDIOPHILE_MUSIC_STUDIO_PLAN.md`  
**Date**: October 31, 2026  
**Primary Hardware Focus**: Audiocular D07 USB DAC (TTGK 32-bit / 384kHz), In-Ear Monitors (IEMs), NVIDIA RTX 3050 CUDA (4GB VRAM), CachyOS Linux (PipeWire 1.6.8 / ALSA)  

---

## 1. Executive Summary & Problem Diagnosis

### 1.1 The User's Requirement
1. **Audiophile Listening Priority**: The primary user exclusively listens to high-resolution lossless music (FLAC, 24-bit / 32-bit float, 96 kHz / 192 kHz / 384 kHz) through a dedicated external **USB DAC (Audiocular D07)** connected to high-sensitivity In-Ear Monitors (IEMs).
2. **The "Bit-Perfect" Bottleneck**: In modern desktop operating systems and standard web browsers, audio streams are forced through software audio servers (e.g. PipeWire / PulseAudio or the browser's internal Web Audio API) which **resample everything to a default 48,000 Hz / 16-bit or float32 buffer**. This destroys bit-perfection, introduces jitter, causes inter-sample clipping, and robs high-res tracks of dynamic range.
3. **Dedicated Music Player Section**: A specialized listening interface separate from the raw ingestion panel, featuring high-fidelity cover art (revolving vinyl, CD jewel case), audiophile telemetry, and Apple Music-style 3D fluid Aurora synced lyrics.
4. **Music Directory Scanner & Asset Pairing**: The ability to mount or point to an entire local music directory (such as `~/Music/mpd`), recursively index all tracks, and automatically pair existing lyrics (`.lrc`, `.elrc`, `.ttml`, `.txt`).
5. **Non-Blocking Background TTML Compiler**: If a track lacks synced lyrics, the user can press Play and listen immediately without waiting; in the background, an asynchronous CUDA worker transcribes the vocal track, aligns syllables to millisecond timestamps, and compiles letter-by-letter `.ttml` lyrics, automatically switching them live when ready.

---

## 2. Hardware Inspection & Bit-Perfect Diagnosis

During system discovery, hardware query on the local Linux host revealed:

```text
Card 2: TTGK Technology Co.,Ltd Audiocular D07 (High Speed USB Audio)
- Altset 1: Format S16_LE  | Rates: 8k, 16k, 32k, 44.1k, 48k, 96k, 192k, 384k Hz (16-bit)
- Altset 2: Format S24_3LE | Rates: 8k, 16k, 32k, 44.1k, 48k, 96k, 192k, 384k Hz (24-bit)
- Altset 3: Format S32_LE  | Rates: 8k, 16k, 32k, 44.1k, 48k, 96k, 192k, 384k Hz (32-bit)
```

However, PipeWire's default sample rate is locked at `48,000 Hz`:
```text
Default Sample Specification: float32le 2ch 48000Hz
Default Sink: alsa_output.usb-TTGK_Technology_Co._Ltd_Audiocular_D07-00.analog-stereo
```

### The Solution: The Dual-Engine Playback Architecture

To ensure 100% mathematical bit-perfection without compromising web usability, we architect a **Dual-Engine Player**:

```
+----------------------------------------------------------------------------------------------------+
|                                    SONICSCRIBE WEB UI (CH-06)                                      |
|            [ 🎛️ DAC DIRECT (BIT-PERFECT) ]      or      [ 🌐 BROWSER WEB AUDIO (UNIVERSAL) ]       |
+---------------------------------------------------+------------------------------------------------+
                                                    |
             +--------------------------------------+---------------------------------------+
             |                                                                              |
             v                                                                              v
+----------------------------------------+                     +----------------------------------------+
|  ENGINE A: HARDWARE DIRECT-TO-DAC      |                     |  ENGINE B: HIGH-RES WEB AUDIO STREAM   |
|  - Spawns mpv IPC daemon in background |                     |  - HTML5 <audio> Range byte streaming  |
|  - ALSA Exclusive: `hw:2,0` (D07)       |                     |  - Web Audio AudioContext (24-bit)     |
|  - Zero resampling (Native 96/192/384k)|                     |  - For remote devices, phones on LAN,  |
|  - Zero software volume/EQ degradation |                     |    or quick browser listening          |
|  - Bi-directional IPC syncs timecodes  |                     |  - Uses visualizer spectrum analyser   |
|    to Web UI at 60 FPS                 |                     |                                        |
+-------------------+--------------------+                     +--------------------+-------------------+
                    |                                                               |
                    v                                                               v
      [ Audiocular D07 USB DAC ]                                      [ System Default / PipeWire ]
      (Pure Bit-Perfect Analog IEMs)                                  (Standard OS Audio Output)
```

---

## 3. System Architecture & Component Design

### 3.1 Component Directory Structure
```text
/home/sumeet/audio-to-text/
├── library/                       # Music Library & Scanner Subsystem
│   ├── scanner.py                 # Recursive directory walker, mutagen tag reader, cover extractor
│   ├── database.py                # Fast SQLite / JSON index cache (~/Music metadata cache)
│   └── watcher.py                 # (Optional) inotify auto-reloader for library changes
├── player/                        # Audiophile Playback Engine Subsystem
│   ├── dac_engine.py              # MPV IPC / ALSA hw direct controller (Bit-Perfect)
│   ├── dac_detector.py            # Real-time ALSA / USB DAC hardware enumerator
│   └── web_streamer.py            # Byte-range FLAC chunk streamer for Engine B
├── workers/                       # Asynchronous Background Syllable Compiler
│   ├── lyrics_queue.py            # FIFO / Priority queue for unaligned tracks
│   └── background_aligner.py      # CUDA Whisper + LRCLIB resolver -> .ttml / .elrc compiler
├── prototypes/
│   └── prototype-neobrutal-2.html # Upgraded with "CH-06 // AUDIOPHILE MUSIC STUDIO"
└── AUDIOPHILE_MUSIC_STUDIO_PLAN.md# This architectural specification document
```

---

## 4. Phase-by-Phase Implementation Blueprint

### Phase 1: Bit-Perfect Hardware Playback Engine (`player/dac_engine.py`)

1. **DAC Hardware Enumeration (`dac_detector.py`)**:
   - Scans `/proc/asound/cards` and `aplay -l` to identify all hardware devices.
   - Detects the **Audiocular D07** (or any plug-and-play USB DAC).
   - Reads `/proc/asound/cardX/stream0` to detect exact supported altsets (16-bit, 24-bit, 32-bit, up to 384 kHz).
   - Exposes endpoint: `GET /api/audiophile/dacs` returning device names, ALSA hardware cards, and supported bitrates.

2. **ALSA Exclusive Direct Daemon (`dac_engine.py`)**:
   - Uses `mpv` with a persistent UNIX domain socket (`/tmp/sonicscribe-mpv.sock`).
   - Launch flags for bit-perfect output:
     ```bash
     mpv --idle=yes \
         --input-ipc-server=/tmp/sonicscribe-mpv.sock \
         --ao=alsa \
         --audio-device=alsa/hw:D07,0 \
         --audio-exclusive=yes \
         --audio-samplerate=0 \
         --audio-format=auto \
         --audio-channels=stereo \
         --gapless-audio=yes \
         --video=no
     ```
   - **Properties Controlled via JSON-IPC**:
     - `loadfile <path> replace`
     - `pause`, `resume`, `seek <seconds> absolute`
     - Polls `time-pos` at 20Hz (every 50ms) and pushes via WebSocket to the frontend for letter-to-letter lyrics synchronization.
   - **Bit-Perfect Verification**:
     - Reads `/proc/asound/card2/pcm0p/sub0/hw_params` during playback to confirm the hardware DAC is receiving the raw native rate (e.g., `rate: 96000 (96000/1)`, `format: S24_3LE`).
     - Exposes real-time verification status to the UI: `BIT-PERFECT CONFIRMED [HW: 96,000 Hz @ 24-bit]`.

3. **Fallback & Toggle to Browser Web Audio**:
   - The user can switch between:
     - **`🎛️ DAC DIRECT (BIT-PERFECT)`**: Sent directly to Audiocular D07 via ALSA hardware exclusive mode.
     - **`🌐 BROWSER AUDIO (PIPEWIRE)`**: Standard HTML5 audio streaming for desktop speaker preview.

---

### Phase 2: Music Directory Scanner & Metadata Indexer (`library/scanner.py`)

1. **Format Compatibility**:
   - Native support for: `.flac`, `.wav`, `.aiff`, `.alac`, `.dsf`, `.dff`, `.m4a`, `.mp3`, `.ogg`, `.opus`.
   - Special focus on FLAC with 24-bit / 32-bit samples and Vorbis comments.

2. **Metadata & Audiophile Specs Extractor**:
   - Uses `mutagen` to extract:
     - Title, Artist, Album, Year, Track Number, Genre.
     - Sample Rate (`sample_rate` e.g. 96,000 Hz, 192,000 Hz).
     - Bit Depth (`bits_per_sample` e.g. 24-bit, 32-bit).
     - Channels (Stereo / 2.0).
     - Exact Duration (seconds to millisecond precision).
     - File Size & Bitrate (kbps).

3. **Cover Art Extractor & Visual Provider**:
   - Searches for embedded cover art inside the audio file (`METADATA_BLOCK_PICTURE` for FLAC, `APIC` for ID3).
   - If no embedded art exists, looks in the track's folder for `cover.jpg`, `cover.png`, `folder.jpg`, `front.png`.
   - Serves extracted art via high-performance thumbnail endpoint: `GET /api/library/cover/{track_id}`.
   - Color Extraction: Samples the dominant 3–5 palette colors from the cover art and sends them to the frontend to automatically paint the Apple Music 3D fluid Aurora canvas!

4. **Lyrics Asset Matcher**:
   - Checks for accompanying synced lyrics files in order of priority:
     1. `{track_name}.ttml` (Apple Music Syllable-by-Syllable sync).
     2. `{track_name}.elrc` (Enhanced word-by-word synced LRC).
     3. `{track_name}.lrc` (Standard line-synced LRC).
     4. `{track_name}.txt` (Unsynced reference text).
     5. Embedded Vorbis tags: `LYRICS`, `UNSYNCEDLYRICS`.
   - Tags each track in the library as:
     - `🟢 SYNCED (TTML / ELRC)`
     - `🟡 LINE-SYNCED (LRC)`
     - `⚪ UNSYNCED (TXT / TAG)`
     - `🔴 NO LYRICS (READY FOR BG COMPILE)`

5. **Persistent Library Storage**:
   - Stores scanned library state in a lightweight SQLite database (`library.db`) in the user's workspace.
   - Initial scan of `~/Music/mpd` (4.5+ GB) takes ~5–8 seconds; subsequent app launches take **< 80 milliseconds** via indexed queries.

---

### Phase 3: Non-Blocking Background Syllable Compiler (`workers/`)

1. **The Instant-Play Experience**:
   - User clicks any track -> Track starts playing immediately through the DAC with zero lag.
   - If the song already has synced lyrics (`.ttml` / `.lrc`), the 3D Aurora lyrics stage illuminates immediately.
   - If the song has NO synced lyrics:
     - The player displays the cover art and audio specs.
     - A status badge in the header shows: `⚡ AUTO-COMPILING SYLLABLES IN BACKGROUND (CUDA)...`
     - The track is enqueued into the background worker.

2. **Asynchronous Alignment Pipeline**:
   ```
   [ Unsynced Song Queued ]
              │
              ▼
   Step 1: Check LRCLIB API / Local Cache for Verified Canonical Lyrics
              │
         ┌────┴────────────────────────┐
         │ Found                       │ Not Found
         ▼                             ▼
   Step 2A: Run Whisper           Step 2B: Run Vocal Stem Separator (separator.py)
   Acoustic Alignment                      │
   (align_canonical_lyrics)                ▼
         │                        Step 2C: Run faster-whisper large-v3-turbo
         │                        with word_timestamps=True on vocal stem
         │                                 │
         └───────────────┬─────────────────┘
                         ▼
   Step 3: Calculate Syllable Singing Weights (estimate_word_singing_weight)
                         │
                         ▼
   Step 4: Generate .ttml and .elrc files alongside track
           (e.g., "~/Music/mpd/01 ☆ Hollywood’s Bleeding.ttml")
                         │
                         ▼
   Step 5: Broadcast WebSocket Event: `LYRICS_READY { track_id, ttml_data }`
                         │
                         ▼
   Web UI seamlessly lights up letter-by-letter synced lyrics with 0 audio pause!
   ```

---

### Phase 4: Frontend "Audiophile Music Studio" UI Experience

We integrate a dedicated, luxury **Audiophile Studio (CH-06)** into `prototype-neobrutal-2.html`:

```
+----------------------------------------------------------------------------------------------------+
|  [01 INGESTION]  [02 ROUTING]  [03 ACOUSTIC]  [04 CHAT]  [05 LYRICS]  | [06 HI-RES MUSIC STUDIO] ★ |
+----------------------------------------------------------------------------------------------------+
|                                                                                                    |
|  +---------------------------+  +--------------------------------+  +---------------------------+  |
|  | LIBRARY EXPLORER (LEFT)   |  | HI-FI CHASSIS & PLAYER (CENTER)|  | 3D AURORA LYRICS (RIGHT)  |  |
|  |                           |  |                                |  |                           |  |
|  | 📁 ~/Music/mpd            |  |  +--------------------------+  |  | [ ≡ CENTER ] [ 🎨 PALETTE]|  |
|  | [ 🔍 Search Library... ]  |  |  |     COVER ART DISPLAY    |  |  |                           |  |
|  |                           |  |  |   Mode A: Vinyl Record   |  |  | "Shout out to what        |  |
|  | ★ Ode to the Mets [24/96] |  |  |   Mode B: CD Jewel Case  |  |  |  is done"                 |  |
|  |   Hollywood's Bleeding    |  |  |   (Spinning with needle) |  |  |                           |  |
|  |   Alone Again             |  |  +--------------------------+  |  | [02:04.56]                |  |
|  |   Reptilia                |  |                                |  | (Letter-by-letter glowing |  |
|  |   Otonoke                 |  |  TITLE: Ode to the Mets        |  |  Apple Music character    |  |
|  |                           |  |  ARTIST: The Strokes           |  |  sweep)                   |  |
|  | Filters:                  |  |  ALBUM: The New Abnormal       |  |                           |  |
|  | [All] [24-bit] [96k+]     |  |                                |  |                           |  |
|  | [Has TTML] [Need Sync]    |  |  +--------------------------+  |  |                           |  |
|  |                           |  |  | AUDIOPHILE HARDWARE HUD  |  |  |                           |  |
|  |                           |  |  | DAC: Audiocular D07 USB  |  |  |                           |  |
|  |                           |  |  | STREAM: 24-bit / 96.0kHz |  |  |                           |  |
|  |                           |  |  | OUTPUT: 🟢 BIT-PERFECT   |  |  |                           |  |
|  |                           |  |  +--------------------------+  |  |                           |  |
|  |                           |  |                                |  |                           |  |
|  |                           |  |  [⏮] [ ▶ / ⏸ ] [⏭]          |  |                           |  |
|  |                           |  |  ═════════●══════════ 02:04/05:51| |                           |  |
|  |                           |  |  OUTPUT: [🎛️ DAC DIRECT] [🌐 WEB]| |                           |  |
|  +---------------------------+  +--------------------------------+  +---------------------------+  |
|                                                                                                    |
+----------------------------------------------------------------------------------------------------+
```

#### Key Visual Features:
1. **Interactive Cover Art Chassis**:
   - **Revolving Vinyl Mode**: High-resolution record spinning at realistic 33⅓ RPM inside an album gatefold sleeve, with an interactive tonearm that moves across the record as the track progresses.
   - **CD Jewel Case Mode**: Glossy polycarbonate CD jewel case with realistic light reflections and holographic disc sheen.
   - **Custom Artwork Swapper**: Drag-and-drop or right-click to replace album art on any track.
2. **Audiophile Telemetry Readout**:
   - Monitors active stream bitrate, dynamic range, bit depth (`24-bit` / `32-bit float`), sample rate (`96 kHz` / `192 kHz`).
   - Confirms ALSA Direct status (`hw:2,0` exclusive lock).
3. **Asset Swapper & Editor Modal**:
   - Allows user to click **"Change Lyrics File"** -> upload or paste custom `.lrc` / `.ttml`.
   - Allows user to click **"Change Audio File"** -> replace single track with high-res remasters.
4. **Instant Fullscreen Cinema Mode**:
   - Press **`F`** anytime to enter borderless cinema mode where the 3D fluid aurora engulfs the screen, the revolving vinyl/cover sits beside the lyrics, and character-by-character glowing lyrics take center stage.

---

## 5. Desktop Application Packaging Roadmap

While the system runs seamlessly as a web app on `http://127.0.0.1:7860`, it is architected with a clean decoupling that allows direct packaging into a standalone **desktop application**:

1. **Tauri / Electron Wrapper**:
   - The frontend (`prototype-neobrutal-2.html`) acts as the native GUI window.
   - The Python backend runs as an embedded lightweight sidecar daemon.
2. **Direct Hardware Audio**:
   - On Linux: Direct ALSA `hw:X,Y` hardware mode (no root privileges needed, uses standard audio group).
   - On Windows: WASAPI Exclusive Mode bypasses Windows Audio Session mixing.
   - On macOS: CoreAudio Integer Mode bypasses macOS aggregate mixing.
3. **Desktop Media Keys & System Tray**:
   - Global media shortcuts (`Play`, `Pause`, `Next`, `Previous`, `Mute`) handled directly by the desktop wrapper.

---

## 6. Implementation Deliverables & Execution Plan

| Phase | Component | Key Tasks | Verification Gate |
|---|---|---|---|
| **Phase 1** | **Audiophile DAC Engine** | • Implement `player/dac_detector.py` to auto-detect Audiocular D07<br>• Implement `player/dac_engine.py` with `mpv` ALSA exclusive IPC<br>• Create `/api/audiophile/status` & verify `/proc/asound/.../hw_params` | `aplay -l` & `hw_params` confirm 24-bit/96kHz pure direct stream |
| **Phase 2** | **Library Scanner & Indexer** | • Install `mutagen`<br>• Implement `library/scanner.py` for recursive scanning of `~/Music`<br>• Extract cover art, Vorbis tags, audio specs, and `.lrc`/`.ttml` pairings<br>• Create SQLite index cache for instant loading | Scanner indexes `~/Music/mpd` in <10s with artwork & specs |
| **Phase 3** | **Background Syllable Compiler** | • Implement background worker queue in FastAPI<br>• Wire LRCLIB canonical fetch + Whisper acoustic alignment<br>• Generate `.ttml` and emit WebSocket updates | User plays unsynced song; lyrics pop up automatically without audio interruption |
| **Phase 4** | **Studio UI & Visuals** | • Add Tab 6 `[06 // HI-RES MUSIC STUDIO]` to `prototype-neobrutal-2.html`<br>• Implement Revolving Vinyl & CD Jewel Case cover art modes<br>• Build library explorer list with Hi-Res badges & search<br>• Connect Apple Music 3D Aurora lyrics with dynamic palette extraction | UI plays high-res FLAC, shows spinning vinyl, and renders glowing letter-by-letter lyrics |
| **Phase 5** | **Sync & Polish** | • Test hotkeys, seek latency, and DAC switching<br>• Commit and push to GitHub repository | Full end-to-end user verification |

---

## 7. Open Questions for User Clarification

To tailor this experience to your exact taste, please share your thoughts on these options:

1. **Default Music Directory**:
   - Would you like the player to default directly to `/home/sumeet/Music/mpd` on startup, with a button to select or browse other directories?
2. **Cover Art Visual Style**:
   - Do you prefer the **Revolving Vinyl Record** as the default visual, the **Glossy CD Jewel Case**, or a toggle button so you can switch between them whenever you want?
3. **Lyrics File Placement**:
   - When the background worker compiles a new `.ttml` or `.lrc` for a song that didn't have one, should it save the file directly in the music folder alongside the song (e.g. `Song.flac` + `Song.ttml`), or in an internal cache directory? *(Saving alongside the track is recommended so it stays permanently with your music collection).*

---
*Plan created in `/home/sumeet/audio-to-text/AUDIOPHILE_MUSIC_STUDIO_PLAN.md`. Ready for review and execution approval.*
