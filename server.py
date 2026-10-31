"""
SonicScribe Industrial Engine - FastAPI Backend.
High-Performance Local Audio-to-English Processing with CUDA, faster-whisper, and Ollama.
"""

import os
import sys
import glob

# Ensure LD_LIBRARY_PATH includes NVIDIA CUDA & cuDNN shared libraries
_nvidia_libs = glob.glob(os.path.join(sys.prefix, "lib", "python*", "site-packages", "nvidia", "*", "lib"))
if _nvidia_libs:
    _existing_ld = os.environ.get("LD_LIBRARY_PATH", "")
    _new_paths = [p for p in _nvidia_libs if p not in _existing_ld]
    if _new_paths:
        os.environ["LD_LIBRARY_PATH"] = ":".join(_nvidia_libs) + ((":" + _existing_ld) if _existing_ld else "")

import time
import json
import shutil
import logging
import subprocess
import numpy as np
from pathlib import Path
from typing import Optional, List, Dict, Any

logger = logging.getLogger(__name__)

from fastapi import FastAPI, UploadFile, File, Form, HTTPException, Query, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse, FileResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from transcriber import WhisperManager, format_timestamp
from refiner import (
    get_available_ollama_models,
    contains_non_english,
    translate_to_english,
    translate_segments_to_english,
    extract_agent_intelligence,
    diarize_dialogue_turns,
    chat_with_transcript,
    query_llm_engine,
    OPENAI_PROVIDERS,
    format_lyrics_with_llm,
    generate_lrc,
    generate_elrc,
    generate_ttml,
)
from lyrics_resolver import (
    extract_audio_metadata,
    query_lrclib,
    parse_lrc_to_lyrics_data,
    align_canonical_lyrics_to_acoustic,
    align_reference_lyrics,
)
from separator import separate_audio_stems, STEMS_DIR
from downloader import download_audio_from_url

from library.scanner import scan_music_directory, get_track_by_id, load_track_lyrics_data, DEFAULT_MUSIC_DIR
from player.dac_detector import list_alsa_playback_devices, verify_live_hardware_params
from player.dac_engine import dac_player
from workers.background_aligner import enqueue_track_for_alignment, get_job_status

BASE_DIR = Path(__file__).resolve().parent
UPLOAD_DIR = BASE_DIR / "uploads"
PROTOTYPES_DIR = BASE_DIR / "prototypes"
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)

# Initialize transcription engine
manager = WhisperManager()

app = FastAPI(title="SonicScribe Industrial Audio Compiler", version="2.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

STATIC_DIR = BASE_DIR / "static"
STATIC_DIR.mkdir(parents=True, exist_ok=True)
app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")

# Startup pre-warm in background thread so server starts instantly
@app.on_event("startup")
def prewarm_gpu():
    import threading
    def _warm():
        print("⚡ Background pre-warming faster-whisper on CUDA (int8_float16)...")
        try:
            manager.get_model(model_name="large-v3-turbo", device="cuda", compute_type="int8_float16")
            print("✅ Whisper large-v3-turbo loaded and ready in VRAM!")
        except Exception as e:
            print(f"⚠️ Pre-warm notice: {e}")
    threading.Thread(target=_warm, daemon=True).start()


@app.get("/", response_class=HTMLResponse)
async def serve_ui():
    """Serve the approved Swiss Industrial Neo-Brutalist UI."""
    ui_path = PROTOTYPES_DIR / "prototype-neobrutal-2.html"
    if ui_path.exists():
        return HTMLResponse(content=ui_path.read_text(encoding="utf-8"))
    return HTMLResponse(content="<h1>SonicScribe UI not found. Check prototypes directory.</h1>", status_code=404)


@app.get("/player", response_class=HTMLResponse)
async def serve_player():
    """Serve the Dedicated Standalone Audiophile Hi-Res Player."""
    ui_path = PROTOTYPES_DIR / "prototype-neobrutal-2.html"
    if ui_path.exists():
        return HTMLResponse(content=ui_path.read_text(encoding="utf-8"))
    return HTMLResponse(content="<h1>SonicScribe UI not found. Check prototypes directory.</h1>", status_code=404)


@app.get("/marked.min.js")
async def serve_marked_js():
    """Serve local vendored marked.js library for offline markdown rendering."""
    marked_path = PROTOTYPES_DIR / "marked.min.js"
    if marked_path.exists():
        return FileResponse(str(marked_path), media_type="application/javascript")
    raise HTTPException(status_code=404, detail="marked.min.js not found")


@app.get("/api/gpu")
async def get_gpu_telemetry():
    """Return live NVIDIA RTX 3050 hardware metrics."""
    try:
        res = subprocess.run(
            ["nvidia-smi", "--query-gpu=memory.used,memory.total,temperature.gpu,utilization.gpu", "--format=csv,noheader,nounits"],
            capture_output=True,
            text=True,
            timeout=2.0
        )
        if res.returncode == 0:
            parts = [p.strip() for p in res.stdout.strip().split(",")]
            return {
                "available": True,
                "vram_used_mb": int(parts[0]),
                "vram_total_mb": int(parts[1]),
                "temp_c": int(parts[2]),
                "util_pct": int(parts[3]),
            }
    except Exception:
        pass
    return {"available": False, "vram_used_mb": 0, "vram_total_mb": 0, "temp_c": 0, "util_pct": 0}


@app.post("/api/upload")
async def upload_audio(file: UploadFile = File(...)):
    """Upload audio/video file for local transcription."""
    sanitized_name = "".join(c for c in file.filename if c.isalnum() or c in "._- ")
    destination = UPLOAD_DIR / sanitized_name

    with open(destination, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    size_kb = round(destination.stat().st_size / 1024, 1)

    return {
        "status": "success",
        "filename": sanitized_name,
        "filepath": str(destination),
        "size_kb": size_kb,
        "audio_url": f"/api/audio/{sanitized_name}",
    }


class UrlRequest(BaseModel):
    url: str

@app.post("/api/download_url")
async def download_url(req: UrlRequest):
    """Download audio stream from YouTube / web URL using yt-dlp."""
    try:
        result = download_audio_from_url(req.url, output_dir=str(UPLOAD_DIR))
        audio_path_str = result["file_path"] if hasattr(result, "__getitem__") else result[0]
        audio_path = Path(audio_path_str)
        title_str = result.get("title", audio_path.name) if hasattr(result, "get") else (result[1] if len(result) > 1 else audio_path.name)
        size_kb = round(audio_path.stat().st_size / 1024, 1)
        return {
            "status": "success",
            "filename": audio_path.name,
            "filepath": str(audio_path),
            "size_kb": size_kb,
            "title": title_str,
            "audio_url": f"/api/audio/{audio_path.name}",
        }
    except Exception as e:
        logger.error(f"Download failed for {req.url}: {e}", exc_info=True)
        raise HTTPException(status_code=400, detail=f"Download failed: {str(e)}")


@app.get("/api/audio/{filename}")
async def serve_audio(filename: str):
    """Stream audio file for HTML5 playback & Web Audio visualization."""
    audio_path = UPLOAD_DIR / filename
    if not audio_path.exists():
        raise HTTPException(status_code=404, detail="Audio file not found")
    
    media_type = "audio/mpeg"
    if filename.endswith(".ogg"):
        media_type = "audio/ogg"
    elif filename.endswith(".wav"):
        media_type = "audio/wav"
    elif filename.endswith(".m4a"):
        media_type = "audio/mp4"
    elif filename.endswith(".flac"):
        media_type = "audio/flac"

    return FileResponse(str(audio_path), media_type=media_type)


@app.post("/api/batch_upload")
async def batch_upload_audio(files: List[UploadFile] = File(...)):
    """Upload multiple audio/video files for batch queue processing."""
    uploaded_files = []
    for file in files:
        sanitized_name = "".join(c for c in file.filename if c.isalnum() or c in "._- ")
        destination = UPLOAD_DIR / sanitized_name
        with open(destination, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        size_kb = round(destination.stat().st_size / 1024, 1)
        uploaded_files.append({
            "filename": sanitized_name,
            "filepath": str(destination),
            "size_kb": size_kb,
            "audio_url": f"/api/audio/{sanitized_name}",
        })
    return {"status": "success", "files": uploaded_files}


class TestLLMRequest(BaseModel):
    provider: str = "groq"
    api_key: Optional[str] = None
    base_url: Optional[str] = None
    model: Optional[str] = None


@app.post("/api/test_llm_connection")
async def test_llm_connection(req: TestLLMRequest):
    """Test live connectivity with external OpenAI-compatible LLM API or local Ollama."""
    t0 = time.time()
    try:
        res = query_llm_engine(
            prompt="Respond with exactly one word: PONG",
            system_prompt="You are a system ping agent. Output ONLY 'PONG'.",
            config=req.dict(),
            max_tokens=15,
            timeout=15.0
        )
        elapsed_ms = round((time.time() - t0) * 1000)
        if res and len(res.strip()) > 0:
            return {
                "status": "success",
                "provider": req.provider,
                "model": req.model or "default",
                "latency_ms": elapsed_ms,
                "response": res.strip(),
                "message": f"Connected to {req.provider.upper()}! Response latency: {elapsed_ms}ms"
            }
        else:
            raise HTTPException(status_code=400, detail="LLM returned an empty response. Verify API key, endpoint URL, and model name.")
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Connection test failed: {str(e)}")


class TranscribeRequest(BaseModel):
    filepath: str
    mode: str = "speech"  # "speech" or "music_lyrics"
    always_translate: bool = True
    beam_size: int = 1
    compute_type: str = "int8_float16"
    language: Optional[str] = None
    target_language: str = "English"
    initial_prompt: Optional[str] = None
    trim_start: Optional[float] = None
    trim_end: Optional[float] = None
    word_timestamps: bool = True
    llm_config: Optional[Dict[str, Any]] = None


@app.post("/api/transcribe")
async def run_transcription(req: TranscribeRequest):
    """
    Run hardware-accelerated transcription on CUDA with optional audio trimming and word timestamps.
    Translates vernacular speech to target language (pure English by default) via local Ollama.
    Supports 'music_lyrics' mode with VAD bypass and Apple Music-style character synchronization.
    """
    if not os.path.exists(req.filepath):
        raise HTTPException(status_code=404, detail=f"File not found: {req.filepath}")

    start_time = time.time()
    
    # Check if trim range is specified
    actual_filepath = req.filepath
    time_offset = 0.0
    if (req.trim_start is not None and req.trim_start > 0) or (req.trim_end is not None and req.trim_end > 0):
        time_offset = max(0.0, req.trim_start or 0.0)
        trimmed_name = f"trim_{int(time.time())}_{Path(req.filepath).name}"
        trimmed_path = UPLOAD_DIR / trimmed_name
        trim_cmd = ["ffmpeg", "-y", "-ss", str(time_offset)]
        if req.trim_end is not None and req.trim_end > time_offset:
            trim_cmd.extend(["-to", str(req.trim_end)])
        trim_cmd.extend(["-i", req.filepath, "-c:a", "libopus", "-b:a", "128k", str(trimmed_path)])
        
        try:
            res = subprocess.run(trim_cmd, capture_output=True, text=True, timeout=30.0)
            if res.returncode == 0 and trimmed_path.exists():
                actual_filepath = str(trimmed_path)
            else:
                time_offset = 0.0
        except Exception as e:
            logger.warning(f"Audio trimming fallback to full file: {e}")
            time_offset = 0.0

    # Run transcription
    model = manager.get_model(
        model_name="large-v3-turbo",
        device="cuda",
        compute_type=req.compute_type
    )

    # Detect if Music & Lyrics mode is active (or FLAC/lossless song)
    is_music = (req.mode == "music_lyrics") or actual_filepath.lower().endswith((".flac",))

    if is_music:
        # Music & Song transcription parameters:
        # VAD filter OFF so backing drums/guitars do not truncate the audio stream after line 1
        # condition_on_previous_text=False prevents hallucination drops across instrumental solos
        vad_filter = False
        vad_params = None
        condition_prev = False
        no_speech_thresh = 0.85
        temperatures = [0.0, 0.2, 0.4, 0.8]
    else:
        vad_filter = True
        vad_params = dict(min_silence_duration_ms=400)
        condition_prev = True
        no_speech_thresh = 0.6
        temperatures = 0.0

    target_lang = None if req.language in [None, "auto", "Auto-Detect", ""] else req.language

    # First pass: transcribe spoken vernacular with accurate timecodes & word timestamps
    segments_iter, info = model.transcribe(
        actual_filepath,
        beam_size=req.beam_size,
        language=target_lang,
        task="transcribe",
        vad_filter=vad_filter,
        vad_parameters=vad_params,
        condition_on_previous_text=condition_prev,
        no_speech_threshold=no_speech_thresh,
        temperature=temperatures,
        initial_prompt=req.initial_prompt,
        word_timestamps=req.word_timestamps,
    )

    detected_lang = info.language
    lang_prob = round(info.language_probability * 100, 1)

    raw_segments: List[Dict[str, Any]] = []
    hindi_texts = []

    for seg in segments_iter:
        text = seg.text.strip()
        if text:
            hindi_texts.append(text)
            
            # Extract word-level timestamps
            words = []
            if hasattr(seg, "words") and seg.words:
                for w in seg.words:
                    words.append({
                        "word": w.word,
                        "start": round(float(w.start) + time_offset, 2),
                        "end": round(float(w.end) + time_offset, 2),
                        "prob": round(float(w.probability), 2),
                    })

            avg_logprob = float(getattr(seg, "avg_logprob", -0.2))
            is_low_conf = avg_logprob < -0.55

            seg_start = round(seg.start + time_offset, 2)
            seg_end = round(seg.end + time_offset, 2)

            raw_segments.append({
                "id": len(raw_segments) + 1,
                "start": seg_start,
                "end": seg_end,
                "start_str": format_timestamp(seg_start),
                "end_str": format_timestamp(seg_end),
                "text_raw": text,
                "confidence": round(avg_logprob, 2),
                "is_low_confidence": is_low_conf,
                "words": words,
            })

    raw_transcript = " ".join(hindi_texts)
    transcription_duration = time.time() - start_time

    # Second pass: Translation Routing
    english_transcript = raw_transcript
    translated_segments = []

    has_foreign_script = contains_non_english(raw_transcript) or (detected_lang != "en")
    target_language = req.target_language or "English"
    ollama_models = get_available_ollama_models()
    preferred_ollama = ollama_models[0] if ollama_models else "None"

    llm_cfg = req.llm_config or {}
    is_cloud_llm = bool(llm_cfg.get("provider") and llm_cfg.get("provider") != "ollama" and llm_cfg.get("api_key"))
    llm_label = f"{llm_cfg.get('provider', '').upper()} [{llm_cfg.get('model', 'default')}]" if is_cloud_llm else f"OLLAMA [{preferred_ollama}]"

    can_translate = is_cloud_llm or bool(ollama_models)

    if req.always_translate and (has_foreign_script or target_language.lower() != "english") and can_translate:
        print(f"⚡ Translating to {target_language} via {llm_label}...")
        t_start = time.time()
        
        # 1. Translate full continuous transcript
        english_transcript = translate_to_english(
            raw_transcript,
            model_name=preferred_ollama,
            target_language=target_language,
            llm_config=llm_cfg
        )
        
        # 2. Fast batch translate segments for synchronized display
        try:
            temp_segs = [{"text": s["text_raw"]} for s in raw_segments]
            translated_batch = translate_segments_to_english(
                temp_segs,
                model_name=preferred_ollama,
                target_language=target_language,
                llm_config=llm_cfg
            )
            for idx, s in enumerate(raw_segments):
                seg_copy = dict(s)
                seg_copy["text_english"] = translated_batch[idx].get("text", s["text_raw"])
                translated_segments.append(seg_copy)
        except Exception as e:
            print(f"Segment translation fallback: {e}")
            for s in raw_segments:
                seg_copy = dict(s)
                seg_copy["text_english"] = s["text_raw"]
                translated_segments.append(seg_copy)
        
        print(f"✅ Translation finished in {round(time.time() - t_start, 2)}s via {llm_label}!")
    else:
        # Audio is already English or verbatim was explicitly requested
        for s in raw_segments:
            seg_copy = dict(s)
            seg_copy["text_english"] = s["text_raw"]
            translated_segments.append(seg_copy)

    # Apply dialogue turn diarization (Speaker 1 vs Speaker 2)
    diarized_segments = diarize_dialogue_turns(translated_segments, model_name=preferred_ollama)

    # Extract multi-tab agent intelligence
    agent_intel = extract_agent_intelligence(english_transcript, model_name=preferred_ollama, llm_config=llm_cfg)

    # Format lyrics if music mode or FLAC audio
    lyrics_payload = None
    if is_music:
        try:
            # 1. First attempt canonical lookup via free LRCLIB API using file tags/filename
            meta = extract_audio_metadata(actual_filepath)
            track_title = meta.get("title") or Path(actual_filepath).stem
            track_artist = meta.get("artist") or ""
            track_duration = meta.get("duration") or info.duration

            print(f"🎵 [LYRICS ENGINE] Querying canonical database for '{track_title}' by '{track_artist}' ({track_duration}s)...")
            lrclib_res = query_lrclib(title=track_title, artist=track_artist, duration=track_duration)

            if lrclib_res and lrclib_res.get("syncedLyrics"):
                print(f"✅ [LRCLIB] Found human-verified synced lyrics for '{track_title}'!")
                initial_payload = parse_lrc_to_lyrics_data(
                    lrclib_res["syncedLyrics"],
                    title=lrclib_res.get("trackName") or track_title,
                    artist=lrclib_res.get("artistName") or track_artist,
                    total_duration=track_duration
                )
                print(f"⚡ [LYRICS ENGINE] Acoustically fusing canonical lyrics onto Whisper word timestamps...")
                lyrics_payload = align_canonical_lyrics_to_acoustic(initial_payload, raw_segments)
            elif lrclib_res and lrclib_res.get("plainLyrics"):
                print(f"✅ [LRCLIB] Found plain canonical text, force-aligning against acoustic segments...")
                lyrics_payload = align_reference_lyrics(
                    reference_text=lrclib_res["plainLyrics"],
                    acoustic_segments=raw_segments,
                    title=lrclib_res.get("trackName") or track_title,
                    artist=lrclib_res.get("artistName") or track_artist
                )
            else:
                # 2. Fallback to LLM stanza & syllable structuring from Whisper acoustic segments
                print(f"ℹ️ [LYRICS ENGINE] LRCLIB unlisted, structuring acoustic segments with LLM...")
                lyrics_payload = format_lyrics_with_llm(
                    raw_segments,
                    llm_config=llm_cfg,
                    song_title=track_title
                )
        except Exception as e:
            print(f"⚠️ Lyrics formatting notice: {e}")

    total_time = round(time.time() - start_time, 2)
    audio_duration = round(info.duration, 2)
    speed_factor = round(audio_duration / total_time, 1) if total_time > 0 else 1.0

    return {
        "status": "success",
        "detected_language": detected_lang,
        "language_probability": lang_prob,
        "audio_duration_sec": audio_duration,
        "processing_time_sec": total_time,
        "speed_factor": f"{speed_factor}x faster than real-time",
        "hindi_transcript": raw_transcript,
        "english_transcript": english_transcript,
        "segments": diarized_segments,
        "ollama_model_used": llm_label,
        "agent_intel": agent_intel,
        "is_music": is_music,
        "lyrics": lyrics_payload,
    }


class LyricsResolveRequest(BaseModel):
    filepath: Optional[str] = None
    title: Optional[str] = None
    artist: Optional[str] = None
    duration: Optional[float] = 0.0

@app.post("/api/lyrics/resolve")
async def resolve_lyrics_endpoint(req: LyricsResolveRequest):
    """Lookup canonical synced lyrics from LRCLIB using metadata from audio file or manual query."""
    title = req.title or ""
    artist = req.artist or ""
    duration = req.duration or 0.0

    if req.filepath and Path(req.filepath).exists():
        meta = extract_audio_metadata(req.filepath)
        if not title:
            title = meta.get("title") or Path(req.filepath).stem
        if not artist:
            artist = meta.get("artist") or ""
        if not duration:
            duration = meta.get("duration") or 0.0

    if not title:
        raise HTTPException(status_code=400, detail="Song title or valid audio filepath is required")

    lrclib_res = query_lrclib(title=title, artist=artist, duration=duration)
    if not lrclib_res:
        return {"status": "not_found", "message": f"No canonical lyrics found on LRCLIB for '{title}'"}

    if lrclib_res.get("syncedLyrics"):
        lyrics_data = parse_lrc_to_lyrics_data(
            lrclib_res["syncedLyrics"],
            title=lrclib_res.get("trackName") or title,
            artist=lrclib_res.get("artistName") or artist,
            total_duration=duration or float(lrclib_res.get("duration", 0.0))
        )
        if req.filepath and Path(req.filepath).exists():
            try:
                vocal_path = STEMS_DIR / "htdemucs" / Path(req.filepath).stem / "vocals.wav"
                target_audio = str(vocal_path) if vocal_path.exists() else req.filepath
                segs, _ = manager.get_model().transcribe(
                    target_audio,
                    beam_size=1,
                    word_timestamps=True,
                    vad_filter=False
                )
                ac_segs = []
                for s in segs:
                    w_list = [{"word": w.word, "start": w.start, "end": w.end} for w in (s.words or [])]
                    ac_segs.append({"start": s.start, "end": s.end, "words": w_list})
                lyrics_data = align_canonical_lyrics_to_acoustic(lyrics_data, ac_segs)
            except Exception as e:
                logger.warning(f"Fast acoustic alignment notice: {e}")
        return {"status": "success", "source": lyrics_data.get("source", "lrclib_synced"), "lyrics": lyrics_data}
    elif lrclib_res.get("plainLyrics"):
        return {
            "status": "plain_only",
            "source": "lrclib_plain",
            "plain_lyrics": lrclib_res["plainLyrics"],
            "title": lrclib_res.get("trackName") or title,
            "artist": lrclib_res.get("artistName") or artist
        }
    else:
        return {"status": "not_found", "message": "LRCLIB returned an entry with no lyrics"}


class LyricsAlignCustomRequest(BaseModel):
    reference_lyrics: str
    segments: Optional[List[Dict[str, Any]]] = None
    filepath: Optional[str] = None
    title: Optional[str] = "Audio Track"
    artist: Optional[str] = "Original Vocalist"

@app.post("/api/lyrics/align_custom")
async def align_custom_lyrics_endpoint(req: LyricsAlignCustomRequest):
    """Force-align user-pasted ground-truth reference lyrics onto acoustic segment landmarks."""
    segments = req.segments or []
    if not segments and req.filepath and Path(req.filepath).exists():
        segs_raw, info = manager.get_model().transcribe(
            req.filepath,
            beam_size=1,
            word_timestamps=True,
            vad_filter=False,
            condition_on_previous_text=False,
            no_speech_threshold=0.85
        )
        for s in segs_raw:
            segments.append({
                "start": s.start,
                "end": s.end,
                "text": s.text.strip(),
                "words": [{"word": w.word, "start": w.start, "end": w.end} for w in (s.words or [])]
            })

    if not segments:
        raise HTTPException(status_code=400, detail="Acoustic segments or a valid audio filepath is required for alignment")

    lyrics_data = align_reference_lyrics(
        reference_text=req.reference_lyrics,
        acoustic_segments=segments,
        title=req.title or "Audio Track",
        artist=req.artist or "Original Vocalist"
    )
    return {"status": "success", "lyrics": lyrics_data}


class LyricsFormatRequest(BaseModel):
    segments: List[Dict[str, Any]]
    llm_config: Optional[Dict[str, Any]] = None
    song_title: Optional[str] = None

@app.post("/api/lyrics/format")
async def format_lyrics_endpoint(req: LyricsFormatRequest):
    """Format transcribed segments into structured stanzas with Apple Music-style char timings."""
    lyrics = format_lyrics_with_llm(req.segments, llm_config=req.llm_config, song_title=req.song_title)
    return {"status": "success", "lyrics": lyrics}


class LyricsExportRequest(BaseModel):
    lyrics: Dict[str, Any]
    format: str = "lrc"  # "lrc", "elrc", "ttml", "json"

@app.post("/api/lyrics/export")
async def export_lyrics_endpoint(req: LyricsExportRequest):
    """Export formatted lyrics to standard LRC, Enhanced LRC, or Apple Music TTML."""
    fmt = req.format.lower()
    if fmt == "lrc":
        content = generate_lrc(req.lyrics)
        ext = "lrc"
    elif fmt == "elrc":
        content = generate_elrc(req.lyrics)
        ext = "elrc"
    elif fmt == "ttml":
        content = generate_ttml(req.lyrics)
        ext = "ttml"
    elif fmt == "json":
        content = json.dumps(req.lyrics, indent=2, ensure_ascii=False)
        ext = "json"
    else:
        raise HTTPException(status_code=400, detail=f"Unsupported format: {fmt}")

    return {
        "status": "success",
        "format": fmt,
        "extension": ext,
        "content": content
    }


class IntelRequest(BaseModel):
    english_transcript: str
    model_name: Optional[str] = None
    llm_config: Optional[Dict[str, Any]] = None

@app.post("/api/extract_agent_intel")
async def get_agent_intel(req: IntelRequest):
    """Re-extract structured intelligence from edited English text."""
    intel = extract_agent_intelligence(req.english_transcript, model_name=req.model_name, llm_config=req.llm_config)
    return {"status": "success", "agent_intel": intel}


class ChatTranscriptRequest(BaseModel):
    question: str
    english_transcript: str
    segments: List[Dict[str, Any]]
    model_name: Optional[str] = None
    llm_config: Optional[Dict[str, Any]] = None

@app.post("/api/chat_transcript")
async def chat_transcript_endpoint(req: ChatTranscriptRequest):
    """Chat over audio transcript with clickable timestamp citations."""
    answer = chat_with_transcript(
        question=req.question,
        english_transcript=req.english_transcript,
        segments=req.segments,
        model_name=req.model_name,
        llm_config=req.llm_config
    )
    return {"status": "success", "answer": answer}


class WebhookDispatchPayload(BaseModel):
    webhook_url: str
    payload: Dict[str, Any]

@app.post("/api/dispatch_webhook")
async def dispatch_webhook(data: WebhookDispatchPayload):
    """Dispatch transcribed English & agent intelligence to external LLM agent webhook."""
    import requests
    try:
        resp = requests.post(
            data.webhook_url,
            json=data.payload,
            timeout=10.0,
            headers={"Content-Type": "application/json", "User-Agent": "SonicScribe-Agent-Dispatcher/2.0"}
        )
        return {
            "status": "success" if resp.status_code < 400 else "warning",
            "status_code": resp.status_code,
            "message": f"Webhook responded with HTTP {resp.status_code}",
        }
    except Exception as e:
        return {
            "status": "error",
            "message": f"Failed to reach webhook at {data.webhook_url}: {str(e)}"
        }


class SubtitleBurnPayload(BaseModel):
    filepath: str
    segments: List[Dict[str, Any]]

@app.post("/api/burn_subtitles")
async def burn_subtitles_to_video(data: SubtitleBurnPayload):
    """
    Burn stylized subtitles (yellow bold text with solid black outline) into video using local FFmpeg.
    """
    input_path = Path(data.filepath)
    if not input_path.exists():
        raise HTTPException(status_code=404, detail="Source media file not found")

    # Generate temporary .srt
    srt_path = UPLOAD_DIR / f"temp_{int(time.time())}.srt"
    srt_lines = []
    for i, s in enumerate(data.segments, 1):
        start = format_timestamp(s["start"], srt_format=True)
        end = format_timestamp(s["end"], srt_format=True)
        text = s.get("text_english", s.get("text_raw", ""))
        srt_lines.append(f"{i}\n{start} --> {end}\n{text}\n")
    srt_path.write_text("\n".join(srt_lines), encoding="utf-8")

    output_filename = f"subtitled_{input_path.stem}.mp4"
    output_path = UPLOAD_DIR / output_filename

    # FFmpeg style definition: bold yellow with black outline
    style = "FontName=DejaVu Sans,FontSize=20,PrimaryColour=&H0000FFFF,OutlineColour=&H00000000,BorderStyle=3,Outline=2,Shadow=0,Alignment=2,MarginV=30"

    try:
        # Check if input has video stream
        probe = subprocess.run(
            ["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries", "stream=codec_type", "-of", "csv=p=0", str(input_path)],
            capture_output=True,
            text=True
        )
        has_video = bool(probe.stdout.strip())

        if has_video:
            cmd = [
                "ffmpeg", "-y", "-i", str(input_path),
                "-vf", f"subtitles={str(srt_path)}:force_style='{style}'",
                "-c:a", "copy",
                "-preset", "veryfast",
                str(output_path)
            ]
        else:
            # Audio-only file: generate video with dark oscilloscope background + burned subtitles
            cmd = [
                "ffmpeg", "-y",
                "-f", "lavfi", "-i", "color=c=#07090E:s=1280x720:r=25",
                "-i", str(input_path),
                "-vf", f"subtitles={str(srt_path)}:force_style='{style}'",
                "-c:v", "libx264", "-tune", "stillimage", "-c:a", "aac", "-b:a", "192k", "-pix_fmt", "yuv420p",
                "-shortest", "-preset", "ultrafast",
                str(output_path)
            ]

        res = subprocess.run(cmd, capture_output=True, text=True, timeout=120.0)
        if res.returncode != 0:
            raise RuntimeError(f"FFmpeg error: {res.stderr[-300:]}")

        return {
            "status": "success",
            "filename": output_filename,
            "download_url": f"/api/download_video/{output_filename}"
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        if srt_path.exists():
            srt_path.unlink()


@app.get("/api/download_video/{filename}")
async def download_video(filename: str):
    """Download burned subtitle video file."""
    video_path = UPLOAD_DIR / filename
    if not video_path.exists():
        raise HTTPException(status_code=404, detail="Video not found")
    return FileResponse(str(video_path), media_type="video/mp4", filename=filename)


class ExportPayload(BaseModel):
    segments: List[Dict[str, Any]]
    full_english: str
    format: str  # "txt", "srt", "vtt", "json"

@app.post("/api/export")
async def export_transcript(payload: ExportPayload):
    """Generate structured downloadable transcript files."""
    fmt = payload.format.lower()
    
    if fmt == "txt":
        content = payload.full_english
        media_type = "text/plain"
        ext = "txt"
    elif fmt == "srt":
        srt_lines = []
        for i, s in enumerate(payload.segments, 1):
            start = format_timestamp(s["start"], srt_format=True)
            end = format_timestamp(s["end"], srt_format=True)
            text = s.get("text_english", s.get("text_raw", ""))
            srt_lines.append(f"{i}\n{start} --> {end}\n{text}\n")
        content = "\n".join(srt_lines)
        media_type = "application/x-subrip"
        ext = "srt"
    elif fmt == "vtt":
        vtt_lines = ["WEBVTT\n"]
        for i, s in enumerate(payload.segments, 1):
            start = format_timestamp(s["start"], srt_format=False)
            end = format_timestamp(s["end"], srt_format=False)
            text = s.get("text_english", s.get("text_raw", ""))
            vtt_lines.append(f"{i}\n{start} --> {end}\n{text}\n")
        content = "\n".join(vtt_lines)
        media_type = "text/vtt"
        ext = "vtt"
    elif fmt == "md":
        md_lines = [
            "# ⚡ SonicScribe Studio Audio Intelligence Transcript",
            f"**Exported:** {time.strftime('%Y-%m-%d %H:%M:%S')}",
            "",
            "### Dialogue Timeline (with Timestamps)",
            ""
        ]
        for s in payload.segments:
            spk = s.get("speaker", "Speaker")
            t_start = s.get("start_str", format_timestamp(s.get("start", 0)))
            t_end = s.get("end_str", format_timestamp(s.get("end", 0)))
            txt = s.get("text_english", s.get("text_raw", ""))
            md_lines.append(f"- **`[{t_start} -> {t_end}]` {spk}:** {txt}")
        
        md_lines.extend([
            "",
            "### Full Continuous Transcript",
            "",
            payload.full_english or ""
        ])
        content = "\n".join(md_lines)
        media_type = "text/markdown"
        ext = "md"
    elif fmt == "json":
        import json
        content = json.dumps({
            "english_transcript": payload.full_english,
            "segments": payload.segments
        }, indent=2, ensure_ascii=False)
        media_type = "application/json"
        ext = "json"
    else:
        raise HTTPException(status_code=400, detail="Invalid format")

    return JSONResponse(content={
        "content": content,
        "filename": f"transcript_english.{ext}",
        "media_type": media_type
    })


class StemSeparateRequest(BaseModel):
    filepath: str
    device: Optional[str] = "auto"

@app.post("/api/stems/separate")
async def separate_stems_endpoint(req: StemSeparateRequest):
    """
    Separate an audio track into isolated vocals and vocal-free karaoke backing tracks (Demucs htdemucs).
    """
    try:
        result = separate_audio_stems(req.filepath, device=req.device or "auto")
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/stems/{track_name}/vocals")
async def stream_vocals_stem(track_name: str):
    """Serve the isolated acapella vocal track."""
    vocals_path = STEMS_DIR / "htdemucs" / track_name / "vocals.wav"
    if not vocals_path.exists():
        raise HTTPException(status_code=404, detail="Vocals stem not found")
    return FileResponse(str(vocals_path), media_type="audio/wav", filename=f"{track_name}_vocals.wav")

@app.get("/api/stems/{track_name}/instrumental")
async def stream_instrumental_stem(track_name: str):
    """Serve the vocal-free karaoke backing track."""
    inst_path = STEMS_DIR / "htdemucs" / track_name / "no_vocals.wav"
    if not inst_path.exists():
        raise HTTPException(status_code=404, detail="Instrumental stem not found")
    return FileResponse(str(inst_path), media_type="audio/wav", filename=f"{track_name}_karaoke.wav")


@app.websocket("/api/ws/dictate")
async def websocket_dictate_endpoint(websocket: WebSocket):
    """
    Real-Time Streaming Dictation WebSocket.
    Receives 16kHz 16-bit PCM (or Float32) audio binary chunks.
    Emits streaming interim transcripts and finalized sentence boundaries.
    """
    await websocket.accept()
    print("⚡ [WS DICTATE] Client connected for real-time dictation.")

    # Audio aggregation state
    audio_buffer = np.array([], dtype=np.float32)
    phrase_start_sec = 0.0
    last_transcribe_time = time.time()
    speaking = False
    silence_frames = 0
    phrase_id = 1

    model = manager.get_model()

    try:
        while True:
            message = await websocket.receive()
            if "bytes" in message and message["bytes"]:
                chunk_bytes = message["bytes"]
                if len(chunk_bytes) % 2 == 0:
                    chunk_samples = np.frombuffer(chunk_bytes, dtype=np.int16).astype(np.float32) / 32768.0
                elif len(chunk_bytes) % 4 == 0:
                    chunk_samples = np.frombuffer(chunk_bytes, dtype=np.float32)
                else:
                    continue

                if len(chunk_samples) == 0:
                    continue

                audio_buffer = np.concatenate([audio_buffer, chunk_samples])
                
                # Energy calculation (RMS)
                rms = float(np.sqrt(np.mean(chunk_samples**2))) if len(chunk_samples) > 0 else 0.0
                is_speech = rms > 0.012

                if is_speech:
                    speaking = True
                    silence_frames = 0
                elif speaking:
                    silence_frames += 1

                now = time.time()
                buf_duration = len(audio_buffer) / 16000.0

                # 1. Interim update (every ~350ms while speech buffer has >= 0.5s)
                if speaking and (now - last_transcribe_time > 0.35) and buf_duration >= 0.5:
                    last_transcribe_time = now
                    try:
                        segs, _ = model.transcribe(
                            audio_buffer,
                            beam_size=1,
                            temperature=0.0,
                            without_timestamps=True,
                            language="en"
                        )
                        interim_text = " ".join([s.text.strip() for s in segs if s.text.strip()]).strip()
                        if interim_text:
                            await websocket.send_json({
                                "type": "interim",
                                "phrase_id": phrase_id,
                                "text": interim_text,
                                "duration": round(buf_duration, 2),
                                "start": round(phrase_start_sec, 2),
                                "rms": round(rms, 4)
                            })
                    except Exception as e:
                        print(f"WS interim transcribe notice: {e}")

                # 2. Finalize phrase on natural pause (silence > 0.8s after speech)
                chunk_dur = len(chunk_samples) / 16000.0
                silence_duration = silence_frames * chunk_dur

                if speaking and silence_duration >= 0.8 and buf_duration >= 0.8:
                    try:
                        segs, _ = model.transcribe(
                            audio_buffer,
                            beam_size=1,
                            word_timestamps=True,
                            language="en"
                        )
                        final_text = " ".join([s.text.strip() for s in segs if s.text.strip()]).strip()
                        if final_text:
                            phrase_end_sec = phrase_start_sec + buf_duration
                            words_list = []
                            for s in segs:
                                if s.words:
                                    for w in s.words:
                                        words_list.append({
                                            "word": w.word.strip(),
                                            "start": round(phrase_start_sec + w.start, 2),
                                            "end": round(phrase_start_sec + w.end, 2)
                                        })

                            await websocket.send_json({
                                "type": "final",
                                "phrase_id": phrase_id,
                                "segment": {
                                    "id": phrase_id,
                                    "start": round(phrase_start_sec, 2),
                                    "end": round(phrase_end_sec, 2),
                                    "start_str": format_timestamp(phrase_start_sec),
                                    "end_str": format_timestamp(phrase_end_sec),
                                    "speaker": "Speaker 1",
                                    "text": final_text,
                                    "text_raw": final_text,
                                    "text_english": final_text,
                                    "words": words_list
                                }
                            })
                            phrase_id += 1
                            phrase_start_sec = phrase_end_sec
                    except Exception as e:
                        print(f"WS finalize error: {e}")

                    audio_buffer = np.array([], dtype=np.float32)
                    speaking = False
                    silence_frames = 0

            elif "text" in message and message["text"]:
                try:
                    data = json.loads(message["text"])
                    action = data.get("action")
                    if action == "stop":
                        if len(audio_buffer) >= 6000:
                            segs, _ = model.transcribe(audio_buffer, beam_size=1, word_timestamps=True)
                            final_text = " ".join([s.text.strip() for s in segs if s.text.strip()]).strip()
                            if final_text:
                                phrase_end_sec = phrase_start_sec + (len(audio_buffer) / 16000.0)
                                await websocket.send_json({
                                    "type": "final",
                                    "phrase_id": phrase_id,
                                    "segment": {
                                        "id": phrase_id,
                                        "start": round(phrase_start_sec, 2),
                                        "end": round(phrase_end_sec, 2),
                                        "start_str": format_timestamp(phrase_start_sec),
                                        "end_str": format_timestamp(phrase_end_sec),
                                        "speaker": "Speaker 1",
                                        "text": final_text,
                                        "text_raw": final_text,
                                        "text_english": final_text,
                                        "words": []
                                    }
                                })
                        await websocket.send_json({"type": "session_complete", "status": "ok"})
                        break
                except Exception as e:
                    print(f"WS control message notice: {e}")

    except WebSocketDisconnect:
        print("⚡ [WS DICTATE] Client disconnected gracefully.")
    except Exception as e:
        print(f"❌ [WS DICTATE] Error: {e}")


# =====================================================================
# 🎛️ AUDIOPHILE HIGH-RES MUSIC PLAYER & LIBRARY ENGINE
# =====================================================================

class PlaybackRequest(BaseModel):
    track_id: str
    mode: str = "dac_direct"  # "dac_direct" or "browser_stream"
    alsa_device: Optional[str] = None

class SeekRequest(BaseModel):
    seconds: float

class SwapLyricsRequest(BaseModel):
    track_id: str
    lyrics_content: str
    format: str = "lrc"

class SwapAudioRequest(BaseModel):
    track_id: str
    new_filepath: str

@app.get("/api/library/scan")
async def scan_library_endpoint(dir: Optional[str] = None):
    """Scan local music directory (defaults to /home/sumeet/Music/mpd)."""
    target_dir = dir if dir else DEFAULT_MUSIC_DIR
    tracks = scan_music_directory(target_dir)
    return {
        "status": "success",
        "directory": target_dir,
        "count": len(tracks),
        "tracks": tracks
    }

@app.get("/api/library/track/{track_id}/lyrics")
async def get_track_lyrics_endpoint(track_id: str):
    """Get parsed lyrics data for a library track."""
    track = get_track_by_id(track_id)
    if not track:
        raise HTTPException(status_code=404, detail="Track not found in library index")
    lyrics_data = load_track_lyrics_data(track)
    return {
        "status": "success",
        "track_id": track_id,
        "lyrics_status": track.get("lyrics_status"),
        "lyrics": lyrics_data
    }

@app.post("/api/library/track/{track_id}/auto_compile")
async def auto_compile_lyrics_endpoint(track_id: str):
    """Enqueue track for asynchronous background CUDA syllable alignment."""
    track = get_track_by_id(track_id)
    if not track:
        raise HTTPException(status_code=404, detail="Track not found in library index")
    job_id = enqueue_track_for_alignment(track)
    return {
        "status": "success",
        "track_id": track_id,
        "job_id": job_id,
        "message": f"Compilation enqueued in background for {track['title']}"
    }

@app.get("/api/library/worker/status/{track_id}")
async def worker_status_endpoint(track_id: str):
    """Check status of background syllable compiler worker."""
    status = get_job_status(track_id)
    return {
        "status": "success",
        "job": status
    }

@app.get("/api/library/stream/{track_id}")
async def stream_library_track(track_id: str):
    """Stream lossless audio file from library with Range header support."""
    track = get_track_by_id(track_id)
    if not track or not os.path.exists(track["filepath"]):
        raise HTTPException(status_code=404, detail="Track file not found on disk")
    filepath = Path(track["filepath"])
    media_type = "audio/flac" if filepath.suffix.lower() == ".flac" else "audio/mpeg"
    return FileResponse(str(filepath), media_type=media_type)

@app.get("/api/audiophile/dacs")
async def get_dacs_endpoint():
    """Enumerate hardware ALSA sound cards with Audiocular D07 highlighted."""
    devices = list_alsa_playback_devices()
    return {
        "status": "success",
        "devices": devices,
        "recommended_device": devices[0]["alsa_target"] if devices else "default"
    }

@app.get("/api/audiophile/status")
async def get_audiophile_status():
    """Return live DAC hardware parameters and player status."""
    player_state = dac_player.get_playback_state()
    hw_params = verify_live_hardware_params(card_num=2, dev_num=0)
    return {
        "status": "success",
        "player": player_state,
        "hardware_dac": hw_params
    }

@app.post("/api/audiophile/play")
async def play_audiophile_track(req: PlaybackRequest):
    """Start bit-perfect hardware playback or stream mode."""
    track = get_track_by_id(req.track_id)
    if not track:
        raise HTTPException(status_code=404, detail="Track not found")

    dac_player.engine_mode = req.mode
    if req.alsa_device:
        dac_player.set_target_device(req.alsa_device)

    if req.mode == "dac_direct":
        ok = dac_player.play(track["filepath"], track_meta=track)
        if not ok:
            raise HTTPException(status_code=500, detail="Failed to initialize bit-perfect DAC playback")
    
    return {
        "status": "success",
        "mode": req.mode,
        "track": track,
        "stream_url": f"/api/library/stream/{track['id']}" if req.mode == "browser_stream" else None,
        "player": dac_player.get_playback_state()
    }

@app.post("/api/audiophile/pause")
async def pause_audiophile():
    dac_player.pause()
    return {"status": "success", "player": dac_player.get_playback_state()}

@app.post("/api/audiophile/resume")
async def resume_audiophile():
    dac_player.resume()
    return {"status": "success", "player": dac_player.get_playback_state()}

@app.post("/api/audiophile/seek")
async def seek_audiophile(req: SeekRequest):
    dac_player.seek(req.seconds)
    return {"status": "success", "player": dac_player.get_playback_state()}

@app.post("/api/audiophile/swap_lyrics")
async def swap_lyrics_endpoint(req: SwapLyricsRequest):
    """Attach user-provided lyrics (.lrc/.ttml) directly alongside the audio file."""
    track = get_track_by_id(req.track_id)
    if not track:
        raise HTTPException(status_code=404, detail="Track not found")
    audio_path = Path(track["filepath"])
    ext = f".{req.format}"
    lyrics_path = audio_path.with_suffix(ext)
    with open(lyrics_path, "w", encoding="utf-8") as f:
        f.write(req.lyrics_content)
    track["lyrics_status"] = req.format
    track["lyrics_path"] = str(lyrics_path)
    track["has_synced_lyrics"] = req.format in ("ttml", "elrc", "lrc")
    return {
        "status": "success",
        "message": f"Saved {ext} alongside audio at {lyrics_path}",
        "track": track
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("server:app", host="0.0.0.0", port=7860, reload=False)
