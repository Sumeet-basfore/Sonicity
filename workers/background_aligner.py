"""
Asynchronous Background Syllable & TTML Lyrics Compiler.
Compiles letter-by-letter synced .ttml and .elrc files alongside audio tracks without interrupting playback.
"""

import os
import time
import queue
import logging
import threading
from pathlib import Path
from typing import Dict, Any, Optional, Callable

from lyrics_resolver import (
    query_lrclib,
    parse_lrc_to_lyrics_data,
    align_canonical_lyrics_to_acoustic
)
from refiner import generate_ttml, generate_elrc, generate_lrc
from transcriber import WhisperManager

logger = logging.getLogger(__name__)

# Background Task Queue
alignment_queue = queue.Queue()
active_jobs: Dict[str, Dict[str, Any]] = {}
completed_callbacks: Dict[str, Callable] = {}


def worker_loop():
    """Worker thread processing background acoustic alignment jobs on CUDA."""
    whisper_mgr = WhisperManager()

    while True:
        try:
            job = alignment_queue.get()
            if job is None:
                break

            track_id = job["track_id"]
            audio_path = Path(job["filepath"])
            title = job.get("title", audio_path.stem)
            artist = job.get("artist", "Original Vocalist")
            duration = job.get("duration", 0.0)

            active_jobs[track_id] = {
                "status": "PROCESSING",
                "track_id": track_id,
                "filepath": str(audio_path),
                "title": title,
                "start_time": time.time(),
                "progress_msg": "Searching canonical lyrics on LRCLIB..."
            }
            logger.info(f"⚡ [BG Worker] Starting syllable compilation for: {title} ({audio_path.name})")

            # 1. Fetch canonical lyrics from LRCLIB
            lrc_result = query_lrclib(title=title, artist=artist, duration=duration)
            canonical_lyrics_data = None

            if lrc_result and lrc_result.get("synced_lyrics"):
                active_jobs[track_id]["progress_msg"] = "Parsing canonical synced lyrics..."
                canonical_lyrics_data = parse_lrc_to_lyrics_data(
                    lrc_text=lrc_result["synced_lyrics"],
                    title=title,
                    artist=artist,
                    total_duration=duration
                )
            elif lrc_result and lrc_result.get("plain_lyrics"):
                active_jobs[track_id]["progress_msg"] = "Parsing plain lyrics..."
                canonical_lyrics_data = parse_lrc_to_lyrics_data(
                    lrc_text=lrc_result["plain_lyrics"],
                    title=title,
                    artist=artist,
                    total_duration=duration
                )

            # 2. Run Whisper acoustic word extraction on CUDA
            active_jobs[track_id]["progress_msg"] = "Running faster-whisper acoustic word alignment on CUDA..."
            model = whisper_mgr.get_model(model_name="large-v3-turbo", device="cuda", compute_type="int8_float16")

            # Transcribe with word timestamps
            segments_gen, _ = model.transcribe(
                str(audio_path),
                word_timestamps=True,
                vad_filter=False, # Keep instrumental pauses intact
                beam_size=1
            )

            acoustic_segments = []
            for seg in segments_gen:
                w_list = []
                if seg.words:
                    for w in seg.words:
                        w_list.append({
                            "word": w.word.strip(),
                            "start": round(w.start, 2),
                            "end": round(w.end, 2)
                        })
                acoustic_segments.append({
                    "start": round(seg.start, 2),
                    "end": round(seg.end, 2),
                    "text": seg.text.strip(),
                    "words": w_list
                })

            # 3. Align canonical lyrics onto acoustic word millisecond boundaries
            final_lyrics_data = None
            if canonical_lyrics_data and canonical_lyrics_data.get("stanzas"):
                active_jobs[track_id]["progress_msg"] = "Mapping syllables onto vocal cords..."
                final_lyrics_data = align_canonical_lyrics_to_acoustic(canonical_lyrics_data, acoustic_segments)
            else:
                # Compile directly from acoustic Whisper segments
                active_jobs[track_id]["progress_msg"] = "Compiling direct acoustic syllables..."
                from lyrics_resolver import align_reference_lyrics
                ref_text = "\n".join(s["text"] for s in acoustic_segments if s["text"])
                final_lyrics_data = align_reference_lyrics(ref_text, acoustic_segments, title=title, artist=artist)

            # 4. Generate .ttml and .lrc files alongside song with identical full name
            ttml_content = generate_ttml(final_lyrics_data)
            lrc_content = generate_lrc(final_lyrics_data)

            target_ttml_path = audio_path.with_suffix(".ttml")
            target_lrc_path = audio_path.with_suffix(".lrc")

            with open(target_ttml_path, "w", encoding="utf-8") as f:
                f.write(ttml_content)
            with open(target_lrc_path, "w", encoding="utf-8") as f:
                f.write(lrc_content)

            logger.info(f"✓ [BG Worker] Saved .ttml & .lrc for '{title}' alongside audio at: {target_ttml_path}")

            active_jobs[track_id] = {
                "status": "COMPLETED",
                "track_id": track_id,
                "filepath": str(audio_path),
                "ttml_path": str(target_ttml_path),
                "lrc_path": str(target_lrc_path),
                "lyrics_data": final_lyrics_data,
                "elapsed": round(time.time() - active_jobs[track_id]["start_time"], 1)
            }

            # Call subscriber callback if registered
            cb = completed_callbacks.get(track_id)
            if cb:
                cb(active_jobs[track_id])

        except Exception as e:
            logger.error(f"❌ [BG Worker] Alignment failed for job: {e}", exc_info=True)
            if "track_id" in locals() and track_id in active_jobs:
                active_jobs[track_id]["status"] = "ERROR"
                active_jobs[track_id]["error"] = str(e)
        finally:
            alignment_queue.task_done()


# Start background worker daemon thread
_bg_thread = threading.Thread(target=worker_loop, daemon=True)
_bg_thread.start()


def enqueue_track_for_alignment(track_data: Dict[str, Any], callback: Optional[Callable] = None) -> str:
    """Add a track to the background compilation queue."""
    track_id = track_data["id"]
    if track_id in active_jobs and active_jobs[track_id]["status"] in ("QUEUED", "PROCESSING"):
        return track_id

    if callback:
        completed_callbacks[track_id] = callback

    job = {
        "track_id": track_id,
        "filepath": track_data["filepath"],
        "title": track_data.get("title", ""),
        "artist": track_data.get("artist", ""),
        "duration": track_data.get("duration", 0.0)
    }

    active_jobs[track_id] = {
        "status": "QUEUED",
        "track_id": track_id,
        "filepath": track_data["filepath"],
        "title": job["title"],
        "progress_msg": "Queued for background CUDA syllable alignment..."
    }

    alignment_queue.put(job)
    return track_id


def get_job_status(track_id: str) -> Optional[Dict[str, Any]]:
    """Query background compilation job status."""
    return active_jobs.get(track_id)
