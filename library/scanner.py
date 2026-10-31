"""
Music Library Scanner & Audiophile Metadata Indexer.
Scans high-res audio tracks (FLAC, WAV, ALAC, DSD, MP3), extracts 24/32-bit sample depth,
retrieves embedded cover art, samples dominant color palettes, and pairs accompanying lyrics (.ttml, .elrc, .lrc).
"""

import os
import io
import json
import hashlib
import sqlite3
import logging
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple

from PIL import Image
import mutagen
from mutagen.flac import FLAC
from mutagen.mp3 import MP3
from mutagen.wave import WAVE
from mutagen.oggvorbis import OggVorbis

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

DEFAULT_MUSIC_DIR = "/home/sumeet/Music/mpd"
COVER_CACHE_DIR = Path("/home/sumeet/audio-to-text/static/covers")
COVER_CACHE_DIR.mkdir(parents=True, exist_ok=True)
DB_PATH = Path("/home/sumeet/audio-to-text/library.db")

AUDIO_EXTENSIONS = {".flac", ".wav", ".dsf", ".dff", ".m4a", ".alac", ".mp3", ".ogg", ".opus"}
LYRICS_EXTENSIONS = {".ttml", ".elrc", ".lrc", ".txt"}

TRACK_CACHE: Dict[str, Dict[str, Any]] = {}


def get_track_by_id(track_id: str) -> Optional[Dict[str, Any]]:
    """Retrieve track from memory cache, sqlite DB, or fallback to scanning."""
    if track_id in TRACK_CACHE:
        return TRACK_CACHE[track_id]

    if DB_PATH.exists():
        try:
            conn = sqlite3.connect(DB_PATH)
            cur = conn.cursor()
            cur.execute("SELECT filepath FROM tracks WHERE id = ?", (track_id,))
            row = cur.fetchone()
            conn.close()
            if row and os.path.exists(row[0]):
                meta = parse_track_metadata(row[0])
                if meta:
                    TRACK_CACHE[track_id] = meta
                    return meta
        except Exception:
            pass

    # If cache is totally empty, trigger scan of DEFAULT_MUSIC_DIR
    if not TRACK_CACHE and os.path.exists(DEFAULT_MUSIC_DIR):
        scan_music_directory(DEFAULT_MUSIC_DIR)
        return TRACK_CACHE.get(track_id)

    return None


def get_track_id(filepath: str) -> str:
    """Generate deterministic track hash from path."""
    return hashlib.sha256(filepath.encode("utf-8")).hexdigest()[:16]


def format_seconds(sec: float) -> str:
    m = int(sec // 60)
    s = int(sec % 60)
    return f"{m:02d}:{s:02d}"


def extract_dominant_colors(image_bytes: bytes) -> Dict[str, Any]:
    """
    Extract 3 dominant colors from album art image bytes to power
    the dynamic 3D Fluid Aurora lyrics canvas.
    """
    try:
        img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        img.thumbnail((80, 80))
        # Quantize to 6 colors
        quantized = img.quantize(colors=6, method=Image.Quantize.MEDIANCUT).convert("RGB")
        palette_colors = quantized.getcolors(maxcolors=256)
        # Sort by frequency
        palette_colors.sort(key=lambda x: x[0], reverse=True)
        
        extracted = []
        for count, (r, g, b) in palette_colors[:5]:
            # Avoid pure black or pure white dominance for vivid aurora
            brightness = (r * 299 + g * 587 + b * 114) / 1000
            if 15 < brightness < 240:
                extracted.append({"r": r, "g": g, "b": b})
        
        if not extracted:
            extracted = [
                {"r": 255, "g": 42, "b": 133},
                {"r": 0, "g": 195, "b": 255},
                {"r": 123, "g": 44, "b": 191}
            ]
        elif len(extracted) < 3:
            extracted.append({"r": 0, "g": 229, "b": 117})
            extracted.append({"r": 255, "g": 229, "b": 0})
            
        return {
            "primary": f"rgb({extracted[0]['r']},{extracted[0]['g']},{extracted[0]['b']})",
            "poles": extracted[:5]
        }
    except Exception as e:
        logger.warning(f"Color extraction error: {e}")
        return {
            "primary": "rgb(0, 229, 117)",
            "poles": [
                {"r": 255, "g": 42, "b": 133},
                {"r": 0, "g": 195, "b": 255},
                {"r": 123, "g": 44, "b": 191},
                {"r": 0, "g": 229, "b": 117},
                {"r": 255, "g": 74, "b": 0}
            ]
        }


def find_associated_lyrics(audio_path: Path) -> Tuple[str, Optional[str]]:
    """
    Search for lyrics matching the identical basename:
    Priority: .ttml -> .elrc -> .lrc -> .txt
    """
    base_stem = audio_path.stem
    parent_dir = audio_path.parent

    for ext in [".ttml", ".elrc", ".lrc", ".txt"]:
        candidate = parent_dir / f"{base_stem}{ext}"
        if candidate.exists() and candidate.is_file():
            status = ext.replace(".", "")
            return status, str(candidate)

    return "none", None


def extract_cover_art(audio_file: mutagen.FileType, audio_path: Path, track_id: str) -> Tuple[bool, Optional[str], Dict[str, Any]]:
    """
    Extract embedded album artwork or find folder/cover.jpg.
    Caches image to static/covers/{track_id}.jpg.
    """
    cache_file = COVER_CACHE_DIR / f"{track_id}.jpg"
    if cache_file.exists():
        try:
            with open(cache_file, "rb") as f:
                colors = extract_dominant_colors(f.read())
            return True, f"/static/covers/{track_id}.jpg", colors
        except Exception:
            pass

    image_data = None

    # 1. Try embedded FLAC pictures
    if isinstance(audio_file, FLAC) and audio_file.pictures:
        image_data = audio_file.pictures[0].data

    # 2. Try MP3 ID3 APIC
    elif isinstance(audio_file, MP3):
        for key in audio_file.tags.keys() if audio_file.tags else []:
            if key.startswith("APIC"):
                image_data = audio_file.tags[key].data
                break

    # 3. Try folder images
    if not image_data:
        parent = audio_path.parent
        for candidate_name in ["cover.jpg", "cover.png", "folder.jpg", "folder.png", "front.jpg", "front.png"]:
            cand = parent / candidate_name
            if cand.exists() and cand.is_file():
                try:
                    with open(cand, "rb") as f:
                        image_data = f.read()
                    break
                except Exception:
                    continue

    if image_data:
        try:
            with open(cache_file, "wb") as f:
                f.write(image_data)
            colors = extract_dominant_colors(image_data)
            return True, f"/static/covers/{track_id}.jpg", colors
        except Exception as e:
            logger.warning(f"Failed to cache cover art for {audio_path}: {e}")

    default_colors = {
        "primary": "rgb(0, 229, 117)",
        "poles": [
            {"r": 255, "g": 42, "b": 133},
            {"r": 0, "g": 195, "b": 255},
            {"r": 123, "g": 44, "b": 191},
            {"r": 0, "g": 229, "b": 117},
            {"r": 255, "g": 74, "b": 0}
        ]
    }
    return False, None, default_colors


def parse_track_metadata(filepath: str) -> Optional[Dict[str, Any]]:
    """Parse complete audiophile metadata, cover art, and lyrics pairing for a single track."""
    path = Path(filepath)
    if not path.exists() or not path.is_file():
        return None

    track_id = get_track_id(filepath)
    try:
        audio = mutagen.File(filepath)
        if not audio:
            return None
    except Exception as e:
        logger.warning(f"Mutagen error parsing {filepath}: {e}")
        return None

    # Technical Specs
    info = audio.info
    sample_rate = getattr(info, "sample_rate", 44100)
    bits_per_sample = getattr(info, "bits_per_sample", 16)
    channels = getattr(info, "channels", 2)
    duration = getattr(info, "length", 0.0)
    bitrate = getattr(info, "bitrate", 0)
    if not bitrate and duration > 0:
        file_size = path.stat().st_size
        bitrate = int((file_size * 8) / duration)

    is_hires = (sample_rate > 48000) or (bits_per_sample > 16)

    # Tags / Vorbis Comments / ID3
    tags = audio.tags or {}
    
    def get_tag_val(keys: List[str], default: str = "") -> str:
        for k in keys:
            if hasattr(tags, "get"):
                v = tags.get(k)
                if v:
                    if isinstance(v, list):
                        return str(v[0]).strip()
                    return str(v).strip()
        return default

    title = get_tag_val(["title", "TITLE", "Title"], default=path.stem)
    artist = get_tag_val(["artist", "ARTIST", "Artist", "albumartist"], default="Original Vocalist")
    album = get_tag_val(["album", "ALBUM", "Album"], default="Single / Audio Track")
    year = get_tag_val(["date", "DATE", "year", "YEAR"], default="")
    genre = get_tag_val(["genre", "GENRE"], default="Lossless Audio")
    track_num = get_tag_val(["tracknumber", "TRACKNUMBER"], default="")

    # Lyrics file pairing
    lyrics_status, lyrics_path = find_associated_lyrics(path)

    # Check for embedded lyrics if no external file found
    embedded_lyrics = get_tag_val(["lyrics", "LYRICS", "unsyncedlyrics", "UNSYNCEDLYRICS"])
    if lyrics_status == "none" and embedded_lyrics:
        lyrics_status = "embedded"

    # Cover art extraction
    has_cover, cover_url, palette = extract_cover_art(audio, path, track_id)

    file_size_kb = round(path.stat().st_size / 1024, 1)

    return {
        "id": track_id,
        "filepath": str(path.resolve()),
        "filename": path.name,
        "title": title,
        "artist": artist,
        "album": album,
        "year": year,
        "genre": genre,
        "track_number": track_num,
        "duration": round(duration, 2),
        "duration_formatted": format_seconds(duration),
        "file_size_kb": file_size_kb,
        "sample_rate": sample_rate,
        "bit_depth": bits_per_sample,
        "channels": channels,
        "bitrate_kbps": round(bitrate / 1000, 1) if bitrate else 0,
        "is_hires": is_hires,
        "has_cover": has_cover,
        "cover_url": cover_url,
        "palette": palette,
        "lyrics_status": lyrics_status,
        "lyrics_path": lyrics_path,
        "has_synced_lyrics": lyrics_status in {"ttml", "elrc", "lrc"}
    }


def scan_music_directory(dir_path: str = DEFAULT_MUSIC_DIR) -> List[Dict[str, Any]]:
    """
    Recursively scan directory for audio files and extract rich metadata.
    """
    root = Path(dir_path).expanduser().resolve()
    if not root.exists() or not root.is_dir():
        logger.error(f"Directory not found: {dir_path}")
        return []

    tracks = []
    logger.info(f"Scanning music library in: {root}")
    
    for current_dir, _, files in os.walk(root):
        for f in sorted(files):
            ext = Path(f).suffix.lower()
            if ext in AUDIO_EXTENSIONS:
                full_path = os.path.join(current_dir, f)
                data = parse_track_metadata(full_path)
                if data:
                    TRACK_CACHE[data["id"]] = data
                    tracks.append(data)

    logger.info(f"Successfully indexed {len(tracks)} tracks from {root}")
    return tracks


def load_track_lyrics_data(track_data: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """Load and parse lyrics from file (.ttml, .elrc, .lrc) or embedded tags."""
    from lyrics_resolver import parse_lrc_to_lyrics_data
    lyrics_path = track_data.get("lyrics_path")
    title = track_data.get("title", "Song Track")
    artist = track_data.get("artist", "Original Vocalist")
    duration = track_data.get("duration", 0.0)

    if lyrics_path and os.path.exists(lyrics_path):
        ext = Path(lyrics_path).suffix.lower()
        try:
            content = Path(lyrics_path).read_text(encoding="utf-8")
            if ext in (".lrc", ".elrc", ".txt"):
                return parse_lrc_to_lyrics_data(content, title=title, artist=artist, total_duration=duration)
            elif ext == ".ttml":
                import xml.etree.ElementTree as ET
                try:
                    root = ET.fromstring(content)
                    stanzas = []
                    lines = []
                    for p in root.iter():
                        if p.tag.endswith("p") or p.tag == "p":
                            p_begin = p.attrib.get("begin", "00:00:00.000")
                            p_end = p.attrib.get("end", "00:00:00.000")
                            def parse_ttml_time(t_str):
                                parts = t_str.split(":")
                                if len(parts) == 3:
                                    return float(parts[0]) * 3600 + float(parts[1]) * 60 + float(parts[2])
                                elif len(parts) == 2:
                                    return float(parts[0]) * 60 + float(parts[1])
                                return float(t_str)

                            p_start_sec = parse_ttml_time(p_begin)
                            p_end_sec = parse_ttml_time(p_end)
                            words = []
                            for span in p.iter():
                                if span.tag.endswith("span") or span.tag == "span":
                                    s_begin = span.attrib.get("begin")
                                    s_end = span.attrib.get("end")
                                    s_text = span.text or ""
                                    if s_begin and s_end and s_text:
                                        words.append({
                                            "word": s_text,
                                            "start": parse_ttml_time(s_begin),
                                            "end": parse_ttml_time(s_end)
                                        })
                            line_text = "".join(w["word"] for w in words) if words else (p.text or "")
                            lines.append({
                                "start": p_start_sec,
                                "end": p_end_sec,
                                "text": line_text.strip(),
                                "words": words
                            })
                    if lines:
                        stanzas.append({"type": "Main", "lines": lines})
                        return {"title": title, "artist": artist, "stanzas": stanzas, "source": "ttml"}
                except Exception:
                    pass
        except Exception as e:
            logger.warning(f"Error loading lyrics from {lyrics_path}: {e}")

    # Fallback: check embedded lyrics tag in audio file
    filepath = track_data.get("filepath")
    if filepath and os.path.exists(filepath):
        try:
            audio = mutagen.File(filepath)
            if audio and audio.tags:
                lyrics_tag = None
                for k in ["lyrics", "LYRICS", "unsyncedlyrics", "UNSYNCEDLYRICS"]:
                    if k in audio.tags:
                        v = audio.tags[k]
                        lyrics_tag = str(v[0] if isinstance(v, list) else v)
                        break
                if lyrics_tag:
                    return parse_lrc_to_lyrics_data(lyrics_tag, title=title, artist=artist, total_duration=duration)
        except Exception:
            pass

    return None
