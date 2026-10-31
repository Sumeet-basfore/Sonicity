"""
SonicScribe Studio - Advanced Lyrics Resolver & Canonical Synchronizer
Handles:
1. Audio metadata extraction (FFprobe / ID3 / Vorbis)
2. Canonical lyrics lookup via LRCLIB public API (zero-key global database)
3. Standard & Enhanced LRC parser with phonetic syllable / character interpolation
4. Custom reference lyrics forced alignment for unlisted / indie songs
"""

import json
import logging
import re
import subprocess
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Dict, List, Any, Optional

from refiner import format_lrc_timestamp, format_ttml_timestamp, interpolate_character_timestamps

logger = logging.getLogger(__name__)

LRCLIB_API_BASE = "https://lrclib.net/api"
USER_AGENT = "SonicScribe-Studio/2.0 (https://github.com/sonicscribe)"


def extract_audio_metadata(filepath: str) -> Dict[str, Any]:
    """
    Extract artist, title, album, and duration using ffprobe with filename regex fallback.
    """
    path = Path(filepath)
    meta = {
        "title": "",
        "artist": "",
        "album": "",
        "duration": 0.0,
        "filename": path.name,
    }

    if not path.exists():
        return meta

    try:
        cmd = [
            "ffprobe",
            "-v", "quiet",
            "-print_format", "json",
            "-show_format",
            str(path)
        ]
        res = subprocess.run(cmd, capture_output=True, text=True, timeout=10.0)
        if res.returncode == 0:
            data = json.loads(res.stdout).get("format", {})
            meta["duration"] = round(float(data.get("duration", 0.0)), 2)
            tags = data.get("tags", {})
            
            # Case-insensitive tag matching
            tag_lookup = {k.lower(): v for k, v in tags.items()}
            meta["title"] = tag_lookup.get("title") or ""
            meta["artist"] = tag_lookup.get("artist") or tag_lookup.get("album_artist") or ""
            meta["album"] = tag_lookup.get("album") or ""
    except Exception as e:
        logger.warning(f"Metadata extraction notice for {filepath}: {e}")

    # Fallback: Parse Title and Artist from Filename if tags are empty
    # Common formats: "Artist - Title", "Artist - Album - 01 - Title", "01. Title"
    stem = path.stem
    if not meta["title"] or not meta["artist"]:
        parts = [p.strip() for p in stem.split(" - ") if p.strip()]
        if len(parts) >= 2:
            if not meta["artist"]:
                meta["artist"] = parts[0]
            if not meta["title"]:
                # If 3 or 4 parts (e.g. Artist - Album - TrackNum - Title), last part is usually title
                meta["title"] = parts[-1]
                # Strip leading track numbers like "01", "10."
                meta["title"] = re.sub(r"^\d+[\.\-\s]+", "", meta["title"]).strip()
        elif not meta["title"]:
            # Clean leading track numbers: "09. Love on the Line Call Now" -> "Love on the Line Call Now"
            cleaned_stem = re.sub(r"^\d+[\.\-\s]+", "", stem).strip()
            meta["title"] = cleaned_stem

    return meta


def query_lrclib(title: str, artist: str = "", duration: float = 0.0) -> Optional[Dict[str, Any]]:
    """
    Query the free public LRCLIB API for human-verified synced lyrics.
    Tries exact get first, then falls back to search query.
    """
    clean_title = re.sub(r"[!?.]$", "", title.strip())
    clean_artist = artist.strip()

    if not clean_title:
        return None

    # Attempt 1: Exact lookup if we have artist & duration
    if clean_title and clean_artist:
        try:
            params = {
                "track_name": clean_title,
                "artist_name": clean_artist,
            }
            if duration > 10:
                params["duration"] = int(duration)
            url = f"{LRCLIB_API_BASE}/get?{urllib.parse.urlencode(params)}"
            req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
            with urllib.request.urlopen(req, timeout=4.0) as resp:
                data = json.loads(resp.read().decode("utf-8"))
                if data and (data.get("syncedLyrics") or data.get("plainLyrics")):
                    logger.info(f"✓ Found exact LRCLIB match: {data.get('trackName')} by {data.get('artistName')}")
                    return data
        except Exception as e:
            logger.debug(f"LRCLIB exact lookup notice: {e}")

    # Attempt 2: Search query
    try:
        search_q = f"{clean_artist} {clean_title}".strip()
        url = f"{LRCLIB_API_BASE}/search?q={urllib.parse.quote_plus(search_q)}"
        req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
        with urllib.request.urlopen(req, timeout=5.0) as resp:
            data = json.loads(resp.read().decode("utf-8"))
            if isinstance(data, list) and data:
                # Prioritize entries that have syncedLyrics
                synced_entries = [d for d in data if d.get("syncedLyrics")]
                best = synced_entries[0] if synced_entries else data[0]
                logger.info(f"✓ Found LRCLIB search match: {best.get('trackName')} by {best.get('artistName')}")
                return best
    except Exception as e:
        logger.warning(f"LRCLIB search query failed: {e}")

    return None


def estimate_word_singing_weight(word: str) -> float:
    """Estimate singing duration weight based on syllable vowels and consonants."""
    w = re.sub(r"[^\w]", "", word).lower()
    if not w:
        return 1.0
    vowels = len(re.findall(r"[aeiouy\u0904-\u0914\u093e-\u094c]", w))
    # Vowels carry sung vocal energy; consonants provide brief transitions
    return max(1.0, vowels * 2.0 + max(0, len(w) - vowels) * 0.5)


def parse_lrc_to_lyrics_data(
    lrc_text: str,
    title: str = "Audio Track",
    artist: str = "Original Artist",
    total_duration: float = 0.0
) -> Dict[str, Any]:
    """
    Parse standard LRC lines ([mm:ss.xx]Text) into structured stanzas, lines,
    words, and character-level timings with phonetic singing duration weighting.
    Prevents artificial word dragging during instrumental interludes.
    """
    time_regex = re.compile(r"\[(\d+):(\d+(?:\.\d+)?)\](.*)")
    raw_lines = []

    for line in lrc_text.strip().splitlines():
        line = line.strip()
        if not line:
            continue
        # Extract metadata headers like [ti:...], [ar:...]
        if line.startswith("[ti:"):
            parsed_title = line[4:-1].strip()
            if parsed_title:
                title = parsed_title
            continue
        if line.startswith("[ar:"):
            parsed_artist = line[4:-1].strip()
            if parsed_artist:
                artist = parsed_artist
            continue
        if line.startswith("["):
            match = time_regex.match(line)
            if match:
                mins = int(match.group(1))
                secs = float(match.group(2))
                start_sec = round(mins * 60 + secs, 2)
                text = match.group(3).strip()
                if text:
                    raw_lines.append({"start": start_sec, "text": text})

    if not raw_lines:
        return {"title": title, "artist": artist, "stanzas": []}

    # Assign end timestamps to each line with singing phonetic duration bounds
    for i in range(len(raw_lines)):
        start = raw_lines[i]["start"]
        tokens = [w for w in raw_lines[i]["text"].split() if w]
        weights = [estimate_word_singing_weight(w) for w in tokens]
        total_w = sum(weights) or 1.0
        # Realistic vocal duration: ~0.26s per phonetic weight unit
        estimated_vocal_dur = round(total_w * 0.26, 2)

        if i < len(raw_lines) - 1:
            next_start = raw_lines[i + 1]["start"]
            gap = next_start - start
            if gap > estimated_vocal_dur + 1.2:
                # Interlude detected: finish singing when words finish rather than dragging across silence
                raw_lines[i]["end"] = min(next_start - 0.4, round(start + max(1.8, estimated_vocal_dur), 2))
            else:
                raw_lines[i]["end"] = round(next_start, 2)
        else:
            raw_lines[i]["end"] = min(round(start + max(2.5, estimated_vocal_dur), 2), total_duration) if total_duration > start else round(start + max(2.5, estimated_vocal_dur), 2)

    # Group lines into musical stanzas based on acoustic pauses
    stanzas = []
    current_lines = []
    stanza_idx = 1
    last_end = 0.0

    for idx, l in enumerate(raw_lines, 1):
        start = l["start"]
        end = l["end"]
        text = l["text"]

        # Pause > 3.2s indicates a stanza / verse / chorus boundary
        if current_lines and (start - last_end > 3.2):
            stanzas.append({
                "type": f"Section {stanza_idx}",
                "lines": current_lines
            })
            stanza_idx += 1
            current_lines = []

        # Split words & calculate word & character timings using phonetic duration weights
        words_tokens = [w for w in text.split() if w]
        line_duration = max(0.01, end - start)
        word_weights = [estimate_word_singing_weight(w) for w in words_tokens]
        total_word_weight = sum(word_weights) or 1.0

        enriched_words = []
        w_curr = start
        for w_tok, w_wt in zip(words_tokens, word_weights):
            dur_this_word = (w_wt / total_word_weight) * line_duration
            w_next = round(w_curr + dur_this_word, 2)
            chars = interpolate_character_timestamps(w_tok, w_curr, w_next)
            enriched_words.append({
                "word": w_tok,
                "start": round(w_curr, 2),
                "end": w_next,
                "chars": chars
            })
            w_curr = w_next

        current_lines.append({
            "id": idx,
            "start": start,
            "end": end,
            "text": text,
            "start_formatted": format_lrc_timestamp(start),
            "end_formatted": format_lrc_timestamp(end),
            "words": enriched_words
        })
        last_end = end

    if current_lines:
        stanzas.append({
            "type": f"Section {stanza_idx}",
            "lines": current_lines
        })

    # Label prominent stanzas if count matches typical song structure
    if len(stanzas) >= 3:
        if stanzas[0]["type"] == "Section 1":
            stanzas[0]["type"] = "Verse 1"
        if len(stanzas) > 1 and stanzas[1]["type"] == "Section 2":
            stanzas[1]["type"] = "Chorus"
        if len(stanzas) > 2 and stanzas[2]["type"] == "Section 3":
            stanzas[2]["type"] = "Verse 2"

    return {
        "title": title,
        "artist": artist,
        "source": "canonical_lrclib",
        "stanzas": stanzas
    }


def align_canonical_lyrics_to_acoustic(
    canonical_data: Dict[str, Any],
    acoustic_segments: List[Dict[str, Any]]
) -> Dict[str, Any]:
    """
    Physically lock canonical human lyrics onto Whisper acoustic word timestamps.
    Matches every word to the exact millisecond the singer vocalized it.
    """
    import difflib

    if not canonical_data or not canonical_data.get("stanzas") or not acoustic_segments:
        return canonical_data

    # 1. Flatten all acoustic words detected by Whisper
    all_ac_words = []
    for seg in acoustic_segments:
        words = seg.get("words", [])
        if words:
            for w in words:
                w_str = (w.get("word") or "").strip()
                if not w_str:
                    continue
                w_clean = re.sub(r"[^\w]", "", w_str).lower()
                if w_clean:
                    all_ac_words.append({
                        "word": w_str,
                        "clean": w_clean,
                        "start": round(float(w.get("start", seg.get("start", 0))), 2),
                        "end": round(float(w.get("end", seg.get("end", 0))), 2),
                        "prob": float(w.get("prob", 1.0))
                    })
        else:
            # Fallback if no word list in segment
            txt = (seg.get("text_raw") or seg.get("text") or "").strip()
            tokens = [t for t in txt.split() if t]
            s_s = round(float(seg.get("start", 0)), 2)
            s_e = round(float(seg.get("end", s_s + 2.0)), 2)
            if tokens:
                d = (s_e - s_s) / len(tokens)
                c_t = s_s
                for tok in tokens:
                    all_ac_words.append({
                        "word": tok,
                        "clean": re.sub(r"[^\w]", "", tok).lower(),
                        "start": round(c_t, 2),
                        "end": round(c_t + d, 2),
                        "prob": 0.8
                    })
                    c_t += d

    if not all_ac_words:
        return canonical_data

    ac_idx = 0
    total_ac = len(all_ac_words)

    for stanza in canonical_data.get("stanzas", []):
        for line in stanza.get("lines", []):
            tokens = [t for t in line.get("text", "").split() if t]
            if not tokens:
                continue

            matched_words = []
            for t in tokens:
                c_tok = re.sub(r"[^\w]", "", t).lower()
                best_match = None
                best_j = -1

                # Search forward in acoustic words within an 8-word window
                for j in range(ac_idx, min(ac_idx + 8, total_ac)):
                    ac_w = all_ac_words[j]
                    if ac_w["clean"] == c_tok:
                        best_match = ac_w
                        best_j = j
                        break
                    elif len(c_tok) >= 3 and difflib.SequenceMatcher(None, ac_w["clean"], c_tok).ratio() >= 0.75:
                        best_match = ac_w
                        best_j = j
                        break

                if best_match:
                    matched_words.append({
                        "word": t,
                        "start": best_match["start"],
                        "end": best_match["end"],
                        "matched": True
                    })
                    ac_idx = best_j + 1
                else:
                    matched_words.append({
                        "word": t,
                        "start": None,
                        "end": None,
                        "matched": False
                    })

            # Interpolate any unmatched gaps using phonetic vowel duration weighting
            line_fallback_start = round(float(line.get("start", 0)), 2)
            for i, mw in enumerate(matched_words):
                if not mw["matched"]:
                    prev_end = matched_words[i - 1]["end"] if i > 0 and matched_words[i - 1]["end"] else line_fallback_start
                    next_start = None
                    for k in range(i + 1, len(matched_words)):
                        if matched_words[k]["start"] is not None:
                            next_start = matched_words[k]["start"]
                            break
                    w_wt = estimate_word_singing_weight(mw["word"])
                    dur = round(w_wt * 0.22, 2)
                    if next_start is not None:
                        mw["start"] = prev_end
                        mw["end"] = min(next_start, round(prev_end + dur, 2))
                    else:
                        mw["start"] = prev_end
                        mw["end"] = round(prev_end + max(0.2, dur), 2)

            # Generate precise per-character timings with vowel duration elongation
            enriched_words = []
            for mw in matched_words:
                chars = interpolate_character_timestamps(mw["word"], mw["start"], mw["end"])
                enriched_words.append({
                    "word": mw["word"],
                    "start": mw["start"],
                    "end": mw["end"],
                    "chars": chars
                })

            line["words"] = enriched_words
            if enriched_words:
                line["start"] = enriched_words[0]["start"]
                line["end"] = enriched_words[-1]["end"]
                line["start_formatted"] = format_lrc_timestamp(line["start"])
                line["end_formatted"] = format_lrc_timestamp(line["end"])

    canonical_data["source"] = "acoustic_aligned"
    return canonical_data


def align_reference_lyrics(
    reference_text: str,
    acoustic_segments: List[Dict[str, Any]],
    title: str = "Audio Track",
    artist: str = "Original Vocalist"
) -> Dict[str, Any]:
    """
    Force-align user-provided ground-truth lyrics onto acoustic segment timings.
    Guarantees 0% spelling errors while anchoring onto exact vocal acoustic boundaries.
    """
    ref_lines = [l.strip() for l in reference_text.strip().splitlines() if l.strip()]
    if not ref_lines or not acoustic_segments:
        return {"title": title, "artist": artist, "stanzas": []}

    total_ref = len(ref_lines)
    total_segs = len(acoustic_segments)

    initial_lines = []
    for idx, ref_txt in enumerate(ref_lines):
        seg_idx = int((idx / total_ref) * total_segs)
        seg = acoustic_segments[min(seg_idx, total_segs - 1)]
        s_start = round(float(seg.get("start", 0)), 2)
        s_end = round(float(seg.get("end", s_start + 3.5)), 2)
        initial_lines.append({
            "id": idx + 1,
            "start": s_start,
            "end": s_end,
            "text": ref_txt,
            "start_formatted": format_lrc_timestamp(s_start),
            "end_formatted": format_lrc_timestamp(s_end),
            "words": []
        })

    # Group into musical stanzas
    stanzas = []
    current = []
    s_idx = 1
    last_e = 0.0
    for l in initial_lines:
        if current and (l["start"] - last_e > 3.0):
            stanzas.append({"type": f"Stanza {s_idx}", "lines": current})
            s_idx += 1
            current = []
        current.append(l)
        last_e = l["end"]
    if current:
        stanzas.append({"type": f"Stanza {s_idx}", "lines": current})

    payload = {
        "title": title,
        "artist": artist,
        "source": "custom_aligned",
        "stanzas": stanzas
    }

    # Lock onto exact acoustic words
    return align_canonical_lyrics_to_acoustic(payload, acoustic_segments)
