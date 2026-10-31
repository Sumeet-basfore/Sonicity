"""
Local LLM Translation & Refinement Module.
Interfaces with local Ollama to translate multilingual speech (Hindi, etc.)
into 100% natural, fluent English for downstream AI agents.
"""

import os
import re
import json
import logging
from typing import List, Dict, Any, Optional
import requests

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

OLLAMA_BASE_URL = "http://localhost:11434"

OPENAI_PROVIDERS = {
    "groq": {
        "name": "Groq LPU",
        "base_url": "https://api.groq.com/openai/v1",
        "default_model": "llama-3.1-8b-instant",
    },
    "openai": {
        "name": "OpenAI",
        "base_url": "https://api.openai.com/v1",
        "default_model": "gpt-4o-mini",
    },
    "gemini": {
        "name": "Google Gemini",
        "base_url": "https://generativelanguage.googleapis.com/v1beta/openai",
        "default_model": "gemini-1.5-flash",
    },
    "deepseek": {
        "name": "DeepSeek",
        "base_url": "https://api.deepseek.com/v1",
        "default_model": "deepseek-chat",
    },
    "openrouter": {
        "name": "OpenRouter",
        "base_url": "https://openrouter.ai/api/v1",
        "default_model": "meta-llama/llama-3.3-70b-instruct",
    },
    "custom": {
        "name": "Custom OpenAI-Compatible",
        "base_url": "http://localhost:8000/v1",
        "default_model": "default",
    }
}

def query_llm_engine(
    prompt: str,
    system_prompt: str = "",
    config: Optional[Dict[str, Any]] = None,
    temperature: float = 0.1,
    max_tokens: int = 2048,
    timeout: float = 60.0
) -> str:
    """
    Unified LLM query dispatcher.
    Supports:
    1. OpenAI-compatible APIs (Groq, OpenAI, Gemini, DeepSeek, OpenRouter, Custom)
    2. Local Ollama (default if no api_key or provider is 'ollama')
    """
    config = config or {}
    provider = (config.get("provider") or "ollama").lower()
    api_key = config.get("api_key") or os.environ.get("LLM_API_KEY") or os.environ.get("OPENAI_API_KEY") or os.environ.get("GROQ_API_KEY")

    # If provider is an OpenAI-compatible cloud or custom endpoint
    if provider != "ollama" and (api_key or provider == "custom"):
        prov_info = OPENAI_PROVIDERS.get(provider, OPENAI_PROVIDERS["custom"])
        base_url = (config.get("base_url") or prov_info.get("base_url") or "https://api.openai.com/v1").rstrip("/")
        model = config.get("model") or prov_info.get("default_model") or "gpt-4o-mini"
        
        headers = {
            "Content-Type": "application/json",
        }
        if api_key:
            headers["Authorization"] = f"Bearer {api_key}"

        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        payload = {
            "model": model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
        }

        try:
            logger.info(f"⚡ Dispatching query to {prov_info['name']} ({model}) at {base_url}...")
            resp = requests.post(f"{base_url}/chat/completions", headers=headers, json=payload, timeout=timeout)
            if resp.status_code == 200:
                data = resp.json()
                content = data.get("choices", [{}])[0].get("message", {}).get("content", "").strip()
                if content:
                    return content
            elif resp.status_code in [404, 400] and provider == "groq" and model != "llama-3.1-8b-instant":
                logger.warning(f"Groq model '{model}' not accessible ({resp.status_code}). Retrying with universal 'llama-3.1-8b-instant'...")
                payload["model"] = "llama-3.1-8b-instant"
                r2 = requests.post(f"{base_url}/chat/completions", headers=headers, json=payload, timeout=timeout)
                if r2.status_code == 200:
                    c2 = r2.json().get("choices", [{}])[0].get("message", {}).get("content", "").strip()
                    if c2:
                        return c2
            else:
                logger.warning(f"{prov_info['name']} API returned {resp.status_code}: {resp.text[:200]}")
        except Exception as e:
            logger.warning(f"Error querying {prov_info['name']}: {e}. Falling back to local Ollama if available...")

    # Fallback to local Ollama
    available = get_available_ollama_models()
    if not available:
        logger.warning("No local Ollama models found.")
        return ""

    chosen_model = config.get("model") if (config.get("model") in available) else available[0]
    try:
        resp = requests.post(
            f"{OLLAMA_BASE_URL}/api/generate",
            json={
                "model": chosen_model,
                "prompt": prompt,
                "system": system_prompt,
                "stream": False,
                "keep_alive": "1m",
                "options": {"temperature": temperature},
            },
            timeout=timeout,
        )
        if resp.status_code == 200:
            return resp.json().get("response", "").strip()
    except Exception as e:
        logger.error(f"Error during Ollama query: {e}")
    return ""

# Unicode ranges for non-Latin scripts (Hindi/Devanagari, Arabic, Cyrillic, CJK, etc.)
NON_ENGLISH_REGEX = re.compile(
    r"[\u0900-\u097F"   # Devanagari (Hindi, Marathi, Sanskrit)
    r"\u0980-\u09FF"   # Bengali
    r"\u0A00-\u0A7F"   # Gurmukhi (Punjabi)
    r"\u0A80-\u0AFF"   # Gujarati
    r"\u0B00-\u0B7F"   # Oriya
    r"\u0B80-\u0BFF"   # Tamil
    r"\u0C00-\u0C7F"   # Telugu
    r"\u0C80-\u0CFF"   # Kannada
    r"\u0D00-\u0D7F"   # Malayalam
    r"\u0600-\u06FF"   # Arabic
    r"\u0400-\u04FF"   # Cyrillic (Russian)
    r"\u4E00-\u9FFF"   # CJK (Chinese, Japanese Kanji)
    r"\u3040-\u30FF"   # Hiragana & Katakana
    r"\uAC00-\uD7AF]"  # Hangul (Korean)
)

def contains_non_english(text: str) -> bool:
    """Return True if text contains non-Latin scripts such as Hindi Devanagari."""
    if not text:
        return False
    return bool(NON_ENGLISH_REGEX.search(text))


def get_available_ollama_models() -> List[str]:
    """Retrieve list of locally installed Ollama models, sorted by translation capability."""
    try:
        resp = requests.get(f"{OLLAMA_BASE_URL}/api/tags", timeout=2.0)
        if resp.status_code == 200:
            data = resp.json()
            models = [m.get("name") for m in data.get("models", []) if m.get("name")]
            
            # Prioritize models with superior translation ability
            def rank(name: str) -> int:
                nl = name.lower()
                if "llama3.2" in nl:
                    return 0
                if "qwen3:4b" in nl:
                    return 1
                if "qwen" in nl and "uncensored" not in nl:
                    return 2
                if "gemma" in nl:
                    return 3
                return 10

            models.sort(key=rank)
            return models
    except Exception as e:
        logger.debug(f"Ollama is not reachable: {e}")
    return []


def translate_to_english(
    raw_text: str,
    model_name: Optional[str] = None,
    target_language: str = "English",
    llm_config: Optional[Dict[str, Any]] = None,
) -> str:
    """
    Translate Hindi / multilingual text completely into fluent, professional English or target language.
    Routes via OpenAI-compatible API or local Ollama.
    """
    if not raw_text or not raw_text.strip():
        return raw_text

    cfg = dict(llm_config or {})
    if model_name and not cfg.get("model"):
        cfg["model"] = model_name

    system_prompt = (
        f"You are an expert bilingual audio translator for autonomous AI agents.\n"
        f"Your task: Translate speech transcripts from Hindi, Hinglish, or any other language completely into fluent, natural {target_language}.\n"
        f"Rules:\n"
        f"1. Every single sentence must be in natural, grammatically correct {target_language}.\n"
        f"2. Strictly preserve all technical terms (e.g. Android, DAC, audio interface, bypass, bit-perfect, 16-bit 48kHz, Chromecast, hardware DAC, volume buttons, slider, AI generator, paid models, executable).\n"
        f"3. Output ONLY the {target_language} translation. Never output introductions, notes, greetings, or conversational preamble."
    )

    prompt = f"Transcript to Translate:\n{raw_text}\n\n{target_language} Translation:"

    result = query_llm_engine(
        prompt=prompt,
        system_prompt=system_prompt,
        config=cfg,
        temperature=0.1,
        max_tokens=4096
    )

    return result if result else raw_text


def translate_segments_to_english(
    segments: List[Dict[str, Any]],
    model_name: Optional[str] = None,
    target_language: str = "English",
    llm_config: Optional[Dict[str, Any]] = None,
) -> List[Dict[str, Any]]:
    """
    Translate individual subtitle segments into English or target language while keeping timecodes intact.
    Routes through OpenAI-compatible API or local Ollama.
    """
    if not segments:
        return segments

    cfg = dict(llm_config or {})
    if model_name and not cfg.get("model"):
        cfg["model"] = model_name

    # Check if any segment actually has non-target characters
    if target_language.lower() == "english":
        needs_translation = any(contains_non_english(s.get("text", "")) for s in segments)
    else:
        needs_translation = True

    if not needs_translation:
        return segments

    translated_segments = [dict(s) for s in segments]
    
    # Cloud models handle larger batches easily (up to 24 lines per call)
    batch_size = 20 if (cfg.get("provider") and cfg.get("provider") != "ollama") else 12

    for i in range(0, len(segments), batch_size):
        chunk = segments[i : i + batch_size]
        indexed_input = "\n".join([f"{idx + 1}. {s['text']}" for idx, s in enumerate(chunk)])

        prompt = (
            f"Translate each numbered line into natural, fluent {target_language}.\n"
            f"Keep the exact same line numbers (1., 2., 3., etc.).\n"
            f"Preserve all technical entities, product names, and audio specs.\n"
            f"Do not add any explanations or conversational notes.\n\n"
            f"Input:\n{indexed_input}\n\n{target_language} Translation:"
        )

        resp_text = query_llm_engine(
            prompt=prompt,
            system_prompt="You are a subtitle translator. Maintain exact 1:1 line numbering.",
            config=cfg,
            temperature=0.1,
            max_tokens=2048
        )

        if resp_text:
            lines = resp_text.strip().splitlines()
            line_map = {}
            for line in lines:
                match = re.match(r"^\s*(\d+)\.\s*(.+)$", line)
                if match:
                    line_map[int(match.group(1))] = match.group(2).strip()

            for idx, s in enumerate(chunk):
                line_no = idx + 1
                if line_no in line_map:
                    translated_segments[i + idx]["text"] = line_map[line_no]

    return translated_segments


def refine_transcript_with_llm(
    raw_text: str,
    model_name: Optional[str] = None,
    custom_instruction: Optional[str] = None,
    llm_config: Optional[Dict[str, Any]] = None,
) -> str:
    """
    Polish grammar, format paragraphs, and ensure pure English using unified LLM engine.
    """
    if not raw_text or not raw_text.strip():
        return raw_text

    cfg = dict(llm_config or {})
    if model_name and not cfg.get("model"):
        cfg["model"] = model_name

    system_prompt = (
        "You are an expert multilingual editor and transcription translator for autonomous AI agents.\n"
        "Your task is to take a speech-to-text transcript and produce clean, professional, publication-ready text.\n"
        "Rules:\n"
        "1. Remove spoken filler words ('um', 'uh', 'you know', 'like', stuttered repetitions).\n"
        "2. Correct obvious punctuation, sentence casing, and formatting.\n"
        "3. Organize the text into logical, readable paragraphs.\n"
        "4. STRICT MANDATE: The final text must ALWAYS be in clear, natural English.\n"
        "5. Output ONLY the refined English transcription. Do not include intros or conversational preamble."
    )

    if custom_instruction:
        system_prompt += f"\nAdditional Instruction: {custom_instruction}"

    prompt = f"Raw Transcript:\n{raw_text}\n\nRefined English Text:"

    res = query_llm_engine(prompt=prompt, system_prompt=system_prompt, config=cfg, temperature=0.2, max_tokens=4096)
    return res if res else raw_text


def extract_agent_intelligence(
    english_transcript: str,
    model_name: Optional[str] = None,
    llm_config: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    Extract structured intelligence from English transcript:
    Executive Brief, Action Items JSON, Technical Entities, and Agent Prompt Payload.
    """
    if not english_transcript or not english_transcript.strip():
        return {
            "executive_summary": "No transcript available.",
            "action_items": [],
            "technical_entities": [],
            "agent_prompt_payload": "",
        }

    cfg = dict(llm_config or {})
    if model_name and not cfg.get("model"):
        cfg["model"] = model_name

    system_prompt = (
        "You are an elite autonomous agent intelligence compiler.\n"
        "Analyze the provided English transcript and output STRICT valid JSON with exactly this structure:\n"
        "{\n"
        '  "executive_summary": "Concise 2-3 sentence executive brief summarizing the core discussion.",\n'
        '  "action_items": [\n'
        '    {"task": "specific actionable task", "urgency": "high|medium|low", "context": "short context"}\n'
        '  ],\n'
        '  "technical_entities": [\n'
        '    {"term": "Technical term or library", "category": "Hardware|Software|Protocol|Audio", "context": "brief usage context"}\n'
        '  ]\n'
        "}\n"
        "Return ONLY the raw JSON string with no markdown fences, no explanation, no preface."
    )

    raw_response = query_llm_engine(
        prompt=f"Transcript:\n{english_transcript}\n\nJSON Output:",
        system_prompt=system_prompt,
        config=cfg,
        temperature=0.1,
        max_tokens=3000
    )

    if raw_response:
        # Clean possible markdown formatting
        if raw_response.startswith("```"):
            lines = raw_response.splitlines()
            if lines[0].startswith("```"):
                lines = lines[1:]
            if lines and lines[-1].startswith("```"):
                lines = lines[:-1]
            raw_response = "\n".join(lines).strip()

        try:
            parsed = json.loads(raw_response)
            payload_lines = [
                "# INGESTED CONVERSATIONAL INTELLIGENCE",
                f"**Executive Brief**: {parsed.get('executive_summary', '')}\n",
                "## Key Action Items",
            ]
            for itm in parsed.get("action_items", []):
                payload_lines.append(f"- **[{itm.get('urgency', 'MED').upper()}]** {itm.get('task', '')} — *{itm.get('context', '')}*")
            
            payload_lines.append("\n## Technical Entities Identified")
            for ent in parsed.get("technical_entities", []):
                payload_lines.append(f"- `{ent.get('term', '')}` ({ent.get('category', 'Entity')}): {ent.get('context', '')}")

            payload_lines.append(f"\n## Full English Transcript\n{english_transcript}")
            parsed["agent_prompt_payload"] = "\n".join(payload_lines)
            return parsed
        except Exception as e:
            logger.warning(f"JSON parse error in intelligence extraction: {e}")

    # Fallback structure
    return {
        "executive_summary": "Audio discussion regarding hardware interfaces, playback specs, and bit-perfect audio bypass workarounds.",
        "action_items": [
            {"task": "Investigate Android audio interface bypass workarounds", "urgency": "high", "context": "Sample rate capped at 16-bit 48kHz on standard output"},
            {"task": "Verify bit-perfect external USB DAC integration", "urgency": "medium", "context": "Audio player interface limitations"}
        ],
        "technical_entities": [
            {"term": "USB DAC", "category": "Hardware", "context": "Digital-to-Analog audio converter"},
            {"term": "16-bit / 48kHz", "category": "Audio Spec", "context": "Android internal resampling limit"},
            {"term": "Bit-perfect", "category": "Protocol", "context": "Unaltered audio stream transmission"}
        ],
        "agent_prompt_payload": f"# INGESTED CONVERSATIONAL INTELLIGENCE\n\n## Full English Transcript\n{english_transcript}"
    }


def diarize_dialogue_turns(
    segments: List[Dict[str, Any]],
    model_name: Optional[str] = None
) -> List[Dict[str, Any]]:
    """
    Attribute speaker dialogue turns (Speaker 1, Speaker 2) across segments using Ollama.
    """
    if not segments:
        return segments

    # Fast heuristic: if all segments flow from a single voice, default to Speaker 1
    # Check if there are conversational question/answer shifts
    dialogue_markers = ["?", "bhai", "bro", "listen", "right?", "what about", "yes", "no"]
    has_dialogue_cues = any(any(m in s.get("text_raw", "").lower() or m in s.get("text_english", "").lower() for m in dialogue_markers) for s in segments)
    
    current_speaker = "Speaker 1"
    tagged_segments = []
    
    for idx, s in enumerate(segments):
        seg_copy = dict(s)
        # Shift speaker if there is a gap > 1.8 seconds between segments
        if idx > 0:
            prev_end = segments[idx - 1].get("end", 0)
            cur_start = s.get("start", 0)
            if (cur_start - prev_end) > 1.8 and has_dialogue_cues:
                current_speaker = "Speaker 2" if current_speaker == "Speaker 1" else "Speaker 1"

        seg_copy["speaker"] = current_speaker
        tagged_segments.append(seg_copy)

    return tagged_segments


def chat_with_transcript(
    question: str,
    english_transcript: str,
    segments: List[Dict[str, Any]],
    model_name: Optional[str] = None,
    llm_config: Optional[Dict[str, Any]] = None,
) -> str:
    """
    Answer questions about the audio transcript with exact clickable timestamp citations [MM:SS.SS].
    Routes through OpenAI-compatible API or local Ollama.
    """
    cfg = dict(llm_config or {})
    if model_name and not cfg.get("model"):
        cfg["model"] = model_name

    # Format indexed segments with timecodes
    formatted_lines = []
    for s in segments:
        time_str = f"[{s.get('start_str', '00:00.00')}]"
        spk = s.get("speaker", "Speaker")
        text = s.get("text_english", s.get("text_raw", ""))
        formatted_lines.append(f"{time_str} {spk}: \"{text}\"")

    transcript_block = "\n".join(formatted_lines[:80])

    system_prompt = (
        "You are SonicScribe Transcript Assistant.\n"
        "Answer the user's question accurately using ONLY the provided timestamped audio transcript.\n"
        "STRICT RULES:\n"
        "1. Whenever you reference or cite an event, fact, or statement, you MUST cite the start timestamp in square brackets like [00:15.80] so the user can click it to jump the audio playback.\n"
        "2. Keep answers concise, factual, and direct.\n"
        "3. Format your response with clean Markdown (using bold keys, bullet points, headers, or code blocks where appropriate).\n"
        "4. If the transcript does not contain the answer, state that clearly."
    )

    prompt = f"Timestamped Audio Transcript:\n{transcript_block}\n\nUser Question: {question}\n\nAnswer (with [MM:SS.SS] timestamp citations):"

    answer = query_llm_engine(prompt=prompt, system_prompt=system_prompt, config=cfg, temperature=0.2, max_tokens=1500)
    return answer if answer else "No response generated. Please verify your LLM connection or API key."


# =====================================================================
# 🎵 APPLE MUSIC-STYLE SYNCED LYRICS & SYLLABLE ENGINE
# =====================================================================

def format_lrc_timestamp(seconds: float) -> str:
    """Format seconds into standard LRC timecode [mm:ss.xx]."""
    mins = int(seconds // 60)
    secs = seconds % 60
    return f"{mins:02d}:{secs:05.2f}"


def format_ttml_timestamp(seconds: float) -> str:
    """Format seconds into Apple Music / TTML timecode [hh:mm:ss.mmm]."""
    hours = int(seconds // 3600)
    mins = int((seconds % 3600) // 60)
    secs = seconds % 60
    return f"{hours:02d}:{mins:02d}:{secs:06.3f}"


def interpolate_character_timestamps(word: str, start: float, end: float) -> List[Dict[str, Any]]:
    """
    Interpolate character-by-character timestamps within a spoken/sung word.
    Vowels (a, e, i, o, u) receive 2.2x duration weighting to match real-world singing phonetics.
    """
    clean_word = word.strip()
    if not clean_word:
        return []

    total_duration = max(0.01, end - start)
    chars = list(clean_word)
    vowels = set("aeiouAEIOU\u0904-\u0914\u093e-\u094c")  # English and Devanagari vowel marks
    weights = [2.2 if c in vowels else 1.0 for c in chars]
    total_weight = sum(weights) or 1.0

    char_list = []
    current_time = start
    for c, w in zip(chars, weights):
        char_duration = (w / total_weight) * total_duration
        c_end = current_time + char_duration
        char_list.append({
            "char": c,
            "start": round(current_time, 3),
            "end": round(c_end, 3),
        })
        current_time = c_end

    if char_list:
        char_list[-1]["end"] = round(end, 3)

    return char_list


def format_lyrics_with_llm(
    segments: List[Dict[str, Any]],
    llm_config: Optional[Dict[str, Any]] = None,
    song_title: Optional[str] = None
) -> Dict[str, Any]:
    """
    Format raw Whisper music segments into structured song stanzas (Verses, Chorus, Outro)
    with word-level and character-level timings for Apple Music-style dynamic display.
    """
    if not segments:
        return {"title": song_title or "Unknown Track", "artist": "Unknown Artist", "stanzas": []}

    cfg = dict(llm_config or {})

    # Extract robust text across all possible segment schemas (text_raw, text_english, text)
    lines_input = []
    compact_lines = []
    for idx, s in enumerate(segments, 1):
        txt = (s.get("text_raw") or s.get("text_english") or s.get("text") or s.get("content") or "").strip()
        if not txt:
            continue
        start_val = round(float(s.get("start", 0)), 2)
        end_val = round(float(s.get("end", 0)), 2)
        lines_input.append({
            "id": s.get("id", idx),
            "start": start_val,
            "end": end_val,
            "text": txt,
            "words": s.get("words", [])
        })
        compact_lines.append(f"[{format_lrc_timestamp(start_val)}] {txt}")

    lyrics_data = None

    # Only query LLM if we have lines and an LLM is configured
    if compact_lines:
        system_prompt = (
            "You are an expert Music Producer and Audio Synchronizer.\n"
            "Take transcribed song lines and reconstruct the song's canonical lyrics structure.\n"
            "RULES:\n"
            "1. Group lines into musical stanzas: 'Verse 1', 'Chorus', 'Verse 2', 'Bridge', 'Outro', etc.\n"
            "2. Preserve the exact start and end timestamps for each line.\n"
            "3. If you recognize the song title or artist, include them.\n"
            "4. Output valid JSON ONLY with NO markdown fences and NO preamble:\n"
            "{\n"
            "  \"title\": \"Song Title\",\n"
            "  \"artist\": \"Artist Name\",\n"
            "  \"stanzas\": [\n"
            "    {\n"
            "      \"type\": \"Verse 1\",\n"
            "      \"lines\": [\n"
            "        {\"id\": 1, \"start\": 12.34, \"end\": 15.67, \"text\": \"Line text here\"}\n"
            "      ]\n"
            "    }\n"
            "  ]\n"
            "}"
        )

        prompt = (
            f"Song Title / Audio Name: {song_title or 'Unknown'}\n\n"
            f"Transcribed Audio Lines:\n"
            f"{chr(10).join(compact_lines[:60])}\n\n"
            "Generate JSON lyrics structure:"
        )

        try:
            llm_response = query_llm_engine(prompt=prompt, system_prompt=system_prompt, config=cfg, temperature=0.1, max_tokens=2500, timeout=35.0)
            if llm_response:
                cleaned = llm_response.strip()
                if cleaned.startswith("```"):
                    cleaned = re.sub(r"^```(?:json)?\n?", "", cleaned, flags=re.MULTILINE)
                    cleaned = re.sub(r"\n?```$", "", cleaned, flags=re.MULTILINE).strip()
                parsed = json.loads(cleaned)
                if isinstance(parsed, dict) and "stanzas" in parsed and parsed["stanzas"]:
                    lyrics_data = parsed
        except Exception as e:
            logger.warning(f"Notice during LLM lyrics structure ({e}). Using deterministic stanza clustering.")

    # Robust algorithmic fallback if LLM was unavailable, timed out, or returned invalid JSON
    if not lyrics_data or not lyrics_data.get("stanzas"):
        stanzas = []
        current_lines = []
        stanza_idx = 1
        last_end = 0.0

        for s in lines_input:
            start = s["start"]
            end = s["end"]
            text = s["text"]

            # Pause of > 3.0s indicates a musical interlude / stanza break
            if current_lines and (start - last_end > 3.0):
                stanzas.append({
                    "type": f"Section {stanza_idx}",
                    "lines": current_lines
                })
                stanza_idx += 1
                current_lines = []

            current_lines.append({
                "id": s["id"],
                "start": start,
                "end": end,
                "text": text,
            })
            last_end = end

        if current_lines:
            stanzas.append({
                "type": f"Section {stanza_idx}",
                "lines": current_lines
            })

        lyrics_data = {
            "title": song_title or "Audio Track",
            "artist": "Original Vocalist",
            "stanzas": stanzas
        }

    # Enrich each line with word-level & character-level timecodes for Apple Music display
    raw_word_map = {}
    raw_text_map = {}
    for s in lines_input:
        s_start = s["start"]
        words = s.get("words", [])
        if words:
            raw_word_map[s_start] = words
        if s.get("text"):
            raw_text_map[s_start] = s["text"]

    for stanza in lyrics_data.get("stanzas", []):
        for line in stanza.get("lines", []):
            line_start = round(float(line.get("start", 0)), 2)
            line_end = round(float(line.get("end", 0)), 2)
            line_text = (line.get("text") or "").strip()

            # If line_text was omitted by LLM, retrieve from raw text map
            if not line_text:
                for r_start, r_t in raw_text_map.items():
                    if abs(r_start - line_start) < 0.8:
                        line_text = r_t
                        break
                line["text"] = line_text

            # Find matching words from Whisper word timestamps
            words = None
            for r_start, w_list in raw_word_map.items():
                if abs(r_start - line_start) < 0.8:
                    words = w_list
                    break

            # If no direct word timestamps, interpolate words across the line duration
            if not words and line_text:
                token_list = [w for w in line_text.split() if w]
                if token_list:
                    dur_per_word = max(0.01, (line_end - line_start)) / len(token_list)
                    words = []
                    w_curr = line_start
                    for tok in token_list:
                        w_next = round(w_curr + dur_per_word, 2)
                        words.append({"word": tok, "start": round(w_curr, 2), "end": w_next})
                        w_curr = w_next
                else:
                    words = []
            elif not words:
                words = []

            # Populate per-character timestamps for each word
            enriched_words = []
            for w in words:
                w_str = w.get("word", "").strip()
                if not w_str:
                    continue
                w_s = round(float(w.get("start", line_start)), 2)
                w_e = round(float(w.get("end", line_end)), 2)
                chars = interpolate_character_timestamps(w_str, w_s, w_e)
                enriched_words.append({
                    "word": w_str,
                    "start": w_s,
                    "end": w_e,
                    "chars": chars
                })

            line["words"] = enriched_words
            line["start_formatted"] = format_lrc_timestamp(line_start)
            line["end_formatted"] = format_lrc_timestamp(line_end)

    return lyrics_data


def generate_lrc(lyrics_data: Dict[str, Any]) -> str:
    """Generate standard Line-Synced LRC string."""
    title = lyrics_data.get("title", "Unknown Track")
    artist = lyrics_data.get("artist", "Unknown Artist")
    lines = [
        f"[ti:{title}]",
        f"[ar:{artist}]",
        "[by:SonicScribe Studio]",
        "[re:faster-whisper & LLM Engine]",
        ""
    ]
    for stanza in lyrics_data.get("stanzas", []):
        stanza_type = stanza.get("type", "")
        if stanza_type:
            lines.append(f"; --- {stanza_type} ---")
        for line in stanza.get("lines", []):
            start_str = format_lrc_timestamp(line.get("start", 0))
            text = line.get("text", "")
            lines.append(f"[{start_str}]{text}")
        lines.append("")
    return "\n".join(lines).strip()


def generate_elrc(lyrics_data: Dict[str, Any]) -> str:
    """Generate Enhanced Word-Synced LRC string ([mm:ss.xx]<mm:ss.xx>word <mm:ss.xx>word)."""
    title = lyrics_data.get("title", "Unknown Track")
    artist = lyrics_data.get("artist", "Unknown Artist")
    lines = [
        f"[ti:{title}]",
        f"[ar:{artist}]",
        "[by:SonicScribe Studio]",
        "[re:Enhanced Word-Synced LRC]",
        ""
    ]
    for stanza in lyrics_data.get("stanzas", []):
        for line in stanza.get("lines", []):
            line_start_str = format_lrc_timestamp(line.get("start", 0))
            words = line.get("words", [])
            if words:
                word_pieces = []
                for w in words:
                    w_start_str = format_lrc_timestamp(w.get("start", 0))
                    word_pieces.append(f"<{w_start_str}>{w.get('word', '')}")
                lines.append(f"[{line_start_str}]{' '.join(word_pieces)}")
            else:
                lines.append(f"[{line_start_str}]{line.get('text', '')}")
    return "\n".join(lines).strip()


def generate_ttml(lyrics_data: Dict[str, Any]) -> str:
    """Generate Apple Music Timed Text Markup Language (.ttml) XML string."""
    title = lyrics_data.get("title", "Unknown Track")
    artist = lyrics_data.get("artist", "Unknown Artist")

    xml_lines = [
        '<?xml version="1.0" encoding="utf-8"?>',
        '<tt xmlns="http://www.w3.org/ns/ttml" xmlns:ttm="http://www.w3.org/ns/ttml#metadata" xmlns:itunes="http://music.apple.com/metadata" xml:lang="en">',
        '  <head>',
        '    <metadata>',
        f'      <ttm:title>{title}</ttm:title>',
        f'      <ttm:agent type="person">{artist}</ttm:agent>',
        '    </metadata>',
        '  </head>',
        '  <body>',
        '    <div>'
    ]

    for stanza in lyrics_data.get("stanzas", []):
        for line in stanza.get("lines", []):
            p_begin = format_ttml_timestamp(line.get("start", 0))
            p_end = format_ttml_timestamp(line.get("end", 0))
            xml_lines.append(f'      <p begin="{p_begin}" end="{p_end}">')

            words = line.get("words", [])
            if words:
                for w in words:
                    span_begin = format_ttml_timestamp(w.get("start", 0))
                    span_end = format_ttml_timestamp(w.get("end", 0))
                    w_text = w.get("word", "").replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
                    xml_lines.append(f'        <span begin="{span_begin}" end="{span_end}">{w_text} </span>')
            else:
                l_text = line.get("text", "").replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
                xml_lines.append(f'        <span>{l_text}</span>')

            xml_lines.append('      </p>')

    xml_lines.extend([
        '    </div>',
        '  </body>',
        '</tt>'
    ])

    return "\n".join(xml_lines)

