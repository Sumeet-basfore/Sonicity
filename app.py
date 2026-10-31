"""
SonicScribe Studio - Premium Local Audio-to-English Studio.
100% Private, Hardware-Accelerated Speech Recognition with faster-whisper & Gradio.
Features automatic speech-to-English translation for multilingual audio (Hindi, etc.).
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
        # Re-exec process once if run directly without LD_LIBRARY_PATH set
        if not os.environ.get("_SONICSCRIBE_CUDA_BOOTSTRAPPED"):
            os.environ["_SONICSCRIBE_CUDA_BOOTSTRAPPED"] = "1"
            try:
                os.execv(sys.executable, [sys.executable] + sys.argv)
            except Exception:
                pass

import time
import tempfile
import gradio as gr
from transcriber import WhisperManager, AVAILABLE_MODELS
from refiner import (
    get_available_ollama_models,
    contains_non_english,
    translate_to_english,
    translate_segments_to_english,
    refine_transcript_with_llm,
)
from downloader import download_audio_from_url

# Initialize backend engine & pre-warm default model
manager = WhisperManager()

print("⚡ Pre-warming faster-whisper large-v3-turbo on CUDA...")
try:
    manager.get_model(model_name="large-v3-turbo", device="cuda", compute_type="int8_float16")
    print("✅ Model warm and ready in GPU VRAM!")
except Exception as e:
    print(f"⚠️ Pre-warm notice: {e}")

ollama_models = get_available_ollama_models()
default_ollama = ollama_models[0] if ollama_models else "None detected"

TEMP_EXPORT_DIR = os.path.join(tempfile.gettempdir(), "sonicscribe_exports")
os.makedirs(TEMP_EXPORT_DIR, exist_ok=True)

POPULAR_LANGUAGES = [
    ("🌐 Auto-Detect Spoken Language", "Auto-Detect"),
    ("🇮🇳 Hindi (hi)", "hi"),
    ("🇺🇸 English (en)", "en"),
    ("🇪🇸 Spanish (es)", "es"),
    ("🇫🇷 French (fr)", "fr"),
    ("🇩🇪 German (de)", "de"),
    ("🇨🇳 Chinese (zh)", "zh"),
    ("🇯🇵 Japanese (ja)", "ja"),
    ("🇰🇷 Korean (ko)", "ko"),
    ("🇵🇹 Portuguese (pt)", "pt"),
    ("🇮🇹 Italian (it)", "it"),
    ("🇷🇺 Russian (ru)", "ru"),
    ("🇸🇦 Arabic (ar)", "ar"),
    ("🇹🇷 Turkish (tr)", "tr"),
    ("🇳🇱 Dutch (nl)", "nl"),
]


def apply_speed_preset(preset_name: str):
    """Update advanced settings dynamically based on chosen preset."""
    if preset_name == "🚀 Turbo (Ultra-Fast)":
        return "large-v3-turbo", 1, True, True
    elif preset_name == "🎯 Deep Focus (Max Accuracy)":
        return "large-v3", 3, True, False
    elif preset_name == "⚡ Instant Draft (Fastest)":
        return "base", 1, True, True
    return "large-v3-turbo", 1, True, True


def run_transcription_stream(
    source_type: str,
    file_path: str,
    mic_path: str,
    url_str: str,
    output_mode: str,
    model_name: str,
    device: str,
    lang_code: str,
    beam_size: int,
    use_vad: bool,
    use_batching: bool,
    auto_llm: bool,
    llm_model: str,
):
    """
    Generator that streams transcription progress and segments in real time.
    Translates Hindi and multilingual speech to 100% natural English.
    """
    audio_file = None
    media_name = "Audio Recording"

    # Determine translation mode
    is_translate_mode = ("Always Translate to English" in output_mode)
    whisper_task = "translate" if is_translate_mode else "transcribe"

    # Resolve input source
    if source_type == "Upload Audio/Video":
        if not file_path:
            yield (
                None,
                "<div class='status-pill error'>⚠️ Please upload an audio or video file first.</div>",
                "",
                "",
                "",
                "",
                None, None, None, None
            )
            return
        audio_file = file_path
        media_name = os.path.splitext(os.path.basename(file_path))[0]

    elif source_type == "Record Microphone":
        if not mic_path:
            yield (
                None,
                "<div class='status-pill error'>⚠️ No microphone audio found. Record voice first.</div>",
                "",
                "",
                "",
                "",
                None, None, None, None
            )
            return
        audio_file = mic_path
        media_name = "Microphone_Recording"

    elif source_type == "Online URL / YouTube":
        if not url_str or not url_str.strip():
            yield (
                None,
                "<div class='status-pill error'>⚠️ Please enter a valid URL.</div>",
                "",
                "",
                "",
                "",
                None, None, None, None
            )
            return
        yield (
            None,
            "<div class='status-pill pulse'>⏳ Extracting audio stream from URL via yt-dlp...</div>",
            "",
            "",
            "",
            "",
            None, None, None, None
        )
        try:
            audio_file, title = download_audio_from_url(url_str)
            media_name = title[:30]
        except Exception as e:
            yield (
                None,
                f"<div class='status-pill error'>❌ Download failed: {str(e)}</div>",
                "",
                "",
                "",
                "",
                None, None, None, None
            )
            return

    # Initial UI state with audio loaded
    task_desc = "Transcribing & Translating speech directly to English" if is_translate_mode else "Transcribing audio verbatim"
    yield (
        audio_file,
        f"<div class='status-pill pulse'>⚡ {task_desc} with GPU acceleration...</div>",
        "",
        "",
        "",
        "",
        None, None, None, None
    )

    raw_text = ""
    raw_segments = []
    final_payload = None

    # Stream segments as they arrive from faster-whisper
    for event in manager.transcribe_stream(
        audio_path=audio_file,
        model_name=model_name,
        device=device,
        language=lang_code,
        task=whisper_task,
        beam_size=int(beam_size),
        use_vad=use_vad,
        use_batching=use_batching,
    ):
        if event["type"] == "info":
            detected_lang = event["language"].upper()
            prob = event["language_probability"] * 100
            if is_translate_mode and detected_lang != "EN":
                lang_display = f"🌐 Spoken: <b>{detected_lang}</b> ({prob:.1f}%) ➔ Target: <b>ENGLISH (AI Agents Ready)</b>"
            else:
                lang_display = f"🌐 Spoken: <b>{detected_lang}</b> ({prob:.1f}%)"

            status_html = (
                f"<div class='status-pill info'>"
                f"{lang_display} | ⏱️ Duration: <b>{event['duration']}s</b>"
                f"</div>"
            )
            yield (
                audio_file,
                status_html,
                "",
                "",
                "",
                "",
                None, None, None, None
            )

        elif event["type"] == "segment":
            raw_text = event["full_text"]
            raw_segments = event["segments"]

            # Render live stream preview cards
            cards_html = "".join([
                f"<div class='segment-card'>"
                f"<span class='seg-time'>[{s['start_formatted']} → {s['end_formatted']}]</span> "
                f"<span class='seg-text'>{s['text']}</span>"
                f"</div>"
                for s in raw_segments[-12:]
            ])

            yield (
                audio_file,
                "<div class='status-pill pulse'>⚡ Decoding speech stream...</div>",
                raw_text,
                cards_html,
                raw_text,
                "",
                None, None, None, None
            )

        elif event["type"] == "done":
            final_payload = event
            raw_text = event["full_text"]
            raw_segments = event["segments"]

    if final_payload is None:
        return

    detected_lang = final_payload["language"].upper()
    prob = final_payload["language_probability"] * 100
    elapsed_time = final_payload["elapsed"]
    speed_ratio = final_payload.get("speed_ratio", 1.0)

    # Check if translation to English is required
    has_non_eng = contains_non_english(raw_text)
    needs_english_conversion = is_translate_mode and (has_non_eng or detected_lang != "EN")

    final_english_text = raw_text
    final_english_segments = raw_segments

    if needs_english_conversion:
        yield (
            audio_file,
            f"<div class='status-pill pulse'>🌍 Detected <b>{detected_lang}</b> ({prob:.1f}%). Translating completely into fluent English via {llm_model}...</div>",
            "Translating into English...",
            "".join([
                f"<div class='segment-card'><span class='seg-time'>[{s['start_formatted']} → {s['end_formatted']}]</span> <span class='seg-text'>{s['text']}</span></div>"
                for s in raw_segments
            ]),
            raw_text,
            "",
            None, None, None, None
        )

        # 1. Translate the continuous full text to English
        final_english_text = translate_to_english(raw_text, model_name=llm_model)

        # 2. Translate individual segments so .srt/.vtt and cards are in English
        final_english_segments = translate_segments_to_english(raw_segments, model_name=llm_model)

    # Render complete interactive segment cards in English
    all_cards_html = "".join([
        f"<div class='segment-card'>"
        f"<div class='seg-header'><span class='seg-time'>[{s['start_formatted']} → {s['end_formatted']}]</span>"
        f"<span class='seg-badge'>Segment #{s['id']}</span></div>"
        f"<div class='seg-text'>{s['text']}</div>"
        f"</div>"
        for s in final_english_segments
    ])

    if needs_english_conversion:
        lang_summary = f"Spoken: <b>{detected_lang}</b> ({prob:.1f}%) ➔ Converted to <b style='color:#38bdf8;'>100% ENGLISH</b>"
    else:
        lang_summary = f"Language: <b>{detected_lang}</b> ({prob:.1f}%)"

    final_status = (
        f"<div class='status-pill success'>"
        f"✅ <b>Done!</b> {lang_summary} "
        f"| Audio: <b>{final_payload['duration']}s</b> | Processed in <b>{elapsed_time}s</b> "
        f"(<span class='highlight-speed'>{speed_ratio}x faster than real-time</span>)"
        f"</div>"
    )

    # Optional additional AI formatting polish
    refined_output = ""
    if auto_llm and final_english_text:
        yield (
            audio_file,
            f"{final_status} <div class='status-pill pulse'>✨ Polishing paragraphs with {llm_model}...</div>",
            final_english_text,
            all_cards_html,
            raw_text,
            "Polishing paragraphs...",
            None, None, None, None
        )
        refined_output = refine_transcript_with_llm(final_english_text, model_name=llm_model)
    else:
        refined_output = final_english_text

    # Generate files for download (All in English)
    timestamp = int(time.time())
    safe_title = "".join(c for c in media_name if c.isalnum() or c in ("-", "_")).rstrip()[:25] or "transcript"

    txt_file = os.path.join(TEMP_EXPORT_DIR, f"{safe_title}_{timestamp}_english.txt")
    srt_file = os.path.join(TEMP_EXPORT_DIR, f"{safe_title}_{timestamp}_english.srt")
    vtt_file = os.path.join(TEMP_EXPORT_DIR, f"{safe_title}_{timestamp}_english.vtt")
    json_file = os.path.join(TEMP_EXPORT_DIR, f"{safe_title}_{timestamp}_english.json")

    with open(txt_file, "w", encoding="utf-8") as f:
        f.write(final_english_text)

    with open(srt_file, "w", encoding="utf-8") as f:
        f.write(manager.generate_srt(final_english_segments))

    with open(vtt_file, "w", encoding="utf-8") as f:
        f.write(manager.generate_vtt(final_english_segments))

    # Save metadata including both English text and raw source
    export_metadata = dict(final_payload)
    export_metadata["english_text"] = final_english_text
    export_metadata["english_segments"] = final_english_segments
    export_metadata["original_spoken_text"] = raw_text

    with open(json_file, "w", encoding="utf-8") as f:
        f.write(manager.generate_json(export_metadata))

    yield (
        audio_file,
        final_status,
        final_english_text,      # Primary Continuous English Transcript
        all_cards_html,          # English Subtitle Cards
        raw_text,                # Original Spoken Transcript (Hindi/Source)
        refined_output,          # AI Polished Paragraphs
        txt_file,
        srt_file,
        vtt_file,
        json_file
    )


def refine_manual(current_text: str, model_name: str):
    """Manually trigger AI polish on existing transcript."""
    if not current_text or not current_text.strip():
        return "Please transcribe an audio file first."
    return refine_transcript_with_llm(current_text, model_name=model_name)


# Premium Cyber-Studio Obsidian Dark Theme CSS
CUSTOM_CSS = """
:root {
    --bg-primary: #07090e;
    --bg-surface: #0e1420;
    --bg-card: #131b2e;
    --border-color: rgba(99, 102, 241, 0.18);
    --border-hover: rgba(99, 102, 241, 0.45);
    --accent-indigo: #6366f1;
    --accent-violet: #8b5cf6;
    --accent-emerald: #10b981;
    --text-primary: #f8fafc;
    --text-secondary: #94a3b8;
    --text-muted: #64748b;
}

body, .gradio-container {
    background-color: var(--bg-primary) !important;
    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, "Inter", sans-serif !important;
    color: var(--text-primary) !important;
}

/* Header bar styling */
.header-container {
    background: linear-gradient(180deg, #101728 0%, #090d16 100%);
    border: 1px solid var(--border-color);
    border-radius: 14px;
    padding: 22px 28px;
    margin-bottom: 24px;
    box-shadow: 0 8px 32px rgba(0, 0, 0, 0.4);
}

.header-title {
    font-size: 2.1rem;
    font-weight: 800;
    letter-spacing: -0.03em;
    background: linear-gradient(135deg, #ffffff 30%, #a5b4fc 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    margin: 0;
}

.header-subtitle {
    color: var(--text-secondary);
    font-size: 0.95rem;
    margin-top: 6px;
}

.badge-row {
    display: flex;
    gap: 10px;
    flex-wrap: wrap;
    margin-top: 14px;
}

.badge {
    display: inline-flex;
    align-items: center;
    gap: 6px;
    padding: 5px 12px;
    font-size: 0.8rem;
    font-weight: 600;
    border-radius: 20px;
    border: 1px solid rgba(255, 255, 255, 0.08);
}

.badge-gpu { background: rgba(16, 185, 129, 0.12); color: #34d399; border-color: rgba(16, 185, 129, 0.25); }
.badge-engine { background: rgba(99, 102, 241, 0.12); color: #a5b4fc; border-color: rgba(99, 102, 241, 0.25); }
.badge-ai { background: rgba(139, 92, 246, 0.12); color: #c4b5fd; border-color: rgba(139, 92, 246, 0.25); }
.badge-lang { background: rgba(56, 189, 248, 0.12); color: #38bdf8; border-color: rgba(56, 189, 248, 0.25); }

/* Status pill */
.status-pill {
    padding: 12px 18px;
    border-radius: 10px;
    font-size: 0.92rem;
    margin-bottom: 14px;
    border: 1px solid transparent;
}
.status-pill.success { background: rgba(16, 185, 129, 0.12); color: #34d399; border-color: rgba(16, 185, 129, 0.3); }
.status-pill.info { background: rgba(99, 102, 241, 0.12); color: #c7d2fe; border-color: rgba(99, 102, 241, 0.3); }
.status-pill.pulse { background: rgba(245, 158, 11, 0.12); color: #fcd34d; border-color: rgba(245, 158, 11, 0.3); animation: pulse 1.5s infinite; }
.status-pill.error { background: rgba(239, 68, 68, 0.12); color: #fca5a5; border-color: rgba(239, 68, 68, 0.3); }

.highlight-speed {
    color: #38bdf8;
    font-weight: 700;
}

/* Transcribe button */
.primary-transcribe-btn {
    background: linear-gradient(135deg, #4f46e5 0%, #7c3aed 100%) !important;
    border: none !important;
    color: #ffffff !important;
    font-weight: 700 !important;
    font-size: 1.05rem !important;
    padding: 14px 24px !important;
    border-radius: 10px !important;
    box-shadow: 0 4px 20px rgba(99, 102, 241, 0.4) !important;
    transition: all 0.2s ease-in-out !important;
}

.primary-transcribe-btn:hover {
    transform: translateY(-2px) !important;
    box-shadow: 0 6px 28px rgba(99, 102, 241, 0.6) !important;
}

/* Segment cards stream */
.segment-stream-box {
    max-height: 480px;
    overflow-y: auto;
    padding-right: 8px;
}

.segment-card {
    background: var(--bg-card);
    border: 1px solid var(--border-color);
    border-radius: 10px;
    padding: 12px 16px;
    margin-bottom: 10px;
    transition: border-color 0.15s ease;
}

.segment-card:hover {
    border-color: var(--border-hover);
}

.seg-header {
    display: flex;
    justify-content: space-between;
    align-items: center;
    margin-bottom: 6px;
}

.seg-time {
    font-family: 'JetBrains Mono', 'Fira Code', monospace;
    font-size: 0.8rem;
    color: #38bdf8;
    background: rgba(56, 189, 248, 0.1);
    padding: 2px 8px;
    border-radius: 4px;
}

.seg-badge {
    font-size: 0.72rem;
    color: var(--text-muted);
}

.seg-text {
    font-size: 0.95rem;
    line-height: 1.55;
    color: var(--text-primary);
}

@keyframes pulse {
    0% { opacity: 0.85; }
    50% { opacity: 1; }
    100% { opacity: 0.85; }
}
"""

with gr.Blocks(title="SonicScribe Studio") as demo:
    # Custom Header Banner
    gr.HTML(
        f"""
        <div class="header-container">
            <h1 class="header-title">🎙️ SonicScribe Studio</h1>
            <div class="header-subtitle">100% Local Audio-to-English Converter for AI Agents & Autonomous Pipelines</div>
            <div class="badge-row">
                <span class="badge badge-gpu">🟢 NVIDIA RTX 3050 CUDA (int8_float16)</span>
                <span class="badge badge-engine">⚡ faster-whisper (Batched Pipeline)</span>
                <span class="badge badge-lang">🌍 Auto-Translate to English (Active)</span>
                <span class="badge badge-ai">✨ Llama 3.2 Translator ({default_ollama})</span>
            </div>
        </div>
        """
    )

    with gr.Row():
        # LEFT COLUMN: Audio Inputs & Configuration
        with gr.Column(scale=5):
            gr.Markdown("### 1. Audio Source")
            source_tabs = gr.Radio(
                choices=["Upload Audio/Video", "Record Microphone", "Online URL / YouTube"],
                value="Upload Audio/Video",
                label="Select Input Method",
                interactive=True,
            )

            with gr.Group(visible=True) as upload_box:
                file_input = gr.File(
                    label="Drop any audio/video file (MP3, WAV, M4A, FLAC, MP4, MKV, WebM)",
                    type="filepath",
                )

            with gr.Group(visible=False) as mic_box:
                mic_input = gr.Audio(
                    sources=["microphone"],
                    type="filepath",
                    label="Speak into your Microphone",
                )

            with gr.Group(visible=False) as url_box:
                url_input = gr.Textbox(
                    label="Video / Audio Link",
                    placeholder="https://www.youtube.com/watch?v=... or direct media link",
                )

            # Toggle input views
            source_tabs.change(
                fn=lambda s: (
                    gr.update(visible=(s == "Upload Audio/Video")),
                    gr.update(visible=(s == "Record Microphone")),
                    gr.update(visible=(s == "Online URL / YouTube")),
                ),
                inputs=[source_tabs],
                outputs=[upload_box, mic_box, url_box],
            )

            gr.Markdown("### 2. Output Language Translation")
            output_lang_mode = gr.Radio(
                choices=[
                    "🇬🇧 Always Translate to English (Recommended for AI Agents)",
                    "🌐 Original Spoken Language (Verbatim)",
                ],
                value="🇬🇧 Always Translate to English (Recommended for AI Agents)",
                label="Target Output Language",
                info="Hindi and other languages will be automatically converted to 100% fluent English for downstream AI agents.",
            )

            gr.Markdown("### 3. Speed & Accuracy Profiles")
            speed_preset = gr.Radio(
                choices=[
                    "🚀 Turbo (Ultra-Fast)",
                    "🎯 Deep Focus (Max Accuracy)",
                    "⚡ Instant Draft (Fastest)",
                ],
                value="🚀 Turbo (Ultra-Fast)",
                label="Inference Profile",
                info="Turbo uses large-v3-turbo with batched GPU acceleration. Sub-second latency & zero loss in accuracy.",
            )

            with gr.Accordion("⚙️ Advanced Tuning & Models", open=False):
                with gr.Row():
                    model_selector = gr.Dropdown(
                        choices=AVAILABLE_MODELS,
                        value="large-v3-turbo",
                        label="Whisper Model",
                    )
                    device_selector = gr.Dropdown(
                        choices=["cuda", "cpu"],
                        value="cuda",
                        label="Compute Device",
                    )
                with gr.Row():
                    lang_selector = gr.Dropdown(
                        choices=[(name, code) for name, code in POPULAR_LANGUAGES],
                        value="Auto-Detect",
                        label="Audio Spoken Language",
                    )
                    beam_slider = gr.Slider(
                        minimum=1,
                        maximum=5,
                        value=1,
                        step=1,
                        label="Beam Size",
                        info="Beam 1 (greedy) is 4x faster on CUDA with identical WER.",
                    )
                with gr.Row():
                    vad_toggle = gr.Checkbox(
                        value=True,
                        label="Silero VAD Silence Removal",
                    )
                    batch_toggle = gr.Checkbox(
                        value=True,
                        label="Batched CUDA Pipeline (Multi-chunk)",
                    )

            # Sync preset with controls
            speed_preset.change(
                fn=apply_speed_preset,
                inputs=[speed_preset],
                outputs=[model_selector, beam_slider, vad_toggle, batch_toggle],
            )

            with gr.Accordion("✨ AI Smart Polish & Translator Model (Ollama)", open=False):
                gr.Markdown("Select local LLM model used for multilingual translation and paragraph structuring.")
                with gr.Row():
                    auto_llm_toggle = gr.Checkbox(
                        value=False,
                        label="Format into clean paragraphs after translation",
                    )
                    llm_select = gr.Dropdown(
                        choices=ollama_models if ollama_models else ["None detected"],
                        value=default_ollama,
                        label="Ollama Translation Model",
                        info="Llama 3.2 is recommended for highest translation accuracy.",
                    )

            start_btn = gr.Button("⚡ Start Transcription & Translation", elem_classes=["primary-transcribe-btn"], size="lg")

        # RIGHT COLUMN: Live Audio & Streaming Results
        with gr.Column(scale=6):
            status_display = gr.HTML(
                "<div class='status-pill info'>💡 Select or record an audio file and click <b>Start Transcription & Translation</b>.</div>"
            )

            audio_preview = gr.Audio(label="Audio Waveform & Playback", interactive=False)

            with gr.Tabs() as result_tabs:
                with gr.Tab("🇬🇧 Full English Text (AI Agents Ready)", id="tab_eng"):
                    full_text_box = gr.Textbox(
                        label="100% English Continuous Output",
                        lines=12,
                        interactive=True,
                    )

                with gr.Tab("⏱️ English Subtitles & Segments", id="tab_segs"):
                    segments_html = gr.HTML(
                        "<div class='segment-stream-box'><p style='color:#64748b;'>English segments will stream here in real time...</p></div>"
                    )

                with gr.Tab("🌐 Original Spoken Transcript", id="tab_raw"):
                    original_text_box = gr.Textbox(
                        label="Raw Spoken Text (Hindi / Source Language)",
                        lines=12,
                        interactive=True,
                    )

                with gr.Tab("✨ AI Polished Paragraphs", id="tab_polish"):
                    with gr.Row():
                        ai_model_manual = gr.Dropdown(
                            choices=ollama_models if ollama_models else ["None detected"],
                            value=default_ollama,
                            label="Target LLM",
                            scale=3,
                        )
                        polish_btn = gr.Button("✨ Polish English Text", scale=1)
                    refined_text_box = gr.Textbox(
                        label="Publication-Ready Clean English Text",
                        lines=10,
                        interactive=True,
                    )
                    polish_btn.click(
                        fn=refine_manual,
                        inputs=[full_text_box, ai_model_manual],
                        outputs=[refined_text_box],
                    )

            gr.Markdown("### 💾 One-Click Export (100% English)")
            with gr.Row():
                txt_down = gr.File(label="📄 English TXT")
                srt_down = gr.File(label="🎬 English Subtitles (.srt)")
                vtt_down = gr.File(label="🌐 WebVTT (.vtt)")
                json_down = gr.File(label="📊 JSON Data")

    # Connect execution stream
    start_btn.click(
        fn=run_transcription_stream,
        inputs=[
            source_tabs,
            file_input,
            mic_input,
            url_input,
            output_lang_mode,
            model_selector,
            device_selector,
            lang_selector,
            beam_slider,
            vad_toggle,
            batch_toggle,
            auto_llm_toggle,
            llm_select,
        ],
        outputs=[
            audio_preview,
            status_display,
            full_text_box,       # Full English text
            segments_html,       # English segment cards
            original_text_box,   # Original verbatim text
            refined_text_box,    # Polished text
            txt_down,
            srt_down,
            vtt_down,
            json_down,
        ],
    )

if __name__ == "__main__":
    demo.launch(
        server_name="127.0.0.1",
        server_port=7860,
        inbrowser=True,
        css=CUSTOM_CSS,
    )
