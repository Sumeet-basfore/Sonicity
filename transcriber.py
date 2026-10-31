"""
High-Performance Transcription Engine powered by faster-whisper.
Optimized for NVIDIA RTX 3050 with Batched Inference & Real-time Streaming.
"""

import os
import sys
import glob

# Auto-configure NVIDIA CUDA & cuDNN shared libraries
_nvidia_libs = glob.glob(os.path.join(sys.prefix, "lib", "python*", "site-packages", "nvidia", "*", "lib"))
if _nvidia_libs:
    _existing_ld = os.environ.get("LD_LIBRARY_PATH", "")
    _new_paths = [p for p in _nvidia_libs if p not in _existing_ld]
    if _new_paths:
        os.environ["LD_LIBRARY_PATH"] = ":".join(_nvidia_libs) + ((":" + _existing_ld) if _existing_ld else "")
        try:
            import ctypes
            for p in _nvidia_libs:
                for so in glob.glob(os.path.join(p, "*.so*")):
                    try:
                        ctypes.CDLL(so)
                    except Exception:
                        pass
        except Exception:
            pass

import time
import json
import logging
from typing import Optional, Dict, Any, List, Generator
from faster_whisper import WhisperModel, BatchedInferencePipeline

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

AVAILABLE_MODELS = [
    "large-v3-turbo",
    "large-v3",
    "medium",
    "small",
    "base",
    "tiny",
]

def format_timestamp(seconds: float, srt_format: bool = False) -> str:
    """Format seconds into standard timecode format."""
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    millis = int((seconds - int(seconds)) * 1000)
    if srt_format:
        return f"{hours:02d}:{minutes:02d}:{secs:02d},{millis:03d}"
    return f"{hours:02d}:{minutes:02d}:{secs:02d}.{millis:03d}"


class WhisperManager:
    def __init__(self):
        self.current_model_name: Optional[str] = None
        self.current_device: Optional[str] = None
        self.current_compute_type: Optional[str] = None
        self.model: Optional[WhisperModel] = None
        self.batched_model: Optional[BatchedInferencePipeline] = None

    def get_model(
        self,
        model_name: str = "large-v3-turbo",
        device: str = "cuda",
        compute_type: Optional[str] = None,
    ) -> WhisperModel:
        """Load or retrieve cached WhisperModel instance with optimal compute type."""
        if device == "cuda":
            # int8_float16 provides 2x lower VRAM & memory bandwidth savings on RTX 3050 Laptop
            compute_type = compute_type or "int8_float16"
        else:
            compute_type = compute_type or "int8"

        if (
            self.model is None
            or self.current_model_name != model_name
            or self.current_device != device
            or self.current_compute_type != compute_type
        ):
            logger.info(f"Loading faster-whisper '{model_name}' on {device} ({compute_type})...")
            try:
                self.model = WhisperModel(model_name, device=device, compute_type=compute_type)
                self.batched_model = BatchedInferencePipeline(model=self.model)
            except Exception as e:
                logger.warning(f"CUDA load failed ({e}), falling back to CPU int8...")
                device = "cpu"
                compute_type = "int8"
                self.model = WhisperModel(model_name, device=device, compute_type=compute_type)
                self.batched_model = BatchedInferencePipeline(model=self.model)

            self.current_model_name = model_name
            self.current_device = device
            self.current_compute_type = compute_type

        return self.model

    def transcribe_stream(
        self,
        audio_path: str,
        model_name: str = "large-v3-turbo",
        device: str = "cuda",
        language: Optional[str] = None,
        task: str = "translate",
        beam_size: int = 1,
        use_vad: bool = True,
        use_batching: bool = True,
    ) -> Generator[Dict[str, Any], None, None]:
        """
        Yield real-time streaming segments as audio is decoded.
        task: "translate" (translates any spoken language directly to English)
              or "transcribe" (keeps original spoken language)
        """
        if not os.path.exists(audio_path):
            raise FileNotFoundError(f"Audio file not found: {audio_path}")

        model = self.get_model(model_name=model_name, device=device)
        vad_params = dict(min_silence_duration_ms=400) if use_vad else None
        target_lang = None if language in [None, "auto", "Auto-Detect", ""] else language

        start_time = time.time()
        logger.info(f"Stream transcription started: {audio_path} (model={model_name}, device={device}, task={task}, beam={beam_size})")

        # Try execution, fallback to CPU if CUDA encounters memory or library error
        try:
            if use_batching and self.batched_model is not None and device == "cuda":
                # Batched pipeline for massive speedup
                segments_iter, info = self.batched_model.transcribe(
                    audio_path,
                    batch_size=8,
                    language=target_lang,
                    task=task,
                    vad_filter=use_vad,
                    vad_parameters=vad_params,
                )
            else:
                segments_iter, info = model.transcribe(
                    audio_path,
                    beam_size=beam_size,
                    language=target_lang,
                    task=task,
                    vad_filter=use_vad,
                    vad_parameters=vad_params,
                    temperature=0.0,
                )

            yield {
                "type": "info",
                "language": info.language,
                "language_probability": round(info.language_probability, 4),
                "duration": round(info.duration, 2),
                "task": task,
            }

            segments = []
            text_pieces = []

            for seg in segments_iter:
                txt = seg.text.strip()
                if not txt:
                    continue
                seg_dict = {
                    "id": seg.id,
                    "start": round(seg.start, 2),
                    "end": round(seg.end, 2),
                    "text": txt,
                    "start_formatted": format_timestamp(seg.start),
                    "end_formatted": format_timestamp(seg.end),
                }
                segments.append(seg_dict)
                text_pieces.append(txt)
                
                # Stream this segment immediately to the UI
                yield {
                    "type": "segment",
                    "segment": seg_dict,
                    "full_text": " ".join(text_pieces),
                    "segments": segments,
                }

            elapsed = round(time.time() - start_time, 2)
            speed_ratio = round(info.duration / elapsed, 1) if elapsed > 0 else 1.0

            yield {
                "type": "done",
                "full_text": " ".join(text_pieces),
                "segments": segments,
                "language": info.language,
                "language_probability": round(info.language_probability, 4),
                "duration": round(info.duration, 2),
                "elapsed": elapsed,
                "speed_ratio": speed_ratio,
                "task": task,
            }

        except RuntimeError as e:
            if "cuda" in str(e).lower() or "libcublas" in str(e) or "out of memory" in str(e).lower():
                logger.warning(f"CUDA error: {e}. Retrying seamlessly on CPU...")
                cpu_model = self.get_model(model_name=model_name, device="cpu", compute_type="int8")
                segments_iter, info = cpu_model.transcribe(
                    audio_path,
                    beam_size=1,
                    language=target_lang,
                    task=task,
                    vad_filter=use_vad,
                    vad_parameters=vad_params,
                )
                yield {
                    "type": "info",
                    "language": info.language,
                    "language_probability": round(info.language_probability, 4),
                    "duration": round(info.duration, 2),
                }
                segments = []
                text_pieces = []
                for seg in segments_iter:
                    txt = seg.text.strip()
                    if not txt:
                        continue
                    seg_dict = {
                        "id": seg.id,
                        "start": round(seg.start, 2),
                        "end": round(seg.end, 2),
                        "text": txt,
                        "start_formatted": format_timestamp(seg.start),
                        "end_formatted": format_timestamp(seg.end),
                    }
                    segments.append(seg_dict)
                    text_pieces.append(txt)
                    yield {
                        "type": "segment",
                        "segment": seg_dict,
                        "full_text": " ".join(text_pieces),
                        "segments": segments,
                    }
                elapsed = round(time.time() - start_time, 2)
                yield {
                    "type": "done",
                    "full_text": " ".join(text_pieces),
                    "segments": segments,
                    "language": info.language,
                    "language_probability": round(info.language_probability, 4),
                    "duration": round(info.duration, 2),
                    "elapsed": elapsed,
                    "speed_ratio": round(info.duration / elapsed, 1) if elapsed > 0 else 1.0,
                }
            else:
                raise

    @staticmethod
    def generate_srt(segments: List[Dict[str, Any]]) -> str:
        """Generate SubRip (.srt) subtitle text."""
        srt_lines = []
        for i, seg in enumerate(segments, start=1):
            start = format_timestamp(seg["start"], srt_format=True)
            end = format_timestamp(seg["end"], srt_format=True)
            srt_lines.append(f"{i}\n{start} --> {end}\n{seg['text']}\n")
        return "\n".join(srt_lines)

    @staticmethod
    def generate_vtt(segments: List[Dict[str, Any]]) -> str:
        """Generate WebVTT (.vtt) subtitle text."""
        vtt_lines = ["WEBVTT\n"]
        for seg in segments:
            start = format_timestamp(seg["start"], srt_format=False)
            end = format_timestamp(seg["end"], srt_format=False)
            vtt_lines.append(f"{start} --> {end}\n{seg['text']}\n")
        return "\n".join(vtt_lines)

    @staticmethod
    def generate_json(data: Dict[str, Any]) -> str:
        """Generate pretty JSON representation."""
        return json.dumps(data, indent=2, ensure_ascii=False)
