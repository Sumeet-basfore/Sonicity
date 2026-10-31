"""
SonicScribe Studio - AI Vocal & Instrumental Stem Separator (Demucs htdemucs).
Extracts isolated acapella vocal tracks and vocal-free karaoke backing tracks.
"""

import os
import sys
import time
import logging
import subprocess
from pathlib import Path
from typing import Dict, Any, Optional

logger = logging.getLogger(__name__)

BASE_DIR = Path(__file__).resolve().parent
STEMS_DIR = BASE_DIR / "uploads" / "stems"
STEMS_DIR.mkdir(parents=True, exist_ok=True)


def get_stem_slug(filepath: str) -> str:
    """Generate a clean slug for caching separated stems."""
    path = Path(filepath)
    stem_name = path.stem.replace(" ", "_")
    return stem_name


def separate_audio_stems(audio_filepath: str, device: str = "auto") -> Dict[str, Any]:
    """
    Separate an audio track into isolated vocals and instrumental backing track using Demucs htdemucs.
    Checks cache first to avoid re-separating already processed tracks.
    """
    path = Path(audio_filepath)
    if not path.exists():
        raise FileNotFoundError(f"Audio file not found: {audio_filepath}")

    track_slug = get_stem_slug(audio_filepath)
    track_dir = STEMS_DIR / "htdemucs" / path.stem

    vocals_file = track_dir / "vocals.wav"
    instrumental_file = track_dir / "no_vocals.wav"

    # Check cache
    if vocals_file.exists() and instrumental_file.exists():
        logger.info(f"✓ Found cached stems for {path.name}")
        return {
            "status": "success",
            "cached": True,
            "track_name": path.stem,
            "vocals_path": str(vocals_file),
            "instrumental_path": str(instrumental_file),
            "vocals_url": f"/api/stems/{path.stem}/vocals",
            "instrumental_url": f"/api/stems/{path.stem}/instrumental",
            "vocals_size_mb": round(vocals_file.stat().st_size / (1024 * 1024), 2),
            "instrumental_size_mb": round(instrumental_file.stat().st_size / (1024 * 1024), 2),
        }

    # Detect CUDA or CPU
    if device == "auto":
        try:
            import torch
            device = "cuda" if torch.cuda.is_available() else "cpu"
        except Exception:
            device = "cuda"

    logger.info(f"⚡ Launching Demucs stem separation on {device} for {path.name}...")
    start_time = time.time()

    cmd = [
        sys.executable,
        "-m", "demucs.separate",
        "-n", "htdemucs",
        "--two-stems", "vocals",
        "-o", str(STEMS_DIR),
        "-d", device,
        str(path)
    ]

    res = subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        timeout=300.0  # 5 min max
    )

    elapsed = round(time.time() - start_time, 2)

    if res.returncode != 0:
        logger.error(f"Demucs separation failed ({res.returncode}): {res.stderr}")
        raise RuntimeError(f"Stem separation failed: {res.stderr[:300]}")

    if not vocals_file.exists() or not instrumental_file.exists():
        # Search for any produced stems in the output directory
        found_stems = list(STEMS_DIR.glob(f"**/{path.stem}/*.wav"))
        for s in found_stems:
            if s.name == "vocals.wav":
                vocals_file = s
            elif s.name in ("no_vocals.wav", "instrumental.wav"):
                instrumental_file = s

    if not vocals_file.exists() or not instrumental_file.exists():
        raise RuntimeError(f"Demucs executed but output stems were not found at {track_dir}")

    logger.info(f"✅ Stems successfully isolated in {elapsed}s on {device}!")

    return {
        "status": "success",
        "cached": False,
        "elapsed_sec": elapsed,
        "device": device,
        "track_name": path.stem,
        "vocals_path": str(vocals_file),
        "instrumental_path": str(instrumental_file),
        "vocals_url": f"/api/stems/{path.stem}/vocals",
        "instrumental_url": f"/api/stems/{path.stem}/instrumental",
        "vocals_size_mb": round(vocals_file.stat().st_size / (1024 * 1024), 2),
        "instrumental_size_mb": round(instrumental_file.stat().st_size / (1024 * 1024), 2),
    }
