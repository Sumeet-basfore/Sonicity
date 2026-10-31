"""
URL Audio Downloader using yt-dlp.
Extracts clean audio streams from YouTube and hundreds of supported online media platforms.
"""

import os
import tempfile
import logging
from typing import Tuple
import yt_dlp

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

TEMP_DOWNLOAD_DIR = os.path.join(tempfile.gettempdir(), "audio_to_text_downloads")
os.makedirs(TEMP_DOWNLOAD_DIR, exist_ok=True)

def download_audio_from_url(url: str) -> Tuple[str, str]:
    """
    Downloads audio from given URL via yt-dlp.
    Returns:
        (filepath, title)
    """
    url = url.strip()
    if not url:
        raise ValueError("URL cannot be empty.")

    out_template = os.path.join(TEMP_DOWNLOAD_DIR, "%(id)s.%(ext)s")

    ydl_opts = {
        "format": "bestaudio/best",
        "outtmpl": out_template,
        "postprocessors": [
            {
                "key": "FFmpegExtractAudio",
                "preferredcodec": "mp3",
                "preferredquality": "192",
            }
        ],
        "quiet": True,
        "no_warnings": True,
    }

    logger.info(f"Downloading audio from URL: {url}")
    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        info = ydl.extract_info(url, download=True)
        title = info.get("title", "Online Media Audio")
        # Output filepath will have .mp3 extension due to FFmpegExtractAudio
        final_filename = ydl.prepare_filename(info)
        base, _ = os.path.splitext(final_filename)
        mp3_path = f"{base}.mp3"

        if os.path.exists(mp3_path):
            return mp3_path, title
        elif os.path.exists(final_filename):
            return final_filename, title
        else:
            raise FileNotFoundError(f"Extracted audio file not found for URL: {url}")
