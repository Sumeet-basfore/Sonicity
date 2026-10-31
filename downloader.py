"""
URL Audio Downloader using yt-dlp.
Extracts clean audio streams from YouTube and hundreds of supported online media platforms.
"""

import os
import tempfile
import logging
from typing import Tuple, Optional
import yt_dlp

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

TEMP_DOWNLOAD_DIR = os.path.join(tempfile.gettempdir(), "audio_to_text_downloads")
os.makedirs(TEMP_DOWNLOAD_DIR, exist_ok=True)


class DownloadResult(tuple):
    """
    Dual-compatibility return type:
    - Supports tuple unpacking: `filepath, title = download_audio_from_url(...)`
    - Supports dict subscripting: `res["file_path"]`, `res["title"]`
    - Supports property access: `res.file_path`, `res.title`
    """
    def __new__(cls, filepath: str, title: str):
        return super().__new__(cls, (filepath, title))

    @property
    def file_path(self) -> str:
        return self[0]

    @property
    def filepath(self) -> str:
        return self[0]

    @property
    def title(self) -> str:
        return self[1]

    def __getitem__(self, item):
        if isinstance(item, str):
            if item in ("file_path", "filepath"):
                return self[0]
            elif item == "title":
                return self[1]
            raise KeyError(item)
        return super().__getitem__(item)

    def get(self, key, default=None):
        if key in ("file_path", "filepath"):
            return self[0]
        elif key == "title":
            return self[1]
        return default


def download_audio_from_url(url: str, output_dir: Optional[str] = None) -> DownloadResult:
    """
    Downloads audio from given URL via yt-dlp.
    Args:
        url: Video or audio stream URL.
        output_dir: Optional output directory to place the extracted file.
    Returns:
        DownloadResult(filepath, title)
    """
    url = url.strip()
    if not url:
        raise ValueError("URL cannot be empty.")

    target_dir = output_dir if output_dir else TEMP_DOWNLOAD_DIR
    os.makedirs(target_dir, exist_ok=True)

    out_template = os.path.join(target_dir, "%(id)s.%(ext)s")

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

    logger.info(f"Downloading audio from URL: {url} into {target_dir}")
    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        info = ydl.extract_info(url, download=True)
        title = info.get("title", "Online Media Audio")
        # Output filepath will have .mp3 extension due to FFmpegExtractAudio
        final_filename = ydl.prepare_filename(info)
        base, _ = os.path.splitext(final_filename)
        mp3_path = f"{base}.mp3"

        if os.path.exists(mp3_path):
            return DownloadResult(mp3_path, title)
        elif os.path.exists(final_filename):
            return DownloadResult(final_filename, title)
        else:
            raise FileNotFoundError(f"Extracted audio file not found for URL: {url}")
