"""
Bit-Perfect Direct Hardware DAC Audio Player using mpv JSON-IPC.
Bypasses PipeWire/PulseAudio software resamplers to deliver pure, untampered 24/32-bit PCM
straight to the Audiocular D07 DAC (or user-selected hardware sound card).
"""

import os
import json
import time
import socket
import logging
import subprocess
import threading
from pathlib import Path
from typing import Dict, Any, Optional

logger = logging.getLogger(__name__)

MPV_SOCKET_PATH = "/tmp/sonicscribe-mpv.sock"


class BitPerfectDacEngine:
    """Controls the background bit-perfect ALSA exclusive hardware player."""

    def __init__(self, default_device: str = "alsa/hw:D07,0"):
        self.target_device = default_device
        self.process: Optional[subprocess.Popen] = None
        self.lock = threading.RLock()
        self.current_filepath: Optional[str] = None
        self.current_track_data: Optional[Dict[str, Any]] = None
        self.is_playing: bool = False
        self.current_time: float = 0.0
        self.duration: float = 0.0
        self.engine_mode: str = "dac_direct"  # 'dac_direct' or 'browser_stream'
        self.monitor_thread: Optional[threading.Thread] = None
        self._stop_monitor = False

    def ensure_mpv_running(self):
        """Ensure mpv process is alive with IPC socket open."""
        with self.lock:
            if self.process and self.process.poll() is None and os.path.exists(MPV_SOCKET_PATH):
                return

            # Clean stale socket
            if os.path.exists(MPV_SOCKET_PATH):
                try:
                    os.unlink(MPV_SOCKET_PATH)
                except Exception:
                    pass

            cmd = [
                "mpv",
                "--idle=yes",
                f"--input-ipc-server={MPV_SOCKET_PATH}",
                "--ao=alsa",
                f"--audio-device={self.target_device}",
                "--audio-exclusive=yes",
                "--audio-samplerate=0",  # Disable forced software resampling
                "--audio-format=auto",   # Preserve 24-bit / 32-bit native format
                "--audio-channels=stereo",
                "--gapless-audio=yes",
                "--video=no",
                "--terminal=no",
                "--really-quiet"
            ]

            logger.info(f"Starting bit-perfect mpv daemon on {self.target_device}")
            try:
                self.process = subprocess.Popen(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                # Wait for socket to become active
                for _ in range(25):
                    if os.path.exists(MPV_SOCKET_PATH):
                        break
                    time.sleep(0.04)
            except Exception as e:
                logger.error(f"Failed to spawn mpv bit-perfect engine: {e}")

    def send_ipc_command(self, command: list, timeout: float = 0.5) -> Optional[Dict[str, Any]]:
        """Send JSON-IPC command to running mpv process."""
        self.ensure_mpv_running()
        if not os.path.exists(MPV_SOCKET_PATH):
            return None

        try:
            sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
            sock.settimeout(timeout)
            sock.connect(MPV_SOCKET_PATH)

            req = json.dumps({"command": command}) + "\n"
            sock.sendall(req.encode("utf-8"))

            buf = b""
            while True:
                chunk = sock.recv(1024)
                if not chunk:
                    break
                buf += chunk
                if b"\n" in chunk:
                    break

            sock.close()
            line = buf.decode("utf-8", errors="ignore").strip().split("\n")[0]
            if line:
                return json.loads(line)
        except Exception as e:
            # Expected during brief startup or socket disconnect
            return None
        return None

    def play(self, filepath: str, track_meta: Optional[Dict[str, Any]] = None) -> bool:
        """Load and start bit-perfect hardware playback of high-res audio file."""
        if not os.path.exists(filepath):
            logger.error(f"File not found: {filepath}")
            return False

        self.current_filepath = filepath
        self.current_track_data = track_meta
        self.ensure_mpv_running()

        res = self.send_ipc_command(["loadfile", filepath, "replace"])
        self.send_ipc_command(["set_property", "pause", False])
        self.is_playing = True

        if not self.monitor_thread or not self.monitor_thread.is_alive():
            self._stop_monitor = False
            self.monitor_thread = threading.Thread(target=self._telemetry_loop, daemon=True)
            self.monitor_thread.start()

        return True

    def pause(self) -> bool:
        """Pause playback."""
        self.send_ipc_command(["set_property", "pause", True])
        self.is_playing = False
        return True

    def resume(self) -> bool:
        """Resume playback."""
        self.send_ipc_command(["set_property", "pause", False])
        self.is_playing = True
        return True

    def toggle_play(self) -> bool:
        """Toggle play/pause."""
        if self.is_playing:
            return self.pause()
        else:
            return self.resume()

    def seek(self, seconds: float) -> bool:
        """Seek to absolute seconds."""
        self.send_ipc_command(["seek", max(0.0, float(seconds)), "absolute"])
        self.current_time = float(seconds)
        return True

    def stop(self) -> bool:
        """Stop playback and clear current file."""
        self.send_ipc_command(["stop"])
        self.is_playing = False
        self.current_time = 0.0
        return True

    def set_target_device(self, alsa_device: str):
        """Change ALSA hardware device (e.g. alsa/hw:D07,0). Restarts mpv daemon if device changed."""
        with self.lock:
            if self.target_device == alsa_device and self.process and self.process.poll() is None:
                return
            self.target_device = alsa_device
            if self.process:
                try:
                    self.process.terminate()
                    self.process.wait(timeout=1.0)
                except Exception:
                    pass
                self.process = None
            if os.path.exists(MPV_SOCKET_PATH):
                try:
                    os.unlink(MPV_SOCKET_PATH)
                except Exception:
                    pass
            self.ensure_mpv_running()

    def _telemetry_loop(self):
        """Poll mpv position and update telemetry state."""
        while not self._stop_monitor:
            try:
                res_time = self.send_ipc_command(["get_property", "time-pos"], timeout=0.2)
                if res_time and "data" in res_time and res_time["data"] is not None:
                    self.current_time = float(res_time["data"])

                res_dur = self.send_ipc_command(["get_property", "duration"], timeout=0.2)
                if res_dur and "data" in res_dur and res_dur["data"] is not None:
                    self.duration = float(res_dur["data"])

                res_pause = self.send_ipc_command(["get_property", "pause"], timeout=0.2)
                if res_pause and "data" in res_pause and res_pause["data"] is not None:
                    self.is_playing = not bool(res_pause["data"])
            except Exception:
                pass
            time.sleep(0.08)

    def get_playback_state(self) -> Dict[str, Any]:
        """Return unified player telemetry state."""
        return {
            "mode": self.engine_mode,
            "target_device": self.target_device,
            "filepath": self.current_filepath,
            "track": self.current_track_data,
            "is_playing": self.is_playing,
            "current_time": round(self.current_time, 2),
            "duration": round(self.duration, 2),
            "bit_perfect": self.engine_mode == "dac_direct" and self.is_playing
        }


# Singleton instance
dac_player = BitPerfectDacEngine()
