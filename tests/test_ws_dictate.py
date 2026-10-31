"""
Test script for SonicScribe Real-Time Streaming Dictation WebSocket.
Connects to ws://127.0.0.1:7860/api/ws/dictate, streams PCM chunks, and validates messages.
"""

import asyncio
import json
import subprocess
import websockets

AUDIO_FILE = "/home/sumeet/audio-to-text/uploads/mic_recording_1793379624573.wav"
WS_URL = "ws://127.0.0.1:7860/api/ws/dictate"

async def test_streaming_dictate():
    print(f"Connecting to {WS_URL}...")
    
    # Use ffmpeg to convert test audio to raw 16kHz 16-bit PCM mono
    cmd = [
        "ffmpeg", "-y", "-i", AUDIO_FILE,
        "-f", "s16le", "-acodec", "pcm_s16le",
        "-ar", "16000", "-ac", "1",
        "pipe:1"
    ]
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)
    pcm_data, _ = proc.communicate()
    print(f"Loaded {len(pcm_data)} bytes of 16kHz PCM audio (~{round(len(pcm_data)/32000, 2)}s).")

    async with websockets.connect(WS_URL) as ws:
        print("✓ Connected to WebSocket!")
        received_messages = []

        async def receiver():
            try:
                while True:
                    msg = await ws.recv()
                    data = json.loads(msg)
                    received_messages.append(data)
                    msg_type = data.get("type")
                    if msg_type == "interim":
                        print(f"  [INTERIM] {data.get('text')} (duration: {data.get('duration')}s)")
                    elif msg_type == "final":
                        print(f"  [FINAL] {data.get('segment', {}).get('text')}")
                    elif msg_type == "session_complete":
                        print("  [COMPLETE] Dictation session ended.")
                        break
            except Exception as e:
                print(f"Receiver ended: {e}")

        recv_task = asyncio.create_task(receiver())

        # Stream in 250ms chunks (4000 samples = 8000 bytes)
        chunk_size = 8000
        for i in range(0, len(pcm_data), chunk_size):
            chunk = pcm_data[i:i + chunk_size]
            await ws.send(chunk)
            await asyncio.sleep(0.1)  # Simulate real-time streaming

        # Wait a moment for silence/finalization
        await asyncio.sleep(1.0)
        await ws.send(json.dumps({"action": "stop"}))
        await recv_task

        assert len(received_messages) > 0, "No messages received from WebSocket!"
        print(f"✓ TEST PASSED! Received {len(received_messages)} streaming dictation messages.")

if __name__ == "__main__":
    asyncio.run(test_streaming_dictate())
